#!/usr/bin/env python3
"""Collect repeated Gemma3 270M four-chip CCE timing checks."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma3_270m_mesh_cce_repeat"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"

VARIANT_COLOR = {"default": "#555555", "cce": "#159A78"}


def write_frame(frame: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)


def extract_artifacts() -> None:
  if RAW_DIR.exists():
    shutil.rmtree(RAW_DIR)
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    target = RAW_DIR / tarball.name.removesuffix(".tar.gz")
    if target.exists():
      shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(target)], check=True)


def read_results() -> pd.DataFrame:
  frames = []
  for path in sorted(RAW_DIR.glob("*/gemma3-270m-cce-mesh-repeat/**/*_results.csv")):
    frame = pd.read_csv(path)
    frame["source_csv"] = str(path.relative_to(SCRIPT_DIR))
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  rows = pd.concat(frames, ignore_index=True, sort=False)
  for col in [
      "batch_size",
      "max_length",
      "chips",
      "mesh_fsdp",
      "mesh_tp",
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
      "valid_tokens_per_sec_excl_first",
      "final_loss",
  ]:
    if col in rows:
      rows[col] = pd.to_numeric(rows[col], errors="coerce")
  rows["repeat"] = rows["suite"].map(
      lambda value: int(re.search(r"repeat(\d+)", str(value)).group(1))
      if re.search(r"repeat(\d+)", str(value))
      else 0
  )
  rows["mesh"] = rows.apply(
      lambda row: f"fsdp{int(row['mesh_fsdp'])}/tp{int(row['mesh_tp'])}",
      axis=1,
  )
  rows["shape"] = rows.apply(
      lambda row: f"b{int(row['batch_size'])}/L{int(row['max_length'])}",
      axis=1,
  )
  rows["ok"] = rows["status"].eq("ok")
  return rows.sort_values(["repeat", "batch_size", "max_length", "variant"])


def build_matched(rows: pd.DataFrame) -> pd.DataFrame:
  ok = rows[rows["ok"]].copy()
  if ok.empty:
    return pd.DataFrame()
  matched = ok.pivot_table(
      index=[
          "repeat",
          "mesh",
          "mesh_fsdp",
          "mesh_tp",
          "shape",
          "batch_size",
          "max_length",
      ],
      columns="variant",
      values=[
          "xla_train_step_gib_per_chip",
          "mean_step_time_sec_excl_first",
          "valid_tokens_per_sec_excl_first",
          "final_loss",
      ],
      aggfunc="first",
  )
  matched.columns = ["_".join(col).strip("_") for col in matched.columns.values]
  matched = matched.reset_index()
  required = {
      "xla_train_step_gib_per_chip_default",
      "xla_train_step_gib_per_chip_cce",
      "mean_step_time_sec_excl_first_default",
      "mean_step_time_sec_excl_first_cce",
  }
  if required <= set(matched.columns):
    matched["xla_reduction_pct"] = (
        1
        - matched["xla_train_step_gib_per_chip_cce"]
        / matched["xla_train_step_gib_per_chip_default"]
    ) * 100
    matched["step_time_multiplier_cce_vs_default"] = (
        matched["mean_step_time_sec_excl_first_cce"]
        / matched["mean_step_time_sec_excl_first_default"]
    )
    matched["step_time_overhead_pct"] = (
        matched["step_time_multiplier_cce_vs_default"] - 1
    ) * 100
  return matched.sort_values(["mesh_fsdp", "mesh_tp", "batch_size", "max_length", "repeat"])


def build_summary(matched: pd.DataFrame) -> pd.DataFrame:
  if matched.empty:
    return matched
  return (
      matched.groupby(["mesh", "mesh_fsdp", "mesh_tp", "shape", "batch_size", "max_length"])
      .agg(
          repeats=("repeat", "nunique"),
          mean_xla_reduction_pct=("xla_reduction_pct", "mean"),
          std_xla_reduction_pct=("xla_reduction_pct", "std"),
          mean_step_time_multiplier=("step_time_multiplier_cce_vs_default", "mean"),
          std_step_time_multiplier=("step_time_multiplier_cce_vs_default", "std"),
          min_step_time_multiplier=("step_time_multiplier_cce_vs_default", "min"),
          max_step_time_multiplier=("step_time_multiplier_cce_vs_default", "max"),
          mean_default_step_s=("mean_step_time_sec_excl_first_default", "mean"),
          mean_cce_step_s=("mean_step_time_sec_excl_first_cce", "mean"),
      )
      .reset_index()
      .sort_values(["mesh_fsdp", "mesh_tp", "batch_size", "max_length"])
  )


def plot_repeat(matched: pd.DataFrame, summary: pd.DataFrame) -> None:
  if matched.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  summary = summary.copy()
  summary["label"] = summary.apply(
      lambda row: f"{row['mesh']}\n{row['shape']}",
      axis=1,
  )
  labels = summary["label"].tolist()
  label_map = {label: idx for idx, label in enumerate(labels)}
  fig, axes = plt.subplots(
      1,
      2,
      figsize=(12.0, 3.8),
      gridspec_kw={"width_ratios": [1.0, 1.05]},
      constrained_layout=True,
  )
  fig.suptitle(
      "Gemma3 270M four-chip CCE repeat check on TPU v5litepod-4",
      fontsize=11.5,
      fontweight="bold",
  )

  ax = axes[0]
  jitter = {1: -0.08, 2: 0.0, 3: 0.08}
  mesh_colors = {
      "fsdp1/tp4": "#2B6CB0",
      "fsdp2/tp2": "#C05621",
      "fsdp4/tp1": "#159A78",
  }
  for _, row in matched.iterrows():
    if pd.isna(row.get("step_time_multiplier_cce_vs_default")):
      continue
    label = f"{row['mesh']}\n{row['shape']}"
    ax.scatter(
        label_map[label] + jitter.get(int(row["repeat"]), 0.0),
        row["step_time_multiplier_cce_vs_default"],
        s=54,
        color=mesh_colors.get(row["mesh"], "#555555"),
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
    )
  for _, row in summary.iterrows():
    label = row["label"]
    ax.hlines(
        row["mean_step_time_multiplier"],
        label_map[label] - 0.22,
        label_map[label] + 0.22,
        color="#333333",
        linewidth=2.0,
    )
    ax.text(
        label_map[label],
        row["mean_step_time_multiplier"] * 1.04,
        f"{row['mean_step_time_multiplier']:.1f}x",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
    )
  ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
  ax.set_yscale("log", base=2)
  ax.set_ylim(0.9, 128)
  ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
  ax.set_yticklabels(["1x", "2x", "4x", "8x", "16x", "32x", "64x", "128x"])
  ax.set_xticks(range(len(labels)))
  ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
  ax.set_ylabel("CCE step-time multiplier vs Default")
  ax.set_title("Repeated matched-row overhead")
  ax.set_axisbelow(True)
  ax.grid(True, axis="y", color="#E6E6E6")

  ax = axes[1]
  width = 0.34
  for idx, variant in enumerate(["default", "cce"]):
    values = []
    for _, row in summary.iterrows():
      part = matched[
          matched["mesh"].eq(row["mesh"])
          & matched["shape"].eq(row["shape"])
      ]
      values.append(part[f"mean_step_time_sec_excl_first_{variant}"].mean())
    offsets = [x + (-0.5 + idx) * width for x in range(len(labels))]
    bars = ax.bar(
        offsets,
        values,
        width=width,
        color=VARIANT_COLOR[variant],
        label="Default CE" if variant == "default" else "CCE",
    )
    for bar, value in zip(bars, values, strict=True):
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          value * 1.04,
          f"{value:.2f}s" if value < 1 else f"{value:.1f}s",
          ha="center",
          va="bottom",
          fontsize=7,
      )
  ax.set_yscale("log", base=10)
  ax.set_ylabel("Mean step time, seconds (log scale)")
  ax.set_xticks(range(len(labels)))
  ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
  ax.set_title("Raw step time explains the ratio")
  ax.set_axisbelow(True)
  ax.grid(True, axis="y", color="#E6E6E6")
  ax.legend(frameon=False, fontsize=8)

  fig.savefig(ASSET_DIR / "gemma3_270m_cce_mesh_2x2_repeat.png", dpi=180)
  plt.close(fig)


def build_manifest() -> pd.DataFrame:
  rows = []
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    rows.append({
        "artifact": str(tarball.relative_to(SCRIPT_DIR)),
        "bytes": tarball.stat().st_size,
        "profile": tarball.name.removesuffix(".tar.gz"),
        "tpu": "v5litepod-4",
        "chips": 4,
        "project": "gcp-ml-172005",
        "zone": "us-west4-a",
        "model": "google/gemma-3-270m-it",
        "mesh": "fsdp/tp repeat grid",
        "repeats": 3,
    })
  return pd.DataFrame(rows)


def main() -> None:
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
  extract_artifacts()
  rows = read_results()
  write_frame(build_manifest(), DATA_DIR / "run_manifest.csv")
  if rows.empty:
    print("No repeat results found")
    return
  matched = build_matched(rows)
  summary = build_summary(matched)
  write_frame(rows, DATA_DIR / "repeat_runs.csv")
  write_frame(matched, DATA_DIR / "repeat_matched.csv")
  write_frame(summary, DATA_DIR / "repeat_summary.csv")
  plot_repeat(matched, summary)
  print(f"wrote={DATA_DIR}")
  print(f"figure={ASSET_DIR / 'gemma3_270m_cce_mesh_2x2_repeat.png'}")


if __name__ == "__main__":
  main()
