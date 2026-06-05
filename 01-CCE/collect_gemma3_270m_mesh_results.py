#!/usr/bin/env python3
"""Collect Gemma3 270M multi-chip mesh CCE compatibility artifacts."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma3_270m_mesh_cce"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"

VARIANT_LABEL = {"default": "Default CE", "cce": "CCE"}
VARIANT_COLOR = {"default": "#555555", "cce": "#159A78"}


def write_frame(frame: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)


def extract_artifacts() -> None:
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    profile = tarball.name.removesuffix(".tar.gz")
    target = RAW_DIR / profile
    if target.exists():
      shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(target)], check=True)


def read_results() -> pd.DataFrame:
  frames = []
  for path in sorted(RAW_DIR.glob("*/gemma3-270m-cce-mesh/**/*_results.csv")):
    frame = pd.read_csv(path)
    frame["source_csv"] = str(path.relative_to(SCRIPT_DIR))
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  rows = pd.concat(frames, ignore_index=True, sort=False)
  for col in [
      "batch_size",
      "max_length",
      "lora_rank",
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
  rows["mesh"] = rows.apply(
      lambda row: f"fsdp{int(row['mesh_fsdp'])}/tp{int(row['mesh_tp'])}",
      axis=1,
  )
  rows["variant_label"] = rows["variant"].map(VARIANT_LABEL).fillna(rows["variant"])
  rows["ok"] = rows["status"].eq("ok")
  return rows


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
  if rows.empty:
    return pd.DataFrame()
  summary_rows = []
  for keys, part in rows.groupby(["mesh", "mesh_fsdp", "mesh_tp", "batch_size", "variant"]):
    ok = part[part["ok"]]
    failed = part[~part["ok"]]
    summary_rows.append({
        "mesh": keys[0],
        "mesh_fsdp": int(keys[1]),
        "mesh_tp": int(keys[2]),
        "batch_size": int(keys[3]),
        "variant": keys[4],
        "max_ok_context": int(ok["max_length"].max()) if not ok.empty else 0,
        "first_failed_context": int(failed["max_length"].min()) if not failed.empty else "",
        "ok_rows": int(ok.shape[0]),
        "failed_rows": int(failed.shape[0]),
    })
  return pd.DataFrame(summary_rows).sort_values(
      ["mesh_fsdp", "mesh_tp", "batch_size", "variant"]
  )


def build_matched(rows: pd.DataFrame) -> pd.DataFrame:
  if rows.empty:
    return pd.DataFrame()
  ok = rows[rows["ok"]].copy()
  if ok.empty:
    return pd.DataFrame()
  matched = ok.pivot_table(
      index=["mesh", "mesh_fsdp", "mesh_tp", "batch_size", "max_length"],
      columns="variant",
      values=["xla_train_step_gib_per_chip", "mean_step_time_sec_excl_first", "final_loss"],
      aggfunc="first",
  )
  matched.columns = ["_".join(col).strip("_") for col in matched.columns.values]
  matched = matched.reset_index()
  if {"xla_train_step_gib_per_chip_default", "xla_train_step_gib_per_chip_cce"} <= set(matched.columns):
    matched["xla_reduction_pct"] = (
        1
        - matched["xla_train_step_gib_per_chip_cce"]
        / matched["xla_train_step_gib_per_chip_default"]
    ) * 100
  return matched.sort_values(["mesh_fsdp", "mesh_tp", "batch_size", "max_length"])


def context_label(value: float) -> str:
  value = int(value)
  return f"{value // 1024}K" if value >= 1024 else str(value)


def plot_mesh(rows: pd.DataFrame, summary: pd.DataFrame, matched: pd.DataFrame) -> None:
  if rows.empty or summary.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), constrained_layout=True)
  fig.suptitle(
      "Gemma3 270M CCE mesh generalization on TPU v5litepod-4 (4 chips)",
      fontsize=14,
      fontweight="bold",
  )

  ax = axes[0]
  summary = summary.copy()
  summary["shape"] = summary.apply(
      lambda row: f"{row['mesh']}\nb{int(row['batch_size'])}",
      axis=1,
  )
  shape_order = (
      summary[["mesh_fsdp", "mesh_tp", "batch_size", "shape"]]
      .drop_duplicates()
      .sort_values(["mesh_fsdp", "mesh_tp", "batch_size"])["shape"]
      .tolist()
  )
  x_base = list(range(len(shape_order)))
  width = 0.36
  for idx, variant in enumerate(["default", "cce"]):
    part = summary[summary["variant"].eq(variant)].set_index("shape")
    heights = [max(part.loc[shape, "max_ok_context"], 1) if shape in part.index else math.nan for shape in shape_order]
    offset = (-0.5 + idx) * width
    bars = ax.bar(
        [x + offset for x in x_base],
        heights,
        width=width,
        color=VARIANT_COLOR[variant],
        label=VARIANT_LABEL[variant],
    )
    for bar, height in zip(bars, heights, strict=True):
      if math.isnan(height):
        continue
      label = "none" if height <= 1 else context_label(height)
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          height * 1.08,
          label,
          ha="center",
          va="bottom",
          fontsize=7,
      )
  ax.set_yscale("log", base=2)
  ax.set_yticks([1, 512, 1024, 2048])
  ax.set_yticklabels(["none", "512", "1K", "2K"])
  ax.set_xticks(x_base)
  ax.set_xticklabels(shape_order, fontsize=8)
  ax.set_ylabel("Maximum completed context")
  ax.set_title("Frontier by mesh and batch")
  ax.set_axisbelow(True)
  ax.grid(True, axis="y", color="#E6E6E6")
  ax.legend(frameon=False)

  ax = axes[1]
  if not matched.empty and "xla_reduction_pct" in matched:
    plot_data = matched.dropna(subset=["xla_reduction_pct"]).copy()
    plot_data["shape"] = plot_data.apply(
        lambda row: f"{row['mesh']} b{int(row['batch_size'])}/L{context_label(row['max_length'])}",
        axis=1,
    )
    plot_data = plot_data.sort_values(["mesh_fsdp", "mesh_tp", "batch_size", "max_length"])
    bars = ax.bar(
        range(len(plot_data)),
        plot_data["xla_reduction_pct"],
        color="#159A78",
        alpha=0.88,
    )
    for bar, value in zip(bars, plot_data["xla_reduction_pct"], strict=True):
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          value + 1.5,
          f"{value:.0f}%",
          ha="center",
          va="bottom",
          fontsize=7,
      )
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data["shape"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("CCE per-chip XLA planned HBM reduction vs Default (%)")
    ax.set_title("Matched passing shapes (both variants OK)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#E6E6E6")
  else:
    ax.text(0.5, 0.5, "No matched passing rows yet", ha="center", va="center")
    ax.axis("off")

  fig.savefig(ASSET_DIR / "gemma3_270m_cce_mesh_generalization.png", dpi=180)
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
    })
  return pd.DataFrame(rows)


def main() -> None:
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
  extract_artifacts()
  rows = read_results()
  write_frame(build_manifest(), DATA_DIR / "run_manifest.csv")
  if rows.empty:
    print("No mesh results found")
    return
  summary = build_summary(rows)
  matched = build_matched(rows)
  write_frame(rows, DATA_DIR / "mesh_runs.csv")
  write_frame(summary, DATA_DIR / "mesh_summary.csv")
  write_frame(matched, DATA_DIR / "matched_memory.csv")
  plot_mesh(rows, summary, matched)
  print(f"wrote={DATA_DIR}")
  print(f"figure={ASSET_DIR / 'gemma3_270m_cce_mesh_generalization.png'}")


if __name__ == "__main__":
  main()
