#!/usr/bin/env python3
"""Collect Gemma3 270M four-chip frontier artifacts."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma3_270m_4chip_frontier"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"

VARIANT_COLOR = {"default": "#555555", "cce": "#159A78"}
MESH_COLOR = {
    "fsdp1/tp4": "#2B6CB0",
    "fsdp2/tp2": "#C05621",
    "fsdp4/tp1": "#159A78",
}


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
  for path in sorted(RAW_DIR.glob("*/gemma3-270m-cce-4chip-frontier/**/*_results.csv")):
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
  rows["mesh"] = rows.apply(
      lambda row: f"fsdp{int(row['mesh_fsdp'])}/tp{int(row['mesh_tp'])}",
      axis=1,
  )
  rows["ok"] = rows["status"].eq("ok")
  return rows.sort_values(["mesh_fsdp", "mesh_tp", "batch_size", "max_length", "variant"])


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
  if rows.empty:
    return pd.DataFrame()
  out = []
  for keys, part in rows.groupby(["mesh", "mesh_fsdp", "mesh_tp", "batch_size", "variant"]):
    ok = part[part["ok"]]
    failed = part[~part["ok"]]
    out.append({
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
  return pd.DataFrame(out).sort_values(["mesh_fsdp", "mesh_tp", "batch_size", "variant"])


def build_matched(rows: pd.DataFrame) -> pd.DataFrame:
  ok = rows[rows["ok"]].copy()
  if ok.empty:
    return pd.DataFrame()
  matched = ok.pivot_table(
      index=["mesh", "mesh_fsdp", "mesh_tp", "batch_size", "max_length"],
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
  if {
      "xla_train_step_gib_per_chip_default",
      "xla_train_step_gib_per_chip_cce",
  } <= set(matched.columns):
    matched["xla_reduction_pct"] = (
        1
        - matched["xla_train_step_gib_per_chip_cce"]
        / matched["xla_train_step_gib_per_chip_default"]
    ) * 100
  if {
      "mean_step_time_sec_excl_first_default",
      "mean_step_time_sec_excl_first_cce",
  } <= set(matched.columns):
    matched["step_time_multiplier_cce_vs_default"] = (
        matched["mean_step_time_sec_excl_first_cce"]
        / matched["mean_step_time_sec_excl_first_default"]
    )
  if {"final_loss_default", "final_loss_cce"} <= set(matched.columns):
    matched["loss_delta_cce_minus_default"] = (
        matched["final_loss_cce"] - matched["final_loss_default"]
    )
    matched["loss_abs_delta"] = matched["loss_delta_cce_minus_default"].abs()
  return matched.sort_values(["mesh_fsdp", "mesh_tp", "batch_size", "max_length"])


def context_label(value: float) -> str:
  value = int(value)
  return f"{value // 1024}K" if value >= 1024 else str(value)


def context_rank(value: float) -> int:
  value = int(value)
  if value <= 0:
    return 0
  return {256: 1, 512: 2, 1024: 3, 2048: 4, 4096: 5}.get(
      value, int(math.log2(value // 256)) + 1
  )


def plot_frontier(summary: pd.DataFrame, matched: pd.DataFrame) -> None:
  if summary.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  meshes = summary[["mesh", "mesh_fsdp", "mesh_tp"]].drop_duplicates().sort_values(
      ["mesh_fsdp", "mesh_tp"]
  )
  fig, axes = plt.subplots(
      len(meshes),
      1,
      figsize=(9.2, 7.6),
      constrained_layout=True,
  )
  if len(meshes) == 1:
    axes = [axes]
  fig.suptitle(
      "Gemma3 270M CCE four-chip frontier on TPU v5litepod-4",
      fontsize=12,
      fontweight="bold",
  )

  for row_idx, mesh_row in enumerate(meshes.itertuples(index=False)):
    mesh = mesh_row.mesh
    part = summary[summary["mesh"].eq(mesh)]
    batches = sorted(part["batch_size"].unique())
    x_base = list(range(len(batches)))
    width = 0.36
    ax = axes[row_idx]
    for idx, variant in enumerate(["default", "cce"]):
      variant_part = part[part["variant"].eq(variant)].set_index("batch_size")
      values = [variant_part.loc[batch, "max_ok_context"] for batch in batches]
      heights = [context_rank(value) for value in values]
      bars = ax.bar(
          [x + (-0.5 + idx) * width for x in x_base],
          heights,
          width=width,
          color=VARIANT_COLOR[variant],
          label="Default CE" if variant == "default" else "CCE",
      )
      for bar, value, height in zip(bars, values, heights, strict=True):
        label = "none" if int(value) <= 0 else context_label(value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_ylim(-0.2, 5.45)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["none", "256", "512", "1K", "2K", "4K"])
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"b{batch}" for batch in batches], fontsize=7)
    ax.set_ylabel(f"{mesh}\nmax context")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#E6E6E6")
    if row_idx == 0:
      ax.set_title("Maximum completed context by batch size")
      ax.legend(frameon=False, fontsize=8, loc="upper right")

  fig.savefig(ASSET_DIR / "gemma3_270m_cce_4chip_frontier.png", dpi=180)
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
    print("No four-chip frontier results found")
    return
  summary = build_summary(rows)
  matched = build_matched(rows)
  write_frame(rows, DATA_DIR / "frontier_runs.csv")
  write_frame(summary, DATA_DIR / "frontier_summary.csv")
  write_frame(matched, DATA_DIR / "matched_metrics.csv")
  plot_frontier(summary, matched)
  print(f"wrote={DATA_DIR}")
  print(f"figure={ASSET_DIR / 'gemma3_270m_cce_4chip_frontier.png'}")


if __name__ == "__main__":
  main()
