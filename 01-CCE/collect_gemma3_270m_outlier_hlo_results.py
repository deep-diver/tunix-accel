#!/usr/bin/env python3
"""Collect HLO/XLA evidence for Gemma3 270M four-chip CCE outliers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma3_270m_outlier_hlo"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"

OPS = [
    "all-gather",
    "all-reduce",
    "collective-permute",
    "reduce-scatter",
    "dot(",
    "reduce(",
    "transpose(",
    "dynamic-slice(",
    "dynamic-update-slice(",
]


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
  for path in sorted(RAW_DIR.glob("*/gemma3-270m-cce-outlier-hlo/**/*_results.csv")):
    frame = pd.read_csv(path)
    frame["source_csv"] = str(path.relative_to(SCRIPT_DIR))
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  rows = pd.concat(frames, ignore_index=True, sort=False)
  for col in [
      "batch_size",
      "max_length",
      "mesh_fsdp",
      "mesh_tp",
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
      "final_loss",
  ]:
    if col in rows:
      rows[col] = pd.to_numeric(rows[col], errors="coerce")
  rows["mesh"] = rows.apply(
      lambda row: f"fsdp{int(row['mesh_fsdp'])}/tp{int(row['mesh_tp'])}",
      axis=1,
  )
  return rows


def case_dir(row: pd.Series) -> Path:
  csv_parent = SCRIPT_DIR / str(row["source_csv"])
  return csv_parent.parent / Path(str(row["run_dir"])).name


def count_hlo_ops(path: Path) -> dict[str, int | str]:
  counter: Counter[str] = Counter()
  text_files = sorted(path.glob("xla/*.txt"))

  def is_hlo_body(file: Path) -> bool:
    name = file.name
    excluded = [
        "memory-usage-report",
        "live-range",
        "buffer-assignment",
        "hlo_module_config",
        "basic_compiler_metadata",
        "tpu_comp_env",
        "target_arguments",
        "transfer_stats",
        "execution_options",
    ]
    if any(item in name for item in excluded):
      return False
    return name.endswith((
        ".before_optimizations.txt",
        ".after_optimizations.txt",
        ".after_optimizations_before_buffer_assignment.txt",
        ".after_optimizations_after_buffer_assignment.txt",
        ".after_codegen.txt",
    ))

  hlo_files = [file for file in text_files if is_hlo_body(file)]
  train_like = [
      file for file in hlo_files
      if "jit__train_step" in file.name or "jit_train_step" in file.name
  ]
  selected = train_like or hlo_files
  text_bytes = 0
  file_count = 0
  for file in selected:
    try:
      text = file.read_text(errors="ignore")
    except OSError:
      continue
    file_count += 1
    text_bytes += len(text)
    lower = text.lower()
    for op in OPS:
      counter[op] += lower.count(op)
  out: dict[str, int | str] = {
      "hlo_files_scanned": file_count,
      "hlo_text_bytes": text_bytes,
  }
  for op in OPS:
    out[op.replace("(", "").replace("-", "_")] = counter[op]
  return out


def build_hlo_summary(rows: pd.DataFrame) -> pd.DataFrame:
  out = []
  for _, row in rows.iterrows():
    run_dir = case_dir(row)
    counts = count_hlo_ops(run_dir)
    out.append({
        "mesh": row["mesh"],
        "mesh_fsdp": row["mesh_fsdp"],
        "mesh_tp": row["mesh_tp"],
        "variant": row["variant"],
        "batch_size": row["batch_size"],
        "max_length": row["max_length"],
        "status": row["status"],
        "xla_train_step_gib_per_chip": row.get("xla_train_step_gib_per_chip"),
        "mean_step_time_sec_excl_first": row.get("mean_step_time_sec_excl_first"),
        "final_loss": row.get("final_loss"),
        "run_dir": row["run_dir"],
        **counts,
    })
  return pd.DataFrame(out).sort_values(
      ["mesh_fsdp", "mesh_tp", "batch_size", "max_length", "variant"]
  )


def build_matched(rows: pd.DataFrame) -> pd.DataFrame:
  ok = rows[rows["status"].eq("ok")].copy()
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
  if {
      "mean_step_time_sec_excl_first_default",
      "mean_step_time_sec_excl_first_cce",
  } <= set(matched.columns):
    matched["step_time_multiplier_cce_vs_default"] = (
        matched["mean_step_time_sec_excl_first_cce"]
        / matched["mean_step_time_sec_excl_first_default"]
    )
  if {
      "xla_train_step_gib_per_chip_default",
      "xla_train_step_gib_per_chip_cce",
  } <= set(matched.columns):
    matched["xla_reduction_pct"] = (
        1
        - matched["xla_train_step_gib_per_chip_cce"]
        / matched["xla_train_step_gib_per_chip_default"]
    ) * 100
  return matched


def plot_hlo(summary: pd.DataFrame, matched: pd.DataFrame) -> None:
  if summary.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)
  fig.suptitle(
      "Gemma3 270M CCE outlier HLO check on TPU v5litepod-4",
      fontsize=11.5,
      fontweight="bold",
  )

  ax = axes[0]
  collective_cols = ["all_gather", "all_reduce", "collective_permute", "reduce_scatter"]
  plot_rows = summary[
      summary["variant"].eq("cce")
      & summary["batch_size"].eq(16)
      & summary["max_length"].eq(512)
  ].copy()
  plot_rows["collective_ops"] = plot_rows[collective_cols].sum(axis=1)
  plot_rows = plot_rows.sort_values(["mesh_fsdp", "mesh_tp"])
  bars = ax.bar(
      plot_rows["mesh"],
      plot_rows["collective_ops"],
      color="#C05621",
      alpha=0.88,
  )
  for bar, value in zip(bars, plot_rows["collective_ops"], strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + max(1, value * 0.02),
        f"{int(value)}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
  ax.set_ylabel("Collective-like HLO op mentions")
  ax.set_title("Collective mentions do not predict the outlier")
  ax.set_axisbelow(True)
  ax.grid(True, axis="y", color="#E6E6E6")

  ax = axes[1]
  if not matched.empty and "step_time_multiplier_cce_vs_default" in matched:
    part = matched[
        matched["batch_size"].eq(16)
        & matched["max_length"].eq(512)
        & matched["step_time_multiplier_cce_vs_default"].notna()
    ].sort_values(["mesh_fsdp", "mesh_tp"])
    ax.bar(
        part["mesh"],
        part["step_time_multiplier_cce_vs_default"],
        color="#159A78",
        alpha=0.88,
    )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}x"))
    ax.set_ylabel("CCE step-time multiplier")
    ax.set_title("Same rows, measured overhead")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#E6E6E6")
  else:
    ax.axis("off")

  fig.savefig(ASSET_DIR / "gemma3_270m_cce_outlier_hlo.png", dpi=180)
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
    print("No outlier HLO results found")
    return
  hlo_summary = build_hlo_summary(rows)
  matched = build_matched(rows)
  write_frame(rows, DATA_DIR / "outlier_runs.csv")
  write_frame(hlo_summary, DATA_DIR / "hlo_op_counts.csv")
  write_frame(matched, DATA_DIR / "matched_metrics.csv")
  plot_hlo(hlo_summary, matched)
  print(f"wrote={DATA_DIR}")
  print(f"figure={ASSET_DIR / 'gemma3_270m_cce_outlier_hlo.png'}")


if __name__ == "__main__":
  main()
