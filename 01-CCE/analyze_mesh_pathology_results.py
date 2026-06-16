#!/usr/bin/env python3
"""Analyze generalized TPU/GPU mesh pathology results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "mesh-pathology-matrix" / "results"
DEFAULT_OUTDIR = SCRIPT_DIR / "results" / "mesh-pathology-matrix" / "analysis"

WORKLOAD_ORDER = [
    "cce_train",
    "chunked_matmul_loop",
    "collective_loop",
    "projection_collective_loop",
]
WORKLOAD_LABELS = {
    "cce_train": "CCE train",
    "chunked_matmul_loop": "Chunked matmul",
    "collective_loop": "Collective only",
    "projection_collective_loop": "Projection + collective",
}
SIGNATURE_ORDER = [
    "bad",
    "good",
    "control-fsdp4-tp1",
    "control-fsdp1-tp4",
]
SIGNATURE_LABELS = {
    "bad": "fsdp2/tp2\n128/8192\nbad",
    "good": "fsdp2/tp2\n512/65536\ngood",
    "control-fsdp4-tp1": "fsdp4/tp1\n128/8192",
    "control-fsdp1-tp4": "fsdp1/tp4\n128/8192",
}
SIGNATURE_COLORS = {
    "bad": "#c94f4f",
    "good": "#2f6f4e",
    "control-fsdp4-tp1": "#669bbc",
    "control-fsdp1-tp4": "#8d80ad",
}


def read_jsonl(path: Path) -> pd.DataFrame:
  rows: list[dict[str, Any]] = []
  with path.open() as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return pd.DataFrame(rows)


def read_input(path: Path) -> pd.DataFrame:
  if not path.exists():
    return pd.DataFrame()
  if path.suffix == ".csv":
    return pd.read_csv(path)
  return read_jsonl(path)


def discover_inputs(args: argparse.Namespace) -> list[Path]:
  if args.input:
    return [path.expanduser().resolve() for path in args.input]
  results_dir = args.results_dir.expanduser().resolve()
  merged = results_dir / "all_results.jsonl"
  if merged.exists():
    return [merged]
  return sorted(results_dir.glob("shard_*.jsonl"))


def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
  for column in columns:
    if column in frame.columns:
      frame[column] = pd.to_numeric(frame[column], errors="coerce")


def normalize(rows: pd.DataFrame) -> pd.DataFrame:
  if rows.empty:
    return pd.DataFrame(
        columns=[
            "experiment_id",
            "hardware_target",
            "workload_family",
            "scenario_group",
            "signature_label",
            "ok",
            "steady_state_mean_step_time_sec",
            "compile_time_sec",
            "xla_planned_hbm_gib_per_chip",
            "chunk_loop_count",
        ]
    )
  rows = rows.copy()
  if "scenario_group" not in rows:
    rows["scenario_group"] = (
        rows.get("hardware_target", "").astype(str)
        + "/"
        + rows.get("workload_family", "").astype(str)
    )
  status = rows["status"] if "status" in rows else pd.Series("", index=rows.index)
  rows["ok"] = status.astype(str).str.lower().isin({"success", "ok"})
  numeric(
      rows,
      [
          "experiment_id",
          "fsdp_degree",
          "tp_degree",
          "global_batch_size",
          "sequence_length",
          "hidden_size",
          "vocab_size",
          "token_chunk",
          "vocab_chunk",
          "token_loop_count",
          "vocab_loop_count",
          "chunk_loop_count",
          "steady_state_mean_step_time_sec",
          "median_step_time_sec",
          "p95_step_time_sec",
          "compile_time_sec",
          "tokens_per_sec",
          "xla_planned_hbm_gib_per_chip",
          "runtime_hbm_peak_gb",
      ],
  )
  return rows


def write_frame(frame: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)


def build_scenario_summary(rows: pd.DataFrame) -> pd.DataFrame:
  if rows.empty:
    return pd.DataFrame()
  grouped = (
      rows.groupby(["hardware_target", "workload_family"], dropna=False)
      .agg(
          rows=("experiment_id", "count"),
          successes=("ok", "sum"),
          median_step_time_sec=("steady_state_mean_step_time_sec", "median"),
          median_compile_time_sec=("compile_time_sec", "median"),
          median_hbm_gib=("xla_planned_hbm_gib_per_chip", "median"),
      )
      .reset_index()
  )
  grouped["scenario_group"] = grouped["hardware_target"] + "/" + grouped["workload_family"]
  return grouped.sort_values(["hardware_target", "workload_family"])


def build_bad_good(rows: pd.DataFrame) -> pd.DataFrame:
  ok = rows[
      rows["ok"]
      & rows["steady_state_mean_step_time_sec"].notna()
      & rows["signature_label"].isin(["bad", "good"])
  ].copy()
  if ok.empty:
    return pd.DataFrame()
  key = ["hardware_target", "workload_family"]
  bad = (
      ok[ok["signature_label"].eq("bad")]
      .groupby(key, dropna=False)
      .agg(
          bad_step_time_sec=("steady_state_mean_step_time_sec", "median"),
          bad_compile_time_sec=("compile_time_sec", "median"),
          bad_hbm_gib=("xla_planned_hbm_gib_per_chip", "median"),
          bad_chunk_loop_count=("chunk_loop_count", "median"),
      )
      .reset_index()
  )
  good = (
      ok[ok["signature_label"].eq("good")]
      .groupby(key, dropna=False)
      .agg(
          good_step_time_sec=("steady_state_mean_step_time_sec", "median"),
          good_compile_time_sec=("compile_time_sec", "median"),
          good_hbm_gib=("xla_planned_hbm_gib_per_chip", "median"),
          good_chunk_loop_count=("chunk_loop_count", "median"),
      )
      .reset_index()
  )
  matched = bad.merge(good, on=key, how="inner")
  if matched.empty:
    return matched
  matched["bad_good_step_time_ratio"] = (
      matched["bad_step_time_sec"] / matched["good_step_time_sec"]
  )
  matched["bad_good_compile_time_ratio"] = (
      matched["bad_compile_time_sec"] / matched["good_compile_time_sec"]
  )
  matched["bad_good_hbm_ratio"] = matched["bad_hbm_gib"] / matched["good_hbm_gib"]
  matched["loop_count_ratio"] = (
      matched["bad_chunk_loop_count"] / matched["good_chunk_loop_count"]
  )
  return matched.sort_values("bad_good_step_time_ratio", ascending=False)


def build_hypothesis(rows: pd.DataFrame, bad_good: pd.DataFrame) -> str:
  hardware_count = rows["hardware_target"].nunique() if "hardware_target" in rows else 0
  workload_count = rows["workload_family"].nunique() if "workload_family" in rows else 0
  lines = [
      "# Mesh/Chunk Pathology Hypothesis Report",
      "",
      "Question: does the small-chunk slowdown generalize beyond CCE and beyond TPU v5e?",
      "",
      f"- Hardware targets observed: {hardware_count}",
      f"- Workload families observed: {workload_count}",
      f"- Result rows: {len(rows)}",
      f"- Successful rows: {int(rows['ok'].sum()) if 'ok' in rows else 0}",
      "",
  ]
  if bad_good.empty:
    lines.append("Bad/good comparisons are not available yet.")
  else:
    repeated = bad_good[bad_good["bad_good_step_time_ratio"] > 2.0]
    lines.extend([
        f"Bad/good matched groups: {len(bad_good)}",
        f"Groups with >2x bad/good slowdown: {len(repeated)}",
        "",
        "Interpretation guide:",
        "- If only `cce_train` shows a large ratio, suspect CCE implementation/lowering.",
        "- If `projection_collective_loop` also shows it, suspect chunked matmul plus TP collectives.",
        "- If `collective_loop` shows it, suspect collective granularity/latency directly.",
        "- If `chunked_matmul_loop` shows it without collectives, suspect compile/lowering/kernel launch or tiny matmul shape.",
    ])
  return "\n".join(lines) + "\n"


def plot_bad_good(bad_good: pd.DataFrame, path: Path) -> None:
  if bad_good.empty:
    return
  part = bad_good.copy()
  part["label"] = part["hardware_target"] + "\n" + part["workload_family"]
  fig, ax = plt.subplots(figsize=(12, 5.5))
  ax.bar(part["label"], part["bad_good_step_time_ratio"], color="#3f6f8f")
  ax.axhline(1.0, color="black", linewidth=1)
  ax.axhline(2.0, color="#c94f4f", linewidth=1, linestyle="--")
  ax.set_ylabel("bad / good step-time ratio")
  ax.set_title("Small-chunk slowdown by hardware and workload")
  ax.tick_params(axis="x", rotation=45)
  ax.grid(axis="y", alpha=0.25)
  fig.tight_layout()
  fig.savefig(path, dpi=170)
  plt.close(fig)


def safe_slug(value: str) -> str:
  return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def format_step_time(value: float) -> str:
  if value < 0.001:
    return f"{value * 1_000_000:.0f}us"
  if value < 1.0:
    return f"{value * 1000:.1f}ms"
  return f"{value:.2f}s"


def successful_step_rows(rows: pd.DataFrame) -> pd.DataFrame:
  if rows.empty:
    return pd.DataFrame()
  return rows[
      rows["ok"]
      & rows["steady_state_mean_step_time_sec"].notna()
      & rows["workload_family"].isin(WORKLOAD_ORDER)
      & rows["signature_label"].isin(SIGNATURE_ORDER)
  ].copy()


def plot_step_time_by_signature(rows: pd.DataFrame, hardware: str, outdir: Path) -> None:
  part = successful_step_rows(rows)
  part = part[part["hardware_target"].astype(str).eq(hardware)]
  if part.empty:
    return

  fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
  axes_flat = axes.flatten()
  for ax, workload in zip(axes_flat, WORKLOAD_ORDER):
    subset = part[part["workload_family"].eq(workload)]
    values = []
    labels = []
    colors = []
    for signature in SIGNATURE_ORDER:
      row = subset[subset["signature_label"].eq(signature)]
      if row.empty:
        continue
      values.append(float(row["steady_state_mean_step_time_sec"].iloc[0]))
      labels.append(SIGNATURE_LABELS[signature])
      colors.append(SIGNATURE_COLORS[signature])

    if not values:
      ax.axis("off")
      continue
    bars = ax.bar(range(len(values)), values, color=colors, width=0.66)
    ymax = max(values) * 1.35 if values else 1.0
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(WORKLOAD_LABELS[workload])
    ax.set_ylabel("step time (s)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + ymax * 0.03,
          format_step_time(value),
          ha="center",
          va="bottom",
          fontsize=8,
      )

  fig.suptitle(f"{hardware}: step time by workload and experiment variable", y=0.995)
  fig.tight_layout()
  fig.savefig(outdir / f"step_time_by_workload_signature_{safe_slug(hardware)}.png", dpi=170)
  plt.close(fig)


def plot_normalized_signature_heatmap(rows: pd.DataFrame, hardware: str, outdir: Path) -> None:
  part = successful_step_rows(rows)
  part = part[part["hardware_target"].astype(str).eq(hardware)]
  if part.empty:
    return

  matrix: list[list[float]] = []
  annotations: list[list[str]] = []
  ylabels: list[str] = []
  for workload in WORKLOAD_ORDER:
    subset = part[part["workload_family"].eq(workload)]
    good = subset[subset["signature_label"].eq("good")]
    if good.empty:
      continue
    good_time = float(good["steady_state_mean_step_time_sec"].iloc[0])
    if good_time <= 0:
      continue
    row_values = []
    row_labels = []
    for signature in SIGNATURE_ORDER:
      match = subset[subset["signature_label"].eq(signature)]
      if match.empty:
        row_values.append(float("nan"))
        row_labels.append("")
        continue
      ratio = float(match["steady_state_mean_step_time_sec"].iloc[0]) / good_time
      row_values.append(ratio)
      row_labels.append(f"{ratio:.2f}x")
    matrix.append(row_values)
    annotations.append(row_labels)
    ylabels.append(WORKLOAD_LABELS[workload])

  if not matrix:
    return

  log_matrix = pd.DataFrame(matrix).map(
      lambda value: math.log2(value) if pd.notna(value) and value > 0 else float("nan")
  )
  finite = [
      abs(float(value))
      for row in log_matrix.to_numpy().tolist()
      for value in row
      if pd.notna(value)
  ]
  bound = max(finite) if finite else 1.0
  bound = max(bound, 1.0)

  fig, ax = plt.subplots(figsize=(9.6, 4.9))
  image = ax.imshow(
      log_matrix,
      cmap="coolwarm",
      norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
      aspect="auto",
  )
  ax.set_xticks(range(len(SIGNATURE_ORDER)))
  ax.set_xticklabels([SIGNATURE_LABELS[item] for item in SIGNATURE_ORDER], fontsize=8)
  ax.set_yticks(range(len(ylabels)))
  ax.set_yticklabels(ylabels)
  ax.set_title(f"{hardware}: step time normalized to the good row")
  for y, row in enumerate(annotations):
    for x, label in enumerate(row):
      if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
  cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
  cbar.set_label("log2(step time / good)")
  fig.tight_layout()
  fig.savefig(outdir / f"normalized_step_time_by_workload_signature_{safe_slug(hardware)}.png", dpi=170)
  plt.close(fig)


def plot_variable_views(rows: pd.DataFrame, outdir: Path) -> None:
  if rows.empty or "hardware_target" not in rows:
    return
  for hardware in sorted(rows["hardware_target"].dropna().astype(str).unique()):
    plot_step_time_by_signature(rows, hardware, outdir)
    plot_normalized_signature_heatmap(rows, hardware, outdir)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, action="append", default=[])
  parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
  parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  outdir = args.outdir.expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  inputs = discover_inputs(args)
  frames = [read_input(path) for path in inputs]
  rows = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
  rows = normalize(rows)
  scenario_summary = build_scenario_summary(rows)
  bad_good = build_bad_good(rows)

  write_frame(rows, outdir / "normalized_results.csv")
  write_frame(scenario_summary, outdir / "scenario_summary.csv")
  write_frame(bad_good, outdir / "bad_good_ratios.csv")
  (outdir / "hypothesis_report.md").write_text(build_hypothesis(rows, bad_good))
  plot_bad_good(bad_good, outdir / "bad_good_slowdown_by_hardware_workload.png")
  plot_variable_views(rows, outdir)

  print(f"inputs={len(inputs)}")
  print(f"rows={len(rows)}")
  print(f"outdir={outdir}")


if __name__ == "__main__":
  main()
