#!/usr/bin/env python3
"""Aggregate dataset/max-length packing sweeps for the 02 report."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tarfile
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
PROCESSED = DATA / "processed"
RAW = DATA / "raw_artifacts"
PROFILE_DIR = ROOT / "results" / "dataset-profile-270m"

DATASET_ORDER = ["opus100", "alpaca", "oasst1"]
DATASET_LABELS = {
    "opus100": "OPUS EN-FR",
    "opus100_en_fr": "OPUS EN-FR",
    "alpaca": "Alpaca",
    "oasst1": "OASST1 EN",
    "oasst1_en": "OASST1 EN",
}
PROFILE_TO_SLUG = {
    "opus100_en_fr": "opus100",
    "alpaca": "alpaca",
    "oasst1_en": "oasst1",
}
ARTIFACT_DIRS = {
    "opus100": "gemma3_270m_dataset_sweep_opus100_v5litepod1",
    "alpaca": "gemma3_270m_dataset_sweep_alpaca_v5litepod1",
    "oasst1": "gemma3_270m_dataset_sweep_oasst1_v5litepod1",
}
MISSING_TPU_ARTIFACTS: set[str] = set()

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#C44E52"
GRAY = "#5F6B7A"
GRID = "#E7E9ED"


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open() as f:
    return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  keys: list[str] = []
  for row in rows:
    for key in row:
      if key not in keys:
        keys.append(key)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def as_float(value: Any, default: float = math.nan) -> float:
  try:
    if value == "":
      return default
    return float(value)
  except (TypeError, ValueError):
    return default


def as_int(value: Any, default: int = 0) -> int:
  try:
    if value == "":
      return default
    return int(float(value))
  except (TypeError, ValueError):
    return default


def ensure_extracted(base_dir: Path, expected_child: str) -> None:
  expected = base_dir / expected_child
  if expected.exists():
    return
  archives = sorted(base_dir.glob("*.tgz")) + sorted(base_dir.glob("*.tar.gz"))
  if not archives:
    raise FileNotFoundError(f"No raw archive found in {base_dir}")
  with tarfile.open(archives[0], "r:*") as archive:
    try:
      archive.extractall(base_dir, filter="data")
    except TypeError:
      archive.extractall(base_dir)


def normalize_profile_row(row: dict[str, str]) -> dict[str, Any] | None:
  slug = PROFILE_TO_SLUG.get(row["dataset"])
  if slug not in DATASET_ORDER:
    return None
  return {
      **row,
      "dataset_slug": slug,
      "dataset_label": DATASET_LABELS[slug],
      "batch_size": as_int(row["batch_size"]),
      "max_length": as_int(row["max_length"]),
      "examples": as_int(row["examples"]),
      "loss_tokens_retained_ratio": as_float(
          row["loss_tokens_retained_ratio"]
      ),
      "overlength_rate": as_float(row["overlength_rate"]),
      "packed_loss_ratio": as_float(row["packed_loss_ratio"]),
      "packed_valid_ratio": as_float(row["packed_valid_ratio"]),
      "packed_vs_fixed_loss_gain_x": as_float(
          row["packed_vs_fixed_loss_gain_x"]
      ),
      "packed_vs_fixed_valid_gain_x": as_float(
          row["packed_vs_fixed_valid_gain_x"]
      ),
      "row_reduction_x": as_float(row["row_reduction_x"]),
      "unpacked_fixed_loss_ratio": as_float(
          row["unpacked_fixed_loss_ratio"]
      ),
      "unpacked_fixed_valid_ratio": as_float(
          row["unpacked_fixed_valid_ratio"]
      ),
  }


def load_profile_rows() -> list[dict[str, Any]]:
  profile_path = PROFILE_DIR / "dataset_profile.csv"
  rows = []
  for row in read_csv(profile_path):
    normalized = normalize_profile_row(row)
    if normalized is not None:
      rows.append(normalized)
  write_csv(PROCESSED / "gemma3_270m_dataset_profile.csv", rows)
  return rows


def normalize_tpu_row(slug: str, row: dict[str, str]) -> dict[str, Any]:
  return {
      **row,
      "dataset_slug": slug,
      "dataset_label": DATASET_LABELS[slug],
      "batch_size": as_int(row.get("batch_size")),
      "max_length": as_int(row.get("max_length")),
      "status": row.get("status", ""),
      "variant": row.get("variant", ""),
      "failure_type": row.get("failure_type", ""),
      "packed_efficiency": as_float(row.get("packed_efficiency")),
      "row_reduction_x": as_float(row.get("row_reduction_x")),
      "loss_tokens_per_sec_excl_first": as_float(
          row.get("loss_tokens_per_sec_excl_first")
      ),
      "valid_tokens_per_sec_excl_first": as_float(
          row.get("valid_tokens_per_sec_excl_first")
      ),
      "mean_step_time_sec_excl_first": as_float(
          row.get("mean_step_time_sec_excl_first")
      ),
      "xla_train_step_gib_per_chip": as_float(
          row.get("xla_train_step_gib_per_chip")
      ),
      "oom_used_gib": as_float(row.get("oom_used_gib")),
      "oom_limit_gib": as_float(row.get("oom_limit_gib")),
      "final_cumulative_loss_tokens": as_int(
          row.get("final_cumulative_loss_tokens")
      ),
      "final_cumulative_valid_tokens": as_int(
          row.get("final_cumulative_valid_tokens")
      ),
  }


def load_tpu_rows() -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for slug in DATASET_ORDER:
    base_dir = RAW / ARTIFACT_DIRS[slug]
    try:
      ensure_extracted(base_dir, "short-throughput")
    except FileNotFoundError:
      MISSING_TPU_ARTIFACTS.add(slug)
      continue
    csv_path = base_dir / "short-throughput" / "short-throughput_results.csv"
    rows.extend(normalize_tpu_row(slug, row) for row in read_csv(csv_path))
  write_csv(PROCESSED / "gemma3_270m_dataset_tpu_sweep_v5litepod1.csv", rows)
  return rows


def setup_axis(axis: plt.Axes) -> None:
  axis.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
  axis.spines["top"].set_visible(False)
  axis.spines["right"].set_visible(False)
  axis.spines["left"].set_color("#D5DAE1")
  axis.spines["bottom"].set_color("#D5DAE1")
  axis.tick_params(colors="#2B3440")


def ratio_formatter(value: float, _pos: int) -> str:
  if value < 1:
    return f"{value:.1f}x"
  return f"{value:g}x"


def pct_formatter(value: float, _pos: int) -> str:
  return f"{value * 100:.0f}%"


def tps_formatter(value: float, _pos: int) -> str:
  if value >= 10000:
    return f"{value / 1000:.0f}k"
  if value >= 1000:
    return f"{value / 1000:.1f}k"
  return f"{value:.0f}"


def json_value(value: Any) -> Any:
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def row_for(
    rows: list[dict[str, Any]],
    *,
    dataset_slug: str,
    batch_size: int,
    max_length: int,
    variant: str | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
  for row in rows:
    if row["dataset_slug"] != dataset_slug:
      continue
    if row["batch_size"] != batch_size or row["max_length"] != max_length:
      continue
    if variant is not None and row.get("variant") != variant:
      continue
    if status is not None and row.get("status") != status:
      continue
    return row
  return None


def plot_dataset_profile_and_tpu(
    profile_rows: list[dict[str, Any]],
    tpu_rows: list[dict[str, Any]],
) -> Path:
  fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
  fig.subplots_adjust(
      left=0.075,
      right=0.985,
      top=0.76,
      bottom=0.20,
      wspace=0.12,
  )
  colors = {
      "opus100": BLUE,
      "alpaca": ORANGE,
      "oasst1": GREEN,
  }
  lengths = [256, 512, 1024, 2048]

  axis = axes[0]
  for slug in DATASET_ORDER:
    values = []
    for length in lengths:
      row = row_for(
          profile_rows,
          dataset_slug=slug,
          batch_size=16,
          max_length=length,
      )
      values.append(row["packed_vs_fixed_loss_gain_x"] if row else math.nan)
    axis.plot(
        lengths,
        values,
        marker="o",
        markersize=7,
        linewidth=2.4,
        label=DATASET_LABELS[slug],
        color=colors[slug],
      )
  axis.set_title("Tokenizer-only target-density opportunity", fontsize=13)
  axis.set_xlabel("Max length")
  axis.set_ylabel("Packed / fixed target density")
  axis.set_xscale("log", base=2)
  axis.set_yscale("log", base=2)
  axis.set_xticks(lengths)
  axis.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
  axis.yaxis.set_major_formatter(FuncFormatter(ratio_formatter))
  setup_axis(axis)

  axis = axes[1]
  for slug in DATASET_ORDER:
    if slug in MISSING_TPU_ARTIFACTS:
      continue
    values = []
    for length in lengths:
      unpacked = row_for(
          tpu_rows,
          dataset_slug=slug,
          batch_size=4,
          max_length=length,
          variant="unpacked",
          status="ok",
      )
      packed = row_for(
          tpu_rows,
          dataset_slug=slug,
          batch_size=4,
          max_length=length,
          variant="packed",
          status="ok",
      )
      if not unpacked or not packed:
        values.append(math.nan)
        continue
      base = unpacked["loss_tokens_per_sec_excl_first"]
      value = packed["loss_tokens_per_sec_excl_first"]
      values.append(value / base if base > 0 else math.nan)
    axis.plot(
        lengths,
        values,
        marker="o",
        markersize=7,
        linewidth=2.4,
        label=DATASET_LABELS[slug],
        color=colors[slug],
      )
  axis.axhline(1, color="#A9B0BA", linewidth=1.2, linestyle="--", zorder=1)
  axis.text(260, 1.08, "no gain", color=GRAY, fontsize=9)
  if MISSING_TPU_ARTIFACTS:
    missing = ", ".join(DATASET_LABELS[slug] for slug in MISSING_TPU_ARTIFACTS)
    axis.text(
        0.98,
        0.05,
        f"TPU artifact not collected: {missing}",
        ha="right",
        va="bottom",
        transform=axis.transAxes,
        color=GRAY,
        fontsize=9,
    )
  axis.set_title("Measured TPU target-token throughput gain", fontsize=13)
  axis.set_xlabel("Max length")
  axis.set_ylabel("Packed / unpacked target tok/s")
  axis.set_xscale("log", base=2)
  axis.set_yscale("log", base=2)
  axis.set_xticks(lengths)
  axis.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
  axis.yaxis.set_major_formatter(FuncFormatter(ratio_formatter))
  setup_axis(axis)

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
      handles,
      labels,
      loc="lower center",
      ncol=len(DATASET_ORDER),
      frameon=False,
      bbox_to_anchor=(0.5, -0.03),
  )
  fig.suptitle(
      "Gemma3 270M packing depends on dataset shape, not model size",
      fontsize=17,
      fontweight="bold",
      y=0.98,
  )
  fig.text(
      0.5,
      0.895,
      "Local tokenizer profile: 5k examples, batch 16. TPU run: Cloud TPU v5litepod-1, one chip, b4, 50 steps, long examples truncated.",
      ha="center",
      color=GRAY,
      fontsize=10.5,
  )
  path = ASSETS / "gemma3_270m_dataset_ablation.png"
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return path


def plot_tpu_absolute_throughput(tpu_rows: list[dict[str, Any]]) -> Path:
  lengths = [512, 1024, 2048]
  fig, axes = plt.subplots(
      1,
      len(DATASET_ORDER),
      figsize=(13.2, 4.9),
      sharey=False,
      squeeze=False,
  )
  fig.subplots_adjust(
      left=0.075,
      right=0.985,
      top=0.76,
      bottom=0.24,
      wspace=0.18,
  )
  for axis, slug in zip(axes.ravel(), DATASET_ORDER):
    if slug in MISSING_TPU_ARTIFACTS:
      axis.set_title(DATASET_LABELS[slug], fontsize=12.5)
      axis.text(
          0.5,
          0.52,
          "TPU artifact\nnot collected",
          ha="center",
          va="center",
          transform=axis.transAxes,
          fontsize=11,
          color=GRAY,
      )
      axis.set_xticks([])
      axis.set_yticks([])
      setup_axis(axis)
      continue
    x = np.arange(len(lengths))
    width = 0.36
    unpacked_values = []
    packed_values = []
    for length in lengths:
      unpacked = row_for(
          tpu_rows,
          dataset_slug=slug,
          batch_size=4,
          max_length=length,
          variant="unpacked",
          status="ok",
      )
      packed = row_for(
          tpu_rows,
          dataset_slug=slug,
          batch_size=4,
          max_length=length,
          variant="packed",
          status="ok",
      )
      unpacked_values.append(
          unpacked["loss_tokens_per_sec_excl_first"] if unpacked else math.nan
      )
      packed_values.append(
          packed["loss_tokens_per_sec_excl_first"] if packed else math.nan
      )
    bars_a = axis.bar(
        x - width / 2,
        unpacked_values,
        width,
        label="Unpacked",
        color=BLUE,
        zorder=3,
    )
    bars_b = axis.bar(
        x + width / 2,
        packed_values,
        width,
        label="Packed",
        color=ORANGE,
        zorder=3,
    )
    for bars in (bars_a, bars_b):
      for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height) or height <= 0:
          continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            tps_formatter(height, 0),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#20262E",
        )
    axis.set_title(DATASET_LABELS[slug], fontsize=12.5)
    axis.set_xticks(x, [str(v) for v in lengths])
    axis.set_xlabel("Max length")
    axis.set_ylabel("Target tok/s")
    axis.yaxis.set_major_formatter(FuncFormatter(tps_formatter))
    setup_axis(axis)
  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(
      handles,
      labels,
      loc="lower center",
      ncol=2,
      frameon=False,
      bbox_to_anchor=(0.5, 0.02),
  )
  fig.suptitle(
      "Measured target-token throughput, same Gemma3 270M TPU setup",
      fontsize=16,
      fontweight="bold",
      y=0.98,
  )
  fig.text(
      0.5,
      0.885,
      "Cloud TPU v5litepod-1, one chip, batch 4, 50-step short-throughput runs. Bars show target tokens per second excluding the first step.",
      ha="center",
      color=GRAY,
      fontsize=10.5,
  )
  path = ASSETS / "gemma3_270m_dataset_tpu_absolute_throughput.png"
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return path


def write_summary(
    profile_rows: list[dict[str, Any]],
    tpu_rows: list[dict[str, Any]],
) -> Path:
  focus = []
  for slug in DATASET_ORDER:
    for length in [512, 1024, 2048]:
      profile = row_for(
          profile_rows,
          dataset_slug=slug,
          batch_size=16,
          max_length=length,
      )
      unpacked = row_for(
          tpu_rows,
          dataset_slug=slug,
          batch_size=4,
          max_length=length,
          variant="unpacked",
          status="ok",
      )
      packed = row_for(
          tpu_rows,
          dataset_slug=slug,
          batch_size=4,
          max_length=length,
          variant="packed",
          status="ok",
      )
      tpu_gain = math.nan
      if unpacked and packed:
        base = unpacked["loss_tokens_per_sec_excl_first"]
        value = packed["loss_tokens_per_sec_excl_first"]
        tpu_gain = value / base if base > 0 else math.nan
      if math.isfinite(tpu_gain):
        tpu_status = "ok"
        tpu_note = None
      elif slug in MISSING_TPU_ARTIFACTS:
        tpu_status = "artifact_missing"
        tpu_note = None
      else:
        tpu_status = "no_matched_tpu_pair"
        tpu_note = None
      focus.append({
          "dataset": DATASET_LABELS[slug],
          "max_length": length,
          "profile_target_density_gain_x": json_value(
              profile["packed_vs_fixed_loss_gain_x"] if profile else math.nan
          ),
          "profile_target_retained_ratio": json_value(
              profile["loss_tokens_retained_ratio"] if profile else math.nan
          ),
          "tpu_b4_target_throughput_gain_x": json_value(tpu_gain),
          "unpacked_target_tps": json_value(
              unpacked["loss_tokens_per_sec_excl_first"]
              if unpacked
              else math.nan
          ),
          "packed_target_tps": json_value(
              packed["loss_tokens_per_sec_excl_first"] if packed else math.nan
          ),
          "tpu_status": tpu_status,
          "tpu_note": tpu_note,
      })
  payload = {
      "model": "google/gemma-3-270m-it",
      "tpu": "Cloud TPU v5litepod-1",
      "chips": 1,
      "mesh": "fsdp=1,tp=1",
      "short_throughput_steps": 50,
      "profile_examples": 5000,
      "long_example_policy": "truncate",
      "missing_tpu_artifacts": sorted(MISSING_TPU_ARTIFACTS),
      "focus_rows": focus,
  }
  path = PROCESSED / "gemma3_270m_dataset_ablation_summary.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2) + "\n")
  return path


def main() -> None:
  ASSETS.mkdir(parents=True, exist_ok=True)
  PROCESSED.mkdir(parents=True, exist_ok=True)
  profile_rows = load_profile_rows()
  tpu_rows = load_tpu_rows()
  profile_plot = plot_dataset_profile_and_tpu(profile_rows, tpu_rows)
  absolute_plot = plot_tpu_absolute_throughput(tpu_rows)
  summary_path = write_summary(profile_rows, tpu_rows)
  print(f"profile_plot={profile_plot}")
  print(f"absolute_plot={absolute_plot}")
  print(f"summary={summary_path}")


if __name__ == "__main__":
  main()
