#!/usr/bin/env python3
"""Aggregate Gemma3 1B / Gemma4 E2B packing transfer checks."""

from __future__ import annotations

import csv
import math
import tarfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
PROCESSED = DATA / "processed"
RAW_TRANSFER = DATA / "transfer_1b_e2b" / "raw"

DATASET_ORDER = ["opus100", "alpaca", "oasst1"]
DATASET_LABELS = {
    "opus100": "OPUS EN-FR",
    "alpaca": "Alpaca",
    "oasst1": "OASST1 EN",
}
COLORS = {
    "unpacked": "#4C78A8",
    "packed": "#F58518",
}
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


def setup_axis(axis: plt.Axes) -> None:
  axis.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
  axis.spines["top"].set_visible(False)
  axis.spines["right"].set_visible(False)
  axis.spines["left"].set_color("#D5DAE1")
  axis.spines["bottom"].set_color("#D5DAE1")
  axis.tick_params(colors="#2B3440")


def ensure_gemma3_1b_raw_base() -> Path:
  extracted_root = RAW_TRANSFER / "gemma3_1b_transfer32_tp4"
  base = extracted_root / "gemma3-1b-transfer32-tp4"
  if base.exists():
    return base
  tar_path = RAW_TRANSFER / "gemma3_1b_transfer32_tp4_results.tar.gz"
  if not tar_path.exists():
    raise FileNotFoundError(
        f"Missing Gemma3 1B transfer artifact: {tar_path}. "
        "Collect the TPU tarball before regenerating transfer plots."
    )
  extracted_root.mkdir(parents=True, exist_ok=True)
  with tarfile.open(tar_path) as tar:
    tar.extractall(extracted_root)
  return base


def load_gemma3_1b_rows() -> list[dict[str, Any]]:
  base = ensure_gemma3_1b_raw_base()
  rows: list[dict[str, Any]] = []
  for dataset in DATASET_ORDER:
    csv_path = base / dataset / "short-throughput" / "short-throughput_results.csv"
    for row in read_csv(csv_path):
      rows.append({
          **row,
          "model_label": "Gemma3 1B",
          "dataset_slug": dataset,
          "dataset_label": DATASET_LABELS[dataset],
          "batch_size": as_int(row.get("batch_size")),
          "max_length": as_int(row.get("max_length")),
          "mesh_fsdp": as_int(row.get("mesh_fsdp")),
          "mesh_tp": as_int(row.get("mesh_tp")),
          "chips": as_int(row.get("chips")),
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
          "wall_time_sec": as_float(row.get("wall_time_sec")),
          "xla_train_step_gib_per_chip": as_float(
              row.get("xla_train_step_gib_per_chip")
          ),
          "final_cumulative_loss_tokens": as_int(
              row.get("final_cumulative_loss_tokens")
          ),
          "final_cumulative_valid_tokens": as_int(
              row.get("final_cumulative_valid_tokens")
          ),
      })
  return rows


def make_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  by_key = {
      (
          row["dataset_slug"],
          row["batch_size"],
          row["max_length"],
          row["variant"],
      ): row
      for row in rows
      if row.get("status") == "ok"
  }
  pairs: list[dict[str, Any]] = []
  for dataset in DATASET_ORDER:
    for batch_size in (4, 8):
      for max_length in (512, 1024, 2048):
        unpacked = by_key[(dataset, batch_size, max_length, "unpacked")]
        packed = by_key[(dataset, batch_size, max_length, "packed")]
        pairs.append({
            "model_label": "Gemma3 1B",
            "dataset_slug": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "batch_size": batch_size,
            "max_length": max_length,
            "tpu": "v5litepod-32",
            "chips": 32,
            "mesh": "fsdp=8,tp=4",
            "unpacked_density": unpacked["packed_efficiency"],
            "packed_density": packed["packed_efficiency"],
            "density_uplift_x": (
                packed["packed_efficiency"] / unpacked["packed_efficiency"]
            ),
            "unpacked_loss_tokens_per_sec": unpacked[
                "loss_tokens_per_sec_excl_first"
            ],
            "packed_loss_tokens_per_sec": packed[
                "loss_tokens_per_sec_excl_first"
            ],
            "loss_token_throughput_uplift_x": (
                packed["loss_tokens_per_sec_excl_first"]
                / unpacked["loss_tokens_per_sec_excl_first"]
            ),
            "unpacked_step_time_sec": unpacked[
                "mean_step_time_sec_excl_first"
            ],
            "packed_step_time_sec": packed["mean_step_time_sec_excl_first"],
            "step_time_ratio_packed_over_unpacked": (
                packed["mean_step_time_sec_excl_first"]
                / unpacked["mean_step_time_sec_excl_first"]
            ),
            "unpacked_xla_gib_per_chip": unpacked[
                "xla_train_step_gib_per_chip"
            ],
            "packed_xla_gib_per_chip": packed["xla_train_step_gib_per_chip"],
            "xla_gib_delta_per_chip": (
                packed["xla_train_step_gib_per_chip"]
                - unpacked["xla_train_step_gib_per_chip"]
            ),
            "row_reduction_x": packed["row_reduction_x"],
        })
  return pairs


def write_e2b_boundary() -> list[dict[str, Any]]:
  rows = [
      {
          "model_label": "Gemma4 E2B",
          "dataset_slug": "opus100",
          "dataset_label": DATASET_LABELS["opus100"],
          "batch_size": 8,
          "max_length": 2048,
          "variant": "unpacked",
          "tpu": "v5litepod-32",
          "chips": 32,
          "mesh": "fsdp=8,tp=4",
          "status": "resource_exhausted",
          "failure_detail": (
              "all-gather allocation 37,580,963,840 bytes exceeds "
              "17,179,869,184-byte per-chip HBM"
          ),
      },
      {
          "model_label": "Gemma4 E2B",
          "dataset_slug": "opus100",
          "dataset_label": DATASET_LABELS["opus100"],
          "batch_size": 8,
          "max_length": 2048,
          "variant": "unpacked",
          "tpu": "v5litepod-32",
          "chips": 32,
          "mesh": "fsdp=4,tp=8",
          "status": "resource_exhausted",
          "failure_detail": (
              "same all-gather allocation observed under TP=8; current "
              "Gemma4 E2B runner did not shard that buffer further"
          ),
      },
  ]
  write_csv(PROCESSED / "gemma4_e2b_packing_boundary_v5litepod32.csv", rows)
  return rows


def plot_gemma3_1b_transfer(pairs: list[dict[str, Any]]) -> Path:
  fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.8), sharex=True)
  fig.subplots_adjust(left=0.09, right=0.985, top=0.9, bottom=0.13, hspace=0.18)

  labels = []
  x = []
  for i, row in enumerate(pairs):
    labels.append(
        f"{DATASET_LABELS[row['dataset_slug']]}\n"
        f"b{row['batch_size']}, L{row['max_length']}"
    )
    x.append(i)
  x_arr = np.asarray(x)

  throughput = [row["loss_token_throughput_uplift_x"] for row in pairs]
  density_unpacked = [row["unpacked_density"] for row in pairs]
  density_packed = [row["packed_density"] for row in pairs]

  axes[0].bar(x_arr, throughput, color="#F58518", width=0.68, zorder=3)
  axes[0].set_ylabel("Packed / Unpacked\nloss-token throughput")
  axes[0].set_yscale("log")
  axes[0].set_ylim(1, max(throughput) * 1.45)
  axes[0].set_title(
      "Gemma3 1B Packing Transfer: Same HBM Shape, More Useful Tokens",
      fontsize=15,
      fontweight="bold",
      pad=12,
  )
  for xi, value in zip(x_arr, throughput):
    axes[0].text(
        xi,
        value * 1.08,
        f"{value:.1f}x",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#2B3440",
        rotation=0,
      )
  setup_axis(axes[0])

  width = 0.34
  axes[1].bar(
      x_arr - width / 2,
      density_unpacked,
      width=width,
      color=COLORS["unpacked"],
      label="Unpacked",
      zorder=3,
  )
  axes[1].bar(
      x_arr + width / 2,
      density_packed,
      width=width,
      color=COLORS["packed"],
      label="Packed",
      zorder=3,
  )
  axes[1].set_ylabel("Token density\n(loss tokens / padded capacity)")
  axes[1].set_ylim(0, 1.12)
  axes[1].legend(frameon=False, loc="upper left", ncols=2)
  axes[1].set_xticks(x_arr)
  axes[1].set_xticklabels(labels, rotation=38, ha="right", fontsize=8.4)
  setup_axis(axes[1])

  fig.text(
      0.09,
      0.035,
      (
          "Measured on Cloud TPU v5litepod-32 in us-west4-a "
          "(32 chips, mesh fsdp=8,tp=4), 50 LoRA SFT steps per case."
      ),
      fontsize=9,
      color="#5F6B7A",
  )
  path = ASSETS / "gemma3_1b_packing_transfer_v5litepod32.png"
  ASSETS.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=190, bbox_inches="tight")
  plt.close(fig)
  return path


def plot_memory_neutrality(pairs: list[dict[str, Any]]) -> Path:
  fig, axis = plt.subplots(figsize=(10.8, 4.7))
  fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.32)
  labels = [
      f"{DATASET_LABELS[row['dataset_slug']]}\n"
      f"b{row['batch_size']}, L{row['max_length']}"
      for row in pairs
  ]
  x = np.arange(len(pairs))
  delta = [row["xla_gib_delta_per_chip"] for row in pairs]
  colors = ["#54A24B" if value <= 0.05 else "#F58518" for value in delta]
  axis.axhline(0, color="#3A4451", linewidth=1.0, zorder=2)
  axis.bar(x, delta, color=colors, width=0.68, zorder=3)
  max_delta = max(delta) if delta else 0.0
  axis.set_ylim(-0.006, max(0.12, max_delta * 1.35))
  for xi, value in zip(x, delta):
    axis.text(
        xi,
        value + (0.005 if value >= 0 else -0.005),
        f"{value:+.2f}",
        ha="center",
        va="bottom" if value >= 0 else "top",
        fontsize=8.5,
        color="#2B3440",
      )
  axis.set_title(
      "Gemma3 1B Packing Is Throughput-Oriented, Not an HBM Reduction",
      fontsize=14,
      fontweight="bold",
      pad=10,
  )
  axis.set_ylabel("Packed - unpacked\nXLA train-step GiB/chip")
  axis.set_xticks(x)
  axis.set_xticklabels(labels, rotation=38, ha="right", fontsize=8.4)
  setup_axis(axis)
  fig.text(
      0.09,
      0.045,
      (
          "Positive values mean packed used slightly more XLA high-water memory. "
          "The measured deltas stay small while throughput changes by 2.3x-73.2x."
      ),
      fontsize=9,
      color="#5F6B7A",
  )
  path = ASSETS / "gemma3_1b_packing_memory_neutrality_v5litepod32.png"
  fig.savefig(path, dpi=190, bbox_inches="tight")
  plt.close(fig)
  return path


def main() -> None:
  rows = load_gemma3_1b_rows()
  pairs = make_pair_rows(rows)
  e2b = write_e2b_boundary()
  write_csv(PROCESSED / "gemma3_1b_packing_transfer_v5litepod32.csv", rows)
  write_csv(PROCESSED / "gemma3_1b_packing_transfer_pairs_v5litepod32.csv", pairs)
  plot_paths = [
      plot_gemma3_1b_transfer(pairs),
      plot_memory_neutrality(pairs),
  ]
  print(f"wrote rows={len(rows)} pairs={len(pairs)} e2b_boundary={len(e2b)}")
  for path in plot_paths:
    print(path)


if __name__ == "__main__":
  main()
