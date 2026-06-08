#!/usr/bin/env python3
"""Aggregate Qwen3 0.6B packing transfer results and figures."""

from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UNPACKED = DATA / "processed" / "qwen3_0p6b_unpack"
PROCESSED = DATA / "processed"
ASSETS = ROOT / "assets"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#2E7D32"
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


def as_float(value: Any) -> float:
  try:
    if value in ("", None):
      return float("nan")
    return float(value)
  except (TypeError, ValueError):
    return float("nan")


def as_int(value: Any) -> int:
  try:
    if value in ("", None):
      return 0
    return int(float(value))
  except (TypeError, ValueError):
    return 0


def normalize_row(row: dict[str, str], chip_count: int) -> dict[str, Any]:
  dataset = row.get("dataset_mode", "")
  if dataset == "opus100":
    dataset_label = "OPUS100"
  elif dataset == "alpaca":
    dataset_label = "Alpaca"
  elif dataset == "oasst1":
    dataset_label = "OASST1"
  else:
    dataset_label = dataset
  return {
      **row,
      "dataset_label": dataset_label,
      "chip_count": chip_count,
      "tpu": f"v5litepod-{chip_count}",
      "batch_size": as_int(row.get("batch_size")),
      "max_length": as_int(row.get("max_length")),
      "xla_train_step_gib_per_chip": as_float(
          row.get("xla_train_step_gib_per_chip")
      ),
      "packed_efficiency": as_float(row.get("packed_efficiency")),
      "row_reduction_x": as_float(row.get("row_reduction_x")),
      "loss_tokens_per_sec_excl_first": as_float(
          row.get("loss_tokens_per_sec_excl_first")
      ),
      "mean_step_time_sec_excl_first": as_float(
          row.get("mean_step_time_sec_excl_first")
      ),
  }


def setup_axis(axis: plt.Axes) -> None:
  axis.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
  axis.spines["top"].set_visible(False)
  axis.spines["right"].set_visible(False)
  axis.spines["left"].set_color("#D5DAE1")
  axis.spines["bottom"].set_color("#D5DAE1")
  axis.tick_params(colors="#2B3440")


def load_rows() -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for path in sorted(UNPACKED.glob("**/*_results.csv")):
    if "prepare" in str(path):
      continue
    chip_count = 4 if "packing4" in str(path) else 1
    for row in read_csv(path):
      rows.append(normalize_row(row, chip_count))
  raw_dir = DATA / "raw_artifacts" / "qwen3_0p6b"
  for path in sorted(raw_dir.glob("*.tar.gz")):
    chip_count = 4 if "pack4" in path.name else 1
    with tarfile.open(path, "r:gz") as tar:
      for member in tar.getmembers():
        if not member.isfile():
          continue
        if not member.name.endswith("_results.csv") or "prepare" in member.name:
          continue
        extracted = tar.extractfile(member)
        if extracted is None:
          continue
        with io.TextIOWrapper(extracted, encoding="utf-8") as f:
          for row in csv.DictReader(f):
            rows.append(normalize_row(row, chip_count))
  return rows


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  index: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
  for row in rows:
    key = (
        row["chip_count"],
        row.get("dataset_mode"),
        row["batch_size"],
        row["max_length"],
    )
    index.setdefault(key, {})[row.get("variant", "")] = row

  pairs: list[dict[str, Any]] = []
  for key, variants in sorted(index.items()):
    if "unpacked" not in variants or "packed" not in variants:
      continue
    unpacked = variants["unpacked"]
    packed = variants["packed"]
    unpacked_status = unpacked.get("status")
    packed_status = packed.get("status")
    gain = float("nan")
    if unpacked_status == "ok" and packed_status == "ok":
      gain = (
          packed["loss_tokens_per_sec_excl_first"]
          / unpacked["loss_tokens_per_sec_excl_first"]
      )
    pairs.append({
        "chip_count": key[0],
        "tpu": f"v5litepod-{key[0]}",
        "dataset_mode": key[1],
        "dataset_label": unpacked["dataset_label"],
        "batch_size": key[2],
        "max_length": key[3],
        "unpacked_status": unpacked_status,
        "packed_status": packed_status,
        "unpacked_density": unpacked["packed_efficiency"],
        "packed_density": packed["packed_efficiency"],
        "unpacked_loss_tokens_per_sec": unpacked[
            "loss_tokens_per_sec_excl_first"
        ],
        "packed_loss_tokens_per_sec": packed[
            "loss_tokens_per_sec_excl_first"
        ],
        "throughput_gain_x": gain,
        "unpacked_xla_gib_per_chip": unpacked["xla_train_step_gib_per_chip"],
        "packed_xla_gib_per_chip": packed["xla_train_step_gib_per_chip"],
    })
  return pairs


def plot(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
  ASSETS.mkdir(parents=True, exist_ok=True)
  figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
  figure.patch.set_facecolor("white")

  # Passing matched gains.
  axis = axes[0]
  passing = [
      row
      for row in pairs
      if row["unpacked_status"] == "ok" and row["packed_status"] == "ok"
  ]
  passing.sort(key=lambda r: (r["chip_count"], r["max_length"], r["dataset_label"], r["batch_size"]))
  labels = [
      f"{row['dataset_label']}\n{row['tpu']} b{row['batch_size']}/L{row['max_length']}"
      for row in passing
  ]
  gains = [row["throughput_gain_x"] for row in passing]
  x = np.arange(len(passing))
  axis.bar(x, gains, color=GREEN, zorder=3)
  for xpos, gain in zip(x, gains):
    axis.text(xpos, gain + max(gains) * 0.025, f"{gain:.1f}x", ha="center", va="bottom", fontsize=8)
  axis.set_title("Useful-token throughput gain where both shapes fit", loc="left", fontsize=13, weight="bold")
  axis.set_ylabel("Packed / unpacked target-token throughput")
  axis.set_xticks(x, labels, rotation=35, ha="right")
  axis.set_ylim(0, max(gains) * 1.22)
  setup_axis(axis)

  # Fit status and memory neutrality.
  axis = axes[1]
  focus = [
      row for row in pairs
      if row["chip_count"] == 4 and row["batch_size"] == 4 and row["max_length"] in (1024, 2048)
  ]
  focus.sort(key=lambda r: (r["dataset_label"], r["max_length"]))
  labels = [f"{row['dataset_label']}\nL{row['max_length']}" for row in focus]
  unpacked_xla = [row["unpacked_xla_gib_per_chip"] for row in focus]
  packed_xla = [row["packed_xla_gib_per_chip"] for row in focus]
  x = np.arange(len(focus))
  width = 0.36
  axis.bar(x - width / 2, unpacked_xla, width, color=BLUE, label="unpacked", zorder=3)
  axis.bar(x + width / 2, packed_xla, width, color=ORANGE, label="packed", zorder=3)
  for xpos, row, value in zip(x - width / 2, focus, unpacked_xla):
    status = "OK" if row["unpacked_status"] == "ok" else "OOM"
    axis.text(xpos, value + 0.7, status, ha="center", va="bottom", fontsize=8)
  for xpos, row, value in zip(x + width / 2, focus, packed_xla):
    status = "OK" if row["packed_status"] == "ok" else "OOM"
    axis.text(xpos, value + 0.7, status, ha="center", va="bottom", fontsize=8)
  axis.set_title("Packing does not change the fixed-shape fit wall", loc="left", fontsize=13, weight="bold")
  axis.set_ylabel("XLA planned HBM per chip")
  axis.set_xticks(x, labels)
  axis.legend(frameon=False, loc="upper left")
  axis.set_ylim(0, max(unpacked_xla + packed_xla) * 1.18)
  setup_axis(axis)

  figure.suptitle(
      "Qwen3 0.6B sequence-packing transfer on Cloud TPU v5e",
      x=0.01,
      ha="left",
      fontsize=16,
      weight="bold",
  )
  figure.savefig(ASSETS / "qwen3_0p6b_packing_transfer.png", dpi=180, bbox_inches="tight")
  plt.close(figure)


def main() -> None:
  rows = load_rows()
  pairs = build_pairs(rows)
  write_csv(PROCESSED / "qwen3_0p6b_packing_transfer.csv", rows)
  write_csv(PROCESSED / "qwen3_0p6b_packing_transfer_pairs.csv", pairs)
  plot(rows, pairs)


if __name__ == "__main__":
  main()
