#!/usr/bin/env python3
"""Create Qwen3 0.6B CCE transfer tables and figures.

The Qwen3 TPU node became SSH-unstable during artifact collection after the
run. The rows below are transcribed from live runner log and CSV-tail checks
captured during the run. They are intentionally small: just the transfer
evidence needed to compare Qwen3 against the Gemma CCE claim.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "qwen3_0p6b_cce_transfer"
ASSETS = ROOT / "assets"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#2E7D32"
RED = "#C44E52"
GRAY = "#5F6B7A"
GRID = "#E7E9ED"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def setup_axis(axis: plt.Axes) -> None:
  axis.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
  axis.spines["top"].set_visible(False)
  axis.spines["right"].set_visible(False)
  axis.spines["left"].set_color("#D5DAE1")
  axis.spines["bottom"].set_color("#D5DAE1")
  axis.tick_params(colors="#2B3440")


FRONTIER = [
    {"batch_size": 1, "variant": "Default CE", "max_context": 2048},
    {"batch_size": 1, "variant": "CCE", "max_context": 2048},
    {"batch_size": 4, "variant": "Default CE", "max_context": 512},
    {"batch_size": 4, "variant": "CCE", "max_context": 1024},
    {"batch_size": 16, "variant": "Default CE", "max_context": 256},
    {"batch_size": 16, "variant": "CCE", "max_context": 256},
    {"batch_size": 64, "variant": "Default CE", "max_context": 0},
    {"batch_size": 64, "variant": "CCE", "max_context": 0},
]

BOUNDARY = [
    {
        "suite": "b4_matched_boundary",
        "variant": "Default CE",
        "batch_size": 4,
        "max_length": 512,
        "lora_rank": 16,
        "status": "ok",
        "xla_gib_per_chip": 8.40,
        "step_time_sec": 0.0933439315,
    },
    {
        "suite": "b4_matched_boundary",
        "variant": "CCE",
        "batch_size": 4,
        "max_length": 512,
        "lora_rank": 16,
        "status": "ok",
        "xla_gib_per_chip": 7.08,
        "step_time_sec": 0.1339941570,
    },
    {
        "suite": "b4_matched_boundary",
        "variant": "Default CE",
        "batch_size": 4,
        "max_length": 1024,
        "lora_rank": 16,
        "status": "compile_oom",
        "xla_gib_per_chip": 16.15,
        "step_time_sec": "",
    },
    {
        "suite": "b4_matched_boundary",
        "variant": "CCE",
        "batch_size": 4,
        "max_length": 1024,
        "lora_rank": 16,
        "status": "ok",
        "xla_gib_per_chip": 14.13,
        "step_time_sec": 0.3213128950,
    },
]

RANK = [
    ("Default CE", 4, 512, "ok", 8.37, 0.1600661160),
    ("Default CE", 4, 1024, "ok", 14.18, 0.2795751470),
    ("CCE", 4, 512, "ok", 7.05, 0.1797523950),
    ("CCE", 4, 1024, "ok", 14.13, 0.3712091790),
    ("Default CE", 16, 512, "ok", 8.40, 0.1580799670),
    ("Default CE", 16, 1024, "compile_oom", 16.15, ""),
    ("CCE", 16, 512, "ok", 7.08, 0.1812821240),
    ("CCE", 16, 1024, "ok", 14.13, 0.3644494090),
    ("Default CE", 64, 512, "ok", 8.53, 0.1152801000),
    ("Default CE", 64, 1024, "compile_oom", 16.26, ""),
    ("CCE", 64, 512, "ok", 7.21, 0.1803141040),
    ("CCE", 64, 1024, "ok", 14.09, 0.3651279380),
]

CHUNK = [
    (32, 4096, 14.14, 0.5113331260),
    (32, 8192, 14.13, 0.4696555490),
    (32, 16384, 14.14, 0.4968258715),
    (32, 32768, 14.14, 0.4903565870),
    (64, 4096, 14.14, 0.3874859410),
    (64, 8192, 14.13, 0.3763901575),
    (64, 16384, 14.14, 0.3767528320),
    (64, 32768, 14.14, 0.3500358045),
    (128, 4096, 14.13, 0.3292957315),
]

AGGRESSIVE = [
    (16, 2048, 20.49),
    (16, 4096, 20.49),
    (16, 8192, 20.49),
    (16, 16384, 20.51),
    (16, 32768, 20.51),
]


def materialize_data() -> None:
  write_csv(DATA / "frontier_summary.csv", FRONTIER)
  write_csv(DATA / "matched_boundary.csv", BOUNDARY)
  write_csv(
      DATA / "rank_sensitivity.csv",
      [
          {
              "variant": variant,
              "lora_rank": rank,
              "batch_size": 4,
              "max_length": length,
              "status": status,
              "xla_gib_per_chip": xla,
              "step_time_sec": step,
          }
          for variant, rank, length, status, xla, step in RANK
      ],
  )
  write_csv(
      DATA / "chunk_tuning.csv",
      [
          {
              "variant": "CCE",
              "batch_size": 4,
              "max_length": 1024,
              "lora_rank": 16,
              "token_chunk": token,
              "vocab_chunk": vocab,
              "status": "ok",
              "xla_gib_per_chip": xla,
              "step_time_sec": step,
          }
          for token, vocab, xla, step in CHUNK
      ],
  )
  write_csv(
      DATA / "aggressive_l4096_negative.csv",
      [
          {
              "variant": "CCE",
              "batch_size": 1,
              "max_length": 4096,
              "lora_rank": 16,
              "token_chunk": token,
              "vocab_chunk": vocab,
              "status": "compile_oom",
              "xla_gib_per_chip": xla,
          }
          for token, vocab, xla in AGGRESSIVE
      ],
  )
  write_csv(
      DATA / "run_manifest.csv",
      [
          {
              "model": "Qwen3 0.6B",
              "model_id": "Qwen/Qwen3-0.6B",
              "tpu": "v5litepod-1",
              "chips": 1,
              "mesh": "fsdp=1,tp=1",
              "zone": "us-west4-a",
              "source": (
                  "transcribed from live runner log and CSV-tail checks; "
                  "TPU VM SSH became unstable before artifact tar recovery"
              ),
          }
      ],
  )


def plot_transfer() -> None:
  ASSETS.mkdir(parents=True, exist_ok=True)
  figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
  figure.patch.set_facecolor("white")

  # Frontier bars.
  axis = axes[0]
  batches = [1, 4, 16, 64]
  x = np.arange(len(batches))
  width = 0.36
  default_values = [
      next(r["max_context"] for r in FRONTIER if r["batch_size"] == b and r["variant"] == "Default CE")
      for b in batches
  ]
  cce_values = [
      next(r["max_context"] for r in FRONTIER if r["batch_size"] == b and r["variant"] == "CCE")
      for b in batches
  ]
  axis.bar(x - width / 2, default_values, width, color=BLUE, label="Default CE", zorder=3)
  axis.bar(x + width / 2, cce_values, width, color=ORANGE, label="CCE", zorder=3)
  for xpos, value in zip(x - width / 2, default_values):
    label = "OOM" if value == 0 else f"L{value}"
    axis.text(xpos, max(value, 70), label, ha="center", va="bottom", fontsize=9, color="#263238")
  for xpos, value in zip(x + width / 2, cce_values):
    label = "OOM" if value == 0 else f"L{value}"
    axis.text(xpos, max(value, 70), label, ha="center", va="bottom", fontsize=9, color="#263238")
  axis.set_title("Fit frontier on Qwen3 0.6B", loc="left", fontsize=13, weight="bold")
  axis.set_ylabel("Max context that compiled")
  axis.set_xticks(x, [f"b{b}" for b in batches])
  axis.set_ylim(0, 2400)
  axis.legend(frameon=False, loc="upper right")
  setup_axis(axis)

  # Boundary bars.
  axis = axes[1]
  groups = ["b4/L512", "b4/L1024"]
  x = np.arange(len(groups))
  default = [8.40, 16.15]
  cce = [7.08, 14.13]
  status_default = ["ok", "OOM"]
  status_cce = ["ok", "ok"]
  axis.bar(x - width / 2, default, width, color=BLUE, label="Default CE", zorder=3)
  axis.bar(x + width / 2, cce, width, color=ORANGE, label="CCE", zorder=3)
  for xpos, value, status in zip(x - width / 2, default, status_default):
    axis.text(xpos, value + 0.35, f"{value:.2f} GiB\n{status}", ha="center", va="bottom", fontsize=9)
  for xpos, value, status in zip(x + width / 2, cce, status_cce):
    axis.text(xpos, value + 0.35, f"{value:.2f} GiB\n{status}", ha="center", va="bottom", fontsize=9)
  axis.set_title("Same model, same TPU, CCE moves b4/L1024", loc="left", fontsize=13, weight="bold")
  axis.set_ylabel("XLA planned HBM per chip")
  axis.set_xticks(x, groups)
  axis.set_ylim(0, 18.5)
  axis.legend(frameon=False, loc="upper left")
  setup_axis(axis)

  figure.suptitle(
      "Qwen3 0.6B CCE transfer check on Cloud TPU v5e-1",
      x=0.01,
      ha="left",
      fontsize=16,
      weight="bold",
  )
  figure.savefig(ASSETS / "qwen3_0p6b_cce_transfer.png", dpi=180, bbox_inches="tight")
  plt.close(figure)


def plot_chunk() -> None:
  figure, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
  figure.patch.set_facecolor("white")
  labels = [f"{token}/{vocab//1024}k" for token, vocab, _, _ in CHUNK]
  times = [step for _, _, _, step in CHUNK]
  colors = [GREEN if time == min(times) else ORANGE for time in times]
  x = np.arange(len(labels))
  axis.bar(x, times, color=colors, zorder=3)
  for xpos, value in zip(x, times):
    axis.text(xpos, value + 0.012, f"{value:.2f}s", ha="center", va="bottom", fontsize=8)
  axis.set_title("Qwen3 0.6B CCE chunk tuning at b4/L1024", loc="left", fontsize=14, weight="bold")
  axis.set_ylabel("Mean step time, excluding first step")
  axis.set_xlabel("token_chunk / vocab_chunk")
  axis.set_xticks(x, labels, rotation=35, ha="right")
  axis.set_ylim(0, max(times) * 1.28)
  setup_axis(axis)
  figure.savefig(ASSETS / "qwen3_0p6b_cce_chunk_tuning.png", dpi=180, bbox_inches="tight")
  plt.close(figure)


def main() -> None:
  materialize_data()
  plot_transfer()
  plot_chunk()


if __name__ == "__main__":
  main()
