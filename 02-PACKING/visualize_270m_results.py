#!/usr/bin/env python3
"""Build Gemma3 270M packing report tables and figures."""

from __future__ import annotations

import csv
import json
import math
import tarfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
PROCESSED = DATA / "processed"

SHORT_DIR = (
    DATA
    / "raw_artifacts"
    / "gemma3_270m_short_throughput_v5litepod1"
    / "short-throughput"
)
QUALITY_UNPACKED_DIR = (
    DATA
    / "raw_artifacts"
    / "gemma3_270m_quality_unpacked_v5litepod1"
    / "quality-unpacked"
    / "quality-unpacked_unpacked_b16_l512_s5000"
)
QUALITY_PACKED_DIR = (
    DATA
    / "raw_artifacts"
    / "gemma3_270m_quality_packed_v5litepod1"
    / "quality-packed"
    / "quality-packed_packed_b16_l512_s1000"
)


BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#2E7D32"
RED = "#C44E52"
GRAY = "#5F6B7A"
LIGHT_GRID = "#E7E9ED"


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open() as f:
    return list(csv.DictReader(f))


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


def ensure_raw_artifacts() -> None:
  ensure_extracted(
      DATA / "raw_artifacts" / "gemma3_270m_short_throughput_v5litepod1",
      "short-throughput",
  )
  ensure_extracted(
      DATA / "raw_artifacts" / "gemma3_270m_quality_unpacked_v5litepod1",
      "quality-unpacked",
  )
  ensure_extracted(
      DATA / "raw_artifacts" / "gemma3_270m_quality_packed_v5litepod1",
      "quality-packed",
  )


def read_json(path: Path) -> Any:
  return json.loads(path.read_text())


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


def gb(value: float) -> str:
  return f"{value:.1f}G"


def setup_axis(axis: plt.Axes) -> None:
  axis.grid(True, axis="y", color=LIGHT_GRID, linewidth=0.8, zorder=0)
  axis.spines["top"].set_visible(False)
  axis.spines["right"].set_visible(False)
  axis.spines["left"].set_color("#D5DAE1")
  axis.spines["bottom"].set_color("#D5DAE1")
  axis.tick_params(colors="#2B3440")


def load_short_rows() -> list[dict[str, Any]]:
  rows = []
  for row in read_csv(SHORT_DIR / "short-throughput_results.csv"):
    rows.append({
        **row,
        "batch_size": as_int(row.get("batch_size")),
        "max_length": as_int(row.get("max_length")),
        "xla_train_step_gib_per_chip": as_float(
            row.get("xla_train_step_gib_per_chip")
        ),
        "mean_step_time_sec_excl_first": as_float(
            row.get("mean_step_time_sec_excl_first")
        ),
        "loss_tokens_per_sec_excl_first": as_float(
            row.get("loss_tokens_per_sec_excl_first")
        ),
        "valid_tokens_per_sec_excl_first": as_float(
            row.get("valid_tokens_per_sec_excl_first")
        ),
        "packed_efficiency": as_float(row.get("packed_efficiency")),
        "row_reduction_x": as_float(row.get("row_reduction_x")),
        "final_cumulative_loss_tokens": as_int(
            row.get("final_cumulative_loss_tokens")
        ),
        "final_cumulative_valid_tokens": as_int(
            row.get("final_cumulative_valid_tokens")
        ),
        "oom_used_gib": as_float(row.get("oom_used_gib")),
        "oom_limit_gib": as_float(row.get("oom_limit_gib")),
    })
  return rows


def load_quality_summary(run_dir: Path) -> dict[str, Any]:
  payload = read_json(run_dir / "summary.json")
  if isinstance(payload, list):
    return payload[0]
  return payload


def load_quality_rows() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
  summaries = []
  histories: dict[str, list[dict[str, Any]]] = {}
  for variant, run_dir in (
      ("unpacked", QUALITY_UNPACKED_DIR),
      ("packed", QUALITY_PACKED_DIR),
  ):
    summary = load_quality_summary(run_dir)
    history = read_csv(run_dir / "history.csv")
    histories[variant] = [
        {
            **row,
            "step": as_int(row["step"]),
            "loss": as_float(row["loss"]),
            "cumulative_loss_tokens": as_int(row["cumulative_loss_tokens"]),
            "cumulative_valid_tokens": as_int(row["cumulative_valid_tokens"]),
            "cumulative_step_time_sec": as_float(
                row["cumulative_step_time_sec"]
            ),
        }
        for row in history
    ]
    quality = summary.get("quality", {})
    summaries.append({
        "variant": variant,
        "steps": summary["steps_recorded"],
        "batch_size": summary["batch_size"],
        "max_length": summary["max_length"],
        "wall_time_sec": summary["wall_time_sec"],
        "mean_step_time_sec_excl_first": summary[
            "mean_step_time_sec_excl_first"
        ],
        "loss_tokens_per_sec_excl_first": summary[
            "loss_tokens_per_sec_excl_first"
        ],
        "valid_tokens_per_sec_excl_first": summary[
            "valid_tokens_per_sec_excl_first"
        ],
        "runtime_peak_hbm_gb": (
            summary["memory_after_train"]["aggregate"]["peak_bytes_in_use"]
            / 1e9
        ),
        "xla_train_step_gib_per_chip": 12.57,
        "packing_efficiency": summary["packing"]["packed_efficiency"],
        "row_reduction_x": summary["packing"]["row_reduction_x"],
        "final_loss": summary["final_loss"],
        "eval_loss": quality.get("eval_loss", math.nan),
        "final_cumulative_loss_tokens": histories[variant][-1][
            "cumulative_loss_tokens"
        ],
        "final_cumulative_valid_tokens": histories[variant][-1][
            "cumulative_valid_tokens"
        ],
        "final_cumulative_step_time_sec": histories[variant][-1][
            "cumulative_step_time_sec"
        ],
    })
  return summaries, histories


def smooth(values: list[float], window: int) -> np.ndarray:
  if not values:
    return np.asarray([])
  window = max(1, min(window, len(values)))
  kernel = np.ones(window) / window
  return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def interpolate_time_at_tokens(
    history: list[dict[str, Any]],
    target_tokens: int,
) -> tuple[int, float, float]:
  for row in history:
    if row["cumulative_loss_tokens"] >= target_tokens:
      return row["step"], row["cumulative_step_time_sec"], row["loss"]
  row = history[-1]
  return row["step"], row["cumulative_step_time_sec"], row["loss"]


def plot_density() -> Path:
  rows = [
      row
      for row in read_csv(DATA / "local_density" / "gemma_tokenizer_20k_packing.csv")
      if as_int(row["batch_size"]) == 16
  ]
  rows.sort(key=lambda row: as_int(row["max_length"]))
  x = [as_int(row["max_length"]) for row in rows]
  fixed = [as_float(row["unpacked_fixed_valid_ratio"]) for row in rows]
  dynamic = [as_float(row["unpacked_dynamic_valid_ratio"]) for row in rows]
  packed = [as_float(row["packed_valid_ratio"]) for row in rows]
  row_reduction = [as_float(row["row_reduction_x"]) for row in rows]
  dynamic_gain = [as_float(row["packed_vs_dynamic_density_gain_x"]) for row in rows]

  fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
  axis = axes[0]
  setup_axis(axis)
  axis.plot(x, fixed, marker="o", linewidth=2.3, color=BLUE, label="Fixed unpacked")
  axis.plot(
      x,
      dynamic,
      marker="o",
      linewidth=2.3,
      color=GRAY,
      label="Dynamic-batch estimate",
  )
  axis.plot(x, packed, marker="o", linewidth=2.6, color=ORANGE, label="Packed")
  axis.set_xscale("log", base=2)
  axis.set_xticks(x)
  axis.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
  axis.set_ylim(0, 1.08)
  axis.set_title("Density Opportunity Before Training", fontweight="bold")
  axis.set_xlabel("Max sequence length")
  axis.set_ylabel("Non-pad token density")
  axis.legend(frameon=False, loc="lower right")
  for xi, yi in zip(x, packed):
    axis.text(xi, yi + 0.035, f"{yi:.1%}", ha="center", fontsize=9, color=ORANGE)

  axis = axes[1]
  setup_axis(axis)
  axis.plot(
      x,
      row_reduction,
      marker="o",
      linewidth=2.6,
      color=ORANGE,
      label="Rows after packing",
  )
  axis.plot(
      x,
      dynamic_gain,
      marker="o",
      linewidth=2.3,
      color=GREEN,
      label="Density gain vs dynamic estimate",
  )
  axis.set_xscale("log", base=2)
  axis.set_yscale("log", base=2)
  axis.set_xticks(x)
  axis.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
  axis.set_yticks([2, 4, 8, 16, 32])
  axis.get_yaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
  axis.set_title("The Gain Comes From Removing Padding", fontweight="bold")
  axis.set_xlabel("Max sequence length")
  axis.set_ylabel("Multiplier")
  axis.legend(frameon=False, loc="upper left")
  for xi, yi in zip(x, row_reduction):
    axis.text(xi, yi * 1.08, f"{yi:.1f}x", ha="center", fontsize=9, color=ORANGE)

  fig.suptitle(
      "Gemma Tokenizer, OPUS100 EN-FR, 20k examples, batch 16",
      y=1.02,
      fontsize=11,
      color=GRAY,
  )
  fig.tight_layout()
  path = ASSETS / "gemma3_270m_density_opportunity.png"
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return path


def shape_label(row: dict[str, Any]) -> str:
  return f"b{row['batch_size']}\nL{row['max_length']}"


def plot_fit_frontier(short_rows: list[dict[str, Any]]) -> Path:
  shapes = []
  for row in short_rows:
    shape = (row["batch_size"], row["max_length"])
    if shape not in shapes:
      shapes.append(shape)
  shapes.sort()
  lookup = {
      (row["batch_size"], row["max_length"], row["variant"]): row
      for row in short_rows
  }
  x = np.arange(len(shapes))
  width = 0.36
  fig, axis = plt.subplots(figsize=(12.2, 5.1))
  setup_axis(axis)
  limit = 15.75
  for variant, offset, color in (
      ("unpacked", -width / 2, BLUE),
      ("packed", width / 2, ORANGE),
  ):
    values = [
        lookup[(batch, length, variant)]["xla_train_step_gib_per_chip"]
        for batch, length in shapes
    ]
    statuses = [
        lookup[(batch, length, variant)]["status"]
        for batch, length in shapes
    ]
    bars = axis.bar(
        x + offset,
        values,
        width,
        color=color,
        edgecolor="#20242A",
        linewidth=0.7,
        label=variant,
        zorder=3,
    )
    for bar, value, status in zip(bars, values, statuses):
      if status != "ok":
        bar.set_hatch("////")
        bar.set_alpha(0.45)
      axis.text(
          bar.get_x() + bar.get_width() / 2,
          value + 0.8,
          f"{value:.1f}",
          ha="center",
          va="bottom",
          fontsize=8.5,
          color="#20242A",
      )
      if status != "ok":
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value, axis.get_ylim()[1]) * 0.18,
            "OOM",
            ha="center",
            va="center",
            rotation=90,
            fontsize=8,
            color="#20242A",
        )
  axis.axhline(limit, color=RED, linewidth=1.8, linestyle="--", zorder=2)
  axis.text(
      len(shapes) - 0.2,
      limit + 0.7,
      "v5litepod-1 HBM limit: 15.75 GiB/chip",
      color=RED,
      fontsize=9,
      ha="right",
  )
  axis.set_xticks(x)
  axis.set_xticklabels([f"b{b}\nL{l}" for b, l in shapes])
  axis.set_ylim(0, 50)
  axis.set_ylabel("XLA train-step planned HBM, GiB/chip")
  axis.set_title("Packing Does Not Move the Fixed-Shape Fit Frontier", fontweight="bold")
  axis.legend(
      handles=[
          Patch(facecolor=BLUE, label="unpacked"),
          Patch(facecolor=ORANGE, label="packed"),
          Patch(
              facecolor="#BBBBBB",
              edgecolor="#20242A",
              hatch="////",
              alpha=0.45,
              label="compile OOM",
          ),
      ],
      frameon=False,
      ncols=3,
      loc="upper left",
  )
  fig.tight_layout()
  path = ASSETS / "gemma3_270m_tpu_fit_frontier.png"
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return path


def plot_throughput(short_rows: list[dict[str, Any]]) -> Path:
  ok_rows = [row for row in short_rows if row["status"] == "ok"]
  shapes = []
  for row in ok_rows:
    shape = (row["batch_size"], row["max_length"])
    if shape not in shapes:
      shapes.append(shape)
  shapes.sort()
  lookup = {
      (row["batch_size"], row["max_length"], row["variant"]): row
      for row in ok_rows
  }
  paired_shapes = [
      shape
      for shape in shapes
      if (shape[0], shape[1], "unpacked") in lookup
      and (shape[0], shape[1], "packed") in lookup
  ]
  labels = [f"b{batch}, L{length}" for batch, length in paired_shapes]
  throughput_speedups = []
  step_ratios = []
  for batch, length in paired_shapes:
    unpacked = lookup[(batch, length, "unpacked")]
    packed = lookup[(batch, length, "packed")]
    throughput_speedups.append(
        packed["loss_tokens_per_sec_excl_first"]
        / unpacked["loss_tokens_per_sec_excl_first"]
    )
    step_ratios.append(
        packed["mean_step_time_sec_excl_first"]
        / unpacked["mean_step_time_sec_excl_first"]
    )

  fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
  axis = axes[0]
  setup_axis(axis)
  bars = axis.bar(labels, throughput_speedups, color=ORANGE, zorder=3)
  axis.axhline(1.0, color=GRAY, linewidth=1.0)
  axis.set_ylabel("Packed / unpacked target-token throughput")
  axis.set_title("Useful Token Throughput", fontweight="bold")
  for bar, value in zip(bars, throughput_speedups):
    axis.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.8,
        f"{value:.1f}x",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

  axis = axes[1]
  setup_axis(axis)
  bars = axis.bar(labels, step_ratios, color=BLUE, zorder=3)
  axis.axhline(1.0, color=GRAY, linewidth=1.4, linestyle="--")
  axis.set_ylim(0.94, 1.08)
  axis.set_ylabel("Packed / unpacked seconds per step")
  axis.set_title("Same-Shape Step Cost", fontweight="bold")
  for bar, value in zip(bars, step_ratios):
    axis.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.006,
        f"{value:.3f}x",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

  fig.suptitle(
      "Gemma3 270M LoRA SFT, v5litepod-1, 1 chip, 50-step short runs",
      y=1.03,
      fontsize=11,
      color=GRAY,
  )
  fig.tight_layout()
  path = ASSETS / "gemma3_270m_tpu_throughput_ratios.png"
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return path


def plot_quality(
    quality_rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
) -> Path:
  unpacked_budget = histories["unpacked"][-1]["cumulative_loss_tokens"]
  packed_step, packed_time, packed_loss = interpolate_time_at_tokens(
      histories["packed"],
      unpacked_budget,
  )
  unpacked_time = histories["unpacked"][-1]["cumulative_step_time_sec"]

  fig, axes = plt.subplots(2, 1, figsize=(10.4, 8.0), height_ratios=[1.35, 1.0])

  axis = axes[0]
  setup_axis(axis)
  for variant, color in (("unpacked", BLUE), ("packed", ORANGE)):
    rows = histories[variant]
    tokens = [row["cumulative_loss_tokens"] for row in rows]
    losses = [row["loss"] for row in rows]
    window = 180 if variant == "unpacked" else 45
    axis.plot(tokens, losses, color=color, alpha=0.12, linewidth=0.7)
    axis.plot(
        tokens,
        smooth(losses, window),
        color=color,
        linewidth=2.3,
        label=variant,
    )
  axis.axvline(unpacked_budget, color=GRAY, linestyle="--", linewidth=1.2)
  axis.text(
      unpacked_budget * 1.02,
      axis.get_ylim()[1] * 0.76,
      f"unpacked final budget\n{unpacked_budget / 1e6:.2f}M target tokens",
      fontsize=9,
      color=GRAY,
      ha="left",
  )
  axis.set_title("Learning View: Compare by Useful Target Tokens", fontweight="bold")
  axis.set_xlabel("Cumulative target tokens used for loss")
  axis.set_ylabel("Training loss")
  axis.legend(frameon=False, loc="upper right")

  axis = axes[1]
  setup_axis(axis)
  labels = [
      "Time to\n1.75M target tokens",
      "Run-end\ntarget tokens",
      "Run-end\neval loss",
      "Train-step\nplanned HBM",
  ]
  unpacked_summary = next(row for row in quality_rows if row["variant"] == "unpacked")
  packed_summary = next(row for row in quality_rows if row["variant"] == "packed")
  raw_values = {
      "unpacked": [
          unpacked_time,
          unpacked_summary["final_cumulative_loss_tokens"] / 1e6,
          unpacked_summary["eval_loss"],
          unpacked_summary["xla_train_step_gib_per_chip"],
      ],
      "packed": [
          packed_time,
          packed_summary["final_cumulative_loss_tokens"] / 1e6,
          packed_summary["eval_loss"],
          packed_summary["xla_train_step_gib_per_chip"],
      ],
  }
  normalizers = raw_values["unpacked"]
  normalized = {
      variant: [
          value / normalizer
          if normalizer else math.nan
          for value, normalizer in zip(series, normalizers)
      ]
      for variant, series in raw_values.items()
  }
  x = np.arange(len(labels))
  width = 0.34
  for variant, offset, color in (
      ("unpacked", -width / 2, BLUE),
      ("packed", width / 2, ORANGE),
  ):
    bars = axis.bar(
        x + offset,
        normalized[variant],
        width,
        color=color,
        label=variant,
        zorder=3,
    )
    for idx, (bar, raw) in enumerate(zip(bars, raw_values[variant])):
      if idx == 0:
        text = f"{raw:.0f}s"
      elif idx == 1:
        text = f"{raw:.2f}M"
      elif idx == 2:
        text = f"{raw:.2f}"
      else:
        text = f"{raw:.2f}GiB"
      axis.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + 0.04,
          text,
          ha="center",
          va="bottom",
          fontsize=9,
      )
  axis.set_xticks(x)
  axis.set_xticklabels(labels)
  axis.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.0)
  axis.set_ylim(0, 2.25)
  axis.set_ylabel("Relative to unpacked baseline")
  axis.set_title("Budget and Sanity Metrics", fontweight="bold")
  axis.legend(frameon=False, loc="upper left", ncols=2)
  axis.text(
      0.98,
      0.93,
      f"Packed reaches the unpacked target-token budget at step {packed_step}.",
      transform=axis.transAxes,
      ha="right",
      va="top",
      fontsize=9,
      color=GRAY,
  )

  fig.suptitle(
      "Gemma3 270M Useful-Token Budget Parity: b16/L512 on v5litepod-1",
      y=1.01,
      fontsize=12,
      color=GRAY,
  )
  fig.tight_layout()
  path = ASSETS / "gemma3_270m_quality_token_budget.png"
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return path


def write_processed(
    short_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
) -> None:
  write_csv(PROCESSED / "gemma3_270m_short_throughput_v5litepod1.csv", short_rows)
  write_csv(PROCESSED / "gemma3_270m_quality_v5litepod1.csv", quality_rows)
  history_rows: list[dict[str, Any]] = []
  for variant, rows in histories.items():
    history_rows.extend(rows)
  write_csv(PROCESSED / "gemma3_270m_quality_history_v5litepod1.csv", history_rows)


def write_summary_json(
    short_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
    figures: list[Path],
) -> None:
  ok_rows = [row for row in short_rows if row["status"] == "ok"]
  failed_rows = [row for row in short_rows if row["status"] != "ok"]
  pairs: list[dict[str, Any]] = []
  for batch in sorted({row["batch_size"] for row in ok_rows}):
    for length in sorted({row["max_length"] for row in ok_rows}):
      unpacked = next(
          (
              row
              for row in ok_rows
              if row["batch_size"] == batch
              and row["max_length"] == length
              and row["variant"] == "unpacked"
          ),
          None,
      )
      packed = next(
          (
              row
              for row in ok_rows
              if row["batch_size"] == batch
              and row["max_length"] == length
              and row["variant"] == "packed"
          ),
          None,
      )
      if unpacked and packed:
        pairs.append({
            "batch_size": batch,
            "max_length": length,
            "target_token_throughput_speedup_x": (
                packed["loss_tokens_per_sec_excl_first"]
                / unpacked["loss_tokens_per_sec_excl_first"]
            ),
            "step_time_ratio": (
                packed["mean_step_time_sec_excl_first"]
                / unpacked["mean_step_time_sec_excl_first"]
            ),
            "xla_train_step_gib_per_chip_unpacked": unpacked[
                "xla_train_step_gib_per_chip"
            ],
            "xla_train_step_gib_per_chip_packed": packed[
                "xla_train_step_gib_per_chip"
            ],
        })
  unpacked_budget = histories["unpacked"][-1]["cumulative_loss_tokens"]
  packed_step, packed_time, _ = interpolate_time_at_tokens(
      histories["packed"],
      unpacked_budget,
  )
  unpacked_time = histories["unpacked"][-1]["cumulative_step_time_sec"]
  summary = {
      "experiment": "Gemma3 270M sequence packing rerun",
      "accelerator": "Cloud TPU v5litepod-1, 1 chip",
      "mesh": {"fsdp": 1, "tp": 1},
      "model": "google/gemma-3-270m-it",
      "training": "LoRA SFT, OPUS100 EN-FR, target-only loss",
      "successful_short_cases": len(ok_rows),
      "failed_short_cases": len(failed_rows),
      "short_pairs": pairs,
      "oom_cases": failed_rows,
      "quality_rows": quality_rows,
      "packed_reaches_unpacked_target_budget": {
          "target_loss_tokens": unpacked_budget,
          "packed_step": packed_step,
          "packed_cumulative_step_time_sec": packed_time,
          "unpacked_cumulative_step_time_sec": unpacked_time,
          "time_speedup_x": unpacked_time / packed_time,
      },
      "figures": [str(path.relative_to(ROOT)) for path in figures],
  }
  (PROCESSED / "gemma3_270m_summary.json").write_text(
      json.dumps(sanitize_json(summary), indent=2, sort_keys=True) + "\n"
  )


def sanitize_json(value: Any) -> Any:
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, dict):
    return {key: sanitize_json(item) for key, item in value.items()}
  if isinstance(value, list):
    return [sanitize_json(item) for item in value]
  return value


def main() -> None:
  ASSETS.mkdir(parents=True, exist_ok=True)
  PROCESSED.mkdir(parents=True, exist_ok=True)
  ensure_raw_artifacts()
  short_rows = load_short_rows()
  quality_rows, histories = load_quality_rows()
  write_processed(short_rows, quality_rows, histories)
  figures = [
      plot_density(),
      plot_fit_frontier(short_rows),
      plot_throughput(short_rows),
      plot_quality(quality_rows, histories),
  ]
  write_summary_json(short_rows, quality_rows, histories, figures)
  for figure in figures:
    print(figure)
  print(PROCESSED / "gemma3_270m_summary.json")


if __name__ == "__main__":
  main()
