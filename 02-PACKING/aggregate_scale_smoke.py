#!/usr/bin/env python3
"""Aggregate short Gemma packing scale-smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "unpacked": "#4C78A8",
    "packed": "#F58518",
}


def read_json(path: Path) -> Any:
  return json.loads(path.read_text())


def read_history(path: Path) -> list[dict[str, str]]:
  with path.open() as f:
    return list(csv.DictReader(f))


def parse_run(raw: str) -> tuple[str, Path]:
  if "=" not in raw:
    path = Path(raw).expanduser().resolve()
    return path.name, path
  label, path = raw.split("=", 1)
  return label, Path(path).expanduser().resolve()


def parse_oom(raw: str) -> dict[str, str]:
  parts = raw.split("|")
  if len(parts) != 5:
    raise ValueError(
        "--oom-note must be model|condition|accelerator|result|detail"
    )
  return {
      "model": parts[0],
      "condition": parts[1],
      "accelerator": parts[2],
      "result": parts[3],
      "detail": parts[4],
  }


def as_float(value: Any, default: float = math.nan) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def as_int(value: Any, default: int = 0) -> int:
  try:
    return int(float(value))
  except (TypeError, ValueError):
    return default


def memory_gb(summary: dict[str, Any]) -> tuple[float, float]:
  memory = summary.get("memory_after_train", {})
  aggregate = memory.get("aggregate", {})
  aggregate_peak = as_float(aggregate.get("peak_bytes_in_use")) / 1e9
  devices = memory.get("devices", [])
  per_chip_peak = math.nan
  if devices:
    per_chip_peak = max(
        as_float(device.get("peak_bytes_in_use")) / 1e9 for device in devices
    )
  return aggregate_peak, per_chip_peak


def history_by_variant(history: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
  grouped: dict[str, list[dict[str, str]]] = {}
  for row in history:
    grouped.setdefault(row["variant"], []).append(row)
  return grouped


def infer_model_label(summary: dict[str, Any]) -> str:
  model_id = str(summary.get("model_id", ""))
  if "4b" in model_id:
    return "Gemma3 4B"
  if "1b" in model_id:
    return "Gemma3 1B"
  if "270m" in model_id or "270M" in model_id:
    return "Gemma3 270M"
  return model_id or "unknown model"


def build_rows(
    runs: list[tuple[str, Path]],
    *,
    accelerator_type: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, str]]]]:
  rows: list[dict[str, Any]] = []
  histories: dict[tuple[str, str], list[dict[str, str]]] = {}
  for run_label, run_dir in runs:
    summaries = read_json(run_dir / "summary.json")
    if not isinstance(summaries, list):
      summaries = [summaries]
    grouped_history = history_by_variant(read_history(run_dir / "history.csv"))
    for summary in summaries:
      variant = str(summary["variant"])
      model = infer_model_label(summary)
      history = grouped_history.get(variant, [])
      histories[(model, variant)] = history
      aggregate_peak_gb, per_chip_peak_gb = memory_gb(summary)
      final = history[-1] if history else {}
      packing = summary.get("packing", {})
      rows.append({
          "model": model,
          "run_label": run_label,
          "variant": variant,
          "model_id": summary.get("model_id", ""),
          "accelerator_type": accelerator_type,
          "tpu_name": summary.get("runtime", {}).get("tpu_name", ""),
          "tpu_zone": summary.get("runtime", {}).get("tpu_zone", ""),
          "jax_devices": len(summary.get("jax_devices", [])),
          "batch_size": summary.get("batch_size", 0),
          "max_length": summary.get("max_length", 0),
          "steps": summary.get("steps_recorded", 0),
          "final_loss": summary.get("final_loss", math.nan),
          "mean_step_time_sec_excl_first": summary.get(
              "mean_step_time_sec_excl_first",
              math.nan,
          ),
          "valid_tokens_per_sec_excl_first": summary.get(
              "valid_tokens_per_sec_excl_first",
              math.nan,
          ),
          "loss_tokens_per_sec_excl_first": summary.get(
              "loss_tokens_per_sec_excl_first",
              math.nan,
          ),
          "packing_efficiency": packing.get("packed_efficiency", math.nan),
          "row_reduction_x": packing.get("row_reduction_x", math.nan),
          "cumulative_valid_tokens": as_int(
              final.get("cumulative_valid_tokens", 0)
          ),
          "cumulative_loss_tokens": as_int(
              final.get("cumulative_loss_tokens", 0)
          ),
          "peak_memory_aggregate_gb_after_train": aggregate_peak_gb,
          "peak_memory_per_chip_gb_after_train": per_chip_peak_gb,
          "wall_time_sec": summary.get("wall_time_sec", math.nan),
      })
  return rows, histories


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys())
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def model_order(rows: list[dict[str, Any]]) -> list[str]:
  order = []
  for row in rows:
    model = str(row["model"])
    if model not in order:
      order.append(model)
  return order


def row_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
  return {(str(row["model"]), str(row["variant"])): row for row in rows}


def plot_loss_vs_tokens(
    rows: list[dict[str, Any]],
    histories: dict[tuple[str, str], list[dict[str, str]]],
    outdir: Path,
) -> Path:
  models = model_order(rows)
  fig, axes = plt.subplots(1, len(models), figsize=(6.6 * len(models), 4.8))
  if len(models) == 1:
    axes = [axes]

  for axis, model in zip(axes, models):
    model_rows = [row for row in rows if row["model"] == model]
    subtitle = ""
    if model_rows:
      first = model_rows[0]
      subtitle = (
          f"b{first['batch_size']}, L{first['max_length']}, "
          f"{first['steps']} steps"
      )
    for variant in ("unpacked", "packed"):
      history = histories.get((model, variant), [])
      if not history:
        continue
      tokens = [as_int(row["cumulative_loss_tokens"]) for row in history]
      losses = [as_float(row["loss"]) for row in history]
      color = COLORS[variant]
      axis.plot(
          tokens,
          losses,
          color=color,
          linewidth=2.1,
          marker="o",
          markersize=3.8,
          label=variant,
      )
      axis.scatter(tokens[-1], losses[-1], s=64, color=color, zorder=3)
      axis.annotate(
          f"{tokens[-1] / 1000:.1f}k",
          xy=(tokens[-1], losses[-1]),
          xytext=(8, 0),
          textcoords="offset points",
          fontsize=9,
          va="center",
          color=color,
      )
    axis.set_title(f"{model}\n{subtitle}")
    axis.set_xlabel("Cumulative target tokens used for loss")
    axis.set_ylabel("Training loss")
    axis.grid(True, color="#E6E8EB", linewidth=0.8)
    axis.legend(frameon=False)

  fig.suptitle(
      "Loss vs Useful Training Tokens",
      fontsize=15,
      fontweight="bold",
      y=1.03,
  )
  fig.tight_layout()
  path = outdir / "loss_vs_useful_tokens.png"
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return path


def plot_throughput_and_density(rows: list[dict[str, Any]], outdir: Path) -> Path:
  models = model_order(rows)
  lookup = row_lookup(rows)
  x = np.arange(len(models))
  width = 0.36
  variants = ("unpacked", "packed")
  offsets = {"unpacked": -width / 2, "packed": width / 2}

  fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
  specs = [
      (
          "loss_tokens_per_sec_excl_first",
          "Target-token throughput",
          "Target tokens/sec",
          "{:.0f}",
      ),
      (
          "packing_efficiency",
          "Batch density",
          "Non-pad token density",
          "{:.1%}",
      ),
      (
          "mean_step_time_sec_excl_first",
          "Step time",
          "Seconds/step",
          "{:.3f}",
      ),
  ]
  for axis, (key, title, ylabel, fmt) in zip(axes, specs):
    max_value = 0.0
    for variant in variants:
      values = [
          as_float(lookup.get((model, variant), {}).get(key))
          for model in models
      ]
      max_value = max(max_value, *(value for value in values if not math.isnan(value)))
      bars = axis.bar(
          x + offsets[variant],
          values,
          width,
          color=COLORS[variant],
          label=variant,
      )
      for bar, value in zip(bars, values):
        if math.isnan(value):
          continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8.5,
            rotation=0,
        )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x)
    axis.set_xticklabels(models)
    axis.grid(True, axis="y", color="#E6E8EB", linewidth=0.8)
    axis.set_axisbelow(True)
    if max_value > 0:
      axis.set_ylim(0, max_value * 1.18)
    if key in {"loss_tokens_per_sec_excl_first", "packing_efficiency"}:
      for idx, model in enumerate(models):
        unpacked = as_float(lookup.get((model, "unpacked"), {}).get(key))
        packed = as_float(lookup.get((model, "packed"), {}).get(key))
        if unpacked > 0 and packed > 0:
          axis.text(
              idx,
              max(unpacked, packed) * 1.09,
              f"{packed / unpacked:.1f}x",
              ha="center",
              va="bottom",
              fontsize=10,
              fontweight="bold",
              color="#2E2E2E",
          )
  axes[0].legend(frameon=False)
  fig.suptitle(
      "Train Throughput Comes From Denser Steps",
      fontsize=15,
      fontweight="bold",
      y=1.03,
  )
  fig.tight_layout()
  path = outdir / "throughput_and_density.png"
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return path


def format_float(value: Any, digits: int = 2) -> str:
  number = as_float(value)
  if math.isnan(number):
    return "n/a"
  return f"{number:.{digits}f}"


def write_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    plots: list[Path],
    oom_notes: list[dict[str, str]],
) -> None:
  lookup = row_lookup(rows)
  lines = [
      "# Gemma3 1B/4B Sequence Packing Scale Smoke",
      "",
      "This is a short-run check for the same sequence-packing effect seen on "
      "Gemma3 270M. It deliberately stops at 50 optimizer steps: the goal is "
      "to see whether packed batches accumulate much more useful training "
      "signal before running a long quality experiment.",
      "",
      "Within each model, packed and unpacked use the same model, TPU, batch "
      "size, max length, LoRA rank, dataset, and optimizer settings. The only "
      "thing changed is whether examples are packed together before they are "
      "passed into the normal Tunix SFT path.",
      "",
  ]
  for plot in plots:
    lines.extend([f"![{plot.stem}]({plot.name})", ""])

  lines.extend([
      "## Run Matrix",
      "",
      "| Model | TPU | Chips | Batch | Max length | Steps | Variant | Density | Target tok/s | Final target tokens | Final loss | JAX peak aggregate |",
      "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
  ])
  for row in rows:
    lines.append(
        "| "
        f"{row['model']} | "
        f"{row['accelerator_type']} / {row['tpu_name']} | "
        f"{row['jax_devices']} | "
        f"{row['batch_size']} | "
        f"{row['max_length']} | "
        f"{row['steps']} | "
        f"{row['variant']} | "
        f"{as_float(row['packing_efficiency']) * 100:.1f}% | "
        f"{as_float(row['loss_tokens_per_sec_excl_first']):.0f} | "
        f"{row['cumulative_loss_tokens']} | "
        f"{as_float(row['final_loss']):.4f} | "
        f"{as_float(row['peak_memory_aggregate_gb_after_train']):.2f} GB |"
    )

  lines.extend([
      "",
      "## Ratios",
      "",
      "| Model | Target-token throughput | Final target tokens in 50 steps | Density change | Step-time change |",
      "| --- | ---: | ---: | ---: | ---: |",
  ])
  for model in model_order(rows):
    unpacked = lookup[(model, "unpacked")]
    packed = lookup[(model, "packed")]
    tps_ratio = (
        as_float(packed["loss_tokens_per_sec_excl_first"])
        / as_float(unpacked["loss_tokens_per_sec_excl_first"])
    )
    token_ratio = (
        as_float(packed["cumulative_loss_tokens"])
        / as_float(unpacked["cumulative_loss_tokens"])
    )
    density_ratio = (
        as_float(packed["packing_efficiency"])
        / as_float(unpacked["packing_efficiency"])
    )
    step_ratio = (
        as_float(packed["mean_step_time_sec_excl_first"])
        / as_float(unpacked["mean_step_time_sec_excl_first"])
    )
    lines.append(
        "| "
        f"{model} | "
        f"{tps_ratio:.1f}x | "
        f"{token_ratio:.1f}x | "
        f"{density_ratio:.1f}x | "
        f"{step_ratio:.3f}x |"
    )

  if oom_notes:
    lines.extend([
        "",
        "## Batch Sizing Notes",
        "",
        "These OOMs were batch-search observations on the same v5litepod-4 "
        "hardware, not evidence that packing itself reduces model memory. "
        "Sequence packing keeps the tensor shape fixed; its win here is that "
        "the fixed shape contains far fewer padding tokens.",
        "",
        "| Model | Tried condition | TPU | Result | Compiler detail |",
        "| --- | --- | --- | --- | --- |",
    ])
    for note in oom_notes:
      lines.append(
          "| "
          f"{note['model']} | "
          f"{note['condition']} | "
          f"{note['accelerator']} | "
          f"{note['result']} | "
          f"{note['detail']} |"
      )

  lines.extend([
      "",
      "## Interpretation",
      "",
      "The key signal reproduced on both larger models is not lower step time. "
      "The step time is nearly unchanged because the model still sees the same "
      "static batch shape. The difference is that the packed batch is almost "
      "full: about 99.3% non-padding density versus about 10.5% for ordinary "
      "fixed-length batches on this OPUS100 EN-FR prompt format.",
      "",
      "That changes the unit economics of training. In 50 steps, Gemma3 1B "
      "processed about "
      f"{lookup[('Gemma3 1B', 'packed')]['cumulative_loss_tokens'] / 1000:.1f}k "
      "target tokens with packing versus "
      f"{lookup[('Gemma3 1B', 'unpacked')]['cumulative_loss_tokens'] / 1000:.1f}k "
      "without it. Gemma3 4B shows the same pattern: "
      f"{lookup[('Gemma3 4B', 'packed')]['cumulative_loss_tokens'] / 1000:.1f}k "
      "versus "
      f"{lookup[('Gemma3 4B', 'unpacked')]['cumulative_loss_tokens'] / 1000:.1f}k.",
      "",
      "This is still a smoke test, not a final quality claim. It is enough to "
      "justify the next long run only if we care about final output quality "
      "for 1B or 4B; for throughput behavior, the effect is already clear.",
      "",
      "The memory column above is the aggregate JAX device-memory snapshot "
      "recorded after training. It is useful as a run sanity check, but it is "
      "not the same as the XLA buffer-assignment peak used in the CCE memory "
      "reports.",
  ])
  path.write_text("\n".join(lines) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", action="append", required=True)
  parser.add_argument("--outdir", required=True)
  parser.add_argument("--accelerator-type", default="v5litepod-4")
  parser.add_argument("--oom-note", action="append", default=[])
  args = parser.parse_args()

  outdir = Path(args.outdir).expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  runs = [parse_run(raw) for raw in args.run]
  rows, histories = build_rows(runs, accelerator_type=args.accelerator_type)
  write_csv(outdir / "summary.csv", rows)
  plots = [
      plot_loss_vs_tokens(rows, histories, outdir),
      plot_throughput_and_density(rows, outdir),
  ]
  oom_notes = [parse_oom(note) for note in args.oom_note]
  write_report(outdir / "README.md", rows=rows, plots=plots, oom_notes=oom_notes)
  print(f"outdir={outdir}")
  for plot in plots:
    print(f"plot={plot}")
  print(f"summary={outdir / 'summary.csv'}")
  print(f"report={outdir / 'README.md'}")


if __name__ == "__main__":
  main()
