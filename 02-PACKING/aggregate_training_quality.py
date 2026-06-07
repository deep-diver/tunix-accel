#!/usr/bin/env python3
"""Aggregate packed vs unpacked Gemma SFT training runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import NullLocator
import numpy as np


def read_json(path: Path) -> Any:
  return json.loads(path.read_text())


def read_summary(run_dir: Path) -> dict[str, Any]:
  payload = read_json(run_dir / "summary.json")
  if isinstance(payload, list):
    if len(payload) != 1:
      raise ValueError(f"{run_dir}/summary.json has {len(payload)} entries.")
    return payload[0]
  return payload


def read_history(run_dir: Path) -> list[dict[str, Any]]:
  with (run_dir / "history.csv").open() as f:
    return list(csv.DictReader(f))


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
  rows = []
  if not path.exists():
    return rows
  with path.open() as f:
    for line in f:
      if not line.strip():
        continue
      rows.append(json.loads(line))
      if limit is not None and len(rows) >= limit:
        break
  return rows


def as_float(row: dict[str, Any], key: str) -> float:
  value = row.get(key, math.nan)
  try:
    return float(value)
  except (TypeError, ValueError):
    return math.nan


def smooth(values: list[float], window: int) -> np.ndarray:
  array = np.asarray(values, dtype=float)
  if len(array) == 0 or window <= 1:
    return array
  window = min(window, len(array))
  kernel = np.ones(window, dtype=float) / window
  return np.convolve(array, kernel, mode="same")


def memory_peak_gb(summary: dict[str, Any]) -> float:
  memory = summary.get("memory_after_train", {})
  aggregate = memory.get("aggregate", {})
  peak = aggregate.get("peak_bytes_in_use", math.nan)
  try:
    return float(peak) / 1e9
  except (TypeError, ValueError):
    return math.nan


def format_steps(steps: Any) -> str:
  try:
    value = int(steps)
  except (TypeError, ValueError):
    return str(steps)
  if value >= 1000 and value % 1000 == 0:
    return f"{value // 1000}K"
  return str(value)


def make_label(summary: dict[str, Any]) -> str:
  return f"{summary['variant'].title()} {format_steps(summary.get('steps_recorded', 0))}"


def loss_tokens(history: list[dict[str, Any]]) -> int:
  if not history:
    return 0
  try:
    return int(float(history[-1].get("cumulative_loss_tokens", 0)))
  except (TypeError, ValueError):
    return 0


def build_summary_rows(
    runs: list[tuple[Path, dict[str, Any], str]],
    histories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
  rows = []
  for run_dir, summary, label in runs:
    quality = summary.get("quality", {})
    rows.append({
        "label": label,
        "variant": summary["variant"],
        "run_dir": str(run_dir),
        "tpu_name": summary.get("runtime", {}).get("tpu_name", ""),
        "tpu_zone": summary.get("runtime", {}).get("tpu_zone", ""),
        "jax_devices": len(summary.get("jax_devices", [])),
        "steps": summary.get("steps_recorded", 0),
        "loss_tokens": loss_tokens(histories.get(label, [])),
        "batch_size": summary.get("batch_size", 0),
        "max_length": summary.get("max_length", 0),
        "final_loss": summary.get("final_loss", math.nan),
        "eval_loss": quality.get("eval_loss", math.nan),
        "bleu": quality.get("bleu", math.nan),
        "chrf": quality.get("chrf", math.nan),
        "wall_time_sec": summary.get("wall_time_sec", math.nan),
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
        "peak_memory_gb_after_train": memory_peak_gb(summary),
        "packing_efficiency": summary.get("packing", {}).get(
            "packed_efficiency",
            math.nan,
        ),
        "prepared_batches": summary.get("prepared_batches", 0),
        "repeats_data": summary.get("repeats_data", False),
    })
  return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys())
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def plot_loss(
    histories: dict[str, list[dict[str, Any]]],
    outdir: Path,
) -> Path:
  colors = ["#4C78A8", "#54A24B", "#F58518", "#B279A2", "#E45756"]
  fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
  for idx, (label, rows) in enumerate(histories.items()):
    color = colors[idx % len(colors)]
    steps = [int(float(row["step"])) for row in rows]
    tokens = [int(float(row["cumulative_loss_tokens"])) for row in rows]
    losses = [as_float(row, "loss") for row in rows]
    axes[0].plot(
        steps,
        losses,
        color=color,
        alpha=0.14,
        linewidth=0.8,
    )
    axes[0].plot(
        steps,
        smooth(losses, 100),
        color=color,
        linewidth=2.0,
        label=label,
    )
    axes[1].plot(
        tokens,
        losses,
        color=color,
        alpha=0.14,
        linewidth=0.8,
    )
    axes[1].plot(
        tokens,
        smooth(losses, 100),
        color=color,
        linewidth=2.0,
        label=label,
    )
  axes[0].set_title("Training Loss by Optimizer Step")
  axes[0].set_xlabel("Optimizer step")
  axes[0].set_ylabel("Loss")
  axes[0].legend()
  axes[1].set_title("Training Loss by Consumed Loss Tokens")
  axes[1].set_xlabel("Cumulative loss tokens")
  axes[1].set_ylabel("Loss")
  axes[1].legend()
  fig.tight_layout()
  path = outdir / "loss_curves.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  return path


def format_metric_value(key: str, value: float) -> str:
  if key == "wall_time_sec":
    return f"{value:.0f}s"
  if key == "mean_step_time_sec_excl_first":
    return f"{value:.3f}s"
  if key == "peak_memory_gb_after_train":
    return f"{value:.2f} GB"
  if key == "loss_tokens":
    return f"{value / 1e6:.2f}M"
  if key == "packing_efficiency":
    return f"{value * 100:.1f}%"
  if key in {
      "valid_tokens_per_sec_excl_first",
      "loss_tokens_per_sec_excl_first",
  }:
    return f"{value / 1000:.1f}k/s"
  if key in {"bleu", "chrf"}:
    return f"{value:.2f}"
  return f"{value:.3f}"


def effect_color(ratio: float, direction: str) -> str:
  if not math.isfinite(ratio):
    return "#6B7280"
  neutral_low = 1 / 1.03
  neutral_high = 1.03
  if neutral_low <= ratio <= neutral_high:
    return "#6B7280"
  improved = ratio > neutral_high if direction == "higher" else ratio < neutral_low
  return "#2E7D32" if improved else "#C46A1A"


def plot_metric_scorecard(
    summary_rows: list[dict[str, Any]],
    outdir: Path,
) -> Path:
  baseline = summary_rows[0]
  candidate = summary_rows[1]
  panels = [
      (
          "Runtime Cost",
          "lower is better",
          [
              ("wall_time_sec", "Wall time", "lower"),
              ("mean_step_time_sec_excl_first", "Step time", "lower"),
              ("peak_memory_gb_after_train", "Peak memory", "lower"),
          ],
          (0.25, 1.18),
          [0.25, 0.5, 1.0],
      ),
      (
          "Useful Work",
          "higher is better",
          [
              ("loss_tokens", "Loss tokens seen", "higher"),
              ("packing_efficiency", "Token density", "higher"),
              ("loss_tokens_per_sec_excl_first", "Loss tokens/sec", "higher"),
          ],
          (0.75, 11.5),
          [1.0, 2.0, 5.0, 10.0],
      ),
      (
          "Quality",
          "near parity is the target",
          [
              ("eval_loss", "Eval loss", "lower"),
              ("bleu", "BLEU", "higher"),
              ("chrf", "chrF", "higher"),
          ],
          (0.94, 1.08),
          [0.95, 1.0, 1.08],
      ),
  ]
  baseline_color = "#4C78A8"
  fig, axes = plt.subplots(
      1,
      3,
      figsize=(15.8, 5.8),
      gridspec_kw={"width_ratios": [1.0, 1.12, 1.0]},
  )
  fig.suptitle(
      f"{candidate['label']} vs {baseline['label']}",
      fontsize=16,
      y=0.98,
  )
  fig.text(
      0.5,
      0.925,
      "Relative scorecard: baseline is 1.0x; small gray band marks +/-3% parity.",
      ha="center",
      fontsize=10,
      color="#4B5563",
  )

  for ax, (title, subtitle, metrics, xlim, ticks) in zip(axes, panels):
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.axvspan(1 / 1.03, 1.03, color="#E5E7EB", alpha=0.65, zorder=0)
    ax.axvline(1.0, color="#111827", linewidth=1.1, alpha=0.75, zorder=1)
    y_positions = np.arange(len(metrics))[::-1]
    for y, (key, metric_label, direction) in zip(y_positions, metrics):
      baseline_value = float(baseline[key])
      candidate_value = float(candidate[key])
      ratio = candidate_value / baseline_value if baseline_value else math.nan
      color = effect_color(ratio, direction)
      if math.isfinite(ratio):
        ax.hlines(
            y,
            min(1.0, ratio),
            max(1.0, ratio),
            color=color,
            linewidth=3.2,
            alpha=0.68,
            zorder=2,
        )
        ax.scatter(
            [ratio],
            [y],
            s=92,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        label_x = ratio * (1.06 if ratio >= 1 else 0.94)
        label_x = min(max(label_x, xlim[0] * 1.04), xlim[1] / 1.04)
        ax.text(
            label_x,
            y + 0.18,
            f"{ratio:.2f}x",
            ha="left" if ratio >= 1 else "right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color,
        )
      ax.scatter(
          [1.0],
          [y],
          s=54,
          color=baseline_color,
          edgecolor="white",
          linewidth=0.8,
          zorder=3,
      )
      ax.text(
          xlim[0] * 1.04,
          y - 0.28,
          (
              f"{format_metric_value(key, candidate_value)} vs "
              f"{format_metric_value(key, baseline_value)}"
          ),
          ha="left",
          va="center",
          fontsize=8.4,
          color="#6B7280",
      )
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, pad=12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([metric[1] for metric in metrics], fontsize=10)
    ax.set_ylim(-0.65, len(metrics) - 0.45)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{tick:g}x" for tick in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
      ax.spines[spine].set_visible(False)
  handles = [
      Line2D(
          [0],
          [0],
          marker="o",
          color="none",
          markerfacecolor=baseline_color,
          markeredgecolor="white",
          markersize=8,
          label=baseline["label"],
      ),
      Line2D(
          [0],
          [0],
          marker="o",
          color="none",
          markerfacecolor="#2E7D32",
          markeredgecolor="white",
          markersize=8,
          label=candidate["label"],
      ),
  ]
  fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
  fig.tight_layout(rect=(0.02, 0.08, 1, 0.9), w_pad=2.4)
  path = outdir / "metric_bars.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  return path


def plot_metric_small_multiples(
    summary_rows: list[dict[str, Any]],
    outdir: Path,
) -> Path:
  metrics = [
      ("wall_time_sec", "Wall Time", "seconds", "{:.0f}s"),
      ("mean_step_time_sec_excl_first", "Step Time", "seconds", "{:.3f}s"),
      ("peak_memory_gb_after_train", "Peak Memory", "GB", "{:.2f}"),
      ("loss_tokens", "Loss Tokens Seen", "tokens", "{:.2e}"),
      ("packing_efficiency", "Token Density", "ratio", "{:.1%}"),
      ("loss_tokens_per_sec_excl_first", "Loss Tokens/sec", "tokens/sec", "{:.0f}"),
      ("eval_loss", "Eval Loss", "loss", "{:.3f}"),
      ("bleu", "BLEU", "score", "{:.2f}"),
      ("chrf", "chrF", "score", "{:.2f}"),
  ]
  colors = ["#4C78A8", "#54A24B", "#F58518", "#B279A2", "#E45756"]
  labels = [row["label"] for row in summary_rows]
  fig, axes = plt.subplots(3, 3, figsize=(15, 10))
  for ax, (key, title, ylabel, fmt) in zip(axes.ravel(), metrics):
    values = [float(row[key]) for row in summary_rows]
    bars = ax.bar(
        labels,
        values,
        color=[colors[idx % len(colors)] for idx, _ in enumerate(labels)],
        width=0.6,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, values):
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height(),
          fmt.format(value),
          ha="center",
          va="bottom",
          fontsize=9,
      )
  fig.tight_layout()
  path = outdir / "metric_bars.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  return path


def plot_metrics(summary_rows: list[dict[str, Any]], outdir: Path) -> Path:
  if len(summary_rows) == 2:
    return plot_metric_scorecard(summary_rows, outdir)
  return plot_metric_small_multiples(summary_rows, outdir)


def escape_md(value: Any) -> str:
  return str(value or "").replace("\n", " ").replace("|", "\\|")


def write_samples(
    path: Path,
    samples: dict[str, list[dict[str, Any]]],
    *,
    limit: int,
) -> None:
  variants = list(samples)
  rows = []
  for idx in range(limit):
    row = {"index": idx + 1}
    for variant in variants:
      if idx >= len(samples[variant]):
        continue
      sample = samples[variant][idx]
      row["source"] = sample.get("source", "")
      row["reference"] = sample.get("reference", "")
      row[f"{variant}_prediction"] = sample.get("prediction", "")
    rows.append(row)
  lines = [
      "| # | Source | Reference | " + " | ".join(f"{v} prediction" for v in variants) + " |",
      "| ---: | --- | --- | " + " | ".join("---" for _ in variants) + " |",
  ]
  for row in rows:
    cells = [
        str(row["index"]),
        escape_md(row.get("source", "")),
        escape_md(row.get("reference", "")),
    ]
    for variant in variants:
      cells.append(escape_md(row.get(f"{variant}_prediction", "")))
    lines.append("| " + " | ".join(cells) + " |")
  path.write_text("\n".join(lines) + "\n")


def write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    plots: list[Path],
    sample_path: Path,
) -> None:
  lines = [
      "# Gemma3 270M EN-FR Packing Quality Comparison",
      "",
      "This report compares Default CE Tunix SFT with sequence packing disabled "
      "and enabled. All runs use Gemma3 270M IT, LoRA rank 16, batch size 16, "
      "max length 512, and OPUS100 EN-FR. The optimizer-step budgets are "
      "intentionally different because packing changes how many useful target "
      "tokens each step carries.",
      "",
  ]
  for plot in plots:
    lines.append(f"![{plot.stem}]({plot.name})")
    lines.append("")
  lines.extend([
      "## Summary Table",
      "",
      "| Run | Variant | TPU | Devices | Steps | Loss tokens | Final loss | Eval loss | BLEU | chrF | Wall time | Peak GB | Packing efficiency |",
      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  for row in summary_rows:
    lines.append(
        "| "
        f"{row['label']} | "
        f"{row['variant']} | "
        f"{row['tpu_name']} | "
        f"{row['jax_devices']} | "
        f"{row['steps']} | "
        f"{row['loss_tokens']} | "
        f"{float(row['final_loss']):.4f} | "
        f"{float(row['eval_loss']):.4f} | "
        f"{float(row['bleu']):.2f} | "
        f"{float(row['chrf']):.2f} | "
        f"{float(row['wall_time_sec']):.0f}s | "
        f"{float(row['peak_memory_gb_after_train']):.2f} | "
        f"{float(row['packing_efficiency']) * 100:.1f}% |"
    )
  lines.extend([
      "",
      "## Translation Samples From These Runs",
      "",
      sample_path.read_text(),
  ])
  path.write_text("\n".join(lines) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", action="append", default=[])
  parser.add_argument("--unpacked-run")
  parser.add_argument("--packed-run")
  parser.add_argument("--outdir", required=True)
  parser.add_argument("--sample-limit", type=int, default=10)
  args = parser.parse_args()

  outdir = Path(args.outdir).expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  if args.run:
    run_dirs = [Path(run).resolve() for run in args.run]
  elif args.unpacked_run and args.packed_run:
    run_dirs = [Path(args.unpacked_run).resolve(), Path(args.packed_run).resolve()]
  else:
    parser.error("provide --run at least once, or both --unpacked-run and --packed-run")

  summaries = [(run_dir, read_summary(run_dir)) for run_dir in run_dirs]
  labels = [make_label(summary) for _, summary in summaries]
  if len(labels) != len(set(labels)):
    seen: dict[str, int] = {}
    unique_labels = []
    for label in labels:
      seen[label] = seen.get(label, 0) + 1
      unique_labels.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    labels = unique_labels
  runs = [
      (run_dir, summary, label)
      for (run_dir, summary), label in zip(summaries, labels)
  ]
  histories = {
      label: read_history(run_dir)
      for run_dir, _summary, label in runs
  }
  samples = {
      label: read_jsonl(
          run_dir / summary["variant"] / "translations.jsonl",
          limit=args.sample_limit,
      )
      for run_dir, summary, label in runs
  }

  summary_rows = build_summary_rows(runs, histories)
  write_csv(outdir / "summary.csv", summary_rows)
  plots = [
      plot_loss(histories, outdir),
      plot_metrics(summary_rows, outdir),
  ]
  sample_path = outdir / "translation_samples.md"
  write_samples(sample_path, samples, limit=args.sample_limit)
  write_report(
      outdir / "README.md",
      summary_rows=summary_rows,
      plots=plots,
      sample_path=sample_path,
  )
  print(f"outdir={outdir}")
  for plot in plots:
    print(f"plot={plot}")
  print(f"summary={outdir / 'summary.csv'}")
  print(f"report={outdir / 'README.md'}")


if __name__ == "__main__":
  main()
