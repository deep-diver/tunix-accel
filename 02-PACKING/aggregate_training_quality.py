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


def plot_metrics(summary_rows: list[dict[str, Any]], outdir: Path) -> Path:
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


def write_cce_aligned_samples(
    md_path: Path,
    csv_path: Path,
    *,
    cce_rows: list[dict[str, Any]],
    samples: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
  by_label = {
      label: {row.get("source", ""): row for row in rows}
      for label, rows in samples.items()
  }
  fieldnames = [
      "index",
      "source",
      "reference",
      "cce_default_b16",
      "cce_cce_b16",
      *[f"{label}_prediction" for label in samples],
  ]
  csv_rows = []
  for idx, cce_row in enumerate(cce_rows[:limit], start=1):
    source = cce_row.get("src", "")
    row = {
        "index": idx,
        "source": source,
        "reference": cce_row.get("reference", ""),
        "cce_default_b16": cce_row.get("predictions", {}).get("default_b16", ""),
        "cce_cce_b16": cce_row.get("predictions", {}).get("cce_b16", ""),
    }
    for label, sample_by_source in by_label.items():
      row[f"{label}_prediction"] = sample_by_source.get(source, {}).get(
          "prediction",
          "",
      )
    csv_rows.append(row)

  with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)

  lines = [
      "| # | Source | Reference | 01-CCE Default | 01-CCE CCE | "
      + " | ".join(f"{label}" for label in samples)
      + " |",
      "| ---: | --- | --- | --- | --- | "
      + " | ".join("---" for _ in samples)
      + " |",
  ]
  for row in csv_rows:
    cells = [
        row["index"],
        row["source"],
        row["reference"],
        row["cce_default_b16"],
        row["cce_cce_b16"],
        *[row[f"{label}_prediction"] for label in samples],
    ]
    lines.append("| " + " | ".join(escape_md(cell) for cell in cells) + " |")
  md_path.write_text("\n".join(lines) + "\n")
  return csv_rows


def compute_translation_quality(
    *,
    predictions: list[str],
    references: list[str],
) -> tuple[float, float]:
  import sacrebleu  # pylint: disable=import-outside-toplevel

  bleu = sacrebleu.corpus_bleu(predictions, [references]).score
  chrf = sacrebleu.corpus_chrf(predictions, [references]).score
  return float(bleu), float(chrf)


def write_cce_quality(
    path: Path,
    *,
    aligned_rows: list[dict[str, Any]],
    run_labels: list[str],
) -> list[dict[str, Any]]:
  references = [str(row["reference"]) for row in aligned_rows]
  columns = [
      ("01-CCE Default", "cce_default_b16"),
      ("01-CCE CCE", "cce_cce_b16"),
      *[(label, f"{label}_prediction") for label in run_labels],
  ]
  rows = []
  for label, key in columns:
    predictions = [str(row.get(key, "")) for row in aligned_rows]
    bleu, chrf = compute_translation_quality(
        predictions=predictions,
        references=references,
    )
    rows.append({
        "label": label,
        "sample_count": len(aligned_rows),
        "bleu": bleu,
        "chrf": chrf,
    })
  write_csv(path, rows)
  return rows


def format_quality_table(rows: list[dict[str, Any]]) -> str:
  lines = [
      "| Run | Samples | BLEU | chrF |",
      "| --- | ---: | ---: | ---: |",
  ]
  for row in rows:
    lines.append(
        "| "
        f"{row['label']} | "
        f"{row['sample_count']} | "
        f"{float(row['bleu']):.2f} | "
        f"{float(row['chrf']):.2f} |"
    )
  return "\n".join(lines)


def write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    plots: list[Path],
    sample_path: Path,
    cce_sample_path: Path | None,
    cce_quality_rows: list[dict[str, Any]] | None,
) -> None:
  lines = [
      "# Gemma3 270M EN-FR Packing Quality Comparison",
      "",
      "This report compares Default CE Tunix SFT with sequence packing disabled "
      "and enabled. All runs use Gemma3 270M IT, LoRA rank 16, batch size 16, "
      "max length 512, and OPUS100 EN-FR. The packed runs differ only in the "
      "number of optimizer steps.",
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
  if cce_sample_path is not None:
    if cce_quality_rows:
      lines.extend([
          "",
          "## 01-CCE-Aligned 16-Sample Quality",
          "",
          format_quality_table(cce_quality_rows),
      ])
    lines.extend([
        "",
        "## Translation Samples Aligned With 01-CCE",
        "",
        cce_sample_path.read_text(),
    ])
  path.write_text("\n".join(lines) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", action="append", default=[])
  parser.add_argument("--unpacked-run")
  parser.add_argument("--packed-run")
  parser.add_argument("--outdir", required=True)
  parser.add_argument("--cce-samples")
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
  cce_sample_path = None
  cce_quality_rows = None
  if args.cce_samples:
    cce_rows = read_jsonl(Path(args.cce_samples), limit=args.sample_limit)
    cce_sample_path = outdir / "cce_aligned_translation_samples.md"
    aligned_rows = write_cce_aligned_samples(
        cce_sample_path,
        outdir / "cce_aligned_translation_samples.csv",
        cce_rows=cce_rows,
        samples=samples,
        limit=args.sample_limit,
    )
    cce_quality_rows = write_cce_quality(
        outdir / "cce_aligned_quality.csv",
        aligned_rows=aligned_rows,
        run_labels=list(samples.keys()),
    )
  write_report(
      outdir / "README.md",
      summary_rows=summary_rows,
      plots=plots,
      sample_path=sample_path,
      cce_sample_path=cce_sample_path,
      cce_quality_rows=cce_quality_rows,
  )
  print(f"outdir={outdir}")
  for plot in plots:
    print(f"plot={plot}")
  print(f"summary={outdir / 'summary.csv'}")
  if cce_sample_path is not None:
    print(f"cce_samples={cce_sample_path}")
    print(f"cce_quality={outdir / 'cce_aligned_quality.csv'}")
  print(f"report={outdir / 'README.md'}")


if __name__ == "__main__":
  main()
