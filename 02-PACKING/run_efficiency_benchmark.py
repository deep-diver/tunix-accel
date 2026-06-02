#!/usr/bin/env python3
"""No-model sequence packing efficiency benchmark.

This benchmark only needs sequence lengths. It measures how much useful-token
density improves when tokenized SFT examples are packed into fixed-length rows.
No Gemma model, tokenizer, TPU, or Tunix runtime is required.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
import re
import sys
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunix_accel.packing import estimate_unpacked_efficiency
from tunix_accel.packing import pack_records


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class LengthDataset:
  name: str
  lengths: list[int]
  source: str


def simple_text_token_count(text: str) -> int:
  """A cheap tokenizer proxy for model-free length benchmarking."""
  return len(TOKEN_RE.findall(text))


def synthetic_sft_lengths(num_examples: int, *, seed: int) -> LengthDataset:
  """Creates a short-heavy SFT-like length distribution."""
  rng = np.random.default_rng(seed)
  buckets = rng.choice(
      np.array([0, 1, 2]),
      size=num_examples,
      p=np.array([0.72, 0.23, 0.05]),
  )
  lengths = np.empty(num_examples, dtype=np.int32)
  short = buckets == 0
  medium = buckets == 1
  long = buckets == 2
  lengths[short] = rng.lognormal(mean=3.7, sigma=0.45, size=short.sum()).astype(int)
  lengths[medium] = rng.lognormal(mean=5.35, sigma=0.35, size=medium.sum()).astype(int)
  lengths[long] = rng.lognormal(mean=6.75, sigma=0.32, size=long.sum()).astype(int)
  lengths = np.clip(lengths, 8, 4096)
  return LengthDataset(
      name="synthetic-sft-lengths",
      lengths=[int(value) for value in lengths],
      source="local synthetic short-heavy SFT-like length mixture",
  )


def opus100_en_fr_lengths(num_examples: int) -> LengthDataset:
  """Loads OPUS100 EN-FR and estimates SFT sequence lengths without Gemma."""
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  dataset = load_dataset(
      "Helsinki-NLP/opus-100",
      "en-fr",
      split=f"train[:{num_examples}]",
  )
  lengths: list[int] = []
  for row in dataset:
    translation = row["translation"]
    source = translation["en"]
    target = translation["fr"]
    prompt_overhead = 12
    length = (
        prompt_overhead
        + simple_text_token_count(source)
        + simple_text_token_count(target)
    )
    lengths.append(max(1, length))
  return LengthDataset(
      name="opus100-en-fr-simple-token-proxy",
      lengths=lengths,
      source=(
          "Helsinki-NLP/opus-100 en-fr train split with regex token-count "
          "proxy, not Gemma tokenizer"
      ),
  )


def load_lengths(dataset: str, num_examples: int, seed: int) -> LengthDataset:
  if dataset == "synthetic":
    return synthetic_sft_lengths(num_examples, seed=seed)
  if dataset == "opus100":
    return opus100_en_fr_lengths(num_examples)
  if dataset == "auto":
    try:
      return opus100_en_fr_lengths(num_examples)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"OPUS100 load failed; falling back to synthetic lengths: {exc}")
      return synthetic_sft_lengths(num_examples, seed=seed)
  raise ValueError(f"Unknown dataset {dataset!r}.")


def parse_ints(raw: str) -> list[int]:
  return [int(part.strip()) for part in raw.split(",") if part.strip()]


def percentile(values: Iterable[int], q: float) -> float:
  return float(np.percentile(np.asarray(list(values), dtype=np.float32), q))


def fixed_max_valid_ratio(
    lengths: list[int],
    *,
    max_length: int,
    batch_size: int,
) -> tuple[float, int, int]:
  clipped = [min(length, max_length) for length in lengths]
  valid_tokens = sum(clipped)
  rows = ceil(len(clipped) / batch_size) * batch_size
  capacity = rows * max_length
  return valid_tokens / capacity, valid_tokens, capacity


def dynamic_capacity(
    lengths: list[int],
    *,
    max_length: int,
    batch_size: int,
) -> int:
  capacity = 0
  for start in range(0, len(lengths), batch_size):
    batch_lengths = [
        min(int(length), max_length)
        for length in lengths[start : start + batch_size]
    ]
    capacity += len(batch_lengths) * max(batch_lengths)
  return capacity


def packed_metrics(
    lengths: list[int],
    *,
    max_length: int,
    batch_size: int,
) -> dict[str, float | int]:
  clipped_lengths = [min(length, max_length) for length in lengths]
  records = [
      {"id": idx, "input_ids": [1] * length}
      for idx, length in enumerate(clipped_lengths)
      if length > 0
  ]
  start = time.perf_counter()
  packed = pack_records(
      records,
      max_length=max_length,
      long_example_policy="truncate",
      return_attention_mask=False,
  )
  elapsed = time.perf_counter() - start

  rounded_rows = ceil(packed.batch_size / batch_size) * batch_size
  no_round_capacity = packed.batch_size * max_length
  rounded_capacity = rounded_rows * max_length
  valid_tokens = packed.valid_tokens
  return {
      "packed_rows": packed.batch_size,
      "packed_rows_rounded": rounded_rows,
      "packed_capacity_tokens": no_round_capacity,
      "packed_capacity_tokens_rounded": rounded_capacity,
      "packed_valid_ratio": valid_tokens / no_round_capacity,
      "packed_valid_ratio_rounded": valid_tokens / rounded_capacity,
      "packing_seconds": elapsed,
  }


def run_benchmark(
    length_dataset: LengthDataset,
    *,
    batch_sizes: list[int],
    max_lengths: list[int],
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
  lengths = length_dataset.lengths
  summary = {
      "dataset": length_dataset.name,
      "source": length_dataset.source,
      "num_examples": len(lengths),
      "min_length": int(min(lengths)),
      "mean_length": float(np.mean(lengths)),
      "median_length": percentile(lengths, 50),
      "p90_length": percentile(lengths, 90),
      "p99_length": percentile(lengths, 99),
      "max_length_observed": int(max(lengths)),
  }
  rows: list[dict[str, float | int | str]] = []
  for max_length in max_lengths:
    clipped = [min(length, max_length) for length in lengths]
    valid_tokens = sum(clipped)
    for batch_size in batch_sizes:
      fixed_ratio, _, fixed_capacity = fixed_max_valid_ratio(
          lengths,
          max_length=max_length,
          batch_size=batch_size,
      )
      dynamic_ratio = estimate_unpacked_efficiency(
          lengths,
          batch_size=batch_size,
          max_length=max_length,
      )
      dyn_capacity = dynamic_capacity(
          lengths,
          max_length=max_length,
          batch_size=batch_size,
      )
      packed = packed_metrics(
          lengths,
          max_length=max_length,
          batch_size=batch_size,
      )
      packed_rows = int(packed["packed_rows"])
      row_reduction = len(lengths) / packed_rows
      rows.append(
          {
              "dataset": length_dataset.name,
              "batch_size": batch_size,
              "max_length": max_length,
              "examples": len(lengths),
              "valid_tokens": valid_tokens,
              "unpacked_rows": len(lengths),
              "packed_rows": packed_rows,
              "row_reduction_x": row_reduction,
              "unpacked_fixed_capacity_tokens": fixed_capacity,
              "unpacked_dynamic_capacity_tokens": dyn_capacity,
              "packed_capacity_tokens": packed["packed_capacity_tokens"],
              "unpacked_fixed_valid_ratio": fixed_ratio,
              "unpacked_dynamic_valid_ratio": dynamic_ratio,
              "packed_valid_ratio": packed["packed_valid_ratio"],
              "packed_valid_ratio_rounded": packed["packed_valid_ratio_rounded"],
              "packed_vs_fixed_density_gain_x": (
                  packed["packed_valid_ratio"] / fixed_ratio
              ),
              "packed_vs_dynamic_density_gain_x": (
                  packed["packed_valid_ratio"] / dynamic_ratio
              ),
              "packing_seconds": packed["packing_seconds"],
          }
      )
  return rows, summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def plot_results(
    rows: list[dict[str, float | int | str]],
    *,
    outdir: Path,
    focus_batch_size: int,
) -> list[Path]:
  outdir.mkdir(parents=True, exist_ok=True)
  focus = [row for row in rows if int(row["batch_size"]) == focus_batch_size]
  max_lengths = [int(row["max_length"]) for row in focus]

  plt.style.use("seaborn-v0_8-whitegrid")
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

  axes[0].plot(
      max_lengths,
      [float(row["unpacked_fixed_valid_ratio"]) * 100 for row in focus],
      marker="o",
      label="Unpacked fixed max",
  )
  axes[0].plot(
      max_lengths,
      [float(row["unpacked_dynamic_valid_ratio"]) * 100 for row in focus],
      marker="o",
      label="Unpacked dynamic",
  )
  axes[0].plot(
      max_lengths,
      [float(row["packed_valid_ratio"]) * 100 for row in focus],
      marker="o",
      label="Packed",
  )
  axes[0].set_xscale("log", base=2)
  axes[0].set_xticks(max_lengths)
  axes[0].set_xticklabels([str(x) for x in max_lengths])
  axes[0].set_ylim(0, 105)
  axes[0].set_xlabel("Max Length")
  axes[0].set_ylabel("Valid Token Ratio (%)")
  axes[0].set_title(f"Useful Token Density (batch {focus_batch_size})")
  axes[0].legend()

  axes[1].plot(
      max_lengths,
      [float(row["row_reduction_x"]) for row in focus],
      marker="o",
      label="Rows Reduction",
  )
  axes[1].plot(
      max_lengths,
      [float(row["packed_vs_fixed_density_gain_x"]) for row in focus],
      marker="o",
      label="Density Gain vs Fixed",
  )
  axes[1].plot(
      max_lengths,
      [float(row["packed_vs_dynamic_density_gain_x"]) for row in focus],
      marker="o",
      label="Density Gain vs Dynamic",
  )
  axes[1].set_xscale("log", base=2)
  axes[1].set_xticks(max_lengths)
  axes[1].set_xticklabels([str(x) for x in max_lengths])
  axes[1].set_xlabel("Max Length")
  axes[1].set_ylabel("Multiplier (x)")
  axes[1].set_title("Packing Improvement")
  axes[1].legend()

  fig.tight_layout()
  overview = outdir / "packing_efficiency_overview.png"
  fig.savefig(overview, dpi=180)
  plt.close(fig)

  batch_focus_length = max_lengths[len(max_lengths) // 2]
  batch_rows = [
      row for row in rows if int(row["max_length"]) == batch_focus_length
  ]
  batch_sizes = [int(row["batch_size"]) for row in batch_rows]
  fig, ax = plt.subplots(figsize=(7.2, 4.4))
  ax.plot(
      batch_sizes,
      [float(row["unpacked_dynamic_valid_ratio"]) * 100 for row in batch_rows],
      marker="o",
      label="Unpacked dynamic",
  )
  ax.plot(
      batch_sizes,
      [float(row["packed_valid_ratio_rounded"]) * 100 for row in batch_rows],
      marker="o",
      label="Packed, batch-rounded",
  )
  ax.set_xscale("log", base=2)
  ax.set_xticks(batch_sizes)
  ax.set_xticklabels([str(x) for x in batch_sizes])
  ax.set_ylim(0, 105)
  ax.set_xlabel("Batch Size")
  ax.set_ylabel("Valid Token Ratio (%)")
  ax.set_title(f"Batch Size Sensitivity (max length {batch_focus_length})")
  ax.legend()
  fig.tight_layout()
  batch_plot = outdir / "packing_batch_sensitivity.png"
  fig.savefig(batch_plot, dpi=180)
  plt.close(fig)

  return [overview, batch_plot]


def write_markdown_summary(
    path: Path,
    *,
    summary: dict[str, float | int | str],
    rows: list[dict[str, float | int | str]],
    plots: list[Path],
    focus_batch_size: int,
) -> None:
  focus_rows = [row for row in rows if int(row["batch_size"]) == focus_batch_size]
  path.parent.mkdir(parents=True, exist_ok=True)
  lines = [
      "# No-Model Packing Efficiency Benchmark",
      "",
      f"- Dataset: `{summary['dataset']}`",
      f"- Source: {summary['source']}",
      f"- Examples: {summary['num_examples']}",
      (
          "- Lengths: "
          f"mean {float(summary['mean_length']):.1f}, "
          f"median {float(summary['median_length']):.1f}, "
          f"p90 {float(summary['p90_length']):.1f}, "
          f"p99 {float(summary['p99_length']):.1f}, "
          f"max {summary['max_length_observed']}"
      ),
      "",
      "## Plots",
      "",
  ]
  for plot in plots:
    rel = plot.relative_to(path.parent)
    lines.append(f"![{plot.stem}]({rel.as_posix()})")
    lines.append("")

  lines.extend(
      [
          f"## Focus Rows (Batch {focus_batch_size})",
          "",
          (
              "| Max Length | Fixed Unpacked | Dynamic Unpacked | Packed | "
              "Rows Reduction | Gain vs Fixed | Gain vs Dynamic |"
          ),
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
      ]
  )
  for row in focus_rows:
    lines.append(
        "| "
        f"{row['max_length']} | "
        f"{float(row['unpacked_fixed_valid_ratio']) * 100:.1f}% | "
        f"{float(row['unpacked_dynamic_valid_ratio']) * 100:.1f}% | "
        f"{float(row['packed_valid_ratio']) * 100:.1f}% | "
        f"{float(row['row_reduction_x']):.2f}x | "
        f"{float(row['packed_vs_fixed_density_gain_x']):.2f}x | "
        f"{float(row['packed_vs_dynamic_density_gain_x']):.2f}x |"
    )
  path.write_text("\n".join(lines) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", choices=["auto", "opus100", "synthetic"], default="auto")
  parser.add_argument("--num-examples", type=int, default=5000)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--batch-sizes", default="8,16,32,64")
  parser.add_argument("--max-lengths", default="256,512,1024,2048")
  parser.add_argument("--outdir", default="02-PACKING/results/no-model")
  parser.add_argument("--focus-batch-size", type=int, default=16)
  args = parser.parse_args()

  outdir = Path(args.outdir)
  length_dataset = load_lengths(args.dataset, args.num_examples, args.seed)
  batch_sizes = parse_ints(args.batch_sizes)
  max_lengths = parse_ints(args.max_lengths)
  rows, summary = run_benchmark(
      length_dataset,
      batch_sizes=batch_sizes,
      max_lengths=max_lengths,
  )

  write_csv(outdir / "packing_efficiency.csv", rows)
  (outdir / "length_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  plots = plot_results(
      rows,
      outdir=outdir,
      focus_batch_size=args.focus_batch_size,
  )
  write_markdown_summary(
      outdir / "README.md",
      summary=summary,
      rows=rows,
      plots=plots,
      focus_batch_size=args.focus_batch_size,
  )

  print(f"dataset={summary['dataset']}")
  print(f"rows_csv={outdir / 'packing_efficiency.csv'}")
  print(f"summary_md={outdir / 'README.md'}")
  for plot in plots:
    print(f"plot={plot}")


if __name__ == "__main__":
  main()
