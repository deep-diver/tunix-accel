#!/usr/bin/env python3
"""Gemma-tokenizer packing benchmark for OPUS100 EN-FR SFT records.

This is the first "actual Gemma" packing step: it uses the real Gemma tokenizer
and a Gemma-style turn format, but still does not instantiate model weights.
That keeps the benchmark cheap while replacing the regex length proxy with real
token ids and real prompt/response loss masks.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_efficiency_benchmark import parse_ints  # pylint: disable=wrong-import-position
from tunix_accel.packing import estimate_unpacked_efficiency  # pylint: disable=wrong-import-position
from tunix_accel.packing import pack_records  # pylint: disable=wrong-import-position


@dataclass(frozen=True)
class TokenizedSftDataset:
  name: str
  model_id: str
  records: list[dict[str, Any]]
  source: str


def _translation_pair(row: dict[str, Any]) -> tuple[str, str]:
  translation = row["translation"]
  return translation["en"], translation["fr"]


def gemma_translation_prompt(source: str) -> str:
  return (
      "<start_of_turn>user\n"
      "Translate the following English text into French.\n\n"
      f"English: {source}\n"
      "<end_of_turn>\n"
      "<start_of_turn>model\n"
  )


def gemma_translation_answer(target: str) -> str:
  return f"{target}<end_of_turn>"


def tokenize_sft_record(tokenizer, source: str, target: str, *, example_id: int):
  prompt = gemma_translation_prompt(source)
  answer = gemma_translation_answer(target)
  prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
  answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
  input_ids = prompt_ids + answer_ids
  loss_mask = [False] * len(prompt_ids) + [True] * len(answer_ids)
  return {
      "id": example_id,
      "input_ids": input_ids,
      "labels": input_ids,
      "loss_mask": loss_mask,
      "source_chars": len(source),
      "target_chars": len(target),
      "prompt_tokens": len(prompt_ids),
      "answer_tokens": len(answer_ids),
  }


def load_opus100_gemma_records(
    *,
    model_id: str,
    num_examples: int,
    local_files_only: bool,
) -> TokenizedSftDataset:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel
  from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

  tokenizer = AutoTokenizer.from_pretrained(
      model_id,
      local_files_only=local_files_only,
  )
  dataset = load_dataset(
      "Helsinki-NLP/opus-100",
      "en-fr",
      split=f"train[:{num_examples}]",
  )
  records = []
  for idx, row in enumerate(dataset):
    source, target = _translation_pair(row)
    records.append(
        tokenize_sft_record(
            tokenizer,
            source,
            target,
            example_id=idx,
        )
    )
  return TokenizedSftDataset(
      name="opus100-en-fr-gemma-tokenizer",
      model_id=model_id,
      records=records,
      source=(
          "Helsinki-NLP/opus-100 en-fr train split, formatted with "
          "Gemma turn tokens and tokenized by the requested Gemma tokenizer"
      ),
  )


def fixed_max_ratio(
    lengths: list[int],
    *,
    max_length: int,
    batch_size: int,
) -> tuple[float, int]:
  clipped = [min(length, max_length) for length in lengths]
  capacity = ceil(len(clipped) / batch_size) * batch_size * max_length
  return sum(clipped) / capacity, capacity


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
    records: list[dict[str, Any]],
    *,
    max_length: int,
    batch_size: int,
) -> dict[str, float | int]:
  start = time.perf_counter()
  packed = pack_records(
      records,
      max_length=max_length,
      long_example_policy="truncate",
      return_attention_mask=False,
  )
  elapsed = time.perf_counter() - start
  rounded_rows = ceil(packed.batch_size / batch_size) * batch_size
  return {
      "packed_rows": packed.batch_size,
      "packed_rows_rounded": rounded_rows,
      "packed_valid_ratio": packed.valid_tokens / (packed.batch_size * max_length),
      "packed_valid_ratio_rounded": (
          packed.valid_tokens / (rounded_rows * max_length)
      ),
      "packing_seconds": elapsed,
  }


def run_benchmark(
    dataset: TokenizedSftDataset,
    *,
    batch_sizes: list[int],
    max_lengths: list[int],
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
  lengths = [len(record["input_ids"]) for record in dataset.records]
  answer_lengths = [int(record["answer_tokens"]) for record in dataset.records]
  summary = {
      "dataset": dataset.name,
      "source": dataset.source,
      "model_id": dataset.model_id,
      "num_examples": len(dataset.records),
      "min_length": int(min(lengths)),
      "mean_length": float(np.mean(lengths)),
      "median_length": float(np.percentile(lengths, 50)),
      "p90_length": float(np.percentile(lengths, 90)),
      "p99_length": float(np.percentile(lengths, 99)),
      "max_length_observed": int(max(lengths)),
      "mean_answer_tokens": float(np.mean(answer_lengths)),
  }

  rows: list[dict[str, float | int | str]] = []
  for max_length in max_lengths:
    clipped_valid_tokens = sum(min(length, max_length) for length in lengths)
    clipped_records = [
        {
            **record,
            "input_ids": record["input_ids"][:max_length],
            "labels": record["labels"][:max_length],
            "loss_mask": record["loss_mask"][:max_length],
        }
        for record in dataset.records
    ]
    trainable_tokens = sum(
        sum(bool(value) for value in record["loss_mask"])
        for record in clipped_records
    )
    for batch_size in batch_sizes:
      fixed_ratio, fixed_capacity = fixed_max_ratio(
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
          clipped_records,
          max_length=max_length,
          batch_size=batch_size,
      )
      packed_ratio = float(packed["packed_valid_ratio"])
      rows.append(
          {
              "dataset": dataset.name,
              "model_id": dataset.model_id,
              "batch_size": batch_size,
              "max_length": max_length,
              "examples": len(dataset.records),
              "valid_tokens": clipped_valid_tokens,
              "trainable_loss_tokens": trainable_tokens,
              "unpacked_rows": len(dataset.records),
              "packed_rows": packed["packed_rows"],
              "row_reduction_x": len(dataset.records) / int(packed["packed_rows"]),
              "unpacked_fixed_capacity_tokens": fixed_capacity,
              "unpacked_dynamic_capacity_tokens": dyn_capacity,
              "unpacked_fixed_valid_ratio": fixed_ratio,
              "unpacked_dynamic_valid_ratio": dynamic_ratio,
              "packed_valid_ratio": packed_ratio,
              "packed_valid_ratio_rounded": packed["packed_valid_ratio_rounded"],
              "packed_vs_fixed_density_gain_x": packed_ratio / fixed_ratio,
              "packed_vs_dynamic_density_gain_x": packed_ratio / dynamic_ratio,
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
  axes[0].set_title(f"Gemma Tokenizer Density (batch {focus_batch_size})")
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
      label="Gain vs Fixed",
  )
  axes[1].plot(
      max_lengths,
      [float(row["packed_vs_dynamic_density_gain_x"]) for row in focus],
      marker="o",
      label="Gain vs Dynamic",
  )
  axes[1].set_xscale("log", base=2)
  axes[1].set_xticks(max_lengths)
  axes[1].set_xticklabels([str(x) for x in max_lengths])
  axes[1].set_xlabel("Max Length")
  axes[1].set_ylabel("Multiplier (x)")
  axes[1].set_title("Packing Improvement")
  axes[1].legend()
  fig.tight_layout()
  overview = outdir / "gemma_tokenizer_packing_overview.png"
  fig.savefig(overview, dpi=180)
  plt.close(fig)

  return [overview]


def write_summary(
    path: Path,
    *,
    summary: dict[str, float | int | str],
    rows: list[dict[str, float | int | str]],
    plots: list[Path],
    focus_batch_size: int,
) -> None:
  focus = [row for row in rows if int(row["batch_size"]) == focus_batch_size]
  lines = [
      "# Gemma Tokenizer Packing Benchmark",
      "",
      f"- Dataset: `{summary['dataset']}`",
      f"- Model tokenizer: `{summary['model_id']}`",
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
      f"- Mean answer/loss tokens: {float(summary['mean_answer_tokens']):.1f}",
      "",
      "## Plot",
      "",
  ]
  for plot in plots:
    lines.append(f"![{plot.stem}]({plot.relative_to(path.parent).as_posix()})")
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
  for row in focus:
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
  parser.add_argument("--model-id", default="google/gemma-3-270m-it")
  parser.add_argument("--num-examples", type=int, default=5000)
  parser.add_argument("--batch-sizes", default="8,16,32,64")
  parser.add_argument("--max-lengths", default="256,512,1024,2048")
  parser.add_argument("--focus-batch-size", type=int, default=16)
  parser.add_argument("--outdir", default="02-PACKING/results/gemma-tokenizer")
  parser.add_argument("--allow-download", action="store_true")
  args = parser.parse_args()

  dataset = load_opus100_gemma_records(
      model_id=args.model_id,
      num_examples=args.num_examples,
      local_files_only=not args.allow_download,
  )
  rows, summary = run_benchmark(
      dataset,
      batch_sizes=parse_ints(args.batch_sizes),
      max_lengths=parse_ints(args.max_lengths),
  )
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  write_csv(outdir / "gemma_tokenizer_packing.csv", rows)
  (outdir / "length_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  plots = plot_results(rows, outdir=outdir, focus_batch_size=args.focus_batch_size)
  write_summary(
      outdir / "README.md",
      summary=summary,
      rows=rows,
      plots=plots,
      focus_batch_size=args.focus_batch_size,
  )

  print(f"dataset={summary['dataset']}")
  print(f"model_id={summary['model_id']}")
  print(f"rows_csv={outdir / 'gemma_tokenizer_packing.csv'}")
  print(f"summary_md={outdir / 'README.md'}")
  for plot in plots:
    print(f"plot={plot}")


if __name__ == "__main__":
  main()
