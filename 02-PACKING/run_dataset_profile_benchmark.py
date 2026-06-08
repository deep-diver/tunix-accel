#!/usr/bin/env python3
"""Dataset and max-length profile sweep for sequence packing.

This benchmark does not instantiate Gemma weights. It formats several SFT-like
datasets with Gemma turn tokens, tokenizes them, and measures how much padding
waste sequence packing can remove as max_length changes.
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
class DatasetSpec:
  name: str
  records: list[dict[str, Any]]
  source: str


def gemma_prompt(text: str) -> str:
  return (
      "<start_of_turn>user\n"
      f"{text.strip()}\n"
      "<end_of_turn>\n"
      "<start_of_turn>model\n"
  )


def tokenized_record(
    tokenizer,
    *,
    prompt: str,
    answer: str,
    example_id: int,
) -> dict[str, Any]:
  prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
  answer_ids = tokenizer(answer.strip(), add_special_tokens=False)["input_ids"]
  eos_id = tokenizer.eos_token_id
  if eos_id is not None:
    answer_ids = answer_ids + [int(eos_id)]
  input_ids = [int(token) for token in prompt_ids + answer_ids]
  loss_mask = [False] * len(prompt_ids) + [True] * len(answer_ids)
  return {
      "id": example_id,
      "input_ids": input_ids,
      "labels": input_ids,
      "loss_mask": loss_mask,
      "prompt_tokens": len(prompt_ids),
      "answer_tokens": len(answer_ids),
  }


def load_opus100(tokenizer, *, num_examples: int) -> DatasetSpec:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  dataset = load_dataset(
      "Helsinki-NLP/opus-100",
      "en-fr",
      split=f"train[:{num_examples}]",
  )
  records = []
  for idx, row in enumerate(dataset):
    translation = row["translation"]
    prompt = gemma_prompt(
        "Translate this into French:\n" + str(translation["en"]).strip()
    )
    records.append(
        tokenized_record(
            tokenizer,
            prompt=prompt,
            answer=str(translation["fr"]),
            example_id=idx,
        )
    )
  return DatasetSpec(
      name="opus100_en_fr",
      records=records,
      source="Helsinki-NLP/opus-100 en-fr train split",
  )


def load_alpaca(tokenizer, *, num_examples: int) -> DatasetSpec:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{num_examples}]")
  records = []
  for idx, row in enumerate(dataset):
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if input_text:
      source = f"Instruction:\n{instruction}\n\nInput:\n{input_text}"
    else:
      source = f"Instruction:\n{instruction}"
    records.append(
        tokenized_record(
            tokenizer,
            prompt=gemma_prompt(source),
            answer=str(row.get("output", "")).strip(),
            example_id=idx,
        )
    )
  return DatasetSpec(
      name="alpaca",
      records=records,
      source="tatsu-lab/alpaca train split",
  )


def load_oasst1(tokenizer, *, num_examples: int) -> DatasetSpec:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  dataset = load_dataset("OpenAssistant/oasst1", split="train")
  rows = [row for row in dataset if str(row.get("lang", "")).lower() == "en"]
  by_id = {row.get("message_id"): row for row in rows}
  records = []
  for row in rows:
    if row.get("role") != "assistant":
      continue
    parent = by_id.get(row.get("parent_id"))
    if not parent or parent.get("role") not in {"prompter", "user"}:
      continue
    prompt_text = str(parent.get("text", "")).strip()
    answer = str(row.get("text", "")).strip()
    if not prompt_text or not answer:
      continue
    records.append(
        tokenized_record(
            tokenizer,
            prompt=gemma_prompt(prompt_text),
            answer=answer,
            example_id=len(records),
        )
    )
    if len(records) >= num_examples:
      break
  return DatasetSpec(
      name="oasst1_en",
      records=records,
      source="OpenAssistant/oasst1 English assistant replies",
  )


def load_cnn_dailymail(tokenizer, *, num_examples: int) -> DatasetSpec:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  dataset = load_dataset(
      "cnn_dailymail",
      "3.0.0",
      split=f"train[:{num_examples}]",
  )
  records = []
  for idx, row in enumerate(dataset):
    prompt = gemma_prompt(
        "Summarize the following article:\n\n"
        + str(row.get("article", "")).strip()
    )
    records.append(
        tokenized_record(
            tokenizer,
            prompt=prompt,
            answer=str(row.get("highlights", "")).strip(),
            example_id=idx,
        )
    )
  return DatasetSpec(
      name="cnn_dailymail",
      records=records,
      source="cnn_dailymail 3.0.0 train split",
  )


LOADERS = {
    "opus100": load_opus100,
    "alpaca": load_alpaca,
    "oasst1": load_oasst1,
    "cnn_dailymail": load_cnn_dailymail,
}


def dynamic_capacity(lengths: list[int], *, batch_size: int, max_length: int) -> int:
  capacity = 0
  for start in range(0, len(lengths), batch_size):
    batch_lengths = [
        min(int(length), max_length)
        for length in lengths[start : start + batch_size]
    ]
    if batch_lengths:
      capacity += len(batch_lengths) * max(batch_lengths)
  return capacity


def mean(values: list[int]) -> float:
  return float(np.mean(values)) if values else 0.0


def percentile(values: list[int], q: float) -> float:
  return float(np.percentile(values, q)) if values else 0.0


def profile_dataset(
    dataset: DatasetSpec,
    *,
    batch_sizes: list[int],
    max_lengths: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  lengths = [len(record["input_ids"]) for record in dataset.records]
  loss_lengths = [sum(bool(v) for v in record["loss_mask"]) for record in dataset.records]
  prompt_lengths = [int(record["prompt_tokens"]) for record in dataset.records]
  answer_lengths = [int(record["answer_tokens"]) for record in dataset.records]
  summary = {
      "dataset": dataset.name,
      "source": dataset.source,
      "num_examples": len(dataset.records),
      "mean_length": mean(lengths),
      "p50_length": percentile(lengths, 50),
      "p90_length": percentile(lengths, 90),
      "p99_length": percentile(lengths, 99),
      "max_length_observed": max(lengths) if lengths else 0,
      "mean_prompt_tokens": mean(prompt_lengths),
      "mean_answer_tokens": mean(answer_lengths),
      "mean_loss_tokens": mean(loss_lengths),
  }

  rows: list[dict[str, Any]] = []
  for max_length in max_lengths:
    truncated_records = [
        {
            **record,
            "input_ids": record["input_ids"][:max_length],
            "labels": record["labels"][:max_length],
            "loss_mask": record["loss_mask"][:max_length],
        }
        for record in dataset.records
    ]
    clipped_lengths = [min(length, max_length) for length in lengths]
    clipped_loss_lengths = [
        sum(bool(v) for v in record["loss_mask"])
        for record in truncated_records
    ]
    total_valid = sum(clipped_lengths)
    total_loss = sum(clipped_loss_lengths)
    original_loss = max(sum(loss_lengths), 1)
    overlength = sum(1 for length in lengths if length > max_length)
    for batch_size in batch_sizes:
      fixed_rows = ceil(len(clipped_lengths) / batch_size) * batch_size
      fixed_capacity = fixed_rows * max_length
      dyn_capacity = dynamic_capacity(
          clipped_lengths,
          batch_size=batch_size,
          max_length=max_length,
      )
      start = time.perf_counter()
      packed = pack_records(
          truncated_records,
          max_length=max_length,
          long_example_policy="truncate",
          return_attention_mask=False,
      )
      packing_sec = time.perf_counter() - start
      packed_rows_rounded = ceil(packed.batch_size / batch_size) * batch_size
      packed_capacity = packed.batch_size * max_length
      packed_capacity_rounded = packed_rows_rounded * max_length
      packed_loss = sum(sum(bool(v) for v in row) for row in packed.loss_mask)
      fixed_valid_ratio = total_valid / fixed_capacity if fixed_capacity else 0.0
      fixed_loss_ratio = total_loss / fixed_capacity if fixed_capacity else 0.0
      dynamic_valid_ratio = estimate_unpacked_efficiency(
          clipped_lengths,
          batch_size=batch_size,
          max_length=max_length,
      )
      dynamic_loss_ratio = total_loss / dyn_capacity if dyn_capacity else 0.0
      packed_valid_ratio = (
          packed.valid_tokens / packed_capacity if packed_capacity else 0.0
      )
      packed_loss_ratio = packed_loss / packed_capacity if packed_capacity else 0.0
      packed_valid_ratio_rounded = (
          packed.valid_tokens / packed_capacity_rounded
          if packed_capacity_rounded
          else 0.0
      )
      packed_loss_ratio_rounded = (
          packed_loss / packed_capacity_rounded
          if packed_capacity_rounded
          else 0.0
      )
      rows.append({
          "dataset": dataset.name,
          "batch_size": batch_size,
          "max_length": max_length,
          "examples": len(dataset.records),
          "overlength_examples": overlength,
          "overlength_rate": overlength / max(len(dataset.records), 1),
          "loss_tokens_retained_ratio": total_loss / original_loss,
          "unpacked_fixed_valid_ratio": fixed_valid_ratio,
          "unpacked_fixed_loss_ratio": fixed_loss_ratio,
          "unpacked_dynamic_valid_ratio": dynamic_valid_ratio,
          "unpacked_dynamic_loss_ratio": dynamic_loss_ratio,
          "packed_valid_ratio": packed_valid_ratio,
          "packed_loss_ratio": packed_loss_ratio,
          "packed_valid_ratio_rounded": packed_valid_ratio_rounded,
          "packed_loss_ratio_rounded": packed_loss_ratio_rounded,
          "row_reduction_x": len(dataset.records) / max(packed.batch_size, 1),
          "packed_rows": packed.batch_size,
          "packed_rows_rounded": packed_rows_rounded,
          "packed_vs_fixed_valid_gain_x": (
              packed_valid_ratio_rounded / fixed_valid_ratio
              if fixed_valid_ratio
              else 0.0
          ),
          "packed_vs_fixed_loss_gain_x": (
              packed_loss_ratio_rounded / fixed_loss_ratio
              if fixed_loss_ratio
              else 0.0
          ),
          "packed_vs_dynamic_loss_gain_x": (
              packed_loss_ratio_rounded / dynamic_loss_ratio
              if dynamic_loss_ratio
              else 0.0
          ),
          "packing_seconds": packing_sec,
      })
  return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    return
  keys = sorted({key for row in rows for key in row})
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def plot_profiles(
    rows: list[dict[str, Any]],
    *,
    outdir: Path,
    focus_batch_size: int,
) -> Path:
  outdir.mkdir(parents=True, exist_ok=True)
  focus = [
      row
      for row in rows
      if int(row["batch_size"]) == focus_batch_size
  ]
  datasets = sorted({str(row["dataset"]) for row in focus})
  max_lengths = sorted({int(row["max_length"]) for row in focus})
  colors = {
      "opus100_en_fr": "#4C78A8",
      "alpaca": "#F58518",
      "oasst1_en": "#54A24B",
      "cnn_dailymail": "#B279A2",
  }

  plt.style.use("seaborn-v0_8-whitegrid")
  fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
  axes = axes.ravel()

  def series(dataset: str, key: str) -> list[float]:
    lookup = {
        int(row["max_length"]): float(row[key])
        for row in focus
        if row["dataset"] == dataset
    }
    return [lookup.get(length, float("nan")) for length in max_lengths]

  for dataset in datasets:
    color = colors.get(dataset)
    axes[0].plot(
        max_lengths,
        [value * 100 for value in series(dataset, "unpacked_fixed_loss_ratio")],
        linestyle="--",
        marker="o",
        color=color,
        alpha=0.55,
    )
    axes[0].plot(
        max_lengths,
        [value * 100 for value in series(dataset, "packed_loss_ratio_rounded")],
        linestyle="-",
        marker="o",
        color=color,
        label=dataset,
    )
    axes[1].plot(
        max_lengths,
        series(dataset, "packed_vs_fixed_loss_gain_x"),
        marker="o",
        color=color,
        label=dataset,
    )
    axes[2].plot(
        max_lengths,
        series(dataset, "row_reduction_x"),
        marker="o",
        color=color,
        label=dataset,
    )
    axes[3].plot(
        max_lengths,
        [value * 100 for value in series(dataset, "loss_tokens_retained_ratio")],
        marker="o",
        color=color,
        label=dataset,
    )

  axes[0].set_title("Target-token density: dashed fixed vs solid packed")
  axes[0].set_ylabel("Loss-token density (%)")
  axes[1].set_title("Packed target-density gain vs fixed rows")
  axes[1].set_ylabel("Gain (x)")
  axes[2].set_title("Row reduction")
  axes[2].set_ylabel("Original examples / packed rows")
  axes[3].set_title("Target tokens retained after truncation")
  axes[3].set_ylabel("Retained target tokens (%)")
  for axis in axes:
    axis.set_xscale("log", base=2)
    axis.set_xticks(max_lengths)
    axis.set_xticklabels([str(length) for length in max_lengths])
    axis.set_xlabel("Max length")
  axes[0].legend(frameon=False, ncols=2, fontsize=9)
  fig.suptitle(
      f"Gemma3 270M Tokenizer Packing Profiles by Dataset (batch {focus_batch_size})",
      fontsize=13,
      color="#5F6B7A",
  )
  fig.tight_layout()
  path = outdir / "dataset_profile_overview.png"
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return path


def write_readme(
    path: Path,
    *,
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    plot: Path,
    focus_batch_size: int,
) -> None:
  focus = [
      row
      for row in rows
      if int(row["batch_size"]) == focus_batch_size
      and int(row["max_length"]) in {512, 1024, 2048}
  ]
  lines = [
      "# Dataset Profile Sweep",
      "",
      "This is a tokenizer-only sequence-packing sweep. It does not load Gemma "
      "weights; it measures whether a dataset has enough padding waste to make "
      "TPU training runs worth spending on.",
      "",
      f"![{plot.stem}]({plot.relative_to(path.parent).as_posix()})",
      "",
      "## Dataset Length Summary",
      "",
      (
          "| Dataset | Examples | Mean len | p90 len | p99 len | Mean answer "
          "tokens | Source |"
      ),
      "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
  ]
  for summary in summaries:
    lines.append(
        "| "
        f"{summary['dataset']} | "
        f"{summary['num_examples']} | "
        f"{float(summary['mean_length']):.1f} | "
        f"{float(summary['p90_length']):.1f} | "
        f"{float(summary['p99_length']):.1f} | "
        f"{float(summary['mean_answer_tokens']):.1f} | "
        f"{summary['source']} |"
    )
  lines.extend([
      "",
      f"## Focus Rows (batch {focus_batch_size})",
      "",
      (
          "| Dataset | Max length | Fixed loss density | Packed loss density | "
          "Gain | Row reduction | Retained target tokens |"
      ),
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  for row in sorted(focus, key=lambda item: (item["dataset"], item["max_length"])):
    lines.append(
        "| "
        f"{row['dataset']} | "
        f"{row['max_length']} | "
        f"{float(row['unpacked_fixed_loss_ratio']) * 100:.2f}% | "
        f"{float(row['packed_loss_ratio_rounded']) * 100:.2f}% | "
        f"{float(row['packed_vs_fixed_loss_gain_x']):.1f}x | "
        f"{float(row['row_reduction_x']):.1f}x | "
        f"{float(row['loss_tokens_retained_ratio']) * 100:.1f}% |"
    )
  path.write_text("\n".join(lines) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", default="google/gemma-3-270m-it")
  parser.add_argument(
      "--datasets",
      default="opus100,alpaca,oasst1",
      help="Comma-separated list from: " + ",".join(sorted(LOADERS)),
  )
  parser.add_argument("--num-examples", type=int, default=5000)
  parser.add_argument("--batch-sizes", default="8,16,32")
  parser.add_argument("--max-lengths", default="256,512,1024,2048,4096")
  parser.add_argument("--focus-batch-size", type=int, default=16)
  parser.add_argument("--outdir", default="02-PACKING/results/dataset-profile-270m")
  parser.add_argument("--allow-download", action="store_true")
  args = parser.parse_args()

  from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

  tokenizer = AutoTokenizer.from_pretrained(
      args.model_id,
      local_files_only=not args.allow_download,
  )
  dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
  unknown = sorted(set(dataset_names) - set(LOADERS))
  if unknown:
    raise ValueError(f"Unknown datasets: {unknown}. Allowed: {sorted(LOADERS)}")

  all_rows: list[dict[str, Any]] = []
  summaries: list[dict[str, Any]] = []
  for name in dataset_names:
    started = time.perf_counter()
    dataset = LOADERS[name](tokenizer, num_examples=args.num_examples)
    rows, summary = profile_dataset(
        dataset,
        batch_sizes=parse_ints(args.batch_sizes),
        max_lengths=parse_ints(args.max_lengths),
    )
    summary["elapsed_sec"] = time.perf_counter() - started
    all_rows.extend(rows)
    summaries.append(summary)
    print(
        "dataset",
        summary["dataset"],
        "examples",
        summary["num_examples"],
        "mean_len",
        f"{summary['mean_length']:.1f}",
        flush=True,
    )

  outdir = Path(args.outdir).expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  write_csv(outdir / "dataset_profile.csv", all_rows)
  write_json(outdir / "dataset_profile_summary.json", summaries)
  plot = plot_profiles(
      all_rows,
      outdir=outdir,
      focus_batch_size=args.focus_batch_size,
  )
  write_readme(
      outdir / "README.md",
      summaries=summaries,
      rows=all_rows,
      plot=plot,
      focus_batch_size=args.focus_batch_size,
  )
  print(f"rows_csv={outdir / 'dataset_profile.csv'}")
  print(f"summary_json={outdir / 'dataset_profile_summary.json'}")
  print(f"plot={plot}")


if __name__ == "__main__":
  main()
