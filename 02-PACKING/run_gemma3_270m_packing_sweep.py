#!/usr/bin/env python3
"""Run Gemma3 270M sequence-packing experiment grids on TPU.

This runner keeps the 02-PACKING workstream focused on one lever: ordinary
fixed-length SFT batches versus sequence-packed SFT batches. It delegates the
actual Tunix training run to `run_gemma_training_benchmark.py`, while adding
case naming, XLA memory-report parsing, and compact CSV/JSON summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAINING_RUNNER = SCRIPT_DIR / "run_gemma_training_benchmark.py"

ENV_KEYS = {
    "TUNIX_ACCEL_DISABLE_AUTOPATCH",
    "TUNIX_ACCEL_DISABLE_CE",
    "TUNIX_ACCEL_CE_TOKEN_CHUNK",
    "TUNIX_ACCEL_CE_VOCAB_CHUNK",
    "TUNIX_ACCEL_DISABLE_TILED_MLP",
    "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY",
    "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION",
}

MODEL_LABELS = {
    "google/gemma-3-270m-it": "Gemma3 270M",
    "google/gemma-3-1b-it": "Gemma3 1B",
    "google/gemma-4-e2b": "Gemma4 E2B",
}


def parse_csv_ints(value: str) -> list[int]:
  return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
  return [item.strip() for item in value.split(",") if item.strip()]


def model_label(model_id: str) -> str:
  return MODEL_LABELS.get(model_id.lower(), model_id.split("/")[-1])


def write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    return
  keys = sorted({key for row in rows for key in row})
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def latest_train_memory_report(xla_dir: Path) -> Path | None:
  reports = sorted(xla_dir.glob("*jit__train_step*memory-usage-report.txt"))
  return reports[-1] if reports else None


def parse_xla_total_gib(report: Path | None) -> float | None:
  if report is None or not report.exists():
    return None
  text = report.read_text(errors="ignore")
  match = re.search(r"Total bytes:\s+\d+\s+\(([\d.]+)GiB\)", text)
  if match:
    return float(match.group(1))
  match = re.search(r"Total bytes:\s+(\d+)", text)
  if match:
    return int(match.group(1)) / (1024**3)
  return None


def parse_failure(log_path: Path) -> dict[str, str]:
  text = log_path.read_text(errors="ignore") if log_path.exists() else ""
  if "CompileTimeHbmOom" in text:
    failure_type = "compile_oom"
  elif "RESOURCE_EXHAUSTED" in text:
    failure_type = "resource_exhausted"
  elif "Traceback" in text:
    failure_type = "error"
  else:
    failure_type = "unknown"
  row = {"failure_type": failure_type}
  match = re.search(
      r"Used\s+([\d.]+)G\s+of\s+([\d.]+)G\s+hbm\.\s+Exceeded.*?by\s+([\d.]+)G",
      text,
      flags=re.DOTALL,
  )
  if match:
    row["oom_used_gib"] = match.group(1)
    row["oom_limit_gib"] = match.group(2)
    row["oom_exceeded_gib"] = match.group(3)
  return row


def first_summary(run_dir: Path) -> dict[str, Any] | None:
  path = run_dir / "summary.json"
  if not path.exists():
    return None
  data = json.loads(path.read_text())
  if isinstance(data, list) and data:
    return dict(data[0])
  if isinstance(data, dict):
    return dict(data)
  return None


def read_history(run_dir: Path) -> list[dict[str, str]]:
  path = run_dir / "history.csv"
  if not path.exists():
    return []
  with path.open() as f:
    return list(csv.DictReader(f))


def cleanup_xla_dir(xla_dir: Path, *, keep_all_xla: bool) -> None:
  if keep_all_xla or not xla_dir.exists():
    return
  for path in xla_dir.iterdir():
    if path.name.endswith("memory-usage-report.txt"):
      continue
    if path.is_dir():
      shutil.rmtree(path, ignore_errors=True)
    else:
      path.unlink(missing_ok=True)


def configure_env(*, xla_dir: Path) -> dict[str, str]:
  env = os.environ.copy()
  for key in ENV_KEYS:
    env.pop(key, None)
  env.update({
      "PYTHONPATH": str(REPO_ROOT),
      "PYTHONUNBUFFERED": "1",
      "TUNIX_ACCEL_DISABLE_AUTOPATCH": "1",
      "TUNIX_ACCEL_DISABLE_CE": "1",
      "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
      "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
      "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "0",
      "XLA_FLAGS": f"--xla_dump_to={xla_dir} --xla_dump_hlo_as_text",
  })
  return env


def command_for_case(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant: str,
    batch_size: int,
    max_length: int,
    max_steps: int,
    skip_quality_eval: bool,
) -> list[str]:
  command = [
      sys.executable,
      str(TRAINING_RUNNER),
      "--model-id",
      args.model_id,
      "--model-source",
      args.model_source,
      "--model-path",
      args.model_path,
      "--model-download-path",
      args.model_download_path,
      "--tokenizer-source",
      args.tokenizer_source,
      "--tokenizer-path",
      args.tokenizer_path,
      "--dataset-mode",
      args.dataset_mode,
      "--long-example-policy",
      args.long_example_policy,
      "--num-examples",
      str(args.num_examples),
      "--variants",
      variant,
      "--batch-size",
      str(batch_size),
      "--max-length",
      str(max_length),
      "--max-steps",
      str(max_steps),
      "--learning-rate",
      str(args.learning_rate),
      "--lora-rank",
      str(args.lora_rank),
      "--lora-alpha",
      str(args.lora_alpha),
      "--mesh-fsdp",
      str(args.mesh_fsdp),
      "--mesh-tp",
      str(args.mesh_tp),
      "--max-inflight",
      str(args.max_inflight),
      "--log-every",
      str(args.log_every),
      "--seed",
      str(args.seed),
      "--outdir",
      str(run_dir),
  ]
  if skip_quality_eval:
    command.append("--skip-quality-eval")
  else:
    command.extend([
        "--eval-examples",
        str(args.eval_examples),
        "--eval-batches",
        str(args.eval_batches),
        "--generation-examples",
        str(args.generation_examples),
        "--generation-batch-size",
        str(args.generation_batch_size),
        "--max-generation-steps",
        str(args.max_generation_steps),
    ])
  if args.initialize_distributed:
    command.append("--initialize-distributed")
  if args.allow_download:
    command.append("--allow-download")
  return command


def run_case(
    *,
    args: argparse.Namespace,
    variant: str,
    batch_size: int,
    max_length: int,
    max_steps: int,
    suite: str,
    skip_quality_eval: bool,
) -> dict[str, Any]:
  case_name = f"{suite}_{variant}_b{batch_size}_l{max_length}_s{max_steps}"
  run_dir = args.outdir / case_name
  xla_dir = run_dir / "xla"
  if run_dir.exists() and not args.force:
    print(f"skip_existing {case_name}", flush=True)
  else:
    shutil.rmtree(run_dir, ignore_errors=True)
    xla_dir.mkdir(parents=True, exist_ok=True)
    env = configure_env(xla_dir=xla_dir)
    command = command_for_case(
        args=args,
        run_dir=run_dir,
        variant=variant,
        batch_size=batch_size,
        max_length=max_length,
        max_steps=max_steps,
        skip_quality_eval=skip_quality_eval,
    )
    log_path = run_dir / "runner.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w") as log:
      log.write("$ " + " ".join(command) + "\n")
      log.flush()
      proc = subprocess.run(
          command,
          cwd=REPO_ROOT,
          env=env,
          stdout=log,
          stderr=subprocess.STDOUT,
          check=False,
      )
    elapsed = time.monotonic() - started
    (run_dir / "elapsed_sec.txt").write_text(f"{elapsed:.6f}\n")
    (run_dir / "returncode.txt").write_text(f"{proc.returncode}\n")
    cleanup_xla_dir(xla_dir, keep_all_xla=args.keep_all_xla)

  summary = first_summary(run_dir)
  history = read_history(run_dir)
  report = latest_train_memory_report(xla_dir)
  row: dict[str, Any] = {
      "suite": suite,
      "case": case_name,
      "model": model_label(args.model_id),
      "model_id": args.model_id,
      "tpu": args.tpu,
      "chips": args.chips,
      "mesh_fsdp": args.mesh_fsdp,
      "mesh_tp": args.mesh_tp,
      "variant": variant,
      "batch_size": batch_size,
      "max_length": max_length,
      "lora_rank": args.lora_rank,
      "max_steps": max_steps,
      "dataset_mode": args.dataset_mode,
      "long_example_policy": args.long_example_policy,
      "run_dir": str(run_dir),
      "xla_report": str(report) if report else "",
      "xla_train_step_gib_per_chip": parse_xla_total_gib(report),
      "status": "ok" if summary else "failed",
  }
  if summary:
    packing = summary.get("packing", {})
    row.update({
        "steps_recorded": summary.get("steps_recorded"),
        "final_loss": summary.get("final_loss"),
        "mean_loss": summary.get("mean_loss"),
        "wall_time_sec": summary.get("wall_time_sec"),
        "mean_step_time_sec_excl_first": summary.get(
            "mean_step_time_sec_excl_first"
        ),
        "valid_tokens_per_sec_excl_first": summary.get(
            "valid_tokens_per_sec_excl_first"
        ),
        "loss_tokens_per_sec_excl_first": summary.get(
            "loss_tokens_per_sec_excl_first"
        ),
        "prepared_batches": summary.get("prepared_batches"),
        "source_examples_fit": summary.get("source_examples_fit"),
        "dropped_overlength": summary.get("dropped_overlength"),
        "packed_efficiency": packing.get("packed_efficiency"),
        "row_reduction_x": packing.get("row_reduction_x"),
        "packed_rows": packing.get("packed_rows"),
        "packed_rows_used": packing.get("packed_rows_used"),
      })
    memory = summary.get("memory_after_train", {}).get("aggregate", {})
    if memory:
      row.update({
          "runtime_peak_hbm_gb": memory.get("peak_bytes_in_use", 0) / 1e9,
          "runtime_hbm_limit_gb": memory.get("bytes_limit", 0) / 1e9,
      })
    quality = summary.get("quality", {})
    if isinstance(quality, dict):
      for key in ["eval_loss", "bleu", "chrf", "eval_batches"]:
        if key in quality:
          row[key] = quality[key]
    if history:
      row["final_cumulative_loss_tokens"] = history[-1].get(
          "cumulative_loss_tokens"
      )
      row["final_cumulative_valid_tokens"] = history[-1].get(
          "cumulative_valid_tokens"
      )
    if len(history) >= 1:
      row["first_step_time_sec"] = history[0].get("step_time_sec")
    if len(history) >= 2:
      row["second_step_time_sec"] = history[1].get("step_time_sec")
  else:
    row.update(parse_failure(run_dir / "runner.log"))

  write_json(run_dir / "case_summary.json", row)
  print(
      "case",
      row["status"],
      suite,
      variant,
      f"b{batch_size}",
      f"l{max_length}",
      "xla_gib",
      row.get("xla_train_step_gib_per_chip"),
      "density",
      row.get("packed_efficiency"),
      "loss_tps",
      row.get("loss_tokens_per_sec_excl_first"),
      flush=True,
  )
  return row


def suite_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
  cases: list[dict[str, Any]] = []
  for batch_size in parse_csv_ints(args.batch_sizes):
    for max_length in parse_csv_ints(args.contexts):
      for variant in parse_csv_strings(args.variants):
        cases.append({
            "suite": args.suite,
            "variant": variant,
            "batch_size": batch_size,
            "max_length": max_length,
            "max_steps": args.max_steps,
            "skip_quality_eval": args.skip_quality_eval,
        })
  return cases


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", default="google/gemma-3-270m-it")
  parser.add_argument("--model-source", default="gcs")
  parser.add_argument("--model-path", default="gs://gemma-data/checkpoints/gemma3-270m-it")
  parser.add_argument("--model-download-path", default="")
  parser.add_argument("--tokenizer-source", default="sentencepiece")
  parser.add_argument("--tokenizer-path", default="gs://gemma-data/tokenizers/tokenizer_gemma3.model")
  parser.add_argument("--allow-download", action="store_true")
  parser.add_argument("--suite", default="short-throughput")
  parser.add_argument("--variants", default="unpacked,packed")
  parser.add_argument("--batch-sizes", default="8,16,32")
  parser.add_argument("--contexts", default="512,1024")
  parser.add_argument(
      "--dataset-mode",
      choices=["synthetic", "opus100", "alpaca", "oasst1", "cnn_dailymail"],
      default="opus100",
  )
  parser.add_argument(
      "--long-example-policy",
      choices=["drop", "truncate"],
      default="drop",
  )
  parser.add_argument("--num-examples", type=int, default=5000)
  parser.add_argument("--max-steps", type=int, default=50)
  parser.add_argument("--learning-rate", type=float, default=2e-4)
  parser.add_argument("--lora-rank", type=int, default=16)
  parser.add_argument("--lora-alpha", type=float, default=32.0)
  parser.add_argument("--max-inflight", type=int, default=1)
  parser.add_argument("--mesh-fsdp", type=int, default=1)
  parser.add_argument("--mesh-tp", type=int, default=1)
  parser.add_argument("--tpu", default="v5litepod-1")
  parser.add_argument("--chips", type=int, default=1)
  parser.add_argument(
      "--initialize-distributed",
      action="store_true",
      help=(
          "Forward distributed JAX initialization to each training subprocess. "
          "Use when this sweep is launched on every host of a TPU pod slice."
      ),
  )
  parser.add_argument("--log-every", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--skip-quality-eval", action="store_true")
  parser.add_argument("--eval-examples", type=int, default=512)
  parser.add_argument("--eval-batches", type=int, default=32)
  parser.add_argument("--generation-examples", type=int, default=0)
  parser.add_argument("--generation-batch-size", type=int, default=8)
  parser.add_argument("--max-generation-steps", type=int, default=128)
  parser.add_argument("--outdir", type=Path, default=Path("/tmp/gemma3-270m-packing"))
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--keep-all-xla", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  args.outdir = args.outdir.expanduser().resolve()
  if args.mesh_fsdp * args.mesh_tp != args.chips:
    raise ValueError("mesh_fsdp * mesh_tp must equal chips.")
  args.outdir.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, Any]] = []
  results_path = args.outdir / f"{args.suite}_results.csv"
  for case in suite_cases(args):
    row = run_case(
        args=args,
        variant=case["variant"],
        batch_size=case["batch_size"],
        max_length=case["max_length"],
        max_steps=case["max_steps"],
        suite=case["suite"],
        skip_quality_eval=case["skip_quality_eval"],
    )
    rows.append(row)
    write_csv(results_path, rows)
    write_json(args.outdir / f"{args.suite}_results.json", rows)
  print(f"results={results_path}")


if __name__ == "__main__":
  main()
