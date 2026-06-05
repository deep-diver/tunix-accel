#!/usr/bin/env python3
"""Runs the LoRA-FA variant matrix via the shared Gemma training runner."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_RUNNER = REPO_ROOT / "02-PACKING" / "run_gemma_training_benchmark.py"


VARIANT_DEFAULTS: dict[str, dict[str, Any]] = {
    "standard_lora_r16": {
        "rank": 16,
        "enable_lora_fa": False,
        "lorafa_mode": "none",
    },
    "standard_lora_r32": {
        "rank": 32,
        "enable_lora_fa": False,
        "lorafa_mode": "none",
    },
    "standard_lora_r64": {
        "rank": 64,
        "enable_lora_fa": False,
        "lorafa_mode": "none",
    },
    "freeze_a_r16": {
        "rank": 16,
        "enable_lora_fa": True,
        "lorafa_mode": "freeze_a",
    },
    "lorafa_r16": {
        "rank": 16,
        "enable_lora_fa": True,
        "lorafa_mode": "corrected_b",
    },
    "lorafa_r32": {
        "rank": 32,
        "enable_lora_fa": True,
        "lorafa_mode": "corrected_b",
    },
    "lorafa_r64": {
        "rank": 64,
        "enable_lora_fa": True,
        "lorafa_mode": "corrected_b",
    },
}


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "gemma3-270m": {
        "model_id": "google/gemma-3-270m-it",
        "model_source": "gcs",
        "model_path": "gs://gemma-data/checkpoints/gemma3-270m-it",
        "tokenizer_source": "sentencepiece",
        "tokenizer_path": "gs://gemma-data/tokenizers/tokenizer_gemma3.model",
        "mesh_fsdp": 1,
        "mesh_tp": 1,
    },
    "gemma3-1b": {
        "model_id": "google/gemma-3-1b-it",
        "model_source": "gcs",
        "model_path": "gs://gemma-data/checkpoints/gemma3-1b-it",
        "tokenizer_source": "sentencepiece",
        "tokenizer_path": "gs://gemma-data/tokenizers/tokenizer_gemma3.model",
        "mesh_fsdp": 4,
        "mesh_tp": 1,
    },
    "gemma3-4b": {
        "model_id": "google/gemma-3-4b-it",
        "model_source": "gcs",
        "model_path": "gs://gemma-data/checkpoints/gemma3-4b-it",
        "tokenizer_source": "sentencepiece",
        "tokenizer_path": "gs://gemma-data/tokenizers/tokenizer_gemma3.model",
        "mesh_fsdp": 8,
        "mesh_tp": 1,
    },
    "gemma3-12b": {
        "model_id": "google/gemma-3-12b-it",
        "model_source": "gcs",
        "model_path": "gs://gemma-data/checkpoints/gemma3-12b-it",
        "tokenizer_source": "sentencepiece",
        "tokenizer_path": "gs://gemma-data/tokenizers/tokenizer_gemma3.model",
        "mesh_fsdp": 4,
        "mesh_tp": 1,
    },
    "gemma3-27b": {
        "model_id": "google/gemma-3-27b-it",
        "model_source": "gcs",
        "model_path": "gs://gemma-data/checkpoints/gemma3-27b-it",
        "tokenizer_source": "sentencepiece",
        "tokenizer_path": "gs://gemma-data/tokenizers/tokenizer_gemma3.model",
        "mesh_fsdp": 8,
        "mesh_tp": 1,
    },
    "gemma4-e2b": {
        "model_id": "google/gemma-4-e2b",
        "model_source": "huggingface",
        "model_path": "",
        "tokenizer_source": "huggingface",
        "tokenizer_path": "google/gemma-4-e2b",
        "hf_cache_name": "gemma4-e2b",
        "mesh_fsdp": 4,
        "mesh_tp": 1,
    },
    "gemma4-e4b": {
        "model_id": "google/gemma-4-e4b",
        "model_source": "huggingface",
        "model_path": "",
        "tokenizer_source": "huggingface",
        "tokenizer_path": "google/gemma-4-e4b",
        "hf_cache_name": "gemma4-e4b",
        "mesh_fsdp": 8,
        "mesh_tp": 1,
    },
}


def parse_csv_list(value: str) -> list[str]:
  return [item.strip() for item in value.split(",") if item.strip()]


def env_for_variant(variant: dict[str, Any], *, alpha: float, correction_eps: float):
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT)
  env["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "false"
  env["TUNIX_ACCEL_DISABLE_CE"] = "true"
  env["TUNIX_ACCEL_DISABLE_TILED_MLP"] = "true"
  env["TUNIX_ACCEL_ACTIVATION_POLICY"] = "none"
  env["TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION"] = "false"
  env["TUNIX_ACCEL_ENABLE_GEMMA4_HF_LOADER"] = "true"
  env["TUNIX_ACCEL_ENABLE_LORA_FA"] = (
      "true" if variant["enable_lora_fa"] else "false"
  )
  env["TUNIX_ACCEL_LORA_FA_MODE"] = variant["lorafa_mode"]
  env["TUNIX_ACCEL_LORA_FA_ALPHA"] = str(alpha)
  env["TUNIX_ACCEL_LORA_FA_CORRECTION_EPS"] = str(correction_eps)
  return env


def command_for_variant(
    *,
    model: dict[str, Any],
    variant: dict[str, Any],
    args: argparse.Namespace,
    outdir: Path,
) -> list[str]:
  cmd = [
      sys.executable,
      str(SHARED_RUNNER),
      "--allow-autopatch",
      "--variants",
      "unpacked",
      "--model-id",
      model["model_id"],
      "--model-source",
      model["model_source"],
      "--tokenizer-source",
      model["tokenizer_source"],
      "--tokenizer-path",
      model["tokenizer_path"],
      "--dataset-mode",
      args.dataset_mode,
      "--num-examples",
      str(args.num_examples),
      "--batch-size",
      str(args.batch_size),
      "--max-length",
      str(args.max_length),
      "--max-steps",
      str(args.max_steps),
      "--learning-rate",
      str(args.learning_rate),
      "--weight-decay",
      str(args.weight_decay),
      "--lora-rank",
      str(variant["rank"]),
      "--lora-alpha",
      str(args.lora_alpha),
      "--mesh-fsdp",
      str(args.mesh_fsdp or model["mesh_fsdp"]),
      "--mesh-tp",
      str(args.mesh_tp or model["mesh_tp"]),
      "--max-inflight",
      str(args.max_inflight),
      "--outdir",
      str(outdir),
      "--capture-lora-value-deltas",
  ]
  if model["model_path"]:
    cmd.extend(["--model-path", model["model_path"]])
  if args.skip_quality_eval:
    cmd.append("--skip-quality-eval")
  else:
    cmd.extend([
        "--eval-examples",
        str(args.eval_examples),
        "--eval-batches",
        str(args.eval_batches),
        "--generation-examples",
        str(args.generation_examples),
    ])
  if args.allow_download:
    cmd.append("--allow-download")
  model_download_path = args.model_download_path
  if not model_download_path and model["model_source"] == "huggingface":
    model_download_path = str(
        Path(args.outdir).expanduser().resolve()
        / "_model_cache"
        / model.get("hf_cache_name", model["model_id"].replace("/", "--"))
    )
  if model_download_path:
    cmd.extend(["--model-download-path", model_download_path])
  return cmd


def read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def gib(value: int | float | None) -> float | None:
  if value is None:
    return None
  return float(value) / (1024**3)


def memory_fields(summary: dict[str, Any]) -> dict[str, Any]:
  memory = summary.get("memory_after_train") or {}
  aggregate = memory.get("aggregate") or {}
  devices = memory.get("devices") or []
  per_chip_peak = [
      device.get("peak_bytes_in_use")
      for device in devices
      if device.get("peak_bytes_in_use") is not None
  ]
  per_chip_limit = [
      device.get("bytes_limit")
      for device in devices
      if device.get("bytes_limit") is not None
  ]
  return {
      "train_peak_aggregate_gib": gib(aggregate.get("peak_bytes_in_use")),
      "train_in_use_aggregate_gib": gib(aggregate.get("bytes_in_use")),
      "train_limit_aggregate_gib": gib(aggregate.get("bytes_limit")),
      "train_peak_max_chip_gib": gib(max(per_chip_peak) if per_chip_peak else None),
      "train_limit_min_chip_gib": gib(min(per_chip_limit) if per_chip_limit else None),
      "device_count": len(devices) or None,
  }


def flatten_summary(model_key: str, variant_name: str, summary: dict[str, Any]):
  accel = summary.get("accel", {})
  lora_fa = summary.get("lora_fa", {})
  before = lora_fa.get("parameter_summary_before", {})
  after = lora_fa.get("parameter_summary_after", {})
  delta = lora_fa.get("value_delta", {})
  quality = summary.get("quality", {})
  return {
      "model_key": model_key,
      "variant_name": variant_name,
      "status": "ok",
      "model_id": summary.get("model_id"),
      "batch_size": summary.get("batch_size"),
      "max_length": summary.get("max_length"),
      "max_steps_requested": summary.get("max_steps_requested"),
      "steps_recorded": summary.get("steps_recorded"),
      "learning_rate": summary.get("learning_rate"),
      "lora_rank": summary.get("lora_rank"),
      "lora_alpha": summary.get("lora_alpha"),
      "lora_fa_installed": accel.get("lora_fa_installed"),
      "final_loss": summary.get("final_loss"),
      "mean_loss": summary.get("mean_loss"),
      "mean_step_time_sec_excl_first": summary.get(
          "mean_step_time_sec_excl_first"
      ),
      "valid_tokens_per_sec_excl_first": summary.get(
          "valid_tokens_per_sec_excl_first"
      ),
      "loss_tokens_per_sec_excl_first": summary.get(
          "loss_tokens_per_sec_excl_first"
      ),
      "wall_time_sec": summary.get("wall_time_sec"),
      "lora_a_tensors": after.get("lora_a_tensors", before.get("lora_a_tensors")),
      "lora_b_tensors": after.get("lora_b_tensors", before.get("lora_b_tensors")),
      "lora_a_params": after.get("lora_a_params", before.get("lora_a_params")),
      "lora_b_params": after.get("lora_b_params", before.get("lora_b_params")),
      "lora_a_bytes": after.get("lora_a_bytes", before.get("lora_a_bytes")),
      "lora_b_bytes": after.get("lora_b_bytes", before.get("lora_b_bytes")),
      "lorafa_a_value_delta_max": delta.get("lorafa_a_value_delta_max"),
      "lorafa_b_value_delta_max": delta.get("lorafa_b_value_delta_max"),
      "lorafa_a_changed_tensors": delta.get("lorafa_a_changed_tensors"),
      "lorafa_b_changed_tensors": delta.get("lorafa_b_changed_tensors"),
      **memory_fields(summary),
      "bleu": quality.get("bleu"),
      "chrf": quality.get("chrf"),
      "quality_num_examples": quality.get("num_examples"),
  }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames: list[str] = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--models", default="gemma3-270m")
  parser.add_argument(
      "--variants",
      default=(
          "standard_lora_r16,standard_lora_r32,standard_lora_r64,"
          "freeze_a_r16,lorafa_r16,lorafa_r32,lorafa_r64"
      ),
  )
  parser.add_argument("--dataset-mode", choices=["opus100", "synthetic"], default="opus100")
  parser.add_argument("--num-examples", type=int, default=5000)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--max-length", type=int, default=512)
  parser.add_argument("--max-steps", type=int, default=50)
  parser.add_argument("--learning-rate", type=float, default=2e-4)
  parser.add_argument("--weight-decay", type=float, default=0.0)
  parser.add_argument("--lora-alpha", type=float, default=32.0)
  parser.add_argument("--correction-eps", type=float, default=1e-8)
  parser.add_argument("--mesh-fsdp", type=int, default=0)
  parser.add_argument("--mesh-tp", type=int, default=0)
  parser.add_argument("--max-inflight", type=int, default=1)
  parser.add_argument("--skip-quality-eval", action="store_true")
  parser.add_argument("--eval-examples", type=int, default=512)
  parser.add_argument("--eval-batches", type=int, default=32)
  parser.add_argument("--generation-examples", type=int, default=16)
  parser.add_argument("--allow-download", action="store_true")
  parser.add_argument("--model-download-path", default=None)
  parser.add_argument("--outdir", default="06-LORA-FA/results/matrix")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  outdir = Path(args.outdir).expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, Any]] = []
  for model_key in parse_csv_list(args.models):
    if model_key not in MODEL_PRESETS:
      raise ValueError(f"Unknown model preset: {model_key}")
    model = MODEL_PRESETS[model_key]
    for variant_name in parse_csv_list(args.variants):
      if variant_name not in VARIANT_DEFAULTS:
        raise ValueError(f"Unknown variant: {variant_name}")
      variant = VARIANT_DEFAULTS[variant_name]
      run_outdir = outdir / model_key / variant_name
      cmd = command_for_variant(
          model=model,
          variant=variant,
          args=args,
          outdir=run_outdir,
      )
      env = env_for_variant(
          variant,
          alpha=args.lora_alpha,
          correction_eps=args.correction_eps,
      )
      command_record = {
          "model_key": model_key,
          "variant_name": variant_name,
          "cmd": cmd,
          "env": {
              key: env[key]
              for key in sorted(env)
              if key.startswith("TUNIX_ACCEL_")
          },
      }
      (run_outdir / "command.json").parent.mkdir(parents=True, exist_ok=True)
      (run_outdir / "command.json").write_text(
          json.dumps(command_record, indent=2) + "\n"
      )
      if args.dry_run:
        print(" ".join(cmd))
        continue
      result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
      if result.returncode != 0:
        rows.append({
            "model_key": model_key,
            "variant_name": variant_name,
            "status": "failed",
            "returncode": result.returncode,
        })
        write_csv(outdir / "matrix_summary.csv", rows)
        raise SystemExit(result.returncode)
      summary_path = run_outdir / "unpacked" / "summary.json"
      rows.append(flatten_summary(model_key, variant_name, read_json(summary_path)))
      write_csv(outdir / "matrix_summary.csv", rows)


if __name__ == "__main__":
  main()
