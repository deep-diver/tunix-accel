#!/usr/bin/env python3
"""Gemma4 base-model benchmark wrapper for existing workstream addenda.

This wrapper deliberately disables quality/generation evaluation. The Gemma4
base-model checks compare memory, compile behavior, and step-time effects for
the existing CCE, packing, tiled-MLP, and activation-policy workstreams, not
translation quality.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAINING_RUNNER = REPO_ROOT / "02-PACKING" / "run_gemma_training_benchmark.py"

MODEL_PRESETS = {
    "e2b": "google/gemma-4-E2B",
    "e4b": "google/gemma-4-E4B",
}

VARIANT_ENV = {
    "default": {
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "cce": {
        "TUNIX_ACCEL_DISABLE_CE": "",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "packed": {
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "tiled_mlp": {
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "split_offload": {
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "split_offload",
    },
    "split_remat": {
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "split_remat",
    },
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="e2b")
  parser.add_argument("--model-id", default=None)
  parser.add_argument(
      "--variant",
      choices=sorted(VARIANT_ENV),
      required=True,
      help="Patch/input variant to run. Quality evaluation is always disabled.",
  )
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--max-length", type=int, default=2048)
  parser.add_argument("--max-steps", type=int, default=5)
  parser.add_argument("--num-examples", type=int, default=512)
  parser.add_argument("--learning-rate", type=float, default=2e-4)
  parser.add_argument("--lora-rank", type=int, default=16)
  parser.add_argument("--lora-alpha", type=float, default=32.0)
  parser.add_argument("--mesh-fsdp", type=int, default=0)
  parser.add_argument("--mesh-tp", type=int, default=0)
  parser.add_argument("--max-inflight", type=int, default=1)
  parser.add_argument("--tiled-mlp-token-chunk", type=int, default=128)
  parser.add_argument("--ce-token-chunk", type=int, default=128)
  parser.add_argument("--ce-vocab-chunk", type=int, default=8192)
  parser.add_argument("--activation-prevent-cse", action="store_true")
  parser.add_argument("--outdir", default="/tmp/tunix-accel-gemma4-base")
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print the delegated runner command without executing it.",
  )
  return parser.parse_args()


def build_env(args: argparse.Namespace) -> dict[str, str]:
  env = os.environ.copy()
  env.setdefault("PYTHONPATH", str(REPO_ROOT))
  env["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = ""
  env["TUNIX_ACCEL_CE_TOKEN_CHUNK"] = str(args.ce_token_chunk)
  env["TUNIX_ACCEL_CE_VOCAB_CHUNK"] = str(args.ce_vocab_chunk)
  env["TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK"] = str(args.tiled_mlp_token_chunk)
  env["TUNIX_ACCEL_TILED_MLP_LORA_ALPHA"] = str(args.lora_alpha)
  env["TUNIX_ACCEL_ACTIVATION_PREVENT_CSE"] = (
      "1" if args.activation_prevent_cse else "0"
  )
  env["TUNIX_ACCEL_ACTIVATION_OFFLOAD_SRC"] = "device"
  env["TUNIX_ACCEL_ACTIVATION_OFFLOAD_DST"] = "pinned_host"
  for key, value in VARIANT_ENV[args.variant].items():
    if value:
      env[key] = value
    else:
      env.pop(key, None)
  return env


def build_command(args: argparse.Namespace) -> list[str]:
  model_id = args.model_id or MODEL_PRESETS[args.model_size]
  input_variant = "packed" if args.variant == "packed" else "unpacked"
  run_outdir = str(Path(args.outdir) / args.model_size / args.variant)
  return [
      sys.executable,
      str(TRAINING_RUNNER),
      "--model-id",
      model_id,
      "--model-source",
      "huggingface",
      "--model-path",
      "",
      "--model-download-path",
      str(Path(args.outdir) / "hf-cache" / args.model_size),
      "--tokenizer-source",
      "huggingface",
      "--allow-download",
      "--num-examples",
      str(args.num_examples),
      "--variants",
      input_variant,
      "--batch-size",
      str(args.batch_size),
      "--max-length",
      str(args.max_length),
      "--max-steps",
      str(args.max_steps),
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
      "--skip-quality-eval",
      "--allow-autopatch",
      "--outdir",
      run_outdir,
  ]


def main() -> None:
  args = parse_args()
  command = build_command(args)
  env = build_env(args)
  print(" ".join(command))
  if args.dry_run:
    for key in sorted(k for k in env if k.startswith("TUNIX_ACCEL_")):
      print(f"{key}={env[key]}")
    return
  subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
  main()
