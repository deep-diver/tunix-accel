#!/usr/bin/env python3
"""Gemma3 training benchmark for the tiled-MLP drop-in patch.

This runner intentionally does not call `gemma3_tiled_mlp.install()`. The tiled
variant relies on the package's sitecustomize/autopatch import hook, matching
the intended drop-in usage after `pip install -e .`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKING_RUNNER = REPO_ROOT / "02-PACKING" / "run_gemma_training_benchmark.py"

GEMMA3_TOKENIZER_GCS = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"
MODEL_PRESETS = {
    "270m": (
        "google/gemma-3-270m-it",
        "gs://gemma-data/checkpoints/gemma3-270m-it",
    ),
    "1b": (
        "google/gemma-3-1b-it",
        "gs://gemma-data/checkpoints/gemma3-1b-it",
    ),
    "4b": (
        "google/gemma-3-4b-it",
        "gs://gemma-data/checkpoints/gemma3-4b-it",
    ),
}


def _disabled(value: str | None) -> bool:
  return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def configure_autopatch_env(args: argparse.Namespace) -> None:
  """Configures env flags before any Tunix/Gemma import happens."""
  if _disabled(os.environ.get("TUNIX_ACCEL_DISABLE_AUTOPATCH")):
    raise RuntimeError(
        "TUNIX_ACCEL_DISABLE_AUTOPATCH disables the Gemma3 import hook. "
        "Unset it for this benchmark; use --mlp-variant default for the "
        "baseline."
    )

  os.environ.setdefault("TUNIX_ACCEL_DISABLE_AUTOPATCH", "")
  os.environ.setdefault("TUNIX_ACCEL_DISABLE_CE", "1")
  os.environ.setdefault(
      "TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK",
      str(args.tiled_mlp_token_chunk),
  )
  if args.mlp_variant == "default":
    os.environ["TUNIX_ACCEL_DISABLE_TILED_MLP"] = "1"
  else:
    os.environ.pop("TUNIX_ACCEL_DISABLE_TILED_MLP", None)


def load_packing_runner():
  spec = importlib.util.spec_from_file_location(
      "tunix_accel_packing_training_runner",
      PACKING_RUNNER,
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {PACKING_RUNNER}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2) + "\n")


def enrich_summary(
    summary: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
  from tunix_accel import gemma3_tiled_mlp  # pylint: disable=import-outside-toplevel

  summary = dict(summary)
  summary.update({
      "variant": args.mlp_variant,
      "mlp_variant": args.mlp_variant,
      "input_variant": args.input_variant,
      "default_ce": True,
      "ce_disabled": _disabled(os.environ.get("TUNIX_ACCEL_DISABLE_CE")),
      "drop_in_path": "sitecustomize/autopatch",
      "explicit_install_called": False,
      "tiled_mlp_expected": args.mlp_variant == "tiled",
      "tiled_mlp_installed": gemma3_tiled_mlp.is_installed(),
      "tiled_mlp_token_chunk": int(
          os.environ.get("TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK", "128")
      ),
      "training_mode": "full-parameter"
      if args.lora_rank == 0
      else "lora-fallback",
      "env": {
          "TUNIX_ACCEL_DISABLE_AUTOPATCH": os.environ.get(
              "TUNIX_ACCEL_DISABLE_AUTOPATCH",
              "",
          ),
          "TUNIX_ACCEL_DISABLE_CE": os.environ.get("TUNIX_ACCEL_DISABLE_CE", ""),
          "TUNIX_ACCEL_DISABLE_TILED_MLP": os.environ.get(
              "TUNIX_ACCEL_DISABLE_TILED_MLP",
              "",
          ),
          "TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK": os.environ.get(
              "TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK",
              "",
          ),
      },
  })
  write_json(run_dir / "summary.json", summary)
  return summary


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="270m")
  parser.add_argument("--model-id", default=None)
  parser.add_argument("--model-source", default="gcs")
  parser.add_argument("--model-path", default=None)
  parser.add_argument("--model-download-path", default=None)
  parser.add_argument("--intermediate-ckpt-dir", default=None)
  parser.add_argument(
      "--tokenizer-source",
      choices=["sentencepiece", "huggingface"],
      default="sentencepiece",
  )
  parser.add_argument("--tokenizer-path", default=GEMMA3_TOKENIZER_GCS)
  parser.add_argument("--allow-download", action="store_true")
  parser.add_argument("--num-examples", type=int, default=5000)
  parser.add_argument("--input-variant", choices=["unpacked", "packed"], default="unpacked")
  parser.add_argument("--mlp-variant", choices=["default", "tiled"], required=True)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--max-length", type=int, default=512)
  parser.add_argument("--max-steps", type=int, default=50)
  parser.add_argument("--learning-rate", type=float, default=2e-4)
  parser.add_argument("--adam-b1", type=float, default=0.9)
  parser.add_argument("--adam-b2", type=float, default=0.999)
  parser.add_argument("--weight-decay", type=float, default=0.0)
  parser.add_argument("--lora-rank", type=int, default=0)
  parser.add_argument("--lora-alpha", type=float, default=32.0)
  parser.add_argument(
      "--lora-module-path",
      default=(
          ".*(q_einsum|kv_einsum|qkv_einsum|attn_vec_einsum|"
          "gate_proj|up_proj|down_proj).*"
      ),
  )
  parser.add_argument(
      "--allow-lora-fallback",
      action="store_true",
      help="Allow LoRA runs, where tiled MLP intentionally falls back to Tunix.",
  )
  parser.add_argument(
      "--packing-strategy",
      choices=[
          "first_fit",
          "best_fit",
          "first_fit_decreasing",
          "best_fit_decreasing",
      ],
      default="best_fit_decreasing",
  )
  parser.add_argument("--mesh-fsdp", type=int, default=0)
  parser.add_argument("--mesh-tp", type=int, default=0)
  parser.add_argument("--max-inflight", type=int, default=1)
  parser.add_argument("--skip-jit", action="store_true")
  parser.add_argument("--skip-quality-eval", action="store_true")
  parser.add_argument("--eval-examples", type=int, default=512)
  parser.add_argument("--eval-batches", type=int, default=32)
  parser.add_argument("--generation-examples", type=int, default=128)
  parser.add_argument("--generation-batch-size", type=int, default=8)
  parser.add_argument("--max-generation-steps", type=int, default=128)
  parser.add_argument("--save-checkpoints", action="store_true")
  parser.add_argument("--log-every", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--tiled-mlp-token-chunk", type=int, default=128)
  parser.add_argument("--outdir", default="03-TILED-MLP/results/gemma-training")
  args = parser.parse_args()

  preset_model_id, preset_model_path = MODEL_PRESETS[args.model_size]
  args.model_id = args.model_id or preset_model_id
  args.model_path = args.model_path or preset_model_path
  if args.lora_rank > 0 and not args.allow_lora_fallback:
    raise ValueError(
        "The tiled MLP patch currently targets non-LoRA projection kernels. "
        "Use --lora-rank 0 for an effective comparison, or pass "
        "--allow-lora-fallback to record an intentional fallback run."
    )
  return args


def main() -> None:
  args = parse_args()
  configure_autopatch_env(args)
  runner = load_packing_runner()

  outdir = Path(args.outdir).expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  tokenizer_bundle = runner.load_tokenizer(args)
  dataset = runner.load_opus100_records(args, tokenizer_bundle)
  eval_examples = None
  if not args.skip_quality_eval:
    eval_examples = runner.load_raw_translation_examples(
        split="validation",
        num_examples=max(args.eval_examples, args.generation_examples * 4),
    )

  prepared = runner.prepare_variant(args.input_variant, dataset, args)
  prepared = replace(prepared, name=args.mlp_variant)
  summary, history = runner.run_variant(
      prepared,
      dataset,
      tokenizer_bundle,
      eval_examples,
      args,
      outdir,
  )
  run_dir = outdir / args.mlp_variant
  summary = enrich_summary(summary, args=args, run_dir=run_dir)
  runner.write_json(outdir / "summary.json", [summary])
  runner.write_csv(outdir / "history.csv", history)

  print(f"outdir={outdir}")
  print(f"run_dir={run_dir}")
  print(f"summary={run_dir / 'summary.json'}")
  print(f"history={run_dir / 'history.csv'}")
  print(f"tiled_mlp_installed={summary['tiled_mlp_installed']}")


if __name__ == "__main__":
  main()
