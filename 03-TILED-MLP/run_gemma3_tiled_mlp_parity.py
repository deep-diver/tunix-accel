#!/usr/bin/env python3
"""Numerical parity check for Gemma3 Tiled MLP on the same model instance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


os.environ.setdefault("TUNIX_ACCEL_DISABLE_AUTOPATCH", "1")
os.environ.setdefault("TUNIX_ACCEL_DISABLE_CE", "1")

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


def tree_global_norm(tree) -> float:
  import jax
  import jax.numpy as jnp

  leaves = jax.tree.leaves(tree)
  if not leaves:
    return math.nan
  total = sum(jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves)
  return float(jnp.sqrt(total))


def tree_diff_stats(actual, expected) -> dict[str, float | int]:
  import jax
  import jax.numpy as jnp

  actual_leaves = jax.tree.leaves(actual)
  expected_leaves = jax.tree.leaves(expected)
  if len(actual_leaves) != len(expected_leaves):
    return {
        "leaf_count_actual": len(actual_leaves),
        "leaf_count_expected": len(expected_leaves),
        "max_abs_diff": math.inf,
        "max_rel_diff": math.inf,
        "rms_abs_diff": math.inf,
    }
  max_abs = 0.0
  max_rel = 0.0
  sum_sq = 0.0
  count = 0
  for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
    actual_f32 = actual_leaf.astype(jnp.float32)
    expected_f32 = expected_leaf.astype(jnp.float32)
    diff = jnp.abs(actual_f32 - expected_f32)
    denom = jnp.maximum(jnp.abs(expected_f32), jnp.asarray(1e-12, dtype=jnp.float32))
    max_abs = max(max_abs, float(jnp.max(diff)))
    max_rel = max(max_rel, float(jnp.max(diff / denom)))
    sum_sq += float(jnp.sum(jnp.square(diff)))
    count += int(diff.size)
  return {
      "leaf_count": len(actual_leaves),
      "element_count": count,
      "max_abs_diff": max_abs,
      "max_rel_diff": max_rel,
      "rms_abs_diff": math.sqrt(sum_sq / max(count, 1)),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="4b")
  parser.add_argument("--model-id", default=None)
  parser.add_argument("--model-source", default="gcs")
  parser.add_argument("--model-path", default=None)
  parser.add_argument("--model-download-path", default=None)
  parser.add_argument("--intermediate-ckpt-dir", default=None)
  parser.add_argument("--tokenizer-source", choices=["sentencepiece", "huggingface"], default="sentencepiece")
  parser.add_argument("--tokenizer-path", default=GEMMA3_TOKENIZER_GCS)
  parser.add_argument("--allow-download", action="store_true")
  parser.add_argument("--num-examples", type=int, default=64)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--max-length", type=int, default=512)
  parser.add_argument("--lora-rank", type=int, default=16)
  parser.add_argument("--lora-alpha", type=float, default=32.0)
  parser.add_argument(
      "--lora-module-path",
      default=(
          ".*(q_einsum|kv_einsum|qkv_einsum|attn_vec_einsum|"
          "gate_proj|up_proj|down_proj).*"
      ),
  )
  parser.add_argument("--mesh-fsdp", type=int, default=0)
  parser.add_argument("--mesh-tp", type=int, default=0)
  parser.add_argument("--max-steps", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--tiled-mlp-token-chunk", type=int, default=128)
  parser.add_argument("--outdir", default="03-TILED-MLP/results/gemma3-tiled-mlp-parity")
  args = parser.parse_args()

  preset_model_id, preset_model_path = MODEL_PRESETS[args.model_size]
  args.model_id = args.model_id or preset_model_id
  args.model_path = args.model_path or preset_model_path
  args.input_variant = "unpacked"
  args.packing_strategy = "best_fit_decreasing"
  args.learning_rate = 2e-4
  args.adam_b1 = 0.9
  args.adam_b2 = 0.999
  args.weight_decay = 0.0
  args.max_inflight = 1
  args.skip_jit = False
  args.save_checkpoints = False
  args.log_every = 1
  return args


def main() -> None:
  if os.environ.get("TUNIX_ACCEL_DISABLE_AUTOPATCH", "").lower() not in {
      "1",
      "true",
      "yes",
      "on",
  }:
    raise RuntimeError(
        "Launch with TUNIX_ACCEL_DISABLE_AUTOPATCH=1 so the baseline block stays "
        "unpatched until this script installs the tiled path explicitly."
    )

  import jax
  import jax.numpy as jnp
  from flax import nnx
  import qwix
  from tunix.sft import peft_trainer
  from tunix.sft import utils as sft_utils

  from tunix_accel import gemma3_tiled_mlp

  runner = load_packing_runner()
  outdir = Path(args.outdir).expanduser().resolve()
  run_dir = outdir / args.model_size
  run_dir.mkdir(parents=True, exist_ok=True)

  tokenizer_bundle = runner.load_tokenizer(args)
  dataset = runner.load_opus100_records(args, tokenizer_bundle)
  prepared = runner.prepare_variant("unpacked", dataset, args)
  batch = prepared.batches[0]

  input_tokens = jnp.asarray(batch["input_tokens"], dtype=jnp.int32)
  input_mask = jnp.asarray(batch["input_mask"], dtype=bool)
  valid_mask = jnp.asarray(batch["valid_mask"], dtype=bool)
  positions = sft_utils.build_positions_from_mask(valid_mask)
  attention_mask = sft_utils.make_causal_attn_mask(valid_mask)

  mesh = runner.create_mesh(jax, args)
  with jax.set_mesh(mesh):
    model = runner.create_model(mesh, args)

    def loss_fn(model_arg):
      return peft_trainer._default_loss_fn(  # pylint: disable=protected-access
          model_arg,
          input_tokens,
          input_mask,
          positions,
          attention_mask,
      )

    if args.lora_rank > 0:
      diff_state = nnx.DiffState(0, nnx.LoRAParam)
    else:
      diff_state = 0

    def loss_and_grad():
      return nnx.value_and_grad(loss_fn, argnums=diff_state)(model)

    gemma3_tiled_mlp.uninstall()
    default_loss, default_grads = loss_and_grad()
    default_loss.block_until_ready()

    gemma3_tiled_mlp.install(
        token_chunk=args.tiled_mlp_token_chunk,
        fallback_to_original_on_lora=False,
        lora_alpha=args.lora_alpha,
    )
    tiled_loss, tiled_grads = loss_and_grad()
    tiled_loss.block_until_ready()

    default_loss_f = float(default_loss)
    tiled_loss_f = float(tiled_loss)
    default_grad_norm = tree_global_norm(default_grads)
    tiled_grad_norm = tree_global_norm(tiled_grads)
    grad_diff = tree_diff_stats(tiled_grads, default_grads)

  result = {
      "model_size": args.model_size,
      "model_id": args.model_id,
      "model_path": args.model_path,
      "batch_size": args.batch_size,
      "max_length": args.max_length,
      "num_examples": args.num_examples,
      "lora_rank": args.lora_rank,
      "lora_alpha": args.lora_alpha,
      "lora_module_path": args.lora_module_path if args.lora_rank > 0 else "",
      "tiled_mlp_token_chunk": args.tiled_mlp_token_chunk,
      "mesh_shape": dict(mesh.shape),
      "jax_devices": [str(device) for device in jax.devices()],
      "input_tokens": {
          "valid_tokens": int(valid_mask.sum()),
          "loss_tokens": int(input_mask.sum()),
          "capacity_tokens": int(input_tokens.size),
      },
      "default": {
          "loss": default_loss_f,
          "grad_norm": default_grad_norm,
      },
      "tiled": {
          "loss": tiled_loss_f,
          "grad_norm": tiled_grad_norm,
          "installed": gemma3_tiled_mlp.is_installed(),
      },
      "diff": {
          "loss_abs_diff": abs(tiled_loss_f - default_loss_f),
          "loss_rel_diff": abs(tiled_loss_f - default_loss_f)
          / max(abs(default_loss_f), 1e-12),
          "grad_norm_abs_diff": abs(tiled_grad_norm - default_grad_norm),
          "grad_norm_rel_diff": abs(tiled_grad_norm - default_grad_norm)
          / max(abs(default_grad_norm), 1e-12),
          **grad_diff,
      },
  }
  write_json(run_dir / "parity_summary.json", result)
  print(json.dumps(result, indent=2))


args = parse_args()


if __name__ == "__main__":
  main()
