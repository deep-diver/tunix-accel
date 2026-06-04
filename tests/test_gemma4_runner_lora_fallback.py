#!/usr/bin/env python3
"""Runner-level Gemma4 LoRA fallback smoke tests."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

os.environ["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "1"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import pytest

nnx = pytest.importorskip("flax.nnx", exc_type=ImportError)
gemma4_model = pytest.importorskip("tunix.models.gemma4.model", exc_type=ImportError)


def _load_runner():
  runner_path = REPO_ROOT / "02-PACKING" / "run_gemma_training_benchmark.py"
  spec = importlib.util.spec_from_file_location(
      "tunix_accel_packing_runner_for_test",
      runner_path,
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {runner_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _tiny_gemma4():
  config = gemma4_model.ModelConfig(
      num_layers=1,
      num_embed=64,
      embed_dim=32,
      hidden_dim=64,
      num_heads=4,
      head_dim=8,
      num_kv_heads=1,
      sliding_window_size=8,
      remat_config=gemma4_model.RematConfig.NONE,
      param_dtype=jnp.float32,
  )
  return gemma4_model.Gemma4(config, rngs=nnx.Rngs(0))


def test_runner_lora_fallback_handles_gemma4_without_get_model_input():
  runner = _load_runner()
  model = _tiny_gemma4()
  assert not hasattr(model, "get_model_input")

  args = argparse.Namespace(
      lora_rank=4,
      lora_alpha=8,
      lora_module_path=(
          ".*(q_einsum|kv_einsum|qkv_einsum|attn_vec_einsum|"
          "gate_proj|up_proj|down_proj).*"
      ),
      max_length=8,
      seed=1,
  )
  model = runner.apply_lora_if_requested(model, None, args)

  assert hasattr(model.layers[0].mlp.gate_proj, "kernel_lora_a")
  assert hasattr(model.layers[0].mlp.up_proj, "kernel_lora_a")
  assert hasattr(model.layers[0].mlp.down_proj, "kernel_lora_a")
