#!/usr/bin/env python3
"""Parity checks for tiled gated-MLP kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunix_accel.tiled_mlp import dense_gated_mlp
from tunix_accel.tiled_mlp import dense_lora_gated_mlp
from tunix_accel.tiled_mlp import estimate_gated_mlp_intermediate_bytes
from tunix_accel.tiled_mlp import tiled_gated_mlp
from tunix_accel.tiled_mlp import tiled_lora_gated_mlp


def _inputs():
  key = jax.random.key(0)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  hidden = jax.random.normal(k1, (2, 7, 16), dtype=jnp.float32)
  gate = jax.random.normal(k2, (16, 48), dtype=jnp.float32) / jnp.sqrt(16)
  up = jax.random.normal(k3, (16, 48), dtype=jnp.float32) / jnp.sqrt(16)
  down = jax.random.normal(k4, (48, 16), dtype=jnp.float32) / jnp.sqrt(48)
  return hidden, gate, up, down


def _tree_allclose(actual, expected, *, atol=2e-5, rtol=2e-5):
  for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
    max_diff = jnp.max(jnp.abs(a - e))
    assert jnp.allclose(a, e, atol=atol, rtol=rtol), f"max_diff={float(max_diff)}"


def test_tiled_gated_mlp_forward_matches_dense_silu():
  hidden, gate, up, down = _inputs()
  dense = dense_gated_mlp(hidden, gate, up, down, activation="silu")
  tiled = tiled_gated_mlp(
      hidden,
      gate,
      up,
      down,
      token_chunk=5,
      activation="silu",
  )
  assert jnp.allclose(tiled, dense, atol=2e-5, rtol=2e-5)


def test_tiled_gated_mlp_forward_matches_dense_gelu_variants():
  hidden, gate, up, down = _inputs()
  for activation in ("gelu", "gelu_approx", "relu"):
    dense = dense_gated_mlp(hidden, gate, up, down, activation=activation)
    tiled = tiled_gated_mlp(
        hidden,
        gate,
        up,
        down,
        token_chunk=4,
        activation=activation,
    )
    assert jnp.allclose(tiled, dense, atol=2e-5, rtol=2e-5)


def test_tiled_gated_mlp_gradients_match_dense():
  hidden, gate, up, down = _inputs()

  def dense_loss(h, g, u, d):
    out = dense_gated_mlp(h, g, u, d, activation="silu")
    return jnp.mean(jnp.square(out))

  def tiled_loss(h, g, u, d):
    out = tiled_gated_mlp(
        h,
        g,
        u,
        d,
        token_chunk=3,
        activation="silu",
    )
    return jnp.mean(jnp.square(out))

  assert jnp.allclose(
      tiled_loss(hidden, gate, up, down),
      dense_loss(hidden, gate, up, down),
      atol=2e-5,
      rtol=2e-5,
  )
  _tree_allclose(
      jax.grad(tiled_loss, argnums=(0, 1, 2, 3))(hidden, gate, up, down),
      jax.grad(dense_loss, argnums=(0, 1, 2, 3))(hidden, gate, up, down),
      atol=4e-5,
      rtol=4e-5,
  )


def test_tiled_gated_mlp_jit_gradients_match_dense():
  hidden, gate, up, down = _inputs()

  @jax.jit
  def tiled_loss_and_grads(h, g, u, d):
    def loss_fn(hh, gg, uu, dd):
      out = tiled_gated_mlp(
          hh,
          gg,
          uu,
          dd,
          token_chunk=6,
          activation="gelu_approx",
      )
      return jnp.mean(out)

    return loss_fn(h, g, u, d), jax.grad(loss_fn, argnums=(0, 1, 2, 3))(h, g, u, d)

  def dense_loss(h, g, u, d):
    return jnp.mean(dense_gated_mlp(h, g, u, d, activation="gelu_approx"))

  dense_value = dense_loss(hidden, gate, up, down)
  dense_grads = jax.grad(dense_loss, argnums=(0, 1, 2, 3))(hidden, gate, up, down)
  tiled_value, tiled_grads = tiled_loss_and_grads(hidden, gate, up, down)

  assert jnp.allclose(tiled_value, dense_value, atol=2e-5, rtol=2e-5)
  _tree_allclose(tiled_grads, dense_grads, atol=4e-5, rtol=4e-5)


def test_tiled_lora_gated_mlp_gradients_match_dense_lora():
  hidden, gate, up, down = _inputs()
  key = jax.random.key(7)
  keys = iter(jax.random.split(key, 6))
  rank = 3
  gate_a = jax.random.normal(next(keys), (gate.shape[0], rank), dtype=jnp.float32) * 0.1
  gate_b = jax.random.normal(next(keys), (rank, gate.shape[1]), dtype=jnp.float32) * 0.1
  up_a = jax.random.normal(next(keys), (up.shape[0], rank), dtype=jnp.float32) * 0.1
  up_b = jax.random.normal(next(keys), (rank, up.shape[1]), dtype=jnp.float32) * 0.1
  down_a = jax.random.normal(next(keys), (down.shape[0], rank), dtype=jnp.float32) * 0.1
  down_b = jax.random.normal(next(keys), (rank, down.shape[1]), dtype=jnp.float32) * 0.1

  def dense_loss(*args):
    out = dense_lora_gated_mlp(
        *args,
        lora_scale=2.0,
        activation="gelu_approx",
    )
    return jnp.mean(jnp.square(out))

  def tiled_loss(*args):
    out = tiled_lora_gated_mlp(
        *args,
        token_chunk=4,
        lora_scale=2.0,
        activation="gelu_approx",
    )
    return jnp.mean(jnp.square(out))

  args = (hidden, gate, gate_a, gate_b, up, up_a, up_b, down, down_a, down_b)
  assert jnp.allclose(tiled_loss(*args), dense_loss(*args), atol=2e-5, rtol=2e-5)
  _tree_allclose(
      jax.grad(tiled_loss, argnums=tuple(range(len(args))))(*args),
      jax.grad(dense_loss, argnums=tuple(range(len(args))))(*args),
      atol=4e-5,
      rtol=4e-5,
  )


def test_estimate_gated_mlp_intermediate_bytes():
  estimate = estimate_gated_mlp_intermediate_bytes(
      batch_size=16,
      sequence_length=2048,
      intermediate_dim=8192,
      dtype_bytes=2,
      token_chunk=256,
  )
  assert estimate["dense_intermediate_bytes"] == 16 * 2048 * 8192 * 2
  assert estimate["tiled_intermediate_bytes"] == 16 * 256 * 8192 * 2
  assert estimate["estimated_reduction_bytes"] > 0
