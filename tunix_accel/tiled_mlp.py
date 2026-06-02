"""Tiled gated-MLP kernels for decoder LM training.

This module targets the SwiGLU/GeGLU-style MLP block used by modern decoder
LMs:

  output = (activation(x @ gate) * (x @ up)) @ down

The tiled implementation streams the token dimension and uses a custom VJP that
recomputes per-tile intermediates during backward. The goal is to avoid keeping
the full `[tokens, intermediate_dim]` MLP activation resident at once while
preserving the exact dense objective.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp


Array = jax.Array
Activation = Literal["silu", "gelu", "gelu_approx", "relu"]


def _validate_token_chunk(token_chunk: int) -> int:
  token_chunk = int(token_chunk)
  if token_chunk <= 0:
    raise ValueError(f"token_chunk must be positive, got {token_chunk}.")
  return token_chunk


def _validate_activation(activation: str) -> Activation:
  if activation not in {"silu", "gelu", "gelu_approx", "relu"}:
    raise ValueError(
        "activation must be one of 'silu', 'gelu', 'gelu_approx', or 'relu', "
        f"got {activation!r}."
    )
  return activation  # type: ignore[return-value]


def _activation(x: Array, activation: Activation) -> Array:
  if activation == "silu":
    return jax.nn.silu(x)
  if activation == "gelu":
    return jax.nn.gelu(x, approximate=False)
  if activation == "gelu_approx":
    return jax.nn.gelu(x, approximate=True)
  return jax.nn.relu(x)


def _activation_grad(x: Array, activation: Activation) -> Array:
  x_f32 = x.astype(jnp.float32)
  if activation == "silu":
    sigmoid = jax.nn.sigmoid(x_f32)
    return sigmoid * (1.0 + x_f32 * (1.0 - sigmoid))
  if activation == "gelu":
    inv_sqrt2 = jnp.array(0.7071067811865476, dtype=jnp.float32)
    inv_sqrt2pi = jnp.array(0.3989422804014327, dtype=jnp.float32)
    cdf = 0.5 * (1.0 + jax.lax.erf(x_f32 * inv_sqrt2))
    pdf = jnp.exp(-0.5 * jnp.square(x_f32)) * inv_sqrt2pi
    return cdf + x_f32 * pdf
  if activation == "gelu_approx":
    sqrt_2_over_pi = jnp.array(0.7978845608028654, dtype=jnp.float32)
    coeff = jnp.array(0.044715, dtype=jnp.float32)
    u = sqrt_2_over_pi * (x_f32 + coeff * jnp.power(x_f32, 3))
    tanh_u = jnp.tanh(u)
    du = sqrt_2_over_pi * (1.0 + 3.0 * coeff * jnp.square(x_f32))
    return 0.5 * (1.0 + tanh_u) + 0.5 * x_f32 * (1.0 - jnp.square(tanh_u)) * du
  return (x_f32 > 0).astype(jnp.float32)


def _check_shapes(
    hidden: Array,
    gate_kernel: Array,
    up_kernel: Array,
    down_kernel: Array,
) -> None:
  hidden_dim = hidden.shape[-1]
  if gate_kernel.ndim != 2 or up_kernel.ndim != 2 or down_kernel.ndim != 2:
    raise ValueError("MLP kernels must all be rank-2 arrays.")
  if gate_kernel.shape[0] != hidden_dim:
    raise ValueError(
        f"gate_kernel input dim {gate_kernel.shape[0]} does not match "
        f"hidden dim {hidden_dim}."
    )
  if up_kernel.shape[0] != hidden_dim:
    raise ValueError(
        f"up_kernel input dim {up_kernel.shape[0]} does not match "
        f"hidden dim {hidden_dim}."
    )
  if gate_kernel.shape[1] != up_kernel.shape[1]:
    raise ValueError(
        "gate_kernel and up_kernel must have the same intermediate dim, got "
        f"{gate_kernel.shape[1]} and {up_kernel.shape[1]}."
    )
  if down_kernel.shape[0] != gate_kernel.shape[1]:
    raise ValueError(
        f"down_kernel input dim {down_kernel.shape[0]} does not match "
        f"intermediate dim {gate_kernel.shape[1]}."
    )


def dense_gated_mlp(
    hidden: Array,
    gate_kernel: Array,
    up_kernel: Array,
    down_kernel: Array,
    *,
    activation: Activation = "silu",
) -> Array:
  """Dense reference implementation for a gated MLP block."""
  activation = _validate_activation(activation)
  _check_shapes(hidden, gate_kernel, up_kernel, down_kernel)
  gate = jnp.matmul(hidden, gate_kernel)
  up = jnp.matmul(hidden, up_kernel)
  intermediate = _activation(gate, activation) * up
  return jnp.matmul(intermediate, down_kernel)


def _flatten_and_pad(hidden: Array, token_chunk: int) -> tuple[Array, int, tuple[int, ...]]:
  original_shape = tuple(hidden.shape)
  hidden_dim = original_shape[-1]
  flat_hidden = hidden.reshape((-1, hidden_dim))
  n_tokens = flat_hidden.shape[0]
  chunks = (n_tokens + token_chunk - 1) // token_chunk
  padded_n = chunks * token_chunk
  pad_n = padded_n - n_tokens
  if pad_n:
    flat_hidden = jnp.pad(flat_hidden, ((0, pad_n), (0, 0)))
  return flat_hidden, n_tokens, original_shape


def _pad_flat_tokens(x: Array, token_chunk: int) -> tuple[Array, int]:
  width = x.shape[-1]
  flat = x.reshape((-1, width))
  n_tokens = flat.shape[0]
  chunks = (n_tokens + token_chunk - 1) // token_chunk
  padded_n = chunks * token_chunk
  pad_n = padded_n - n_tokens
  if pad_n:
    flat = jnp.pad(flat, ((0, pad_n), (0, 0)))
  return flat, n_tokens


def _tile_forward(
    x: Array,
    gate_kernel: Array,
    up_kernel: Array,
    down_kernel: Array,
    activation: Activation,
) -> tuple[Array, Array, Array, Array]:
  gate = jnp.dot(x, gate_kernel)
  up = jnp.dot(x, up_kernel)
  activated = _activation(gate, activation)
  intermediate = activated * up
  out = jnp.dot(intermediate, down_kernel)
  return out, gate, up, intermediate


@lru_cache(maxsize=32)
def make_tiled_gated_mlp(
    token_chunk: int = 128,
    *,
    activation: Activation = "silu",
) -> Callable[[Array, Array, Array, Array], Array]:
  """Returns a tiled custom-VJP gated MLP function.

  The returned function accepts `(hidden, gate_kernel, up_kernel, down_kernel)`.
  It is mathematically equivalent to `dense_gated_mlp` for the same activation.
  """
  token_chunk = _validate_token_chunk(token_chunk)
  activation = _validate_activation(activation)

  @jax.custom_vjp
  def tiled_gated_mlp(
      hidden: Array,
      gate_kernel: Array,
      up_kernel: Array,
      down_kernel: Array,
  ) -> Array:
    _check_shapes(hidden, gate_kernel, up_kernel, down_kernel)
    flat_hidden, n_tokens, original_shape = _flatten_and_pad(hidden, token_chunk)
    token_chunks = flat_hidden.shape[0] // token_chunk
    output_dim = down_kernel.shape[-1]
    out = jnp.zeros((flat_hidden.shape[0], output_dim), dtype=hidden.dtype)

    def body(i: Array, acc: Array) -> Array:
      start = i * token_chunk
      x = jax.lax.dynamic_slice(
          flat_hidden,
          (start, 0),
          (token_chunk, flat_hidden.shape[-1]),
      )
      tile_out, _, _, _ = _tile_forward(
          x,
          gate_kernel,
          up_kernel,
          down_kernel,
          activation,
      )
      return jax.lax.dynamic_update_slice(acc, tile_out.astype(out.dtype), (start, 0))

    out = jax.lax.fori_loop(0, token_chunks, body, out)
    out = out[:n_tokens]
    return out.reshape(original_shape[:-1] + (output_dim,))

  def fwd(
      hidden: Array,
      gate_kernel: Array,
      up_kernel: Array,
      down_kernel: Array,
  ):
    out = tiled_gated_mlp(hidden, gate_kernel, up_kernel, down_kernel)
    flat_hidden, n_tokens, original_shape = _flatten_and_pad(hidden, token_chunk)
    return out, (
        flat_hidden,
        gate_kernel,
        up_kernel,
        down_kernel,
        n_tokens,
        original_shape,
    )

  def bwd(residual, grad_out: Array):
    (
        flat_hidden,
        gate_kernel,
        up_kernel,
        down_kernel,
        n_tokens,
        original_shape,
    ) = residual
    flat_grad_out, _ = _pad_flat_tokens(grad_out, token_chunk)
    token_chunks = flat_hidden.shape[0] // token_chunk
    hidden_dim = flat_hidden.shape[-1]

    grad_hidden = jnp.zeros_like(flat_hidden)
    grad_gate = jnp.zeros_like(gate_kernel)
    grad_up = jnp.zeros_like(up_kernel)
    grad_down = jnp.zeros_like(down_kernel)

    def body(i: Array, state: tuple[Array, Array, Array, Array]):
      gh_acc, gg_acc, gu_acc, gd_acc = state
      start = i * token_chunk
      x = jax.lax.dynamic_slice(
          flat_hidden,
          (start, 0),
          (token_chunk, hidden_dim),
      )
      go = jax.lax.dynamic_slice(
          flat_grad_out,
          (start, 0),
          (token_chunk, down_kernel.shape[-1]),
      )

      _, gate, up, intermediate = _tile_forward(
          x,
          gate_kernel,
          up_kernel,
          down_kernel,
          activation,
      )
      grad_intermediate = jnp.dot(go, down_kernel.T)
      grad_down_tile = jnp.dot(intermediate.T, go)

      grad_up_pre = grad_intermediate * _activation(gate, activation)
      grad_gate_pre = (
          grad_intermediate
          * up
          * _activation_grad(gate, activation).astype(grad_intermediate.dtype)
      )

      grad_x = (
          jnp.dot(grad_gate_pre, gate_kernel.T)
          + jnp.dot(grad_up_pre, up_kernel.T)
      )
      grad_gate_tile = jnp.dot(x.T, grad_gate_pre)
      grad_up_tile = jnp.dot(x.T, grad_up_pre)

      gh_acc = jax.lax.dynamic_update_slice(
          gh_acc,
          grad_x.astype(gh_acc.dtype),
          (start, 0),
      )
      return (
          gh_acc,
          gg_acc + grad_gate_tile.astype(gg_acc.dtype),
          gu_acc + grad_up_tile.astype(gu_acc.dtype),
          gd_acc + grad_down_tile.astype(gd_acc.dtype),
      )

    grad_hidden, grad_gate, grad_up, grad_down = jax.lax.fori_loop(
        0,
        token_chunks,
        body,
        (grad_hidden, grad_gate, grad_up, grad_down),
    )
    grad_hidden = grad_hidden[:n_tokens].reshape(original_shape)
    return grad_hidden, grad_gate, grad_up, grad_down

  tiled_gated_mlp.defvjp(fwd, bwd)
  return tiled_gated_mlp


def tiled_gated_mlp(
    hidden: Array,
    gate_kernel: Array,
    up_kernel: Array,
    down_kernel: Array,
    *,
    token_chunk: int = 128,
    activation: Activation = "silu",
) -> Array:
  """Computes a gated MLP by streaming the token dimension."""
  return make_tiled_gated_mlp(
      token_chunk=token_chunk,
      activation=activation,
  )(hidden, gate_kernel, up_kernel, down_kernel)


def estimate_gated_mlp_intermediate_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    intermediate_dim: int,
    dtype_bytes: int = 2,
    token_chunk: int | None = None,
) -> dict[str, int]:
  """Estimates dense vs tiled gated-MLP intermediate size in bytes.

  This is a simple reporting helper, not a replacement for XLA or TPU profiler
  memory reports.
  """
  tokens = int(batch_size) * int(sequence_length)
  if tokens <= 0:
    raise ValueError("batch_size * sequence_length must be positive.")
  if intermediate_dim <= 0:
    raise ValueError("intermediate_dim must be positive.")
  if dtype_bytes <= 0:
    raise ValueError("dtype_bytes must be positive.")
  dense = tokens * int(intermediate_dim) * int(dtype_bytes)
  if token_chunk is None:
    tile_tokens = tokens
  else:
    tile_tokens = min(_validate_token_chunk(token_chunk), tokens)
  tiled = tile_tokens * int(intermediate_dim) * int(dtype_bytes)
  return {
      "dense_intermediate_bytes": dense,
      "tiled_intermediate_bytes": tiled,
      "estimated_reduction_bytes": dense - tiled,
  }
