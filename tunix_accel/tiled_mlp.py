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


def _linear(x: Array, kernel: Array) -> Array:
  return jax.lax.dot_general(
      x,
      kernel,
      (((x.ndim - 1,), (0,)), ((), ())),
      precision=None,
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
  gate = _linear(hidden, gate_kernel)
  up = _linear(hidden, up_kernel)
  intermediate = _activation(gate, activation) * up
  return _linear(intermediate, down_kernel)


def _linear_lora(
    x: Array,
    kernel: Array,
    lora_a: Array,
    lora_b: Array,
    lora_scale: float,
) -> Array:
  base = _linear(x, kernel)
  delta = _linear(_linear(x, lora_a), lora_b)
  return base + delta * jnp.asarray(lora_scale, dtype=base.dtype)


def dense_lora_gated_mlp(
    hidden: Array,
    gate_kernel: Array,
    gate_lora_a: Array,
    gate_lora_b: Array,
    up_kernel: Array,
    up_lora_a: Array,
    up_lora_b: Array,
    down_kernel: Array,
    down_lora_a: Array,
    down_lora_b: Array,
    *,
    lora_scale: float,
    activation: Activation = "silu",
) -> Array:
  """Dense reference implementation for a LoRA-wrapped gated MLP block."""
  activation = _validate_activation(activation)
  _check_shapes(hidden, gate_kernel, up_kernel, down_kernel)
  gate = _linear_lora(hidden, gate_kernel, gate_lora_a, gate_lora_b, lora_scale)
  up = _linear_lora(hidden, up_kernel, up_lora_a, up_lora_b, lora_scale)
  intermediate = _activation(gate, activation) * up
  return _linear_lora(
      intermediate,
      down_kernel,
      down_lora_a,
      down_lora_b,
      lora_scale,
  )


def _token_axis(hidden: Array) -> int:
  if hidden.ndim < 2:
    raise ValueError(f"hidden must be rank >= 2, got shape {hidden.shape}.")
  return hidden.ndim - 2


def _token_axis_length(x: Array) -> int:
  return int(x.shape[_token_axis(x)])


def _pad_token_axis(x: Array, token_chunk: int) -> tuple[Array, int]:
  axis = _token_axis(x)
  n_tokens = int(x.shape[axis])
  chunks = (n_tokens + token_chunk - 1) // token_chunk
  padded_n = chunks * token_chunk
  pad_n = padded_n - n_tokens
  if not pad_n:
    return x, n_tokens
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, pad_n)
  return jnp.pad(x, pad_width), n_tokens


def _slice_token_tile(x: Array, start: Array, token_chunk: int) -> Array:
  axis = _token_axis(x)
  starts = [0] * x.ndim
  starts[axis] = start
  sizes = list(x.shape)
  sizes[axis] = token_chunk
  return jax.lax.dynamic_slice(x, tuple(starts), tuple(sizes))


def _update_token_tile(acc: Array, tile: Array, start: Array) -> Array:
  axis = _token_axis(acc)
  starts = [0] * acc.ndim
  starts[axis] = start
  return jax.lax.dynamic_update_slice(acc, tile, tuple(starts))


def _trim_token_axis(x: Array, n_tokens: int) -> Array:
  axis = _token_axis(x)
  index = [slice(None)] * x.ndim
  index[axis] = slice(0, n_tokens)
  return x[tuple(index)]


def _kernel_grad(x: Array, y: Array) -> Array:
  return jnp.einsum("...d,...h->dh", x, y)


def _flatten_and_pad(hidden: Array, token_chunk: int) -> tuple[Array, int, tuple[int, ...]]:
  """Deprecated compatibility wrapper for older callers."""
  original_shape = tuple(hidden.shape)
  padded, n_tokens = _pad_token_axis(hidden, token_chunk)
  return padded, n_tokens, original_shape


def _pad_flat_tokens(x: Array, token_chunk: int) -> tuple[Array, int]:
  return _pad_token_axis(x, token_chunk)


def _tile_forward(
    x: Array,
    gate_kernel: Array,
    up_kernel: Array,
    down_kernel: Array,
    activation: Activation,
) -> tuple[Array, Array, Array, Array]:
  gate = _linear(x, gate_kernel)
  up = _linear(x, up_kernel)
  activated = _activation(gate, activation)
  intermediate = activated * up
  out = _linear(intermediate, down_kernel)
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
    return dense_gated_mlp(
        hidden,
        gate_kernel,
        up_kernel,
        down_kernel,
        activation=activation,
    )

  def fwd(
      hidden: Array,
      gate_kernel: Array,
      up_kernel: Array,
      down_kernel: Array,
  ):
    out = dense_gated_mlp(
        hidden,
        gate_kernel,
        up_kernel,
        down_kernel,
        activation=activation,
    )
    hidden_padded, n_tokens, original_shape = _flatten_and_pad(hidden, token_chunk)
    return out, (
        hidden_padded,
        gate_kernel,
        up_kernel,
        down_kernel,
        n_tokens,
        original_shape,
    )

  def bwd(residual, grad_out: Array):
    (
        hidden_padded,
        gate_kernel,
        up_kernel,
        down_kernel,
        n_tokens,
        original_shape,
    ) = residual
    grad_out_padded, _ = _pad_flat_tokens(grad_out, token_chunk)
    token_chunks = _token_axis_length(hidden_padded) // token_chunk

    grad_hidden = jnp.zeros_like(hidden_padded)
    grad_gate = jnp.zeros_like(gate_kernel)
    grad_up = jnp.zeros_like(up_kernel)
    grad_down = jnp.zeros_like(down_kernel)

    def body(i: Array, state: tuple[Array, Array, Array, Array]):
      gh_acc, gg_acc, gu_acc, gd_acc = state
      start = i * token_chunk
      x = _slice_token_tile(hidden_padded, start, token_chunk)
      go = _slice_token_tile(grad_out_padded, start, token_chunk)

      def tile_fn(tile_x, tile_gate, tile_up, tile_down):
        return dense_gated_mlp(
            tile_x,
            tile_gate,
            tile_up,
            tile_down,
            activation=activation,
        )

      _, pullback = jax.vjp(
          tile_fn,
          x,
          gate_kernel,
          up_kernel,
          down_kernel,
      )
      grad_x, grad_gate_tile, grad_up_tile, grad_down_tile = pullback(go)

      gh_acc = _update_token_tile(gh_acc, grad_x.astype(gh_acc.dtype), start)
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
    grad_hidden = _trim_token_axis(grad_hidden, n_tokens).reshape(original_shape)
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


@lru_cache(maxsize=32)
def make_tiled_lora_gated_mlp(
    token_chunk: int = 128,
    *,
    activation: Activation = "silu",
    lora_scale: float = 1.0,
) -> Callable[
    [Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    Array,
]:
  """Returns a tiled custom-VJP gated MLP for LoRA-wrapped projections."""
  token_chunk = _validate_token_chunk(token_chunk)
  activation = _validate_activation(activation)
  lora_scale = float(lora_scale)

  @jax.custom_vjp
  def tiled_lora_gated_mlp(
      hidden: Array,
      gate_kernel: Array,
      gate_lora_a: Array,
      gate_lora_b: Array,
      up_kernel: Array,
      up_lora_a: Array,
      up_lora_b: Array,
      down_kernel: Array,
      down_lora_a: Array,
      down_lora_b: Array,
  ) -> Array:
    return dense_lora_gated_mlp(
        hidden,
        gate_kernel,
        gate_lora_a,
        gate_lora_b,
        up_kernel,
        up_lora_a,
        up_lora_b,
        down_kernel,
        down_lora_a,
        down_lora_b,
        lora_scale=lora_scale,
        activation=activation,
    )

  def fwd(
      hidden: Array,
      gate_kernel: Array,
      gate_lora_a: Array,
      gate_lora_b: Array,
      up_kernel: Array,
      up_lora_a: Array,
      up_lora_b: Array,
      down_kernel: Array,
      down_lora_a: Array,
      down_lora_b: Array,
  ):
    out = dense_lora_gated_mlp(
        hidden,
        gate_kernel,
        gate_lora_a,
        gate_lora_b,
        up_kernel,
        up_lora_a,
        up_lora_b,
        down_kernel,
        down_lora_a,
        down_lora_b,
        lora_scale=lora_scale,
        activation=activation,
    )
    hidden_padded, n_tokens, original_shape = _flatten_and_pad(hidden, token_chunk)
    return out, (
        hidden_padded,
        gate_kernel,
        gate_lora_a,
        gate_lora_b,
        up_kernel,
        up_lora_a,
        up_lora_b,
        down_kernel,
        down_lora_a,
        down_lora_b,
        n_tokens,
        original_shape,
    )

  def bwd(residual, grad_out: Array):
    (
        hidden_padded,
        gate_kernel,
        gate_lora_a,
        gate_lora_b,
        up_kernel,
        up_lora_a,
        up_lora_b,
        down_kernel,
        down_lora_a,
        down_lora_b,
        n_tokens,
        original_shape,
    ) = residual
    grad_out_padded, _ = _pad_flat_tokens(grad_out, token_chunk)
    token_chunks = _token_axis_length(hidden_padded) // token_chunk

    init = (
        jnp.zeros_like(hidden_padded),
        jnp.zeros_like(gate_kernel),
        jnp.zeros_like(gate_lora_a),
        jnp.zeros_like(gate_lora_b),
        jnp.zeros_like(up_kernel),
        jnp.zeros_like(up_lora_a),
        jnp.zeros_like(up_lora_b),
        jnp.zeros_like(down_kernel),
        jnp.zeros_like(down_lora_a),
        jnp.zeros_like(down_lora_b),
    )

    def body(i: Array, state):
      start = i * token_chunk
      x = _slice_token_tile(hidden_padded, start, token_chunk)
      go = _slice_token_tile(grad_out_padded, start, token_chunk)

      def tile_fn(
          tile_x,
          tile_gate,
          tile_gate_a,
          tile_gate_b,
          tile_up,
          tile_up_a,
          tile_up_b,
          tile_down,
          tile_down_a,
          tile_down_b,
      ):
        return dense_lora_gated_mlp(
            tile_x,
            tile_gate,
            tile_gate_a,
            tile_gate_b,
            tile_up,
            tile_up_a,
            tile_up_b,
            tile_down,
            tile_down_a,
            tile_down_b,
            lora_scale=lora_scale,
            activation=activation,
        )

      _, pullback = jax.vjp(
          tile_fn,
          x,
          gate_kernel,
          gate_lora_a,
          gate_lora_b,
          up_kernel,
          up_lora_a,
          up_lora_b,
          down_kernel,
          down_lora_a,
          down_lora_b,
      )
      grads = pullback(go)
      gh = _update_token_tile(state[0], grads[0].astype(state[0].dtype), start)
      accum = [gh]
      for acc, grad in zip(state[1:], grads[1:], strict=True):
        accum.append(acc + grad.astype(acc.dtype))
      return tuple(accum)

    grads = jax.lax.fori_loop(0, token_chunks, body, init)
    grad_hidden = _trim_token_axis(grads[0], n_tokens).reshape(original_shape)
    return (grad_hidden, *grads[1:])

  tiled_lora_gated_mlp.defvjp(fwd, bwd)
  return tiled_lora_gated_mlp


def tiled_lora_gated_mlp(
    hidden: Array,
    gate_kernel: Array,
    gate_lora_a: Array,
    gate_lora_b: Array,
    up_kernel: Array,
    up_lora_a: Array,
    up_lora_b: Array,
    down_kernel: Array,
    down_lora_a: Array,
    down_lora_b: Array,
    *,
    token_chunk: int = 128,
    activation: Activation = "silu",
    lora_scale: float = 1.0,
) -> Array:
  """Computes a LoRA-wrapped gated MLP by streaming the token dimension."""
  return make_tiled_lora_gated_mlp(
      token_chunk=token_chunk,
      activation=activation,
      lora_scale=float(lora_scale),
  )(
      hidden,
      gate_kernel,
      gate_lora_a,
      gate_lora_b,
      up_kernel,
      up_lora_a,
      up_lora_b,
      down_kernel,
      down_lora_a,
      down_lora_b,
  )


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
  batch_size = int(batch_size)
  sequence_length = int(sequence_length)
  if batch_size * sequence_length <= 0:
    raise ValueError("batch_size * sequence_length must be positive.")
  if intermediate_dim <= 0:
    raise ValueError("intermediate_dim must be positive.")
  if dtype_bytes <= 0:
    raise ValueError("dtype_bytes must be positive.")
  dense = batch_size * sequence_length * int(intermediate_dim) * int(dtype_bytes)
  if token_chunk is None:
    tile_sequence = sequence_length
  else:
    tile_sequence = min(_validate_token_chunk(token_chunk), sequence_length)
  tiled = batch_size * tile_sequence * int(intermediate_dim) * int(dtype_bytes)
  return {
      "dense_intermediate_bytes": dense,
      "tiled_intermediate_bytes": tiled,
      "estimated_reduction_bytes": dense - tiled,
  }
