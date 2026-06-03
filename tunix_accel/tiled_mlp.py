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
import math
import os
from typing import Literal

import jax
import jax.numpy as jnp


Array = jax.Array
Activation = Literal["silu", "gelu", "gelu_approx", "relu"]
IntermediateSharding = tuple[str | None, ...] | None
MatmulBackend = Literal["xla", "pallas"]


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


def _validate_matmul_backend(matmul_backend: str) -> MatmulBackend:
  if matmul_backend not in {"xla", "pallas"}:
    raise ValueError(
        "matmul_backend must be one of 'xla' or 'pallas', "
        f"got {matmul_backend!r}."
    )
  return matmul_backend  # type: ignore[return-value]


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


def _has_tpu_device() -> bool:
  return any(device.platform == "tpu" for device in jax.devices())


def _pallas_block_k(k_dim: int) -> int | None:
  for block_k in (256, 128):
    if k_dim % block_k == 0:
      return block_k
  if k_dim % 8 == 0:
    return k_dim
  return None


def _linear_pallas_tpu_local(
    x: Array,
    kernel: Array,
    *,
    block_m: int = 128,
    block_n: int = 256,
) -> Array:
  """Runs a rank-2 Pallas TPU matmul when the shape is compatible."""
  if x.ndim < 2 or kernel.ndim != 2:
    return _linear(x, kernel)
  if not _has_tpu_device():
    return _linear(x, kernel)

  leading_shape = tuple(x.shape[:-1])
  m_dim = math.prod(leading_shape)
  k_dim = int(x.shape[-1])
  n_dim = int(kernel.shape[-1])
  block_k = _pallas_block_k(k_dim)
  if (
      block_k is None
      or m_dim % block_m != 0
      or n_dim % block_n != 0
      or kernel.shape[0] != k_dim
  ):
    return _linear(x, kernel)

  from jax.experimental.pallas.ops.tpu import matmul as pallas_tpu_matmul  # pylint: disable=import-outside-toplevel

  flat_x = jnp.reshape(x, (m_dim, k_dim))
  out = pallas_tpu_matmul.matmul(
      flat_x,
      kernel,
      block_shape=(block_m, block_n),
      block_k=block_k,
      out_dtype=x.dtype,
  )
  return jnp.reshape(out, (*leading_shape, n_dim))


def _array_sharding_parts(x: Array):
  sharding = getattr(x, "sharding", None)
  mesh = getattr(sharding, "mesh", None)
  spec = getattr(sharding, "spec", None)
  return mesh, tuple(spec) if spec is not None else None


def _pallas_debug(message: str) -> None:
  if os.environ.get("TUNIX_ACCEL_TILED_MLP_DEBUG", "").lower() in {
      "1",
      "true",
      "yes",
      "on",
  }:
    print(f"[tunix_accel.tiled_mlp] {message}", flush=True)


def _linear_pallas_tpu_shard_map(
    x: Array,
    kernel: Array,
    *,
    block_m: int,
    block_n: int,
) -> Array | None:
  """Runs a Pallas matmul under Gemma3's FSDP mesh via shard_map.

  Mosaic/Pallas kernels are not automatically partitioned by GSPMD. For the
  Gemma3 experiments retained in this repo, the mesh is `fsdp=8,tp=1`; model
  weights are sharded over `fsdp`, while the local Pallas kernel needs the full
  contraction/output dimension. We gather only that one weight axis inside
  shard_map, run the local Mosaic kernel, and return the usual activation
  sharding.
  """
  if x.ndim != 3 or kernel.ndim != 2:
    _pallas_debug(
        f"skip shard_map: unsupported ranks x={x.ndim} kernel={kernel.ndim}"
    )
    return None
  x_mesh, x_spec = _array_sharding_parts(x)
  kernel_mesh, kernel_spec = _array_sharding_parts(kernel)
  mesh = x_mesh or kernel_mesh or jax.sharding.get_abstract_mesh()
  if mesh is None:
    _pallas_debug("skip shard_map: mesh is None")
    return None
  mesh_shape = getattr(mesh, "shape", {})
  if "fsdp" not in mesh_shape or "tp" not in mesh_shape:
    _pallas_debug(
        "skip shard_map: mesh lacks fsdp/tp "
        f"shape={mesh_shape} x_spec={x_spec} kernel_spec={kernel_spec}"
    )
    return None
  if int(mesh_shape["tp"]) != 1:
    _pallas_debug(f"skip shard_map: tp={mesh_shape['tp']} is not supported")
    return None
  if kernel_spec is None:
    if int(kernel.shape[0]) < int(kernel.shape[1]):
      kernel_spec = ("fsdp", "tp")
    elif int(kernel.shape[0]) > int(kernel.shape[1]):
      kernel_spec = ("tp", "fsdp")
    else:
      _pallas_debug(f"skip shard_map: square kernel shape={kernel.shape}")
      return None
  if x_spec is None:
    if kernel_spec == ("fsdp", "tp"):
      x_spec = (None, None, "fsdp")
    else:
      x_spec = (None, None, "tp")
  gather_axis = None
  reduce_axis = None
  if kernel_spec == ("fsdp", "tp"):
    if x_spec[-1] == "fsdp":
      reduce_axis = "fsdp"
      out_spec = (*x_spec[:-1], "tp")
    elif x_spec[-1] in {None, "tp"}:
      gather_axis = 0
      out_spec = x_spec
    else:
      _pallas_debug(
          f"skip shard_map: unsupported x/kernel specs {x_spec}/{kernel_spec}"
      )
      return None
  elif kernel_spec == ("tp", "fsdp"):
    if x_spec[-1] in {None, "tp"}:
      out_spec = (*x_spec[:-1], "fsdp")
    elif x_spec[-1] == "fsdp":
      gather_axis = 1
      out_spec = x_spec
    else:
      _pallas_debug(
          f"skip shard_map: unsupported x/kernel specs {x_spec}/{kernel_spec}"
      )
      return None
  else:
    _pallas_debug(f"skip shard_map: unsupported kernel_spec={kernel_spec}")
    return None

  from jax.sharding import PartitionSpec as P  # pylint: disable=import-outside-toplevel

  x_pspec = P(*x_spec)
  kernel_pspec = P(*kernel_spec)
  out_pspec = P(*out_spec)

  def local_matmul(x_local: Array, kernel_local: Array) -> Array:
    if gather_axis is not None:
      kernel_local = jax.lax.all_gather(
          kernel_local,
          "fsdp",
          axis=gather_axis,
          tiled=True,
      )
    out = _linear_pallas_tpu_local(
        x_local,
        kernel_local,
        block_m=block_m,
        block_n=block_n,
    )
    if reduce_axis is not None:
      out = jax.lax.psum(out, reduce_axis)
    return out

  return jax.shard_map(
      local_matmul,
      mesh=mesh,
      in_specs=(x_pspec, kernel_pspec),
      out_specs=out_pspec,
      axis_names={"fsdp", "tp"},
      check_vma=False,
  )(x, kernel)


def _linear_pallas_tpu(
    x: Array,
    kernel: Array,
    *,
    block_m: int = 128,
    block_n: int = 256,
) -> Array:
  out = _linear_pallas_tpu_shard_map(
      x,
      kernel,
      block_m=block_m,
      block_n=block_n,
  )
  if out is not None:
    return out
  return _linear_pallas_tpu_local(
      x,
      kernel,
      block_m=block_m,
      block_n=block_n,
  )


def _linear_backend(
    x: Array,
    kernel: Array,
    *,
    matmul_backend: MatmulBackend,
) -> Array:
  if matmul_backend == "pallas":
    return _linear_pallas_tpu(x, kernel)
  return _linear(x, kernel)


def _maybe_shard_intermediate(
    x: Array,
    intermediate_sharding: IntermediateSharding,
) -> Array:
  if intermediate_sharding is None:
    return x
  from tunix.utils import sharding_utils  # pylint: disable=import-outside-toplevel

  return sharding_utils.shard(x, tuple(intermediate_sharding))


def dense_gated_mlp(
    hidden: Array,
    gate_kernel: Array,
    up_kernel: Array,
    down_kernel: Array,
    *,
    activation: Activation = "silu",
    intermediate_sharding: IntermediateSharding = None,
) -> Array:
  """Dense reference implementation for a gated MLP block."""
  activation = _validate_activation(activation)
  _check_shapes(hidden, gate_kernel, up_kernel, down_kernel)
  return _dense_gated_mlp_backend(
      hidden,
      gate_kernel,
      up_kernel,
      down_kernel,
      activation=activation,
      intermediate_sharding=intermediate_sharding,
      matmul_backend="xla",
  )


def _dense_gated_mlp_backend(
    hidden: Array,
    gate_kernel: Array,
    up_kernel: Array,
    down_kernel: Array,
    *,
    activation: Activation,
    intermediate_sharding: IntermediateSharding,
    matmul_backend: MatmulBackend,
) -> Array:
  gate = _linear_backend(hidden, gate_kernel, matmul_backend=matmul_backend)
  up = _linear_backend(hidden, up_kernel, matmul_backend=matmul_backend)
  intermediate = _activation(gate, activation) * up
  intermediate = _maybe_shard_intermediate(intermediate, intermediate_sharding)
  return _linear_backend(intermediate, down_kernel, matmul_backend=matmul_backend)


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


def _linear_lora_backend(
    x: Array,
    kernel: Array,
    lora_a: Array,
    lora_b: Array,
    lora_scale: float,
    *,
    matmul_backend: MatmulBackend,
) -> Array:
  base = _linear_backend(x, kernel, matmul_backend=matmul_backend)
  delta = _linear(_linear(x, lora_a), lora_b)
  return base + delta * jnp.asarray(lora_scale, dtype=base.dtype)


def _linear_lora_backward_explicit(
    x: Array,
    kernel: Array,
    lora_a: Array,
    lora_b: Array,
    grad_out: Array,
    lora_scale: float,
    *,
    matmul_backend: MatmulBackend,
) -> tuple[Array, Array, Array, Array]:
  scale = jnp.asarray(lora_scale, dtype=grad_out.dtype)
  low_rank = _linear(x, lora_a)
  grad_kernel = _kernel_grad_backend(
      x,
      grad_out,
      matmul_backend=matmul_backend,
  )
  grad_x_base = _linear_backend(
      grad_out,
      jnp.swapaxes(kernel, 0, 1),
      matmul_backend=matmul_backend,
  )
  grad_lora_b = _kernel_grad(low_rank, grad_out) * scale
  grad_low_rank = _linear(grad_out, jnp.swapaxes(lora_b, 0, 1)) * scale
  grad_lora_a = _kernel_grad(x, grad_low_rank)
  grad_x_lora = _linear(grad_low_rank, jnp.swapaxes(lora_a, 0, 1))
  return grad_x_base + grad_x_lora, grad_kernel, grad_lora_a, grad_lora_b


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
    intermediate_sharding: IntermediateSharding = None,
) -> Array:
  """Dense reference implementation for a LoRA-wrapped gated MLP block."""
  activation = _validate_activation(activation)
  _check_shapes(hidden, gate_kernel, up_kernel, down_kernel)
  return _dense_lora_gated_mlp_backend(
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
      intermediate_sharding=intermediate_sharding,
      matmul_backend="xla",
  )


def _dense_lora_gated_mlp_backend(
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
    activation: Activation,
    intermediate_sharding: IntermediateSharding,
    matmul_backend: MatmulBackend,
) -> Array:
  gate = _linear_lora_backend(
      hidden,
      gate_kernel,
      gate_lora_a,
      gate_lora_b,
      lora_scale,
      matmul_backend=matmul_backend,
  )
  up = _linear_lora_backend(
      hidden,
      up_kernel,
      up_lora_a,
      up_lora_b,
      lora_scale,
      matmul_backend=matmul_backend,
  )
  intermediate = _activation(gate, activation) * up
  intermediate = _maybe_shard_intermediate(intermediate, intermediate_sharding)
  return _linear_lora_backend(
      intermediate,
      down_kernel,
      down_lora_a,
      down_lora_b,
      lora_scale,
      matmul_backend=matmul_backend,
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


def _kernel_grad_backend(
    x: Array,
    y: Array,
    *,
    matmul_backend: MatmulBackend,
) -> Array:
  del matmul_backend
  return _kernel_grad(x, y)


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
    matmul_backend: MatmulBackend = "xla",
) -> tuple[Array, Array, Array, Array]:
  gate = _linear_backend(x, gate_kernel, matmul_backend=matmul_backend)
  up = _linear_backend(x, up_kernel, matmul_backend=matmul_backend)
  activated = _activation(gate, activation)
  intermediate = activated * up
  out = _linear_backend(intermediate, down_kernel, matmul_backend=matmul_backend)
  return out, gate, up, intermediate


def _tile_backward_explicit(
    x: Array,
    gate_kernel: Array,
    up_kernel: Array,
    down_kernel: Array,
    grad_out: Array,
    activation: Activation,
    matmul_backend: MatmulBackend,
) -> tuple[Array, Array, Array, Array]:
  _, gate, up, intermediate = _tile_forward(
      x,
      gate_kernel,
      up_kernel,
      down_kernel,
      activation,
      matmul_backend,
  )
  grad_intermediate = _linear_backend(
      grad_out,
      jnp.swapaxes(down_kernel, 0, 1),
      matmul_backend=matmul_backend,
  )
  grad_down = _kernel_grad_backend(
      intermediate,
      grad_out,
      matmul_backend=matmul_backend,
  )
  activated = _activation(gate, activation)
  grad_gate_pre = grad_intermediate * up * _activation_grad(gate, activation)
  grad_up_pre = grad_intermediate * activated
  grad_gate = _kernel_grad_backend(x, grad_gate_pre, matmul_backend=matmul_backend)
  grad_up = _kernel_grad_backend(x, grad_up_pre, matmul_backend=matmul_backend)
  grad_x = _linear_backend(
      grad_gate_pre,
      jnp.swapaxes(gate_kernel, 0, 1),
      matmul_backend=matmul_backend,
  ) + _linear_backend(
      grad_up_pre,
      jnp.swapaxes(up_kernel, 0, 1),
      matmul_backend=matmul_backend,
  )
  return grad_x, grad_gate, grad_up, grad_down


def _tile_lora_forward(
    x: Array,
    gate_kernel: Array,
    gate_lora_a: Array,
    gate_lora_b: Array,
    up_kernel: Array,
    up_lora_a: Array,
    up_lora_b: Array,
    down_kernel: Array,
    down_lora_a: Array,
    down_lora_b: Array,
    lora_scale: float,
    activation: Activation,
    matmul_backend: MatmulBackend,
) -> tuple[Array, Array, Array, Array]:
  gate = _linear_lora_backend(
      x,
      gate_kernel,
      gate_lora_a,
      gate_lora_b,
      lora_scale,
      matmul_backend=matmul_backend,
  )
  up = _linear_lora_backend(
      x,
      up_kernel,
      up_lora_a,
      up_lora_b,
      lora_scale,
      matmul_backend=matmul_backend,
  )
  activated = _activation(gate, activation)
  intermediate = activated * up
  out = _linear_lora_backend(
      intermediate,
      down_kernel,
      down_lora_a,
      down_lora_b,
      lora_scale,
      matmul_backend=matmul_backend,
  )
  return out, gate, up, intermediate


def _tile_lora_backward_explicit(
    x: Array,
    gate_kernel: Array,
    gate_lora_a: Array,
    gate_lora_b: Array,
    up_kernel: Array,
    up_lora_a: Array,
    up_lora_b: Array,
    down_kernel: Array,
    down_lora_a: Array,
    down_lora_b: Array,
    grad_out: Array,
    lora_scale: float,
    activation: Activation,
    matmul_backend: MatmulBackend,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
  _, gate, up, intermediate = _tile_lora_forward(
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
      lora_scale,
      activation,
      matmul_backend,
  )
  (
      grad_intermediate,
      grad_down,
      grad_down_a,
      grad_down_b,
  ) = _linear_lora_backward_explicit(
      intermediate,
      down_kernel,
      down_lora_a,
      down_lora_b,
      grad_out,
      lora_scale,
      matmul_backend=matmul_backend,
  )
  activated = _activation(gate, activation)
  grad_gate_pre = grad_intermediate * up * _activation_grad(gate, activation)
  grad_up_pre = grad_intermediate * activated
  (
      grad_x_gate,
      grad_gate,
      grad_gate_a,
      grad_gate_b,
  ) = _linear_lora_backward_explicit(
      x,
      gate_kernel,
      gate_lora_a,
      gate_lora_b,
      grad_gate_pre,
      lora_scale,
      matmul_backend=matmul_backend,
  )
  (
      grad_x_up,
      grad_up,
      grad_up_a,
      grad_up_b,
  ) = _linear_lora_backward_explicit(
      x,
      up_kernel,
      up_lora_a,
      up_lora_b,
      grad_up_pre,
      lora_scale,
      matmul_backend=matmul_backend,
  )
  grad_x = grad_x_gate + grad_x_up
  return (
      grad_x,
      grad_gate,
      grad_gate_a,
      grad_gate_b,
      grad_up,
      grad_up_a,
      grad_up_b,
      grad_down,
      grad_down_a,
      grad_down_b,
  )


@lru_cache(maxsize=32)
def make_tiled_gated_mlp(
    token_chunk: int = 128,
    *,
    activation: Activation = "silu",
    intermediate_sharding: IntermediateSharding = None,
    matmul_backend: MatmulBackend = "xla",
) -> Callable[[Array, Array, Array, Array], Array]:
  """Returns a tiled custom-VJP gated MLP function.

  The returned function accepts `(hidden, gate_kernel, up_kernel, down_kernel)`.
  It is mathematically equivalent to `dense_gated_mlp` for the same activation.
  """
  token_chunk = _validate_token_chunk(token_chunk)
  activation = _validate_activation(activation)
  matmul_backend = _validate_matmul_backend(matmul_backend)

  @jax.custom_vjp
  def tiled_gated_mlp(
      hidden: Array,
      gate_kernel: Array,
      up_kernel: Array,
      down_kernel: Array,
  ) -> Array:
    return _dense_gated_mlp_backend(
        hidden,
        gate_kernel,
        up_kernel,
        down_kernel,
        activation=activation,
        intermediate_sharding=intermediate_sharding,
        matmul_backend=matmul_backend,
    )

  def fwd(
      hidden: Array,
      gate_kernel: Array,
      up_kernel: Array,
      down_kernel: Array,
  ):
    out = _dense_gated_mlp_backend(
        hidden,
        gate_kernel,
        up_kernel,
        down_kernel,
        activation=activation,
        intermediate_sharding=intermediate_sharding,
        matmul_backend=matmul_backend,
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

      if matmul_backend == "pallas":
        grad_x, grad_gate_tile, grad_up_tile, grad_down_tile = (
            _tile_backward_explicit(
                x,
                gate_kernel,
                up_kernel,
                down_kernel,
                go,
                activation,
                matmul_backend,
            )
        )
      else:

        def tile_fn(tile_x, tile_gate, tile_up, tile_down):
          return dense_gated_mlp(
              tile_x,
              tile_gate,
              tile_up,
              tile_down,
              activation=activation,
              intermediate_sharding=intermediate_sharding,
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
    intermediate_sharding: IntermediateSharding = None,
    matmul_backend: MatmulBackend = "xla",
) -> Array:
  """Computes a gated MLP by streaming the token dimension."""
  return make_tiled_gated_mlp(
      token_chunk=token_chunk,
      activation=activation,
      intermediate_sharding=intermediate_sharding,
      matmul_backend=matmul_backend,
  )(hidden, gate_kernel, up_kernel, down_kernel)


@lru_cache(maxsize=32)
def make_tiled_lora_gated_mlp(
    token_chunk: int = 128,
    *,
    activation: Activation = "silu",
    lora_scale: float = 1.0,
    intermediate_sharding: IntermediateSharding = None,
    matmul_backend: MatmulBackend = "xla",
) -> Callable[
    [Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    Array,
]:
  """Returns a tiled custom-VJP gated MLP for LoRA-wrapped projections."""
  token_chunk = _validate_token_chunk(token_chunk)
  activation = _validate_activation(activation)
  matmul_backend = _validate_matmul_backend(matmul_backend)
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
    return _dense_lora_gated_mlp_backend(
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
        intermediate_sharding=intermediate_sharding,
        matmul_backend=matmul_backend,
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
    out = _dense_lora_gated_mlp_backend(
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
        intermediate_sharding=intermediate_sharding,
        matmul_backend=matmul_backend,
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

      if matmul_backend == "pallas":
        grads = _tile_lora_backward_explicit(
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
            go,
            lora_scale,
            activation,
            matmul_backend,
        )
      else:

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
              intermediate_sharding=intermediate_sharding,
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
    intermediate_sharding: IntermediateSharding = None,
    matmul_backend: MatmulBackend = "xla",
) -> Array:
  """Computes a LoRA-wrapped gated MLP by streaming the token dimension."""
  return make_tiled_lora_gated_mlp(
      token_chunk=token_chunk,
      activation=activation,
      lora_scale=float(lora_scale),
      intermediate_sharding=intermediate_sharding,
      matmul_backend=matmul_backend,
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
