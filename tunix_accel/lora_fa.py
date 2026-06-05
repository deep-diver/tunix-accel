"""LoRA-FA helpers for Tunix/Qwix LoRA training.

LoRA-FA freezes the projection-down LoRA matrix A and trains only the
projection-up matrix B. The B gradient can optionally be corrected in rank space
so the induced low-rank update better approximates the full fine-tuning
gradient. This module keeps A and B as ``nnx.LoRAParam`` values for checkpoint
compatibility, but narrows the trainer's differentiation and optimizer filters
to B parameters only.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


LoRAFAMode = Literal["freeze_a", "corrected_b"]


@dataclass(frozen=True)
class LoRAFAConfig:
  """Configuration for LoRA-FA training."""

  mode: LoRAFAMode = "corrected_b"
  lora_alpha: float = 32.0
  correction_eps: float = 1e-8
  use_rslora: bool = False


@dataclass
class _PatchState:
  installed: bool = False
  original_trainer_init: Any | None = None
  original_train_step: Any | None = None
  config: LoRAFAConfig = LoRAFAConfig()


_STATE = _PatchState()


def _path_name(path: tuple[Any, ...]) -> str:
  return str(path[-1]) if path else ""


def is_lora_a_path(path: tuple[Any, ...], value: Any) -> bool:
  """Returns true for Qwix LoRA A parameters."""
  return isinstance(value, nnx.LoRAParam) and _path_name(path).endswith("_lora_a")


def is_lora_b_path(path: tuple[Any, ...], value: Any) -> bool:
  """Returns true for Qwix LoRA B parameters."""
  return isinstance(value, nnx.LoRAParam) and _path_name(path).endswith("_lora_b")


def lora_fa_trainable_filter(path: tuple[Any, ...], value: Any) -> bool:
  """NNX filter for LoRA-FA trainable parameters.

  A remains an ``nnx.LoRAParam`` so Tunix LoRA checkpoint paths can still retain
  it, but only B participates in differentiation and optimizer state.
  """
  return is_lora_b_path(path, value)


def lora_fa_diff_state(argnum: int = 0) -> nnx.DiffState:
  return nnx.DiffState(argnum, lora_fa_trainable_filter)


def _a_path_for_b_path(path: tuple[Any, ...]) -> tuple[Any, ...]:
  name = _path_name(path)
  if not name.endswith("_lora_b"):
    raise ValueError(f"Expected a LoRA B path, got {path!r}.")
  return (*path[:-1], name.removesuffix("_lora_b") + "_lora_a")


def _lora_scaling(rank: int, config: LoRAFAConfig) -> jax.Array:
  denominator = jnp.sqrt(rank) if config.use_rslora else rank
  return jnp.asarray(config.lora_alpha, dtype=jnp.float32) / jnp.asarray(
      denominator,
      dtype=jnp.float32,
  )


def _lora_scaling_value(rank: int, config: LoRAFAConfig) -> float:
  denominator = np.sqrt(rank) if config.use_rslora else rank
  return float(config.lora_alpha) / float(denominator)


def _flat_path_key(path: tuple[Any, ...]) -> str:
  return "/".join(map(str, path))


def _correct_b_gradient(
    lora_a: jax.Array,
    grad_b: jax.Array,
    config: LoRAFAConfig,
) -> jax.Array:
  rank = int(lora_a.shape[-1])
  if grad_b.shape[0] != rank:
    raise ValueError(
        "Expected Qwix LoRA B to keep rank on axis 0; "
        f"got A rank {rank} and B grad shape {grad_b.shape}."
    )
  a_flat = jnp.reshape(lora_a, (-1, rank)).astype(jnp.float32)
  grad_dtype = grad_b.dtype
  grad_flat = jnp.reshape(grad_b, (rank, -1)).astype(jnp.float32)
  gram = a_flat.T @ a_flat
  eye = jnp.eye(rank, dtype=jnp.float32)
  inv_gram = jnp.linalg.pinv(gram + config.correction_eps * eye)
  scaling = _lora_scaling(rank, config)
  corrected = (inv_gram @ grad_flat) / (scaling * scaling)
  return jnp.reshape(corrected, grad_b.shape).astype(grad_dtype)


def _correct_b_gradient_with_matrix(
    correction_matrix: jax.Array,
    grad_b: jax.Array,
) -> jax.Array:
  grad_dtype = grad_b.dtype
  rank = int(grad_b.shape[0])
  grad_flat = jnp.reshape(grad_b, (rank, -1)).astype(jnp.float32)
  corrected = jnp.asarray(correction_matrix, dtype=jnp.float32) @ grad_flat
  return jnp.reshape(corrected, grad_b.shape).astype(grad_dtype)


def build_lora_fa_correction_cache(
    model: nnx.Module,
    config: LoRAFAConfig | None = None,
) -> dict[str, jax.Array]:
  """Builds per-LoRA-B correction matrices from frozen A parameters.

  The correction matrix depends only on A and LoRA scaling. Since A is frozen
  under LoRA-FA, computing the pseudo-inverse inside the jitted train step is
  unnecessary and can inflate compile-time memory. The cache stores only small
  rank-by-rank matrices keyed by the matching B parameter path.
  """
  config = config or _STATE.config
  if config.mode == "freeze_a":
    return {}

  cache: dict[str, jax.Array] = {}
  lora_state = dict(nnx.to_flat_state(nnx.state(model, nnx.LoRAParam)))
  for path, variable in lora_state.items():
    if not is_lora_a_path(path, variable):
      continue
    rank = int(variable[...].shape[-1])
    a_flat = np.asarray(jax.device_get(variable[...]), dtype=np.float32).reshape(
        -1,
        rank,
    )
    gram = a_flat.T @ a_flat
    eye = np.eye(rank, dtype=np.float32)
    inv_gram = np.linalg.pinv(gram + config.correction_eps * eye).astype(
        np.float32,
        copy=False,
    )
    scaling = _lora_scaling_value(rank, config)
    b_path = _flat_path_key((*path[:-1], _path_name(path).removesuffix("_lora_a") + "_lora_b"))
    cache[b_path] = jnp.asarray(inv_gram / (scaling * scaling), dtype=jnp.float32)
  return cache


def correct_lora_fa_grads(
    model: nnx.Module,
    grads: nnx.State,
    config: LoRAFAConfig | None = None,
    correction_cache: dict[str, jax.Array] | None = None,
) -> nnx.State:
  """Applies the LoRA-FA B-gradient correction to an NNX gradient state."""
  config = config or _STATE.config
  if config.mode == "freeze_a":
    return grads

  lora_state = None
  if correction_cache is None:
    lora_state = dict(nnx.to_flat_state(nnx.state(model, nnx.LoRAParam)))
  corrected_items = []
  for path, variable in nnx.to_flat_state(grads):
    if not is_lora_b_path(path, variable):
      corrected_items.append((path, variable))
      continue
    if correction_cache is None:
      assert lora_state is not None
      a_path = _a_path_for_b_path(path)
      if a_path not in lora_state:
        raise KeyError(f"Missing LoRA A parameter for B path {path!r}.")
      corrected_value = _correct_b_gradient(
          lora_state[a_path][...],
          variable[...],
          config,
      )
    else:
      b_key = _flat_path_key(path)
      if b_key not in correction_cache:
        raise KeyError(f"Missing LoRA-FA correction matrix for B path {path!r}.")
      corrected_value = _correct_b_gradient_with_matrix(
          correction_cache[b_key],
          variable[...],
      )
    corrected_items.append((path, variable.replace(value=corrected_value)))
  return nnx.State.from_flat_path(corrected_items)


def lora_fa_parameter_summary(model: nnx.Module) -> dict[str, int]:
  """Counts LoRA-FA A/B parameter elements and bytes."""
  summary = {
      "lora_a_params": 0,
      "lora_b_params": 0,
      "lora_a_bytes": 0,
      "lora_b_bytes": 0,
      "lora_a_tensors": 0,
      "lora_b_tensors": 0,
  }
  for path, value in nnx.iter_graph(model):
    if not isinstance(value, nnx.LoRAParam):
      continue
    array = value[...]
    size = int(array.size)
    bytes_ = int(size * array.dtype.itemsize)
    if is_lora_a_path(path, value):
      summary["lora_a_params"] += size
      summary["lora_a_bytes"] += bytes_
      summary["lora_a_tensors"] += 1
    elif is_lora_b_path(path, value):
      summary["lora_b_params"] += size
      summary["lora_b_bytes"] += bytes_
      summary["lora_b_tensors"] += 1
  return summary


def lora_value_snapshot(model: nnx.Module) -> dict[str, np.ndarray]:
  """Copies current LoRA parameter values to host memory."""
  snapshot: dict[str, np.ndarray] = {}
  for path, value in nnx.iter_graph(model):
    if isinstance(value, nnx.LoRAParam):
      snapshot["/".join(map(str, path))] = np.asarray(jax.device_get(value[...]))
  return snapshot


def lora_value_delta_summary(
    before: dict[str, np.ndarray],
    model: nnx.Module,
) -> dict[str, float | int]:
  """Summarizes A/B value changes relative to a host snapshot."""
  max_a_delta = 0.0
  max_b_delta = 0.0
  changed_a = 0
  changed_b = 0
  missing_before = 0
  for path, value in nnx.iter_graph(model):
    if not isinstance(value, nnx.LoRAParam):
      continue
    key = "/".join(map(str, path))
    if key not in before:
      missing_before += 1
      continue
    delta = np.asarray(jax.device_get(value[...])) - before[key]
    max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    if key.endswith("_lora_a"):
      max_a_delta = max(max_a_delta, max_delta)
      changed_a += int(max_delta != 0.0)
    elif key.endswith("_lora_b"):
      max_b_delta = max(max_b_delta, max_delta)
      changed_b += int(max_delta != 0.0)
  return {
      "lorafa_a_value_delta_max": max_a_delta,
      "lorafa_b_value_delta_max": max_b_delta,
      "lorafa_a_changed_tensors": changed_a,
      "lorafa_b_changed_tensors": changed_b,
      "lorafa_snapshot_missing_tensors": missing_before,
  }


def install(config: LoRAFAConfig | None = None) -> None:
  """Installs a process-local Tunix ``PeftTrainer`` LoRA-FA patch."""
  from tunix.sft import peft_trainer  # pylint: disable=import-outside-toplevel
  from tunix.sft import utils as sft_utils  # pylint: disable=import-outside-toplevel

  config = config or LoRAFAConfig()
  if not _STATE.installed:
    _STATE.original_trainer_init = peft_trainer.PeftTrainer.__init__
    _STATE.original_train_step = peft_trainer.PeftTrainer._train_step  # pylint: disable=protected-access

    def _patched_trainer_init(self, model, optimizer, training_config, *args, **kwargs):
      _STATE.original_trainer_init(
          self,
          model,
          optimizer,
          training_config,
          *args,
          **kwargs,
      )
      if sft_utils.is_lora_enabled(self.model):
        self.optimizer = nnx.Optimizer(
            self.model,
            optimizer,
            wrt=lora_fa_trainable_filter,
        )
        self._tunix_accel_lora_fa_correction_cache = (  # pylint: disable=protected-access
            build_lora_fa_correction_cache(self.model, _STATE.config)
        )

    def _patched_train_step(self, model, optimizer, inputs):
      inputs = self.gen_model_input_fn(inputs)
      grad_fn = nnx.value_and_grad(
          self.loss_fn,
          argnums=lora_fa_diff_state(0) if self._lora_enabled else 0,
          has_aux=self._has_aux,
      )
      out, grads = grad_fn(model, **inputs)
      grads = correct_lora_fa_grads(
          model,
          grads,
          _STATE.config,
          getattr(self, "_tunix_accel_lora_fa_correction_cache", None),
      )
      grad_norm = optax.global_norm(grads)
      optimizer.update(model, grads)
      if self._has_aux:
        loss, aux = out
        return loss, aux, grad_norm
      return out, None, grad_norm

    peft_trainer.PeftTrainer.__init__ = _patched_trainer_init
    peft_trainer.PeftTrainer._train_step = _patched_train_step  # pylint: disable=protected-access
    _STATE.installed = True

  _STATE.config = config


def uninstall() -> None:
  """Restores Tunix's original ``PeftTrainer`` methods."""
  if not _STATE.installed:
    return
  from tunix.sft import peft_trainer  # pylint: disable=import-outside-toplevel

  peft_trainer.PeftTrainer.__init__ = _STATE.original_trainer_init
  peft_trainer.PeftTrainer._train_step = _STATE.original_train_step  # pylint: disable=protected-access
  _STATE.installed = False


def is_installed() -> bool:
  return _STATE.installed


@contextmanager
def patched(config: LoRAFAConfig | None = None):
  """Temporarily installs the Tunix LoRA-FA trainer patch."""
  was_installed = _STATE.installed
  old_config = _STATE.config
  install(config)
  try:
    yield
  finally:
    if was_installed:
      _STATE.config = old_config
    else:
      uninstall()
