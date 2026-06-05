"""Drop-in Tunix adapters for sequence-packed SFT batches.

The core packer in :mod:`tunix_accel.packing` works on tokenized records. This
module adapts ordinary Tunix SFT batches, where examples are usually already
padded into ``input_tokens`` / ``input_mask`` arrays, into packed Tunix batches.

The adapter intentionally does not touch the loss function. It can therefore be
used with Tunix's default CE path or with the tunix-accel CCE patch.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from tunix_accel.packing import LongExamplePolicy
from tunix_accel.packing import PackingStrategy
from tunix_accel.packing import pack_records


_TRAINER_PACKING_CONFIG_ATTR = "_tunix_accel_packing_config"
_PACKING_DISABLED = object()


@dataclass(frozen=True)
class TunixPackingConfig:
  """Configuration for adapting Tunix batches to sequence-packed batches."""

  batch_size: int | None = None
  max_length: int | None = None
  pad_token_id: int = 0
  strategy: PackingStrategy = "best_fit_decreasing"
  long_example_policy: LongExamplePolicy = "error"
  drop_remainder: bool = True
  token_key: str = "input_tokens"
  loss_mask_key: str = "input_mask"
  valid_mask_key: str = "valid_mask"


@dataclass
class _PatchState:
  installed: bool = False
  api_patched: bool = False
  original_train: Any | None = None
  original_with_gen_model_input_fn: Any | None = None
  trainer_cls: Any | None = None
  config: TunixPackingConfig = TunixPackingConfig()


_STATE = _PatchState()


def _as_2d_array(value: Any, *, key: str) -> np.ndarray:
  array = np.asarray(value)
  if array.ndim == 1:
    array = array[None, :]
  if array.ndim != 2:
    raise ValueError(f"{key!r} must be rank 1 or 2, got shape {array.shape}.")
  return array


def _first_batch_and_iterator(
    batches: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
  iterator = iter(batches)
  try:
    first = next(iterator)
  except StopIteration as exc:
    raise ValueError("Cannot pack an empty Tunix dataset.") from exc
  return first, iterator


def _resolve_config(
    config: TunixPackingConfig,
    first_batch: Mapping[str, Any],
) -> TunixPackingConfig:
  input_tokens = _as_2d_array(first_batch[config.token_key], key=config.token_key)
  batch_size, max_length = input_tokens.shape
  resolved = config
  if config.batch_size is None:
    resolved = replace(resolved, batch_size=int(batch_size))
  if config.max_length is None:
    resolved = replace(resolved, max_length=int(max_length))
  if resolved.batch_size is None or resolved.batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {resolved.batch_size}.")
  if resolved.max_length is None or resolved.max_length <= 0:
    raise ValueError(f"max_length must be positive, got {resolved.max_length}.")
  return resolved


def _iter_batches_with_first(
    first: Mapping[str, Any],
    rest: Iterator[Mapping[str, Any]],
) -> Iterator[Mapping[str, Any]]:
  yield first
  yield from rest


def _valid_mask_for_batch(
    batch: Mapping[str, Any],
    input_tokens: np.ndarray,
    config: TunixPackingConfig,
) -> np.ndarray:
  if config.valid_mask_key in batch:
    return _as_2d_array(batch[config.valid_mask_key], key=config.valid_mask_key).astype(
        bool
    )
  return input_tokens != config.pad_token_id


def _loss_mask_for_batch(
    batch: Mapping[str, Any],
    valid_mask: np.ndarray,
    config: TunixPackingConfig,
) -> np.ndarray:
  if config.loss_mask_key in batch:
    return _as_2d_array(batch[config.loss_mask_key], key=config.loss_mask_key).astype(
        bool
    )
  return valid_mask


def _examples_from_tunix_batches(
    batches: Iterable[Mapping[str, Any]],
    config: TunixPackingConfig,
) -> list[dict[str, Any]]:
  records: list[dict[str, Any]] = []
  next_id = 0
  for batch in batches:
    input_tokens = _as_2d_array(batch[config.token_key], key=config.token_key)
    valid_mask = _valid_mask_for_batch(batch, input_tokens, config)
    loss_mask = _loss_mask_for_batch(batch, valid_mask, config)
    if valid_mask.shape != input_tokens.shape:
      raise ValueError(
          f"{config.valid_mask_key!r} shape {valid_mask.shape} does not match "
          f"{config.token_key!r} shape {input_tokens.shape}."
      )
    if loss_mask.shape != input_tokens.shape:
      raise ValueError(
          f"{config.loss_mask_key!r} shape {loss_mask.shape} does not match "
          f"{config.token_key!r} shape {input_tokens.shape}."
      )
    for row_idx in range(input_tokens.shape[0]):
      valid_positions = np.flatnonzero(valid_mask[row_idx])
      if valid_positions.size == 0:
        continue
      end = int(valid_positions[-1]) + 1
      records.append({
          "id": next_id,
          "input_ids": input_tokens[row_idx, :end].astype(np.int32).tolist(),
          "labels": input_tokens[row_idx, :end].astype(np.int32).tolist(),
          "loss_mask": loss_mask[row_idx, :end].astype(bool).tolist(),
      })
      next_id += 1
  return records


def pack_tunix_batches(
    batches: Iterable[Mapping[str, Any]],
    config: TunixPackingConfig | None = None,
) -> Iterator[dict[str, np.ndarray]]:
  """Yields sequence-packed Tunix batches from ordinary Tunix batches.

  ``batch_size`` and ``max_length`` default to the first incoming batch shape, so
  callers can wrap an existing Tunix dataset without restating those values.
  """
  config = config or TunixPackingConfig()
  first, rest = _first_batch_and_iterator(batches)
  config = _resolve_config(config, first)
  records = _examples_from_tunix_batches(
      _iter_batches_with_first(first, rest),
      config,
  )
  packed = pack_records(
      records,
      max_length=int(config.max_length),
      pad_token_id=config.pad_token_id,
      strategy=config.strategy,
      long_example_policy=config.long_example_policy,
      return_attention_mask=True,
  ).as_tunix(token_key=config.token_key)

  assert config.batch_size is not None
  total_rows = int(packed[config.token_key].shape[0])
  usable_rows = (total_rows // config.batch_size) * config.batch_size
  if not config.drop_remainder and usable_rows < total_rows:
    usable_rows = total_rows

  for start in range(0, usable_rows, config.batch_size):
    end = min(start + config.batch_size, total_rows)
    batch = {
        key: value[start:end]
        for key, value in packed.items()
        if isinstance(value, np.ndarray)
    }
    if config.drop_remainder and batch[config.token_key].shape[0] != config.batch_size:
      continue
    yield batch


def packed_input_fn(
    *,
    pad_token_id: int = 0,
    token_key: str = "input_tokens",
):
  """Returns a Tunix ``gen_model_input_fn`` that respects packed masks."""

  def gen_model_input_fn(batch):
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from tunix.sft import utils as sft_utils  # pylint: disable=import-outside-toplevel

    input_tokens = jnp.asarray(batch[token_key], dtype=jnp.int32)
    input_mask = jnp.asarray(batch["input_mask"], dtype=bool)
    valid_mask = jnp.asarray(
        batch.get("valid_mask", input_tokens != pad_token_id),
        dtype=bool,
    )
    positions = batch.get("positions")
    attention_mask = batch.get("attention_mask")
    if positions is None:
      positions = sft_utils.build_positions_from_mask(valid_mask)
    else:
      positions = jnp.asarray(positions, dtype=jnp.int32)
    if attention_mask is None:
      attention_mask = sft_utils.make_causal_attn_mask(valid_mask)
    else:
      attention_mask = jnp.asarray(attention_mask, dtype=bool)
    return {
        "input_tokens": input_tokens,
        "input_mask": input_mask,
        "positions": positions,
        "attention_mask": attention_mask,
    }

  return gen_model_input_fn


def _coerce_packing_config(packing: Any) -> TunixPackingConfig | None:
  if packing is None:
    return None
  if packing is False:
    return None
  if packing is True:
    return TunixPackingConfig()
  if isinstance(packing, TunixPackingConfig):
    return packing
  if isinstance(packing, Mapping):
    return TunixPackingConfig(**packing)
  raise TypeError(
      "packing must be None, False, True, a TunixPackingConfig, or a mapping "
      f"of TunixPackingConfig fields; got {type(packing)!r}."
  )


def _wrap_gen_model_input_fn(
    gen_model_input_fn: Any,
    config: TunixPackingConfig,
):
  if getattr(gen_model_input_fn, "_tunix_accel_packing_wrapped", False):
    return gen_model_input_fn

  packed_fn = packed_input_fn(
      pad_token_id=config.pad_token_id,
      token_key=config.token_key,
  )

  def _packed_gen_model_input_fn(batch):
    model_inputs = {}
    if gen_model_input_fn is not None:
      model_inputs = dict(gen_model_input_fn(batch))
    model_inputs.update(packed_fn(batch))
    return model_inputs

  setattr(_packed_gen_model_input_fn, "_tunix_accel_packing_wrapped", True)
  return _packed_gen_model_input_fn


def _trainer_config(self) -> TunixPackingConfig | None:
  config = getattr(self, _TRAINER_PACKING_CONFIG_ATTR, None)
  if config is _PACKING_DISABLED:
    return None
  if isinstance(config, TunixPackingConfig):
    return config
  if _STATE.installed:
    return _STATE.config
  return None


def _ensure_packed_input_fn(self, config: TunixPackingConfig) -> None:
  gen_model_input_fn = getattr(self, "gen_model_input_fn", None)
  if getattr(gen_model_input_fn, "_tunix_accel_packing_wrapped", False):
    return
  assert _STATE.original_with_gen_model_input_fn is not None
  _STATE.original_with_gen_model_input_fn(
      self,
      _wrap_gen_model_input_fn(gen_model_input_fn, config),
  )


def patch_trainer_api(peft_trainer_module: Any | None = None) -> None:
  """Adds optional ``packing=`` support to Tunix ``PeftTrainer``.

  This only widens the trainer API. It is a no-op for normal Tunix code unless a
  caller passes ``packing=`` to ``with_gen_model_input_fn`` or enables the legacy
  process-wide ``install()`` wrapper below.
  """
  if peft_trainer_module is None:
    from tunix.sft import peft_trainer as peft_trainer_module  # pylint: disable=import-outside-toplevel

  trainer_cls = peft_trainer_module.PeftTrainer
  if _STATE.api_patched:
    if _STATE.trainer_cls is trainer_cls:
      return
    raise RuntimeError("Tunix packing API is already patched on another trainer.")

  _STATE.original_train = trainer_cls.train
  _STATE.original_with_gen_model_input_fn = trainer_cls.with_gen_model_input_fn
  _STATE.trainer_cls = trainer_cls

  def _with_gen_model_input_fn(self, gen_model_input_fn, *, packing=None):
    if packing is False:
      setattr(self, _TRAINER_PACKING_CONFIG_ATTR, _PACKING_DISABLED)
      config = None
    else:
      config = _coerce_packing_config(packing)
      setattr(self, _TRAINER_PACKING_CONFIG_ATTR, config)
    if config is not None:
      gen_model_input_fn = _wrap_gen_model_input_fn(gen_model_input_fn, config)
    return _STATE.original_with_gen_model_input_fn(self, gen_model_input_fn)

  def _packed_aware_train(
      self,
      train_ds,
      eval_ds=None,
      skip_jit=False,
      *,
      cache_nnx_graph=True,
  ):
    config = _trainer_config(self)
    if config is not None:
      train_ds = pack_tunix_batches(train_ds, config)
      _ensure_packed_input_fn(self, config)
    return _STATE.original_train(
        self,
        train_ds,
        eval_ds=eval_ds,
        skip_jit=skip_jit,
        cache_nnx_graph=cache_nnx_graph,
    )

  trainer_cls.with_gen_model_input_fn = _with_gen_model_input_fn
  trainer_cls.train = _packed_aware_train
  _STATE.api_patched = True


def restore_trainer_api() -> None:
  """Restores the original Tunix trainer API.

  This is mostly useful for tests. Normal installed environments keep the
  widened API active because it is inert until ``packing=`` is supplied.
  """
  if not _STATE.api_patched or _STATE.trainer_cls is None:
    return
  _STATE.trainer_cls.train = _STATE.original_train
  _STATE.trainer_cls.with_gen_model_input_fn = (
      _STATE.original_with_gen_model_input_fn
  )
  _STATE.installed = False
  _STATE.api_patched = False
  _STATE.original_train = None
  _STATE.original_with_gen_model_input_fn = None
  _STATE.trainer_cls = None


def install(config: TunixPackingConfig | None = None) -> None:
  """Enables process-wide Tunix packing for legacy experiments.

  The wrapper only transforms the training dataset and packed input function. It
  leaves Tunix's loss function alone, so it composes with the CCE patch.
  """
  config = config or TunixPackingConfig()
  patch_trainer_api()
  _STATE.config = config
  _STATE.installed = True


def uninstall() -> None:
  """Disables process-wide Tunix packing installed by ``install()``."""
  _STATE.installed = False


def is_installed() -> bool:
  return _STATE.installed


@contextmanager
def packed(config: TunixPackingConfig | None = None):
  """Temporarily installs the Tunix packing wrapper."""
  was_installed = _STATE.installed
  was_api_patched = _STATE.api_patched
  old_config = _STATE.config
  install(config)
  try:
    yield
  finally:
    if was_installed:
      _STATE.config = old_config
    else:
      uninstall()
    if not was_api_patched:
      restore_trainer_api()
