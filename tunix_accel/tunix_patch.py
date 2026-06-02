"""Monkey-patch Tunix defaults with memory-efficient decoder-LM losses.

This is the closest form of drop-in replacement for existing Tunix code:

  from tunix_accel.tunix_patch import install
  install()

  trainer = peft_trainer.PeftTrainer(...)
  trainer.train(...)

After installation, newly-created Tunix `PeftTrainer` instances keep using the
normal Tunix API, but their default loss is replaced for supported decoder-only
LMs. LoRA/PEFT models use a frozen-head backward path; full-parameter models use
a trainable-head backward path. Unsupported models fall back to Tunix's original
default loss when requested.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from flax import nnx
import jax

from tunix_accel import model_adapters
from tunix_accel import tunix_lora_ce


@dataclass
class _PatchState:
  installed: bool = False
  original_default_loss_fn: Any | None = None
  original_trainer_init: Any | None = None
  token_chunk: int = 128
  vocab_chunk: int = 8192
  fallback_to_original: bool = True


_STATE = _PatchState()


def _has_lora_params(model: nnx.Module) -> bool:
  try:
    leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.LoRAParam))
  except Exception:  # pylint: disable=broad-exception-caught
    return False
  return bool(leaves)


def _patched_default_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
    images: jax.Array | None = None,
):
  inputs = {
      "input_tokens": input_tokens,
      "input_mask": input_mask,
      "positions": positions,
      "attention_mask": attention_mask,
      "images": images,
  }
  if model_adapters.is_supported_decoder_lm(model):
    train_lm_head = not _has_lora_params(model)
    loss_fn = tunix_lora_ce.chunked_lm_head_ce_loss_fn(
        _STATE.token_chunk,
        train_lm_head=train_lm_head,
        vocab_chunk=_STATE.vocab_chunk,
    )
    return loss_fn(model, **inputs)

  if _STATE.fallback_to_original and _STATE.original_default_loss_fn is not None:
    return _STATE.original_default_loss_fn(model, **inputs)

  raise TypeError(
      "tunix_accel chunked CE default patch only supports Tunix decoder-only "
      "models exposing embedder/layers/final_norm/LM-head pieces. Call "
      "uninstall(), pass "
      "fallback_to_original=True, or use trainer.with_loss_fn(...) explicitly."
  )


def _prepare_model_if_needed(model: nnx.Module) -> None:
  if model_adapters.is_supported_decoder_lm(model) and _has_lora_params(model):
    model_adapters.prepare_intercepted_lora_model(model)


def install(
    *,
    token_chunk: int = 128,
    vocab_chunk: int = 8192,
    fallback_to_original: bool = True,
) -> None:
  """Installs a process-local Tunix default loss override.

  Existing `PeftTrainer` objects are not mutated. Create trainers after calling
  this function. Calling `install()` again updates the default token chunk.
  """
  if token_chunk <= 0:
    raise ValueError(f"token_chunk must be positive, got {token_chunk}.")
  if vocab_chunk <= 0:
    raise ValueError(f"vocab_chunk must be positive, got {vocab_chunk}.")

  from tunix.sft import peft_trainer  # pylint: disable=import-outside-toplevel

  if not _STATE.installed:
    _STATE.original_default_loss_fn = peft_trainer._default_loss_fn  # pylint: disable=protected-access
    _STATE.original_trainer_init = peft_trainer.PeftTrainer.__init__
    peft_trainer._default_loss_fn = _patched_default_loss_fn  # pylint: disable=protected-access

    def _patched_trainer_init(self, model, *args, **kwargs):
      _prepare_model_if_needed(model)
      return _STATE.original_trainer_init(self, model, *args, **kwargs)

    peft_trainer.PeftTrainer.__init__ = _patched_trainer_init
    _STATE.installed = True

  _STATE.token_chunk = int(token_chunk)
  _STATE.vocab_chunk = int(vocab_chunk)
  _STATE.fallback_to_original = bool(fallback_to_original)


def uninstall() -> None:
  """Restores Tunix's original default loss."""
  if not _STATE.installed:
    return

  from tunix.sft import peft_trainer  # pylint: disable=import-outside-toplevel

  peft_trainer._default_loss_fn = _STATE.original_default_loss_fn  # pylint: disable=protected-access
  if _STATE.original_trainer_init is not None:
    peft_trainer.PeftTrainer.__init__ = _STATE.original_trainer_init
  _STATE.installed = False


def is_installed() -> bool:
  return _STATE.installed


@contextmanager
def patched(
    *,
    token_chunk: int = 128,
    vocab_chunk: int = 8192,
    fallback_to_original: bool = True,
):
  """Temporarily installs the default loss patch."""
  was_installed = _STATE.installed
  old_original = _STATE.original_default_loss_fn
  old_chunk = _STATE.token_chunk
  old_vocab_chunk = _STATE.vocab_chunk
  old_fallback = _STATE.fallback_to_original
  install(
      token_chunk=token_chunk,
      vocab_chunk=vocab_chunk,
      fallback_to_original=fallback_to_original,
  )
  try:
    yield
  finally:
    if was_installed:
      install(
          token_chunk=old_chunk,
          vocab_chunk=old_vocab_chunk,
          fallback_to_original=old_fallback,
      )
      _STATE.original_default_loss_fn = old_original
    else:
      uninstall()
