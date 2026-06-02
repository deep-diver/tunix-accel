"""Drop-in Tunix loss helpers for Cut Cross Entropy (CCE)."""

from __future__ import annotations

from flax import nnx
import jax

from tunix_accel.chunked_linear_ce import make_lm_head_ce
from tunix_accel import model_adapters


def chunked_lm_head_ce_loss_fn(
    token_chunk: int = 128,
    *,
    train_lm_head: bool = False,
    vocab_chunk: int = 8192,
):
  """Builds a Tunix-compatible CCE loss.

  The returned function has the same signature expected by
  `tunix.sft.peft_trainer.PeftTrainer.with_loss_fn`. Internally, the
  implementation streams the LM-head loss over token/vocab chunks.
  """

  def _loss_fn(
      model: nnx.Module,
      input_tokens: jax.Array,
      input_mask: jax.Array,
      positions: jax.Array,
      attention_mask: jax.Array,
      images: jax.Array | None = None,
  ) -> jax.Array:
    parts = model_adapters.extract_lm_head_parts(
        model,
        input_tokens,
        positions,
        attention_mask,
        images,
    )
    ce = make_lm_head_ce(
        token_chunk,
        train_lm_head=train_lm_head,
        vocab_chunk=vocab_chunk,
        logit_softcap=parts.logit_softcap,
    )
    return ce(
        parts.hidden[:, :-1, :],
        parts.head_kernel,
        input_tokens[:, 1:],
        input_mask[:, 1:],
    )

  return _loss_fn


def frozen_lm_head_ce_loss_fn(token_chunk: int = 128, vocab_chunk: int = 8192):
  """Builds a frozen-head Tunix loss for LoRA/PEFT."""
  return chunked_lm_head_ce_loss_fn(
      token_chunk,
      train_lm_head=False,
      vocab_chunk=vocab_chunk,
  )


def trainable_lm_head_ce_loss_fn(token_chunk: int = 128, vocab_chunk: int = 8192):
  """Builds a trainable-head Tunix loss for full fine-tuning."""
  return chunked_lm_head_ce_loss_fn(
      token_chunk,
      train_lm_head=True,
      vocab_chunk=vocab_chunk,
  )


def use_frozen_lm_head_ce(
    trainer,
    *,
    token_chunk: int = 128,
    vocab_chunk: int = 8192,
):
  """Applies the frozen-head CCE loss to a Tunix PeftTrainer.

  Example:

    trainer = peft_trainer.PeftTrainer(...).with_gen_model_input_fn(...)
    trainer = use_frozen_lm_head_ce(trainer, token_chunk=128)

  Returns the same trainer object returned by `trainer.with_loss_fn(...)`.
  """
  model_adapters.prepare_intercepted_lora_model(trainer.model)
  return trainer.with_loss_fn(frozen_lm_head_ce_loss_fn(token_chunk, vocab_chunk))


def use_trainable_lm_head_ce(
    trainer,
    *,
    token_chunk: int = 128,
    vocab_chunk: int = 8192,
):
  """Applies trainable-head CCE to a Tunix PeftTrainer."""
  model_adapters.prepare_intercepted_lora_model(trainer.model)
  return trainer.with_loss_fn(trainable_lm_head_ce_loss_fn(token_chunk, vocab_chunk))
