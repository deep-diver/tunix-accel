"""Memory-efficient linear-head cross entropy for decoder LM training.

This module computes exact next-token cross entropy from final hidden states and
an LM-head kernel without materializing a full [batch, length, vocab] logits
tensor. It streams both token and vocabulary chunks: the forward pass computes
log-sum-exp over vocabulary chunks, and the custom VJP recomputes those chunks
during backward.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import jax
import jax.numpy as jnp


Array = jax.Array


def _validate_token_chunk(token_chunk: int) -> int:
  token_chunk = int(token_chunk)
  if token_chunk <= 0:
    raise ValueError(f"token_chunk must be positive, got {token_chunk}.")
  return token_chunk


def _validate_vocab_chunk(vocab_chunk: int) -> int:
  vocab_chunk = int(vocab_chunk)
  if vocab_chunk <= 0:
    raise ValueError(f"vocab_chunk must be positive, got {vocab_chunk}.")
  return vocab_chunk


def _flatten_and_pad(
    hidden: Array,
    target_tokens: Array,
    target_mask: Array,
    token_chunk: int,
) -> tuple[Array, Array, Array, int, tuple[int, ...]]:
  """Flattens token dimensions and pads to a static chunk multiple."""
  original_shape = tuple(hidden.shape)
  hidden_dim = original_shape[-1]
  flat_hidden = hidden.reshape((-1, hidden_dim))
  flat_targets = target_tokens.reshape((-1,))
  flat_mask = target_mask.reshape((-1,)).astype(jnp.float32)

  n_tokens = flat_hidden.shape[0]
  chunks = (n_tokens + token_chunk - 1) // token_chunk
  padded_n = chunks * token_chunk
  pad_n = padded_n - n_tokens
  if pad_n:
    flat_hidden = jnp.pad(flat_hidden, ((0, pad_n), (0, 0)))
    flat_targets = jnp.pad(flat_targets, ((0, pad_n),), constant_values=0)
    flat_mask = jnp.pad(flat_mask, ((0, pad_n),))
  return flat_hidden, flat_targets, flat_mask, n_tokens, original_shape


def _pad_head_kernel(head_kernel: Array, vocab_chunk: int) -> tuple[Array, int]:
  """Pads the LM head so vocab chunks have a static size."""
  vocab_size = head_kernel.shape[-1]
  chunks = (vocab_size + vocab_chunk - 1) // vocab_chunk
  padded_vocab = chunks * vocab_chunk
  pad_v = padded_vocab - vocab_size
  if pad_v:
    head_kernel = jnp.pad(head_kernel, ((0, 0), (0, pad_v)))
  return head_kernel, vocab_size


def _maybe_softcap(logits: Array, logit_softcap: float | None) -> Array:
  if logit_softcap is None:
    return logits
  cap = jnp.array(logit_softcap, dtype=logits.dtype)
  return jnp.tanh(logits / cap) * cap


def _softcap_grad(raw_logits: Array, logit_softcap: float | None) -> Array:
  if logit_softcap is None:
    return jnp.ones_like(raw_logits, dtype=jnp.float32)
  cap = jnp.array(logit_softcap, dtype=jnp.float32)
  y = jnp.tanh(raw_logits.astype(jnp.float32) / cap)
  return 1.0 - jnp.square(y)


def _vocab_chunk_logits(
    x: Array,
    head_kernel: Array,
    vocab_start: Array,
    vocab_chunk: int,
    vocab_size: int,
    logit_softcap: float | None,
) -> tuple[Array, Array, Array]:
  """Computes one vocab block and masks padded columns."""
  hidden_dim = x.shape[-1]
  head = jax.lax.dynamic_slice(
      head_kernel,
      (0, vocab_start),
      (hidden_dim, vocab_chunk),
  )
  raw_logits = jnp.dot(x, head).astype(jnp.float32)
  logits = _maybe_softcap(raw_logits, logit_softcap)
  cols = vocab_start + jnp.arange(vocab_chunk)
  valid_cols = cols < vocab_size
  logits = jnp.where(valid_cols[None, :], logits, -jnp.inf)
  return raw_logits, logits, valid_cols


def _token_chunk_lse_and_target(
    x: Array,
    head_kernel: Array,
    y: Array,
    vocab_chunk: int,
    vocab_size: int,
    logit_softcap: float | None,
) -> tuple[Array, Array]:
  """Streams log-sum-exp and target logits across vocab chunks."""
  vocab_chunks = head_kernel.shape[-1] // vocab_chunk
  init_max = jnp.full((x.shape[0],), -jnp.inf, dtype=jnp.float32)
  init_sum = jnp.zeros((x.shape[0],), dtype=jnp.float32)
  init_target = jnp.zeros((x.shape[0],), dtype=jnp.float32)

  def body(i: Array, state: tuple[Array, Array, Array]):
    running_max, running_sum, target_logits = state
    vocab_start = i * vocab_chunk
    _, logits, _ = _vocab_chunk_logits(
        x,
        head_kernel,
        vocab_start,
        vocab_chunk,
        vocab_size,
        logit_softcap,
    )
    chunk_max = jnp.max(logits, axis=-1)
    new_max = jnp.maximum(running_max, chunk_max)
    new_sum = (
        running_sum * jnp.exp(running_max - new_max)
        + jnp.sum(jnp.exp(logits - new_max[:, None]), axis=-1)
    )

    local_y = y - vocab_start
    in_chunk = (local_y >= 0) & (local_y < vocab_chunk)
    local_y = jnp.clip(local_y, 0, vocab_chunk - 1)
    selected = jnp.take_along_axis(logits, local_y[:, None], axis=-1)[:, 0]
    target_logits = jnp.where(in_chunk, selected, target_logits)
    return new_max, new_sum, target_logits

  max_logits, sum_exp, target_logits = jax.lax.fori_loop(
      0,
      vocab_chunks,
      body,
      (init_max, init_sum, init_target),
  )
  return max_logits + jnp.log(sum_exp), target_logits


def _streaming_kernel_loss_impl(
    flat_hidden: Array,
    head_kernel: Array,
    target_tokens: Array,
    target_mask: Array,
    token_chunk: int,
    vocab_chunk: int,
    vocab_size: int,
    logit_softcap: float | None,
) -> Array:
  hidden_dim = flat_hidden.shape[-1]
  token_chunks = flat_hidden.shape[0] // token_chunk
  norm = jnp.sum(target_mask) + jnp.array(1e-8, dtype=jnp.float32)

  def body(i: Array, total: Array) -> Array:
    start = i * token_chunk
    x = jax.lax.dynamic_slice(
        flat_hidden,
        (start, 0),
        (token_chunk, hidden_dim),
    )
    y = jax.lax.dynamic_slice(target_tokens, (start,), (token_chunk,))
    mask = jax.lax.dynamic_slice(target_mask, (start,), (token_chunk,))
    lse, target_logits = _token_chunk_lse_and_target(
        x,
        head_kernel,
        y,
        vocab_chunk,
        vocab_size,
        logit_softcap,
    )
    return total + jnp.sum((lse - target_logits) * mask)

  loss_sum = jax.lax.fori_loop(0, token_chunks, body, jnp.array(0.0, jnp.float32))
  return loss_sum / norm


@lru_cache(maxsize=32)
def make_lm_head_ce(
    token_chunk: int,
    *,
    train_lm_head: bool,
    vocab_chunk: int = 8192,
    logit_softcap: float | None = None,
) -> Callable[[Array, Array, Array, Array], Array]:
  """Returns a custom-VJP CE function for a static token chunk size.

  The returned function has signature:

    loss(hidden, head_kernel, target_tokens, target_mask) -> scalar loss

  `head_kernel` must have shape [hidden_dim, vocab]. When `train_lm_head` is
  true, the backward pass returns an exact gradient for that kernel. When false,
  it only returns a hidden-state gradient, matching frozen-head LoRA/PEFT.
  """
  token_chunk = _validate_token_chunk(token_chunk)
  vocab_chunk = _validate_vocab_chunk(vocab_chunk)

  @jax.custom_vjp
  def lm_head_ce(
      hidden: Array,
      head_kernel: Array,
      target_tokens: Array,
      target_mask: Array,
  ) -> Array:
    flat_hidden, flat_targets, flat_mask, _, _ = _flatten_and_pad(
        hidden,
        target_tokens,
        target_mask,
        token_chunk,
    )
    padded_head, vocab_size = _pad_head_kernel(head_kernel, vocab_chunk)
    return _streaming_kernel_loss_impl(
        flat_hidden,
        padded_head,
        flat_targets,
        flat_mask,
        token_chunk,
        vocab_chunk,
        vocab_size,
        logit_softcap,
    )

  def fwd(
      hidden: Array,
      head_kernel: Array,
      target_tokens: Array,
      target_mask: Array,
  ) -> tuple[Array, tuple[Array, Array, Array, Array, Array, int, int, tuple[int, ...]]]:
    flat_hidden, flat_targets, flat_mask, n_tokens, original_shape = _flatten_and_pad(
        hidden,
        target_tokens,
        target_mask,
        token_chunk,
    )
    padded_head, vocab_size = _pad_head_kernel(head_kernel, vocab_chunk)
    loss = _streaming_kernel_loss_impl(
        flat_hidden,
        padded_head,
        flat_targets,
        flat_mask,
        token_chunk,
        vocab_chunk,
        vocab_size,
        logit_softcap,
    )
    norm = jnp.sum(flat_mask) + jnp.array(1e-8, dtype=jnp.float32)
    return loss, (
        flat_hidden,
        padded_head,
        flat_targets,
        flat_mask,
        norm,
        n_tokens,
        vocab_size,
        original_shape,
    )

  def bwd(
      residuals: tuple[Array, Array, Array, Array, Array, int, int, tuple[int, ...]],
      loss_cotangent: Array,
  ) -> tuple[Array, Array | None, None, None]:
    (
        flat_hidden,
        head_kernel,
        flat_targets,
        flat_mask,
        norm,
        n_tokens,
        vocab_size,
        original_shape,
    ) = residuals
    hidden_dim = flat_hidden.shape[-1]
    padded_vocab_size = head_kernel.shape[-1]
    token_chunks = flat_hidden.shape[0] // token_chunk
    vocab_chunks = padded_vocab_size // vocab_chunk
    head_kernel_f32 = head_kernel.astype(jnp.float32)
    grad_hidden = jnp.zeros(flat_hidden.shape, dtype=jnp.float32)
    scale = loss_cotangent.astype(jnp.float32) / norm

    if train_lm_head:
      grad_head = jnp.zeros((hidden_dim, padded_vocab_size), dtype=jnp.float32)

    def token_body_frozen(i: Array, hidden_acc: Array) -> Array:
      start = i * token_chunk
      x = jax.lax.dynamic_slice(
          flat_hidden,
          (start, 0),
          (token_chunk, hidden_dim),
      )
      y = jax.lax.dynamic_slice(flat_targets, (start,), (token_chunk,))
      mask = jax.lax.dynamic_slice(flat_mask, (start,), (token_chunk,))
      lse, _ = _token_chunk_lse_and_target(
          x,
          head_kernel,
          y,
          vocab_chunk,
          vocab_size,
          logit_softcap,
      )

      def vocab_body(j: Array, grad_x: Array) -> Array:
        vocab_start = j * vocab_chunk
        raw_logits, logits, valid_cols = _vocab_chunk_logits(
            x,
            head_kernel,
            vocab_start,
            vocab_chunk,
            vocab_size,
            logit_softcap,
        )
        cols = vocab_start + jnp.arange(vocab_chunk)
        grad_logits = jnp.exp(logits - lse[:, None])
        grad_logits = jnp.where(valid_cols[None, :], grad_logits, 0.0)
        target_cols = cols[None, :] == y[:, None]
        grad_logits = grad_logits - target_cols.astype(jnp.float32)
        grad_logits = grad_logits * (mask * scale)[:, None]
        grad_logits = grad_logits * _softcap_grad(raw_logits, logit_softcap)
        grad_logits = jnp.where(valid_cols[None, :], grad_logits, 0.0)
        head = jax.lax.dynamic_slice(
            head_kernel_f32,
            (0, vocab_start),
            (hidden_dim, vocab_chunk),
        )
        return grad_x + jnp.dot(grad_logits, head.T)

      grad_x = jax.lax.fori_loop(
          0,
          vocab_chunks,
          vocab_body,
          jnp.zeros((token_chunk, hidden_dim), dtype=jnp.float32),
      )
      return jax.lax.dynamic_update_slice(hidden_acc, grad_x, (start, 0))

    def token_body_trainable(i: Array, acc: tuple[Array, Array]) -> tuple[Array, Array]:
      hidden_acc, head_acc = acc
      start = i * token_chunk
      x = jax.lax.dynamic_slice(
          flat_hidden,
          (start, 0),
          (token_chunk, hidden_dim),
      )
      y = jax.lax.dynamic_slice(flat_targets, (start,), (token_chunk,))
      mask = jax.lax.dynamic_slice(flat_mask, (start,), (token_chunk,))
      lse, _ = _token_chunk_lse_and_target(
          x,
          head_kernel,
          y,
          vocab_chunk,
          vocab_size,
          logit_softcap,
      )

      def vocab_body(
          j: Array,
          vocab_acc: tuple[Array, Array],
      ) -> tuple[Array, Array]:
        grad_x_acc, head_acc_inner = vocab_acc
        vocab_start = j * vocab_chunk
        raw_logits, logits, valid_cols = _vocab_chunk_logits(
            x,
            head_kernel,
            vocab_start,
            vocab_chunk,
            vocab_size,
            logit_softcap,
        )
        cols = vocab_start + jnp.arange(vocab_chunk)
        grad_logits = jnp.exp(logits - lse[:, None])
        grad_logits = jnp.where(valid_cols[None, :], grad_logits, 0.0)
        target_cols = cols[None, :] == y[:, None]
        grad_logits = grad_logits - target_cols.astype(jnp.float32)
        grad_logits = grad_logits * (mask * scale)[:, None]
        grad_logits = grad_logits * _softcap_grad(raw_logits, logit_softcap)
        grad_logits = jnp.where(valid_cols[None, :], grad_logits, 0.0)
        head = jax.lax.dynamic_slice(
            head_kernel_f32,
            (0, vocab_start),
            (hidden_dim, vocab_chunk),
        )
        grad_x_acc = grad_x_acc + jnp.dot(grad_logits, head.T)
        old_head_grad = jax.lax.dynamic_slice(
            head_acc_inner,
            (0, vocab_start),
            (hidden_dim, vocab_chunk),
        )
        new_head_grad = old_head_grad + jnp.dot(x.astype(jnp.float32).T, grad_logits)
        head_acc_inner = jax.lax.dynamic_update_slice(
            head_acc_inner,
            new_head_grad,
            (0, vocab_start),
        )
        return grad_x_acc, head_acc_inner

      grad_x, head_acc = jax.lax.fori_loop(
          0,
          vocab_chunks,
          vocab_body,
          (
              jnp.zeros((token_chunk, hidden_dim), dtype=jnp.float32),
              head_acc,
          ),
      )
      hidden_acc = jax.lax.dynamic_update_slice(hidden_acc, grad_x, (start, 0))
      return hidden_acc, head_acc

    if train_lm_head:
      grad_hidden, grad_head = jax.lax.fori_loop(
          0,
          token_chunks,
          token_body_trainable,
          (grad_hidden, grad_head),
      )
    else:
      grad_hidden = jax.lax.fori_loop(
          0,
          token_chunks,
          token_body_frozen,
          grad_hidden,
      )
    grad_hidden = grad_hidden[:n_tokens].reshape(original_shape)
    if train_lm_head:
      return (
          grad_hidden.astype(flat_hidden.dtype),
          grad_head[:, :vocab_size].astype(head_kernel.dtype),
          None,
          None,
      )
    return grad_hidden.astype(flat_hidden.dtype), None, None, None

  lm_head_ce.defvjp(fwd, bwd)
  return lm_head_ce


def lm_head_cross_entropy(
    hidden: Array,
    head_kernel: Array,
    target_tokens: Array,
    target_mask: Array,
    *,
    token_chunk: int,
    vocab_chunk: int = 8192,
    train_lm_head: bool,
    logit_softcap: float | None = None,
) -> Array:
  """Computes chunked LM-head CE with a cached custom-VJP implementation."""
  return make_lm_head_ce(
      token_chunk,
      train_lm_head=train_lm_head,
      vocab_chunk=vocab_chunk,
      logit_softcap=logit_softcap,
  )(
      hidden,
      head_kernel,
      target_tokens,
      target_mask,
  )


@lru_cache(maxsize=32)
def make_frozen_lm_head_ce(
    token_chunk: int,
    vocab_chunk: int = 8192,
) -> Callable[[Array, Array, Array, Array], Array]:
  """Returns a frozen tied-embedding CE function.

  This backward-compatible helper accepts `embedding` as [vocab, hidden_dim].
  """
  ce = make_lm_head_ce(
      token_chunk,
      train_lm_head=False,
      vocab_chunk=vocab_chunk,
  )

  def _loss(
      hidden: Array,
      embedding: Array,
      target_tokens: Array,
      target_mask: Array,
  ) -> Array:
    return ce(hidden, embedding.T, target_tokens, target_mask)

  return _loss


def frozen_lm_head_cross_entropy(
    hidden: Array,
    embedding: Array,
    target_tokens: Array,
    target_mask: Array,
    *,
    token_chunk: int,
    vocab_chunk: int = 8192,
) -> Array:
  """Computes frozen tied-embedding CE for PEFT/LoRA."""
  return make_frozen_lm_head_ce(token_chunk, vocab_chunk)(
      hidden,
      embedding,
      target_tokens,
      target_mask,
  )
