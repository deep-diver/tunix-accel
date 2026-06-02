#!/usr/bin/env python3
"""Numerical checks for the chunked implementation of CCE."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunix_accel.chunked_linear_ce import frozen_lm_head_cross_entropy
from tunix_accel.chunked_linear_ce import lm_head_cross_entropy
from tunix_accel.chunked_linear_ce import make_frozen_lm_head_ce
from tunix_accel.chunked_linear_ce import make_lm_head_ce


def maybe_softcap(logits, logit_softcap):
  if logit_softcap is None:
    return logits
  return jnp.tanh(logits / logit_softcap) * logit_softcap


def dense_ce(hidden, head_kernel, target_tokens, target_mask, logit_softcap=None):
  flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
  flat_targets = target_tokens.reshape((-1,))
  flat_mask = target_mask.reshape((-1,)).astype(jnp.float32)
  logits = jnp.dot(flat_hidden, head_kernel).astype(jnp.float32)
  logits = maybe_softcap(logits, logit_softcap)
  token_log_probs = jnp.take_along_axis(
      jax.nn.log_softmax(logits, axis=-1),
      flat_targets[:, None],
      axis=-1,
  )[:, 0]
  return -jnp.sum(token_log_probs * flat_mask) / (jnp.sum(flat_mask) + 1e-8)


def assert_close(name, actual, expected, atol=5e-3, rtol=5e-3):
  max_diff = jnp.max(jnp.abs(actual - expected))
  if not bool(jnp.allclose(actual, expected, atol=atol, rtol=rtol)):
    raise AssertionError(f"{name} mismatch: max_diff={float(max_diff)}")
  print(f"{name}=ok max_diff={float(max_diff):.6g}")


def test_chunked_linear_ce_numerics() -> None:
  key = jax.random.PRNGKey(0)
  hidden = jax.random.normal(key, (3, 7, 11), dtype=jnp.float32)
  embedding = jax.random.normal(jax.random.fold_in(key, 1), (37, 11), dtype=jnp.float32)
  head_kernel = embedding.T
  targets = jax.random.randint(jax.random.fold_in(key, 2), (3, 7), 0, 37)
  mask = jnp.array(
      [
          [1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0],
      ],
      dtype=bool,
  )

  for chunk in (1, 2, 5, 8, 64):
    vocab_chunk = 7
    expected = dense_ce(hidden, head_kernel, targets, mask)
    loss = frozen_lm_head_cross_entropy(
        hidden,
        embedding,
        targets,
        mask,
        token_chunk=chunk,
        vocab_chunk=vocab_chunk,
    )
    assert_close(f"frozen_loss_chunk_{chunk}", loss, expected)

    grad_dense = jax.grad(lambda x: dense_ce(x, head_kernel, targets, mask))(hidden)
    grad_chunked = jax.grad(
        lambda x: frozen_lm_head_cross_entropy(
            x,
            embedding,
            targets,
            mask,
            token_chunk=chunk,
            vocab_chunk=vocab_chunk,
        )
    )(hidden)
    assert_close(f"frozen_grad_hidden_chunk_{chunk}", grad_chunked, grad_dense)

    full_loss = lm_head_cross_entropy(
        hidden,
        head_kernel,
        targets,
        mask,
        token_chunk=chunk,
        vocab_chunk=vocab_chunk,
        train_lm_head=True,
    )
    assert_close(f"full_loss_chunk_{chunk}", full_loss, expected)
    grad_dense_hidden, grad_dense_head = jax.grad(
        lambda x, w: dense_ce(x, w, targets, mask),
        argnums=(0, 1),
    )(hidden, head_kernel)
    grad_chunk_hidden, grad_chunk_head = jax.grad(
        lambda x, w: lm_head_cross_entropy(
            x,
            w,
            targets,
            mask,
            token_chunk=chunk,
            vocab_chunk=vocab_chunk,
            train_lm_head=True,
        ),
        argnums=(0, 1),
    )(hidden, head_kernel)
    assert_close(f"full_grad_hidden_chunk_{chunk}", grad_chunk_hidden, grad_dense_hidden)
    assert_close(f"full_grad_head_chunk_{chunk}", grad_chunk_head, grad_dense_head)

  softcap = 3.0
  softcap_loss = lm_head_cross_entropy(
      hidden,
      head_kernel,
      targets,
      mask,
      token_chunk=5,
      vocab_chunk=6,
      train_lm_head=True,
      logit_softcap=softcap,
  )
  assert_close(
      "softcap_loss",
      softcap_loss,
      dense_ce(hidden, head_kernel, targets, mask, logit_softcap=softcap),
  )
  soft_grad_dense = jax.grad(
      lambda x: dense_ce(x, head_kernel, targets, mask, logit_softcap=softcap)
  )(hidden)
  soft_grad_chunked = jax.grad(
      lambda x: lm_head_cross_entropy(
          x,
          head_kernel,
          targets,
          mask,
          token_chunk=5,
          vocab_chunk=6,
          train_lm_head=True,
          logit_softcap=softcap,
      )
  )(hidden)
  assert_close("softcap_grad_hidden", soft_grad_chunked, soft_grad_dense)

  jit_loss = jax.jit(make_frozen_lm_head_ce(5, vocab_chunk=7))(
      hidden,
      embedding,
      targets,
      mask,
  )
  assert_close("jit_frozen_loss", jit_loss, expected)
  jit_full_loss = jax.jit(make_lm_head_ce(5, train_lm_head=True, vocab_chunk=7))(
      hidden,
      head_kernel,
      targets,
      mask,
  )
  assert_close("jit_full_loss", jit_full_loss, expected)
  jit_grad_hidden, jit_grad_head = jax.jit(
      jax.grad(
          lambda x, w: lm_head_cross_entropy(
              x,
              w,
              targets,
              mask,
              token_chunk=5,
              vocab_chunk=7,
              train_lm_head=True,
          ),
          argnums=(0, 1),
      )
  )(hidden, head_kernel)
  grad_dense_hidden, grad_dense_head = jax.grad(
      lambda x, w: dense_ce(x, w, targets, mask),
      argnums=(0, 1),
  )(hidden, head_kernel)
  assert_close("jit_full_grad_hidden", jit_grad_hidden, grad_dense_hidden)
  assert_close("jit_full_grad_head", jit_grad_head, grad_dense_head)


if __name__ == "__main__":
  test_chunked_linear_ce_numerics()
