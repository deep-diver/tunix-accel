#!/usr/bin/env python3
"""End-to-end packing parity with a tiny causal LM.

This test is intentionally Gemma-free. It verifies the invariant that matters
before wiring into Tunix/Gemma: packed rows with reset positions and
block-causal masks produce the same loss as running the examples separately.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from tunix_accel.packing import build_block_causal_attention_mask
from tunix_accel.packing import pack_records


def _records():
  return [
      {
          "id": "a",
          "input_ids": [2, 3, 5, 7],
          "labels": [3, 5, 7, -100],
      },
      {
          "id": "b",
          "input_ids": [11, 13],
          "labels": [13, -100],
      },
      {
          "id": "c",
          "input_ids": [17, 19, 23],
          "labels": [19, 23, -100],
      },
      {
          "id": "d",
          "input_ids": [29, 31],
          "labels": [31, -100],
      },
      {
          "id": "e",
          "input_ids": [37, 41, 43],
          "labels": [41, 43, -100],
      },
  ]


def _separate_batch(records, *, max_length: int):
  rows = []
  for record in records:
    length = len(record["input_ids"])
    input_ids = list(record["input_ids"]) + [0] * (max_length - length)
    labels = list(record["labels"]) + [-100] * (max_length - length)
    loss_mask = [label != -100 for label in labels]
    input_mask = [True] * length + [False] * (max_length - length)
    positions = list(range(length)) + [0] * (max_length - length)
    segment_ids = [0] * length + [-1] * (max_length - length)
    rows.append(
        {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "input_mask": input_mask,
            "positions": positions,
            "attention_mask": build_block_causal_attention_mask(
                segment_ids,
                input_mask,
            ),
        }
    )

  return {
      key: np.asarray([row[key] for row in rows])
      for key in rows[0]
  }


def _ordinary_causal_mask(input_mask):
  batch, length = input_mask.shape
  causal = np.tril(np.ones((length, length), dtype=bool))
  valid = input_mask[:, :, None] & input_mask[:, None, :]
  return np.broadcast_to(causal, (batch, length, length)) & valid


def _params(*, vocab_size: int = 64, max_length: int = 6, dim: int = 16):
  key = jax.random.PRNGKey(7)
  keys = jax.random.split(key, 7)
  scale = 0.13
  return {
      "tok": jax.random.normal(keys[0], (vocab_size, dim)) * scale,
      "pos": jax.random.normal(keys[1], (max_length, dim)) * scale,
      "wq": jax.random.normal(keys[2], (dim, dim)) * scale,
      "wk": jax.random.normal(keys[3], (dim, dim)) * scale,
      "wv": jax.random.normal(keys[4], (dim, dim)) * scale,
      "wo": jax.random.normal(keys[5], (dim, dim)) * scale,
      "head": jax.random.normal(keys[6], (dim, vocab_size)) * scale,
  }


def _tiny_causal_lm(params, input_ids, positions, attention_mask):
  x = params["tok"][input_ids] + params["pos"][positions]
  q = jnp.einsum("bld,df->blf", x, params["wq"])
  k = jnp.einsum("bld,df->blf", x, params["wk"])
  v = jnp.einsum("bld,df->blf", x, params["wv"])
  scale = jnp.sqrt(jnp.asarray(q.shape[-1], dtype=jnp.float32))
  scores = jnp.einsum("bqd,bkd->bqk", q, k) / scale
  scores = jnp.where(attention_mask, scores, -1e9)
  attn = jax.nn.softmax(scores, axis=-1)
  h = jnp.einsum("bqk,bkd->bqd", attn, v)
  h = jnp.tanh(jnp.einsum("bld,df->blf", h, params["wo"]) + x)
  return jnp.einsum("bld,dv->blv", h, params["head"])


def _loss(params, batch):
  input_ids = jnp.asarray(batch["input_ids"], dtype=jnp.int32)
  labels = jnp.asarray(batch["labels"], dtype=jnp.int32)
  loss_mask = jnp.asarray(batch["loss_mask"], dtype=bool)
  positions = jnp.asarray(batch["positions"], dtype=jnp.int32)
  attention_mask = jnp.asarray(batch["attention_mask"], dtype=bool)

  logits = _tiny_causal_lm(params, input_ids, positions, attention_mask)
  safe_labels = jnp.where(loss_mask, labels, 0)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  token_log_probs = jnp.take_along_axis(
      log_probs,
      safe_labels[..., None],
      axis=-1,
  )[..., 0]
  mask = loss_mask.astype(jnp.float32)
  return -jnp.sum(token_log_probs * mask) / jnp.sum(mask)


def test_packed_batch_matches_separate_examples_for_tiny_causal_lm():
  records = _records()
  max_length = 6
  params = _params(max_length=max_length)

  separate = _separate_batch(records, max_length=max_length)
  packed = pack_records(
      records,
      max_length=max_length,
      strategy="best_fit_decreasing",
  ).as_numpy()

  separate_loss = _loss(params, separate)
  packed_loss = _loss(params, packed)

  assert packed["input_ids"].shape[0] < separate["input_ids"].shape[0]
  assert jnp.allclose(packed_loss, separate_loss, atol=1e-6, rtol=1e-6), (
      float(packed_loss),
      float(separate_loss),
  )


def test_plain_causal_mask_contaminates_packed_segments():
  records = _records()
  max_length = 6
  params = _params(max_length=max_length)
  packed = pack_records(
      records,
      max_length=max_length,
      strategy="best_fit_decreasing",
  ).as_numpy()

  block_loss = _loss(params, packed)
  contaminated = dict(packed)
  contaminated["attention_mask"] = _ordinary_causal_mask(packed["input_mask"])
  contaminated_loss = _loss(params, contaminated)

  assert abs(float(contaminated_loss - block_loss)) > 1e-5
