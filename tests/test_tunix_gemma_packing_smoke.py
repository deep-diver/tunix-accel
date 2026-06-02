#!/usr/bin/env python3
"""Optional Gemma/Tunix smoke test for packed SFT batches."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import numpy as np
import pytest

from tunix_accel.packing import build_block_causal_attention_mask
from tunix_accel.packing import pack_records


def _records():
  return [
      {
          "id": "a",
          "input_ids": [2, 3, 5, 7],
          "loss_mask": [False, True, True, True],
      },
      {"id": "b", "input_ids": [11, 13], "loss_mask": [False, True]},
      {"id": "c", "input_ids": [17, 19, 23], "loss_mask": [False, True, True]},
      {"id": "d", "input_ids": [29, 31], "loss_mask": [False, True]},
      {"id": "e", "input_ids": [37, 41, 43], "loss_mask": [False, True, True]},
  ]


def _tiny_gemma(nnx, gemma3_model):
  config = gemma3_model.ModelConfig(
      num_layers=1,
      num_embed=64,
      embed_dim=32,
      hidden_dim=64,
      num_heads=4,
      head_dim=8,
      num_kv_heads=1,
      sliding_window_size=8,
      param_dtype=jnp.float32,
  )
  return gemma3_model.Gemma3(config, rngs=nnx.Rngs(0))


def _mask_rank_like_tunix(attention_mask, valid_mask, utils):
  reference = utils.make_causal_attn_mask(jnp.asarray(valid_mask, dtype=bool))
  attention_mask = jnp.asarray(attention_mask, dtype=bool)
  if attention_mask.ndim == reference.ndim:
    return attention_mask
  if attention_mask.ndim + 1 == reference.ndim:
    return attention_mask[:, None, :, :]
  raise AssertionError(
      f"Unexpected attention mask rank {attention_mask.ndim}; "
      f"Tunix reference rank is {reference.ndim}."
  )


def _separate_tunix_batch(records, *, max_length: int, utils):
  rows = []
  for record in records:
    length = len(record["input_ids"])
    valid_mask = [True] * length + [False] * (max_length - length)
    loss_mask = list(record["loss_mask"]) + [False] * (max_length - length)
    segment_ids = [0] * length + [-1] * (max_length - length)
    rows.append(
        {
            "input_tokens": list(record["input_ids"]) + [0] * (max_length - length),
            "input_mask": loss_mask,
            "valid_mask": valid_mask,
            "positions": list(range(length)) + [0] * (max_length - length),
            "attention_mask": build_block_causal_attention_mask(
                segment_ids,
                valid_mask,
            ),
        }
    )

  batch = {
      key: np.asarray([row[key] for row in rows])
      for key in rows[0]
  }
  batch["attention_mask"] = _mask_rank_like_tunix(
      batch["attention_mask"],
      batch["valid_mask"],
      utils,
  )
  return batch


def _packed_tunix_batch(records, *, max_length: int, utils):
  packed = pack_records(
      records,
      max_length=max_length,
      strategy="best_fit_decreasing",
  ).as_tunix()
  packed["attention_mask"] = _mask_rank_like_tunix(
      packed["attention_mask"],
      packed["valid_mask"],
      utils,
  )
  return packed


def _default_loss(model, batch, peft_trainer):
  return peft_trainer._default_loss_fn(  # pylint: disable=protected-access
      model,
      jnp.asarray(batch["input_tokens"], dtype=jnp.int32),
      jnp.asarray(batch["input_mask"], dtype=bool),
      jnp.asarray(batch["positions"], dtype=jnp.int32),
      jnp.asarray(batch["attention_mask"], dtype=bool),
  )


def test_tiny_gemma_default_loss_matches_for_packed_batch():
  nnx = pytest.importorskip("flax.nnx", exc_type=ImportError)
  gemma3_model = pytest.importorskip(
      "tunix.models.gemma3.model",
      exc_type=ImportError,
  )
  peft_trainer = pytest.importorskip(
      "tunix.sft.peft_trainer",
      exc_type=ImportError,
  )
  utils = pytest.importorskip("tunix.sft.utils", exc_type=ImportError)

  records = _records()
  max_length = 6
  model = _tiny_gemma(nnx, gemma3_model)

  separate = _separate_tunix_batch(records, max_length=max_length, utils=utils)
  packed = _packed_tunix_batch(records, max_length=max_length, utils=utils)

  assert packed["input_tokens"].shape[0] < separate["input_tokens"].shape[0]
  assert jnp.allclose(
      _default_loss(model, packed, peft_trainer),
      _default_loss(model, separate, peft_trainer),
      atol=1e-5,
      rtol=1e-5,
  )


def test_tiny_gemma_cce_loss_matches_for_packed_batch():
  nnx = pytest.importorskip("flax.nnx", exc_type=ImportError)
  gemma3_model = pytest.importorskip(
      "tunix.models.gemma3.model",
      exc_type=ImportError,
  )
  peft_trainer = pytest.importorskip(
      "tunix.sft.peft_trainer",
      exc_type=ImportError,
  )
  utils = pytest.importorskip("tunix.sft.utils", exc_type=ImportError)
  tunix_patch = pytest.importorskip(
      "tunix_accel.tunix_patch",
      exc_type=ImportError,
  )

  records = _records()
  max_length = 6
  model = _tiny_gemma(nnx, gemma3_model)
  separate = _separate_tunix_batch(records, max_length=max_length, utils=utils)
  packed = _packed_tunix_batch(records, max_length=max_length, utils=utils)

  tunix_patch.install(token_chunk=2, vocab_chunk=32)
  try:
    assert jnp.allclose(
        _default_loss(model, packed, peft_trainer),
        _default_loss(model, separate, peft_trainer),
        atol=1e-5,
        rtol=1e-5,
    )
  finally:
    tunix_patch.uninstall()
