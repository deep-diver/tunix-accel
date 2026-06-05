#!/usr/bin/env python3
"""Tests for Tunix-facing sequence packing adapters."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tunix_accel.tunix_packing import patch_trainer_api
from tunix_accel.tunix_packing import TunixPackingConfig
from tunix_accel.tunix_packing import pack_tunix_batches
from tunix_accel.tunix_packing import restore_trainer_api


def test_pack_tunix_batches_infers_shape_and_preserves_loss_mask():
  source_batch = {
      "input_tokens": np.asarray(
          [
              [10, 11, 12, 0, 0, 0],
              [20, 21, 0, 0, 0, 0],
              [30, 31, 32, 0, 0, 0],
              [40, 41, 0, 0, 0, 0],
          ],
          dtype=np.int32,
      ),
      "input_mask": np.asarray(
          [
              [False, True, True, False, False, False],
              [False, True, False, False, False, False],
              [False, True, True, False, False, False],
              [False, True, False, False, False, False],
          ],
          dtype=bool,
      ),
  }

  packed_batches = list(
      pack_tunix_batches(
          [source_batch],
          TunixPackingConfig(batch_size=1, max_length=6),
      )
  )

  assert len(packed_batches) == 2
  merged = np.concatenate([batch["input_tokens"] for batch in packed_batches])
  merged_loss = np.concatenate([batch["input_mask"] for batch in packed_batches])
  assert merged.shape == (2, 6)
  assert int((merged != 0).sum()) == 10
  assert int(merged_loss.sum()) == 6
  assert {"valid_mask", "positions", "segment_ids", "attention_mask"}.issubset(
      packed_batches[0]
  )


def test_pack_tunix_batches_uses_valid_mask_for_pad_like_tokens():
  source_batch = {
      "input_tokens": np.asarray([[0, 7, 0, 0]], dtype=np.int32),
      "valid_mask": np.asarray([[True, True, True, False]], dtype=bool),
      "input_mask": np.asarray([[False, True, True, False]], dtype=bool),
  }

  [packed] = list(
      pack_tunix_batches(
          [source_batch],
          TunixPackingConfig(batch_size=1, max_length=4, pad_token_id=0),
      )
  )

  assert packed["input_tokens"].tolist() == [[0, 7, 0, 0]]
  assert packed["valid_mask"].tolist() == [[True, True, True, False]]
  assert packed["input_mask"].tolist() == [[False, True, True, False]]


def test_packing_kwarg_wraps_trainer_dataset_and_input_fn():
  class FakeTrainer:
    def __init__(self):
      self.gen_model_input_fn = None
      self.seen_batches = None

    def clear_jit_cache(self):
      return None

    def with_gen_model_input_fn(self, gen_model_input_fn):
      self.clear_jit_cache()
      self.gen_model_input_fn = gen_model_input_fn
      return self

    def train(self, train_ds, eval_ds=None, skip_jit=False, *, cache_nnx_graph=True):
      del eval_ds, skip_jit, cache_nnx_graph
      self.seen_batches = list(train_ds)
      return "trained"

  source_batch = {
      "input_tokens": np.asarray(
          [
              [10, 11, 0, 0, 0, 0],
              [20, 21, 0, 0, 0, 0],
          ],
          dtype=np.int32,
      ),
      "input_mask": np.asarray(
          [
              [False, True, False, False, False, False],
              [False, True, False, False, False, False],
          ],
          dtype=bool,
      ),
  }

  patch_trainer_api(SimpleNamespace(PeftTrainer=FakeTrainer))
  try:
    trainer = FakeTrainer().with_gen_model_input_fn(
        lambda batch: {
            "kept_extra_field": np.asarray(batch["input_tokens"]) + 1,
            "attention_mask": "should be overwritten",
        },
        packing=TunixPackingConfig(batch_size=1, max_length=6),
    )

    assert trainer.train([source_batch]) == "trained"
    assert trainer.seen_batches is not None
    assert len(trainer.seen_batches) == 1

    batch = trainer.seen_batches[0]
    model_inputs = trainer.gen_model_input_fn(batch)
    assert model_inputs["input_tokens"].shape == (1, 6)
    assert model_inputs["kept_extra_field"].shape == (1, 6)
    assert model_inputs["attention_mask"].shape == (1, 6, 6)
    assert model_inputs["input_tokens"].tolist()[0][:4] == [10, 11, 20, 21]
  finally:
    restore_trainer_api()
