#!/usr/bin/env python3
"""Tests for Tunix-facing sequence packing adapters."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tunix_accel.tunix_packing import TunixPackingConfig
from tunix_accel.tunix_packing import pack_tunix_batches


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

