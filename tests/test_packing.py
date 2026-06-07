#!/usr/bin/env python3
"""Tests for decoder-LM sequence packing."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from tunix_accel.packing import PackingConfig
from tunix_accel.packing import build_block_causal_attention_mask
from tunix_accel.packing import estimate_packed_efficiency
from tunix_accel.packing import estimate_unpacked_efficiency
from tunix_accel.packing import pack_records


def test_pack_records_resets_positions_and_blocks_attention():
  records = [
      {"id": "a", "input_ids": [10, 11, 12, 13]},
      {"id": "b", "input_ids": [20, 21]},
      {"id": "c", "input_ids": [30, 31, 32]},
      {"id": "d", "input_ids": [40, 41, 42]},
  ]

  packed = pack_records(
      records,
      max_length=6,
      pad_token_id=0,
      strategy="best_fit_decreasing",
  )

  assert packed.batch_size == 2
  assert packed.sequence_length == 6
  assert packed.valid_tokens == 12
  assert packed.packing_efficiency == 1.0

  rows_by_sources = {
      tuple(source for source in row if source is not None): idx
      for idx, row in enumerate(packed.source_ids)
  }
  row_idx = rows_by_sources[("a", "a", "a", "a", "b", "b")]
  assert packed.input_ids[row_idx] == [10, 11, 12, 13, 20, 21]
  assert packed.positions[row_idx] == [0, 1, 2, 3, 0, 1]
  assert packed.segment_ids[row_idx] == [0, 0, 0, 0, 1, 1]

  attention = packed.attention_mask[row_idx]
  assert attention[3][0]
  assert attention[4][3] is False
  assert attention[5][4]
  assert attention[2][4] is False


def test_pack_records_preserves_labels_and_loss_mask():
  packed = pack_records(
      [
          {
              "id": 0,
              "input_ids": [1, 2, 3],
              "labels": [-100, 2, 3],
          },
          {
              "id": 1,
              "input_ids": [4, 5],
              "labels": [4, -100],
              "loss_mask": [True, False],
          },
      ],
      max_length=6,
      strategy="first_fit",
  )

  assert packed.input_ids == [[1, 2, 3, 4, 5, 0]]
  assert packed.labels == [[-100, 2, 3, 4, -100, -100]]
  assert packed.loss_mask == [[False, True, True, True, False, False]]
  assert packed.input_mask == [[True, True, True, True, True, False]]

  tunix_batch = packed.as_tunix()
  assert tunix_batch["input_tokens"].tolist() == packed.input_ids
  assert tunix_batch["input_mask"].tolist() == packed.loss_mask
  assert tunix_batch["valid_mask"].tolist() == packed.input_mask


def test_long_example_policies():
  records = [{"input_ids": [1, 2, 3, 4, 5]}]

  with pytest.raises(ValueError, match="exceeds max_length"):
    pack_records(records, max_length=3)

  truncated = pack_records(
      records,
      max_length=3,
      long_example_policy="truncate",
  )
  assert truncated.input_ids == [[1, 2, 3]]

  split = pack_records(
      records,
      max_length=3,
      long_example_policy="split",
      strategy="first_fit",
  )
  assert split.input_ids == [[1, 2, 3], [4, 5, 0]]
  assert split.positions == [[0, 1, 2], [0, 1, 0]]


def test_build_block_causal_attention_mask():
  mask = build_block_causal_attention_mask(
      segment_ids=[0, 0, 1, 1, -1],
      input_mask=[True, True, True, True, False],
  )
  assert mask == [
      [True, False, False, False, False],
      [True, True, False, False, False],
      [False, False, True, False, False],
      [False, False, True, True, False],
      [False, False, False, False, False],
  ]


def test_efficiency_estimators_show_padding_gain():
  lengths = [8, 2, 2, 2, 7, 1, 1, 1]
  unpacked = estimate_unpacked_efficiency(
      lengths,
      batch_size=4,
      max_length=8,
  )
  packed = estimate_packed_efficiency(lengths, max_length=8)

  assert unpacked == pytest.approx(24 / 60)
  assert packed == pytest.approx(1.0)


def test_config_rejects_invalid_max_length():
  with pytest.raises(ValueError, match="max_length"):
    PackingConfig(max_length=0)
