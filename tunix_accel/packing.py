"""Sequence packing helpers for decoder-LM SFT.

Packing combines multiple short tokenized examples into one fixed-length model
row. It reduces padding waste while preserving correctness by resetting
positions per segment and by building a block-causal attention mask that blocks
attention between packed examples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


PackingStrategy = Literal[
    "first_fit",
    "best_fit",
    "first_fit_decreasing",
    "best_fit_decreasing",
]
LongExamplePolicy = Literal["error", "truncate", "split"]


@dataclass(frozen=True)
class PackingConfig:
  """Configuration for fixed-length decoder-LM sequence packing."""

  max_length: int
  pad_token_id: int = 0
  label_pad_token_id: int = -100
  strategy: PackingStrategy = "best_fit_decreasing"
  long_example_policy: LongExamplePolicy = "error"
  return_attention_mask: bool = True

  def __post_init__(self) -> None:
    if self.max_length <= 0:
      raise ValueError(f"max_length must be positive, got {self.max_length}.")


@dataclass(frozen=True)
class TokenizedExample:
  """A single tokenized SFT example before packing."""

  input_ids: tuple[int, ...]
  labels: tuple[int, ...]
  loss_mask: tuple[bool, ...]
  example_id: int | str | None = None

  @property
  def length(self) -> int:
    return len(self.input_ids)


@dataclass(frozen=True)
class PackedBatch:
  """Packed fixed-shape batch represented as Python lists.

  `attention_mask` is rank-3 `[batch, query, key]` when requested. It is a
  block-causal mask: tokens can attend to earlier tokens in the same segment,
  but never to tokens from another packed example in the same row.
  """

  input_ids: list[list[int]]
  labels: list[list[int]]
  loss_mask: list[list[bool]]
  input_mask: list[list[bool]]
  positions: list[list[int]]
  segment_ids: list[list[int]]
  source_ids: list[list[int | str | None]]
  attention_mask: list[list[list[bool]]] | None
  original_lengths: list[int]
  packed_lengths: list[int]

  @property
  def batch_size(self) -> int:
    return len(self.input_ids)

  @property
  def sequence_length(self) -> int:
    return len(self.input_ids[0]) if self.input_ids else 0

  @property
  def valid_tokens(self) -> int:
    return sum(self.packed_lengths)

  @property
  def capacity_tokens(self) -> int:
    return self.batch_size * self.sequence_length

  @property
  def packing_efficiency(self) -> float:
    if self.capacity_tokens == 0:
      return 0.0
    return self.valid_tokens / self.capacity_tokens

  @property
  def num_source_examples(self) -> int:
    return len(self.original_lengths)

  def as_dict(self) -> dict[str, Any]:
    result: dict[str, Any] = {
        "input_ids": self.input_ids,
        "labels": self.labels,
        "loss_mask": self.loss_mask,
        "input_mask": self.input_mask,
        "positions": self.positions,
        "segment_ids": self.segment_ids,
        "source_ids": self.source_ids,
    }
    if self.attention_mask is not None:
      result["attention_mask"] = self.attention_mask
    return result

  def as_numpy(self) -> dict[str, Any]:
    """Returns the packed batch as NumPy arrays.

    NumPy is intentionally imported lazily so the packing engine itself stays
    dependency-light.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    result = {
        "input_ids": np.asarray(self.input_ids, dtype=np.int32),
        "labels": np.asarray(self.labels, dtype=np.int32),
        "loss_mask": np.asarray(self.loss_mask, dtype=np.bool_),
        "input_mask": np.asarray(self.input_mask, dtype=np.bool_),
        "positions": np.asarray(self.positions, dtype=np.int32),
        "segment_ids": np.asarray(self.segment_ids, dtype=np.int32),
        "source_ids": np.asarray(self.source_ids, dtype=object),
    }
    if self.attention_mask is not None:
      result["attention_mask"] = np.asarray(self.attention_mask, dtype=np.bool_)
    return result


@dataclass
class _Bin:
  segments: list[TokenizedExample]
  capacity: int
  used: int = 0

  @property
  def remaining(self) -> int:
    return self.capacity - self.used


def _tuple_ints(values: Sequence[int], *, name: str) -> tuple[int, ...]:
  try:
    return tuple(int(value) for value in values)
  except TypeError as exc:
    raise TypeError(f"{name} must be a sequence of integers.") from exc


def _tuple_bools(values: Sequence[Any], *, name: str) -> tuple[bool, ...]:
  try:
    return tuple(bool(value) for value in values)
  except TypeError as exc:
    raise TypeError(f"{name} must be a sequence of booleans.") from exc


def _coerce_example(
    example: TokenizedExample | Mapping[str, Any],
    *,
    index: int,
    token_key: str,
    label_key: str,
    loss_mask_key: str,
    label_pad_token_id: int,
) -> TokenizedExample:
  if isinstance(example, TokenizedExample):
    return example
  if not isinstance(example, Mapping):
    raise TypeError(
        "Examples must be TokenizedExample instances or mapping records."
    )
  if token_key not in example:
    raise KeyError(f"Missing token key {token_key!r} in example {index}.")

  input_ids = _tuple_ints(example[token_key], name=token_key)
  labels_raw = example.get(label_key, input_ids)
  labels = _tuple_ints(labels_raw, name=label_key)
  if loss_mask_key in example:
    loss_mask = _tuple_bools(example[loss_mask_key], name=loss_mask_key)
  else:
    loss_mask = tuple(label != label_pad_token_id for label in labels)
  example_id = example.get("id", index)
  coerced = TokenizedExample(
      input_ids=input_ids,
      labels=labels,
      loss_mask=loss_mask,
      example_id=example_id,
  )
  _validate_example(coerced, index=index)
  return coerced


def _validate_example(example: TokenizedExample, *, index: int) -> None:
  length = example.length
  if length == 0:
    raise ValueError(f"Example {index} is empty.")
  if len(example.labels) != length:
    raise ValueError(
        f"Example {index} has {length} input ids but "
        f"{len(example.labels)} labels."
    )
  if len(example.loss_mask) != length:
    raise ValueError(
        f"Example {index} has {length} input ids but "
        f"{len(example.loss_mask)} loss-mask values."
    )


def _handle_long_example(
    example: TokenizedExample,
    *,
    max_length: int,
    policy: LongExamplePolicy,
) -> list[TokenizedExample]:
  if example.length <= max_length:
    return [example]
  if policy == "error":
    raise ValueError(
        f"Example {example.example_id!r} length {example.length} exceeds "
        f"max_length {max_length}. Use long_example_policy='truncate' or "
        "'split' if this is intended."
    )
  if policy == "truncate":
    return [
        TokenizedExample(
            input_ids=example.input_ids[:max_length],
            labels=example.labels[:max_length],
            loss_mask=example.loss_mask[:max_length],
            example_id=example.example_id,
        )
    ]
  if policy == "split":
    chunks = []
    for start in range(0, example.length, max_length):
      end = min(start + max_length, example.length)
      chunks.append(
          TokenizedExample(
              input_ids=example.input_ids[start:end],
              labels=example.labels[start:end],
              loss_mask=example.loss_mask[start:end],
              example_id=example.example_id,
          )
      )
    return chunks
  raise ValueError(f"Unknown long_example_policy {policy!r}.")


def _materialize_examples(
    examples: Sequence[TokenizedExample | Mapping[str, Any]],
    *,
    config: PackingConfig,
    token_key: str,
    label_key: str,
    loss_mask_key: str,
) -> list[TokenizedExample]:
  materialized: list[TokenizedExample] = []
  for index, raw_example in enumerate(examples):
    example = _coerce_example(
        raw_example,
        index=index,
        token_key=token_key,
        label_key=label_key,
        loss_mask_key=loss_mask_key,
        label_pad_token_id=config.label_pad_token_id,
    )
    materialized.extend(
        _handle_long_example(
            example,
            max_length=config.max_length,
            policy=config.long_example_policy,
        )
    )
  return materialized


def _ordered_for_strategy(
    examples: list[TokenizedExample],
    strategy: PackingStrategy,
) -> list[TokenizedExample]:
  if strategy in {"first_fit_decreasing", "best_fit_decreasing"}:
    return sorted(
        examples,
        key=lambda example: (-example.length, str(example.example_id)),
    )
  return list(examples)


def _place_example(
    bins: list[_Bin],
    example: TokenizedExample,
    strategy: PackingStrategy,
    *,
    max_length: int,
) -> None:
  candidates = [
      (idx, bin_)
      for idx, bin_ in enumerate(bins)
      if bin_.used + example.length <= bin_.capacity
  ]
  if not candidates:
    bins.append(_Bin(segments=[example], capacity=max_length, used=example.length))
    return

  if strategy in {"best_fit", "best_fit_decreasing"}:
    idx, target = min(
        candidates,
        key=lambda item: item[1].capacity - (item[1].used + example.length),
    )
  else:
    idx, target = candidates[0]
  bins[idx] = target
  target.segments.append(example)
  target.used += example.length


def _pack_bins(
    examples: list[TokenizedExample],
    config: PackingConfig,
) -> list[_Bin]:
  bins: list[_Bin] = []
  for example in _ordered_for_strategy(examples, config.strategy):
    _place_example(
        bins,
        example,
        config.strategy,
        max_length=config.max_length,
    )
  return bins


def build_block_causal_attention_mask(
    segment_ids: Sequence[int],
    input_mask: Sequence[bool],
) -> list[list[bool]]:
  """Builds a block-causal mask for one packed row."""
  if len(segment_ids) != len(input_mask):
    raise ValueError("segment_ids and input_mask must have the same length.")

  length = len(segment_ids)
  attention_mask: list[list[bool]] = []
  for query in range(length):
    row = []
    for key in range(length):
      same_segment = segment_ids[query] == segment_ids[key]
      causal = key <= query
      valid = input_mask[query] and input_mask[key] and segment_ids[query] >= 0
      row.append(bool(valid and same_segment and causal))
    attention_mask.append(row)
  return attention_mask


def pack_examples(
    examples: Sequence[TokenizedExample | Mapping[str, Any]],
    config: PackingConfig,
    *,
    token_key: str = "input_ids",
    label_key: str = "labels",
    loss_mask_key: str = "loss_mask",
) -> PackedBatch:
  """Packs tokenized examples into fixed-length decoder-LM rows.

  Mapping examples are expected to contain `input_ids` by default. `labels` and
  `loss_mask` are optional; labels default to input ids, and loss mask defaults
  to labels that are not equal to `label_pad_token_id`.
  """
  materialized = _materialize_examples(
      examples,
      config=config,
      token_key=token_key,
      label_key=label_key,
      loss_mask_key=loss_mask_key,
  )
  bins = _pack_bins(materialized, config)

  batch_input_ids: list[list[int]] = []
  batch_labels: list[list[int]] = []
  batch_loss_mask: list[list[bool]] = []
  batch_input_mask: list[list[bool]] = []
  batch_positions: list[list[int]] = []
  batch_segment_ids: list[list[int]] = []
  batch_source_ids: list[list[int | str | None]] = []
  batch_attention_mask: list[list[list[bool]]] | None = (
      [] if config.return_attention_mask else None
  )
  packed_lengths: list[int] = []

  for bin_ in bins:
    input_ids = [config.pad_token_id] * config.max_length
    labels = [config.label_pad_token_id] * config.max_length
    loss_mask = [False] * config.max_length
    input_mask = [False] * config.max_length
    positions = [0] * config.max_length
    segment_ids = [-1] * config.max_length
    source_ids: list[int | str | None] = [None] * config.max_length

    cursor = 0
    for segment_idx, segment in enumerate(bin_.segments):
      end = cursor + segment.length
      input_ids[cursor:end] = list(segment.input_ids)
      labels[cursor:end] = list(segment.labels)
      loss_mask[cursor:end] = list(segment.loss_mask)
      input_mask[cursor:end] = [True] * segment.length
      positions[cursor:end] = list(range(segment.length))
      segment_ids[cursor:end] = [segment_idx] * segment.length
      source_ids[cursor:end] = [segment.example_id] * segment.length
      cursor = end

    batch_input_ids.append(input_ids)
    batch_labels.append(labels)
    batch_loss_mask.append(loss_mask)
    batch_input_mask.append(input_mask)
    batch_positions.append(positions)
    batch_segment_ids.append(segment_ids)
    batch_source_ids.append(source_ids)
    packed_lengths.append(cursor)
    if batch_attention_mask is not None:
      batch_attention_mask.append(
          build_block_causal_attention_mask(segment_ids, input_mask)
      )

  return PackedBatch(
      input_ids=batch_input_ids,
      labels=batch_labels,
      loss_mask=batch_loss_mask,
      input_mask=batch_input_mask,
      positions=batch_positions,
      segment_ids=batch_segment_ids,
      source_ids=batch_source_ids,
      attention_mask=batch_attention_mask,
      original_lengths=[example.length for example in materialized],
      packed_lengths=packed_lengths,
  )


def pack_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    pad_token_id: int = 0,
    label_pad_token_id: int = -100,
    strategy: PackingStrategy = "best_fit_decreasing",
    long_example_policy: LongExamplePolicy = "error",
    token_key: str = "input_ids",
    label_key: str = "labels",
    loss_mask_key: str = "loss_mask",
    return_attention_mask: bool = True,
) -> PackedBatch:
  """Convenience wrapper for dict-style tokenized datasets."""
  return pack_examples(
      records,
      PackingConfig(
          max_length=max_length,
          pad_token_id=pad_token_id,
          label_pad_token_id=label_pad_token_id,
          strategy=strategy,
          long_example_policy=long_example_policy,
          return_attention_mask=return_attention_mask,
      ),
      token_key=token_key,
      label_key=label_key,
      loss_mask_key=loss_mask_key,
  )


def estimate_unpacked_efficiency(
    lengths: Sequence[int],
    *,
    batch_size: int,
    max_length: int,
) -> float:
  """Estimates valid-token ratio for ordinary padded batching.

  Batches are formed in the provided order. Each batch pads to the longest
  example in that batch, capped by `max_length`.
  """
  if batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size}.")
  if max_length <= 0:
    raise ValueError(f"max_length must be positive, got {max_length}.")
  if not lengths:
    return 0.0

  valid = 0
  capacity = 0
  for start in range(0, len(lengths), batch_size):
    batch_lengths = [
        min(int(length), max_length)
        for length in lengths[start : start + batch_size]
    ]
    padded_length = max(batch_lengths)
    valid += sum(batch_lengths)
    capacity += len(batch_lengths) * padded_length
  return valid / capacity if capacity else 0.0


def estimate_packed_efficiency(
    lengths: Sequence[int],
    *,
    max_length: int,
    strategy: PackingStrategy = "best_fit_decreasing",
) -> float:
  """Estimates valid-token ratio after fixed-row packing."""
  examples = [
      TokenizedExample(
          input_ids=tuple([1] * int(length)),
          labels=tuple([1] * int(length)),
          loss_mask=tuple([True] * int(length)),
          example_id=index,
      )
      for index, length in enumerate(lengths)
      if int(length) > 0
  ]
  packed = pack_examples(
      examples,
      PackingConfig(
          max_length=max_length,
          strategy=strategy,
          return_attention_mask=False,
      ),
  )
  return packed.packing_efficiency
