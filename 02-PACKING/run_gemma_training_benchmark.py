#!/usr/bin/env python3
"""Actual Tunix/Gemma training benchmark for sequence packing.

Set TUNIX_ACCEL_DISABLE_AUTOPATCH=true before launching Python so this benchmark
stays focused on sequence packing and does not pick up repository-level
autopatches at interpreter startup.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.metadata as importlib_metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Iterable, Iterator

os.environ.setdefault("TUNIX_ACCEL_DISABLE_AUTOPATCH", "true")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from tunix_accel.packing import pack_records  # pylint: disable=wrong-import-position


GEMMA3_270M_IT_MODEL_ID = "google/gemma-3-270m-it"
GEMMA3_270M_IT_GCS = "gs://gemma-data/checkpoints/gemma3-270m-it"
GEMMA3_TOKENIZER_GCS = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

INPUT_TEMPLATE_IT = {
    "prefix": "<start_of_turn>user\nTranslate this into French:\n",
    "suffix": "\n<end_of_turn>\n<start_of_turn>model\n",
}


@dataclass(frozen=True)
class TokenizedSftDataset:
  name: str
  model_id: str
  tokenizer_source: str
  pad_token_id: int
  records: list[dict[str, Any]]
  source: str


@dataclass(frozen=True)
class RawTranslationDataset:
  name: str
  split: str
  examples: list[dict[str, str]]


@dataclass(frozen=True)
class TokenizerBundle:
  tokenizer: Any
  encode: Any
  pad_id: int
  eos_id: int


@dataclass(frozen=True)
class PreparedVariant:
  name: str
  batches: list[dict[str, np.ndarray]]
  batch_metrics: list[dict[str, float | int]]
  source_examples: int
  dropped_overlength: int
  packing_summary: dict[str, float | int | str]


def parse_variants(value: str) -> list[str]:
  variants = [item.strip() for item in value.split(",") if item.strip()]
  allowed = {"unpacked", "packed"}
  unknown = sorted(set(variants) - allowed)
  if unknown:
    raise ValueError(f"Unknown variants: {unknown}. Allowed: {sorted(allowed)}")
  return variants


def infer_model_name(model_id: str) -> str:
  return model_id.split("/")[-1].lower()


def translation_pair(row: dict[str, Any]) -> tuple[str, str]:
  translation = row["translation"]
  return translation["en"], translation["fr"]


def configure_autopatch(*, allow_autopatch: bool) -> None:
  disabled = os.environ.get("TUNIX_ACCEL_DISABLE_AUTOPATCH", "").lower() in {
      "1",
      "true",
      "yes",
      "on",
  }
  if allow_autopatch:
    if not disabled:
      from tunix_accel import autopatch  # pylint: disable=import-outside-toplevel

      autopatch.enable()
    return

  if not disabled:
    raise RuntimeError(
        "This benchmark should run with repository autopatches disabled. Launch "
        "with TUNIX_ACCEL_DISABLE_AUTOPATCH=true, or pass --allow-autopatch for "
        "explicit patch-comparison runs."
    )


def collect_accel_status() -> dict[str, Any]:
  """Returns lightweight patch-status instrumentation for benchmark summaries."""
  status: dict[str, Any] = {
      "disable_autopatch_env": os.environ.get(
          "TUNIX_ACCEL_DISABLE_AUTOPATCH",
          "",
      ),
      "disable_ce_env": os.environ.get("TUNIX_ACCEL_DISABLE_CE", ""),
      "disable_tiled_mlp_env": os.environ.get(
          "TUNIX_ACCEL_DISABLE_TILED_MLP",
          "",
      ),
      "disable_activation_policy_env": os.environ.get(
          "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY",
          "",
      ),
      "activation_policy_env": os.environ.get(
          "TUNIX_ACCEL_ACTIVATION_POLICY",
          "",
      ),
      "enable_splash_attention_env": os.environ.get(
          "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION",
          "",
      ),
      "enable_gemma4_hf_loader_env": os.environ.get(
          "TUNIX_ACCEL_ENABLE_GEMMA4_HF_LOADER",
          "",
      ),
  }
  modules = {
      "cce_installed": "tunix_accel.tunix_patch",
      "gemma3_tiled_mlp_installed": "tunix_accel.gemma3_tiled_mlp",
      "gemma3_activation_policy_installed": (
          "tunix_accel.gemma3_activation_policy"
      ),
      "gemma3_splash_attention_installed": (
          "tunix_accel.gemma3_splash_attention"
      ),
      "gemma4_tiled_mlp_installed": "tunix_accel.gemma4_tiled_mlp",
      "gemma4_activation_policy_installed": (
          "tunix_accel.gemma4_activation_policy"
      ),
  }
  for key, module_name in modules.items():
    try:
      module = __import__(module_name, fromlist=["is_installed"])
      is_installed = getattr(module, "is_installed", None)
      status[key] = bool(is_installed()) if callable(is_installed) else False
    except Exception as exc:  # pylint: disable=broad-exception-caught
      status[key] = False
      status[f"{key}_error"] = type(exc).__name__
  return status


def load_tokenizer(args: argparse.Namespace) -> TokenizerBundle:
  if args.tokenizer_source == "sentencepiece":
    from tunix.models.gemma3 import params as gemma3_params  # pylint: disable=import-outside-toplevel

    tokenizer = gemma3_params.create_tokenizer(args.tokenizer_path)
    pad_id = int(tokenizer.pad_id())
    if pad_id < 0:
      pad_id = 0
    eos_id = int(tokenizer.eos_id())

    def encode(text: str) -> list[int]:
      return [int(x) for x in tokenizer.EncodeAsIds(text)]

    return TokenizerBundle(
        tokenizer=tokenizer,
        encode=encode,
        pad_id=pad_id,
        eos_id=eos_id,
    )

  if args.tokenizer_source == "huggingface":
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer_kwargs: dict[str, Any] = {}
    if "gemma-4" in args.model_id.lower():
      tokenizer_kwargs["extra_special_tokens"] = {}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=not args.allow_download,
        **tokenizer_kwargs,
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
      pad_id = 0
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
      raise ValueError("The Hugging Face tokenizer has no eos_token_id.")

    def encode(text: str) -> list[int]:
      return [int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]]

    return TokenizerBundle(
        tokenizer=tokenizer,
        encode=encode,
        pad_id=int(pad_id),
        eos_id=int(eos_id),
    )

  raise ValueError(f"Unsupported tokenizer source: {args.tokenizer_source!r}")


def tokenize_sft_record(
    *,
    encode,
    eos_id: int,
    source: str,
    target: str,
    example_id: int,
    prompt_text: str | None = None,
) -> dict[str, Any]:
  if prompt_text is None:
    prompt_text = (
        INPUT_TEMPLATE_IT["prefix"] + source + INPUT_TEMPLATE_IT["suffix"]
    )
  prompt_ids = encode(prompt_text)
  answer_ids = encode(target) + [eos_id]
  input_ids = prompt_ids + answer_ids
  loss_mask = [False] * len(prompt_ids) + [True] * len(answer_ids)
  return {
      "id": example_id,
      "input_ids": input_ids,
      "labels": input_ids,
      "loss_mask": loss_mask,
      "source_chars": len(source),
      "target_chars": len(target),
      "prompt_tokens": len(prompt_ids),
      "answer_tokens": len(answer_ids),
  }


def load_opus100_records(
    args: argparse.Namespace,
    tokenizer_bundle: TokenizerBundle | None = None,
) -> TokenizedSftDataset:
  if args.dataset_mode == "synthetic":
    return load_synthetic_records(args, tokenizer_bundle)
  if args.dataset_mode == "alpaca":
    return load_alpaca_records(args, tokenizer_bundle)
  if args.dataset_mode == "oasst1":
    return load_oasst1_records(args, tokenizer_bundle)
  if args.dataset_mode == "cnn_dailymail":
    return load_cnn_dailymail_records(args, tokenizer_bundle)

  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  tokenizer_bundle = tokenizer_bundle or load_tokenizer(args)
  dataset = load_dataset(
      "Helsinki-NLP/opus-100",
      "en-fr",
      split=f"train[:{args.num_examples}]",
  )
  records = []
  for idx, row in enumerate(dataset):
    source, target = translation_pair(row)
    records.append(
        tokenize_sft_record(
            encode=tokenizer_bundle.encode,
            eos_id=tokenizer_bundle.eos_id,
            source=source,
            target=target,
            example_id=idx,
        )
    )

  return TokenizedSftDataset(
      name="opus100-en-fr-gemma3-it",
      model_id=args.model_id,
      tokenizer_source=args.tokenizer_source,
      pad_token_id=tokenizer_bundle.pad_id,
      records=records,
      source=(
          "Helsinki-NLP/opus-100 en-fr train split, Tunix Gemma3 IT prompt "
          "wrapper, target-only loss mask, target EOS"
      ),
  )


def gemma_instruction_prompt(text: str) -> str:
  return (
      "<start_of_turn>user\n"
      f"{text.strip()}\n"
      "<end_of_turn>\n"
      "<start_of_turn>model\n"
  )


def load_alpaca_records(
    args: argparse.Namespace,
    tokenizer_bundle: TokenizerBundle | None = None,
) -> TokenizedSftDataset:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  tokenizer_bundle = tokenizer_bundle or load_tokenizer(args)
  dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{args.num_examples}]")
  records = []
  for idx, row in enumerate(dataset):
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if input_text:
      source = f"Instruction:\n{instruction}\n\nInput:\n{input_text}"
    else:
      source = f"Instruction:\n{instruction}"
    target = str(row.get("output", "")).strip()
    records.append(
        tokenize_sft_record(
            encode=tokenizer_bundle.encode,
            eos_id=tokenizer_bundle.eos_id,
            source=source,
            target=target,
            prompt_text=gemma_instruction_prompt(source),
            example_id=idx,
        )
    )
  return TokenizedSftDataset(
      name="alpaca-gemma3-it",
      model_id=args.model_id,
      tokenizer_source=args.tokenizer_source,
      pad_token_id=tokenizer_bundle.pad_id,
      records=records,
      source=(
          "tatsu-lab/alpaca train split, Gemma3 IT instruction wrapper, "
          "target-only loss mask, target EOS"
      ),
  )


def load_oasst1_records(
    args: argparse.Namespace,
    tokenizer_bundle: TokenizerBundle | None = None,
) -> TokenizedSftDataset:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  tokenizer_bundle = tokenizer_bundle or load_tokenizer(args)
  dataset = load_dataset("OpenAssistant/oasst1", split="train")
  rows = [row for row in dataset if str(row.get("lang", "")).lower() == "en"]
  by_id = {row.get("message_id"): row for row in rows}
  records = []
  for row in rows:
    if row.get("role") != "assistant":
      continue
    parent = by_id.get(row.get("parent_id"))
    if not parent or parent.get("role") not in {"prompter", "user"}:
      continue
    source = str(parent.get("text", "")).strip()
    target = str(row.get("text", "")).strip()
    if not source or not target:
      continue
    records.append(
        tokenize_sft_record(
            encode=tokenizer_bundle.encode,
            eos_id=tokenizer_bundle.eos_id,
            source=source,
            target=target,
            prompt_text=gemma_instruction_prompt(source),
            example_id=len(records),
        )
    )
    if len(records) >= args.num_examples:
      break
  return TokenizedSftDataset(
      name="oasst1-en-assistant-gemma3-it",
      model_id=args.model_id,
      tokenizer_source=args.tokenizer_source,
      pad_token_id=tokenizer_bundle.pad_id,
      records=records,
      source=(
          "OpenAssistant/oasst1 English assistant replies with immediate "
          "prompter parent as prompt, Gemma3 IT wrapper, target-only loss"
      ),
  )


def load_cnn_dailymail_records(
    args: argparse.Namespace,
    tokenizer_bundle: TokenizerBundle | None = None,
) -> TokenizedSftDataset:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  tokenizer_bundle = tokenizer_bundle or load_tokenizer(args)
  dataset = load_dataset(
      "cnn_dailymail",
      "3.0.0",
      split=f"train[:{args.num_examples}]",
  )
  records = []
  for idx, row in enumerate(dataset):
    source = (
        "Summarize the following article:\n\n"
        + str(row.get("article", "")).strip()
    )
    target = str(row.get("highlights", "")).strip()
    records.append(
        tokenize_sft_record(
            encode=tokenizer_bundle.encode,
            eos_id=tokenizer_bundle.eos_id,
            source=source,
            target=target,
            prompt_text=gemma_instruction_prompt(source),
            example_id=idx,
        )
    )
  return TokenizedSftDataset(
      name="cnn-dailymail-summary-gemma3-it",
      model_id=args.model_id,
      tokenizer_source=args.tokenizer_source,
      pad_token_id=tokenizer_bundle.pad_id,
      records=records,
      source=(
          "cnn_dailymail 3.0.0 train split, article summarization prompt, "
          "Gemma3 IT wrapper, target-only loss mask"
      ),
  )


def load_synthetic_records(
    args: argparse.Namespace,
    tokenizer_bundle: TokenizerBundle | None = None,
) -> TokenizedSftDataset:
  """Creates deterministic EN-FR shaped records without network access."""
  tokenizer_bundle = tokenizer_bundle or load_tokenizer(args)
  records = []
  for idx in range(args.num_examples):
    source = (
        f"synthetic source sentence {idx}. "
        + "The quick training probe uses repeated neutral text. " * 4
    )
    target = (
        f"phrase cible synthetique {idx}. "
        + "bonjour monde entrainement verification. " * 6
    )
    records.append(
        tokenize_sft_record(
            encode=tokenizer_bundle.encode,
            eos_id=tokenizer_bundle.eos_id,
            source=source,
            target=target,
            example_id=idx,
        )
    )
  return TokenizedSftDataset(
      name="synthetic-en-fr-shaped",
      model_id=args.model_id,
      tokenizer_source=args.tokenizer_source,
      pad_token_id=tokenizer_bundle.pad_id,
      records=records,
      source=(
          "Deterministic synthetic EN-FR shaped text, Tunix Gemma prompt "
          "wrapper, target-only loss mask, target EOS"
      ),
  )


def load_raw_translation_examples(
    *,
    split: str,
    num_examples: int,
) -> RawTranslationDataset:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  dataset = load_dataset(
      "Helsinki-NLP/opus-100",
      "en-fr",
      split=f"{split}[:{num_examples}]",
  )
  examples = []
  for row in dataset:
    source, target = translation_pair(row)
    examples.append({"source": source, "reference": target})
  return RawTranslationDataset(
      name="opus100-en-fr",
      split=split,
      examples=examples,
  )


def tokenize_raw_examples(
    examples: RawTranslationDataset,
    *,
    encode,
    eos_id: int,
) -> list[dict[str, Any]]:
  return [
      tokenize_sft_record(
          encode=encode,
          eos_id=eos_id,
          source=example["source"],
          target=example["reference"],
          example_id=idx,
      )
      for idx, example in enumerate(examples.examples)
  ]


def select_generation_examples(
    examples: RawTranslationDataset,
    *,
    encode,
    max_prompt_length: int,
    num_examples: int,
) -> RawTranslationDataset:
  selected = []
  for example in examples.examples:
    prompt_length = len(encode(gemma_generation_prompt(example["source"])))
    if prompt_length > max_prompt_length:
      continue
    selected.append({
        "source": example["source"],
        "reference": example["reference"],
        "prompt_tokens": str(prompt_length),
    })
    if len(selected) >= num_examples:
      break
  return RawTranslationDataset(
      name=examples.name,
      split=examples.split,
      examples=selected,
  )


def filter_overlength(
    records: list[dict[str, Any]],
    *,
    max_length: int,
    policy: str = "drop",
) -> tuple[list[dict[str, Any]], int]:
  if policy == "drop":
    kept = [record for record in records if len(record["input_ids"]) <= max_length]
    return kept, len(records) - len(kept)
  if policy == "truncate":
    kept = []
    changed = 0
    for record in records:
      if len(record["input_ids"]) > max_length:
        changed += 1
      kept.append({
          **record,
          "input_ids": record["input_ids"][:max_length],
          "labels": record["labels"][:max_length],
          "loss_mask": record["loss_mask"][:max_length],
      })
    return kept, changed
  raise ValueError(f"Unknown long example policy: {policy!r}")


def make_unpacked_batches(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    pad_token_id: int,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, float | int]]]:
  batches: list[dict[str, np.ndarray]] = []
  metrics: list[dict[str, float | int]] = []
  usable = (len(records) // batch_size) * batch_size
  for start in range(0, usable, batch_size):
    rows = records[start : start + batch_size]
    input_tokens = np.full((batch_size, max_length), pad_token_id, dtype=np.int32)
    loss_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
    valid_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
    valid_tokens = 0
    loss_tokens = 0
    for row_idx, record in enumerate(rows):
      length = len(record["input_ids"])
      input_tokens[row_idx, :length] = np.asarray(record["input_ids"], dtype=np.int32)
      loss_row = np.asarray(record["loss_mask"], dtype=np.bool_)
      loss_mask[row_idx, :length] = loss_row
      valid_mask[row_idx, :length] = True
      valid_tokens += length
      loss_tokens += int(loss_row.sum())
    batches.append({
        "input_tokens": input_tokens,
        "input_mask": loss_mask,
        "valid_mask": valid_mask,
    })
    capacity = batch_size * max_length
    metrics.append({
        "valid_tokens": valid_tokens,
        "loss_tokens": loss_tokens,
        "capacity_tokens": capacity,
        "valid_ratio": valid_tokens / capacity,
        "loss_ratio": loss_tokens / capacity,
    })
  return batches, metrics


def make_packed_batches(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    pad_token_id: int,
    strategy: str,
) -> tuple[
    list[dict[str, np.ndarray]],
    list[dict[str, float | int]],
    dict[str, float | int | str],
]:
  packed = pack_records(
      records,
      max_length=max_length,
      pad_token_id=pad_token_id,
      strategy=strategy,
      return_attention_mask=True,
  )
  arrays = packed.as_tunix()
  usable_rows = (packed.batch_size // batch_size) * batch_size
  batches: list[dict[str, np.ndarray]] = []
  metrics: list[dict[str, float | int]] = []
  for start in range(0, usable_rows, batch_size):
    batch = {
        "input_tokens": arrays["input_tokens"][start : start + batch_size],
        "input_mask": arrays["input_mask"][start : start + batch_size],
        "valid_mask": arrays["valid_mask"][start : start + batch_size],
        "positions": arrays["positions"][start : start + batch_size],
        "segment_ids": arrays["segment_ids"][start : start + batch_size],
        "attention_mask": arrays["attention_mask"][start : start + batch_size],
    }
    valid_tokens = int(batch["valid_mask"].sum())
    loss_tokens = int(batch["input_mask"].sum())
    capacity = batch_size * max_length
    batches.append(batch)
    metrics.append({
        "valid_tokens": valid_tokens,
        "loss_tokens": loss_tokens,
        "capacity_tokens": capacity,
        "valid_ratio": valid_tokens / capacity,
        "loss_ratio": loss_tokens / capacity,
    })
  summary = {
      "packing_strategy": strategy,
      "source_examples": packed.num_source_examples,
      "packed_rows": packed.batch_size,
      "packed_rows_used": usable_rows,
      "packed_valid_tokens": packed.valid_tokens,
      "packed_capacity_tokens": packed.capacity_tokens,
      "packed_efficiency": packed.packing_efficiency,
      "row_reduction_x": packed.num_source_examples / max(packed.batch_size, 1),
  }
  return batches, metrics, summary


def prepare_variant(
    variant: str,
    dataset: TokenizedSftDataset,
    args: argparse.Namespace,
) -> PreparedVariant:
  records, dropped = filter_overlength(
      dataset.records,
      max_length=args.max_length,
      policy=args.long_example_policy,
  )
  if not records:
    raise ValueError(
        f"No records fit max_length={args.max_length}; cannot train {variant}."
    )

  if variant == "unpacked":
    batches, metrics = make_unpacked_batches(
        records,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pad_token_id=dataset.pad_token_id,
    )
    packing_summary: dict[str, float | int | str] = {
        "packing_strategy": "none",
        "source_examples": len(records),
        "packed_rows": len(records),
        "packed_rows_used": (len(records) // args.batch_size) * args.batch_size,
        "packed_efficiency": np.mean([m["valid_ratio"] for m in metrics])
        if metrics
        else 0.0,
        "row_reduction_x": 1.0,
    }
  elif variant == "packed":
    batches, metrics, packing_summary = make_packed_batches(
        records,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pad_token_id=dataset.pad_token_id,
        strategy=args.packing_strategy,
    )
  else:
    raise ValueError(f"Unsupported variant: {variant}")

  if not batches:
    raise ValueError(
        f"Variant {variant!r} produced no full batches. Increase num_examples "
        f"or reduce batch_size={args.batch_size}."
    )
  return PreparedVariant(
      name=variant,
      batches=batches,
      batch_metrics=metrics,
      source_examples=len(records),
      dropped_overlength=dropped,
      packing_summary=packing_summary,
  )


def cycled_batches(
    prepared: PreparedVariant,
    *,
    max_steps: int,
) -> Iterator[dict[str, np.ndarray]]:
  for step in range(max_steps):
    yield prepared.batches[step % len(prepared.batches)]


def timed_batches(
    batches: Iterator[dict[str, np.ndarray]],
    timings: list[float],
) -> Iterator[dict[str, np.ndarray]]:
  for batch in batches:
    start = time.perf_counter()
    yield batch
    timings.append(time.perf_counter() - start)


def cycled_batch_metrics(
    prepared: PreparedVariant,
    *,
    steps: int,
) -> list[dict[str, float | int]]:
  return [
      prepared.batch_metrics[step % len(prepared.batch_metrics)]
      for step in range(steps)
  ]


def create_mesh(jax, args: argparse.Namespace):
  devices = np.asarray(jax.devices())
  if devices.size == 0:
    raise RuntimeError("No JAX devices are available.")
  fsdp = args.mesh_fsdp if args.mesh_fsdp > 0 else int(devices.size)
  tp = args.mesh_tp if args.mesh_tp > 0 else int(devices.size // fsdp)
  if fsdp * tp != devices.size:
    raise ValueError(
        f"mesh_fsdp * mesh_tp must equal device count. Got {fsdp} * {tp} "
        f"for {devices.size} devices."
    )
  return jax.sharding.Mesh(devices.reshape((fsdp, tp)), ("fsdp", "tp"))


def apply_lora_if_requested(model, mesh, args: argparse.Namespace):
  from flax import nnx  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  import qwix  # pylint: disable=import-outside-toplevel
  from tunix.sft import utils as sft_utils  # pylint: disable=import-outside-toplevel
  from tunix.rl import reshard  # pylint: disable=import-outside-toplevel

  if args.lora_rank <= 0:
    return model

  lora_provider = qwix.LoraProvider(
      module_path=args.lora_module_path,
      rank=args.lora_rank,
      alpha=args.lora_alpha,
  )
  if hasattr(model, "get_model_input"):
    model = qwix.apply_lora_to_model(
        model,
        lora_provider,
        **model.get_model_input(),
        rngs=nnx.Rngs(args.seed),
    )
  else:
    sample_len = min(max(args.max_length, 1), 8)
    sample_tokens = jnp.ones((1, sample_len), dtype=jnp.int32)
    sample_valid = jnp.ones_like(sample_tokens, dtype=bool)
    sample_positions = sft_utils.build_positions_from_mask(sample_valid)
    sample_attention_mask = sft_utils.make_causal_attn_mask(sample_valid)
    model = qwix.apply_lora_to_model(
        model,
        lora_provider,
        sample_tokens,
        sample_positions,
        None,
        sample_attention_mask,
        rngs=nnx.Rngs(args.seed),
    )
  if mesh is not None:
    model = reshard.reshard_model_to_mesh(model, mesh)
  return model


def create_model(mesh, args: argparse.Namespace):
  from tunix.cli.utils import model as model_lib  # pylint: disable=import-outside-toplevel

  if should_use_gemma4_hf_loader(args):
    return create_gemma4_hf_model(mesh, args)

  model_config = {
      "model_name": infer_model_name(args.model_id),
      "model_source": args.model_source,
      "model_id": args.model_id,
      "model_path": args.model_path,
      "model_download_path": args.model_download_path,
      "intermediate_ckpt_dir": args.intermediate_ckpt_dir,
      "rng_seed": args.seed,
      "model_display": False,
  }
  tokenizer_config = {
      "tokenizer_path": args.tokenizer_path,
      "tokenizer_type": "sentencepiece",
      "add_bos": False,
      "add_eos": False,
  }
  model, _ = model_lib.create_model(model_config, tokenizer_config, mesh)
  return apply_lora_if_requested(model, mesh, args)


def should_use_gemma4_hf_loader(args: argparse.Namespace) -> bool:
  """Whether to bypass Tunix AutoModel's current Gemma4 HF guard."""
  enabled = os.environ.get("TUNIX_ACCEL_ENABLE_GEMMA4_HF_LOADER", "").lower()
  return (
      enabled in {"1", "true", "yes", "on"}
      and args.model_source == "huggingface"
      and "gemma-4" in args.model_id.lower()
  )


def create_gemma4_hf_model(mesh, args: argparse.Namespace):
  """Loads Gemma4 HF safetensors through Tunix's native Gemma4 tensor loader.

  google-tunix 0.1.6 contains Gemma4 configs and safetensor parameter loaders,
  but AutoModel's public Hugging Face path still rejects Gemma-family models.
  This opt-in path keeps the same config/parameter loader while skipping only
  that source guard.
  """
  from tunix.models import automodel  # pylint: disable=import-outside-toplevel

  model_name = infer_model_name(args.model_id)
  model_dir = args.model_download_path or args.model_path
  if not model_dir:
    raise ValueError(
        "Gemma4 HF loader requires --model-download-path or --model-path."
    )
  model_dir_path = Path(model_dir)
  if not (model_dir_path / "model.safetensors.index.json").exists():
    if not args.allow_download:
      raise FileNotFoundError(
          "Gemma4 HF safetensors snapshot is missing and --allow-download was "
          "not set: " + str(model_dir_path)
      )
    model_dir = automodel.download_model(
        args.model_id,
        args.model_download_path,
        automodel.ModelSource.HUGGINGFACE,
    )
    model_dir_path = Path(model_dir)

  try:
    model_config = automodel.call_model_config(model_name)
  except AttributeError:
    model_config = automodel.call_model_config(
        model_name.replace("gemma-4-", "gemma4_")
    )
  if mesh is None:
    model = automodel.create_model_from_safe_tensors(
        model_name,
        str(model_dir_path),
        model_config,
        mesh,
    )
  else:
    with mesh:
      model = automodel.create_model_from_safe_tensors(
          model_name,
          str(model_dir_path),
          model_config,
          mesh,
      )
  return apply_lora_if_requested(model, mesh, args)


def create_trainer(model, args: argparse.Namespace, run_dir: Path):
  import optax  # pylint: disable=import-outside-toplevel
  from tunix.sft import metrics_logger as metrics_logger_lib  # pylint: disable=import-outside-toplevel
  from tunix.sft import peft_trainer  # pylint: disable=import-outside-toplevel

  optimizer = optax.adamw(
      learning_rate=args.learning_rate,
      b1=args.adam_b1,
      b2=args.adam_b2,
      weight_decay=args.weight_decay,
  )
  metrics_options_kwargs: dict[str, Any] = {
      "log_dir": str(run_dir / "tensorboard"),
      "project_name": "tunix-accel",
      "run_name": run_dir.name,
      "flush_every_n_steps": max(args.log_every, 1),
  }
  if "backend_factories" in inspect.signature(
      metrics_logger_lib.MetricsLoggerOptions
  ).parameters:
    metrics_options_kwargs["backend_factories"] = []
  logger = metrics_logger_lib.MetricsLogger(
      metrics_logger_lib.MetricsLoggerOptions(**metrics_options_kwargs)
  )
  config = peft_trainer.TrainingConfig(
      eval_every_n_steps=max(args.max_steps + 1, 1),
      max_steps=args.max_steps,
      checkpoint_root_directory=str(run_dir / "checkpoints")
      if args.save_checkpoints
      else None,
      metrics_logging_options=None,
      data_sharding_axis=("fsdp",),
      max_inflight_computations=args.max_inflight,
      metrics_prefix="",
      pbar_description=run_dir.name,
  )
  trainer = peft_trainer.PeftTrainer(
      model,
      optimizer,
      config,
      metrics_logger=logger,
  )
  return trainer, logger


def create_gen_model_input_fn(pad_token_id: int):
  def gen_model_input_fn(batch):
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from tunix.sft import utils as sft_utils  # pylint: disable=import-outside-toplevel

    input_tokens = jnp.asarray(batch["input_tokens"], dtype=jnp.int32)
    input_mask = jnp.asarray(batch["input_mask"], dtype=bool)
    if "valid_mask" in batch:
      valid_mask = jnp.asarray(batch["valid_mask"], dtype=bool)
    else:
      valid_mask = input_tokens != pad_token_id
    positions = batch.get("positions")
    attention_mask = batch.get("attention_mask")
    if positions is None:
      positions = sft_utils.build_positions_from_mask(valid_mask)
    else:
      positions = jnp.asarray(positions, dtype=jnp.int32)
    if attention_mask is None:
      attention_mask = sft_utils.make_causal_attn_mask(valid_mask)
    else:
      attention_mask = jnp.asarray(attention_mask, dtype=bool)
    return {
        "input_tokens": input_tokens,
        "input_mask": input_mask,
        "positions": positions,
        "attention_mask": attention_mask,
    }

  return gen_model_input_fn


def metric_history(logger, metric_name: str) -> list[float]:
  from tunix.sft import metrics_logger as metrics_logger_lib  # pylint: disable=import-outside-toplevel

  try:
    values = logger.get_metric_history(
        "",
        metric_name,
        metrics_logger_lib.Mode.TRAIN,
    )
  except (KeyError, ValueError):
    return []
  return [float(np.asarray(value)) for value in values]


def device_memory_snapshot(jax) -> dict[str, Any]:
  snapshots = []
  for device in jax.local_devices():
    try:
      stats = device.memory_stats() or {}
    except Exception as exc:  # pylint: disable=broad-exception-caught
      snapshots.append({
          "device": str(device),
          "error": str(exc),
      })
      continue
    snapshots.append({
        "device": str(device),
        "platform": getattr(device, "platform", ""),
        "bytes_in_use": int(stats.get("bytes_in_use", 0) or 0),
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0) or 0),
        "bytes_limit": int(stats.get("bytes_limit", 0) or 0),
        "raw_keys": sorted(stats.keys()),
    })
  aggregate = {
      "bytes_in_use": sum(item.get("bytes_in_use", 0) for item in snapshots),
      "peak_bytes_in_use": sum(
          item.get("peak_bytes_in_use", 0) for item in snapshots
      ),
      "bytes_limit": sum(item.get("bytes_limit", 0) for item in snapshots),
  }
  return {"devices": snapshots, "aggregate": aggregate}


def package_version(name: str) -> str:
  try:
    return importlib_metadata.version(name)
  except importlib_metadata.PackageNotFoundError:
    return "not-installed"


def clean_translation(text: str) -> str:
  text = text.strip()
  for stop in ("<end_of_turn>", "<eos>", "</s>"):
    if stop in text:
      text = text.split(stop, 1)[0]
  return text.strip()


def compute_eval_loss(
    trainer,
    prepared_eval: PreparedVariant,
    *,
    max_batches: int,
) -> dict[str, Any]:
  from tunix.sft import metrics_logger as metrics_logger_lib  # pylint: disable=import-outside-toplevel

  _, eval_step = trainer.jit_train_and_eval_step(
      skip_jit=False,
      cache_nnx_graph=True,
  )
  eval_batches = prepared_eval.batches[:max_batches]
  if not eval_batches:
    return {"eval_loss": math.nan, "eval_batches": 0}
  try:
    existing_values = trainer.metrics_logger.get_metric_history(
        "",
        "loss",
        metrics_logger_lib.Mode.EVAL,
    )
    before = len(existing_values)
  except (KeyError, ValueError):
    before = 0
  trainer._run_eval(eval_batches, eval_step)  # pylint: disable=protected-access
  try:
    values = trainer.metrics_logger.get_metric_history(
        "",
        "loss",
        metrics_logger_lib.Mode.EVAL,
    )
  except (KeyError, ValueError):
    return {"eval_loss": math.nan, "eval_batches": len(eval_batches)}
  new_values = [float(np.asarray(value)) for value in values[before:]]
  if not new_values:
    new_values = [float(np.asarray(value)) for value in values]
  return {
      "eval_loss": float(np.mean(new_values)),
      "eval_batches": len(eval_batches),
      "eval_loss_values": new_values,
  }


def gemma_generation_prompt(source: str) -> str:
  return INPUT_TEMPLATE_IT["prefix"] + source + INPUT_TEMPLATE_IT["suffix"]


def generate_translations(
    model,
    tokenizer,
    examples: RawTranslationDataset,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
  from tunix.generate import sampler as sampler_lib  # pylint: disable=import-outside-toplevel

  cache_config = sampler_lib.CacheConfig(
      cache_size=args.max_length + args.max_generation_steps,
      num_layers=model.config.num_layers,
      num_kv_heads=model.config.num_kv_heads,
      head_dim=model.config.head_dim,
  )
  decode_sampler = sampler_lib.Sampler(model, tokenizer, cache_config)
  rows: list[dict[str, str]] = []
  selected = examples.examples[: args.generation_examples]
  for start in range(0, len(selected), args.generation_batch_size):
    chunk = selected[start : start + args.generation_batch_size]
    prompts = [gemma_generation_prompt(example["source"]) for example in chunk]
    outputs = decode_sampler(
        prompts,
        max_generation_steps=args.max_generation_steps,
        max_prompt_length=args.max_length,
        temperature=0.0,
        echo=False,
        pad_output=False,
    )
    for example, prediction in zip(chunk, outputs.text):
      rows.append({
          "source": example["source"],
          "reference": example["reference"],
          "prompt_tokens": example.get("prompt_tokens", ""),
          "prediction": clean_translation(prediction),
      })
  return rows


def compute_generation_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
  if not rows:
    return {"bleu": math.nan, "chrf": math.nan, "num_examples": 0}
  try:
    import sacrebleu  # pylint: disable=import-outside-toplevel
  except ImportError:
    return {
        "bleu": math.nan,
        "chrf": math.nan,
        "num_examples": len(rows),
        "error": "sacrebleu is not installed",
    }
  predictions = [row["prediction"] for row in rows]
  references = [row["reference"] for row in rows]
  return {
      "bleu": float(sacrebleu.corpus_bleu(predictions, [references]).score),
      "chrf": float(sacrebleu.corpus_chrf(predictions, [references]).score),
      "num_examples": len(rows),
  }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_variant(
    prepared: PreparedVariant,
    dataset: TokenizedSftDataset,
    tokenizer_bundle: TokenizerBundle,
    eval_examples: RawTranslationDataset | None,
    args: argparse.Namespace,
    outdir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  import jax  # pylint: disable=import-outside-toplevel

  run_dir = outdir / prepared.name
  run_dir.mkdir(parents=True, exist_ok=True)
  mesh = create_mesh(jax, args)
  start_wall = time.perf_counter()
  with jax.set_mesh(mesh):
    model = create_model(mesh, args)
    trainer, logger = create_trainer(model, args, run_dir)
    trainer.is_managed_externally = True
    local_step_times: list[float] = []
    try:
      trainer = trainer.with_gen_model_input_fn(
          create_gen_model_input_fn(dataset.pad_token_id)
      )
      memory_before_train = device_memory_snapshot(jax)
      train_iter = timed_batches(
          cycled_batches(prepared, max_steps=args.max_steps),
          local_step_times,
      )
      trainer.train(train_iter, eval_ds=None, skip_jit=args.skip_jit)
      memory_after_train = device_memory_snapshot(jax)
      quality: dict[str, Any] = {}
      generation_rows: list[dict[str, str]] = []
      if not args.skip_quality_eval and eval_examples is not None:
        eval_records = tokenize_raw_examples(
            eval_examples,
            encode=tokenizer_bundle.encode,
            eos_id=tokenizer_bundle.eos_id,
        )
        eval_records, eval_dropped = filter_overlength(
            eval_records,
            max_length=args.max_length,
            policy=args.long_example_policy,
        )
        eval_batches, eval_metrics = make_unpacked_batches(
            eval_records,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pad_token_id=dataset.pad_token_id,
        )
        prepared_eval = PreparedVariant(
            name="eval",
            batches=eval_batches,
            batch_metrics=eval_metrics,
            source_examples=len(eval_records),
            dropped_overlength=eval_dropped,
            packing_summary={
                "packing_strategy": "none",
                "source_examples": len(eval_records),
                "packed_rows": len(eval_records),
                "packed_rows_used": (len(eval_records) // args.batch_size)
                * args.batch_size,
                "packed_efficiency": np.mean(
                    [metric["valid_ratio"] for metric in eval_metrics]
                )
                if eval_metrics
                else 0.0,
                "row_reduction_x": 1.0,
            },
        )
        quality.update(
            compute_eval_loss(
                trainer,
                prepared_eval,
                max_batches=args.eval_batches,
            )
        )
        generation_examples = select_generation_examples(
            eval_examples,
            encode=tokenizer_bundle.encode,
            max_prompt_length=args.max_length,
            num_examples=args.generation_examples,
        )
        generation_rows = generate_translations(
            model,
            tokenizer_bundle.tokenizer,
            generation_examples,
            args,
        )
        quality.update(compute_generation_metrics(generation_rows))
        quality["eval_examples_requested"] = args.eval_examples
        quality["eval_examples_fit"] = len(eval_records)
        quality["eval_dropped_overlength"] = eval_dropped
        quality["generation_examples_requested"] = args.generation_examples
        quality["generation_examples_fit"] = len(generation_rows)
        write_jsonl(run_dir / "translations.jsonl", generation_rows)
      memory_after_quality = device_memory_snapshot(jax)
    finally:
      trainer.close()
  wall_time_sec = time.perf_counter() - start_wall

  losses = metric_history(logger, "loss")
  step_times = metric_history(logger, "step_time_sec")
  if (
      len(step_times) < len(losses)
      or all(math.isnan(value) for value in step_times[: len(losses)])
  ):
    step_times = local_step_times
  grad_norms = metric_history(logger, "grad_norm")
  steps_recorded = len(losses)
  token_metrics = cycled_batch_metrics(prepared, steps=steps_recorded)
  rows: list[dict[str, Any]] = []
  cumulative_valid = 0
  cumulative_loss = 0
  cumulative_time = 0.0
  for idx in range(steps_recorded):
    step_time = step_times[idx] if idx < len(step_times) else math.nan
    if not math.isnan(step_time):
      cumulative_time += step_time
    batch_metric = token_metrics[idx]
    cumulative_valid += int(batch_metric["valid_tokens"])
    cumulative_loss += int(batch_metric["loss_tokens"])
    rows.append({
        "variant": prepared.name,
        "step": idx + 1,
        "loss": losses[idx],
        "step_time_sec": step_time,
        "grad_norm": grad_norms[idx] if idx < len(grad_norms) else math.nan,
        "valid_tokens": int(batch_metric["valid_tokens"]),
        "loss_tokens": int(batch_metric["loss_tokens"]),
        "capacity_tokens": int(batch_metric["capacity_tokens"]),
        "valid_ratio": float(batch_metric["valid_ratio"]),
        "loss_ratio": float(batch_metric["loss_ratio"]),
        "cumulative_valid_tokens": cumulative_valid,
        "cumulative_loss_tokens": cumulative_loss,
        "cumulative_step_time_sec": cumulative_time,
    })

  timed_rows = [
      row for row in rows[1:] if not math.isnan(float(row["step_time_sec"]))
  ]
  if not timed_rows:
    timed_rows = [
        row for row in rows if not math.isnan(float(row["step_time_sec"]))
    ]
  measured_time = sum(float(row["step_time_sec"]) for row in timed_rows)
  measured_valid = sum(int(row["valid_tokens"]) for row in timed_rows)
  measured_loss = sum(int(row["loss_tokens"]) for row in timed_rows)
  summary = {
      "variant": prepared.name,
      "default_ce": os.environ.get("TUNIX_ACCEL_DISABLE_AUTOPATCH") in {
          "1",
          "true",
          "yes",
          "on",
      },
      "model_id": dataset.model_id,
      "model_source": args.model_source,
      "model_path": args.model_path,
      "tokenizer_source": dataset.tokenizer_source,
      "batch_size": args.batch_size,
      "max_length": args.max_length,
      "max_steps_requested": args.max_steps,
      "steps_recorded": steps_recorded,
      "learning_rate": args.learning_rate,
      "lora_rank": args.lora_rank,
      "lora_alpha": args.lora_alpha,
      "lora_module_path": args.lora_module_path if args.lora_rank > 0 else "",
      "source_examples_loaded": len(dataset.records),
      "source_examples_fit": prepared.source_examples,
      "dropped_overlength": prepared.dropped_overlength,
      "long_example_policy": args.long_example_policy,
      "prepared_batches": len(prepared.batches),
      "repeats_data": args.max_steps > len(prepared.batches),
      "final_loss": losses[-1] if losses else math.nan,
      "mean_loss": float(np.mean(losses)) if losses else math.nan,
      "mean_step_time_sec_excl_first": (
          measured_time / len(timed_rows) if timed_rows else math.nan
      ),
      "valid_tokens_per_sec_excl_first": (
          measured_valid / measured_time if measured_time > 0 else math.nan
      ),
      "loss_tokens_per_sec_excl_first": (
          measured_loss / measured_time if measured_time > 0 else math.nan
      ),
      "wall_time_sec": wall_time_sec,
      "memory_before_train": memory_before_train,
      "memory_after_train": memory_after_train,
      "memory_after_quality": memory_after_quality,
      "mesh_shape": dict(mesh.shape),
      "jax_devices": [str(device) for device in jax.devices()],
      "runtime": {
          "hostname": socket.gethostname(),
          "platform": platform.platform(),
          "tpu_name": os.environ.get("TUNIX_ACCEL_TPU_NAME", ""),
          "tpu_zone": os.environ.get("TUNIX_ACCEL_TPU_ZONE", ""),
          "jax_version": getattr(jax, "__version__", "unknown"),
          "google_tunix_version": package_version("google-tunix"),
          "tunix_accel_version": package_version("tunix-accel"),
      },
      "accel": collect_accel_status(),
      "packing": prepared.packing_summary,
      "quality": quality,
  }
  write_json(run_dir / "summary.json", summary)
  write_csv(run_dir / "history.csv", rows)
  return summary, rows


def write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    return
  fieldnames: list[str] = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def plot_results(
    summaries: list[dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
    *,
    outdir: Path,
) -> list[Path]:
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  if not summaries:
    return []
  outdir.mkdir(parents=True, exist_ok=True)
  plt.style.use("seaborn-v0_8-whitegrid")
  colors = {"unpacked": "#4C78A8", "packed": "#F58518"}

  fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
  for summary in summaries:
    variant = summary["variant"]
    rows = histories.get(variant, [])
    axes[0].plot(
        [row["cumulative_loss_tokens"] for row in rows],
        [row["loss"] for row in rows],
        marker="o",
        linewidth=1.8,
        markersize=3,
        color=colors.get(variant),
        label=variant,
    )
  axes[0].set_xlabel("Consumed Loss Tokens")
  axes[0].set_ylabel("Training Loss")
  axes[0].set_title("Loss vs Useful Training Tokens")
  axes[0].legend()

  labels = [summary["variant"] for summary in summaries]
  valid_tps = [summary["valid_tokens_per_sec_excl_first"] for summary in summaries]
  loss_tps = [summary["loss_tokens_per_sec_excl_first"] for summary in summaries]
  x = np.arange(len(labels))
  width = 0.36
  axes[1].bar(
      x - width / 2,
      valid_tps,
      width,
      label="valid tokens/sec",
      color="#54A24B",
  )
  axes[1].bar(
      x + width / 2,
      loss_tps,
      width,
      label="loss tokens/sec",
      color="#E45756",
  )
  axes[1].set_xticks(x)
  axes[1].set_xticklabels(labels)
  axes[1].set_ylabel("Tokens/sec, excluding first logged step")
  axes[1].set_title("Actual Train Throughput")
  axes[1].legend()
  for xpos, value in zip(x - width / 2, valid_tps):
    axes[1].text(xpos, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
  for xpos, value in zip(x + width / 2, loss_tps):
    axes[1].text(xpos, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
  fig.tight_layout()
  path = outdir / "training_comparison.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  return [path]


def write_readme(
    path: Path,
    *,
    dataset: TokenizedSftDataset,
    summaries: list[dict[str, Any]],
    plots: list[Path],
) -> None:
  lines = [
      "# Gemma3 Packing Training Benchmark",
      "",
      "This run compares ordinary fixed-length Tunix SFT batches against packed "
      "batches using Default CE only.",
      "",
      f"- Dataset: `{dataset.name}`",
      f"- Source: {dataset.source}",
      f"- Model: `{dataset.model_id}`",
      f"- Tokenizer source: `{dataset.tokenizer_source}`",
      "",
  ]
  for plot in plots:
    lines.append(f"![{plot.stem}]({plot.relative_to(path.parent).as_posix()})")
    lines.append("")
  lines.extend([
      "## Summary",
      "",
      (
          "| Variant | Steps | Batch | Max length | Fit examples | Rows/batches | "
          "Final loss | Eval loss | BLEU | chrF | Step time | Valid tok/s | "
          "Loss tok/s | Packing density |"
      ),
      (
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
          "---: | ---: | ---: | ---: | ---: |"
      ),
  ])
  for summary in summaries:
    packing = summary["packing"]
    quality = summary.get("quality", {})
    lines.append(
        "| "
        f"{summary['variant']} | "
        f"{summary['steps_recorded']} | "
        f"{summary['batch_size']} | "
        f"{summary['max_length']} | "
        f"{summary['source_examples_fit']} | "
        f"{summary['prepared_batches']} | "
        f"{summary['final_loss']:.4f} | "
        f"{float(quality.get('eval_loss', math.nan)):.4f} | "
        f"{float(quality.get('bleu', math.nan)):.2f} | "
        f"{float(quality.get('chrf', math.nan)):.2f} | "
        f"{summary['mean_step_time_sec_excl_first']:.3f}s | "
        f"{summary['valid_tokens_per_sec_excl_first']:.0f} | "
        f"{summary['loss_tokens_per_sec_excl_first']:.0f} | "
        f"{float(packing['packed_efficiency']) * 100:.1f}% |"
    )
  path.write_text("\n".join(lines) + "\n")


def write_prepare_only_outputs(
    *,
    dataset: TokenizedSftDataset,
    prepared_variants: list[PreparedVariant],
    args: argparse.Namespace,
    outdir: Path,
) -> None:
  rows = []
  for prepared in prepared_variants:
    mean_valid_ratio = float(
        np.mean([metric["valid_ratio"] for metric in prepared.batch_metrics])
    )
    mean_loss_ratio = float(
        np.mean([metric["loss_ratio"] for metric in prepared.batch_metrics])
    )
    rows.append({
        "variant": prepared.name,
        "source_examples_loaded": len(dataset.records),
        "source_examples_fit": prepared.source_examples,
      "dropped_overlength": prepared.dropped_overlength,
      "long_example_policy": args.long_example_policy,
      "prepared_batches": len(prepared.batches),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "mean_valid_ratio": mean_valid_ratio,
        "mean_loss_ratio": mean_loss_ratio,
        "packing": prepared.packing_summary,
    })
  write_json(outdir / "prepare_summary.json", rows)
  write_csv(
      outdir / "prepare_summary.csv",
      [
          {
              **{k: v for k, v in row.items() if k != "packing"},
              **{
                  f"packing_{k}": v
                  for k, v in dict(row["packing"]).items()
              },
          }
          for row in rows
      ],
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", default=GEMMA3_270M_IT_MODEL_ID)
  parser.add_argument("--model-source", default="gcs")
  parser.add_argument("--model-path", default=GEMMA3_270M_IT_GCS)
  parser.add_argument("--model-download-path", default=None)
  parser.add_argument("--intermediate-ckpt-dir", default=None)
  parser.add_argument("--tokenizer-source", choices=["sentencepiece", "huggingface"], default="sentencepiece")
  parser.add_argument("--tokenizer-path", default=GEMMA3_TOKENIZER_GCS)
  parser.add_argument("--allow-download", action="store_true")
  parser.add_argument(
      "--dataset-mode",
      choices=["opus100", "synthetic", "alpaca", "oasst1", "cnn_dailymail"],
      default="opus100",
  )
  parser.add_argument("--num-examples", type=int, default=5000)
  parser.add_argument("--variants", default="unpacked,packed")
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--max-length", type=int, default=512)
  parser.add_argument("--max-steps", type=int, default=50)
  parser.add_argument("--learning-rate", type=float, default=2e-4)
  parser.add_argument("--adam-b1", type=float, default=0.9)
  parser.add_argument("--adam-b2", type=float, default=0.999)
  parser.add_argument("--weight-decay", type=float, default=0.0)
  parser.add_argument("--lora-rank", type=int, default=16)
  parser.add_argument("--lora-alpha", type=float, default=32.0)
  parser.add_argument(
      "--lora-module-path",
      default=(
          ".*(q_einsum|kv_einsum|qkv_einsum|attn_vec_einsum|"
          "gate_proj|up_proj|down_proj).*"
      ),
  )
  parser.add_argument(
      "--packing-strategy",
      choices=[
          "first_fit",
          "best_fit",
          "first_fit_decreasing",
          "best_fit_decreasing",
      ],
      default="best_fit_decreasing",
  )
  parser.add_argument(
      "--long-example-policy",
      choices=["drop", "truncate"],
      default="drop",
      help=(
          "How to handle examples longer than max_length before batching. "
          "Existing OPUS100 report runs used drop; dataset-profile sweeps "
          "usually use truncate to preserve the length distribution."
      ),
  )
  parser.add_argument("--mesh-fsdp", type=int, default=0)
  parser.add_argument("--mesh-tp", type=int, default=0)
  parser.add_argument("--max-inflight", type=int, default=1)
  parser.add_argument("--skip-jit", action="store_true")
  parser.add_argument("--prepare-only", action="store_true")
  parser.add_argument("--skip-quality-eval", action="store_true")
  parser.add_argument("--eval-examples", type=int, default=512)
  parser.add_argument("--eval-batches", type=int, default=32)
  parser.add_argument("--generation-examples", type=int, default=128)
  parser.add_argument("--generation-batch-size", type=int, default=8)
  parser.add_argument("--max-generation-steps", type=int, default=128)
  parser.add_argument("--save-checkpoints", action="store_true")
  parser.add_argument("--log-every", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--outdir", default="02-PACKING/results/gemma-training-default-ce")
  parser.add_argument(
      "--allow-autopatch",
      action="store_true",
      help="Allow repository autopatches for explicit patch-comparison runs.",
  )
  args = parser.parse_args()

  configure_autopatch(allow_autopatch=args.allow_autopatch)

  outdir = Path(args.outdir).expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  tokenizer_bundle = load_tokenizer(args)
  dataset = load_opus100_records(args, tokenizer_bundle)
  eval_examples = None
  if not args.skip_quality_eval:
    eval_examples = load_raw_translation_examples(
        split="validation",
        num_examples=max(args.eval_examples, args.generation_examples * 4),
    )
  write_json(
      outdir / "dataset_summary.json",
      {
          "name": dataset.name,
          "model_id": dataset.model_id,
          "tokenizer_source": dataset.tokenizer_source,
          "pad_token_id": dataset.pad_token_id,
          "source": dataset.source,
          "num_examples": len(dataset.records),
          "mean_length": float(np.mean([len(r["input_ids"]) for r in dataset.records])),
          "p50_length": float(np.percentile([len(r["input_ids"]) for r in dataset.records], 50)),
          "p90_length": float(np.percentile([len(r["input_ids"]) for r in dataset.records], 90)),
          "max_length": int(max(len(r["input_ids"]) for r in dataset.records)),
          "eval_examples": len(eval_examples.examples)
          if eval_examples is not None
          else 0,
      },
  )

  prepared_variants = [
      prepare_variant(variant, dataset, args)
      for variant in parse_variants(args.variants)
  ]
  if args.prepare_only:
    write_prepare_only_outputs(
        dataset=dataset,
        prepared_variants=prepared_variants,
        args=args,
        outdir=outdir,
    )
    print(f"outdir={outdir}")
    print(f"prepare_summary={outdir / 'prepare_summary.json'}")
    return

  summaries: list[dict[str, Any]] = []
  histories: dict[str, list[dict[str, Any]]] = {}
  for prepared in prepared_variants:
    summary, history = run_variant(
        prepared,
        dataset,
        tokenizer_bundle,
        eval_examples,
        args,
        outdir,
    )
    summaries.append(summary)
    histories[prepared.name] = history

  write_json(outdir / "summary.json", summaries)
  all_history = [row for variant in histories.values() for row in variant]
  write_csv(outdir / "history.csv", all_history)
  plots = plot_results(summaries, histories, outdir=outdir)
  write_readme(
      outdir / "README.md",
      dataset=dataset,
      summaries=summaries,
      plots=plots,
  )

  print(f"outdir={outdir}")
  print(f"summary={outdir / 'summary.json'}")
  print(f"history={outdir / 'history.csv'}")
  for plot in plots:
    print(f"plot={plot}")


if __name__ == "__main__":
  main()
