#!/usr/bin/env python3
"""Actual Tunix/Gemma training benchmark for sequence packing.

This script intentionally runs the Default CE path. Set
TUNIX_ACCEL_DISABLE_AUTOPATCH=1 before launching Python so the repository's
sitecustomize hook cannot install the CCE patch at interpreter startup.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Iterator

os.environ.setdefault("TUNIX_ACCEL_DISABLE_AUTOPATCH", "1")

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


def load_tokenizer(args: argparse.Namespace):
  if args.tokenizer_source == "sentencepiece":
    from tunix.models.gemma3 import params as gemma3_params  # pylint: disable=import-outside-toplevel

    tokenizer = gemma3_params.create_tokenizer(args.tokenizer_path)
    pad_id = int(tokenizer.pad_id())
    if pad_id < 0:
      pad_id = 0
    eos_id = int(tokenizer.eos_id())

    def encode(text: str) -> list[int]:
      return [int(x) for x in tokenizer.EncodeAsIds(text)]

    return encode, pad_id, eos_id

  if args.tokenizer_source == "huggingface":
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=not args.allow_download,
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
      pad_id = 0
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
      raise ValueError("The Hugging Face tokenizer has no eos_token_id.")

    def encode(text: str) -> list[int]:
      return [int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]]

    return encode, int(pad_id), int(eos_id)

  raise ValueError(f"Unsupported tokenizer source: {args.tokenizer_source!r}")


def tokenize_sft_record(
    *,
    encode,
    eos_id: int,
    source: str,
    target: str,
    example_id: int,
) -> dict[str, Any]:
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


def load_opus100_records(args: argparse.Namespace) -> TokenizedSftDataset:
  from datasets import load_dataset  # pylint: disable=import-outside-toplevel

  encode, pad_id, eos_id = load_tokenizer(args)
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
            encode=encode,
            eos_id=eos_id,
            source=source,
            target=target,
            example_id=idx,
        )
    )

  return TokenizedSftDataset(
      name="opus100-en-fr-gemma3-it",
      model_id=args.model_id,
      tokenizer_source=args.tokenizer_source,
      pad_token_id=pad_id,
      records=records,
      source=(
          "Helsinki-NLP/opus-100 en-fr train split, Tunix Gemma3 IT prompt "
          "wrapper, target-only loss mask, target EOS"
      ),
  )


def filter_overlength(
    records: list[dict[str, Any]],
    *,
    max_length: int,
) -> tuple[list[dict[str, Any]], int]:
  kept = [record for record in records if len(record["input_ids"]) <= max_length]
  return kept, len(records) - len(kept)


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
  records, dropped = filter_overlength(dataset.records, max_length=args.max_length)
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


def create_model(mesh, args: argparse.Namespace):
  from flax import nnx  # pylint: disable=import-outside-toplevel
  import qwix  # pylint: disable=import-outside-toplevel
  from tunix.cli.utils import model as model_lib  # pylint: disable=import-outside-toplevel
  from tunix.rl import reshard  # pylint: disable=import-outside-toplevel

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
  if args.lora_rank > 0:
    lora_provider = qwix.LoraProvider(
        module_path=args.lora_module_path,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    model = qwix.apply_lora_to_model(
        model,
        lora_provider,
        **model.get_model_input(),
        rngs=nnx.Rngs(args.seed),
    )
    if mesh is not None:
      model = reshard.reshard_model_to_mesh(model, mesh)
  return model


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
  logger = metrics_logger_lib.MetricsLogger(
      metrics_logger_lib.MetricsLoggerOptions(
          log_dir=str(run_dir / "tensorboard"),
          project_name="tunix-accel",
          run_name=run_dir.name,
          flush_every_n_steps=max(args.log_every, 1),
          backend_factories=[],
      )
  )
  config = peft_trainer.TrainingConfig(
      eval_every_n_steps=max(args.max_steps + 1, 1),
      max_steps=args.max_steps,
      checkpoint_root_directory=None,
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
  except ValueError:
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


def run_variant(
    prepared: PreparedVariant,
    dataset: TokenizedSftDataset,
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
    trainer = trainer.with_gen_model_input_fn(
        create_gen_model_input_fn(dataset.pad_token_id)
    )
    memory_before_train = device_memory_snapshot(jax)
    train_iter = cycled_batches(prepared, max_steps=args.max_steps)
    trainer.train(train_iter, eval_ds=None, skip_jit=args.skip_jit)
    jax.block_until_ready(np.asarray(metric_history(logger, "loss")[-1:]))
    memory_after_train = device_memory_snapshot(jax)
  wall_time_sec = time.perf_counter() - start_wall

  losses = metric_history(logger, "loss")
  step_times = metric_history(logger, "step_time_sec")
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
      "mesh_shape": dict(mesh.shape),
      "jax_devices": [str(device) for device in jax.devices()],
      "packing": prepared.packing_summary,
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
          "Final loss | Step time | Valid tok/s | Loss tok/s | Packing density |"
      ),
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  for summary in summaries:
    packing = summary["packing"]
    lines.append(
        "| "
        f"{summary['variant']} | "
        f"{summary['steps_recorded']} | "
        f"{summary['batch_size']} | "
        f"{summary['max_length']} | "
        f"{summary['source_examples_fit']} | "
        f"{summary['prepared_batches']} | "
        f"{summary['final_loss']:.4f} | "
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
  parser.add_argument("--mesh-fsdp", type=int, default=0)
  parser.add_argument("--mesh-tp", type=int, default=0)
  parser.add_argument("--max-inflight", type=int, default=1)
  parser.add_argument("--skip-jit", action="store_true")
  parser.add_argument("--prepare-only", action="store_true")
  parser.add_argument("--log-every", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--outdir", default="02-PACKING/results/gemma-training-default-ce")
  args = parser.parse_args()

  if os.environ.get("TUNIX_ACCEL_DISABLE_AUTOPATCH") not in {
      "1",
      "true",
      "yes",
      "on",
  }:
    raise RuntimeError(
        "This benchmark is the Default CE baseline. Launch with "
        "TUNIX_ACCEL_DISABLE_AUTOPATCH=1."
    )

  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  dataset = load_opus100_records(args)
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
    summary, history = run_variant(prepared, dataset, args, outdir)
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
