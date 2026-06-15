#!/usr/bin/env python3
"""Structured TPU v5e CCE mesh/chunk experiment runner.

This runner is intentionally a thin orchestration layer over
`run_gemma_training_benchmark.py`.  It owns matrix generation, sharding,
profiler selection, and structured result rows; the benchmark still owns the
actual Tunix/JAX training step.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAINING_RUNNER = SCRIPT_DIR / "run_gemma_training_benchmark.py"

DEFAULT_CORE_MESHES = [(4, 1), (2, 2), (1, 4)]
DEFAULT_CORE_SHAPES = [(16, 512), (16, 1024), (32, 512)]
DEFAULT_CORE_CHUNKS = [
    (128, 8192),
    (128, 32768),
    (256, 32768),
    (512, 32768),
    (512, 65536),
]
BAD_ROW_CHUNKS = [
    (128, 8192),
    (256, 32768),
    (512, 65536),
]
PROFILER_CASES = [
    (2, 2, 16, 512, 128, 8192),
    (2, 2, 16, 512, 512, 65536),
    (4, 1, 16, 512, 128, 8192),
    (1, 4, 16, 512, 128, 8192),
]

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "270m": {
        "label": "Gemma3 270M",
        "model_id": "google/gemma-3-270m-it",
        "model_source": "gcs",
        "model_path": "gs://gemma-data/checkpoints/gemma3-270m-it",
        "tokenizer_source": "sentencepiece",
        "tokenizer_path": "gs://gemma-data/tokenizers/tokenizer_gemma3.model",
        "allow_download": False,
        "enable_gemma4_hf_loader": False,
        "vocab_size": 262144,
    },
    "1b": {
        "label": "Gemma3 1B",
        "model_id": "google/gemma-3-1b-it",
        "model_source": "gcs",
        "model_path": "gs://gemma-data/checkpoints/gemma3-1b-it",
        "tokenizer_source": "sentencepiece",
        "tokenizer_path": "gs://gemma-data/tokenizers/tokenizer_gemma3.model",
        "allow_download": False,
        "enable_gemma4_hf_loader": False,
        "vocab_size": 262144,
    },
    "qwen3_0p6b": {
        "label": "Qwen3 0.6B",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_source": "huggingface",
        "model_path": "",
        "tokenizer_source": "huggingface",
        "tokenizer_path": "",
        "allow_download": True,
        "enable_gemma4_hf_loader": False,
        "vocab_size": 151936,
    },
}

ENV_KEYS = {
    "TUNIX_ACCEL_DISABLE_AUTOPATCH",
    "TUNIX_ACCEL_DISABLE_CE",
    "TUNIX_ACCEL_CE_TOKEN_CHUNK",
    "TUNIX_ACCEL_CE_VOCAB_CHUNK",
    "TUNIX_ACCEL_DISABLE_TILED_MLP",
    "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA",
    "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY",
    "TUNIX_ACCEL_ACTIVATION_POLICY",
    "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION",
    "TUNIX_ACCEL_ENABLE_GEMMA4_HF_LOADER",
}


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_int_list(value: str) -> list[int]:
  return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_meshes(value: str) -> list[tuple[int, int]]:
  explicit = re.findall(
      r"fsdp\s*=\s*(\d+)\s*,?\s*tp\s*=\s*(\d+)",
      value,
      flags=re.IGNORECASE,
  )
  if explicit:
    return [(int(fsdp), int(tp)) for fsdp, tp in explicit]

  meshes: list[tuple[int, int]] = []
  for raw in value.split(","):
    item = raw.strip().lower()
    if not item:
      continue
    match = re.fullmatch(r"(?:fsdp=)?(\d+)\s*(?:x|/|:|tp=)\s*(\d+)", item)
    if not match:
      match = re.fullmatch(r"fsdp(\d+)[_/,-]?tp(\d+)", item)
    if not match:
      raise ValueError(
          "Mesh values must look like '4x1', 'fsdp=4,tp=1', or 'fsdp4_tp1'. "
          f"Got {raw!r}."
      )
    meshes.append((int(match.group(1)), int(match.group(2))))
  if not meshes:
    raise ValueError("At least one mesh must be provided.")
  return meshes


def parse_shapes(value: str) -> list[tuple[int, int]]:
  shapes: list[tuple[int, int]] = []
  for raw in value.split(","):
    item = raw.strip().lower()
    if not item:
      continue
    match = re.fullmatch(r"b?(\d+)\s*(?:/|x|:|l)\s*l?(\d+)", item)
    if not match:
      raise ValueError(
          "Shape values must look like 'b16/L512' or '16x512'. "
          f"Got {raw!r}."
      )
    shapes.append((int(match.group(1)), int(match.group(2))))
  if not shapes:
    raise ValueError("At least one shape must be provided.")
  return shapes


def parse_chunks(value: str) -> list[tuple[int, int]]:
  chunks: list[tuple[int, int]] = []
  for raw in value.split(","):
    item = raw.strip().lower()
    if not item:
      continue
    match = re.fullmatch(r"(\d+)\s*(?:/|x|:)\s*(\d+)", item)
    if not match:
      raise ValueError(
          "Chunk values must look like '128/8192' or '512x65536'. "
          f"Got {raw!r}."
      )
    chunks.append((int(match.group(1)), int(match.group(2))))
  if not chunks:
    raise ValueError("At least one chunk pair must be provided.")
  return chunks


def maybe_parse_meshes(value: str | None) -> list[tuple[int, int]] | None:
  return parse_meshes(value) if value else None


def maybe_parse_shapes(value: str | None) -> list[tuple[int, int]] | None:
  return parse_shapes(value) if value else None


def maybe_parse_chunks(value: str | None) -> list[tuple[int, int]] | None:
  return parse_chunks(value) if value else None


def safe_float(value: Any) -> float | None:
  if value is None or value == "":
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def safe_int(value: Any) -> int | None:
  if value is None or value == "":
    return None
  try:
    return int(float(value))
  except (TypeError, ValueError):
    return None


def percentile_nearest(values: list[float], q: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  index = math.ceil(q / 100.0 * len(ordered)) - 1
  index = min(max(index, 0), len(ordered) - 1)
  return ordered[index]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  mode = "a" if append else "w"
  with path.open(mode) as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")
      f.flush()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  keys: list[str] = []
  for row in rows:
    for key in row:
      if key not in keys:
        keys.append(key)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def read_json(path: Path) -> Any | None:
  if not path.exists():
    return None
  return json.loads(path.read_text())


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open() as f:
    return list(csv.DictReader(f))


def first_summary(run_dir: Path) -> dict[str, Any] | None:
  data = read_json(run_dir / "summary.json")
  if isinstance(data, list) and data:
    return dict(data[0])
  if isinstance(data, dict):
    return dict(data)
  return None


def latest_train_memory_report(xla_dir: Path) -> Path | None:
  reports = sorted(xla_dir.glob("*jit__train_step*memory-usage-report.txt"))
  return reports[-1] if reports else None


def parse_xla_total_gib(report: Path | None) -> float | None:
  if report is None or not report.exists():
    return None
  text = report.read_text(errors="ignore")
  match = re.search(r"Total bytes:\s+\d+\s+\(([\d.]+)GiB\)", text)
  if match:
    return float(match.group(1))
  match = re.search(r"Total bytes:\s+(\d+)", text)
  if match:
    return int(match.group(1)) / (1024**3)
  return None


def cleanup_xla_dir(xla_dir: Path, *, keep_all_xla: bool) -> None:
  if keep_all_xla or not xla_dir.exists():
    return
  for path in xla_dir.iterdir():
    if path.name.endswith("memory-usage-report.txt"):
      continue
    if path.is_dir():
      shutil.rmtree(path, ignore_errors=True)
    else:
      path.unlink(missing_ok=True)


def tpu_worker_id() -> str:
  for key in [
      "TPU_WORKER_ID",
      "CLOUD_TPU_TASK_ID",
      "JAX_PROCESS_INDEX",
      "TPU_PROCESS_INDEX",
      "HOSTNAME",
  ]:
    value = os.environ.get(key)
    if value:
      return value
  return ""


def model_args(args: argparse.Namespace) -> dict[str, Any]:
  preset = dict(MODEL_PRESETS[args.model_size])
  preset["model_id"] = args.model_id or preset["model_id"]
  preset["model_source"] = args.model_source or preset["model_source"]
  preset["model_path"] = (
      args.model_path if args.model_path is not None else preset["model_path"]
  )
  preset["tokenizer_source"] = args.tokenizer_source or preset["tokenizer_source"]
  preset["tokenizer_path"] = (
      args.tokenizer_path
      if args.tokenizer_path is not None
      else preset["tokenizer_path"]
  )
  preset["allow_download"] = bool(args.allow_download or preset["allow_download"])
  preset["vocab_size"] = args.vocab_size or preset["vocab_size"]
  return preset


def is_profiler_case(case: dict[str, Any]) -> bool:
  if not case["cce_enabled"]:
    return False
  signature = (
      case["fsdp_degree"],
      case["tp_degree"],
      case["global_batch_size"],
      case["sequence_length"],
      case["token_chunk"],
      case["vocab_chunk"],
  )
  return signature in PROFILER_CASES


def case_name(case: dict[str, Any]) -> str:
  chunk = (
      f"tc{case['token_chunk']}_vc{case['vocab_chunk']}"
      if case["cce_enabled"]
      else "default_ce"
  )
  return (
      f"exp{case['experiment_id']:04d}_"
      f"fsdp{case['fsdp_degree']}_tp{case['tp_degree']}_"
      f"b{case['global_batch_size']}_l{case['sequence_length']}_"
      f"{chunk}_r{case['repeat_index']}"
  )


def base_case(
    *,
    args: argparse.Namespace,
    model: dict[str, Any],
    experiment_id: int,
    preset: str,
    fsdp: int,
    tp: int,
    batch_size: int,
    sequence_length: int,
    cce_enabled: bool,
    token_chunk: int | None,
    vocab_chunk: int | None,
    repeat_index: int,
) -> dict[str, Any]:
  token_loop_count = (
      math.ceil(sequence_length / token_chunk)
      if cce_enabled and token_chunk
      else None
  )
  vocab_loop_count = (
      math.ceil(int(model["vocab_size"]) / vocab_chunk)
      if cce_enabled and vocab_chunk
      else None
  )
  return {
      "experiment_id": experiment_id,
      "run_id": args.run_id,
      "preset": preset,
      "backend": args.backend,
      "timestamp_manifest": utc_now(),
      "hostname_manifest": socket.gethostname(),
      "tpu_worker_id_manifest": tpu_worker_id(),
      "tpu_type": args.tpu_type,
      "chips": args.chips,
      "model_size": args.model_size,
      "model_name": model["label"],
      "model_id": model["model_id"],
      "model_source": model["model_source"],
      "model_path": model["model_path"],
      "tokenizer_source": model["tokenizer_source"],
      "tokenizer_path": model["tokenizer_path"],
      "allow_download": bool(model["allow_download"]),
      "enable_gemma4_hf_loader": bool(model["enable_gemma4_hf_loader"]),
      "vocab_size": int(model["vocab_size"]),
      "global_batch_size": batch_size,
      "sequence_length": sequence_length,
      "shape": f"b{batch_size}/L{sequence_length}",
      "mesh_configuration": f"fsdp={fsdp},tp={tp}",
      "fsdp_degree": fsdp,
      "tp_degree": tp,
      "cce_enabled": cce_enabled,
      "loss_impl": "cce" if cce_enabled else "default_ce",
      "token_chunk": token_chunk,
      "vocab_chunk": vocab_chunk,
      "chunk_configuration": (
          f"{token_chunk}/{vocab_chunk}" if cce_enabled else "default_ce"
      ),
      "token_loop_count": token_loop_count,
      "vocab_loop_count": vocab_loop_count,
      "cce_loop_count": (
          token_loop_count * vocab_loop_count
          if token_loop_count is not None and vocab_loop_count is not None
          else None
      ),
      "warmup_steps": args.warmup_steps,
      "measured_steps": args.measured_steps,
      "max_steps": args.warmup_steps + args.measured_steps,
      "dataset_mode": args.dataset_mode,
      "num_examples": args.num_examples,
      "lora_rank": args.lora_rank,
      "lora_alpha": args.lora_alpha,
      "learning_rate": args.learning_rate,
      "max_inflight": args.max_inflight,
      "seed": args.seed + repeat_index,
      "repeat_index": repeat_index,
  }


def build_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
  model = model_args(args)
  meshes_override = maybe_parse_meshes(args.meshes)
  shapes_override = maybe_parse_shapes(args.shapes)
  chunks_override = maybe_parse_chunks(args.chunk_configs)
  repeats = args.repeats
  include_default = not args.no_default_ce

  if args.preset == "core":
    meshes = meshes_override or DEFAULT_CORE_MESHES
    shapes = shapes_override or DEFAULT_CORE_SHAPES
    chunks = chunks_override or DEFAULT_CORE_CHUNKS
    if repeats is None:
      repeats = 1
  elif args.preset == "bad-row":
    meshes = meshes_override or [(2, 2)]
    shapes = shapes_override or [(16, 512), (16, 1024)]
    chunks = chunks_override or BAD_ROW_CHUNKS
    include_default = args.include_default_ce
    if repeats is None:
      repeats = 3
  elif args.preset == "profiler":
    if repeats is None:
      repeats = 1
    cases: list[dict[str, Any]] = []
    experiment_id = 0
    for repeat_index in range(repeats):
      for fsdp, tp, batch_size, sequence_length, token_chunk, vocab_chunk in PROFILER_CASES:
        case = base_case(
            args=args,
            model=model,
            experiment_id=experiment_id,
            preset=args.preset,
            fsdp=fsdp,
            tp=tp,
            batch_size=batch_size,
            sequence_length=sequence_length,
            cce_enabled=True,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
            repeat_index=repeat_index,
        )
        cases.append(case)
        experiment_id += 1
    return attach_paths_and_profiler(args, cases)
  else:
    raise ValueError(f"Unknown preset: {args.preset!r}")

  cases = []
  experiment_id = 0
  for repeat_index in range(repeats):
    for fsdp, tp in meshes:
      for batch_size, sequence_length in shapes:
        if include_default:
          cases.append(
              base_case(
                  args=args,
                  model=model,
                  experiment_id=experiment_id,
                  preset=args.preset,
                  fsdp=fsdp,
                  tp=tp,
                  batch_size=batch_size,
                  sequence_length=sequence_length,
                  cce_enabled=False,
                  token_chunk=None,
                  vocab_chunk=None,
                  repeat_index=repeat_index,
              )
          )
          experiment_id += 1
        for token_chunk, vocab_chunk in chunks:
          cases.append(
              base_case(
                  args=args,
                  model=model,
                  experiment_id=experiment_id,
                  preset=args.preset,
                  fsdp=fsdp,
                  tp=tp,
                  batch_size=batch_size,
                  sequence_length=sequence_length,
                  cce_enabled=True,
                  token_chunk=token_chunk,
                  vocab_chunk=vocab_chunk,
                  repeat_index=repeat_index,
              )
          )
          experiment_id += 1
  return attach_paths_and_profiler(args, cases)


def attach_paths_and_profiler(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  for case in cases:
    name = case_name(case)
    case["case_name"] = name
    case["run_dir"] = str((args.outdir / "runs" / name).resolve())
    case["xla_dir"] = str((Path(case["run_dir"]) / "xla").resolve())
    case["profiler_enabled"] = bool(args.enable_profiler and is_profiler_case(case))
    if case["profiler_enabled"]:
      case["profiler_path"] = str((args.profiler_dir / name).resolve())
    else:
      case["profiler_path"] = ""
  return cases


def validate_matrix(args: argparse.Namespace, cases: list[dict[str, Any]]) -> None:
  if args.num_shards < 1:
    raise ValueError("--num-shards must be >= 1.")
  if args.shard_index < 0 or args.shard_index >= args.num_shards:
    raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
  for case in cases:
    if case["fsdp_degree"] * case["tp_degree"] != args.chips:
      raise ValueError(
          "Every mesh must multiply to --chips. "
          f"Experiment {case['experiment_id']} has "
          f"{case['fsdp_degree']} * {case['tp_degree']} != {args.chips}."
      )


def select_cases(args: argparse.Namespace, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
  selected = cases
  if args.experiment_id:
    ids = set(args.experiment_id)
    selected = [case for case in selected if case["experiment_id"] in ids]
  else:
    selected = [
        case
        for case in selected
        if case["experiment_id"] % args.num_shards == args.shard_index
    ]
  if args.limit is not None:
    selected = selected[: args.limit]
  return selected


def configure_env(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, str]:
  env = os.environ.copy()
  for key in ENV_KEYS:
    env.pop(key, None)
  pythonpath = str(REPO_ROOT)
  if env.get("PYTHONPATH"):
    pythonpath += os.pathsep + env["PYTHONPATH"]
  env.update({
      "PYTHONPATH": pythonpath,
      "PYTHONUNBUFFERED": "1",
      "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
      "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
      "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
      "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "0",
  })
  if case["model_source"] == "huggingface":
    env.setdefault("HF_TOKEN", "")
  if case.get("enable_gemma4_hf_loader"):
    env["TUNIX_ACCEL_ENABLE_GEMMA4_HF_LOADER"] = "1"
  if case["cce_enabled"]:
    env["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "0"
    env["TUNIX_ACCEL_DISABLE_CE"] = "0"
    env["TUNIX_ACCEL_CE_TOKEN_CHUNK"] = str(case["token_chunk"])
    env["TUNIX_ACCEL_CE_VOCAB_CHUNK"] = str(case["vocab_chunk"])
  else:
    env["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "1"

  if not args.disable_xla_dump:
    xla_flags = env.get("XLA_FLAGS", "").strip()
    dump_parts = [
        f"--xla_dump_to={case['xla_dir']}",
        "--xla_dump_hlo_as_text",
    ]
    if args.full_hlo_dump:
      dump_parts.extend([
          "--xla_dump_hlo_as_proto",
          "--xla_dump_hlo_module_re=.*train_step.*",
          "--xla_dump_hlo_pass_re=.*",
      ])
    dump_flags = " ".join(dump_parts)
    env["XLA_FLAGS"] = f"{xla_flags} {dump_flags}".strip()
  return env


def command_for_case(args: argparse.Namespace, case: dict[str, Any]) -> list[str]:
  command = [
      sys.executable,
      str(args.training_runner),
      "--model-id",
      case["model_id"],
      "--model-source",
      case["model_source"],
      "--model-path",
      case["model_path"],
      "--tokenizer-source",
      case["tokenizer_source"],
      "--tokenizer-path",
      case["tokenizer_path"],
      "--dataset-mode",
      case["dataset_mode"],
      "--num-examples",
      str(case["num_examples"]),
      "--variants",
      "unpacked",
      "--batch-size",
      str(case["global_batch_size"]),
      "--max-length",
      str(case["sequence_length"]),
      "--max-steps",
      str(case["max_steps"]),
      "--learning-rate",
      str(case["learning_rate"]),
      "--lora-rank",
      str(case["lora_rank"]),
      "--lora-alpha",
      str(case["lora_alpha"]),
      "--mesh-fsdp",
      str(case["fsdp_degree"]),
      "--mesh-tp",
      str(case["tp_degree"]),
      "--max-inflight",
      str(case["max_inflight"]),
      "--log-every",
      str(args.log_every),
      "--seed",
      str(case["seed"]),
      "--outdir",
      case["run_dir"],
      "--skip-quality-eval",
  ]
  if case["cce_enabled"]:
    command.append("--allow-autopatch")
  if args.initialize_distributed:
    command.append("--initialize-distributed")
  if args.model_download_path:
    command.extend(["--model-download-path", args.model_download_path])
  if case["model_source"] == "huggingface" and case.get("allow_download", False):
    command.append("--allow-download")
  if case["profiler_enabled"]:
    command.extend(["--profiler-dir", case["profiler_path"]])
  return command


def flatten_memory(summary: dict[str, Any] | None) -> dict[str, Any]:
  if not summary:
    return {}
  memory = summary.get("memory_after_train", {}).get("aggregate", {})
  if not isinstance(memory, dict) or not memory:
    return {}
  peak = int(memory.get("peak_bytes_in_use", 0) or 0)
  limit = int(memory.get("bytes_limit", 0) or 0)
  return {
      "runtime_hbm_peak_gb": peak / 1e9 if peak else None,
      "runtime_hbm_limit_gb": limit / 1e9 if limit else None,
      "runtime_hbm_headroom_gb": (limit - peak) / 1e9
      if limit and peak
      else None,
  }


def step_stats(
    history: list[dict[str, str]],
    *,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
  all_times = [safe_float(row.get("step_time_sec")) for row in history]
  all_times = [value for value in all_times if value is not None]
  measured_rows = history[warmup_steps : warmup_steps + measured_steps]
  measured_times = [
      safe_float(row.get("step_time_sec"))
      for row in measured_rows
      if safe_float(row.get("step_time_sec")) is not None
  ]
  measured_valid_tokens = sum(
      safe_int(row.get("valid_tokens")) or 0 for row in measured_rows
  )
  measured_capacity_tokens = sum(
      safe_int(row.get("capacity_tokens")) or 0 for row in measured_rows
  )
  measured_time = sum(measured_times)
  median = statistics.median(measured_times) if measured_times else None
  first_step = all_times[0] if all_times else None
  compile_time = (
      max(0.0, first_step - median)
      if first_step is not None and median is not None
      else None
  )
  return {
      "steps_recorded": len(history),
      "first_step_time_sec": first_step,
      "compile_time_sec": compile_time,
      "compile_time_estimate_method": "max(first_step_time - steady_median, 0)"
      if compile_time is not None
      else "",
      "steady_state_mean_step_time_sec": (
          statistics.mean(measured_times) if measured_times else None
      ),
      "median_step_time_sec": median,
      "p95_step_time_sec": percentile_nearest(measured_times, 95),
      "measured_step_count": len(measured_times),
      "measured_time_sec": measured_time if measured_times else None,
      "tokens_per_sec": measured_valid_tokens / measured_time
      if measured_time > 0
      else None,
      "capacity_tokens_per_sec": measured_capacity_tokens / measured_time
      if measured_time > 0
      else None,
  }


def parse_failure(log_path: Path) -> dict[str, Any]:
  text = log_path.read_text(errors="ignore") if log_path.exists() else ""
  if "CompileTimeHbmOom" in text:
    failure_type = "compile_oom"
  elif "RESOURCE_EXHAUSTED" in text:
    failure_type = "resource_exhausted"
  elif "Traceback" in text:
    failure_type = "error"
  else:
    failure_type = "unknown"
  lines = [line.strip() for line in text.splitlines() if line.strip()]
  tail = "\n".join(lines[-20:])
  row: dict[str, Any] = {
      "failure_type": failure_type,
      "error_message": tail[-4000:],
  }
  match = re.search(
      r"Used\s+([\d.]+)G\s+of\s+([\d.]+)G\s+hbm\.\s+Exceeded.*?by\s+([\d.]+)G",
      text,
      flags=re.DOTALL,
  )
  if match:
    row.update({
        "oom_used_gib": float(match.group(1)),
        "oom_limit_gib": float(match.group(2)),
        "oom_exceeded_gib": float(match.group(3)),
    })
  return row


def result_base(case: dict[str, Any]) -> dict[str, Any]:
  return {
      **case,
      "timestamp": utc_now(),
      "hostname": socket.gethostname(),
      "tpu_worker_id": tpu_worker_id(),
  }


def run_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
  run_dir = Path(case["run_dir"])
  xla_dir = Path(case["xla_dir"])
  log_path = run_dir / "runner.log"
  if run_dir.exists() and args.force:
    shutil.rmtree(run_dir)
  run_dir.mkdir(parents=True, exist_ok=True)
  xla_dir.mkdir(parents=True, exist_ok=True)

  command = command_for_case(args, case)
  env = configure_env(args, case)
  started = time.monotonic()
  returncode: int | None = None
  with log_path.open("w") as log:
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
    returncode = proc.returncode
  elapsed = time.monotonic() - started
  (run_dir / "elapsed_sec.txt").write_text(f"{elapsed:.6f}\n")
  (run_dir / "returncode.txt").write_text(f"{returncode}\n")
  cleanup_xla_dir(xla_dir, keep_all_xla=args.keep_all_xla)

  summary = first_summary(run_dir)
  history = read_csv_dicts(run_dir / "history.csv")
  report = latest_train_memory_report(xla_dir)
  row = {
      **result_base(case),
      "status": "success" if summary else "failure",
      "returncode": returncode,
      "elapsed_sec": elapsed,
      "log_path": str(log_path.resolve()),
      "xla_report_path": str(report.resolve()) if report else "",
      "xla_planned_hbm_gib_per_chip": parse_xla_total_gib(report),
  }
  row.update(
      step_stats(
          history,
          warmup_steps=int(case["warmup_steps"]),
          measured_steps=int(case["measured_steps"]),
      )
  )
  row.update(flatten_memory(summary))
  if summary:
    runtime = summary.get("runtime", {})
    if isinstance(runtime, dict):
      row.update({
          "summary_hostname": runtime.get("hostname", ""),
          "summary_tpu_name": runtime.get("tpu_name", ""),
          "summary_tpu_zone": runtime.get("tpu_zone", ""),
          "jax_version": runtime.get("jax_version", ""),
          "google_tunix_version": runtime.get("google_tunix_version", ""),
          "tunix_accel_version": runtime.get("tunix_accel_version", ""),
      })
    row.update({
        "final_loss": summary.get("final_loss"),
        "mean_loss": summary.get("mean_loss"),
        "benchmark_wall_time_sec": summary.get("wall_time_sec"),
        "summary_valid_tokens_per_sec_excl_first": summary.get(
            "valid_tokens_per_sec_excl_first"
        ),
        "summary_mean_step_time_sec_excl_first": summary.get(
            "mean_step_time_sec_excl_first"
        ),
      })
  else:
    row.update(parse_failure(log_path))
  (run_dir / "case_summary.json").write_text(
      json.dumps(row, indent=2, sort_keys=True) + "\n"
  )
  return row


def manifest_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
  keys_to_drop = {
      "model_path",
      "tokenizer_path",
      "hostname_manifest",
      "tpu_worker_id_manifest",
  }
  return [
      {key: value for key, value in case.items() if key not in keys_to_drop}
      for case in cases
  ]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--preset", choices=["core", "bad-row", "profiler"], default="core")
  parser.add_argument("--backend", default="tpu-v5e")
  parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="270m")
  parser.add_argument("--model-id", default=None)
  parser.add_argument("--model-source", choices=["gcs", "huggingface"], default=None)
  parser.add_argument("--model-path", default=None)
  parser.add_argument("--model-download-path", default="")
  parser.add_argument(
      "--tokenizer-source",
      choices=["sentencepiece", "huggingface"],
      default=None,
  )
  parser.add_argument("--tokenizer-path", default=None)
  parser.add_argument("--allow-download", action="store_true")
  parser.add_argument("--vocab-size", type=int, default=0)
  parser.add_argument("--meshes", default=None)
  parser.add_argument("--shapes", default=None)
  parser.add_argument("--chunk-configs", default=None)
  parser.add_argument("--repeats", type=int, default=None)
  parser.add_argument("--no-default-ce", action="store_true")
  parser.add_argument(
      "--include-default-ce",
      action="store_true",
      help="Include Default CE rows for presets that default to CCE-only.",
  )
  parser.add_argument("--warmup-steps", type=int, default=3)
  parser.add_argument("--measured-steps", type=int, default=10)
  parser.add_argument("--dataset-mode", choices=["synthetic", "opus100"], default="synthetic")
  parser.add_argument("--num-examples", type=int, default=2048)
  parser.add_argument("--learning-rate", type=float, default=2e-4)
  parser.add_argument("--lora-rank", type=int, default=16)
  parser.add_argument("--lora-alpha", type=float, default=32.0)
  parser.add_argument("--max-inflight", type=int, default=1)
  parser.add_argument("--log-every", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--tpu-type", default="v5litepod-4")
  parser.add_argument("--chips", type=int, default=4)
  parser.add_argument("--initialize-distributed", action="store_true")
  parser.add_argument("--enable-profiler", action="store_true")
  parser.add_argument("--disable-xla-dump", action="store_true")
  parser.add_argument(
      "--full-hlo-dump",
      action="store_true",
      help=(
          "Retain text/proto HLO for train_step pass dumps. Intended for the "
          "small profiler preset with --keep-all-xla, not the full grid."
      ),
  )
  parser.add_argument("--keep-all-xla", action="store_true")
  parser.add_argument("--shard-index", type=int, default=0)
  parser.add_argument("--num-shards", type=int, default=1)
  parser.add_argument("--experiment-id", type=int, action="append", default=[])
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--run-id", default="")
  parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "results" / "v5e-cce-matrix")
  parser.add_argument("--profiler-dir", type=Path, default=None)
  parser.add_argument("--manifest-path", type=Path, default=None)
  parser.add_argument("--results-path", type=Path, default=None)
  parser.add_argument("--training-runner", type=Path, default=TRAINING_RUNNER)
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
  args.outdir = args.outdir.expanduser().resolve()
  args.training_runner = args.training_runner.expanduser().resolve()
  if not args.run_id:
    args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  if args.profiler_dir is None:
    args.profiler_dir = args.outdir / "profiler"
  else:
    args.profiler_dir = args.profiler_dir.expanduser().resolve()
  if args.manifest_path is None:
    args.manifest_path = args.outdir / "manifest.jsonl"
  else:
    args.manifest_path = args.manifest_path.expanduser().resolve()
  if args.results_path is None:
    args.results_path = args.outdir / "results" / f"shard_{args.shard_index}.jsonl"
  else:
    args.results_path = args.results_path.expanduser().resolve()
  if args.warmup_steps < 0 or args.measured_steps < 1:
    raise ValueError("--warmup-steps must be >= 0 and --measured-steps must be >= 1.")
  if args.repeats is not None and args.repeats < 1:
    raise ValueError("--repeats must be >= 1.")
  if args.no_default_ce and args.include_default_ce:
    raise ValueError("--no-default-ce and --include-default-ce cannot both be set.")
  return args


def main() -> None:
  args = normalize_args(parse_args())
  cases = build_matrix(args)
  validate_matrix(args, cases)
  selected = select_cases(args, cases)

  args.outdir.mkdir(parents=True, exist_ok=True)
  write_jsonl(args.manifest_path, manifest_rows(cases))
  write_csv(args.manifest_path.with_suffix(".csv"), manifest_rows(cases))
  if args.dry_run:
    print(f"manifest_jsonl={args.manifest_path}")
    print(f"manifest_csv={args.manifest_path.with_suffix('.csv')}")
    print(f"total_experiments={len(cases)}")
    print(f"selected_experiments={len(selected)}")
    return

  print(f"total_experiments={len(cases)} selected_experiments={len(selected)}")
  print(f"results_jsonl={args.results_path}")
  for index, case in enumerate(selected, start=1):
    print(
        "run",
        f"{index}/{len(selected)}",
        f"experiment_id={case['experiment_id']}",
        case["case_name"],
        flush=True,
    )
    row = run_case(args, case)
    write_jsonl(args.results_path, [row], append=True)
    print(
        "done",
        row["status"],
        f"experiment_id={row['experiment_id']}",
        f"step_s={row.get('steady_state_mean_step_time_sec')}",
        f"hbm_gib={row.get('xla_planned_hbm_gib_per_chip')}",
        flush=True,
    )


if __name__ == "__main__":
  main()
