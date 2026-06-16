#!/usr/bin/env python3
"""General TPU/GPU mesh/chunk pathology matrix runner.

This extends the CCE-only v5e experiment into a broader question:
do tiny chunk-loop workloads fail generally when mixed with FSDP/TP layout and
collectives?  The default pilot matrix is:

* 4 hardware targets: TPU v5e, A100, L40S, H100
* 4 workload families: CCE training plus three synthetic JAX microbenches
* 4 target signatures: bad/good/control rows from the CCE profiler study

So there are 16 scenario groups and 64 concrete experiment rows.  The
`dense-synthetic` preset keeps CCE out of the default path and expands the
microbench matrix over token/vocab chunk axes so TPU and GPU stacks can be
compared with the same experimental design.
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
import signal
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MICROBENCH_RUNNER = SCRIPT_DIR / "run_mesh_pathology_microbench.py"
TORCH_MICROBENCH_RUNNER = SCRIPT_DIR / "run_mesh_pathology_torchbench.py"
CCE_RUNNER = SCRIPT_DIR / "run_v5e_cce_matrix.py"

HARDWARE_TARGETS: dict[str, dict[str, Any]] = {
    "tpu-v5e-4": {
        "backend": "tpu",
        "accelerator_type": "v5litepod-4",
        "device_count": 4,
        "dstack_gpu_spec": "",
        "role": "tpu_baseline",
    },
    "gpu-a100-80gb-4": {
        "backend": "gpu",
        "accelerator_type": "A100-80GB",
        "device_count": 4,
        "dstack_gpu_spec": "A100:4:80GB",
        "role": "primary_gpu_baseline",
    },
    "gpu-l40s-48gb-4": {
        "backend": "gpu",
        "accelerator_type": "L40S-48GB",
        "device_count": 4,
        "dstack_gpu_spec": "L40S:4:48GB",
        "role": "pcie_communication_stress",
    },
    "gpu-h100-80gb-4": {
        "backend": "gpu",
        "accelerator_type": "H100-80GB",
        "device_count": 4,
        "dstack_gpu_spec": "H100:4:80GB",
        "role": "upper_bound_gpu",
    },
}

WORKLOADS: dict[str, dict[str, Any]] = {
    "cce_train": {
        "runner": "tunix_cce",
        "description": "Existing Tunix CCE training path.",
    },
    "chunked_matmul_loop": {
        "runner": "jax_microbench",
        "description": "Chunk-loop projection matmul without explicit collectives.",
    },
    "collective_loop": {
        "runner": "jax_microbench",
        "description": "Repeated same-shape TP all-reduce over chunk-sized tensors.",
    },
    "projection_collective_loop": {
        "runner": "jax_microbench",
        "description": "Chunk-loop projection matmul plus TP all-reduce.",
    },
    "torch_eager_chunked_matmul_loop": {
        "runner": "torch_microbench",
        "execution_mode": "eager",
        "operation_family": "chunked_matmul_loop",
        "description": "PyTorch eager chunk-loop projection matmul without XLA.",
    },
    "torch_eager_collective_loop": {
        "runner": "torch_microbench",
        "execution_mode": "eager",
        "operation_family": "collective_loop",
        "description": "PyTorch eager repeated NCCL all-reduce over chunk-sized tensors.",
    },
    "torch_eager_projection_collective_loop": {
        "runner": "torch_microbench",
        "execution_mode": "eager",
        "operation_family": "projection_collective_loop",
        "description": "PyTorch eager projection matmul plus NCCL all-reduce.",
    },
    "torch_compile_chunked_matmul_loop": {
        "runner": "torch_microbench",
        "execution_mode": "compile",
        "operation_family": "chunked_matmul_loop",
        "description": "torch.compile/Inductor chunk-loop projection matmul without XLA.",
    },
    "torch_compile_collective_loop": {
        "runner": "torch_microbench",
        "execution_mode": "compile",
        "operation_family": "collective_loop",
        "description": "torch.compile/Inductor repeated NCCL all-reduce.",
    },
    "torch_compile_projection_collective_loop": {
        "runner": "torch_microbench",
        "execution_mode": "compile",
        "operation_family": "projection_collective_loop",
        "description": "torch.compile/Inductor projection matmul plus NCCL all-reduce.",
    },
}

PILOT_DEFAULT_WORKLOADS = {
    "cce_train",
    "chunked_matmul_loop",
    "collective_loop",
    "projection_collective_loop",
}
SYNTHETIC_JAX_WORKLOADS = {
    "chunked_matmul_loop",
    "collective_loop",
    "projection_collective_loop",
}
DENSE_DEFAULT_TOKEN_CHUNKS = [64, 128, 256, 512, 1024]
DENSE_DEFAULT_VOCAB_CHUNKS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
DENSE_DEFAULT_SHAPES = [(16, 512)]
DENSE_DEFAULT_MESH_CONFIGS = [(4, 1), (2, 2), (1, 4)]

TARGET_SIGNATURES = [
    {
        "signature_label": "bad",
        "fsdp_degree": 2,
        "tp_degree": 2,
        "global_batch_size": 16,
        "sequence_length": 512,
        "token_chunk": 128,
        "vocab_chunk": 8192,
        "cce_profiler_experiment_id": 0,
    },
    {
        "signature_label": "good",
        "fsdp_degree": 2,
        "tp_degree": 2,
        "global_batch_size": 16,
        "sequence_length": 512,
        "token_chunk": 512,
        "vocab_chunk": 65536,
        "cce_profiler_experiment_id": 1,
    },
    {
        "signature_label": "control-fsdp4-tp1",
        "fsdp_degree": 4,
        "tp_degree": 1,
        "global_batch_size": 16,
        "sequence_length": 512,
        "token_chunk": 128,
        "vocab_chunk": 8192,
        "cce_profiler_experiment_id": 2,
    },
    {
        "signature_label": "control-fsdp1-tp4",
        "fsdp_degree": 1,
        "tp_degree": 4,
        "global_batch_size": 16,
        "sequence_length": 512,
        "token_chunk": 128,
        "vocab_chunk": 8192,
        "cce_profiler_experiment_id": 3,
    },
]


def build_signatures(args: argparse.Namespace) -> list[dict[str, Any]]:
  if args.preset == "pilot":
    return list(TARGET_SIGNATURES)
  if args.preset != "dense-synthetic":
    raise ValueError(f"Unknown preset: {args.preset}")

  token_chunks = parse_int_csv_values(args.token_chunks, DENSE_DEFAULT_TOKEN_CHUNKS, "token-chunks")
  vocab_chunks = parse_int_csv_values(args.vocab_chunks, DENSE_DEFAULT_VOCAB_CHUNKS, "vocab-chunks")
  shapes = parse_shapes(args.shapes)
  mesh_configs = parse_mesh_configs(args.mesh_configs)
  signatures: list[dict[str, Any]] = []
  skipped = 0
  for global_batch_size, sequence_length in shapes:
    tokens = global_batch_size * sequence_length
    for fsdp_degree, tp_degree in mesh_configs:
      for token_chunk in token_chunks:
        for vocab_chunk in vocab_chunks:
          if tokens % token_chunk != 0 or args.vocab_size % vocab_chunk != 0:
            skipped += 1
            continue
          signatures.append({
              "signature_label": (
                  f"b{global_batch_size}_L{sequence_length}_"
                  f"fsdp{fsdp_degree}_tp{tp_degree}_"
                  f"tc{token_chunk}_vc{vocab_chunk}"
              ),
              "fsdp_degree": fsdp_degree,
              "tp_degree": tp_degree,
              "global_batch_size": global_batch_size,
              "sequence_length": sequence_length,
              "token_chunk": token_chunk,
              "vocab_chunk": vocab_chunk,
              "cce_profiler_experiment_id": -1,
          })
  if not signatures:
    raise ValueError(
        "Dense synthetic preset produced no valid signatures. "
        "Check that token chunks divide batch*sequence and vocab chunks divide vocab size."
    )
  if skipped:
    print(f"skipped_invalid_dense_chunk_configs={skipped}", file=sys.stderr)
  return signatures


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv_values(value: str | None, allowed: set[str], label: str) -> list[str]:
  if not value:
    return sorted(allowed)
  parsed = [item.strip() for item in value.split(",") if item.strip()]
  unknown = [item for item in parsed if item not in allowed]
  if unknown:
    raise ValueError(f"Unknown {label}: {unknown}. Allowed: {sorted(allowed)}")
  return parsed


def parse_int_csv_values(value: str | None, default: list[int], label: str) -> list[int]:
  if not value:
    return list(default)
  parsed = []
  for item in value.split(","):
    item = item.strip()
    if not item:
      continue
    try:
      parsed.append(int(item))
    except ValueError as exc:
      raise ValueError(f"Invalid {label} value {item!r}; expected integers.") from exc
  if not parsed:
    raise ValueError(f"--{label} did not contain any values.")
  if any(item <= 0 for item in parsed):
    raise ValueError(f"All {label} values must be positive: {parsed}")
  return parsed


def parse_shapes(value: str | None) -> list[tuple[int, int]]:
  if not value:
    return list(DENSE_DEFAULT_SHAPES)
  shapes = []
  for item in value.split(","):
    item = item.strip()
    if not item:
      continue
    match = re.fullmatch(r"[bB]?(\d+)\s*(?:/|x|X|:|_)\s*[lL]?(\d+)", item)
    if not match:
      raise ValueError(
          f"Invalid shape {item!r}; use forms such as b16/L512, 16x512, or 16:512."
      )
    batch, sequence = int(match.group(1)), int(match.group(2))
    if batch <= 0 or sequence <= 0:
      raise ValueError(f"Shape values must be positive: {item!r}")
    shapes.append((batch, sequence))
  if not shapes:
    raise ValueError("--shapes did not contain any values.")
  return shapes


def parse_mesh_configs(value: str | None) -> list[tuple[int, int]]:
  if not value:
    return list(DENSE_DEFAULT_MESH_CONFIGS)
  configs = []
  compact_value = value.replace(" ", "")
  fsdp_tp_pattern = r"fsdp=(\d+),tp=(\d+)"
  items = (
      compact_value.split(";")
      if ";" in compact_value
      else [compact_value]
      if re.fullmatch(fsdp_tp_pattern, compact_value)
      else compact_value.split(",")
  )
  for item in items:
    item = item.strip()
    if not item:
      continue
    match = (
        re.fullmatch(r"(\d+)x(\d+)", item)
        or re.fullmatch(fsdp_tp_pattern, item)
        or re.fullmatch(r"fsdp(\d+)-tp(\d+)", item)
    )
    if not match:
      raise ValueError(
          f"Invalid mesh config {item!r}; use forms such as 4x1, fsdp=2,tp=2, or fsdp2-tp2."
      )
    fsdp, tp = int(match.group(1)), int(match.group(2))
    if fsdp <= 0 or tp <= 0:
      raise ValueError(f"Mesh degrees must be positive: {item!r}")
    configs.append((fsdp, tp))
  if not configs:
    raise ValueError("--mesh-configs did not contain any values.")
  return configs


def default_workloads_for_preset(preset: str) -> set[str]:
  if preset == "pilot":
    return set(PILOT_DEFAULT_WORKLOADS)
  if preset == "dense-synthetic":
    return set(SYNTHETIC_JAX_WORKLOADS)
  raise ValueError(f"Unknown preset: {preset}")


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  rows = []
  with path.open() as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def case_name(case: dict[str, Any]) -> str:
  return (
      f"exp{case['experiment_id']:04d}_"
      f"{case['hardware_target']}_"
      f"{case['workload_family']}_"
      f"{case['signature_label']}_"
      f"fsdp{case['fsdp_degree']}_tp{case['tp_degree']}_"
      f"tc{case['token_chunk']}_vc{case['vocab_chunk']}_"
      f"r{case['repeat_index']}"
  )


def base_case(
    *,
    args: argparse.Namespace,
    experiment_id: int,
    hardware_target: str,
    workload_family: str,
    signature: dict[str, Any],
    repeat_index: int,
) -> dict[str, Any]:
  hardware = HARDWARE_TARGETS[hardware_target]
  workload = WORKLOADS[workload_family]
  tokens = signature["global_batch_size"] * signature["sequence_length"]
  token_loop_count = math.ceil(tokens / signature["token_chunk"])
  vocab_loop_count = math.ceil(args.vocab_size / signature["vocab_chunk"])
  loop_count = token_loop_count * vocab_loop_count
  row = {
      "experiment_id": experiment_id,
      "run_id": args.run_id,
      "preset": args.preset,
      "timestamp_manifest": utc_now(),
      "hostname_manifest": socket.gethostname(),
      "hardware_target": hardware_target,
      "hardware_role": hardware["role"],
      "backend": hardware["backend"],
      "accelerator_type": hardware["accelerator_type"],
      "device_count": hardware["device_count"],
      "dstack_gpu_spec": hardware["dstack_gpu_spec"],
      "workload_family": workload_family,
      "workload_runner": workload["runner"],
      "workload_description": workload["description"],
      "operation_family": workload.get("operation_family", workload_family),
      "execution_mode": workload.get("execution_mode", ""),
      "signature_label": signature["signature_label"],
      "fsdp_degree": signature["fsdp_degree"],
      "tp_degree": signature["tp_degree"],
      "global_batch_size": signature["global_batch_size"],
      "sequence_length": signature["sequence_length"],
      "shape": f"b{signature['global_batch_size']}/L{signature['sequence_length']}",
      "hidden_size": args.hidden_size,
      "vocab_size": args.vocab_size,
      "token_chunk": signature["token_chunk"],
      "vocab_chunk": signature["vocab_chunk"],
      "chunk_configuration": f"{signature['token_chunk']}/{signature['vocab_chunk']}",
      "token_loop_count": token_loop_count,
      "vocab_loop_count": vocab_loop_count,
      "chunk_loop_count": loop_count,
      "cce_loop_count": loop_count if workload_family == "cce_train" else "",
      "mesh_configuration": f"fsdp={signature['fsdp_degree']},tp={signature['tp_degree']}",
      "warmup_steps": args.warmup_steps,
      "measured_steps": args.measured_steps,
      "seed": args.seed + repeat_index,
      "repeat_index": repeat_index,
      "cce_profiler_experiment_id": signature["cce_profiler_experiment_id"],
  }
  row["case_name"] = case_name(row)
  row["run_dir"] = str((args.outdir / "runs" / row["case_name"]).resolve())
  row["xla_dir"] = str((Path(row["run_dir"]) / "xla").resolve())
  row["profiler_path"] = (
      str((args.outdir / "profiler" / row["case_name"]).resolve())
      if args.enable_profiler
      else ""
  )
  row["scenario_group"] = f"{hardware_target}/{workload_family}"
  return row


def build_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
  hardware_targets = parse_csv_values(
      args.hardware_targets,
      set(HARDWARE_TARGETS),
      "hardware target",
  )
  workloads = (
      parse_csv_values(args.workloads, set(WORKLOADS), "workload")
      if args.workloads
      else sorted(default_workloads_for_preset(args.preset))
  )
  signatures = build_signatures(args)
  cases = []
  experiment_id = 0
  for repeat_index in range(args.repeats):
    for hardware_target in hardware_targets:
      for workload_family in workloads:
        for signature in signatures:
          cases.append(
              base_case(
                  args=args,
                  experiment_id=experiment_id,
                  hardware_target=hardware_target,
                  workload_family=workload_family,
                  signature=signature,
                  repeat_index=repeat_index,
              )
          )
          experiment_id += 1
  return cases


def validate_matrix(args: argparse.Namespace, cases: list[dict[str, Any]]) -> None:
  if args.num_shards < 1:
    raise ValueError("--num-shards must be >= 1.")
  if args.shard_index < 0 or args.shard_index >= args.num_shards:
    raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
  for case in cases:
    if case["fsdp_degree"] * case["tp_degree"] != case["device_count"]:
      raise ValueError(
          f"{case['case_name']} mesh does not match device_count: "
          f"{case['fsdp_degree']} * {case['tp_degree']} != {case['device_count']}"
      )
    if case["workload_runner"] == "torch_microbench" and case["backend"] != "gpu":
      raise ValueError(f"{case['case_name']} is a PyTorch GPU workload but backend={case['backend']}.")
    if case["workload_runner"] == "tunix_cce" and args.preset != "pilot":
      raise ValueError(
          f"{case['case_name']} uses cce_train, but dense CCE rows are not implemented. "
          "Use --preset pilot for CCE profiler cases, or synthetic workloads for dense sweeps."
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
  pythonpath = str(REPO_ROOT)
  if env.get("PYTHONPATH"):
    pythonpath += os.pathsep + env["PYTHONPATH"]
  env["PYTHONPATH"] = pythonpath
  env["PYTHONUNBUFFERED"] = "1"
  if case["workload_runner"] in {"jax_microbench", "tunix_cce"} and not args.disable_xla_dump:
    xla_flags = env.get("XLA_FLAGS", "").strip()
    dump_parts = [
        f"--xla_dump_to={case['xla_dir']}",
        "--xla_dump_hlo_as_text",
    ]
    if args.full_hlo_dump:
      dump_parts.extend([
          "--xla_dump_hlo_as_proto",
          "--xla_dump_hlo_pass_re=.*",
      ])
    env["XLA_FLAGS"] = f"{xla_flags} {' '.join(dump_parts)}".strip()
  return env


def microbench_command(case: dict[str, Any]) -> list[str]:
  command = [
      sys.executable,
      str(MICROBENCH_RUNNER),
      "--workload",
      case["workload_family"],
      "--backend",
      case["backend"],
      "--hardware-target",
      case["hardware_target"],
      "--accelerator-type",
      case["accelerator_type"],
      "--mesh-fsdp",
      str(case["fsdp_degree"]),
      "--mesh-tp",
      str(case["tp_degree"]),
      "--global-batch-size",
      str(case["global_batch_size"]),
      "--sequence-length",
      str(case["sequence_length"]),
      "--hidden-size",
      str(case["hidden_size"]),
      "--vocab-size",
      str(case["vocab_size"]),
      "--token-chunk",
      str(case["token_chunk"]),
      "--vocab-chunk",
      str(case["vocab_chunk"]),
      "--warmup-steps",
      str(case["warmup_steps"]),
      "--measured-steps",
      str(case["measured_steps"]),
      "--seed",
      str(case["seed"]),
      "--outdir",
      case["run_dir"],
      "--xla-dir",
      case["xla_dir"],
  ]
  if case["profiler_path"]:
    command.extend(["--profiler-dir", case["profiler_path"]])
  return command


def torch_microbench_command(case: dict[str, Any]) -> list[str]:
  return [
      sys.executable,
      "-m",
      "torch.distributed.run",
      "--standalone",
      "--nnodes",
      "1",
      "--nproc_per_node",
      str(case["device_count"]),
      str(TORCH_MICROBENCH_RUNNER),
      "--workload",
      case["operation_family"],
      "--execution-mode",
      case["execution_mode"],
      "--backend",
      case["backend"],
      "--hardware-target",
      case["hardware_target"],
      "--accelerator-type",
      case["accelerator_type"],
      "--mesh-fsdp",
      str(case["fsdp_degree"]),
      "--mesh-tp",
      str(case["tp_degree"]),
      "--global-batch-size",
      str(case["global_batch_size"]),
      "--sequence-length",
      str(case["sequence_length"]),
      "--hidden-size",
      str(case["hidden_size"]),
      "--vocab-size",
      str(case["vocab_size"]),
      "--token-chunk",
      str(case["token_chunk"]),
      "--vocab-chunk",
      str(case["vocab_chunk"]),
      "--warmup-steps",
      str(case["warmup_steps"]),
      "--measured-steps",
      str(case["measured_steps"]),
      "--seed",
      str(case["seed"]),
      "--outdir",
      case["run_dir"],
  ]


def cce_command(args: argparse.Namespace, case: dict[str, Any], cce_results_path: Path) -> list[str]:
  command = [
      sys.executable,
      str(CCE_RUNNER),
      "--preset",
      "profiler",
      "--backend",
      case["hardware_target"],
      "--chips",
      str(case["device_count"]),
      "--tpu-type",
      case["accelerator_type"],
      "--experiment-id",
      str(case["cce_profiler_experiment_id"]),
      "--warmup-steps",
      str(case["warmup_steps"]),
      "--measured-steps",
      str(case["measured_steps"]),
      "--outdir",
      str((Path(case["run_dir"]) / "cce_delegate").resolve()),
      "--results-path",
      str(cce_results_path.resolve()),
      "--force",
      "--model-size",
      args.cce_model_size,
  ]
  if args.cce_model_id:
    command.extend(["--model-id", args.cce_model_id])
  if args.cce_model_source:
    command.extend(["--model-source", args.cce_model_source])
  if args.cce_model_path is not None:
    command.extend(["--model-path", args.cce_model_path])
  if args.cce_tokenizer_source:
    command.extend(["--tokenizer-source", args.cce_tokenizer_source])
  if args.cce_tokenizer_path is not None:
    command.extend(["--tokenizer-path", args.cce_tokenizer_path])
  if args.cce_allow_download:
    command.append("--allow-download")
  if args.enable_profiler:
    command.append("--enable-profiler")
  if args.keep_all_xla:
    command.append("--keep-all-xla")
  if args.full_hlo_dump:
    command.append("--full-hlo-dump")
  if args.disable_xla_dump:
    command.append("--disable-xla-dump")
  return command


def parse_failure(log_path: Path) -> dict[str, Any]:
  text = log_path.read_text(errors="ignore") if log_path.exists() else ""
  tail = "\n".join(line.strip() for line in text.splitlines() if line.strip())[-4000:]
  return {
      "failure_type": "error" if "Traceback" in text else "unknown",
      "error_message": tail,
  }


def run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Any,
    timeout_sec: int,
) -> tuple[subprocess.CompletedProcess, bool]:
  """Run a command and kill the whole process group on timeout."""
  timeout = timeout_sec if timeout_sec > 0 else None
  proc = subprocess.Popen(
      command,
      cwd=cwd,
      env=env,
      stdout=log,
      stderr=subprocess.STDOUT,
      start_new_session=True,
  )
  try:
    returncode = proc.wait(timeout=timeout)
    return subprocess.CompletedProcess(command, returncode=returncode), False
  except subprocess.TimeoutExpired:
    log.write(f"\nTIMEOUT after {timeout_sec} seconds; terminating process group\n")
    log.flush()
    try:
      os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
      pass
    except Exception:
      proc.terminate()
    try:
      proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
      log.write("Process group did not exit after SIGTERM; sending SIGKILL\n")
      log.flush()
      try:
        os.killpg(proc.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      except Exception:
        proc.kill()
      proc.wait()
    return subprocess.CompletedProcess(command, returncode=124), True


def run_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
  run_dir = Path(case["run_dir"])
  if run_dir.exists() and args.force:
    shutil.rmtree(run_dir)
  run_dir.mkdir(parents=True, exist_ok=True)
  Path(case["xla_dir"]).mkdir(parents=True, exist_ok=True)
  log_path = run_dir / "runner.log"
  started = time.monotonic()

  if case["workload_runner"] == "jax_microbench":
    command = microbench_command(case)
    result_path = run_dir / "summary.json"
  elif case["workload_runner"] == "torch_microbench":
    command = torch_microbench_command(case)
    result_path = run_dir / "summary.json"
  else:
    cce_results_path = run_dir / "cce_delegate_result.jsonl"
    command = cce_command(args, case, cce_results_path)
    result_path = cce_results_path

  with log_path.open("w") as log:
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    proc, timed_out = run_subprocess(
        command,
        cwd=REPO_ROOT,
        env=configure_env(args, case),
        log=log,
        timeout_sec=args.case_timeout_sec,
    )
  elapsed = time.monotonic() - started

  row = {
      **case,
      "timestamp": utc_now(),
      "hostname": socket.gethostname(),
      "status": "failure",
      "returncode": proc.returncode,
      "elapsed_sec": elapsed,
      "log_path": str(log_path.resolve()),
  }
  if proc.returncode == 0 and result_path.exists():
    if case["workload_runner"] in {"jax_microbench", "torch_microbench"}:
      summary = json.loads(result_path.read_text())
    else:
      rows = read_jsonl(result_path)
      summary = rows[-1] if rows else {}
    row.update(summary)
    row["status"] = summary.get("status", "success")
  else:
    row.update(parse_failure(log_path))
    if timed_out:
      row["failure_type"] = "timeout"
      row["error_message"] = f"Case exceeded timeout_sec={args.case_timeout_sec}."
  (run_dir / "case_summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
  return row


def dstack_command_rows(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
  rows = []
  seen = set()
  for case in cases:
    if case["backend"] != "gpu":
      continue
    key = (case["hardware_target"], case["dstack_gpu_spec"])
    if key in seen:
      continue
    seen.add(key)
    outdir = f"/tmp/mesh-pathology-{case['hardware_target']}"
    command = (
        "uvx --from dstack dstack apply -f 01-CCE/dstack_mesh_pathology_gpu.yml "
        f"--gpu {case['dstack_gpu_spec']} "
        f"-n mesh-pathology-{case['hardware_target']} -- "
        f"--hardware-targets {case['hardware_target']} --outdir {outdir}"
    )
    offer = (
        "uvx --from dstack dstack offer "
        f"--gpu {case['dstack_gpu_spec']} --max-offers 5"
    )
    rows.append({
        "hardware_target": case["hardware_target"],
        "dstack_gpu_spec": case["dstack_gpu_spec"],
        "offer_command": offer,
        "apply_command": command,
        "note": "Inspect offers before apply; apply provisions paid GPU capacity.",
    })
  return rows


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--preset", choices=["pilot", "dense-synthetic"], default="pilot")
  parser.add_argument("--hardware-targets", default=None)
  parser.add_argument("--workloads", default=None)
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument(
      "--shapes",
      default=None,
      help="Dense preset shapes, e.g. b16/L512,b16/L1024,b32/L512.",
  )
  parser.add_argument(
      "--mesh-configs",
      default=None,
      help="Dense preset mesh configs, e.g. 4x1,2x2,1x4 or fsdp=2,tp=2.",
  )
  parser.add_argument(
      "--token-chunks",
      default=None,
      help="Dense preset token chunks, e.g. 64,128,256,512,1024.",
  )
  parser.add_argument(
      "--vocab-chunks",
      default=None,
      help="Dense preset vocab chunks, e.g. 4096,8192,16384,32768,65536,131072,262144.",
  )
  parser.add_argument("--hidden-size", type=int, default=320)
  parser.add_argument("--vocab-size", type=int, default=262144)
  parser.add_argument("--cce-model-size", default="270m")
  parser.add_argument("--cce-model-id", default="")
  parser.add_argument("--cce-model-source", choices=["gcs", "huggingface"], default=None)
  parser.add_argument("--cce-model-path", default=None)
  parser.add_argument(
      "--cce-tokenizer-source",
      choices=["sentencepiece", "huggingface"],
      default=None,
  )
  parser.add_argument("--cce-tokenizer-path", default=None)
  parser.add_argument("--cce-allow-download", action="store_true")
  parser.add_argument("--warmup-steps", type=int, default=3)
  parser.add_argument("--measured-steps", type=int, default=10)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--enable-profiler", action="store_true")
  parser.add_argument("--disable-xla-dump", action="store_true")
  parser.add_argument("--full-hlo-dump", action="store_true")
  parser.add_argument("--keep-all-xla", action="store_true")
  parser.add_argument("--shard-index", type=int, default=0)
  parser.add_argument("--num-shards", type=int, default=1)
  parser.add_argument("--experiment-id", type=int, action="append", default=[])
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--case-timeout-sec", type=int, default=0)
  parser.add_argument("--run-id", default="")
  parser.add_argument(
      "--outdir",
      type=Path,
      default=SCRIPT_DIR / "results" / "mesh-pathology-matrix",
  )
  parser.add_argument("--manifest-path", type=Path, default=None)
  parser.add_argument("--results-path", type=Path, default=None)
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--write-dstack-commands", action="store_true")
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
  args.outdir = args.outdir.expanduser().resolve()
  if not args.run_id:
    args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  if args.manifest_path is None:
    args.manifest_path = args.outdir / "manifest.jsonl"
  else:
    args.manifest_path = args.manifest_path.expanduser().resolve()
  if args.results_path is None:
    args.results_path = args.outdir / "results" / f"shard_{args.shard_index}.jsonl"
  else:
    args.results_path = args.results_path.expanduser().resolve()
  if args.repeats < 1:
    raise ValueError("--repeats must be >= 1.")
  if args.warmup_steps < 0 or args.measured_steps < 1:
    raise ValueError("--warmup-steps must be >= 0 and --measured-steps must be >= 1.")
  return args


def main() -> None:
  args = normalize_args(parse_args())
  cases = build_matrix(args)
  validate_matrix(args, cases)
  selected = select_cases(args, cases)
  args.outdir.mkdir(parents=True, exist_ok=True)
  write_jsonl(args.manifest_path, cases)
  write_csv(args.manifest_path.with_suffix(".csv"), cases)

  if args.write_dstack_commands:
    rows = dstack_command_rows(cases, args)
    path = args.outdir / "dstack_commands.csv"
    write_csv(path, rows)
    print(f"dstack_commands={path}")

  if args.dry_run:
    scenario_groups = {(case["hardware_target"], case["workload_family"]) for case in cases}
    print(f"manifest_jsonl={args.manifest_path}")
    print(f"manifest_csv={args.manifest_path.with_suffix('.csv')}")
    print(f"scenario_groups={len(scenario_groups)}")
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
        row.get("status"),
        f"experiment_id={row['experiment_id']}",
        f"step_s={row.get('steady_state_mean_step_time_sec')}",
        flush=True,
    )


if __name__ == "__main__":
  main()
