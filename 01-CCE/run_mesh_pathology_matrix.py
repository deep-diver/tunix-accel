#!/usr/bin/env python3
"""General TPU/GPU mesh/chunk pathology matrix runner.

This extends the CCE-only v5e experiment into a broader question:
do tiny chunk-loop workloads fail generally when mixed with FSDP/TP layout and
collectives?  The default pilot matrix is:

* 4 hardware targets: TPU v5e, A100, L40S, H100
* 4 workload families: CCE training plus three synthetic JAX microbenches
* 4 target signatures: bad/good/control rows from the CCE profiler study

So there are 16 scenario groups and 64 concrete experiment rows.
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
import subprocess
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MICROBENCH_RUNNER = SCRIPT_DIR / "run_mesh_pathology_microbench.py"
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
}

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
  workloads = parse_csv_values(args.workloads, set(WORKLOADS), "workload")
  cases = []
  experiment_id = 0
  for repeat_index in range(args.repeats):
    for hardware_target in hardware_targets:
      for workload_family in workloads:
        for signature in TARGET_SIGNATURES:
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
  if not args.disable_xla_dump:
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
  else:
    cce_results_path = run_dir / "cce_delegate_result.jsonl"
    command = cce_command(args, case, cce_results_path)
    result_path = cce_results_path

  with log_path.open("w") as log:
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=configure_env(args, case),
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
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
    if case["workload_runner"] == "jax_microbench":
      summary = json.loads(result_path.read_text())
    else:
      rows = read_jsonl(result_path)
      summary = rows[-1] if rows else {}
    row.update(summary)
    row["status"] = summary.get("status", "success")
  else:
    row.update(parse_failure(log_path))
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
  parser.add_argument("--preset", choices=["pilot"], default="pilot")
  parser.add_argument("--hardware-targets", default=None)
  parser.add_argument("--workloads", default=None)
  parser.add_argument("--repeats", type=int, default=1)
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
