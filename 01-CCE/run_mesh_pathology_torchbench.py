#!/usr/bin/env python3
"""PyTorch GPU microbenchmarks for mesh/chunk pathology checks.

This runner mirrors `run_mesh_pathology_microbench.py`, but deliberately avoids
JAX/XLA. On NVIDIA GPUs it uses PyTorch eager or `torch.compile`/Inductor, CUDA
matmul kernels, and NCCL collectives through `torch.distributed`.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time
from typing import Any, Callable

import torch
import torch.distributed as dist


WORKLOADS = {
    "chunked_matmul_loop",
    "collective_loop",
    "projection_collective_loop",
}


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def percentile_nearest(values: list[float], q: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  index = math.ceil(q / 100.0 * len(ordered)) - 1
  index = min(max(index, 0), len(ordered) - 1)
  return ordered[index]


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


def distributed_context() -> tuple[int, int, int]:
  rank = int(os.environ.get("RANK", "0"))
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  if world_size > 1 and not dist.is_initialized():
    dist.init_process_group(backend="nccl")
  torch.cuda.set_device(local_rank)
  return rank, local_rank, world_size


def tp_group_for_rank(rank: int, fsdp: int, tp: int, world_size: int) -> dist.ProcessGroup | None:
  if world_size == 1 or tp == 1:
    return None
  needed = fsdp * tp
  if world_size < needed:
    raise ValueError(f"Need {needed} ranks for fsdp={fsdp},tp={tp}; got {world_size}.")
  groups = []
  for fsdp_index in range(fsdp):
    ranks = [fsdp_index * tp + tp_index for tp_index in range(tp)]
    groups.append((ranks, dist.new_group(ranks=ranks, backend="nccl")))
  for ranks, group in groups:
    if rank in ranks:
      return group
  return None


def cuda_sync() -> None:
  torch.cuda.synchronize()
  if dist.is_available() and dist.is_initialized():
    dist.barrier()


def max_rank_elapsed(elapsed: float, device: torch.device) -> float:
  if not dist.is_available() or not dist.is_initialized():
    return elapsed
  tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
  dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
  return float(tensor.item())


def runtime_memory_gb() -> tuple[float | None, float | None]:
  if not torch.cuda.is_available():
    return None, None
  peak = torch.cuda.max_memory_allocated() / 1e9
  props = torch.cuda.get_device_properties(torch.cuda.current_device())
  return peak, props.total_memory / 1e9


def step_stats(history: list[dict[str, Any]], warmup_steps: int, measured_steps: int) -> dict[str, Any]:
  all_times = [float(row["step_time_sec"]) for row in history]
  measured = all_times[warmup_steps : warmup_steps + measured_steps]
  median = statistics.median(measured) if measured else None
  first = all_times[0] if all_times else None
  compile_time = max(first - median, 0.0) if first is not None and median is not None else None
  return {
      "steps_recorded": len(history),
      "first_step_time_sec": first,
      "compile_time_sec": compile_time,
      "compile_time_estimate_method": "max(first_step_time - measured_median, 0)"
      if compile_time is not None
      else "",
      "steady_state_mean_step_time_sec": statistics.mean(measured) if measured else None,
      "median_step_time_sec": median,
      "p95_step_time_sec": percentile_nearest(measured, 95),
      "measured_step_count": len(measured),
      "measured_time_sec": sum(measured) if measured else None,
  }


def make_projection_step(
    *,
    workload: str,
    token_chunk: int,
    vocab_chunk: int,
    token_loop_count: int,
    vocab_loop_count: int,
    hidden_size: int,
    tp_group: dist.ProcessGroup | None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
  def step(hidden: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for token_index in range(token_loop_count):
      hidden_chunk = hidden[
          token_index * token_chunk : (token_index + 1) * token_chunk,
          :hidden_size,
      ]
      for vocab_index in range(vocab_loop_count):
        head_chunk = head[
            :hidden_size,
            vocab_index * vocab_chunk : (vocab_index + 1) * vocab_chunk,
        ]
        logits = hidden_chunk @ head_chunk
        if workload == "projection_collective_loop" and tp_group is not None:
          dist.all_reduce(logits, op=dist.ReduceOp.SUM, group=tp_group)
        acc = acc + torch.tanh(logits.float()).sum() * 1e-7
    return acc

  return step


def make_collective_step(
    *,
    loop_count: int,
    tp_group: dist.ProcessGroup | None,
) -> Callable[[torch.Tensor], torch.Tensor]:
  def step(tile: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros((), dtype=torch.float32, device=tile.device)
    for _ in range(loop_count):
      reduced = tile
      if tp_group is not None:
        reduced = tile.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=tp_group)
      acc = acc + reduced.float().sum() * 1e-7
    return acc

  return step


def compile_step(fn: Callable[..., torch.Tensor], args: argparse.Namespace) -> Callable[..., torch.Tensor]:
  if args.execution_mode != "compile":
    return fn
  return torch.compile(
      fn,
      backend=args.torch_compile_backend,
      mode=args.torch_compile_mode or None,
      fullgraph=False,
      dynamic=False,
  )


def run(args: argparse.Namespace) -> dict[str, Any]:
  if args.global_batch_size * args.sequence_length % args.token_chunk:
    raise ValueError("global_batch_size * sequence_length must be divisible by token_chunk.")
  if args.vocab_size % args.vocab_chunk:
    raise ValueError("vocab_size must be divisible by vocab_chunk.")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this PyTorch GPU microbench.")

  rank, local_rank, world_size = distributed_context()
  device = torch.device(f"cuda:{local_rank}")
  tp_group = tp_group_for_rank(rank, args.mesh_fsdp, args.mesh_tp, world_size)
  torch.manual_seed(args.seed + rank)
  torch.cuda.manual_seed_all(args.seed + rank)
  torch.set_float32_matmul_precision(args.float32_matmul_precision)

  outdir = args.outdir.expanduser().resolve()
  if rank == 0:
    outdir.mkdir(parents=True, exist_ok=True)

  tokens = args.global_batch_size * args.sequence_length
  token_loop_count = math.ceil(tokens / args.token_chunk)
  vocab_loop_count = math.ceil(args.vocab_size / args.vocab_chunk)
  loop_count = token_loop_count * vocab_loop_count

  if args.workload == "collective_loop":
    tile = torch.randn(
        (args.token_chunk, args.vocab_chunk),
        dtype=torch.bfloat16,
        device=device,
    )
    step = make_collective_step(loop_count=loop_count, tp_group=tp_group)
    step = compile_step(step, args)
    call = lambda: step(tile)
  else:
    hidden = torch.randn(
        (tokens, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    head = torch.randn(
        (args.hidden_size, args.vocab_size),
        dtype=torch.bfloat16,
        device=device,
    )
    step = make_projection_step(
        workload=args.workload,
        token_chunk=args.token_chunk,
        vocab_chunk=args.vocab_chunk,
        token_loop_count=token_loop_count,
        vocab_loop_count=vocab_loop_count,
        hidden_size=args.hidden_size,
        tp_group=tp_group,
    )
    step = compile_step(step, args)
    call = lambda: step(hidden, head)

  cuda_sync()
  torch.cuda.reset_peak_memory_stats(device)
  history: list[dict[str, Any]] = []
  for step_index in range(args.warmup_steps + args.measured_steps):
    cuda_sync()
    started = time.perf_counter()
    value = call()
    cuda_sync()
    elapsed = max_rank_elapsed(time.perf_counter() - started, device)
    if rank == 0:
      history.append({
          "step": step_index,
          "phase": "warmup" if step_index < args.warmup_steps else "measured",
          "step_time_sec": elapsed,
          "result": float(value.detach().float().cpu().item()),
      })

  runtime_peak, runtime_limit = runtime_memory_gb()
  if rank != 0:
    return {}

  write_csv(outdir / "history.csv", history)
  stats = step_stats(history, args.warmup_steps, args.measured_steps)
  measured_time = stats.get("measured_time_sec") or 0.0
  tokens_per_sec = tokens * args.measured_steps / measured_time if measured_time > 0 else None
  row = {
      "timestamp": utc_now(),
      "hostname": socket.gethostname(),
      "backend": args.backend,
      "hardware_target": args.hardware_target,
      "accelerator_type": args.accelerator_type,
      "workload_family": f"torch_{args.execution_mode}_{args.workload}",
      "operation_family": args.workload,
      "framework": "pytorch",
      "compiler_stack": "torch_inductor" if args.execution_mode == "compile" else "pytorch_eager",
      "execution_mode": args.execution_mode,
      "status": "success",
      "torch_version": torch.__version__,
      "torch_cuda_version": torch.version.cuda,
      "torch_compile_backend": args.torch_compile_backend if args.execution_mode == "compile" else "",
      "torch_compile_mode": args.torch_compile_mode if args.execution_mode == "compile" else "",
      "cuda_device_count": torch.cuda.device_count(),
      "world_size": world_size,
      "local_rank": local_rank,
      "rank": rank,
      "mesh_configuration": f"fsdp={args.mesh_fsdp},tp={args.mesh_tp}",
      "fsdp_degree": args.mesh_fsdp,
      "tp_degree": args.mesh_tp,
      "global_batch_size": args.global_batch_size,
      "sequence_length": args.sequence_length,
      "shape": f"b{args.global_batch_size}/L{args.sequence_length}",
      "hidden_size": args.hidden_size,
      "vocab_size": args.vocab_size,
      "token_chunk": args.token_chunk,
      "vocab_chunk": args.vocab_chunk,
      "chunk_configuration": f"{args.token_chunk}/{args.vocab_chunk}",
      "token_loop_count": token_loop_count,
      "vocab_loop_count": vocab_loop_count,
      "chunk_loop_count": loop_count,
      "warmup_steps": args.warmup_steps,
      "measured_steps": args.measured_steps,
      "tokens_per_sec": tokens_per_sec,
      "xla_planned_hbm_gib_per_chip": None,
      "xla_report_path": "",
      "runtime_hbm_peak_gb": runtime_peak,
      "runtime_hbm_limit_gb": runtime_limit,
      "profiler_path": "",
      **stats,
  }
  (outdir / "summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
  return row


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
  parser.add_argument("--execution-mode", choices=["eager", "compile"], required=True)
  parser.add_argument("--backend", default="")
  parser.add_argument("--hardware-target", default="")
  parser.add_argument("--accelerator-type", default="")
  parser.add_argument("--mesh-fsdp", type=int, default=1)
  parser.add_argument("--mesh-tp", type=int, default=1)
  parser.add_argument("--global-batch-size", type=int, default=16)
  parser.add_argument("--sequence-length", type=int, default=512)
  parser.add_argument("--hidden-size", type=int, default=320)
  parser.add_argument("--vocab-size", type=int, default=262144)
  parser.add_argument("--token-chunk", type=int, default=128)
  parser.add_argument("--vocab-chunk", type=int, default=8192)
  parser.add_argument("--warmup-steps", type=int, default=3)
  parser.add_argument("--measured-steps", type=int, default=10)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--torch-compile-backend", default="inductor")
  parser.add_argument("--torch-compile-mode", default="")
  parser.add_argument("--float32-matmul-precision", default="high")
  parser.add_argument("--outdir", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  row = run(parse_args())
  if row:
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
  main()
