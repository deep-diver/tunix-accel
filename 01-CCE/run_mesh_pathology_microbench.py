#!/usr/bin/env python3
"""JAX microbenchmarks for chunk-loop and mesh collective pathologies.

The benchmark is intentionally synthetic.  It isolates the ingredients that
made the CCE bad row suspicious:

* many small chunk-loop iterations,
* loss-head/projection-shaped matmuls,
* tensor-parallel collectives inside the loop,
* the same code path on TPU and GPU.
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
import socket
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P


WORKLOADS = {
    "chunked_matmul_loop",
    "collective_loop",
    "projection_collective_loop",
}

MEM_TOTAL_RE = re.compile(
    r"Total bytes(?: used)?:\s+(\d+)(?:\s+\(([\d.]+)([KMGT]iB)\))?"
)
UNIT_TO_GIB = {
    "KiB": 1.0 / (1024**2),
    "MiB": 1.0 / 1024,
    "GiB": 1.0,
    "TiB": 1024.0,
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


def parse_xla_total_gib(xla_dir: Path | None) -> tuple[float | None, str]:
  if not xla_dir or not xla_dir.exists():
    return None, ""
  reports = sorted(xla_dir.glob("*memory-usage-report.txt"))
  if not reports:
    return None, ""
  report = reports[-1]
  text = report.read_text(errors="ignore")
  match = MEM_TOTAL_RE.search(text)
  if match:
    if match.group(2) and match.group(3):
      return float(match.group(2)) * UNIT_TO_GIB[match.group(3)], str(report.resolve())
    return int(match.group(1)) / (1024**3), str(report.resolve())
  return None, str(report.resolve())


def runtime_memory_gb() -> tuple[float | None, float | None]:
  peaks = []
  limits = []
  for device in jax.devices():
    try:
      stats = device.memory_stats() or {}
    except Exception:
      stats = {}
    peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
    limit = stats.get("bytes_limit")
    if peak:
      peaks.append(float(peak))
    if limit:
      limits.append(float(limit))
  peak_gb = max(peaks) / 1e9 if peaks else None
  limit_gb = max(limits) / 1e9 if limits else None
  return peak_gb, limit_gb


def make_mesh(fsdp: int, tp: int) -> Mesh:
  devices = np.array(jax.devices())
  needed = fsdp * tp
  if len(devices) < needed:
    raise ValueError(
        f"Need at least {needed} JAX devices for fsdp={fsdp},tp={tp}; "
        f"found {len(devices)} devices: {devices.tolist()}"
    )
  return Mesh(devices[:needed].reshape((fsdp, tp)), ("fsdp", "tp"))


def make_projection_step(
    *,
    mesh: Mesh,
    workload: str,
    token_chunk: int,
    vocab_chunk: int,
    token_loop_count: int,
    vocab_loop_count: int,
    hidden_size: int,
):
  def step(hidden: jax.Array, head: jax.Array) -> jax.Array:
    def token_body(i: int, acc: jax.Array) -> jax.Array:
      hidden_chunk = jax.lax.dynamic_slice(
          hidden,
          (i * token_chunk, 0),
          (token_chunk, hidden_size),
      )

      def vocab_body(j: int, inner_acc: jax.Array) -> jax.Array:
        head_chunk = jax.lax.dynamic_slice(
            head,
            (0, j * vocab_chunk),
            (hidden_size, vocab_chunk),
        )
        logits = hidden_chunk @ head_chunk
        if workload == "projection_collective_loop":
          logits = jax.lax.psum(logits, "tp")
        return inner_acc + jnp.sum(jnp.tanh(logits.astype(jnp.float32))) * 1e-7

      return jax.lax.fori_loop(0, vocab_loop_count, vocab_body, acc)

    return jax.lax.fori_loop(
        0,
        token_loop_count,
        token_body,
        jnp.array(0.0, dtype=jnp.float32),
    )

  mapped = jax.shard_map(
      step,
      mesh=mesh,
      in_specs=(P(), P()),
      out_specs=P(),
      axis_names={"fsdp", "tp"},
  )
  return jax.jit(mapped)


def make_collective_step(
    *,
    mesh: Mesh,
    token_chunk: int,
    vocab_chunk: int,
    loop_count: int,
):
  def step(tile: jax.Array) -> jax.Array:
    def body(_: int, acc: jax.Array) -> jax.Array:
      reduced = jax.lax.psum(tile, "tp")
      return acc + jnp.sum(reduced.astype(jnp.float32)) * 1e-7

    return jax.lax.fori_loop(
        0,
        loop_count,
        body,
        jnp.array(0.0, dtype=jnp.float32),
    )

  del token_chunk, vocab_chunk
  mapped = jax.shard_map(
      step,
      mesh=mesh,
      in_specs=P(),
      out_specs=P(),
      axis_names={"fsdp", "tp"},
  )
  return jax.jit(mapped)


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


def run(args: argparse.Namespace) -> dict[str, Any]:
  if args.global_batch_size * args.sequence_length % args.token_chunk:
    raise ValueError("global_batch_size * sequence_length must be divisible by token_chunk.")
  if args.vocab_size % args.vocab_chunk:
    raise ValueError("vocab_size must be divisible by vocab_chunk.")

  outdir = args.outdir.expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  xla_dir = args.xla_dir.expanduser().resolve() if args.xla_dir else None
  if xla_dir:
    xla_dir.mkdir(parents=True, exist_ok=True)

  tokens = args.global_batch_size * args.sequence_length
  token_loop_count = math.ceil(tokens / args.token_chunk)
  vocab_loop_count = math.ceil(args.vocab_size / args.vocab_chunk)
  loop_count = token_loop_count * vocab_loop_count
  mesh = make_mesh(args.mesh_fsdp, args.mesh_tp)

  key = jax.random.PRNGKey(args.seed)
  if args.workload == "collective_loop":
    tile = jax.random.normal(
        key,
        (args.token_chunk, args.vocab_chunk),
        dtype=jnp.bfloat16,
    )
    step = make_collective_step(
        mesh=mesh,
        token_chunk=args.token_chunk,
        vocab_chunk=args.vocab_chunk,
        loop_count=loop_count,
    )
    call = lambda: step(tile).block_until_ready()
  else:
    hidden = jax.random.normal(
        key,
        (tokens, args.hidden_size),
        dtype=jnp.bfloat16,
    )
    head = jax.random.normal(
        jax.random.fold_in(key, 1),
        (args.hidden_size, args.vocab_size),
        dtype=jnp.bfloat16,
    )
    step = make_projection_step(
        mesh=mesh,
        workload=args.workload,
        token_chunk=args.token_chunk,
        vocab_chunk=args.vocab_chunk,
        token_loop_count=token_loop_count,
        vocab_loop_count=vocab_loop_count,
        hidden_size=args.hidden_size,
    )
    call = lambda: step(hidden, head).block_until_ready()

  profiler_started = False
  if args.profiler_dir:
    profiler_dir = args.profiler_dir.expanduser().resolve()
    profiler_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(profiler_dir))
    profiler_started = True

  history: list[dict[str, Any]] = []
  try:
    for step_index in range(args.warmup_steps + args.measured_steps):
      started = time.perf_counter()
      value = call()
      elapsed = time.perf_counter() - started
      history.append({
          "step": step_index,
          "phase": "warmup" if step_index < args.warmup_steps else "measured",
          "step_time_sec": elapsed,
          "result": float(np.asarray(value)),
      })
  finally:
    if profiler_started:
      jax.profiler.stop_trace()

  write_csv(outdir / "history.csv", history)
  xla_hbm, xla_report = parse_xla_total_gib(xla_dir)
  runtime_peak, runtime_limit = runtime_memory_gb()
  stats = step_stats(history, args.warmup_steps, args.measured_steps)
  measured_time = stats.get("measured_time_sec") or 0.0
  tokens_per_sec = (
      tokens * args.measured_steps / measured_time if measured_time > 0 else None
  )
  row = {
      "timestamp": utc_now(),
      "hostname": socket.gethostname(),
      "backend": args.backend,
      "hardware_target": args.hardware_target,
      "accelerator_type": args.accelerator_type,
      "workload_family": args.workload,
      "status": "success",
      "jax_version": jax.__version__,
      "jax_backend": jax.default_backend(),
      "jax_device_count": len(jax.devices()),
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
      "cce_loop_count": loop_count if args.workload == "projection_collective_loop" else "",
      "warmup_steps": args.warmup_steps,
      "measured_steps": args.measured_steps,
      "tokens_per_sec": tokens_per_sec,
      "xla_planned_hbm_gib_per_chip": xla_hbm,
      "xla_report_path": xla_report,
      "runtime_hbm_peak_gb": runtime_peak,
      "runtime_hbm_limit_gb": runtime_limit,
      "profiler_path": str(args.profiler_dir.expanduser().resolve())
      if args.profiler_dir
      else "",
      **stats,
  }
  (outdir / "summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
  return row


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
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
  parser.add_argument("--outdir", type=Path, required=True)
  parser.add_argument("--xla-dir", type=Path, default=None)
  parser.add_argument("--profiler-dir", type=Path, default=None)
  return parser.parse_args()


def main() -> None:
  row = run(parse_args())
  print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
  main()
