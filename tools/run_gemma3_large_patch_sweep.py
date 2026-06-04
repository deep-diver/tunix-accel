#!/usr/bin/env python3
"""Run Gemma3 12B/27B LoRA patch sweeps on TPU.

The script is intentionally self-contained so it can be copied to TPU VMs and
run unattended. It records success/failure, step time, runtime memory, and the
XLA train-step memory report for each run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAINING_RUNNER = REPO_ROOT / "02-PACKING" / "run_gemma_training_benchmark.py"
TOKENIZER_GCS = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

MODEL_PRESETS = {
    "12b": {
        "model_id": "google/gemma-3-12b-it",
        "model_path": "gs://gemma-data/checkpoints/gemma3-12b-it",
        "mesh_fsdp": 4,
        "mesh_tp": 1,
        "tpu": "v5litepod-4",
    },
    "27b": {
        "model_id": "google/gemma-3-27b-it",
        "model_path": "gs://gemma-data/checkpoints/gemma3-27b-it",
        "mesh_fsdp": 8,
        "mesh_tp": 1,
        "tpu": "v5litepod-8",
    },
}

VARIANT_ENV = {
    "default": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "1",
    },
    "cce": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "tiled_mlp": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "",
        "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA": "0",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "tiled_mlp_c64": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "",
        "TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK": "64",
        "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA": "0",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "tiled_mlp_c256": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "",
        "TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK": "256",
        "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA": "0",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "tiled_cce": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "",
        "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA": "0",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "tiled_splash": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "",
        "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA": "0",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "fast_stack": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "",
        "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA": "0",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "fast_stack_split_remat": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "",
        "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA": "0",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "split_remat",
    },
    "layer_remat": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "layer_remat",
    },
    "split_remat": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "split_remat",
    },
    "split_offload": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "split_offload",
    },
    "splash": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "1",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "1",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "none",
    },
    "stacked": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "0",
        "TUNIX_ACCEL_DISABLE_CE": "",
        "TUNIX_ACCEL_DISABLE_TILED_MLP": "",
        "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA": "0",
        "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY": "",
        "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION": "1",
        "TUNIX_ACCEL_ACTIVATION_POLICY": "split_offload",
    },
    "packed_default": {
        "TUNIX_ACCEL_DISABLE_AUTOPATCH": "1",
    },
}

ENV_KEYS = sorted({
    "TUNIX_ACCEL_DISABLE_AUTOPATCH",
    "TUNIX_ACCEL_DISABLE_CE",
    "TUNIX_ACCEL_CE_TOKEN_CHUNK",
    "TUNIX_ACCEL_CE_VOCAB_CHUNK",
    "TUNIX_ACCEL_DISABLE_TILED_MLP",
    "TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK",
    "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA",
    "TUNIX_ACCEL_TILED_MLP_LORA_ALPHA",
    "TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY",
    "TUNIX_ACCEL_ACTIVATION_POLICY",
    "TUNIX_ACCEL_ACTIVATION_PREVENT_CSE",
    "TUNIX_ACCEL_ACTIVATION_OFFLOAD_SRC",
    "TUNIX_ACCEL_ACTIVATION_OFFLOAD_DST",
    "TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION",
    "TUNIX_ACCEL_SPLASH_ATTENTION_INTERPRET",
})


def parse_csv_ints(value: str) -> list[int]:
  return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
  return [item.strip() for item in value.split(",") if item.strip()]


def set_env(env: dict[str, str], updates: dict[str, str]) -> None:
  for key in ENV_KEYS:
    env.pop(key, None)
  env.update({
      "PYTHONPATH": str(REPO_ROOT),
      "PYTHONUNBUFFERED": "1",
      "TUNIX_ACCEL_CE_TOKEN_CHUNK": "128",
      "TUNIX_ACCEL_CE_VOCAB_CHUNK": "8192",
      "TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK": "128",
      "TUNIX_ACCEL_TILED_MLP_LORA_ALPHA": "32.0",
      "TUNIX_ACCEL_ACTIVATION_PREVENT_CSE": "1",
      "TUNIX_ACCEL_ACTIVATION_OFFLOAD_SRC": "device",
      "TUNIX_ACCEL_ACTIVATION_OFFLOAD_DST": "pinned_host",
      "TUNIX_ACCEL_SPLASH_ATTENTION_INTERPRET": "0",
  })
  for key, value in updates.items():
    if value == "":
      env.pop(key, None)
    else:
      env[key] = value


def first_summary(run_dir: Path) -> dict[str, Any] | None:
  path = run_dir / "summary.json"
  if not path.exists():
    return None
  data = json.loads(path.read_text())
  if isinstance(data, list) and data:
    return dict(data[0])
  if isinstance(data, dict):
    return dict(data)
  return None


def read_history(run_dir: Path) -> list[dict[str, str]]:
  path = run_dir / "history.csv"
  if not path.exists():
    return []
  with path.open() as f:
    return list(csv.DictReader(f))


def latest_train_memory_report(xla_dir: Path) -> Path | None:
  reports = sorted(xla_dir.glob("*jit__train_step*memory-usage-report.txt"))
  return reports[-1] if reports else None


def parse_xla_total_gib(report: Path | None) -> float | None:
  if report is None or not report.exists():
    return None
  text = report.read_text(errors="ignore")
  match = re.search(r"Total bytes:\s+(\d+)\s+\(([\d.]+)GiB\)", text)
  if match:
    return float(match.group(2))
  match = re.search(r"Total bytes:\s+(\d+)", text)
  if match:
    return int(match.group(1)) / (1024**3)
  return None


def parse_failure(log_path: Path) -> dict[str, str]:
  text = log_path.read_text(errors="ignore") if log_path.exists() else ""
  result: dict[str, str] = {}
  if "RESOURCE_EXHAUSTED" in text or "CompileTimeHbmOom" in text:
    result["failure_type"] = "compile_oom"
  elif "Traceback" in text:
    result["failure_type"] = "error"
  else:
    result["failure_type"] = "unknown"
  match = re.search(
      r"Used\s+([\d.]+)G\s+of\s+([\d.]+)G\s+hbm\.\s+Exceeded.*?by\s+([\d.]+)G",
      text,
      flags=re.DOTALL,
  )
  if match:
    result["oom_used_gib"] = match.group(1)
    result["oom_limit_gib"] = match.group(2)
    result["oom_exceeded_gib"] = match.group(3)
  return result


def command_for_run(
    *,
    model: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    variant: str,
    max_length: int,
    batch_size: int,
) -> list[str]:
  input_variant = "packed" if variant == "packed_default" else "unpacked"
  command = [
      sys.executable,
      str(TRAINING_RUNNER),
      "--model-id",
      str(model["model_id"]),
      "--model-source",
      "gcs",
      "--model-path",
      str(model["model_path"]),
      "--tokenizer-source",
      "sentencepiece",
      "--tokenizer-path",
      TOKENIZER_GCS,
      "--num-examples",
      str(args.num_examples),
      "--variants",
      input_variant,
      "--batch-size",
      str(batch_size),
      "--max-length",
      str(max_length),
      "--max-steps",
      str(args.max_steps),
      "--learning-rate",
      str(args.learning_rate),
      "--lora-rank",
      str(args.lora_rank),
      "--lora-alpha",
      str(args.lora_alpha),
      "--mesh-fsdp",
      str(args.mesh_fsdp or model["mesh_fsdp"]),
      "--mesh-tp",
      str(args.mesh_tp or model["mesh_tp"]),
      "--max-inflight",
      str(args.max_inflight),
      "--skip-quality-eval",
      "--outdir",
      str(run_dir),
  ]
  if variant not in {"default", "packed_default"}:
    command.insert(-2, "--allow-autopatch")
  return command


def run_case(
    *,
    model_size: str,
    model: dict[str, Any],
    args: argparse.Namespace,
    variant: str,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
  case_name = f"{variant}_b{batch_size}_l{max_length}"
  run_dir = args.outdir / model_size / case_name
  xla_dir = run_dir / "xla"
  if run_dir.exists() and not args.force:
    print(f"skip_existing {case_name}", flush=True)
  else:
    shutil.rmtree(run_dir, ignore_errors=True)
    xla_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    set_env(env, VARIANT_ENV[variant])
    env["XLA_FLAGS"] = (
        f"--xla_dump_to={xla_dir} --xla_dump_hlo_as_text"
    )
    command = command_for_run(
        model=model,
        args=args,
        run_dir=run_dir,
        variant=variant,
        max_length=max_length,
        batch_size=batch_size,
    )
    log_path = run_dir / "runner.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
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
    elapsed = time.monotonic() - started
    (run_dir / "elapsed_sec.txt").write_text(f"{elapsed:.6f}\n")
    (run_dir / "returncode.txt").write_text(f"{proc.returncode}\n")

  summary = first_summary(run_dir)
  history = read_history(run_dir)
  report = latest_train_memory_report(xla_dir)
  row: dict[str, Any] = {
      "model_size": model_size,
      "model_id": model["model_id"],
      "tpu": model["tpu"],
      "chips": int((args.mesh_fsdp or model["mesh_fsdp"]) * (args.mesh_tp or model["mesh_tp"])),
      "mesh_fsdp": args.mesh_fsdp or model["mesh_fsdp"],
      "mesh_tp": args.mesh_tp or model["mesh_tp"],
      "variant": variant,
      "batch_size": batch_size,
      "max_length": max_length,
      "max_steps": args.max_steps,
      "lora_rank": args.lora_rank,
      "run_dir": str(run_dir),
      "xla_report": str(report) if report else "",
      "xla_train_step_gib_per_chip": parse_xla_total_gib(report),
      "requested_autopatch": variant not in {"default", "packed_default"},
      "status": "ok" if summary else "failed",
  }
  if summary:
    row["default_ce"] = summary.get("default_ce")
    row["autopatch_effective"] = not bool(summary.get("default_ce"))
    accel = summary.get("accel", {})
    if isinstance(accel, dict):
      row.update({
          "cce_installed": accel.get("cce_installed"),
          "gemma3_tiled_mlp_installed": accel.get(
              "gemma3_tiled_mlp_installed"
          ),
          "gemma3_activation_policy_installed": accel.get(
              "gemma3_activation_policy_installed"
          ),
          "gemma3_splash_attention_installed": accel.get(
              "gemma3_splash_attention_installed"
          ),
      })
    row.update({
        "steps_recorded": summary.get("steps_recorded"),
        "final_loss": summary.get("final_loss"),
        "mean_loss": summary.get("mean_loss"),
        "wall_time_sec": summary.get("wall_time_sec"),
        "mean_step_time_sec_excl_first": summary.get(
            "mean_step_time_sec_excl_first"
        ),
        "valid_tokens_per_sec_excl_first": summary.get(
            "valid_tokens_per_sec_excl_first"
        ),
        "loss_tokens_per_sec_excl_first": summary.get(
            "loss_tokens_per_sec_excl_first"
        ),
    })
    memory = summary.get("memory_after_train", {}).get("aggregate", {})
    if memory:
      row.update({
          "runtime_peak_hbm_gb_aggregate": memory.get("peak_bytes_in_use", 0) / 1e9,
          "runtime_hbm_limit_gb_aggregate": memory.get("bytes_limit", 0) / 1e9,
          "runtime_hbm_headroom_gb_aggregate": (
              memory.get("bytes_limit", 0) - memory.get("peak_bytes_in_use", 0)
          ) / 1e9,
      })
    if len(history) >= 1:
      row["first_step_time_sec"] = history[0].get("step_time_sec")
    if len(history) >= 2:
      row["second_step_time_sec"] = history[1].get("step_time_sec")
  else:
    row.update(parse_failure(run_dir / "runner.log"))
  write_json(run_dir / "case_summary.json", row)
  print(
      "case",
      row["status"],
      variant,
      f"b{batch_size}",
      f"l{max_length}",
      "xla_gib",
      row.get("xla_train_step_gib_per_chip"),
      "step_s",
      row.get("mean_step_time_sec_excl_first"),
      flush=True,
  )
  return row


def write_results(outdir: Path, rows: list[dict[str, Any]]) -> None:
  outdir.mkdir(parents=True, exist_ok=True)
  write_json(outdir / "sweep_results.json", rows)
  keys = sorted({key for row in rows for key in row})
  with (outdir / "sweep_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), required=True)
  parser.add_argument("--variants", default="default,cce,tiled_mlp,split_offload,splash,stacked")
  parser.add_argument("--contexts", default="512,1024,2048")
  parser.add_argument("--extra-stacked-contexts", default="4096,8192")
  parser.add_argument("--batch-sizes", default="1")
  parser.add_argument("--num-examples", type=int, default=64)
  parser.add_argument("--max-steps", type=int, default=2)
  parser.add_argument("--learning-rate", type=float, default=2e-4)
  parser.add_argument("--lora-rank", type=int, default=16)
  parser.add_argument("--lora-alpha", type=float, default=32.0)
  parser.add_argument("--mesh-fsdp", type=int, default=0)
  parser.add_argument("--mesh-tp", type=int, default=0)
  parser.add_argument("--max-inflight", type=int, default=1)
  parser.add_argument("--outdir", type=Path, default=Path("/tmp/gemma3-large-patch-sweep"))
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  model = MODEL_PRESETS[args.model_size]
  variants = parse_csv_strings(args.variants)
  contexts = parse_csv_ints(args.contexts)
  extra_stacked_contexts = parse_csv_ints(args.extra_stacked_contexts)
  batch_sizes = parse_csv_ints(args.batch_sizes)
  unknown = sorted(set(variants) - set(VARIANT_ENV))
  if unknown:
    raise ValueError(f"Unknown variants: {unknown}")

  rows: list[dict[str, Any]] = []
  model_outdir = args.outdir / args.model_size
  for batch_size in batch_sizes:
    for max_length in contexts:
      for variant in variants:
        rows.append(
            run_case(
                model_size=args.model_size,
                model=model,
                args=args,
                variant=variant,
                max_length=max_length,
                batch_size=batch_size,
            )
        )
        write_results(model_outdir, rows)

    if "stacked" in variants:
      for max_length in extra_stacked_contexts:
        rows.append(
            run_case(
                model_size=args.model_size,
                model=model,
                args=args,
                variant="stacked",
                max_length=max_length,
                batch_size=batch_size,
            )
        )
        write_results(model_outdir, rows)
        if rows[-1]["status"] != "ok":
          break

  print(f"results={model_outdir / 'sweep_results.csv'}")


if __name__ == "__main__":
  main()
