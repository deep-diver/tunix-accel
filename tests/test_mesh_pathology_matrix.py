from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "01-CCE" / "run_mesh_pathology_matrix.py"


def load_runner():
  spec = importlib.util.spec_from_file_location("run_mesh_pathology_matrix", RUNNER_PATH)
  assert spec is not None
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def args(**overrides):
  defaults = {
      "preset": "pilot",
      "hardware_targets": None,
      "workloads": None,
      "repeats": 1,
      "shapes": None,
      "mesh_configs": None,
      "token_chunks": None,
      "vocab_chunks": None,
      "hidden_size": 320,
      "vocab_size": 262144,
      "warmup_steps": 3,
      "measured_steps": 10,
      "seed": 0,
      "enable_profiler": False,
      "disable_xla_dump": False,
      "full_hlo_dump": False,
      "keep_all_xla": False,
      "shard_index": 0,
      "num_shards": 1,
      "experiment_id": [],
      "limit": None,
      "case_timeout_sec": 0,
      "run_id": "test-run",
      "outdir": Path("/tmp/mesh-pathology-test"),
      "manifest_path": Path("/tmp/mesh-pathology-test/manifest.jsonl"),
      "results_path": Path("/tmp/mesh-pathology-test/results/shard_0.jsonl"),
      "dry_run": True,
      "write_dstack_commands": False,
      "force": False,
      "cce_model_size": "270m",
      "cce_model_id": "",
      "cce_model_source": None,
      "cce_model_path": None,
      "cce_tokenizer_source": None,
      "cce_tokenizer_path": None,
      "cce_allow_download": False,
  }
  defaults.update(overrides)
  return SimpleNamespace(**defaults)


def test_pilot_matrix_has_16_scenario_groups_and_64_rows():
  runner = load_runner()
  cases = runner.build_matrix(args())
  groups = {(case["hardware_target"], case["workload_family"]) for case in cases}
  assert len(groups) == 16
  assert len(cases) == 64
  assert {case["signature_label"] for case in cases} == {
      "bad",
      "good",
      "control-fsdp4-tp1",
      "control-fsdp1-tp4",
  }


def test_torch_workloads_are_opt_in_gpu_rows():
  runner = load_runner()
  cases = runner.build_matrix(
      args(
          hardware_targets="gpu-a100-80gb-4",
          workloads="torch_eager_chunked_matmul_loop,torch_compile_collective_loop",
      )
  )
  assert len(cases) == 8
  assert {case["workload_runner"] for case in cases} == {"torch_microbench"}
  assert {case["execution_mode"] for case in cases} == {"eager", "compile"}
  assert {case["operation_family"] for case in cases} == {
      "chunked_matmul_loop",
      "collective_loop",
  }


def test_dense_synthetic_default_matrix_decouples_chunk_axes():
  runner = load_runner()
  cases = runner.build_matrix(
      args(
          preset="dense-synthetic",
          hardware_targets="tpu-v5e-4",
      )
  )
  assert len(cases) == 315
  assert {case["workload_family"] for case in cases} == {
      "chunked_matmul_loop",
      "collective_loop",
      "projection_collective_loop",
  }
  assert {case["mesh_configuration"] for case in cases} == {
      "fsdp=4,tp=1",
      "fsdp=2,tp=2",
      "fsdp=1,tp=4",
  }
  assert {case["token_chunk"] for case in cases} == {64, 128, 256, 512, 1024}
  assert {case["vocab_chunk"] for case in cases} == {
      4096,
      8192,
      16384,
      32768,
      65536,
      131072,
      262144,
  }
  assert all(case["chunk_loop_count"] == case["token_loop_count"] * case["vocab_loop_count"] for case in cases)


def test_dense_synthetic_can_select_torch_eager_gpu_stack():
  runner = load_runner()
  cases = runner.build_matrix(
      args(
          preset="dense-synthetic",
          hardware_targets="gpu-a100-80gb-4",
          workloads="torch_eager_chunked_matmul_loop",
          token_chunks="128,512",
          vocab_chunks="8192,65536",
          mesh_configs="2x2",
          shapes="b16/L512",
      )
  )
  assert len(cases) == 4
  assert {case["workload_runner"] for case in cases} == {"torch_microbench"}
  assert {case["execution_mode"] for case in cases} == {"eager"}
  assert {case["chunk_configuration"] for case in cases} == {
      "128/8192",
      "128/65536",
      "512/8192",
      "512/65536",
  }


def test_dense_mesh_config_accepts_fsdp_tp_form():
  runner = load_runner()
  cases = runner.build_matrix(
      args(
          preset="dense-synthetic",
          hardware_targets="tpu-v5e-4",
          workloads="chunked_matmul_loop",
          token_chunks="128",
          vocab_chunks="8192",
          mesh_configs="fsdp=2,tp=2",
      )
  )
  assert len(cases) == 1
  assert cases[0]["mesh_configuration"] == "fsdp=2,tp=2"


def test_primary_gpu_target_is_a100_80gb_four_way():
  runner = load_runner()
  a100 = runner.HARDWARE_TARGETS["gpu-a100-80gb-4"]
  assert a100["dstack_gpu_spec"] == "A100:4:80GB"
  assert a100["device_count"] == 4
  assert a100["role"] == "primary_gpu_baseline"


def test_dstack_commands_are_generated_for_gpu_targets_only():
  runner = load_runner()
  cases = runner.build_matrix(args())
  rows = runner.dstack_command_rows(cases, args())
  assert {row["hardware_target"] for row in rows} == {
      "gpu-a100-80gb-4",
      "gpu-l40s-48gb-4",
      "gpu-h100-80gb-4",
  }
  assert all("dstack apply" in row["apply_command"] for row in rows)
