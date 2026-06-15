from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "01-CCE" / "run_v5e_cce_matrix.py"


def load_runner():
  spec = importlib.util.spec_from_file_location("run_v5e_cce_matrix", RUNNER_PATH)
  assert spec is not None
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def args(**overrides):
  defaults = {
      "preset": "core",
      "backend": "tpu-v5e",
      "model_size": "270m",
      "model_id": None,
      "model_source": None,
      "model_path": None,
      "tokenizer_source": None,
      "tokenizer_path": None,
      "allow_download": False,
      "vocab_size": 0,
      "meshes": None,
      "shapes": None,
      "chunk_configs": None,
      "repeats": None,
      "no_default_ce": False,
      "include_default_ce": False,
      "warmup_steps": 3,
      "measured_steps": 10,
      "dataset_mode": "synthetic",
      "num_examples": 2048,
      "learning_rate": 2e-4,
      "lora_rank": 16,
      "lora_alpha": 32.0,
      "max_inflight": 1,
      "seed": 0,
      "run_id": "test-run",
      "tpu_type": "v5litepod-4",
      "chips": 4,
      "enable_profiler": False,
      "outdir": Path("/tmp/v5e-cce-test"),
      "profiler_dir": Path("/tmp/v5e-cce-test/profiler"),
      "shard_index": 0,
      "num_shards": 1,
      "experiment_id": [],
      "limit": None,
  }
  defaults.update(overrides)
  return SimpleNamespace(**defaults)


def test_core_matrix_shape_and_loop_counts():
  runner = load_runner()
  cases = runner.build_matrix(args())
  assert len(cases) == 54
  assert cases[0]["loss_impl"] == "default_ce"
  assert cases[0]["cce_loop_count"] is None
  assert cases[1]["loss_impl"] == "cce"
  assert cases[1]["token_loop_count"] == 4
  assert cases[1]["vocab_loop_count"] == 32
  assert cases[1]["cce_loop_count"] == 128


def test_bad_row_reproduction_defaults():
  runner = load_runner()
  cases = runner.build_matrix(args(preset="bad-row"))
  assert len(cases) == 18
  assert {case["mesh_configuration"] for case in cases} == {"fsdp=2,tp=2"}
  assert {case["chunk_configuration"] for case in cases} == {
      "128/8192",
      "256/32768",
      "512/65536",
  }
  assert {case["repeat_index"] for case in cases} == {0, 1, 2}


def test_profiler_preset_marks_only_selected_cases():
  runner = load_runner()
  cases = runner.build_matrix(args(preset="profiler", enable_profiler=True))
  assert len(cases) == 4
  assert all(case["profiler_enabled"] for case in cases)


def test_sharding_uses_experiment_id_modulo():
  runner = load_runner()
  cases = runner.build_matrix(args())
  shards = [
      runner.select_cases(args(num_shards=4, shard_index=index), cases)
      for index in range(4)
  ]
  shard_ids = [{case["experiment_id"] for case in shard} for shard in shards]
  assert sum(len(ids) for ids in shard_ids) == len(cases)
  assert len(set().union(*shard_ids)) == len(cases)
  assert all(case_id % 4 == 2 for case_id in shard_ids[2])


def test_parse_explicit_meshes():
  runner = load_runner()
  assert runner.parse_meshes("fsdp=4,tp=1; fsdp=2,tp=2") == [(4, 1), (2, 2)]
