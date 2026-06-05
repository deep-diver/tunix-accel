#!/usr/bin/env python3
"""Subprocess smoke checks for import-time autopatching."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str, *, env: dict[str, str] | None = None) -> str:
  run_env = os.environ.copy()
  run_env.pop("TUNIX_ACCEL_DISABLE_AUTOPATCH", None)
  run_env.pop("TUNIX_ACCEL_DISABLE_CE", None)
  run_env.pop("TUNIX_ACCEL_ENABLE_LORA_FA", None)
  run_env.pop("TUNIX_ACCEL_LORA_FA_MODE", None)
  run_env.pop("TUNIX_ACCEL_LORA_FA_ALPHA", None)
  run_env.pop("TUNIX_ACCEL_LORA_FA_CORRECTION_EPS", None)
  run_env.pop("TUNIX_ACCEL_LORA_FA_USE_RSLORA", None)
  run_env.pop("TUNIX_ACCEL_DISABLE_TILED_MLP", None)
  run_env.pop("TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY", None)
  run_env.pop("TUNIX_ACCEL_ACTIVATION_POLICY", None)
  run_env["PYTHONPATH"] = str(REPO_ROOT)
  if env:
    run_env.update(env)

  result = subprocess.run(
      [sys.executable, "-c", textwrap.dedent(code)],
      cwd=REPO_ROOT,
      env=run_env,
      check=True,
      capture_output=True,
      text=True,
  )
  return result.stdout.strip()


def test_env_booleans_are_case_insensitive(monkeypatch) -> None:
  sys.path.insert(0, str(REPO_ROOT))

  from tunix_accel import autopatch

  monkeypatch.setenv("TUNIX_ACCEL_DISABLE_AUTOPATCH", "TrUe")
  assert not autopatch._env_enabled()

  monkeypatch.setenv("TUNIX_ACCEL_DISABLE_AUTOPATCH", "FaLsE")
  assert autopatch._env_enabled()

  monkeypatch.setenv("TUNIX_ACCEL_DISABLE_CE", "YeS")
  assert autopatch._env_bool("TUNIX_ACCEL_DISABLE_CE", default=False)

  monkeypatch.setenv("TUNIX_ACCEL_DISABLE_CE", "oFf")
  assert not autopatch._env_bool("TUNIX_ACCEL_DISABLE_CE", default=True)


def test_gemma3_tiled_mlp_autopatch_installs_on_import() -> None:
  output = _run_python(
      """
      from tunix.models.gemma3 import model as gemma3_model
      from tunix_accel import gemma3_tiled_mlp

      assert gemma3_tiled_mlp.is_installed()
      assert getattr(gemma3_model, "_tunix_accel_tiled_mlp_autopatched", False)
      print("gemma3_tiled_mlp_autopatch=ok")
      """
  )
  assert output.endswith("gemma3_tiled_mlp_autopatch=ok")


def test_tunix_packing_api_autopatches_on_peft_trainer_import() -> None:
  output = _run_python(
      """
      import inspect
      from tunix.sft import peft_trainer

      signature = inspect.signature(
          peft_trainer.PeftTrainer.with_gen_model_input_fn
      )
      assert "packing" in signature.parameters
      assert getattr(peft_trainer, "_tunix_accel_packing_api_autopatched", False)
      print("tunix_packing_api_autopatch=ok")
      """,
      env={"TUNIX_ACCEL_DISABLE_CE": "true"},
  )
  assert output.endswith("tunix_packing_api_autopatch=ok")


def test_lora_fa_autopatch_is_opt_in() -> None:
  output = _run_python(
      """
      from tunix.sft import peft_trainer
      from tunix_accel import lora_fa

      assert lora_fa.is_installed()
      assert lora_fa._STATE.config.mode == "freeze_a"
      assert lora_fa._STATE.config.lora_alpha == 8.0
      assert getattr(peft_trainer, "_tunix_accel_lora_fa_autopatched", False)
      print("lora_fa_autopatch=ok")
      """,
      env={
          "TUNIX_ACCEL_DISABLE_CE": "true",
          "TUNIX_ACCEL_ENABLE_LORA_FA": "true",
          "TUNIX_ACCEL_LORA_FA_MODE": "freeze_a",
          "TUNIX_ACCEL_LORA_FA_ALPHA": "8",
      },
  )
  assert output.endswith("lora_fa_autopatch=ok")


def test_benchmark_allow_autopatch_overrides_disabled_env(monkeypatch) -> None:
  module_path = REPO_ROOT / "02-PACKING" / "run_gemma_training_benchmark.py"
  spec = importlib.util.spec_from_file_location("packing_benchmark", module_path)
  assert spec is not None
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)

  from tunix_accel import autopatch

  calls = []
  monkeypatch.setenv("TUNIX_ACCEL_DISABLE_AUTOPATCH", "true")
  monkeypatch.setattr(autopatch, "enable", lambda: calls.append("enabled"))

  module.configure_autopatch(allow_autopatch=True)

  assert os.environ["TUNIX_ACCEL_DISABLE_AUTOPATCH"] == "false"
  assert calls == ["enabled"]


def test_gemma3_tiled_mlp_autopatch_can_be_disabled() -> None:
  output = _run_python(
      """
      from tunix.models.gemma3 import model as gemma3_model
      from tunix_accel import gemma3_tiled_mlp

      assert not gemma3_tiled_mlp.is_installed()
      assert not getattr(gemma3_model, "_tunix_accel_tiled_mlp_autopatched", False)
      print("gemma3_tiled_mlp_autopatch_disabled=ok")
      """,
      env={"TUNIX_ACCEL_DISABLE_TILED_MLP": "1"},
  )
  assert output.endswith("gemma3_tiled_mlp_autopatch_disabled=ok")


def test_disabling_ce_does_not_disable_gemma3_tiled_mlp() -> None:
  output = _run_python(
      """
      from tunix.models.gemma3 import model as gemma3_model
      from tunix_accel import gemma3_tiled_mlp

      assert gemma3_tiled_mlp.is_installed()
      assert getattr(gemma3_model, "_tunix_accel_tiled_mlp_autopatched", False)
      print("ce_disabled_gemma3_tiled_mlp_autopatch=ok")
      """,
      env={"TUNIX_ACCEL_DISABLE_CE": "1"},
  )
  assert output.endswith("ce_disabled_gemma3_tiled_mlp_autopatch=ok")


def test_gemma3_activation_policy_autopatch_reads_policy_env() -> None:
  output = _run_python(
      """
      from tunix.models.gemma3 import model as gemma3_model
      from tunix_accel import gemma3_activation_policy

      assert gemma3_activation_policy.is_installed()
      assert getattr(gemma3_model, "_tunix_accel_activation_policy_autopatched", False)
      assert gemma3_activation_policy._STATE.policy == "split_remat"
      print("gemma3_activation_policy_autopatch=ok")
      """,
      env={
          "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
          "TUNIX_ACCEL_ACTIVATION_POLICY": "split_remat",
      },
  )
  assert output.endswith("gemma3_activation_policy_autopatch=ok")


def test_gemma4_tiled_mlp_autopatch_installs_on_import() -> None:
  output = _run_python(
      """
      from tunix.models.gemma4 import model as gemma4_model
      from tunix_accel import gemma4_tiled_mlp

      assert gemma4_tiled_mlp.is_installed()
      assert getattr(gemma4_model, "_tunix_accel_tiled_mlp_autopatched", False)
      print("gemma4_tiled_mlp_autopatch=ok")
      """
  )
  assert output.endswith("gemma4_tiled_mlp_autopatch=ok")


def test_gemma4_activation_policy_autopatch_reads_policy_env() -> None:
  output = _run_python(
      """
      from tunix.models.gemma4 import model as gemma4_model
      from tunix_accel import gemma4_activation_policy

      assert gemma4_activation_policy.is_installed()
      assert getattr(gemma4_model, "_tunix_accel_activation_policy_autopatched", False)
      assert gemma4_activation_policy._STATE.policy == "split_remat"
      print("gemma4_activation_policy_autopatch=ok")
      """,
      env={
          "TUNIX_ACCEL_DISABLE_TILED_MLP": "1",
          "TUNIX_ACCEL_ACTIVATION_POLICY": "split_remat",
      },
  )
  assert output.endswith("gemma4_activation_policy_autopatch=ok")
