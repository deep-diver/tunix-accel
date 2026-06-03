#!/usr/bin/env python3
"""Subprocess smoke checks for import-time autopatching."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str, *, env: dict[str, str] | None = None) -> str:
  run_env = os.environ.copy()
  run_env.pop("TUNIX_ACCEL_DISABLE_AUTOPATCH", None)
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
