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
  run_env.pop("TUNIX_ACCEL_TILED_MLP_BACKEND", None)
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


def test_gemma3_tiled_mlp_autopatch_reads_backend_env() -> None:
  output = _run_python(
      """
      from tunix.models.gemma3 import model as gemma3_model
      from tunix_accel import gemma3_tiled_mlp

      assert gemma3_tiled_mlp.is_installed()
      assert getattr(gemma3_model, "_tunix_accel_tiled_mlp_autopatched", False)
      assert gemma3_tiled_mlp._STATE.matmul_backend == "pallas"
      print("gemma3_tiled_mlp_backend_env=ok")
      """,
      env={"TUNIX_ACCEL_TILED_MLP_BACKEND": "pallas"},
  )
  assert output.endswith("gemma3_tiled_mlp_backend_env=ok")
