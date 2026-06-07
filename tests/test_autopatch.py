#!/usr/bin/env python3
"""Subprocess smoke checks for import-time CCE autopatching."""

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
  run_env.pop("TUNIX_ACCEL_DISABLE_CE", None)
  run_env.pop("TUNIX_ACCEL_CE_PRESET", None)
  run_env.pop("TUNIX_ACCEL_CE_TOKEN_CHUNK", None)
  run_env.pop("TUNIX_ACCEL_CE_VOCAB_CHUNK", None)
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


def test_cce_autopatch_reads_tpu_large_chunks_preset() -> None:
  output = _run_python(
      """
      from tunix.sft import peft_trainer  # noqa: F401
      from tunix_accel import tunix_patch

      assert tunix_patch.is_installed()
      assert tunix_patch._STATE.token_chunk == 512
      assert tunix_patch._STATE.vocab_chunk == 65536
      print("cce_tpu_large_chunks_preset=ok")
      """,
      env={"TUNIX_ACCEL_CE_PRESET": "tpu_large_chunks"},
  )
  assert output.endswith("cce_tpu_large_chunks_preset=ok")


def test_cce_explicit_chunks_override_preset() -> None:
  output = _run_python(
      """
      from tunix.sft import peft_trainer  # noqa: F401
      from tunix_accel import tunix_patch

      assert tunix_patch.is_installed()
      assert tunix_patch._STATE.token_chunk == 256
      assert tunix_patch._STATE.vocab_chunk == 32768
      print("cce_explicit_chunks_override_preset=ok")
      """,
      env={
          "TUNIX_ACCEL_CE_PRESET": "tpu_large_chunks",
          "TUNIX_ACCEL_CE_TOKEN_CHUNK": "256",
          "TUNIX_ACCEL_CE_VOCAB_CHUNK": "32768",
      },
  )
  assert output.endswith("cce_explicit_chunks_override_preset=ok")


def test_cce_autopatch_can_be_disabled() -> None:
  output = _run_python(
      """
      from tunix.sft import peft_trainer  # noqa: F401
      from tunix_accel import tunix_patch

      assert not tunix_patch.is_installed()
      print("cce_autopatch_disabled=ok")
      """,
      env={"TUNIX_ACCEL_DISABLE_CE": "true"},
  )
  assert output.endswith("cce_autopatch_disabled=ok")
