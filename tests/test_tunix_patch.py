#!/usr/bin/env python3
"""Smoke checks for Tunix default-loss monkey patching."""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "1"

import pytest

peft_trainer = pytest.importorskip("tunix.sft.peft_trainer")
from tunix_accel import tunix_patch


def test_tunix_patch_install_uninstall() -> None:
  original = peft_trainer._default_loss_fn  # pylint: disable=protected-access
  assert not tunix_patch.is_installed()

  tunix_patch.install()
  assert tunix_patch.is_installed()
  assert peft_trainer._default_loss_fn is not original  # pylint: disable=protected-access

  tunix_patch.install(token_chunk=64)
  assert tunix_patch.is_installed()
  assert peft_trainer._default_loss_fn is not original  # pylint: disable=protected-access

  tunix_patch.uninstall()
  assert not tunix_patch.is_installed()
  assert peft_trainer._default_loss_fn is original  # pylint: disable=protected-access

  with tunix_patch.patched(token_chunk=32):
    assert tunix_patch.is_installed()
    assert peft_trainer._default_loss_fn is not original  # pylint: disable=protected-access
  assert peft_trainer._default_loss_fn is original  # pylint: disable=protected-access
  print("tunix_patch=ok")


if __name__ == "__main__":
  test_tunix_patch_install_uninstall()
