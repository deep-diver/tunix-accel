"""Startup hook for automatic Tunix CCE patching.

This module is imported by the package's `sitecustomize` startup hook when the
package is installed into a Python environment. It stays intentionally light:
it registers an import hook and only imports Tunix/JAX-heavy modules when the
supported Tunix trainer module is actually loaded by user code.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType


CCE_TARGET_MODULE = "tunix.sft.peft_trainer"
ENV_DISABLE = "TUNIX_ACCEL_DISABLE_AUTOPATCH"
ENV_DISABLE_CE = "TUNIX_ACCEL_DISABLE_CE"
ENV_CE_PRESET = "TUNIX_ACCEL_CE_PRESET"
ENV_TOKEN_CHUNK = "TUNIX_ACCEL_CE_TOKEN_CHUNK"
ENV_VOCAB_CHUNK = "TUNIX_ACCEL_CE_VOCAB_CHUNK"
DEFAULT_TOKEN_CHUNK = 128
DEFAULT_VOCAB_CHUNK = 8192
CE_CHUNK_PRESETS = {
    "default": (DEFAULT_TOKEN_CHUNK, DEFAULT_VOCAB_CHUNK),
    "portable": (DEFAULT_TOKEN_CHUNK, DEFAULT_VOCAB_CHUNK),
    "tpu_large_chunks": (512, 65536),
    "tpu-large-chunks": (512, 65536),
}


def _env_enabled() -> bool:
  value = os.environ.get(ENV_DISABLE, "").strip().lower()
  return value not in {"1", "true", "yes", "on"}


def _env_bool(name: str, *, default: bool) -> bool:
  raw = os.environ.get(name)
  if raw is None:
    return default
  value = raw.strip().lower()
  if value in {"1", "true", "yes", "on"}:
    return True
  if value in {"0", "false", "no", "off"}:
    return False
  return default


def _ce_preset_chunks() -> tuple[int, int]:
  preset = os.environ.get(ENV_CE_PRESET, "default").strip().lower()
  return CE_CHUNK_PRESETS.get(preset, CE_CHUNK_PRESETS["default"])


def _token_chunk_from_env() -> int:
  preset_token_chunk, _ = _ce_preset_chunks()
  raw = os.environ.get(ENV_TOKEN_CHUNK)
  if not raw:
    return preset_token_chunk
  try:
    token_chunk = int(raw)
  except ValueError:
    return preset_token_chunk
  return token_chunk if token_chunk > 0 else preset_token_chunk


def _vocab_chunk_from_env() -> int:
  _, preset_vocab_chunk = _ce_preset_chunks()
  raw = os.environ.get(ENV_VOCAB_CHUNK)
  if not raw:
    return preset_vocab_chunk
  try:
    vocab_chunk = int(raw)
  except ValueError:
    return preset_vocab_chunk
  return vocab_chunk if vocab_chunk > 0 else preset_vocab_chunk


def _patch_cce(module: ModuleType | None = None) -> None:
  if not _env_enabled() or _env_bool(ENV_DISABLE_CE, default=False):
    return
  target = module or sys.modules.get(CCE_TARGET_MODULE)
  if target is None or getattr(target, "_tunix_accel_cce_autopatched", False):
    return

  from tunix_accel import tunix_patch  # pylint: disable=import-outside-toplevel

  tunix_patch.install(
      token_chunk=_token_chunk_from_env(),
      vocab_chunk=_vocab_chunk_from_env(),
  )
  setattr(target, "_tunix_accel_cce_autopatched", True)


def enable() -> None:
  """Enables import-time patching and patches already-imported targets."""
  _patch_cce()
  if not any(isinstance(finder, _TunixAccelFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _TunixAccelFinder())


class _TunixAccelLoader(importlib.abc.Loader):
  """Delegating loader that patches supported modules after normal import."""

  def __init__(self, wrapped: importlib.abc.Loader, fullname: str):
    self._wrapped = wrapped
    self._fullname = fullname

  def create_module(self, spec):
    if hasattr(self._wrapped, "create_module"):
      return self._wrapped.create_module(spec)
    return None

  def exec_module(self, module: ModuleType) -> None:
    self._wrapped.exec_module(module)
    if self._fullname == CCE_TARGET_MODULE:
      _patch_cce(module)


class _TunixAccelFinder(importlib.abc.MetaPathFinder):
  """Meta path finder that wraps the Tunix trainer loader."""

  def find_spec(self, fullname: str, path=None, target=None):
    if fullname != CCE_TARGET_MODULE:
      return None
    for finder in sys.meta_path:
      if finder is self:
        continue
      if not hasattr(finder, "find_spec"):
        continue
      spec = finder.find_spec(fullname, path, target)
      if spec is None or spec.loader is None:
        continue
      if isinstance(spec.loader, _TunixAccelLoader):
        return spec
      if not isinstance(spec.loader, importlib.abc.Loader):
        return spec
      spec.loader = _TunixAccelLoader(spec.loader, fullname)
      return spec
    return None


enable()
