"""Startup hook for automatic Tunix loss patching.

This module is imported by the package's `sitecustomize` startup hook when the
package is installed into a Python environment. It stays intentionally light: it
registers an import hook and only imports JAX/Tunix-heavy modules when
`tunix.sft.peft_trainer` is actually loaded by user code.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType


TARGET_MODULE = "tunix.sft.peft_trainer"
ENV_DISABLE = "TUNIX_ACCEL_DISABLE_AUTOPATCH"
ENV_TOKEN_CHUNK = "TUNIX_ACCEL_CE_TOKEN_CHUNK"
ENV_VOCAB_CHUNK = "TUNIX_ACCEL_CE_VOCAB_CHUNK"
DEFAULT_TOKEN_CHUNK = 128
DEFAULT_VOCAB_CHUNK = 8192


def _env_enabled() -> bool:
  value = os.environ.get(ENV_DISABLE, "").strip().lower()
  return value not in {"1", "true", "yes", "on"}


def _token_chunk_from_env() -> int:
  raw = os.environ.get(ENV_TOKEN_CHUNK)
  if not raw:
    return DEFAULT_TOKEN_CHUNK
  try:
    token_chunk = int(raw)
  except ValueError:
    return DEFAULT_TOKEN_CHUNK
  return token_chunk if token_chunk > 0 else DEFAULT_TOKEN_CHUNK


def _vocab_chunk_from_env() -> int:
  raw = os.environ.get(ENV_VOCAB_CHUNK)
  if not raw:
    return DEFAULT_VOCAB_CHUNK
  try:
    vocab_chunk = int(raw)
  except ValueError:
    return DEFAULT_VOCAB_CHUNK
  return vocab_chunk if vocab_chunk > 0 else DEFAULT_VOCAB_CHUNK


def _patch(module: ModuleType | None = None) -> None:
  if not _env_enabled():
    return
  target = module or sys.modules.get(TARGET_MODULE)
  if target is None or getattr(target, "_tunix_accel_autopatched", False):
    return

  from tunix_accel import tunix_patch  # pylint: disable=import-outside-toplevel

  tunix_patch.install(
      token_chunk=_token_chunk_from_env(),
      vocab_chunk=_vocab_chunk_from_env(),
  )
  setattr(target, "_tunix_accel_autopatched", True)


class _PatchLoader(importlib.abc.Loader):
  def __init__(self, wrapped: importlib.abc.Loader):
    self._wrapped = wrapped

  def create_module(self, spec):
    create_module = getattr(self._wrapped, "create_module", None)
    if create_module is None:
      return None
    return create_module(spec)

  def exec_module(self, module: ModuleType) -> None:
    self._wrapped.exec_module(module)
    _patch(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path, target=None):
    if fullname != TARGET_MODULE:
      return None

    for finder in sys.meta_path:
      if finder is self:
        continue
      find_spec = getattr(finder, "find_spec", None)
      if find_spec is None:
        continue
      spec = find_spec(fullname, path, target)
      if spec is not None:
        if spec.loader is not None:
          spec.loader = _PatchLoader(spec.loader)
        return spec
    return None


def enable() -> None:
  """Registers the lazy Tunix import hook, or patches an already-loaded module."""
  if not _env_enabled():
    return
  if TARGET_MODULE in sys.modules:
    _patch(sys.modules[TARGET_MODULE])
    return
  if not any(isinstance(finder, _PatchFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _PatchFinder())


enable()
