"""Startup hook for automatic Tunix acceleration patching.

This module is imported by the package's `sitecustomize` startup hook when the
package is installed into a Python environment. It stays intentionally light: it
registers an import hook and only imports JAX/Tunix-heavy modules when
supported Tunix modules are actually loaded by user code.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType


CCE_TARGET_MODULE = "tunix.sft.peft_trainer"
GEMMA3_TARGET_MODULE = "tunix.models.gemma3.model"
ENV_DISABLE = "TUNIX_ACCEL_DISABLE_AUTOPATCH"
ENV_DISABLE_CE = "TUNIX_ACCEL_DISABLE_CE"
ENV_TOKEN_CHUNK = "TUNIX_ACCEL_CE_TOKEN_CHUNK"
ENV_VOCAB_CHUNK = "TUNIX_ACCEL_CE_VOCAB_CHUNK"
ENV_DISABLE_TILED_MLP = "TUNIX_ACCEL_DISABLE_TILED_MLP"
ENV_TILED_MLP_TOKEN_CHUNK = "TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK"
ENV_TILED_MLP_FALLBACK_ON_LORA = "TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA"
DEFAULT_TOKEN_CHUNK = 128
DEFAULT_VOCAB_CHUNK = 8192
DEFAULT_TILED_MLP_TOKEN_CHUNK = 128


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


def _tiled_mlp_token_chunk_from_env() -> int:
  raw = os.environ.get(ENV_TILED_MLP_TOKEN_CHUNK)
  if not raw:
    return DEFAULT_TILED_MLP_TOKEN_CHUNK
  try:
    token_chunk = int(raw)
  except ValueError:
    return DEFAULT_TILED_MLP_TOKEN_CHUNK
  return token_chunk if token_chunk > 0 else DEFAULT_TILED_MLP_TOKEN_CHUNK


def _patch_cce(module: ModuleType | None = None) -> None:
  if not _env_enabled() or _env_bool(ENV_DISABLE_CE, default=False):
    return
  target = module or sys.modules.get(CCE_TARGET_MODULE)
  if target is None or getattr(target, "_tunix_accel_autopatched", False):
    return

  from tunix_accel import tunix_patch  # pylint: disable=import-outside-toplevel

  tunix_patch.install(
      token_chunk=_token_chunk_from_env(),
      vocab_chunk=_vocab_chunk_from_env(),
  )
  setattr(target, "_tunix_accel_autopatched", True)


def _patch_gemma3_tiled_mlp(module: ModuleType | None = None) -> None:
  if not _env_enabled() or _env_bool(ENV_DISABLE_TILED_MLP, default=False):
    return
  target = module or sys.modules.get(GEMMA3_TARGET_MODULE)
  if target is None or getattr(target, "_tunix_accel_tiled_mlp_autopatched", False):
    return

  from tunix_accel import gemma3_tiled_mlp  # pylint: disable=import-outside-toplevel

  gemma3_tiled_mlp.install(
      token_chunk=_tiled_mlp_token_chunk_from_env(),
      fallback_to_original_on_lora=_env_bool(
          ENV_TILED_MLP_FALLBACK_ON_LORA,
          default=True,
      ),
  )
  setattr(target, "_tunix_accel_tiled_mlp_autopatched", True)


_PATCHERS = {
    CCE_TARGET_MODULE: _patch_cce,
    GEMMA3_TARGET_MODULE: _patch_gemma3_tiled_mlp,
}


class _PatchLoader(importlib.abc.Loader):
  def __init__(self, wrapped: importlib.abc.Loader, fullname: str):
    self._wrapped = wrapped
    self._fullname = fullname

  def create_module(self, spec):
    create_module = getattr(self._wrapped, "create_module", None)
    if create_module is None:
      return None
    return create_module(spec)

  def exec_module(self, module: ModuleType) -> None:
    self._wrapped.exec_module(module)
    _PATCHERS[self._fullname](module)


class _PatchFinder(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path, target=None):
    if fullname not in _PATCHERS:
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
          spec.loader = _PatchLoader(spec.loader, fullname)
        return spec
    return None


def enable() -> None:
  """Registers the lazy Tunix import hook, or patches already-loaded modules."""
  if not _env_enabled():
    return
  for target_module, patcher in _PATCHERS.items():
    if target_module in sys.modules:
      patcher(sys.modules[target_module])
  if not any(isinstance(finder, _PatchFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _PatchFinder())


enable()
