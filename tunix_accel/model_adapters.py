"""Model adapters for chunked LM-head CE.

The loss kernel is model-agnostic once it receives final hidden states and an
LM-head kernel shaped [hidden_dim, vocab]. This module contains best-effort
adapters for Tunix decoder-only models that expose the common
embedder/layers/final_norm/lm_head structure.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib

from flax import nnx
import jax
import jax.numpy as jnp


@dataclass
class LMHeadParts:
  hidden: jax.Array
  head_kernel: jax.Array
  logit_softcap: float | None = None


def is_supported_decoder_lm(model: nnx.Module) -> bool:
  return all(hasattr(model, attr) for attr in ("embedder", "layers", "final_norm"))


def _module_name(model: nnx.Module) -> str:
  return model.__class__.__module__


def _encode(model: nnx.Module, input_tokens: jax.Array, images=None) -> jax.Array:
  if hasattr(model, "_encode_and_get_inputs"):
    kwargs = {} if images is None else {"images": images}
    return model._encode_and_get_inputs(tokens=input_tokens, **kwargs)  # pylint: disable=protected-access
  if images is not None:
    raise TypeError(
        f"{type(model).__name__} adapter does not support image inputs."
    )
  return model.embedder.encode(input_tokens)


def _has_lora_params(model: nnx.Module) -> bool:
  try:
    return bool(jax.tree_util.tree_leaves(nnx.state(model, nnx.LoRAParam)))
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def _identity_decode(x):
  return x


def prepare_intercepted_lora_model(model: nnx.Module) -> bool:
  """Makes Qwix-LoRA decoder models return hidden states from `__call__`.

  Qwix LoRA for NNX works by intercepting the model's top-level `__call__`.
  Directly replaying sublayers bypasses that interception and silently drops
  LoRA deltas. For LoRA models, CCE therefore runs the real intercepted model
  call, but the final tied-embedding decode must be an identity so the forward
  returns final hidden states instead of full vocab logits.

  This preparation must happen before NNX/JAX tracing. Mutating the graph from
  inside the loss function is rejected by NNX cached_partial/JIT.
  """
  if not _has_lora_params(model):
    return False
  if not hasattr(model, "embedder") or not hasattr(model.embedder, "decode"):
    raise TypeError(
        f"{type(model).__name__} has LoRA params, but this adapter cannot run "
        "an intercepted hidden-state forward without embedder.decode."
    )
  if getattr(model.embedder, "_tunix_accel_decode_identity", False):
    return True
  model.embedder._tunix_accel_original_decode = model.embedder.decode  # pylint: disable=protected-access
  model.embedder.decode = _identity_decode
  model.embedder._tunix_accel_decode_identity = True  # pylint: disable=protected-access
  return True


def _intercepted_hidden_via_model_call(
    model: nnx.Module,
    input_tokens: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
    images: jax.Array | None = None,
) -> jax.Array:
  if not hasattr(model, "embedder") or not hasattr(model.embedder, "decode"):
    raise TypeError(
        f"{type(model).__name__} has LoRA params, but this adapter cannot run "
        "an intercepted hidden-state forward without embedder.decode."
    )
  if not getattr(model.embedder, "_tunix_accel_decode_identity", False):
    raise RuntimeError(
        "Qwix-LoRA chunked CE models must be prepared before JIT tracing. "
        "Call tunix_accel.model_adapters.prepare_intercepted_lora_model(model) "
        "before constructing/tracing the trainer, or use tunix_patch.install() "
        "before creating PeftTrainer."
    )
  kwargs = {} if images is None else {"images": images}
  hidden, _ = model(input_tokens, positions, None, attention_mask, **kwargs)
  return hidden


def _run_layers(
    model: nnx.Module,
    x: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
  module_name = _module_name(model)
  if module_name.endswith(".qwen2.model"):
    model_module = importlib.import_module(module_name)
    sin, cos = model_module._generate_pos_embeddings(  # pylint: disable=protected-access
        positions,
        model.config.head_dim,
        model.config.rope_theta,
    )
    sin, cos = sin.astype(x.dtype), cos.astype(x.dtype)
    for i, layer in enumerate(model.layers):
      with jax.named_scope(f"layer_{i}"):
        _, x = layer(
            x,
            None,
            attention_mask,
            sin,
            cos,
        )
    return x

  for i, layer in enumerate(model.layers):
    with jax.named_scope(f"layer_{i}"):
      _, x = layer(
          x,
          positions,
          None,
          attention_mask,
      )
  return x


def _head_kernel(model: nnx.Module) -> jax.Array:
  config = getattr(model, "config", None)
  tied = (
      getattr(config, "weight_tying", False)
      or getattr(config, "use_tied_embedding", False)
      or not hasattr(model, "lm_head")
  )
  if tied:
    return model.embedder.input_embedding[...].T
  return model.lm_head.w[...]


def extract_lm_head_parts(
    model: nnx.Module,
    input_tokens: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
    images: jax.Array | None = None,
) -> LMHeadParts:
  """Extracts final hidden states and LM-head kernel for supported Tunix LMs."""
  if not is_supported_decoder_lm(model):
    raise TypeError(f"Unsupported model for chunked LM-head CE: {type(model)}")

  if _has_lora_params(model):
    hidden = _intercepted_hidden_via_model_call(
        model,
        input_tokens,
        positions,
        attention_mask,
        images=images,
    )
  else:
    x = _encode(model, input_tokens, images=images)
    x = _run_layers(model, x, positions, attention_mask)
    hidden = model.final_norm(x)
  softcap = getattr(model, "final_logits_softcap", None)
  return LMHeadParts(
      hidden=hidden,
      head_kernel=_head_kernel(model),
      logit_softcap=softcap,
  )
