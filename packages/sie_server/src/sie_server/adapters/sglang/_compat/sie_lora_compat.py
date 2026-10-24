"""Load partial QKV LoRAs and release their GPU slots on pinned SGLang engines.

Upstream issue https://github.com/sgl-project/sglang/issues/21380 describes the
missing cleanup: CPU weights disappear, but the GPU pool keeps the old UID.
Repeated load/unload cycles then exhaust the pool. The CUDA 13 bundle's
SGLang 0.5.20 already implements this cleanup and keeps its native unload path.
Both pinned engines assume split attention adapters include q_proj and v_proj.
Complete missing projections with zero factors before their normal packing.

Only the SGLang child opts into this hook. It waits for the engine's normal
import rather than importing CUDA or SGLang during Python site initialization.
The engine already drains requests before calling the method wrapped here.
"""

from __future__ import annotations

import builtins
import importlib.metadata
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import Any

_AFFECTED_VERSIONS = frozenset({"0.5.10", "0.5.10.post1"})
_PARTIAL_QKV_VERSIONS = _AFFECTED_VERSIONS | {"0.5.20"}
_ADAPTER_MODULE = "sglang.srt.lora.lora"
_MANAGER_MODULE = "sglang.srt.lora.lora_manager"
_POOL_MODULE = "sglang.srt.lora.mem_pool"
_CLASS_MARKER = "_sie_lora_slot_release_compat"
_ADAPTER_MARKER = "_sie_lora_partial_qkv_compat"
_IMPORT_MARKER = "_sie_lora_slot_release_deferred"
_SPLIT_QKV_WEIGHT = re.compile(r"((?:.*\.)?)([qkv])_proj\.(lora_[AB]\.weight)")
logger = logging.getLogger(__name__)


def _positive_attention_dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Partial QKV LoRA requires standard positive attention dimensions")
    return value


def _complete_partial_qkv(adapter: Any, weight_names: list[str], weights: dict[str, Any]) -> list[str]:
    """Complete standard attention factors on CPU, before the first GPU use.

    Keep every supplied factor unchanged. A missing projection gets zero A and
    B factors, with B sized from attention heads rather than hidden_size (GQA
    and explicit head_dim can make those sizes different). Unknown layouts fail
    during adapter loading instead of reaching the scheduler with invalid keys.
    """
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for name in weight_names:
        match = _SPLIT_QKV_WEIGHT.fullmatch(name)
        if match is not None:
            prefix, projection, factor = match.groups()
            groups.setdefault(prefix, {}).setdefault(projection, {})[factor] = weights[name]

    additions: dict[str, Any] = {}
    for prefix, projections in groups.items():
        if any(f"{prefix}qkv_proj.{factor}" in weights for factor in ("lora_A.weight", "lora_B.weight")):
            raise ValueError("LoRA cannot mix fused and split QKV factors for one attention projection")
        if any(set(factors) != {"lora_A.weight", "lora_B.weight"} for factors in projections.values()):
            raise ValueError("Split QKV LoRA projections require both lora_A.weight and lora_B.weight")
        if len(projections) == 3:
            continue

        config = adapter.base_hf_config
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        hidden_size = _positive_attention_dimension(getattr(config, "hidden_size", None))
        query_heads = _positive_attention_dimension(getattr(config, "num_attention_heads", None))
        kv_heads = _positive_attention_dimension(getattr(config, "num_key_value_heads", query_heads))
        rank = _positive_attention_dimension(adapter.config.r)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            if hidden_size % query_heads:
                raise ValueError("Partial QKV LoRA requires an explicit head_dim for this attention layout")
            head_dim = hidden_size // query_heads
        head_dim = _positive_attention_dimension(head_dim)
        output_dims = {"q": query_heads * head_dim, "k": kv_heads * head_dim, "v": kv_heads * head_dim}

        for projection, factors in projections.items():
            a, b = factors["lora_A.weight"], factors["lora_B.weight"]
            if tuple(a.shape) != (rank, hidden_size) or tuple(b.shape) != (output_dims[projection], rank):
                raise ValueError("Partial QKV LoRA tensor shapes do not match the base attention dimensions")
        reference = next(iter(projections.values()))
        for projection in ("q", "k", "v"):
            if projection not in projections:
                additions[f"{prefix}{projection}_proj.lora_A.weight"] = reference["lora_A.weight"].new_zeros(
                    (rank, hidden_size)
                )
                additions[f"{prefix}{projection}_proj.lora_B.weight"] = reference["lora_B.weight"].new_zeros(
                    (output_dims[projection], rank)
                )

    # Validate every group before changing the input mapping.
    weights.update(additions)
    return [*weight_names, *additions]


def _patch_adapter_module(module: ModuleType) -> None:
    adapter = getattr(module, "LoRAAdapter", None)
    if adapter is None:
        raise RuntimeError("SGLang does not expose the expected LoRAAdapter")
    if getattr(adapter, _ADAPTER_MARKER, False):
        return
    original_normalize = adapter.normalize_qkv_proj

    @wraps(original_normalize)
    def normalize_with_partial_qkv(self: Any, weight_names: list[str], weights: dict[str, Any]) -> Any:
        names = _complete_partial_qkv(self, weight_names, weights)
        return original_normalize(self, names, weights)

    adapter.normalize_qkv_proj = normalize_with_partial_qkv
    setattr(adapter, _ADAPTER_MARKER, True)


def _release_slot(pool: Any, lora_id: str, empty_slot: Any) -> None:
    """Forget one drained adapter, preserving the base model's ``None`` UID."""
    if lora_id is None:
        raise ValueError("a LoRA unload must never release the base model slot")
    slot = pool.uid_to_buffer_id.get(lora_id)
    if slot is not None and pool.buffer_id_to_uid[slot] != lora_id:
        raise RuntimeError("LoRA GPU slot mapping is inconsistent")
    # The policy can remember an adapter even when it is not currently in a
    # GPU slot. remove() is idempotent in both pinned upstream policies.
    pool.eviction_policy.remove(lora_id)
    if slot is not None:
        pool.uid_to_buffer_id.pop(lora_id)
        pool.buffer_id_to_uid[slot] = empty_slot


def _patch_manager_module(module: ModuleType) -> None:
    manager = getattr(module, "LoRAManager", None)
    if manager is None:
        raise RuntimeError("SGLang does not expose the expected LoRAManager")
    if getattr(manager, _CLASS_MARKER, False):
        return
    pool_module = sys.modules.get(_POOL_MODULE)
    if pool_module is None or not hasattr(pool_module, "EMPTY_SLOT"):
        raise RuntimeError("SGLang does not expose the expected LoRA pool sentinel")
    empty_slot = pool_module.EMPTY_SLOT
    original_unload = manager.unload_lora_adapter

    @wraps(original_unload)
    def unload_with_slot_release(self: Any, lora_ref: Any) -> Any:
        # Do not alter a pool for an upstream operation that did not succeed.
        result = original_unload(self, lora_ref)
        if result.success:
            try:
                _release_slot(self.memory_pool, lora_ref.lora_id, empty_slot)
            except Exception:
                logger.exception("SIE could not release an unloaded LoRA GPU slot")
                return self.create_lora_update_result(
                    success=False,
                    error_message="LoRA GPU slot cleanup failed; reload this model before loading more adapters",
                )
        return result

    manager.unload_lora_adapter = unload_with_slot_release
    setattr(manager, _CLASS_MARKER, True)


def _patch_ready_modules(*, release_slots: bool) -> bool:
    adapter = sys.modules.get(_ADAPTER_MODULE)
    if not isinstance(adapter, ModuleType) or not hasattr(adapter, "LoRAAdapter"):
        return False
    _patch_adapter_module(adapter)
    if release_slots:
        manager = sys.modules.get(_MANAGER_MODULE)
        if not isinstance(manager, ModuleType) or not hasattr(manager, "LoRAManager"):
            return False
        _patch_manager_module(manager)
    return True


def install_lora_compat() -> bool:
    """Install only reviewed version backports; leave unknown engines native."""
    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        return False
    if version not in _PARTIAL_QKV_VERSIONS:
        return False
    release_slots = version in _AFFECTED_VERSIONS
    if _patch_ready_modules(release_slots=release_slots):
        return True

    current_import = builtins.__import__
    if getattr(current_import, _IMPORT_MARKER, False):
        return True

    def deferred_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        module = current_import(name, globals, locals, fromlist, level)
        complete = False
        try:
            complete = _patch_ready_modules(release_slots=release_slots)
        except Exception:
            if builtins.__import__ is deferred_import:
                setattr(builtins, "__import__", current_import)  # noqa: B010
            raise
        finally:
            if complete and builtins.__import__ is deferred_import:
                setattr(builtins, "__import__", current_import)  # noqa: B010
        return module

    setattr(deferred_import, _IMPORT_MARKER, True)
    setattr(builtins, "__import__", deferred_import)  # noqa: B010
    return True
