"""Pinned LoRA loading preserves PEFT factors and reuses drained GPU slots."""

from __future__ import annotations

import builtins
import importlib.util
import itertools
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

_HOOK_PATH = Path(__file__).parents[2] / "src/sie_server/adapters/sglang/_compat/sie_lora_compat.py"


@pytest.fixture
def hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sie_lora_compat_test", _HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """CPU metadata contract of the pinned manager and two-slot GPU pool."""
    empty_slot = object()

    class Manager:
        def __init__(self) -> None:
            self.used: set[str | None] = {None}
            self.memory_pool = SimpleNamespace(
                uid_to_buffer_id={None: 1},
                buffer_id_to_uid=[empty_slot, None],
                eviction_policy=SimpleNamespace(remove=self.used.discard),
            )
            self.loras: dict[str, object] = {}
            self.fail_unload = False

        def add(self, name: str) -> SimpleNamespace:
            slot = self.memory_pool.buffer_id_to_uid.index(empty_slot)
            self.memory_pool.buffer_id_to_uid[slot] = name
            self.memory_pool.uid_to_buffer_id[name] = slot
            self.used.add(name)
            self.loras[name] = object()
            return SimpleNamespace(lora_id=name)

        def create_lora_update_result(self, success: bool, error_message: str = "") -> SimpleNamespace:
            return SimpleNamespace(success=success, error_message=error_message)

        def unload_lora_adapter(self, ref: Any) -> SimpleNamespace:
            if self.fail_unload:
                return self.create_lora_update_result(False, "upstream refused")
            del self.loras[ref.lora_id]
            return self.create_lora_update_result(True)

    manager_module = ModuleType("sglang.srt.lora.lora_manager")
    manager_module.__dict__["LoRAManager"] = Manager
    pool_module = ModuleType("sglang.srt.lora.mem_pool")
    pool_module.__dict__["EMPTY_SLOT"] = empty_slot
    monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
    monkeypatch.setitem(sys.modules, pool_module.__name__, pool_module)
    adapter_module = ModuleType("sglang.srt.lora.lora")

    class Adapter:
        """Pinned engine's packing contract, without importing its CUDA runtime.

        The exact 0.5.10.post1 and 0.5.20 wheel implementations were separately
        reproduced: split weights are packed only when q_proj is present, and
        q_proj also requires v_proj. The hook must supply every absent slice.
        """

        def __init__(self, config: Any, rank: int = 4) -> None:
            self.base_hf_config = config
            self.config = SimpleNamespace(r=rank)
            self.calls = 0

        def normalize_qkv_proj(self, names: list[str], weights: dict[str, torch.Tensor]) -> None:
            self.calls += 1
            for name in names:
                if ".q_proj." in name:
                    parts = [name.replace(".q_proj.", f".{projection}_proj.") for projection in ("q", "k", "v")]
                    weights[name.replace(".q_proj.", ".qkv_proj.")] = torch.cat([weights[part] for part in parts])
                    for part in parts:
                        weights.pop(part)
                elif ".qkv_proj." in name and "lora_A" in name:
                    weights[name] = weights[name].repeat(3, 1)

    adapter_module.__dict__["LoRAAdapter"] = Adapter
    monkeypatch.setitem(sys.modules, adapter_module.__name__, adapter_module)
    return SimpleNamespace(
        manager=Manager, module=manager_module, empty=empty_slot, adapter=Adapter, adapter_module=adapter_module
    )


def test_repeated_unload_reuses_slot_without_evicting_base(hook: ModuleType, engine: SimpleNamespace) -> None:
    broken = engine.manager()
    assert broken.unload_lora_adapter(broken.add("first")).success
    with pytest.raises(ValueError, match="not in list"):
        broken.add("second")

    hook._patch_manager_module(engine.module)
    fixed = engine.manager()
    for index in range(20):
        ref = fixed.add(f"adapter-{index}")
        assert fixed.unload_lora_adapter(ref).success
        assert fixed.loras == {}
        assert fixed.memory_pool.uid_to_buffer_id == {None: 1}
        assert fixed.memory_pool.buffer_id_to_uid == [engine.empty, None]
        assert fixed.used == {None}


def test_upstream_failure_preserves_residency(hook: ModuleType, engine: SimpleNamespace) -> None:
    hook._patch_manager_module(engine.module)
    manager = engine.manager()
    ref = manager.add("busy")
    manager.fail_unload = True
    assert not manager.unload_lora_adapter(ref).success
    assert "busy" in manager.loras
    assert manager.memory_pool.uid_to_buffer_id == {None: 1, "busy": 0}
    assert "busy" in manager.used


def test_nonresident_adapter_cleans_only_policy(hook: ModuleType, engine: SimpleNamespace) -> None:
    hook._patch_manager_module(engine.module)
    manager = engine.manager()
    manager.loras["cpu-only"] = object()
    manager.used.add("cpu-only")
    assert manager.unload_lora_adapter(SimpleNamespace(lora_id="cpu-only")).success
    assert manager.used == {None}
    assert manager.memory_pool.uid_to_buffer_id == {None: 1}


def test_cleanup_failure_cannot_be_reported_as_success(hook: ModuleType, engine: SimpleNamespace) -> None:
    hook._patch_manager_module(engine.module)
    manager = engine.manager()
    ref = manager.add("adapter")
    manager.memory_pool.buffer_id_to_uid[0] = "different-adapter"
    result = manager.unload_lora_adapter(ref)
    assert not result.success
    assert "reload this model" in result.error_message
    assert manager.memory_pool.buffer_id_to_uid[0] == "different-adapter"


def test_base_slot_is_never_released(hook: ModuleType, engine: SimpleNamespace) -> None:
    manager = engine.manager()
    with pytest.raises(ValueError, match="base model"):
        hook._release_slot(manager.memory_pool, None, engine.empty)
    assert manager.memory_pool.uid_to_buffer_id == {None: 1}


@pytest.mark.parametrize("version", ["0.6.0", "0.5.11"])
def test_other_engine_versions_keep_native_method(
    hook: ModuleType, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(hook.importlib.metadata, "version", lambda _: version)
    original = engine.manager.unload_lora_adapter
    original_normalize = engine.adapter.normalize_qkv_proj
    assert hook.install_lora_compat() is False
    assert engine.manager.unload_lora_adapter is original
    assert engine.adapter.normalize_qkv_proj is original_normalize


def test_cuda13_normalization_keeps_native_unload(
    hook: ModuleType, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hook.importlib.metadata, "version", lambda _: "0.5.20")
    original = engine.manager.unload_lora_adapter
    assert hook.install_lora_compat()
    assert engine.manager.unload_lora_adapter is original
    assert getattr(engine.adapter, hook._ADAPTER_MARKER)


def test_exact_version_install_is_idempotent(
    hook: ModuleType, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hook.importlib.metadata, "version", lambda _: "0.5.10.post1")
    assert hook.install_lora_compat()
    first = engine.manager.unload_lora_adapter
    first_normalize = engine.adapter.normalize_qkv_proj
    assert hook.install_lora_compat()
    assert engine.manager.unload_lora_adapter is first
    assert engine.adapter.normalize_qkv_proj is first_normalize


def test_deferred_import_restores_existing_hook(
    hook: ModuleType, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hook.importlib.metadata, "version", lambda _: "0.5.10.post1")
    monkeypatch.delitem(sys.modules, engine.module.__name__)
    previous_import = builtins.__import__
    # Register restoration before installing the hook, including on assertion failure.
    monkeypatch.setattr(builtins, "__import__", previous_import)
    assert hook.install_lora_compat()
    installed = builtins.__import__
    assert installed is not previous_import
    assert hook.install_lora_compat()
    assert builtins.__import__ is installed
    monkeypatch.setitem(sys.modules, engine.module.__name__, engine.module)
    builtins.__import__("math")
    assert builtins.__import__ is previous_import
    assert getattr(engine.manager, hook._CLASS_MARKER)
    assert getattr(engine.adapter, hook._ADAPTER_MARKER)


def test_child_sitecustomize_installs_the_hook(tmp_path: Path) -> None:
    for package in ("sglang", "sglang/srt", "sglang/srt/lora"):
        directory = tmp_path / package
        directory.mkdir(exist_ok=True)
        (directory / "__init__.py").write_text("")
    metadata = tmp_path / "sglang-0.5.10.post1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: sglang\nVersion: 0.5.10.post1\n")
    (tmp_path / "sglang/srt/lora/mem_pool.py").write_text("EMPTY_SLOT = object()\n")
    (tmp_path / "sglang/srt/lora/lora.py").write_text(
        "class LoRAAdapter:\n    def normalize_qkv_proj(self, names, weights):\n        raise NotImplementedError\n"
    )
    (tmp_path / "sglang/srt/lora/lora_manager.py").write_text(
        "from sglang.srt.lora.mem_pool import EMPTY_SLOT\n"
        "from sglang.srt.lora.lora import LoRAAdapter\n"
        "class LoRAManager:\n"
        "    def unload_lora_adapter(self, ref):\n"
        "        raise NotImplementedError\n"
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SIE_SGLANG_")}
    environment["PYTHONPATH"] = os.pathsep.join([str(_HOOK_PATH.parent), str(tmp_path)])
    environment["SIE_SGLANG_LORA_COMPAT"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "assert 'sglang.srt.lora.lora_manager' not in sys.modules\n"
            "from sglang.srt.lora.lora_manager import LoRAManager\n"
            "from sglang.srt.lora.lora import LoRAAdapter\n"
            "assert LoRAManager._sie_lora_slot_release_compat\n"
            "assert LoRAAdapter._sie_lora_partial_qkv_compat\n",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "Error in sitecustomize" not in result.stderr


_SUBSETS = [subset for size in (1, 2, 3) for subset in itertools.combinations(("q", "k", "v"), size)]


def _weights(
    projections: tuple[str, ...], *, query_dim: int, kv_dim: int, prefix: str = "base.model.layers.0.attn."
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261024)
    result = {}
    for projection in projections:
        result[f"{prefix}{projection}_proj.lora_A.weight"] = torch.randn(
            4, 12, generator=generator, dtype=torch.float64
        )
        result[f"{prefix}{projection}_proj.lora_B.weight"] = torch.randn(
            query_dim if projection == "q" else kv_dim, 4, generator=generator, dtype=torch.float64
        )
    return result


@pytest.mark.parametrize("projections", _SUBSETS)
@pytest.mark.parametrize(("kv_heads", "head_dim"), [(3, None), (1, None), (1, 8)])
def test_packed_qkv_matches_independent_peft_products(
    hook: ModuleType, engine: SimpleNamespace, projections: tuple[str, ...], kv_heads: int, head_dim: int | None
) -> None:
    config = SimpleNamespace(hidden_size=12, num_attention_heads=3, num_key_value_heads=kv_heads, head_dim=head_dim)
    query_dim, kv_dim = 3 * (head_dim or 4), kv_heads * (head_dim or 4)
    weights = _weights(projections, query_dim=query_dim, kv_dim=kv_dim)
    original = {name: value.clone() for name, value in weights.items()}
    prefix = "base.model.layers.0.attn."
    x = torch.arange(36, dtype=torch.float64).reshape(3, 12) / 36
    expected = []
    for projection, dim in zip(("q", "k", "v"), (query_dim, kv_dim, kv_dim), strict=True):
        expected.append(
            x
            @ original[f"{prefix}{projection}_proj.lora_A.weight"].T
            @ original[f"{prefix}{projection}_proj.lora_B.weight"].T
            if projection in projections
            else torch.zeros(3, dim, dtype=torch.float64)
        )

    hook._patch_adapter_module(engine.adapter_module)
    adapter = engine.adapter(config)
    adapter.normalize_qkv_proj(list(weights), weights)
    assert adapter.calls == 1
    assert set(weights) == {f"{prefix}qkv_proj.lora_A.weight", f"{prefix}qkv_proj.lora_B.weight"}
    a = weights[f"{prefix}qkv_proj.lora_A.weight"]
    b = weights[f"{prefix}qkv_proj.lora_B.weight"]
    assert a.shape == (12, 12)
    assert b.shape == (query_dim + 2 * kv_dim, 4)
    projected = x @ a.T
    offset = 0
    actual = []
    for index, dim in enumerate((query_dim, kv_dim, kv_dim)):
        actual.append(projected[:, index * 4 : (index + 1) * 4] @ b[offset : offset + dim].T)
        offset += dim
    torch.testing.assert_close(torch.cat(actual, dim=1), torch.cat(expected, dim=1), rtol=1e-12, atol=1e-12)


def test_partial_groups_are_completed_independently(hook: ModuleType, engine: SimpleNamespace) -> None:
    config = SimpleNamespace(hidden_size=12, num_attention_heads=3, num_key_value_heads=1)
    weights = _weights(("q",), query_dim=12, kv_dim=4)
    weights.update(_weights(("v",), query_dim=12, kv_dim=4, prefix="base.model.layers.1.attn."))
    hook._patch_adapter_module(engine.adapter_module)
    engine.adapter(config).normalize_qkv_proj(list(weights), weights)
    assert len(weights) == 4
    for layer, active_slice in ((0, 0), (1, 2)):
        a = weights[f"base.model.layers.{layer}.attn.qkv_proj.lora_A.weight"]
        for index in range(3):
            assert bool(torch.count_nonzero(a[index * 4 : (index + 1) * 4])) is (index == active_slice)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_padding_preserves_factor_dtype_device_and_values(
    hook: ModuleType, engine: SimpleNamespace, dtype: torch.dtype
) -> None:
    config = SimpleNamespace(hidden_size=12, num_attention_heads=3, num_key_value_heads=1, head_dim=8)
    weights = {name: value.to(dtype) for name, value in _weights(("v",), query_dim=24, kv_dim=8).items()}
    original = dict(weights)
    hook._complete_partial_qkv(engine.adapter(config), list(weights), weights)
    assert all(value.dtype == dtype and value.device.type == "cpu" for value in weights.values())
    assert all(weights[name] is value for name, value in original.items())
    assert all(torch.count_nonzero(value) == 0 for name, value in weights.items() if name not in original)


@pytest.mark.parametrize(
    "problem", ["missing_a", "missing_b", "wrong_rank", "wrong_input", "wrong_output", "fused_collision"]
)
def test_invalid_partial_factors_fail_before_mutating_weights(
    hook: ModuleType, engine: SimpleNamespace, problem: str
) -> None:
    config = SimpleNamespace(hidden_size=12, num_attention_heads=3, num_key_value_heads=1)
    weights = _weights(("v",), query_dim=12, kv_dim=4)
    a_name, b_name = sorted(weights)
    if problem == "missing_a":
        weights.pop(a_name)
    elif problem == "missing_b":
        weights.pop(b_name)
    elif problem == "wrong_rank":
        weights[a_name] = torch.zeros(8, 12)
    elif problem == "wrong_input":
        weights[a_name] = torch.zeros(4, 13)
    elif problem == "wrong_output":
        weights[b_name] = torch.zeros(12, 4)
    else:
        weights[a_name.replace("v_proj", "qkv_proj")] = torch.zeros(4, 12)
    before = dict(weights)
    with pytest.raises(ValueError, match=r"QKV LoRA|fused and split QKV"):
        hook._complete_partial_qkv(engine.adapter(config), list(weights), weights)
    assert weights.keys() == before.keys()
    assert all(weights[name] is value for name, value in before.items())


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(),
        SimpleNamespace(hidden_size=13, num_attention_heads=3),
        SimpleNamespace(hidden_size=12, num_attention_heads=3, head_dim=False),
    ],
)
def test_unknown_attention_dimensions_reject_partial_adapters(
    hook: ModuleType, engine: SimpleNamespace, config: SimpleNamespace
) -> None:
    weights = _weights(("v",), query_dim=12, kv_dim=4)
    with pytest.raises(ValueError, match="Partial QKV LoRA"):
        hook._complete_partial_qkv(engine.adapter(config), list(weights), weights)


def test_full_and_fused_qkv_keep_native_packing(hook: ModuleType, engine: SimpleNamespace) -> None:
    hook._patch_adapter_module(engine.adapter_module)
    adapter = engine.adapter(SimpleNamespace())  # Full factors do not need dimension inference.
    weights = _weights(("q", "k", "v"), query_dim=24, kv_dim=8)
    adapter.normalize_qkv_proj(list(weights), weights)
    assert len(weights) == 2
    prefix = "base.model.layers.0.attn.qkv_proj."
    fused_a, fused_b = torch.ones(4, 12), torch.ones(40, 4)
    fused = {prefix + "lora_A.weight": fused_a, prefix + "lora_B.weight": fused_b}
    adapter.normalize_qkv_proj(list(fused), fused)
    assert torch.equal(fused[prefix + "lora_A.weight"], fused_a.repeat(3, 1))
    assert fused[prefix + "lora_B.weight"] is fused_b


def test_no_qkv_weights_do_not_require_attention_config(hook: ModuleType, engine: SimpleNamespace) -> None:
    weights = {"base.model.layers.0.attn.o_proj.lora_A.weight": torch.ones(4, 12)}
    before = dict(weights)
    assert hook._complete_partial_qkv(engine.adapter(SimpleNamespace()), list(weights), weights) == list(weights)
    assert weights == before


def test_unconfigured_child_does_not_import_lora_hook() -> None:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SIE_SGLANG_")}
    environment["PYTHONPATH"] = str(_HOOK_PATH.parent)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; assert 'sie_lora_compat' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
