"""Native decision checkpoint lookup respects the same cache and offline policy."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from sie_server.adapters import _checkpoint
from sie_server.adapters.gliner2.decide import GLiNER2DecideAdapter
from sie_server.adapters.laya.adapter import LayaAdapter
from sie_server.adapters.pplx_decider import adapter as pplx_module

REVISION = "a" * 40
MODEL = "public-fixture/decision"


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "configured-cache"
    monkeypatch.setenv("SIE_LOCAL_CACHE", str(directory))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "different-hf-home"))
    snapshot = directory / "models--public-fixture--decision" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    return snapshot


@pytest.mark.parametrize(("offline", "fallback"), [(True, True), (True, False), (False, False)])
def test_cached_exact_revision_never_opens_a_socket_when_downloads_are_disabled(
    cache: Path, monkeypatch: pytest.MonkeyPatch, offline: bool, fallback: bool
) -> None:
    monkeypatch.setattr(_checkpoint.constants, "HF_HUB_OFFLINE", offline)
    monkeypatch.setenv("SIE_HF_FALLBACK", str(fallback).lower())
    denied = Mock(side_effect=AssertionError("offline checkpoint attempted network access"))
    monkeypatch.setattr("socket.socket.connect", denied)
    assert _checkpoint.resolve_checkpoint(MODEL, revision=REVISION, allow_patterns=["config.json"]) == cache
    denied.assert_not_called()


@pytest.mark.parametrize(("offline", "fallback"), [(True, True), (False, False)])
def test_missing_pinned_revision_does_not_reuse_another_snapshot_or_download(
    cache: Path, monkeypatch: pytest.MonkeyPatch, offline: bool, fallback: bool
) -> None:
    monkeypatch.setattr(_checkpoint.constants, "HF_HUB_OFFLINE", offline)
    monkeypatch.setenv("SIE_HF_FALLBACK", str(fallback).lower())
    denied = Mock(side_effect=AssertionError("missing offline checkpoint attempted network access"))
    monkeypatch.setattr("socket.socket.connect", denied)
    with pytest.raises(LocalEntryNotFoundError):
        _checkpoint.resolve_checkpoint(MODEL, revision="b" * 40, allow_patterns=["config.json"])
    assert cache.is_dir()
    denied.assert_not_called()


def test_online_download_preserves_revision_patterns_and_configured_cache(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checkpoint.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setenv("SIE_HF_FALLBACK", "true")
    download = Mock(return_value=str(cache))
    monkeypatch.setattr(_checkpoint, "snapshot_download", download)
    patterns = ("config.json", "model*.safetensors")
    assert _checkpoint.resolve_checkpoint(MODEL, revision=REVISION, allow_patterns=patterns) == cache
    download.assert_called_once_with(
        repo_id=MODEL,
        revision=REVISION,
        allow_patterns=list(patterns),
        cache_dir=cache.parents[2],
        local_files_only=False,
    )


def test_local_directory_bypasses_hub_and_cache_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    denied = Mock(side_effect=AssertionError("local checkpoint used Hub/cache lookup"))
    monkeypatch.setattr(_checkpoint, "snapshot_download", denied)
    monkeypatch.setattr(_checkpoint, "get_cache_config", denied)
    assert _checkpoint.resolve_checkpoint(tmp_path, revision=REVISION, allow_patterns=["config.json"]) == tmp_path
    denied.assert_not_called()


@pytest.mark.parametrize("adapter_class", [GLiNER2DecideAdapter, LayaAdapter])
def test_decide_and_laya_use_the_configured_cache_without_network(
    cache: Path, monkeypatch: pytest.MonkeyPatch, adapter_class: type[GLiNER2DecideAdapter] | type[LayaAdapter]
) -> None:
    monkeypatch.setattr(_checkpoint.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setenv("SIE_HF_FALLBACK", "false")
    denied = Mock(side_effect=AssertionError("adapter checkpoint attempted network access"))
    monkeypatch.setattr("socket.socket.connect", denied)
    adapter = adapter_class(MODEL, revision=REVISION)
    assert adapter._checkpoint_dir() == cache
    denied.assert_not_called()


def test_pplx_load_uses_the_configured_cache_before_reading_weights(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checkpoint.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setenv("SIE_HF_FALLBACK", "false")
    denied = Mock(side_effect=AssertionError("adapter checkpoint attempted network access"))
    monkeypatch.setattr("socket.socket.connect", denied)
    # Stop at the native config reader: resolution must finish before any model
    # initialization, and this cache fixture intentionally contains no weights.
    read_config = Mock(side_effect=ValueError("checkpoint reader reached"))
    monkeypatch.setattr(pplx_module, "checkpoint_config", read_config)
    adapter = pplx_module.PerplexityDecisionAdapter(MODEL, revision=REVISION)
    with pytest.raises(ValueError, match="checkpoint reader reached"):
        adapter.load("cpu")
    read_config.assert_called_once_with(cache)
    denied.assert_not_called()
