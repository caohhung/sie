"""Resolve native checkpoint paths using the configured local cache policy."""

from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import constants, snapshot_download
from sie_sdk.cache import get_cache_config


def resolve_checkpoint(model_name_or_path: str | Path, *, revision: str | None, allow_patterns: Sequence[str]) -> Path:
    """Keep local paths local and resolve Hub snapshots under the serving cache policy."""
    directory = Path(model_name_or_path)
    if directory.is_dir():
        return directory
    config = get_cache_config()
    return Path(
        snapshot_download(
            repo_id=str(model_name_or_path),
            revision=revision,
            allow_patterns=list(allow_patterns),
            cache_dir=config.local_cache,
            local_files_only=constants.HF_HUB_OFFLINE or not config.hf_fallback,
        )
    )
