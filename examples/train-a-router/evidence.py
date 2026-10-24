"""Download the example's public artifacts and verify their published hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import urlopen

FORMAT = "train-a-router-artifacts-v1"
GROUPS = frozenset({"core", "weights", "recordings"})
MANIFEST_LIMIT = 2 * 1024 * 1024


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_manifest(body, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Supply the 64-character SHA-256 published with the artifact manifest")
    if len(body) > MANIFEST_LIMIT or hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("The downloaded manifest does not match the published SHA-256")
    manifest = json.loads(body)
    if manifest.get("format") != FORMAT or not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("Unknown or empty artifact manifest")
    names = set()
    for entry in manifest["files"]:
        name = entry.get("path")
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("Artifact paths must be relative POSIX paths")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name in (".", "manifest.json"):
            raise ValueError("Artifact path can escape or overwrite the manifest")
        if name in names or any(parent.as_posix() in names for parent in path.parents):
            raise ValueError("Duplicate or conflicting artifact path")
        if any(PurePosixPath(other).is_relative_to(path) for other in names):
            raise ValueError("Artifact path conflicts with an earlier directory")
        names.add(name)
        if (
            entry.get("group") not in GROUPS
            or type(entry.get("size")) is not int
            or entry["size"] < 0
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise ValueError("Artifact is missing its group, byte count, or SHA-256")
    return manifest


def artifact_path(root, relative):
    root = Path(root).resolve()
    target = root / relative
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError("Artifact destination escapes the download directory through a symlink")
    return target


def verify(directory, *, manifest_sha256, groups=("core",)):
    directory = Path(directory)
    if not groups or not set(groups).issubset(GROUPS):
        raise ValueError("Choose core, weights, or recordings artifact groups")
    manifest = checked_manifest((directory / "manifest.json").read_bytes(), manifest_sha256)
    for entry in manifest["files"]:
        if entry["group"] not in groups:
            continue
        path = artifact_path(directory, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["size"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Missing or changed artifact: {entry['path']}; repeat fetch to repair the download")
    return manifest


def download(url, target, entry):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=target.name + ".", suffix=".partial", delete=False
        ) as stream:
            temporary = Path(stream.name)
            received, digest = 0, hashlib.sha256()
            with urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > entry["size"]:
                        raise ValueError("Artifact is larger than its published byte count")
                    digest.update(chunk)
                    stream.write(chunk)
        if received != entry["size"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Downloaded artifact failed verification: {entry['path']}")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch(manifest_url, manifest_sha256, output, groups=("core",)):
    """Reuse verified downloads; optional weights are unnecessary for CPU replay."""
    if not groups or not set(groups).issubset(GROUPS):
        raise ValueError("Choose core, weights, or recordings artifact groups")
    url = urlsplit(manifest_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use the public HTTPS manifest URL without credentials, query parameters, or fragments")
    with urlopen(manifest_url, timeout=60) as response:
        body = response.read(MANIFEST_LIMIT + 1)
    manifest = checked_manifest(body, manifest_sha256)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("The manifest destination must not be a symlink")
    if manifest_path.exists():
        if manifest_path.read_bytes() != body:
            raise ValueError("This directory contains another study; choose a new output directory")
    else:
        if any(output.iterdir()):
            raise ValueError("Use an empty directory for the first download")
        manifest_path.write_bytes(body)
    downloaded, reused, total = 0, 0, 0
    for entry in manifest["files"]:
        if entry["group"] not in groups:
            continue
        target = artifact_path(output, entry["path"])
        total += entry["size"]
        if target.is_file() and target.stat().st_size == entry["size"] and sha(target) == entry["sha256"]:
            reused += 1
            continue
        download(urljoin(manifest_url, quote(entry["path"], safe="/")), target, entry)
        downloaded += 1
    return {
        "directory": str(output),
        "manifest_sha256": manifest_sha256,
        "downloaded_files": downloaded,
        "reused_files": reused,
        "selected_bytes": total,
        "groups": list(groups),
    }
