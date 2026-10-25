"""Small download-boundary tests using invented local artifacts."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO

import evidence


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        content = b"invented data\n"
        manifest = {
            "format": evidence.FORMAT,
            "files": [
                {
                    "path": "source/rows.json",
                    "group": "core",
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                {
                    "path": "reference/weights.bin",
                    "group": "weights",
                    "size": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
            ],
        }
        body = json.dumps(manifest).encode()
        return manifest, body, hashlib.sha256(body).hexdigest(), content

    def test_manifest_and_path_tampering_rejected(self):
        manifest, body, digest, _ = self.fixture()
        with self.assertRaises(ValueError):
            evidence.checked_manifest(body + b" ", digest)
        for name in ("../outside", "/absolute", "manifest.json", "source/../outside", "source//rows.json"):
            changed = deepcopy(manifest)
            changed["files"][0]["path"] = name
            value = json.dumps(changed).encode()
            with self.subTest(name=name), self.assertRaises(ValueError):
                evidence.checked_manifest(value, hashlib.sha256(value).hexdigest())

    def test_fetch_reuses_valid_core_and_repairs_corruption_without_optional_weights(self):
        _, body, digest, content = self.fixture()
        urls = []

        def opening(url, timeout):
            urls.append(url)
            return BytesIO(body if url.endswith("manifest.json") else content)

        with tempfile.TemporaryDirectory() as temporary, patch.object(evidence, "urlopen", side_effect=opening):
            path = Path(temporary) / "data"
            first = evidence.fetch("https://example.test/pinned/manifest.json", digest, path)
            self.assertEqual(first["downloaded_files"], 1)
            self.assertFalse((path / "reference/weights.bin").exists())
            again = evidence.fetch("https://example.test/pinned/manifest.json", digest, path)
            self.assertEqual(again["reused_files"], 1)
            (path / "source/rows.json").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                evidence.verify(path, manifest_sha256=digest)
            evidence.fetch("https://example.test/pinned/manifest.json", digest, path)
            evidence.verify(path, manifest_sha256=digest)
            self.assertEqual(urls.count("https://example.test/pinned/source/rows.json"), 2)

    def test_bad_download_does_not_replace_existing_file(self):
        manifest, _, _, _ = self.fixture()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(evidence, "urlopen", return_value=BytesIO(b"wrong")),
        ):
            target = Path(temporary) / "rows.json"
            target.write_bytes(b"preserved")
            with self.assertRaises(ValueError):
                evidence.download("https://example.test/rows.json", target, manifest["files"][0])
            self.assertEqual(target.read_bytes(), b"preserved")
            self.assertEqual(list(Path(temporary).glob("*.partial")), [])

    def test_destination_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "download"
            root.mkdir()
            (root / "source").symlink_to(Path(temporary), target_is_directory=True)
            with self.assertRaises(ValueError):
                evidence.artifact_path(root, "source/outside")


if __name__ == "__main__":
    unittest.main()
