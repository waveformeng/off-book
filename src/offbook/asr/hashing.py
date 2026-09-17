"""Model identity: pinned revisions and content hashes, logged with every result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

LOCK_PATH = Path(__file__).resolve().parents[3] / "models.lock.json"


def sha256_files(paths: list[Path]) -> str:
    """Content hash over the given files, in sorted-name order."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: p.name):
        h.update(p.name.encode())
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def load_lock() -> dict[str, dict[str, str]]:
    with LOCK_PATH.open() as f:
        data: dict[str, dict[str, str]] = json.load(f)
    return data


def pinned_snapshot(key: str, allow_patterns: list[str]) -> tuple[Path, str, str]:
    """Return (snapshot dir, model id, revision) for a lock entry. Downloads only if absent."""
    entry = load_lock()[key]
    path = snapshot_download(
        entry["model_id"], revision=entry["revision"], allow_patterns=allow_patterns
    )
    return Path(path), entry["model_id"], entry["revision"]


def verify_hash(key: str, computed: str) -> None:
    expected = load_lock()[key]["model_hash"]
    if expected and expected != computed:
        raise RuntimeError(
            f"model hash mismatch for {key}: lock={expected[:16]} computed={computed[:16]}"
        )
