"""Confirm a dataset CSV on disk matches its published Zenodo manifest.

Loaders call :func:`verify_file` so training only ever runs on bytes whose
sha256/md5 equals the checksum the fetcher verified against the Zenodo archive.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1 << 20

# (resolved path, size, mtime_ns) -> already verified, so a grid cell hashes each
# multi-GB CSV at most once per process (its train and test loads share the result).
_verified: set[tuple[str, int, int]] = set()


def file_digest(path: Path, algo: str) -> str:
    """Streaming hex digest of ``path`` (integrity-only, not for security)."""
    h = hashlib.new(algo, usedforsecurity=False)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, entry: Mapping[str, Any]) -> None:
    """Raise ValueError if ``path`` does not match the sha256/md5 recorded in ``entry``.

    A no-op when ``entry`` carries no checksum (locally-generated families).
    Memoised on the file's size and mtime.
    """
    algo = "sha256" if entry.get("sha256") else "md5" if entry.get("md5") else None
    if algo is None:
        return

    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key in _verified:
        return

    expected_bytes = entry.get("bytes")
    if expected_bytes is not None and stat.st_size != expected_bytes:
        raise ValueError(
            f"{path} is {stat.st_size} bytes but the manifest records {expected_bytes}; the file "
            f"differs from the Zenodo archive. Re-fetch it with the dataset fetcher."
        )

    expected = entry[algo]
    actual = file_digest(path, algo)
    if actual != expected:
        raise ValueError(
            f"{path} {algo} {actual} != manifest {expected}; the file differs from the Zenodo "
            f"archive (modified or regenerated). Re-fetch it with the dataset fetcher."
        )

    _verified.add(key)
    logger.info("Verified %s against the Zenodo manifest (%s).", path.name, algo)
