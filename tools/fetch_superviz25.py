"""Download the Superviz25-SQL dataset from Zenodo with checksum verification.

Superviz25 is a single ``dataset.csv`` (~1.1 GB). The manifest carries an MD5
(that is what the Zenodo record publishes) and no byte count, so verification
falls back to MD5 and skips the size check when ``bytes`` is absent.

Usage:
    python -m tools.fetch_superviz25
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from urllib.request import Request, urlopen

import typer

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "superviz25"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class FileSpec:
    """One entry from the manifest's ``files`` mapping."""

    name: str
    url: str
    bytes_: int | None
    sha256: str | None
    md5: str | None

    @property
    def path(self) -> Path:
        return DATA_DIR / self.name


def load_manifest() -> tuple[dict, list[FileSpec]]:
    """Parse MANIFEST.json into typed specs."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    pattern = manifest["url_pattern"]
    specs = [
        FileSpec(
            name=name,
            url=pattern.format(filename=name),
            bytes_=meta.get("bytes"),
            sha256=meta.get("sha256"),
            md5=meta.get("md5"),
        )
        for name, meta in manifest["files"].items()
    ]
    return manifest, specs


def _digest_file(path: Path, algo: str) -> str:
    """Compute a hash of a file in streaming fashion."""
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def _checksum_ok(spec: FileSpec) -> bool:
    """Verify whichever checksum the manifest provides (sha256 preferred, else md5)."""
    if spec.sha256:
        return _digest_file(spec.path, "sha256") == spec.sha256
    if spec.md5:
        return _digest_file(spec.path, "md5") == spec.md5
    return True  # nothing to check against


def _is_valid(spec: FileSpec) -> bool:
    """True iff ``spec.path`` exists, matches the size (when known) and checksum."""
    if not spec.path.exists():
        return False
    if spec.bytes_ is not None and spec.path.stat().st_size != spec.bytes_:
        return False
    return _checksum_ok(spec)


def _download(spec: FileSpec, *, resume: bool) -> None:
    """Download ``spec`` to disk. Supports HTTP Range resume on partial files."""
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = spec.path.with_suffix(spec.path.suffix + ".part")

    start = tmp.stat().st_size if resume and tmp.exists() else 0
    headers: dict[str, str] = {}
    mode = "wb"
    if start > 0:
        headers["Range"] = f"bytes={start}-"
        mode = "ab"
        logger.info(f"Resuming {spec.name} from byte {start}")

    if not spec.url.startswith(("https://", "http://")):
        raise ValueError(f"Unexpected URL scheme: {spec.url}")
    req = Request(spec.url, headers=headers)  # noqa: S310
    with urlopen(req) as response, tmp.open(mode) as out:  # noqa: S310
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)

    tmp.replace(spec.path)


def _ensure_one(spec: FileSpec, *, force: bool) -> bool:
    """Make ``spec.path`` exist and match the manifest. Returns True on success."""
    if not force and _is_valid(spec):
        logger.info(f"OK   {spec.name} (checksum verified)")
        return True

    logger.info(f"Downloading  {spec.name} -> {spec.path}")
    _download(spec, resume=not force)

    if spec.bytes_ is not None and spec.path.stat().st_size != spec.bytes_:
        logger.error(f"FAIL {spec.name}: expected {spec.bytes_} bytes, got {spec.path.stat().st_size}")
        return False
    if not _checksum_ok(spec):
        logger.error(f"FAIL {spec.name}: checksum mismatch")
        return False

    logger.info(f"OK   {spec.name} (verified)")
    return True


def fetch(
    force: Annotated[bool, typer.Option(help="Re-download even if a valid file is present.")] = False,
    check: Annotated[bool, typer.Option(help="Verify checksums of present files; do not download.")] = False,
) -> None:
    """Download (or verify) Superviz25-SQL into ``data/raw/superviz25/``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _, specs = load_manifest()

    failures: list[str] = []
    for spec in specs:
        if check:
            ok = _is_valid(spec)
            logger.info(f"{'OK  ' if ok else 'BAD '} {spec.name}")
            if not ok:
                failures.append(spec.name)
            continue
        if not _ensure_one(spec, force=force):
            failures.append(spec.name)

    if failures:
        logger.error(f"{len(failures)} file(s) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    typer.run(fetch)
