"""Download the Superviz26-SQL dataset from Zenodo with checksum verification.

Usage:
    python -m tools.fetch_superviz26
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
DATA_DIR = REPO_ROOT / "data" / "raw" / "superviz26"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class FileSpec:
    """One entry from the manifest's ``files`` mapping."""

    name: str
    url: str
    bytes_: int
    sha256: str

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
            bytes_=meta["bytes"],
            sha256=meta["sha256"],
        )
        for name, meta in manifest["files"].items()
    ]
    return manifest, specs


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 of a file in streaming fashion."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def _is_valid(spec: FileSpec) -> bool:
    """True iff ``spec.path`` exists with the right size and SHA-256."""
    if not spec.path.exists():
        return False
    if spec.path.stat().st_size != spec.bytes_:
        return False
    return _sha256_file(spec.path) == spec.sha256


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
        logger.info(f"OK   {spec.name} ({spec.bytes_:,} bytes, sha256 verified)")
        return True

    logger.info(f"Downloading  {spec.name} -> {spec.path}")
    _download(spec, resume=not force)

    actual_size = spec.path.stat().st_size
    if actual_size != spec.bytes_:
        logger.error(f"FAIL {spec.name}: expected {spec.bytes_} bytes, got {actual_size}")
        return False

    actual_sha256 = _sha256_file(spec.path)
    if actual_sha256 != spec.sha256:
        logger.error(f"FAIL {spec.name}: expected sha256 {spec.sha256}, got {actual_sha256}")
        return False

    logger.info(f"OK   {spec.name} (verified)")
    return True


def fetch(
    datasets: Annotated[
        str,
        typer.Option(help="Comma-separated short names (e.g. 'a-a,bcd-a'). Omit for all."),
    ] = "",
    force: Annotated[bool, typer.Option(help="Re-download even if a valid file is present.")] = False,
    check: Annotated[bool, typer.Option(help="Verify checksums of present files; do not download.")] = False,
) -> None:
    """Download (or verify) Superviz26-SQL CSVs into ``data/raw/superviz26/``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _, specs = load_manifest()

    if datasets:
        wanted = {s.strip() for s in datasets.split(",") if s.strip()}
        unknown = wanted - {s.name.removesuffix(".csv") for s in specs}
        if unknown:
            raise typer.BadParameter(f"Unknown dataset(s): {sorted(unknown)}")
        specs = [s for s in specs if s.name.removesuffix(".csv") in wanted]

    failures: list[str] = []
    for spec in specs:
        if check:
            ok = _is_valid(spec)
            status = "OK  " if ok else "BAD "
            logger.info(f"{status} {spec.name}")
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
