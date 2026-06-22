"""Download the heavy Superviz26-SQL supplementary datasets from Zenodo.

The Big (``200k-training``) and concept-drift (``concept-drift``) datasets back the
size-sufficiency and concept-drift experiments. They are *not* vendored and are
deliberately left out of the default ``fetch_data`` flow: extracted they are several
GB. Zenodo hosts them inside a single ``supplementary-materials.zip`` archive (record
20795955); this tool downloads that archive once (MD5-verified, HTTP-Range resume),
extracts the requested group(s) into their loader default roots, and verifies each
extracted CSV's size and SHA-256 against ``MANIFEST.json``.

Usage:
    python -m tools.fetch_supplementary --groups big,drift
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from urllib.request import Request, urlopen

import typer

from mlops_sqldetect.datasets import superviz26_big, superviz26_drift

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "supplementary" / "MANIFEST.json"
CHUNK_BYTES = 1 << 20

# Where each group's CSVs are extracted to. Sourced from the loader default roots so
# the download target never drifts from where the dataset modules read from.
GROUP_ROOTS: dict[str, Callable[[], Path]] = {
    "big": superviz26_big.default_root,
    "drift": superviz26_drift.default_root,
}


@dataclass(frozen=True)
class FileSpec:
    """One extractable CSV: its name in a group, its zip member, and its checksums."""

    group: str
    name: str
    member: str
    dest: Path
    bytes_: int
    sha256: str

    @property
    def path(self) -> Path:
        return self.dest / self.name


@dataclass(frozen=True)
class Archive:
    """The single Zenodo zip that bundles every supplementary group."""

    name: str
    url: str
    bytes_: int
    md5: str


def _archive_spec(manifest: dict) -> Archive:
    """Build the :class:`Archive` spec from a parsed manifest."""
    a = manifest["archive"]
    return Archive(name=a["name"], url=a["url"], bytes_=a["bytes"], md5=a["md5"])


def group_specs(group: str, root: Path | None = None) -> list[FileSpec]:
    """File specs for one group, extracted under ``root`` (default: the loader root)."""
    if group not in GROUP_ROOTS:
        raise ValueError(f"Unknown group {group!r}. Known: {sorted(GROUP_ROOTS)}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    meta = manifest["groups"][group]
    dest = root or GROUP_ROOTS[group]()
    prefix = meta["member_prefix"]
    return [
        FileSpec(
            group=group,
            name=name,
            member=f"{prefix}{name}",
            dest=dest,
            bytes_=fmeta["bytes"],
            sha256=fmeta["sha256"],
        )
        for name, fmeta in meta["files"].items()
    ]


def load_manifest() -> tuple[Archive, dict[str, list[FileSpec]]]:
    """Parse MANIFEST.json into the archive spec and per-group file specs."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    archive = _archive_spec(manifest)
    groups = {group: group_specs(group) for group in manifest["groups"]}
    return archive, groups


def _hash_file(path: Path, algo: str) -> str:
    """Compute a hex digest of a file in streaming fashion (digests are integrity-only)."""
    h = hashlib.new(algo, usedforsecurity=False)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def _is_valid(spec: FileSpec) -> bool:
    """True iff ``spec.path`` exists with the right size and SHA-256."""
    if not spec.path.exists() or spec.path.stat().st_size != spec.bytes_:
        return False
    return _hash_file(spec.path, "sha256") == spec.sha256


def _download_archive(archive: Archive, path: Path, *, force: bool) -> None:
    """Download the zip to ``path``, verifying MD5. Supports HTTP-Range resume."""
    if not force and path.exists() and path.stat().st_size == archive.bytes_:
        if _hash_file(path, "md5") == archive.md5:
            logger.info(f"OK   {archive.name} already present (md5 verified)")
            return
        logger.info(f"Re-downloading {archive.name}: md5 mismatch on cached archive")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")

    start = tmp.stat().st_size if not force and tmp.exists() else 0
    headers: dict[str, str] = {}
    mode = "wb"
    if start > 0:
        headers["Range"] = f"bytes={start}-"
        mode = "ab"
        logger.info(f"Resuming {archive.name} from byte {start:,}")

    if not archive.url.startswith(("https://", "http://")):
        raise ValueError(f"Unexpected URL scheme: {archive.url}")
    logger.info(f"Downloading  {archive.name} ({archive.bytes_:,} bytes) -> {path}")
    req = Request(archive.url, headers=headers)  # noqa: S310
    with urlopen(req) as response, tmp.open(mode) as out:  # noqa: S310
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)

    actual_md5 = _hash_file(path, "md5")
    if actual_md5 != archive.md5:
        raise ValueError(f"{archive.name}: expected md5 {archive.md5}, got {actual_md5}")
    logger.info(f"OK   {archive.name} (md5 verified)")


def _extract_one(zf: zipfile.ZipFile, spec: FileSpec) -> bool:
    """Stream one member out of the open zip and verify its size and SHA-256."""
    spec.dest.mkdir(parents=True, exist_ok=True)
    tmp = spec.path.with_suffix(spec.path.suffix + ".part")
    logger.info(f"Extracting   {spec.member} -> {spec.path}")
    with zf.open(spec.member) as src, tmp.open("wb") as out:
        for block in iter(lambda: src.read(CHUNK_BYTES), b""):
            out.write(block)
    tmp.replace(spec.path)

    actual_size = spec.path.stat().st_size
    if actual_size != spec.bytes_:
        logger.error(f"FAIL {spec.name}: expected {spec.bytes_} bytes, got {actual_size}")
        return False
    actual_sha256 = _hash_file(spec.path, "sha256")
    if actual_sha256 != spec.sha256:
        logger.error(f"FAIL {spec.name}: expected sha256 {spec.sha256}, got {actual_sha256}")
        return False
    logger.info(f"OK   {spec.name} (verified)")
    return True


def _extract_pending(
    archive: Archive, pending: list[FileSpec], *, archive_path: Path, force: bool, keep_archive: bool
) -> list[str]:
    """Download the archive and extract the ``pending`` specs. Returns failed ``group/name``."""
    _download_archive(archive, archive_path, force=force)
    failures: list[str] = []
    with zipfile.ZipFile(archive_path) as zf:
        for spec in pending:
            if not _extract_one(zf, spec):
                failures.append(f"{spec.group}/{spec.name}")
    if not keep_archive:
        logger.info(f"Removing cached archive {archive_path} (pass --keep-archive to retain)")
        archive_path.unlink(missing_ok=True)
    return failures


def ensure_group(group: str, *, root: Path | None = None, force: bool = False, keep_archive: bool = False) -> None:
    """Ensure every CSV of ``group`` exists under ``root`` and matches the manifest.

    Missing or corrupt files are fetched by downloading the Zenodo archive once and
    extracting just the pending members; a no-op when everything is already valid, so
    it is cheap to call on every load. This is the programmatic auto-fetch entry point
    used by the dataset loaders; :func:`fetch` is the multi-group CLI equivalent.

    Raises:
        RuntimeError: If any file still fails verification after extraction.
    """
    specs = group_specs(group, root)
    pending = specs if force else [s for s in specs if not _is_valid(s)]
    if not pending:
        return
    archive = _archive_spec(json.loads(MANIFEST_PATH.read_text()))
    archive_path = (root or GROUP_ROOTS[group]()).parent / archive.name
    logger.info(f"Auto-fetching {len(pending)} missing {group!r} dataset file(s) from {archive.url}")
    failures = _extract_pending(archive, pending, archive_path=archive_path, force=force, keep_archive=keep_archive)
    if failures:
        raise RuntimeError(f"Failed to fetch {group!r} file(s): {failures}")


def fetch(
    groups: Annotated[
        str,
        typer.Option(help="Comma-separated groups to fetch ('big', 'drift'). Omit for all."),
    ] = "",
    force: Annotated[bool, typer.Option(help="Re-download/re-extract even if valid files are present.")] = False,
    check: Annotated[bool, typer.Option(help="Verify checksums of present files; do not download.")] = False,
    keep_archive: Annotated[bool, typer.Option(help="Keep the downloaded zip after extraction.")] = False,
    archive_dir: Annotated[
        Path | None,
        typer.Option(help="Where to cache the downloaded zip (default: parent of the Big root, i.e. ~/datasets)."),
    ] = None,
) -> None:
    """Download (or verify) the heavy Big/concept-drift CSVs from the Zenodo archive."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    archive, all_groups = load_manifest()

    if groups:
        wanted = {g.strip() for g in groups.split(",") if g.strip()}
        unknown = wanted - all_groups.keys()
        if unknown:
            raise typer.BadParameter(f"Unknown group(s): {sorted(unknown)}. Known: {sorted(all_groups)}")
    else:
        wanted = set(all_groups)

    specs = [spec for group in sorted(wanted) for spec in all_groups[group]]

    if check:
        failures = []
        for spec in specs:
            ok = _is_valid(spec)
            logger.info(f"{'OK  ' if ok else 'BAD '} {spec.group}/{spec.name}")
            if not ok:
                failures.append(f"{spec.group}/{spec.name}")
        if failures:
            logger.error(f"{len(failures)} file(s) failed: {failures}")
            sys.exit(1)
        return

    pending = specs if force else [s for s in specs if not _is_valid(s)]
    if not pending:
        logger.info("All requested files already present and verified; nothing to do.")
        return

    archive_path = (archive_dir or GROUP_ROOTS["big"]().parent) / archive.name
    failures = _extract_pending(archive, pending, archive_path=archive_path, force=force, keep_archive=keep_archive)
    if failures:
        logger.error(f"{len(failures)} file(s) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    typer.run(fetch)
