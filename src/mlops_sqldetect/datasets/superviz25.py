"""Superviz25-SQL dataset registry and split loader.

Wraps the single CSV published at https://zenodo.org/records/17086037 (one
OurAirports domain, ``train``/``test`` ``split`` column). Mirrors the
:mod:`superviz26` interface so both plug into the same ``DatasetFamily`` registry
and evaluation suite. Use ``tools.fetch_superviz25`` to put the file on disk.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd

from mlops_sqldetect.data import load_split_csv

Split = Literal["train", "test"]

_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "superviz25" / "MANIFEST.json"


class Superviz25(StrEnum):
    """Short name of the single Superviz25-SQL CSV (file is ``dataset.csv``)."""

    MAIN = "dataset"


ALL: tuple[Superviz25, ...] = (Superviz25.MAIN,)


def default_root() -> Path:
    """Repository default location for the downloaded CSV."""
    return _MANIFEST_PATH.parent


def manifest() -> dict:
    """Parsed MANIFEST.json — the single source of truth on file metadata."""
    return json.loads(_MANIFEST_PATH.read_text())


def resolve_path(name: Superviz25, root: Path | None = None) -> Path:
    """Absolute path to the CSV for ``name`` under ``root`` (default: repo data dir)."""
    return (root or default_root()) / f"{name.value}.csv"


def load_split(
    name: Superviz25,
    split: Split,
    *,
    root: Path | None = None,
    columns: tuple[str, ...] = ("full_query", "label", "split"),
    limit: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load rows of the Superviz25 CSV that belong to ``split``.

    Args:
        name: Dataset member (only ``MAIN``).
        split: ``train`` or ``test``. The train split is benign-only.
        root: Directory holding the CSV. Defaults to ``data/raw/superviz25/``.
        columns: Columns to keep (``split`` is always included).

    Raises:
        FileNotFoundError: If the CSV is not on disk. Hints at the fetcher.
        ValueError: If the ``split`` column is missing.
    """
    path = resolve_path(name, root)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m tools.fetch_superviz25")
    return load_split_csv(path, split, columns=columns, limit=limit, seed=seed)
