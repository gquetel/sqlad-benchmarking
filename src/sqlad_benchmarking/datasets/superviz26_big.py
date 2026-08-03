"""Superviz26 variant with an even larger training set.

A family backed by a training set larger than the standard full-split Superviz26
(which backs the default ``superviz26`` family). Identical 8 scenarios, columns, and
``manifest`` reused verbatim from :mod:`superviz26` so both plug into the same
``DatasetFamily``; its CSVs live outside the repo under ``~/datasets/superviz26-big/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sqlad_benchmarking.data import load_split_csv
from sqlad_benchmarking.datasets.superviz26 import (
    IN_DOMAIN,
    LODO,
    Split,
    Superviz26,
    manifest,
)

__all__ = ["IN_DOMAIN", "LODO", "Split", "Superviz26", "default_root", "load_split", "manifest", "resolve_path"]


def default_root() -> Path:
    """Location of the Big CSVs (outside the repo)."""
    return Path("~/datasets/superviz26-big").expanduser()


def resolve_path(name: Superviz26, root: Path | None = None) -> Path:
    """Absolute path to the Big CSV for ``name`` under ``root`` (default: ~/datasets/superviz26-big)."""
    return (root or default_root()) / f"{name.value}.csv"


def load_split(
    name: Superviz26,
    split: Split,
    *,
    root: Path | None = None,
    columns: tuple[str, ...] = ("full_query", "label", "split"),
    limit: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load rows of the Big Superviz26 CSV that belong to ``split``.

    The Big CSVs hold a training set larger than the standard full-split Superviz26 and
    are generated locally (not on Zenodo), so a missing file raises ``FileNotFoundError``
    rather than triggering a download.
    """
    path = resolve_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The Big training set is generated locally; it is expected at "
            f"~/datasets/superviz26-big/."
        )
    return load_split_csv(path, split, columns=columns, limit=limit, seed=seed)
