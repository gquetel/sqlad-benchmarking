"""Reserved Superviz26 variant with an even larger training set (future work).

A placeholder family for a future training set larger than the standard full-split
Superviz26 (which now backs the default ``superviz26`` family). Identical 8 scenarios,
columns, and ``manifest`` reused verbatim from :mod:`superviz26` so both plug into the
same ``DatasetFamily``; its CSVs are not yet generated and are expected outside the repo
under ``~/datasets/superviz26-xl/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlops_sqldetect.data import load_split_csv
from mlops_sqldetect.datasets.superviz26 import (
    IN_DOMAIN,
    LODO,
    Split,
    Superviz26,
    manifest,
)

__all__ = ["IN_DOMAIN", "LODO", "Split", "Superviz26", "default_root", "load_split", "manifest", "resolve_path"]


def default_root() -> Path:
    """Location of the reserved XL CSVs (outside the repo; not yet generated)."""
    return Path("~/datasets/superviz26-xl").expanduser()


def resolve_path(name: Superviz26, root: Path | None = None) -> Path:
    """Absolute path to the XL CSV for ``name`` under ``root`` (default: ~/datasets/superviz26-xl)."""
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
    """Load rows of the reserved XL Superviz26 CSV that belong to ``split``.

    The XL CSVs hold a training set larger than the standard full-split Superviz26 and
    are generated locally (not on Zenodo); they are not yet produced, so a missing file
    raises ``FileNotFoundError`` rather than triggering a download.
    """
    path = resolve_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The XL training set is reserved for future work and is not yet "
            f"generated; it is expected at ~/datasets/superviz26-xl/."
        )
    return load_split_csv(path, split, columns=columns, limit=limit, seed=seed)
