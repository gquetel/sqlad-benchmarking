"""Big-Superviz26-SQL dataset registry and split loader.

The ANUBIS size-sufficiency variant of Superviz26: identical 8 scenarios and 1M
test splits, but each train split is doubled to 200k (``Big = Normal ∪ Extra``, a
strict superset of the 100k Normal train set). The CSVs are produced once by
``experiments.build_big_trainsets`` and stored outside the repo under
``~/datasets/200k-training/``. The scenario names, columns, and ``manifest`` are reused
verbatim from :mod:`superviz26` so both plug into the same ``DatasetFamily``.
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
    """Location of the 200k Big CSVs (outside the repo; see build_big_trainsets)."""
    return Path("~/datasets/200k-training").expanduser()


def resolve_path(name: Superviz26, root: Path | None = None) -> Path:
    """Absolute path to the Big CSV for ``name`` under ``root`` (default: ~/datasets/200k-training)."""
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

    The train split is the 200k benign-only Big set; the test split is the same 1M
    split as the Normal Superviz26 runs. When the CSV is absent from the default root it
    is auto-downloaded from Zenodo; a ``FileNotFoundError`` (pointing at the
    fetcher/builder) is raised only if it is still missing afterwards.
    """
    path = resolve_path(name, root)
    if not path.exists() and root is None:
        from tools.fetch_supplementary import ensure_group

        ensure_group("big")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it: python -m tools.fetch_supplementary --groups big "
            f"(or rebuild: python -m experiments.build_big_trainsets --scenario {name.value})"
        )
    return load_split_csv(path, split, columns=columns, limit=limit, seed=seed)
