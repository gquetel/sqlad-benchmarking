"""Big-Superviz26-SQL dataset registry and split loader.

The size-sufficiency variant of Superviz26: identical 8 scenarios, but each split is
drawn from the *full* per-domain pools instead of the 100k/1M Normal subsamples. The
CSVs are produced by the dataset generator's full-split mode
(``legacy-sqlia-dataset-generator``: ``python experiments/generate_splits.py --full``),
which downsamples every domain to the smallest domain's train/test pool so all
scenarios stay equal-sized, and stores them outside the repo under
``~/datasets/full-lodo/``. The scenario names, columns, and ``manifest`` are reused
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
    """Location of the full-split Big CSVs (outside the repo; see generate_splits --full)."""
    return Path("~/datasets/full-lodo").expanduser()


def resolve_path(name: Superviz26, root: Path | None = None) -> Path:
    """Absolute path to the Big CSV for ``name`` under ``root`` (default: ~/datasets/full-lodo)."""
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
    """Load rows of the full-split Big Superviz26 CSV that belong to ``split``.

    Both the train and test splits are drawn from the full per-domain pools, downsampled
    to the smallest domain so every scenario stays equal-sized. The CSVs are generated
    locally (not on Zenodo), so a missing file raises ``FileNotFoundError`` pointing at
    the generator rather than triggering a download.
    """
    path = resolve_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it with the dataset generator's full-split mode: "
            f"`python experiments/generate_splits.py --full` (in legacy-sqlia-dataset-generator), "
            f"which writes the 8 scenario CSVs to ~/datasets/full-lodo/."
        )
    return load_split_csv(path, split, columns=columns, limit=limit, seed=seed)
