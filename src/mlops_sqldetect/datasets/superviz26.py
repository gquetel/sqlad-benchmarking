"""Superviz26-SQL dataset registry and split loader.

Wraps the 8 full-split CSVs in a typed enum, so callers cannot mistype dataset
names, and exposes a single ``load_split`` entry point that filters by the
``split`` column without materialising the whole file in memory unnecessarily.
The CSVs live under ``~/datasets/superviz26-lodo/`` and are either generated locally
by the dataset generator's full-split mode or downloaded from Zenodo with
``python -m tools.fetch_superviz26 --groups main``; the in-repo ``MANIFEST.json``
supplies the shared column/scenario metadata and the archive checksums.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd

from mlops_sqldetect.data import load_split_csv

Split = Literal["train", "test"]

_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "superviz26" / "MANIFEST.json"


class Superviz26(StrEnum):
    """Short names of the 8 Superviz26-SQL CSVs.

    In-domain files are ``A_A`` … ``D_D`` (train and test from the same domain).
    LODO files are ``BCD_A`` … ``ABC_D`` (train on three domains, test on the held-out one).
    """

    A_A = "a-a"
    B_B = "b-b"
    C_C = "c-c"
    D_D = "d-d"
    BCD_A = "bcd-a"
    ACD_B = "acd-b"
    ABD_C = "abd-c"
    ABC_D = "abc-d"


IN_DOMAIN: tuple[Superviz26, ...] = (
    Superviz26.A_A,
    Superviz26.B_B,
    Superviz26.C_C,
    Superviz26.D_D,
)
LODO: tuple[Superviz26, ...] = (
    Superviz26.BCD_A,
    Superviz26.ACD_B,
    Superviz26.ABD_C,
    Superviz26.ABC_D,
)


def default_root() -> Path:
    """Location of the full-split CSVs (outside the repo; see datasets.md)."""
    return Path("~/datasets/superviz26-lodo").expanduser()


def manifest() -> dict:
    """Parsed MANIFEST.json — the single source of truth on file metadata."""
    return json.loads(_MANIFEST_PATH.read_text())


def resolve_path(name: Superviz26, root: Path | None = None) -> Path:
    """Absolute path to the CSV for ``name`` under ``root`` (default: repo data dir)."""
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
    """Load rows from the chosen Superviz26 CSV that belong to ``split``.

    Args:
        name: Which of the 8 datasets to load.
        split: ``train`` or ``test``.
        root: Directory holding the CSVs. Defaults to ``~/datasets/superviz26-lodo/``.
        columns: Columns to keep. The default trims the file to the three columns
            actually needed by the Li pipelines, cutting memory by ~10x.

    Returns:
        DataFrame containing only rows where the ``split`` column equals ``split``.

    Raises:
        FileNotFoundError: If the CSV is not on disk. Hints at the fetcher.
        ValueError: If ``split`` column or required columns are missing.
    """
    path = resolve_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the 8 scenario CSVs with "
            f"`python -m tools.fetch_superviz26 --groups main`, or generate them locally with the "
            f"dataset generator's full-split mode: `python experiments/generate_splits.py --full` "
            f"(in legacy-sqlia-dataset-generator). Both write to ~/datasets/superviz26-lodo/."
        )
    return load_split_csv(path, split, columns=columns, limit=limit, seed=seed)
