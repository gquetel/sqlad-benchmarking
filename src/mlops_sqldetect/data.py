"""Dataset loading for the SQL injection detection pipelines.

Expected CSV schema: a ``full_query`` column (raw SQL string) and a ``label``
column (``0`` for normal samples, ``1`` for SQLi attacks). Optional columns such
as ``split`` are preserved when requested via ``usecols``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("full_query", "label")

logger = logging.getLogger(__name__)


def load_dataset(
    data_path: Path,
    usecols: Iterable[str] | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load a labelled SQL dataset from CSV.

    Args:
        data_path: Path to a CSV containing at least ``full_query`` and ``label``.
        usecols: Optional iterable of columns to load. Required columns are
            always included even if omitted. Trimming columns at read time keeps
            memory bounded on 300 MB+ Superviz26 files.
        limit: If set and smaller than the loaded row count, return a
            label-stratified random subset of approximately ``limit`` rows.
            Intended for smoke tests.
        seed: Random state for the subsample so smoke runs are deterministic.

    Returns:
        DataFrame with rows containing NaN in the required columns dropped and
        labels coerced to int.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required column is missing.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dtype: dict = {"full_query": str, "label": "Int64"}
    read_kwargs: dict = {"dtype": dtype}
    if usecols is not None:
        cols = list({*usecols, *REQUIRED_COLUMNS})
        read_kwargs["usecols"] = cols
        # Pin metadata columns to str so chunked reads can't infer mixed
        # object/float chunks (pandas low-memory concat raises IndexError).
        for c in cols:
            dtype.setdefault(c, "str")

    # Trace the exact file opened so a cache hit can be tied back to its source.
    logger.info("Loading dataset: %s", data_path.resolve())
    df = pd.read_csv(data_path, **read_kwargs)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return stratified_subsample(df, limit=limit, seed=seed)


def stratified_subsample(
    df: pd.DataFrame,
    limit: int | None,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a label-stratified subsample of ``df`` of approximately ``limit`` rows."""
    if limit is None or len(df) <= limit:
        return df
    return df.groupby("label", group_keys=False).sample(frac=limit / len(df), random_state=seed).reset_index(drop=True)


def load_split_csv(
    path: Path,
    split: str,
    *,
    columns: tuple[str, ...] = ("full_query", "label", "split"),
    limit: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load the rows of a CSV whose ``split`` column equals ``split``.

    Shared by every dataset family (Superviz25, Superviz26, ...): they resolve the
    file path and report a family-specific fetch hint, then defer the column
    trimming + split filtering + subsampling to here.

    Raises:
        ValueError: If the file has no ``split`` column.
    """
    cols = tuple(columns)
    if "split" not in cols:
        cols = cols + ("split",)
    df = load_dataset(path, usecols=cols)
    if "split" not in df.columns:
        raise ValueError(f"{path} has no 'split' column")
    out = df[df["split"] == split].reset_index(drop=True)
    return stratified_subsample(out, limit=limit, seed=seed)


def load_whole_sampled(
    path: Path,
    *,
    columns: tuple[str, ...] = ("full_query", "label", "split"),
    limit: int | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Read the whole CSV and (for smoke runs) sample ``limit`` rows before splitting.

    Samples ``limit`` rows from the *entire* file (all splits, both labels, no
    stratification) and leaves the ``split`` column intact for the caller to
    filter into train/test. This differs from :func:`load_split_csv`, which
    subsamples each split independently; here the split proportions follow the
    file's overall distribution. ``random_state=seed`` makes the selection
    deterministic.
    """
    cols = tuple(columns)
    if "split" not in cols:
        cols = cols + ("split",)
    df = load_dataset(path, usecols=cols)
    if "split" not in df.columns:
        raise ValueError(f"{path} has no 'split' column")
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
    return df


def split_normals(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the normal (label == 0) rows for one-class training."""
    return df[df["label"] == 0].reset_index(drop=True)
