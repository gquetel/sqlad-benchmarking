"""Same-domain concept-drift registry and split loader for Superviz26.

The concept-drift protocol re-partitions a *single* domain through its per-sample
query-template metadata: for one domain, the templates are split 50/50 per
statement type into an origin set (S1) and a held-out shifted set (S2). A detector
is trained on the benign S1 train rows and evaluated twice — on the S1 test set
(reference) and on the never-seen S2 set (post-drift) — and drift robustness is the
AUROC drop between the two. See :mod:`mlops_sqldetect.evaluate_drift`.

The four per-domain CSVs (``a.csv`` … ``d.csv``) carry the standard Superviz26
columns plus a ``drift_set`` column (``origin``/``shifted``) that, together with
``split``, tells the three partitions apart:

  * ``(train, origin)``  -> ``origin_train``  (benign rows train the detector)
  * ``(test,  origin)``  -> ``origin_test``   (S1 reference evaluation)
  * ``(test,  shifted)`` -> ``shifted_test``  (S2 post-drift evaluation)

They are built outside the repo by ``experiments/build_concept_drift.py`` and
stored under ``~/datasets/concept-drift/``. The columns and ``manifest`` are reused
verbatim from :mod:`superviz26`.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd

from mlops_sqldetect.data import load_dataset, load_split_csv, stratified_subsample
from mlops_sqldetect.datasets.superviz26 import Split, manifest

__all__ = [
    "DOMAINS",
    "Split",
    "Superviz26Drift",
    "default_root",
    "load_drift",
    "load_split",
    "manifest",
    "resolve_path",
]

DriftSet = Literal["origin", "shifted"]


class Superviz26Drift(StrEnum):
    """Short names of the four per-domain concept-drift CSVs (file is ``<x>.csv``)."""

    A = "a"
    B = "b"
    C = "c"
    D = "d"


DOMAINS: tuple[Superviz26Drift, ...] = (
    Superviz26Drift.A,
    Superviz26Drift.B,
    Superviz26Drift.C,
    Superviz26Drift.D,
)


def default_root() -> Path:
    """Location of the per-domain concept-drift CSVs (outside the repo)."""
    return Path("~/datasets/concept-drift").expanduser()


def resolve_path(name: Superviz26Drift, root: Path | None = None) -> Path:
    """Absolute path to the drift CSV for ``name`` under ``root`` (default: ~/datasets/concept-drift)."""
    return (root or default_root()) / f"{name.value}.csv"


def load_split(
    name: Superviz26Drift,
    split: Split,
    *,
    root: Path | None = None,
    columns: tuple[str, ...] = ("full_query", "label", "split"),
    limit: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load the rows of one domain's CSV that belong to ``split`` (both drift sets).

    Provided for ``DatasetFamily`` contract uniformity; the drift evaluator uses the
    three-way :func:`load_drift` instead, because a single split mixes the origin and
    shifted partitions that the protocol must keep apart.
    """
    path = resolve_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python -m experiments.build_concept_drift --domain {name.value}"
        )
    return load_split_csv(path, split, columns=columns, limit=limit, seed=seed)


def load_drift(
    name: Superviz26Drift,
    *,
    root: Path | None = None,
    columns: tuple[str, ...] = ("full_query", "label", "attack_technique"),
    limit: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(origin_train, origin_test, shifted_test)`` for one domain.

    ``origin_train`` are the S1 train rows (filter to ``label == 0`` to train the
    one-class detector), ``origin_test`` is the S1 test set (reference AUROC), and
    ``shifted_test`` is the held-out S2 test set (post-drift AUROC). When ``limit``
    is set, each partition is independently label-stratified down for smoke runs.

    Raises:
        FileNotFoundError: If the domain CSV is not on disk. Hints at the builder.
        ValueError: If the ``split`` or ``drift_set`` column is missing.
    """
    path = resolve_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python -m experiments.build_concept_drift --domain {name.value}"
        )
    cols = tuple({*columns, "full_query", "label", "split", "drift_set"})
    df = load_dataset(path, usecols=cols)
    for required in ("split", "drift_set"):
        if required not in df.columns:
            raise ValueError(f"{path} has no '{required}' column")

    origin = df[df["drift_set"] == "origin"]
    origin_train = origin[origin["split"] == "train"].reset_index(drop=True)
    origin_test = origin[origin["split"] == "test"].reset_index(drop=True)
    shifted_test = df[(df["drift_set"] == "shifted") & (df["split"] == "test")].reset_index(drop=True)

    return (
        stratified_subsample(origin_train, limit, seed),
        stratified_subsample(origin_test, limit, seed),
        stratified_subsample(shifted_test, limit, seed),
    )
