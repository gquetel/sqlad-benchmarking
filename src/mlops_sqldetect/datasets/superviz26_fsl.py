"""Few-shot domain-adaptation registry and loader for Superviz26.

The few-shot adaptation protocol does not introduce new data: it re-uses the
standard Superviz26 splits to ask how many unlabelled *target-domain* benign
queries suffice to recover in-domain detection performance. For one target domain
``X``:

  * the detector starts from the **pretrained LODO model** trained on the other
    three domains (e.g. for target ``a`` that is the ``bcd-a`` autoencoder);
  * it is fine-tuned on ``k`` benign samples drawn from ``X``'s own *train* split
    (the in-domain ``x-x`` CSV), with the feature extractor kept frozen;
  * it is evaluated on ``X``'s *test* split (the same held-out domain test set the
    LODO model is scored on).

So each target domain maps to two existing Superviz26 files: the in-domain
``x-x`` CSV (source of the ``k`` adaptation samples and of the evaluation test
set) and the LODO ``(others)-x`` CSV (whose pretrained checkpoint is the starting
point). See :mod:`mlops_sqldetect.evaluate_fsl`. ``manifest``/``default_root``/
``resolve_path`` are reused verbatim from :mod:`superviz26`, since the files are
the Superviz26 ones.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pandas as pd

from mlops_sqldetect.datasets import superviz26
from mlops_sqldetect.datasets.superviz26 import Split, Superviz26, default_root, manifest

__all__ = [
    "DOMAINS",
    "Split",
    "Superviz26FSL",
    "default_root",
    "lodo_source",
    "load_fsl",
    "load_split",
    "manifest",
    "resolve_path",
]


class Superviz26FSL(StrEnum):
    """The four target domains of the few-shot adaptation protocol (one per domain)."""

    A = "a"
    B = "b"
    C = "c"
    D = "d"


DOMAINS: tuple[Superviz26FSL, ...] = (
    Superviz26FSL.A,
    Superviz26FSL.B,
    Superviz26FSL.C,
    Superviz26FSL.D,
)

# Each target domain X maps to its in-domain file (x-x: source of the k adaptation
# samples and of the evaluation test set) and to the LODO file ((others)-x) whose
# pretrained checkpoint the few-shot model adapts from.
_IN_DOMAIN: dict[Superviz26FSL, Superviz26] = {
    Superviz26FSL.A: Superviz26.A_A,
    Superviz26FSL.B: Superviz26.B_B,
    Superviz26FSL.C: Superviz26.C_C,
    Superviz26FSL.D: Superviz26.D_D,
}
_LODO: dict[Superviz26FSL, Superviz26] = {
    Superviz26FSL.A: Superviz26.BCD_A,
    Superviz26FSL.B: Superviz26.ACD_B,
    Superviz26FSL.C: Superviz26.ABD_C,
    Superviz26FSL.D: Superviz26.ABC_D,
}


def lodo_source(target: Superviz26FSL) -> Superviz26:
    """Return the LODO scenario whose pretrained model is adapted for ``target``."""
    return _LODO[target]


def resolve_path(name: Superviz26FSL, root: Path | None = None) -> Path:
    """Absolute path to the in-domain (``x-x``) CSV backing target domain ``name``."""
    return superviz26.resolve_path(_IN_DOMAIN[name], root)


def load_split(
    name: Superviz26FSL,
    split: Split,
    *,
    root: Path | None = None,
    columns: tuple[str, ...] = ("full_query", "label", "split"),
    limit: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load the target domain's in-domain (``x-x``) rows for ``split``.

    Provided for ``DatasetFamily`` contract uniformity; the few-shot evaluator uses
    :func:`load_fsl` instead, which returns the train and test partitions together.
    """
    return superviz26.load_split(_IN_DOMAIN[name], split, root=root, columns=columns, limit=limit, seed=seed)


def load_fsl(
    target: Superviz26FSL,
    *,
    root: Path | None = None,
    test_limit: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(target_train, target_test)`` for one target domain.

    ``target_train`` are the in-domain train rows (filter to ``label == 0`` to draw
    the ``k`` benign adaptation samples); ``target_test`` is the in-domain test set
    (carrying ``attack_technique``), optionally label-stratified down to
    ``test_limit`` rows to keep the AUROC computation tractable.

    Raises:
        FileNotFoundError: If the in-domain CSV is not on disk (hints at the fetcher).
    """
    in_domain = _IN_DOMAIN[target]
    train = superviz26.load_split(in_domain, "train", root=root)
    test = superviz26.load_split(
        in_domain,
        "test",
        root=root,
        columns=("full_query", "label", "attack_technique"),
        limit=test_limit,
        seed=seed,
    )
    return train, test
