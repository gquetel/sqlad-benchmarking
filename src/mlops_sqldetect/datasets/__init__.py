"""Dataset families and the registry that exposes them by name.

A *dataset family* (Superviz25, Superviz26, ...) bundles its scenarios into named
suites and provides a uniform ``load_split``/``manifest``/``default_root`` surface.
The evaluation suite looks one up in :data:`FAMILIES` by ``--dataset`` name and
stays agnostic to how many scenarios it holds (Superviz25 has one; Superviz26 has
eight split into in-domain and LODO).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from mlops_sqldetect.datasets import superviz25, superviz26, superviz26_big, superviz26_drift, superviz26_fsl
from mlops_sqldetect.datasets.superviz25 import Superviz25
from mlops_sqldetect.datasets.superviz26 import (
    IN_DOMAIN,
    LODO,
    Split,
    Superviz26,
    default_root,
    load_split,
    manifest,
    resolve_path,
)


@dataclass(frozen=True)
class DatasetFamily:
    """A registered dataset and its named evaluation suites.

    ``protocol`` selects the evaluator a family is run through: ``"suite"`` for the
    standard train-once/evaluate-once grid (:mod:`mlops_sqldetect.evaluate_suite`),
    ``"drift"`` for the train-once/evaluate-twice concept-drift protocol
    (:mod:`mlops_sqldetect.evaluate_drift`), ``"fsl"`` for the few-shot
    domain-adaptation protocol that fine-tunes a pretrained LODO autoencoder
    (:mod:`mlops_sqldetect.evaluate_fsl`). The SLURM runner dispatches on it.
    """

    name: str
    suites: dict[str, tuple[StrEnum, ...]]
    load_split: Callable[..., object]
    manifest: Callable[[], dict]
    default_root: Callable[[], Path]
    resolve_path: Callable[..., Path]
    protocol: str = "suite"


FAMILIES: dict[str, DatasetFamily] = {
    "superviz26": DatasetFamily(
        name="superviz26",
        suites={
            "in_domain": superviz26.IN_DOMAIN,
            "lodo": superviz26.LODO,
            "all": superviz26.IN_DOMAIN + superviz26.LODO,
        },
        load_split=superviz26.load_split,
        manifest=superviz26.manifest,
        default_root=superviz26.default_root,
        resolve_path=superviz26.resolve_path,
    ),
    "superviz25": DatasetFamily(
        name="superviz25",
        suites={"all": superviz25.ALL},
        load_split=superviz25.load_split,
        manifest=superviz25.manifest,
        default_root=superviz25.default_root,
        resolve_path=superviz25.resolve_path,
    ),
    "superviz26-big": DatasetFamily(
        name="superviz26-big",
        suites={
            "in_domain": superviz26_big.IN_DOMAIN,
            "lodo": superviz26_big.LODO,
            "all": superviz26_big.IN_DOMAIN + superviz26_big.LODO,
        },
        load_split=superviz26_big.load_split,
        manifest=superviz26_big.manifest,
        default_root=superviz26_big.default_root,
        resolve_path=superviz26_big.resolve_path,
    ),
    "superviz26-drift": DatasetFamily(
        name="superviz26-drift",
        # One scenario per domain; the drift evaluator trains once and evaluates the
        # origin (S1) and shifted (S2) test sets of that domain.
        suites={"all": superviz26_drift.DOMAINS},
        load_split=superviz26_drift.load_split,
        manifest=superviz26_drift.manifest,
        default_root=superviz26_drift.default_root,
        resolve_path=superviz26_drift.resolve_path,
        protocol="drift",
    ),
    "superviz26-fsl": DatasetFamily(
        name="superviz26-fsl",
        # One scenario per target domain; the few-shot evaluator adapts that domain's
        # pretrained LODO autoencoder on k benign target samples and evaluates it on
        # the domain's own test set. Data is re-used from the base superviz26 family.
        suites={"all": superviz26_fsl.DOMAINS},
        load_split=superviz26_fsl.load_split,
        manifest=superviz26_fsl.manifest,
        default_root=superviz26_fsl.default_root,
        resolve_path=superviz26_fsl.resolve_path,
        protocol="fsl",
    ),
}


__all__ = [
    "FAMILIES",
    "IN_DOMAIN",
    "LODO",
    "DatasetFamily",
    "Split",
    "Superviz25",
    "Superviz26",
    "default_root",
    "load_split",
    "manifest",
    "resolve_path",
]
