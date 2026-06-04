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

from mlops_sqldetect.datasets import superviz25, superviz26
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
    """A registered dataset and its named evaluation suites."""

    name: str
    suites: dict[str, tuple[StrEnum, ...]]
    load_split: Callable[..., object]
    manifest: Callable[[], dict]
    default_root: Callable[[], Path]
    resolve_path: Callable[..., Path]


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
