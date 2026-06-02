"""Feature extractors and the registry that exposes them by short name.

A *feature extractor* is any sklearn-compatible transformer mapping a column of
raw SQL queries to a fixed-width numeric matrix. They are registered in
:data:`EXTRACTORS` so decision models and the evaluation suite can request one
by name (``"li"``, ...) and stay agnostic to which extractor is in use. To add a
new extractor: implement the transformer, then register it here.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import TransformerMixin

from mlops_sqldetect.features.li import LiExtractor, extract_li_features

# Short name -> zero-arg factory returning a fresh, unfitted extractor.
EXTRACTORS: dict[str, Callable[[], TransformerMixin]] = {
    "li": LiExtractor,
}

# Default extractor used when a caller does not specify one.
DEFAULT_EXTRACTOR = "li"


def build_extractor(name: str = DEFAULT_EXTRACTOR) -> TransformerMixin:
    """Instantiate a feature extractor by short name (see :data:`EXTRACTORS`)."""
    try:
        factory = EXTRACTORS[name]
    except KeyError:
        raise ValueError(f"Unknown feature extractor: {name!r} (expected one of {sorted(EXTRACTORS)})") from None
    return factory()


__all__ = [
    "DEFAULT_EXTRACTOR",
    "EXTRACTORS",
    "LiExtractor",
    "build_extractor",
    "extract_li_features",
]
