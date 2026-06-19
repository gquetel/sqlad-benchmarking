"""Feature extractors and the registry that exposes them by short name.

A *feature extractor* is any sklearn-compatible transformer mapping a column of
raw SQL queries to a numeric matrix (dense or sparse). They are registered in
:data:`EXTRACTORS` so decision models and the evaluation suite can request one
by name (``"li"``, ``"cv"``, ``"sbert"``) and stay agnostic to which extractor is
in use. To add a new extractor: implement the transformer, then register it here.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from sklearn.base import TransformerMixin

from mlops_sqldetect.features.cache import CachingExtractor, maybe_wrap, resolve_cache_dir
from mlops_sqldetect.features.codet5 import CodeT5Extractor
from mlops_sqldetect.features.countvect import CountVectorizerExtractor
from mlops_sqldetect.features.li import LiExtractor, extract_li_features
from mlops_sqldetect.features.loginov import LoginovExtractor, extract_loginov_features
from mlops_sqldetect.features.securebert import SecureBertExtractor

# Short name -> zero-arg factory returning a fresh, unfitted extractor.
EXTRACTORS: dict[str, Callable[[], TransformerMixin]] = {
    "li": LiExtractor,
    "cv": CountVectorizerExtractor,
    "sbert": SecureBertExtractor,
    "loginov": LoginovExtractor,
    "codet5": CodeT5Extractor,
}

# Default extractor used when a caller does not specify one.
DEFAULT_EXTRACTOR = "li"

# Human-readable labels for logs and MLflow run names. Acronyms stay uppercase;
# labels mirror their reference works (Li et al., SecureBERT).
EXTRACTOR_LABELS: dict[str, str] = {
    "li": "Li",
    "cv": "CountVectorizer",
    "sbert": "SecureBERT",
    "loginov": "Loginov",
    "codet5": "CodeT5+",
}


def build_extractor(name: str = DEFAULT_EXTRACTOR, cache_dir: str | os.PathLike | None = None) -> TransformerMixin:
    """Instantiate a feature extractor by short name (see :data:`EXTRACTORS`).

    When ``cache_dir`` is set the extractor is wrapped in a
    :class:`~mlops_sqldetect.features.cache.CachingExtractor` that memoises its
    ``transform`` output to disk (see :func:`resolve_cache_dir`).
    """
    try:
        factory = EXTRACTORS[name]
    except KeyError:
        raise ValueError(f"Unknown feature extractor: {name!r} (expected one of {sorted(EXTRACTORS)})") from None
    return maybe_wrap(factory(), cache_dir)


__all__ = [
    "DEFAULT_EXTRACTOR",
    "EXTRACTORS",
    "EXTRACTOR_LABELS",
    "CachingExtractor",
    "CodeT5Extractor",
    "CountVectorizerExtractor",
    "LiExtractor",
    "LoginovExtractor",
    "SecureBertExtractor",
    "build_extractor",
    "extract_li_features",
    "extract_loginov_features",
    "resolve_cache_dir",
]
