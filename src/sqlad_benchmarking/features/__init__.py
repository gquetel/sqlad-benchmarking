"""Feature extractors and the registry that exposes them by short name.

A *feature extractor* is any sklearn-compatible transformer mapping a column of
raw SQL queries to a numeric matrix (dense or sparse). They are registered in
:data:`EXTRACTORS` so decision models and the evaluation suite can request one
by name (``"li"``, ``"cv"``, ``"tfidf"``, ``"sbert"``) and stay agnostic to which extractor is
in use. To add a new extractor: implement the transformer, then register it here.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable

from sklearn.base import TransformerMixin

from sqlad_benchmarking.features.cache import CachingExtractor, maybe_wrap, resolve_cache_dir
from sqlad_benchmarking.features.codebert import CodeBertExtractor
from sqlad_benchmarking.features.codet5 import CodeT5Extractor
from sqlad_benchmarking.features.countvect import CountVectorizerExtractor
from sqlad_benchmarking.features.flan_t5 import FlanT5Extractor
from sqlad_benchmarking.features.gaur import GaurExtractor
from sqlad_benchmarking.features.kakisim import KakisimExtractor
from sqlad_benchmarking.features.li import LiExtractor, extract_li_features
from sqlad_benchmarking.features.loginov import LoginovExtractor, extract_loginov_features
from sqlad_benchmarking.features.modernbert import ModernBertExtractor
from sqlad_benchmarking.features.qwen3_emb import Qwen3EmbExtractor
from sqlad_benchmarking.features.roberta import RobertaExtractor
from sqlad_benchmarking.features.securebert import SecureBert2Extractor, SecureBertExtractor
from sqlad_benchmarking.features.sentbert import SentBertExtractor
from sqlad_benchmarking.features.tfidf import TfidfExtractor

# Short name -> zero-arg factory returning a fresh, unfitted extractor.
EXTRACTORS: dict[str, Callable[[], TransformerMixin]] = {
    "li": LiExtractor,
    "cv": CountVectorizerExtractor,
    "tfidf": TfidfExtractor,
    "sbert": SecureBertExtractor,
    "sbert2": SecureBert2Extractor,
    "loginov": LoginovExtractor,
    "kakisim": KakisimExtractor,
    "codet5": CodeT5Extractor,
    "roberta": RobertaExtractor,
    "modernbert": ModernBertExtractor,
    "codebert": CodeBertExtractor,
    "flan-t5": FlanT5Extractor,
    "sentbert": SentBertExtractor,
    "qwen3-emb": Qwen3EmbExtractor,
    "gaur-expert": functools.partial(GaurExtractor, mode="expert"),
    "gaur-chatgpt": functools.partial(GaurExtractor, mode="chatgpt"),
    "gaur-claude": functools.partial(GaurExtractor, mode="claude"),
    "gaur-llama": functools.partial(GaurExtractor, mode="llama"),
    "gaur-mistral": functools.partial(GaurExtractor, mode="mistral"),
    "gaur-gpt-oss": functools.partial(GaurExtractor, mode="gpt-oss"),
    "gaur-ruleid": functools.partial(GaurExtractor, mode="ruleid"),
}

GPU_EXTRACTORS = frozenset(
    {
        "sbert",
        "sbert2",
        "codet5",
        "roberta",
        "modernbert",
        "codebert",
        "flan-t5",
        "sentbert",
        "qwen3-emb",
    }
)

# Default extractor used when a caller does not specify one.
DEFAULT_EXTRACTOR = "li"

# Human-readable labels for logs and MLflow run names. Acronyms stay uppercase;
# labels mirror their reference works (Li et al., SecureBERT, GAUR).
EXTRACTOR_LABELS: dict[str, str] = {
    "li": "Li",
    "cv": "CountVectorizer",
    "tfidf": "TF-IDF",
    "sbert": "SecureBERT",
    "sbert2": "SecureBERT2",
    "loginov": "Loginov",
    "kakisim": "Kakisim",
    "codet5": "CodeT5+",
    "roberta": "RoBERTa-base",
    "modernbert": "ModernBERT-base",
    "codebert": "CodeBERT",
    "flan-t5": "Flan-T5-Small",
    "sentbert": "SentenceBERT-mpnet",
    "qwen3-emb": "Qwen3-Emb-0.6B",
    "gaur-expert": "GAUR (Expert)",
    "gaur-chatgpt": "GAUR (ChatGPT)",
    "gaur-claude": "GAUR (Claude)",
    "gaur-llama": "GAUR (Llama)",
    "gaur-mistral": "GAUR (Mistral)",
    "gaur-gpt-oss": "GAUR (GPT-OSS)",
    "gaur-ruleid": "GAUR (RuleID)",
}


def extractor_observes_insider(name: str) -> bool:
    """Whether the named collector observes insider traffic (queries run directly against the DBMS).

    Declared per collector via its ``observes_insider`` class attribute; resolved off the
    registered factory (unwrapping ``functools.partial``) so no extractor is built.
    """
    factory = EXTRACTORS.get(name)
    cls = getattr(factory, "func", factory)  # unwrap functools.partial(GaurExtractor, ...)
    return bool(getattr(cls, "observes_insider", False))


def build_extractor(name: str = DEFAULT_EXTRACTOR, cache_dir: str | os.PathLike | None = None) -> TransformerMixin:
    """Instantiate a feature extractor by short name (see :data:`EXTRACTORS`).

    When ``cache_dir`` is set the extractor is wrapped in a
    :class:`~sqlad_benchmarking.features.cache.CachingExtractor` that memoises its
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
    "GPU_EXTRACTORS",
    "CachingExtractor",
    "CodeBertExtractor",
    "CodeT5Extractor",
    "CountVectorizerExtractor",
    "FlanT5Extractor",
    "GaurExtractor",
    "KakisimExtractor",
    "LiExtractor",
    "LoginovExtractor",
    "ModernBertExtractor",
    "Qwen3EmbExtractor",
    "RobertaExtractor",
    "SecureBert2Extractor",
    "SecureBertExtractor",
    "SentBertExtractor",
    "TfidfExtractor",
    "build_extractor",
    "extractor_observes_insider",
    "extract_li_features",
    "extract_loginov_features",
    "resolve_cache_dir",
]
