"""CountVectorizer bag-of-words feature extractor.

Wraps :class:`sklearn.feature_extraction.text.CountVectorizer` so a column of raw
SQL queries maps to a sparse word-count matrix. Unlike the Li extractor this is
*stateful*: the vocabulary is learned at ``fit`` time, so it must be fitted on the
training queries before ``transform``. Output is kept sparse — downstream heads
densify per batch only when needed.
"""

from __future__ import annotations

import hashlib

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


def _as_queries(X) -> list[str]:  # noqa: N803
    """Coerce a DataFrame (``full_query`` column) or iterable of strings to a list."""
    if isinstance(X, pd.DataFrame):
        return X["full_query"].astype(str).tolist()
    return [str(q) for q in X]


class CountVectorizerExtractor(BaseEstimator, TransformerMixin):
    """Sklearn transformer producing a sparse CountVectorizer feature matrix.

    Accepts either a pandas DataFrame with a ``full_query`` column or any iterable
    of strings. ``transform`` returns a ``(n_samples, vocab_size)`` sparse CSR
    matrix of word counts.
    """

    def __init__(self, max_features: int | None = None) -> None:
        self.max_features = max_features

    def fit(self, X, y=None) -> "CountVectorizerExtractor":  # noqa: N803
        self.vectorizer_ = CountVectorizer(max_features=self.max_features)
        self.vectorizer_.fit(_as_queries(X))
        return self

    def transform(self, X) -> csr_matrix:  # noqa: N803
        return self.vectorizer_.transform(_as_queries(X))

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer_.get_feature_names_out()

    def cache_key_state(self) -> str:
        """Fold the fitted vocabulary into the cache key (see CachingExtractor).

        transform output depends on the learned vocabulary, not just the init
        params, so the key must change when the vocabulary does.
        """
        vocab = getattr(self, "vectorizer_", None)
        if vocab is None:
            return ""
        # Stable (not PYTHONHASHSEED-salted) digest so the key matches across runs.
        h = hashlib.blake2b(digest_size=16)
        for term, idx in sorted(vocab.vocabulary_.items()):
            h.update(f"{term}\0{idx}\0".encode())
        return h.hexdigest()
