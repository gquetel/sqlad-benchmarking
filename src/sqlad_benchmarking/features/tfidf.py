"""TF-IDF bag-of-words feature extractor.

Wraps :class:`sklearn.feature_extraction.text.TfidfVectorizer` so a column of raw
SQL queries maps to a sparse TF-IDF matrix. Like the CountVectorizer extractor this
is *stateful*: the vocabulary and IDF weights are learned at ``fit`` time, so it
must be fitted on the training queries before ``transform``. Output is kept sparse —
downstream heads densify per batch only when needed.
"""

from __future__ import annotations

import hashlib

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# Vocabulary cap, keeping the matrix narrow enough for the dense decision heads.
DEFAULT_MAX_FEATURES = 200


def _as_queries(X) -> list[str]:  # noqa: N803
    """Coerce a DataFrame (``full_query`` column) or iterable of strings to a list."""
    if isinstance(X, pd.DataFrame):
        return X["full_query"].astype(str).tolist()
    return [str(q) for q in X]


class TfidfExtractor(BaseEstimator, TransformerMixin):
    """Sklearn transformer producing a sparse TF-IDF feature matrix.

    Accepts either a pandas DataFrame with a ``full_query`` column or any iterable
    of strings. ``transform`` returns a ``(n_samples, vocab_size)`` sparse CSR
    matrix of TF-IDF weights.
    """

    def __init__(self, max_features: int | None = DEFAULT_MAX_FEATURES) -> None:
        self.max_features = max_features

    def fit(self, X, y=None) -> "TfidfExtractor":  # noqa: N803
        self.vectorizer_ = TfidfVectorizer(max_features=self.max_features)
        self.vectorizer_.fit(_as_queries(X))
        return self

    def transform(self, X) -> csr_matrix:  # noqa: N803
        return self.vectorizer_.transform(_as_queries(X))

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer_.get_feature_names_out()

    def cache_key_state(self) -> str:
        """Fold the fitted vocabulary and IDF weights into the cache key (see CachingExtractor).

        transform output depends on what was learned at fit time, not just the init
        params, so the key must change when the vocabulary or IDF weights do.
        """
        vec = getattr(self, "vectorizer_", None)
        if vec is None:
            return ""
        # Stable (not PYTHONHASHSEED-salted) digest so the key matches across runs.
        h = hashlib.blake2b(digest_size=16)
        idf = vec.idf_
        for term, idx in sorted(vec.vocabulary_.items()):
            h.update(f"{term}\0{idx}\0{idf[idx]:.12g}\0".encode())
        return h.hexdigest()
