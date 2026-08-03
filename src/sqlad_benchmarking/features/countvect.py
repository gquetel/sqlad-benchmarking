"""CountVectorizer bag-of-words feature extractors.

Wraps :class:`sklearn.feature_extraction.text.CountVectorizer` so a column of raw
SQL queries maps to a sparse word-count matrix. Unlike the Li extractor this is
*stateful*: the vocabulary is learned at ``fit`` time, so it must be fitted on the
training queries before ``transform``. Output is kept sparse — downstream heads
densify per batch only when needed.

Three variants split the token space: the full bag of words, the SQL keywords and
builtin functions only, and everything except those. Comparing the last two
separates what a detector learns from query *structure* (a vocabulary shared by
every application) from what it learns from the identifiers and literals of the
schema it was trained on.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from sqlad_benchmarking.features.loginov import SQL_KEYWORDS

# Vocabulary cap, applied wherever an unbounded vocabulary would be a problem: the
# AE densifies its input and its width equals the feature count, and the
# non-keyword variant is capped to stay comparable to it.
CV_MAX_FEATURES = 20000

# sklearn's default word tokenizer. SQL terms it can never emit (the single-character
# "!") are dropped below, so they add no dead vocabulary entry and do not trip
# sklearn's stop-word consistency check.
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# Lower-cased SQL keywords and builtin function names, matching the casefolding
# CountVectorizer applies before lookup.
SQL_TERMS: tuple[str, ...] = tuple(sorted({kw.lower() for kw in SQL_KEYWORDS if _TOKEN_RE.fullmatch(kw.lower())}))


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

    def _make_vectorizer(self) -> CountVectorizer:
        """Build the unfitted sklearn vectorizer; subclasses override to restrict the token space."""
        return CountVectorizer(max_features=self.max_features)

    def fit(self, X, y=None) -> "CountVectorizerExtractor":  # noqa: N803
        self.vectorizer_ = self._make_vectorizer()
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


class SqlKeywordCountVectorizerExtractor(CountVectorizerExtractor):
    """CountVectorizer counting only SQL keywords and builtin function names.

    :data:`SQL_TERMS` is used as a fixed vocabulary, so the feature space is
    identical whatever corpus the extractor is fitted on and encodes query
    structure alone. ``fit`` is still required (it validates the vocabulary) but
    learns nothing, and ``max_features`` is inert because the vocabulary is fixed.
    """

    def _make_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(vocabulary=SQL_TERMS)


class NonSqlKeywordCountVectorizerExtractor(CountVectorizerExtractor):
    """CountVectorizer counting everything that is *not* a SQL keyword or builtin function.

    :data:`SQL_TERMS` is passed as stop words, leaving a vocabulary of identifiers,
    literals and other application-specific tokens learned from the training
    queries. Capped at :data:`CV_MAX_FEATURES` terms by default so its width stays
    comparable to the plain extractor's under the AE head.
    """

    def __init__(self, max_features: int | None = CV_MAX_FEATURES) -> None:
        super().__init__(max_features=max_features)

    def _make_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(max_features=self.max_features, stop_words=list(SQL_TERMS))
