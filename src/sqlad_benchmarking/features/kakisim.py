"""Kakisim semantic-tokenization feature extractor (CountVectorizer variant).

Each query is parsed with ``sqlparse`` and every token tagged with a semantic
role (keyword, identifier, function call, comparison operator, ...). Tagged
tokens are then folded into up to three parallel "views" of the query — T
(token values, numbers dropped), C (tag sequence only), E (tag sequence with
numbers/punctuation/parens spelled out) — each vectorised independently with
its own :class:`~sklearn.feature_extraction.text.CountVectorizer` and
concatenated into a single sparse matrix. Like CountVectorizer, this is
*stateful*: the per-view vocabularies are learned at ``fit`` time.
"""

from __future__ import annotations

import hashlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import sqlparse
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sqlparse import tokens as T

# ---- Tokenization & semantic tagging ----

_KEYWORD_OVERRIDES: dict[str, str] = {
    "GRANT": "DCL",
    "REVOKE": "DCL",
    "WITH": "CTE",
    "WHERE": "Where",
    "ASC": "Order",
    "DESC": "Order",
    "DATE": "Builtin",
    "TIME": "Builtin",
    "TIMESTAMP": "Builtin",
    "VARCHAR": "Builtin",
    "CHAR": "Builtin",
    "INT": "Builtin",
    "INTEGER": "Builtin",
    "UNSIGNED": "Builtin",
    "BINARY": "Builtin",
    "SLEEP": "Func",
    "BENCHMARK": "Func",
    "SUBSTRING": "Func",
    "SUBSTR": "Func",
    "MID": "Func",
    "LENGTH": "Func",
    "LEN": "Func",
    "COUNT": "Func",
    "SUM": "Func",
    "AVG": "Func",
    "MAX": "Func",
    "MIN": "Func",
    "UPPER": "Func",
    "LOWER": "Func",
    "CONCAT": "Func",
    "GROUP_CONCAT": "Func",
    "COALESCE": "Func",
    "IFNULL": "Func",
    "IF": "Func",
    "CAST": "Func",
    "CONVERT": "Func",
    "CHAR_LENGTH": "Func",
    "LOAD_FILE": "Func",
    "VERSION": "Func",
    "DATABASE": "Func",
    "USER": "Func",
    "CURRENT_USER": "Func",
    "SCHEMA": "Func",
    "HEX": "Func",
    "UNHEX": "Func",
    "ASCII": "Func",
    "ORD": "Func",
    "FLOOR": "Func",
    "CEILING": "Func",
    "RAND": "Func",
    "EXTRACTVALUE": "Func",
    "UPDATEXML": "Func",
}

_NOISY_FOR_T: set[str] = {"Int"}
_NOISY_FOR_E: set[str] = {"Int", "Punct", "Par"}


def _get_tag(ttype, token_val: str, next_tok=None) -> str:
    if token_val == "(" or token_val == ")":
        return "Par"
    if ttype is T.Keyword.DDL:
        return "DLL"
    if ttype is T.Keyword.DML:
        return "DML"
    if ttype in T.Keyword:
        upper = token_val.upper()
        if upper in _KEYWORD_OVERRIDES:
            return _KEYWORD_OVERRIDES[upper]
        return "Keyw"
    if ttype is T.Number.Integer or ttype is T.Number.Float:
        return "Int"
    if ttype is T.Number.Hexadecimal:
        return "Hexadecimal"
    if ttype is T.Literal.String.Single:
        return "Quot"
    if ttype is T.Punctuation:
        return "Punct"
    if ttype is T.Wildcard:
        return "Wildcard"
    if ttype is T.Comparison:
        return "Comparison"
    if ttype in T.Operator:
        return "Oper"
    if ttype in T.Name:
        if ttype is T.Name.Builtin:
            return "Builtin"
        if next_tok is not None and next_tok.value == "(":
            return "Func"
        upper = token_val.upper()
        if upper in _KEYWORD_OVERRIDES:
            return _KEYWORD_OVERRIDES[upper]
        return "Identifi"
    if ttype in T.Comment:
        return "Escap"
    if ttype is T.Error:
        if token_val == "'":
            return "Escap"
        return "Error"
    return "Unknown"


def _walk_tree(node, result: list[tuple[str, str]]) -> None:
    if isinstance(node, sqlparse.sql.IdentifierList):
        result.append((node.value, "Identifierlist"))
        return
    if node.ttype is not None:
        if node.is_whitespace:
            return
        tag = _get_tag(node.ttype, node.value, next_tok=None)
        result.append((node.value, tag))
    else:
        for child in node.tokens:
            _walk_tree(child, result)


def _tokenize_and_tag(query: str) -> list[tuple[str, str]]:
    statements = sqlparse.parse(query)
    if not statements:
        return []
    result: list[tuple[str, str]] = []
    _walk_tree(statements[0], result)
    fixed: list[tuple[str, str]] = []
    for i, (val, tag) in enumerate(result):
        if tag == "Identifi" and i + 1 < len(result) and result[i + 1][0] == "(":
            fixed.append((val, "Func"))
        else:
            fixed.append((val, tag))
    return fixed


def _sql_to_views(query: str) -> tuple[str, str, str]:
    tagged = _tokenize_and_tag(query)
    t_parts: list[str] = []
    c_parts: list[str] = []
    e_parts: list[str] = []
    for val, tag in tagged:
        c_parts.append(tag)
        if tag not in _NOISY_FOR_T:
            t_parts.append(val)
        e_parts.append(tag)
        if tag not in _NOISY_FOR_E:
            e_parts.append(val)
    return " ".join(t_parts), " ".join(c_parts), " ".join(e_parts)


def _as_queries(X) -> list[str]:  # noqa: N803
    """Coerce a DataFrame (``full_query`` column) or iterable of strings to a list."""
    if isinstance(X, pd.DataFrame):
        return X["full_query"].astype(str).tolist()
    return [str(q) for q in X]


class KakisimExtractor(BaseEstimator, TransformerMixin):
    """Sklearn transformer producing the Kakisim multi-view count matrix.

    ``views`` selects which of T / C / E to include (default: all three).
    Views with more than 1000 queries are tokenised in a process pool since
    ``sqlparse`` parsing is the dominant cost.
    """

    def __init__(self, views: list[str] | None = None, min_df: int = 1) -> None:
        self.views = views
        self.min_df = min_df

    def _views_set(self) -> set[str]:
        return set(self.views) if self.views is not None else {"T", "C", "E"}

    def _to_view_strings(self, queries: list[str]) -> tuple[list[str], list[str], list[str]]:
        if len(queries) > 1000:
            with Pool() as pool:
                results = pool.map(_sql_to_views, queries, chunksize=500)
        else:
            results = [_sql_to_views(q) for q in queries]
        return [r[0] for r in results], [r[1] for r in results], [r[2] for r in results]

    def fit(self, X, y=None) -> "KakisimExtractor":  # noqa: N803
        views = self._views_set()
        t_strs, c_strs, e_strs = self._to_view_strings(_as_queries(X))
        self.vectorizers_: dict[str, CountVectorizer] = {}
        if "T" in views:
            self.vectorizers_["T"] = CountVectorizer(min_df=self.min_df).fit(t_strs)
        if "C" in views:
            self.vectorizers_["C"] = CountVectorizer(min_df=self.min_df).fit(c_strs)
        if "E" in views:
            self.vectorizers_["E"] = CountVectorizer(min_df=self.min_df).fit(e_strs)
        return self

    def transform(self, X) -> csr_matrix:  # noqa: N803
        t_strs, c_strs, e_strs = self._to_view_strings(_as_queries(X))
        view_strs = {"T": t_strs, "C": c_strs, "E": e_strs}
        mats = [vec.transform(view_strs[name]) for name, vec in self.vectorizers_.items()]
        return hstack(mats, format="csr")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.concatenate([vec.get_feature_names_out() for vec in self.vectorizers_.values()])

    def cache_key_state(self) -> str:
        """Fold the fitted per-view vocabularies into the cache key (see CachingExtractor).

        transform output depends on the learned vocabularies, not just init params,
        so the key must change when they do.
        """
        vectorizers = getattr(self, "vectorizers_", None)
        if not vectorizers:
            return ""
        h = hashlib.blake2b(digest_size=16)
        for name, vec in sorted(vectorizers.items()):
            h.update(f"\0view={name}\0".encode())
            for term, idx in sorted(vec.vocabulary_.items()):
                h.update(f"{term}\0{idx}\0".encode())
        return h.hexdigest()
