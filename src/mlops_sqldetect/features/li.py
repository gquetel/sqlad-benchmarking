"""Li et al. hand-crafted SQL feature extractor.

Produces a fixed-width numeric feature matrix (boolean flags and character counts)
from a column of raw SQL queries. Pure-function extraction keeps it deterministic
and trivially serialisable: the only state the transformer carries is the ordered
list of output feature names.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_QUERY_KEYWORDS = ("select", "update", "drop", "insert", "create")
_DATABASE_KEYWORDS = ("sysobjects", "msysobjects", "version", "information_schema")
_CONNECTION_KEYWORDS = ("inner join", "and", "or", "xor")
_FILE_KEYWORDS = ("load_file", "infile", "outfile", "dumpfile")
_STRING_FUNCTIONS = ("substr", "substring", "mid", "asc")
_COMPARISON_OPERATORS = ("=", "<", ">", "<=", ">=", "<>", "!=")

_WORD_PATTERNS = {
    "has_exec": re.compile(r"\bexec\b"),
    "has_exists": re.compile(r"\bexists\b"),
    "has_floor": re.compile(r"\bfloor\b"),
    "has_rand": re.compile(r"\brand\b"),
    "has_group": re.compile(r"\bgroup\b"),
    "has_order": re.compile(r"\border\b"),
    "has_length": re.compile(r"\blength\b"),
    "has_ascii": re.compile(r"\bascii\b"),
    "has_concat": re.compile(r"\bconcat\b"),
    "has_if": re.compile(r"\bif\b"),
    "has_count": re.compile(r"\bcount\b"),
    "has_sleep": re.compile(r"\bsleep\b"),
    "has_null": re.compile(r"\bnull\b"),
    "has_union": re.compile(r"\bunion\b"),
}
_QUERY_KW_RE = re.compile(r"\b(" + "|".join(_QUERY_KEYWORDS) + r")\b")
_STRING_FN_RE = re.compile(r"\b(" + "|".join(_STRING_FUNCTIONS) + r")\b")
_TAUTOLOGY_RE = re.compile(r"(\w+)=\1")

FEATURE_NAMES: tuple[str, ...] = (
    "len_query",
    "has_null",
    "has_comment",
    "has_query_keywords",
    "has_union",
    "has_database_keywords",
    "has_connection_keywords",
    "has_file_keywords",
    "has_exec",
    "has_string_functions",
    "c_comparison",
    "has_exists",
    "has_floor",
    "has_rand",
    "has_group",
    "has_order",
    "has_length",
    "has_ascii",
    "has_concat",
    "has_if",
    "has_count",
    "has_sleep",
    "has_tautology",
    "c_num",
    "c_upper",
    "c_space",
    "c_special",
    "c_arith",
    "c_square_brackets",
    "c_round_brackets",
    "has_multiline_comment",
    "c_curly_brackets",
)


def _char_features(query: str) -> dict[str, int]:
    """Single-pass character-class counts and multiline-comment detection."""
    c_num = c_upper = c_space = c_special = 0
    c_arith = c_square = c_round = c_curly = 0
    has_multiline_comment = 0
    in_squot = in_dquot = False
    n = len(query)
    for i, ch in enumerate(query):
        if ch.isdigit():
            c_num += 1
        elif ch.isupper():
            c_upper += 1
        elif ch.isspace():
            c_space += 1
        elif ch == "/":
            if i + 1 < n and query[i + 1] == "*":
                has_multiline_comment = 1
            else:
                c_arith += 1
        elif ch == "*":
            if i + 1 < n and query[i + 1] == "/":
                has_multiline_comment = 1
            else:
                c_arith += 1
        elif ch in ("+", "-"):
            c_arith += 1
        elif ch in ("[", "]"):
            c_square += 1
        elif ch in ("(", ")"):
            c_round += 1
        elif ch in ("{", "}"):
            c_curly += 1
        elif ch == "'":
            if in_dquot:
                pass
            else:
                in_squot = True
            c_special += 1
        elif ch == '"':
            if in_squot:
                pass
            else:
                in_dquot = True
            c_special += 1
        elif not ch.isalnum():
            c_special += 1
    return {
        "c_num": c_num,
        "c_upper": c_upper,
        "c_space": c_space,
        "c_special": c_special,
        "c_arith": c_arith,
        "c_square_brackets": c_square,
        "c_round_brackets": c_round,
        "has_multiline_comment": has_multiline_comment,
        "c_curly_brackets": c_curly,
    }


def extract_li_features(query: str) -> dict[str, int]:
    """Extract the full Li feature vector from a single SQL query.

    Args:
        query: Raw SQL query string.

    Returns:
        Dict mapping each name in :data:`FEATURE_NAMES` to an int value.
    """
    q = query.lower()
    feats: dict[str, int] = {
        "len_query": len(query),
        "has_comment": int("--" in q or "#" in q),
        "has_query_keywords": int(_QUERY_KW_RE.search(q) is not None),
        "has_database_keywords": int(any(k in q for k in _DATABASE_KEYWORDS)),
        "has_connection_keywords": int(any(k in q for k in _CONNECTION_KEYWORDS)),
        "has_file_keywords": int(any(k in q for k in _FILE_KEYWORDS)),
        "has_string_functions": int(_STRING_FN_RE.search(q) is not None),
        "c_comparison": sum(q.count(op) for op in _COMPARISON_OPERATORS),
        "has_tautology": int(_TAUTOLOGY_RE.search(query) is not None),
    }
    for name, pattern in _WORD_PATTERNS.items():
        feats[name] = int(pattern.search(q) is not None)
    feats.update(_char_features(query))
    return feats


class LiExtractor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer producing the Li feature matrix.

    The transformer is stateless: ``fit`` is a no-op. ``transform`` accepts either
    a pandas DataFrame with a ``full_query`` column or any iterable of strings,
    and returns a ``(n_samples, n_features)`` float32 ndarray with columns ordered
    as :data:`FEATURE_NAMES`.
    """

    def fit(self, X, y=None) -> "LiExtractor":  # noqa: N803
        return self

    def transform(self, X) -> np.ndarray:  # noqa: N803
        if isinstance(X, pd.DataFrame):
            queries = X["full_query"].astype(str).tolist()
        else:
            queries = [str(q) for q in X]
        rows = [extract_li_features(q) for q in queries]
        return np.asarray(
            [[r[name] for name in FEATURE_NAMES] for r in rows],
            dtype=np.float32,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(FEATURE_NAMES, dtype=object)
