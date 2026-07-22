"""GAUR feature extractor: parser-instrumentation-derived features for MySQL queries.

GAUR (https://github.com/gquetel/gaur) instruments a Bison grammar so that the
generated parser emits, for every query, a trace mirroring its parse tree:
lexical tokens, syntactic node identifiers (``symbkind``), and — where a
semantic model has been attached to a grammar rule — a semantic tag. This
module talks to :mod:`gaur_sqld` (https://github.com/gquetel/gaur-sql-detect),
which drives a GAUR-instrumented MySQL server (auto-provisioned via Nix from
gaur-instrumented-apps) to collect one trace per query, then turns each trace
into a fixed-width feature vector.

Seven modes are supported, one per semantic model instantiation evaluated in
the paper: ``expert`` (hand-crafted action/object tags), five LLM-instantiated
tag sets (``chatgpt``, ``claude``, ``llama``, ``mistral``, ``gpt-oss``), and
``ruleid``, a purely syntactic ablation that maps every grammar rule to its own
tag instead of a semantic one. ``ruleid`` carries no semantic tags of its own,
so it reuses the ``expert`` GAUR-instrumented server and only its rule
identifiers (``symbkind``) are read from the trace.

Every mode concatenates its own features with Li et al.'s hand-crafted features
(:mod:`mlops_sqldetect.features.li`) — the "hybrid" combination is the default,
not an opt-in, for every ``gaur-*`` extractor registered in this project.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin

from mlops_sqldetect.features.li import FEATURE_NAMES as LI_FEATURE_NAMES
from mlops_sqldetect.features.li import extract_li_features

logger = logging.getLogger(__name__)

GaurMode = str  # one of _SERVER_TRACE_TYPE's keys; kept as str for sklearn get_params round-tripping.

# ruleid reads plain rule identifiers out of a trace, no semantic tags: it reuses
# the expert-tagged server rather than needing an instrumented server of its own.
_SERVER_TRACE_TYPE: dict[str, str] = {
    "expert": "expert",
    "chatgpt": "chatgpt",
    "claude": "claude",
    "llama": "llama",
    "mistral": "mistral",
    "gpt-oss": "gpt-oss",
    "ruleid": "expert",
}

# Syntactic fields present on every GAUR trace row, regardless of semantic model.
GAUR_SYNT_NAMES: tuple[str, ...] = (
    "n_terminal",
    "n_nonterminal",
    "is_syntax_error",
    "depth",
    "n_parser_invoc",
)

_KEYWORD_STAT_NAMES: tuple[str, ...] = ("avg_c_sqlkywds", "max_c_sqlkywds", "min_c_sqlkywds")

_EXPERT_ACTION_TAGS: tuple[str, ...] = ("CREATE", "DELETE", "MODIFY", "EXECUTE", "READ")
_EXPERT_OBJECT_TAGS: tuple[str, ...] = (
    "TABLESPACE",
    "TABLE",
    "INDEX",
    "VIEW",
    "USER",
    "PROCEDURE",
    "DATABASE",
    "FUNCTION",
    "INSTANCE",
    "LOGFILE",
    "SERVER",
    "TRIGGER",
)
_EXPERT_TAGS: tuple[str, ...] = _EXPERT_ACTION_TAGS + _EXPERT_OBJECT_TAGS

_CHATGPT_TAGS: tuple[str, ...] = (
    "DDL_ALTER",
    "DDL_CREATE",
    "DDL_DROP",
    "DML_DELETE_TRUNCATE",
    "DML_INSERT_REPLACE",
    "DML_MAINTENANCE",
    "DML_SELECT",
    "DML_UPDATE",
    "EXPRESSION_LOGIC",
    "PARTITIONING_STORAGE",
    "PRIVILEGES_SECURITY",
    "PROCEDURAL_LOGIC",
    "REPLICATION_MANAGEMENT",
    "SERVER_ADMIN",
    "SHOW_DESCRIBE_EXPLAIN",
    "STATEMENT_CONTROL",
    "STATEMENT_HELP",
    "STATEMENT_MANAGEMENT",
    "TRANSACTION_CONTROL",
    "WINDOW_ANALYTICS",
)

_CLAUDE_TAGS: tuple[str, ...] = (
    "ADMINISTRATIVE",
    "CLAUSE_COMPONENT",
    "CONSTRAINT_DEFINITION",
    "DATA_IMPORT_EXPORT",
    "DATA_TYPE",
    "DDL_STATEMENT",
    "DML_STATEMENT",
    "ENTRY_POINT",
    "EXPRESSION",
    "FUNCTION_CALL",
    "IDENTIFIER",
    "LITERAL_VALUE",
    "OPTIONAL_MODIFIER",
    "QUERY_STRUCTURE",
    "REPLICATION_CLUSTER",
    "STORED_PROCEDURE",
    "SYNTAX_ELEMENT",
    "TABLE_REFERENCE",
    "TRANSACTION_CONTROL",
    "USER_MANAGEMENT",
)

_LLAMA_TAGS: tuple[str, ...] = (
    "DCL",
    "DDL",
    "DML",
    "Database",
    "Event",
    "Function",
    "Indexing",
    "Locking",
    "Procedure",
    "Query",
    "Role",
    "Security",
    "Server",
    "Table",
    "Tablespace",
    "Transaction",
    "Trigger",
    "User",
    "Utility",
    "View",
)

_GPT_OSS_TAGS: tuple[str, ...] = (
    "Clause-Modifier",
    "Constraint",
    "DDL-Statement",
    "DML-Statement",
    "Data-Type",
    "Event-Scheduling",
    "Expression",
    "Identifier",
    "Index-Definition",
    "Join-Clause",
    "Literal",
    "Options-List",
    "Partition-Clause",
    "Predicate",
    "Privilege-Control",
    "Replication-Control",
    "Stored-Program",
    "Transaction-Control",
    "Utility-Statement",
    "Window-Function",
)

_MISTRAL_TAGS: tuple[str, ...] = (
    "Data Definition",
    "Data Import/Export",
    "Data Manipulation",
    "Data Query",
    "Database Management",
    "Locking & Concurrency",
    "Miscellaneous Operations",
    "Replication & Clustering",
    "Resource Management",
    "Security & Privileges",
    "Statement Control",
    "Stored Procedures & Functions",
    "System Information",
    "System Maintenance",
    "System Variables",
    "Temporary Objects",
    "Transaction Control",
    "Triggers & Events",
    "User Management",
    "Views",
)

_LLM_TAGS: dict[str, tuple[str, ...]] = {
    "chatgpt": _CHATGPT_TAGS,
    "claude": _CLAUDE_TAGS,
    "llama": _LLAMA_TAGS,
    "gpt-oss": _GPT_OSS_TAGS,
    "mistral": _MISTRAL_TAGS,
}

# The range of grammar-rule symbol kinds seen in the instrumented MySQL parser:
# 832 is YYSYMBOL_YYACCEPT, 1844 is YYSYMBOL_json_attribute (both found in the
# generated sql_yacc.cc). Fixed at build time, so this range is stable across runs.
_RULEID_MIN, _RULEID_MAX = 832, 1844
_RULEID_TAGS: tuple[str, ...] = tuple(f"kind_{i}" for i in range(_RULEID_MIN, _RULEID_MAX + 1))

_MODES: tuple[str, ...] = ("expert", "chatgpt", "claude", "llama", "mistral", "gpt-oss", "ruleid")


def _tag_names(mode: str) -> tuple[str, ...]:
    if mode == "expert":
        return _EXPERT_TAGS
    if mode == "ruleid":
        return _RULEID_TAGS
    return _LLM_TAGS[mode]


def gaur_feature_names(mode: str) -> tuple[str, ...]:
    """Fixed-width column names ``transform`` produces for ``mode`` (GAUR side only)."""
    return GAUR_SYNT_NAMES + _tag_names(mode) + _KEYWORD_STAT_NAMES


# ----- MySQL keyword counting -------------------------------------------------


@functools.lru_cache(maxsize=1)
def _sql_keyword_re() -> re.Pattern[str]:
    """Regex matching any MySQL reserved word or built-in function, word-bounded.

    Imports ``gaur_sqld`` lazily (rather than at module load) so importing this
    module — and the extractor registry that pulls it in — does not require
    ``gaur_sqld`` to be installed unless a ``gaur-*`` extractor is actually used.
    """
    from gaur_sqld.utils.constants import mysql_functions, mysql_keywords

    keywords = mysql_functions | mysql_keywords
    return re.compile(r"\b(?:%s)\b" % "|".join(keywords), flags=re.IGNORECASE)


def _count_sql_keywords(text: str) -> int:
    return len(_sql_keyword_re().findall(text))


# ----- Trace parsing -----------------------------------------------------------

TraceNode = tuple[str, str, str, str]  # (symbkind, tag1, tag2, sem_value)


def parse_semantic_tree(trace: str | float | None) -> list[TraceNode]:
    """Parse a GAUR ``semantic_tree`` trace into its ``(symbkind, tag1, tag2, value)`` nodes.

    Each node in the node section of the trace (before the ``||-||`` edge
    separator) is formatted as ``order:symbkind:id:tag1:tag2:sem_value``. Nodes
    that fail to unpack are skipped and logged; a missing/``NaN`` trace (e.g. a
    query GAUR could not collect a trace for) yields an empty list, so downstream
    feature counts are zero rather than raising.
    """
    if pd.isna(trace):
        return []
    nodes: list[TraceNode] = []
    for node in trace.split("||-||")[0].split("|"):
        if not node:
            continue
        try:
            _order, symbkind, _id, tag1, tag2, sem_value = node.split(":")
        except ValueError:
            logger.warning("Failed to parse GAUR trace node: %s", node)
            continue
        nodes.append((symbkind, tag1, tag2, sem_value))
    return nodes


def _keyword_stats(nodes: Iterable[TraceNode]) -> tuple[float, float, float]:
    counts = [_count_sql_keywords(value) for *_rest, value in nodes if value]
    if not counts:
        return 0.0, 0.0, 0.0
    return sum(counts) / len(counts), float(max(counts)), float(min(counts))


def _expert_tag_counts(nodes: Iterable[TraceNode]) -> dict[str, float]:
    counts = dict.fromkeys(_EXPERT_TAGS, 0.0)
    for _symbkind, action, obj, _value in nodes:
        if action in counts:
            counts[action] += 1
        if obj in counts:
            counts[obj] += 1
    return counts


def _llm_tag_counts(nodes: Iterable[TraceNode], tags: tuple[str, ...]) -> dict[str, float]:
    counts = dict.fromkeys(tags, 0.0)
    for _symbkind, tag, _tag2, _value in nodes:
        if tag in counts:
            counts[tag] += 1
    return counts


def _ruleid_counts(nodes: Iterable[TraceNode]) -> dict[str, float]:
    counts = dict.fromkeys(_RULEID_TAGS, 0.0)
    for symbkind, *_rest in nodes:
        try:
            kind = int(symbkind)
        except ValueError:
            continue
        if _RULEID_MIN <= kind <= _RULEID_MAX:
            counts[f"kind_{kind}"] += 1
    return counts


def _sparse_block(rows: list[dict[str, float]], names: tuple[str, ...]) -> csr_matrix:
    """Build a sparse ``(len(rows), len(names))`` matrix from per-row feature dicts.

    Only the nonzero entries of each row are visited, so memory scales with the
    number of nonzero counts rather than ``len(rows) * len(names)`` — the point of
    keeping ``ruleid``'s ~1,013 mostly-zero rule-identifier columns out of a dense
    array in the first place.
    """
    col_of = {name: i for i, name in enumerate(names)}
    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []
    for r, feats in enumerate(rows):
        for name, value in feats.items():
            if value:
                row_idx.append(r)
                col_idx.append(col_of[name])
                data.append(value)
    return csr_matrix((data, (row_idx, col_idx)), shape=(len(rows), len(names)), dtype=np.float32)


def _tag_counts(mode: str, nodes: list[TraceNode]) -> dict[str, float]:
    if mode == "expert":
        return _expert_tag_counts(nodes)
    if mode == "ruleid":
        return _ruleid_counts(nodes)
    return _llm_tag_counts(nodes, _LLM_TAGS[mode])


# ----- Trace collection ---------------------------------------------------------


def _collect_traces(df: pd.DataFrame, trace_type: str) -> pd.DataFrame:
    """Collect one GAUR trace row per query, via ``gaur_sqld``'s own collector.

    Delegates to ``gaur_sqld.utils.traces_collector.get_traces_from_df``, which
    starts (or reuses) the instrumented server for ``trace_type`` — auto-
    provisioned through Nix from gaur-instrumented-apps on first use — and caches
    collected traces to disk (zstd-compressed, keyed by a hash of ``df``, with
    incremental checkpoints), so repeated calls over the same queries, or a run
    interrupted partway through a large collection, don't restart from scratch.
    ``expert`` and ``ruleid`` share this cache automatically: both resolve to the
    ``expert`` trace type, and their cache key only depends on ``df`` and the
    trace type, not the mode requesting it.

    ``get_traces_from_df`` drops rows whose trace collection failed, after first
    setting the result's index to ``df.index`` (so surviving rows keep their
    original position). Reindexing onto ``df.index`` here reinstates the dropped
    rows as all-NaN, which ``_row_features`` already treats as "no trace
    collected" (zero tag counts, ``is_syntax_error`` defaulting to 1), so the
    extractor's row count always matches the input regardless of collection
    failures.
    """
    from gaur_sqld import config as gcfg
    from gaur_sqld.utils.traces_collector import get_traces_from_df

    gcfg.update_location_mysqlfiles(trace_type)
    traces = get_traces_from_df(df[["full_query"]])
    return traces.reindex(df.index)


def _row_features(mode: str, trace_row: dict) -> dict[str, float]:
    feats: dict[str, float] = {
        "n_terminal": float(trace_row["n_terminal"]) if pd.notna(trace_row["n_terminal"]) else 0.0,
        "n_nonterminal": float(trace_row["n_nonterminal"]) if pd.notna(trace_row["n_nonterminal"]) else 0.0,
        # A trace GAUR failed to collect is itself an anomalous signal.
        "is_syntax_error": float(trace_row["is_syntax_error"]) if pd.notna(trace_row["is_syntax_error"]) else 1.0,
        "depth": float(trace_row["depth"]) if pd.notna(trace_row["depth"]) else 0.0,
        "n_parser_invoc": float(trace_row["n_parser_invoc"]) if pd.notna(trace_row["n_parser_invoc"]) else 0.0,
    }
    nodes = parse_semantic_tree(trace_row["semantic_tree"])
    feats.update(_tag_counts(mode, nodes))
    avg, mx, mn = _keyword_stats(nodes)
    feats["avg_c_sqlkywds"] = avg
    feats["max_c_sqlkywds"] = mx
    feats["min_c_sqlkywds"] = mn
    return feats


class GaurExtractor(BaseEstimator, TransformerMixin):
    """GAUR features (one semantic model) concatenated with Li et al.'s features.

    Stateless like :class:`~mlops_sqldetect.features.li.LiExtractor`: ``fit`` is a
    no-op, and every mode produces a fixed-width vector so ``transform`` never
    depends on what it has seen before. Collecting the GAUR side requires a live
    GAUR-instrumented MySQL server for ``mode`` (see :func:`_collect_traces`),
    though its own on-disk trace cache already avoids re-collecting an input
    already seen; :func:`~mlops_sqldetect.features.build_extractor` additionally
    wraps the extractor in a
    :class:`~mlops_sqldetect.features.cache.CachingExtractor`, caching the
    (smaller, derived) feature matrix this class returns.
    """

    def __init__(self, mode: GaurMode = "expert") -> None:
        if mode not in _MODES:
            raise ValueError(f"Unknown GAUR mode: {mode!r} (expected one of {_MODES})")
        self.mode = mode

    def fit(self, X, y=None) -> "GaurExtractor":  # noqa: N803
        return self

    def transform(self, X) -> np.ndarray:  # noqa: N803
        if isinstance(X, pd.DataFrame):
            queries = X["full_query"].astype(str).tolist()
        else:
            queries = [str(q) for q in X]
        query_df = pd.DataFrame({"full_query": queries})

        traces = _collect_traces(query_df, _SERVER_TRACE_TYPE[self.mode])
        gaur_names = gaur_feature_names(self.mode)
        gaur_rows = [_row_features(self.mode, row) for row in traces.to_dict("records")]

        li_rows = [extract_li_features(q) for q in queries]
        li_matrix = np.asarray([[r[name] for name in LI_FEATURE_NAMES] for r in li_rows], dtype=np.float32)

        if self.mode == "ruleid":
            # ruleid's one-hot rule-identifier counts are ~1,013 columns and almost
            # entirely zero per query: build them sparse instead of densifying the
            # full matrix, mirroring CountVectorizerExtractor's sparse output — the
            # other wide, high-cardinality extractor in this project.
            gaur_block = _sparse_block(gaur_rows, gaur_names)
            return hstack([gaur_block, csr_matrix(li_matrix)], format="csr")

        gaur_matrix = np.asarray([[r[name] for name in gaur_names] for r in gaur_rows], dtype=np.float32)
        return np.concatenate([gaur_matrix, li_matrix], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = gaur_feature_names(self.mode) + LI_FEATURE_NAMES
        return np.asarray(names, dtype=object)
