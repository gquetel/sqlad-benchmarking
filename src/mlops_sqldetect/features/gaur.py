"""GAUR feature extractor: parser-instrumentation-derived features for MySQL queries.

This module talks to :mod:`gaur_sqld` (https://github.com/gquetel/gaur-sql-detect),
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
(:mod:`mlops_sqldetect.features.li`) to create a collector with a full view on
lexical, syntactic and semantic features.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mlops_sqldetect.features.li import FEATURE_NAMES as LI_FEATURE_NAMES
from mlops_sqldetect.features.li import extract_li_features

logger = logging.getLogger(__name__)

GaurMode = str


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

# Trace collection checkpoints every this fraction
_CHECKPOINT_FRACTION = 0.05

# Where checkpoints (derived feature rows, never raw traces) are written.
_TRACE_CHECKPOINT_DIR = Path("data/processed/gaur_trace_checkpoints")


def _trace_checkpoint_dir() -> Path:
    return _TRACE_CHECKPOINT_DIR


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


def _tag_counts(mode: str, nodes: list[TraceNode]) -> dict[str, float]:
    if mode == "expert":
        return _expert_tag_counts(nodes)
    if mode == "ruleid":
        return _ruleid_counts(nodes)
    return _llm_tag_counts(nodes, _LLM_TAGS[mode])


# ----- Trace collection ---------------------------------------------------------
#
# GAUR traces are collected one query at a time via gaur_sqld's own
# get_traces_from_query (rather than its df-level get_traces_from_df) so a single
# MySQL connection is kept open for the whole collection instead of reconnecting
# per checkpoint. Each raw trace is reduced to its feature row immediately and
# never accumulated or written to disk; only the small derived rows are
# checkpointed, every ~5% of input rows (see _collect_and_featurize). Every input
# row always yields exactly one feature row — a per-query collection failure
# produces a row of defaults (via _row_features) rather than being dropped, unlike
# get_traces_from_df, which silently drops failed rows from its output.


def _configure_trace_type(trace_type: str) -> None:
    """Point gaur_sqld's MySQL socket/datadir at the server for ``trace_type``."""
    from gaur_sqld import config as gcfg

    gcfg.update_location_mysqlfiles(trace_type)


def _ensure_server(trace_type: str) -> None:
    """Start (or reuse) the GAUR-instrumented MySQL server for ``trace_type``."""
    from gaur_sqld.server import ensure_server

    ensure_server(trace_type)


def _new_connector_and_collector():
    """Build a fresh, unconnected ``(SQLConnector, GaurTraceCollector)`` pair.

    The connection itself is established lazily, by ``get_traces_from_query`` on
    its first call, exactly as ``get_traces_from_df`` does internally.
    """
    from gaur_sqld import config as gcfg
    from gaur_sqld.utils.mysql_wrapper import SQLConnector
    from gaur_sqld.utils.traces_collector import GaurTraceCollector

    sqlc = SQLConnector(
        user=gcfg.mysql_info.user,
        pwd=gcfg.mysql_info.password,
        socket_path=gcfg.mysql_info.socket_path,
        database=gcfg.mysql_info.database,
    )
    gtc = GaurTraceCollector(fp_datadir=gcfg.mysql_info.datadir_path)
    return sqlc, gtc


def _collect_one_trace(query: str, sqlc, gtc) -> dict[str, object]:
    """Collect the raw trace fields for one query.

    Delegates to ``gaur_sqld.utils.traces_collector.get_traces_from_query``, which
    (re)establishes the MySQL connection as needed, executes the query, and merges
    multiple parser invocations into a single row. A per-query collection failure
    (e.g. a syntax error) comes back as a row of NaNs rather than raising —
    ``_row_features`` already treats that as "no trace collected" — so only a
    connection-level failure propagates out of here.
    """
    from gaur_sqld.utils.traces_collector import get_traces_from_query

    return get_traces_from_query(query, sqlc, gtc).iloc[0].to_dict()


def _checkpoint_path(mode: str, query_df: pd.DataFrame) -> Path:
    """On-disk path for this collection's resume checkpoint.

    Keyed by mode + the exact ordered queries, so an interrupted run only ever
    resumes against a rerun of the identical input.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(mode.encode())
    for q in query_df["full_query"]:
        h.update(b"\0")
        h.update(str(q).encode("utf-8", "surrogatepass"))
    directory = _trace_checkpoint_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{mode}-{h.hexdigest()}.pkl"


def _load_checkpoint(path: Path) -> tuple[list[dict[str, float]], int]:
    """Return ``(feature rows collected so far, how many input rows they cover)``."""
    if not path.exists():
        return [], 0
    state = joblib.load(path)
    return state["rows"], state["n_done"]


def _save_checkpoint(path: Path, rows: list[dict[str, float]], n_done: int) -> None:
    # Atomic write (tmp + rename), pid-suffixed so a concurrent job never sees a
    # half-written checkpoint (mirrors CachingExtractor._save).
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    joblib.dump({"rows": rows, "n_done": n_done}, tmp)
    os.replace(tmp, path)


def _collect_and_featurize(mode: str, query_df: pd.DataFrame) -> list[dict[str, float]]:
    """Collect GAUR traces for ``mode`` and reduce each one to its feature row immediately.

    Progress is checkpointed to disk every ``_CHECKPOINT_FRACTION`` of rows, as the
    feature rows collected so far (never the raw traces), keyed by a hash of
    ``query_df`` so a rerun over the same input resumes from the last checkpoint
    instead of recollecting from scratch. On any failure while collecting a row
    (most commonly a dropped MySQL connection), whatever has been collected so far
    is checkpointed before the exception is re-raised. The checkpoint is deleted
    once collection finishes successfully — nothing is meant to outlive a run.

    Every input row always yields exactly one feature row: a per-query collection
    failure produces a row of defaults (``_row_features`` already treats an all-NaN
    trace as "no trace collected"), so the output length always equals ``len(query_df)``
    — this function never drops an observation the way ``get_traces_from_df`` does.
    """
    trace_type = _SERVER_TRACE_TYPE[mode]
    n = len(query_df)
    ckpt_path = _checkpoint_path(mode, query_df)
    rows, start_idx = _load_checkpoint(ckpt_path)

    if start_idx >= n:
        ckpt_path.unlink(missing_ok=True)
        return rows
    if start_idx:
        logger.info("Resuming GAUR collection for %s from row %d/%d", mode, start_idx, n)

    _configure_trace_type(trace_type)
    _ensure_server(trace_type)
    sqlc, gtc = _new_connector_and_collector()
    checkpoint_interval = max(1, round(n * _CHECKPOINT_FRACTION))

    for i in range(start_idx, n):
        query = str(query_df.iloc[i]["full_query"])
        try:
            trace_row = _collect_one_trace(query, sqlc, gtc)
        except Exception:
            logger.error(
                "GAUR collection failed on row %d/%d for %s; checkpointing %d completed rows",
                i,
                n,
                mode,
                len(rows),
            )
            _save_checkpoint(ckpt_path, rows, i)
            raise
        rows.append(_row_features(mode, trace_row))

        done = i + 1
        if done % checkpoint_interval == 0:
            _save_checkpoint(ckpt_path, rows, done)
            logger.info("GAUR collection checkpoint: %d/%d rows (%s)", done, n, mode)

    if len(rows) != n:
        raise RuntimeError(f"GAUR collection for {mode} produced {len(rows)} feature rows for {n} input rows")
    ckpt_path.unlink(missing_ok=True)
    return rows


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
    GAUR-instrumented MySQL server for ``mode`` (see :func:`_collect_and_featurize`),
    which checkpoints its (feature-row, never raw-trace) progress to disk so an
    interrupted collection resumes rather than restarting from scratch;
    :func:`~mlops_sqldetect.features.build_extractor` additionally wraps the
    extractor in a :class:`~mlops_sqldetect.features.cache.CachingExtractor`,
    caching the (smaller, derived) feature matrix this class returns.
    """

    # GAUR instruments the DBMS parser, so its collector observes every executed query,
    # including insider attacks issued directly against the DBMS. External collectors sit
    # above the DBMS and never see that traffic (they leave this attribute at its False
    # default), which is what the evaluation's insider blind-spot models.
    observes_insider: bool = True

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

        gaur_names = gaur_feature_names(self.mode)
        gaur_rows = _collect_and_featurize(self.mode, query_df)

        li_rows = [extract_li_features(q) for q in queries]
        li_matrix = np.asarray([[r[name] for name in LI_FEATURE_NAMES] for r in li_rows], dtype=np.float32)

        # Every mode (including ruleid's ~1,013 rule-identifier columns) is returned
        # dense: the reference gaur-sql-detect scales the whole feature frame with
        # StandardScaler/MaxAbsScaler, which needs mean-centrable dense input.
        gaur_matrix = np.asarray([[r[name] for name in gaur_names] for r in gaur_rows], dtype=np.float32)
        return np.concatenate([gaur_matrix, li_matrix], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = gaur_feature_names(self.mode) + LI_FEATURE_NAMES
        return np.asarray(names, dtype=object)
