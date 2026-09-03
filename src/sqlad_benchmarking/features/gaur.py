"""GAUR feature extractor: turns MySQL parser traces into fixed-width features.

Uses :mod:`gaur_sqld` to run a GAUR-instrumented MySQL server and collect one
trace per query. 7 modes: ``expert`` (hand-tagged), 5 LLM-tagged sets, and
``ruleid`` (raw grammar-rule ids, no semantics). Every mode is concatenated
with Li et al.'s features (:mod:`sqlad_benchmarking.features.li`).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sqlad_benchmarking.features.li import FEATURE_NAMES as LI_FEATURE_NAMES
from sqlad_benchmarking.features.li import extract_li_features

logger = logging.getLogger(__name__)

GaurMode = str


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

# Grammar-rule symbol kind range in the instrumented parser (832=YYSYMBOL_YYACCEPT,
# 1844=YYSYMBOL_json_attribute in sql_yacc.cc). Fixed at build time.
_RULEID_MIN, _RULEID_MAX = 832, 1844
_RULEID_TAGS: tuple[str, ...] = tuple(f"kind_{i}" for i in range(_RULEID_MIN, _RULEID_MAX + 1))

_MODES: tuple[str, ...] = ("expert", "chatgpt", "claude", "llama", "mistral", "gpt-oss", "ruleid")

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
    """Regex matching any MySQL keyword or builtin function, word-bounded.

    Imports gaur_sqld lazily so this module doesn't require it unless a gaur-* extractor is used.
    """
    from gaur_sqld.utils.constants import mysql_functions, mysql_keywords

    keywords = mysql_functions | mysql_keywords
    return re.compile(r"\b(?:%s)\b" % "|".join(keywords), flags=re.IGNORECASE)


def _count_sql_keywords(text: str) -> int:
    return len(_sql_keyword_re().findall(text))


# ----- Trace parsing -----------------------------------------------------------

TraceNode = tuple[str, str, str, str]  # (symbkind, tag1, tag2, sem_value)


def parse_semantic_tree(trace: str) -> list[TraceNode]:
    """Parse a GAUR ``semantic_tree`` trace into its ``(symbkind, tag1, tag2, value)`` nodes.

    Malformed nodes are skipped and logged. gaur_sqld writes ``"||-||"`` for a query it
    could not trace, which gives an empty list.
    """
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
# gaur_sqld returns one trace row for each query, and marks a query it could not
# trace with n_parser_invoc = 0. The collection runs in parts, and each part
# writes its derived feature rows to disk. Raw traces never go to disk.

# The collection is divided into this many parts, so progress is reported every
# 10% and a failed run continues at the last complete part.
_N_PARTS = 10


def _part_edges(n: int) -> list[int]:
    """Row index of every part boundary, from 0 to ``n``.

    Args:
        n: Number of queries to divide.

    Returns:
        ``n_parts + 1`` indexes. The parts differ by at most one row, and a short
        input gives one part for each row.
    """
    n_parts = max(1, min(_N_PARTS, n))
    return [(k * n) // n_parts for k in range(n_parts + 1)]


def _quiet_collection_logs() -> None:
    """Keep the dependency logs out of the progress lines.

    gaur_sqld writes one INFO line for each part, and mysql.connector writes three
    for each new connection. Warnings and errors still come through.
    """
    for name in ("gaur_sqld", "mysql.connector"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _fmt_duration(seconds: float) -> str:
    """Format a duration as ``1h02m``, ``2m03s`` or ``4.1s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{sec:02d}s"


def _configure_trace_type(trace_type: str) -> None:
    """Point gaur_sqld's MySQL socket/datadir at the server for ``trace_type``."""
    from gaur_sqld import config as gcfg

    gcfg.update_location_mysqlfiles(trace_type)


def _trace_type(mode: str) -> str:
    """Return the server that supplies the traces for ``mode``.

    Args:
        mode: A GAUR mode.

    Returns:
        The trace type of the server. ``ruleid`` reads the expert server, because it
        only keeps the grammar-rule ids, which every server writes.
    """
    return "expert" if mode == "ruleid" else mode


def _get_traces(part: pd.DataFrame) -> pd.DataFrame:
    """Collect the GAUR traces for one part of the input.

    Args:
        part: Queries in a ``full_query`` column.

    Returns:
        One trace row for each query, in the order of ``part``.
    """
    from gaur_sqld.utils.traces_collector import get_traces_from_df

    return get_traces_from_df(part, use_cache=False, disable_tqdm=True)


def _checkpoint_path(mode: str, query_df: pd.DataFrame) -> Path:
    """Checkpoint path, keyed by mode + exact query order so a rerun only resumes against identical input."""
    directory = _trace_checkpoint_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_checkpoint_stem(mode, query_df)}.pkl"


def _checkpoint_stem(mode: str, query_df: pd.DataFrame) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(mode.encode())
    for q in query_df["full_query"]:
        h.update(b"\0")
        h.update(str(q).encode("utf-8", "surrogatepass"))
    return f"{mode}-{h.hexdigest()}"


def _load_checkpoint(path: Path) -> list[dict[str, float]]:
    """Return the feature rows collected so far, or an empty list."""
    if not path.exists():
        return []
    return joblib.load(path)


def _save_checkpoint(path: Path, rows: list[dict[str, float]]) -> None:
    # Atomic write (tmp + rename), pid-suffixed so a concurrent job never sees a
    # half-written checkpoint (mirrors CachingExtractor._save).
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    joblib.dump(rows, tmp)
    os.replace(tmp, path)


def _collect_and_featurize(mode: str, query_df: pd.DataFrame) -> list[dict[str, float]]:
    """Collect the GAUR traces for ``mode`` and make one feature row for each query.

    The collection runs in ``_N_PARTS`` equal parts. Each part writes the feature rows
    to disk and logs the progress, so a failed run continues at the last complete part.
    Each query makes exactly one feature row, so the number of rows on disk is also the
    number of queries that they cover.

    Args:
        mode: A GAUR mode.
        query_df: Queries in a ``full_query`` column.

    Returns:
        One feature row for each query, in the order of ``query_df``.

    Raises:
        RuntimeError: If the rows do not agree with the number of queries.
    """
    n = len(query_df)
    if n == 0:
        return []

    _quiet_collection_logs()
    _configure_trace_type(_trace_type(mode))
    path = _checkpoint_path(mode, query_df)
    rows = _load_checkpoint(path)
    edges = _part_edges(n)
    n_parts = len(edges) - 1
    # A checkpoint only names mode + queries, not the part scheme that produced it, so one
    # left over from a different _N_PARTS -- or a deprecated checkpoint format -- is not
    # usable here. Discard it rather than resuming from a row count we cannot place.
    if len(rows) not in edges:
        if rows:
            logger.warning(f"GAUR {mode}: discarding an incompatible checkpoint ({len(rows)} rows)")
        rows = []
    first = edges.index(len(rows))

    logger.info(f"GAUR {mode}: collecting {n} queries in {n_parts} parts, started {time.strftime('%H:%M:%S')}")
    if first:
        logger.info(f"GAUR {mode}: resuming at row {len(rows)}/{n}, part {first + 1}/{n_parts}")

    started = time.monotonic()
    n_failed = 0
    for k in range(first, n_parts):
        traces = _get_traces(query_df.iloc[edges[k] : edges[k + 1]])
        n_failed += int((traces["n_parser_invoc"] == 0).sum())
        rows.extend(_row_features(mode, trace) for trace in traces.to_dict("records"))
        _save_checkpoint(path, rows)
        logger.info(f"GAUR {mode}: {len(rows)}/{n} rows done at {time.strftime('%H:%M:%S')}")

    if first < n_parts:
        elapsed = _fmt_duration(time.monotonic() - started)
        logger.info(f"GAUR {mode}: done in {elapsed}, {n_failed} queries without a trace")

    # This guards the checkpoint, not the library: a file that holds more rows than
    # the input gives a bad resume point. The library makes its own count agree.
    if len(rows) != n:
        raise RuntimeError(f"GAUR collection for {mode} made {len(rows)} feature rows for {n} queries")
    path.unlink(missing_ok=True)
    return rows


def _row_features(mode: str, trace_row: dict) -> dict[str, float]:
    """Reduce one GAUR trace row to its feature row. A failed trace shows ``n_parser_invoc`` 0."""
    feats = {name: float(trace_row[name]) for name in GAUR_SYNT_NAMES}
    nodes = parse_semantic_tree(trace_row["semantic_tree"])
    feats.update(_tag_counts(mode, nodes))
    avg, mx, mn = _keyword_stats(nodes)
    feats["avg_c_sqlkywds"] = avg
    feats["max_c_sqlkywds"] = mx
    feats["min_c_sqlkywds"] = mn
    return feats


class GaurExtractor(BaseEstimator, TransformerMixin):
    """GAUR features (one semantic model) concatenated with Li et al.'s features.

    Stateless: ``fit`` is a no-op. Collecting the GAUR side needs a live
    GAUR-instrumented MySQL server for ``mode``, and checkpoints progress to disk
    so an interrupted collection resumes instead of restarting.
    """

    # GAUR instruments the DBMS parser, so it sees insider attacks run directly
    # against the DBMS. External collectors sit above it and miss that traffic.
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

        # Dense for every mode: the reference impl's StandardScaler/MaxAbsScaler
        # needs mean-centrable input, even for ruleid's ~1,013 columns.
        gaur_matrix = np.asarray([[r[name] for name in gaur_names] for r in gaur_rows], dtype=np.float32)
        return np.concatenate([gaur_matrix, li_matrix], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = gaur_feature_names(self.mode) + LI_FEATURE_NAMES
        return np.asarray(names, dtype=object)

    def cache_key_state(self) -> str:
        """Fold the gaur_sqld version into the cache key (see CachingExtractor).

        The trace values come from the library, so a bump that changes them must
        also change the key. The version must go up in gaur-sql-detect each time.
        """
        from gaur_sqld import __version__

        return __version__
