"""Unit tests for the GAUR feature extractor.

Trace collection needs a live GAUR-instrumented MySQL server (via ``gaur_sqld``),
so these tests exercise trace parsing, tag counting, and the extractor's output
shape/order by monkeypatching ``_collect_traces`` and the keyword regex rather
than talking to a real server or requiring ``gaur_sqld`` to be installed.
``test_collect_traces_reindexes_dropped_rows`` covers ``_collect_traces`` itself
by injecting a fake ``gaur_sqld`` module into ``sys.modules``, since that
function's lazy ``from gaur_sqld...`` imports would otherwise need the real
package.
"""

from __future__ import annotations

import re
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from mlops_sqldetect.features import gaur
from mlops_sqldetect.features.gaur import (
    GAUR_SYNT_NAMES,
    GaurExtractor,
    _sparse_block,
    gaur_feature_names,
    parse_semantic_tree,
)
from mlops_sqldetect.features.li import FEATURE_NAMES as LI_FEATURE_NAMES

# Captured before the autouse fixture below replaces gaur._collect_traces, so
# test_collect_traces_reindexes_dropped_rows can still exercise the real thing.
_REAL_COLLECT_TRACES = gaur._collect_traces

# A trace with two nodes: one tagged CREATE/USER with a literal, one untagged.
_SAMPLE_TRACE = "1:2979:100:CREATE:USER:admin|2:15:101::: ||-||edges"


def test_parse_semantic_tree_extracts_nodes():
    nodes = parse_semantic_tree(_SAMPLE_TRACE)
    assert nodes == [
        ("2979", "CREATE", "USER", "admin"),
        ("15", "", "", " "),
    ]


def test_parse_semantic_tree_handles_missing_trace():
    assert parse_semantic_tree(pd.NA) == []
    assert parse_semantic_tree(float("nan")) == []
    assert parse_semantic_tree(None) == []


def test_parse_semantic_tree_skips_malformed_node():
    nodes = parse_semantic_tree("1:2979:100:CREATE|2:15:101::: ||-||")
    assert nodes == [("15", "", "", " ")]


@pytest.mark.parametrize(
    "mode",
    ["expert", "chatgpt", "claude", "llama", "mistral", "gpt-oss", "ruleid"],
)
def test_gaur_feature_names_start_with_syntactic_fields(mode):
    names = gaur_feature_names(mode)
    assert names[: len(GAUR_SYNT_NAMES)] == GAUR_SYNT_NAMES
    assert names[-3:] == ("avg_c_sqlkywds", "max_c_sqlkywds", "min_c_sqlkywds")
    assert len(names) == len(set(names))  # no accidental collisions


def test_ruleid_feature_names_cover_expected_range():
    names = gaur_feature_names("ruleid")
    assert "kind_832" in names
    assert "kind_1844" in names
    assert len(names) == len(GAUR_SYNT_NAMES) + (1844 - 832 + 1) + 3


def test_expert_tag_counts_ignore_none_buckets():
    nodes = parse_semantic_tree(_SAMPLE_TRACE)
    counts = gaur._expert_tag_counts(nodes)
    assert counts["CREATE"] == 1
    assert counts["USER"] == 1
    assert "A_NONE" not in counts
    assert "O_NONE" not in counts


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown GAUR mode"):
        GaurExtractor(mode="not-a-mode")


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"full_query": ["select 1", "create user 'a'@'b' identified by 'x'"]})


def _stub_traces(df: pd.DataFrame, trace_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_terminal": 3,
            "n_nonterminal": 2,
            "is_syntax_error": 0,
            "semantic_tree": _SAMPLE_TRACE,
            "depth": 4,
            "n_parser_invoc": 1,
        },
        index=df.index,
    )


@pytest.fixture(autouse=True)
def _stub_gaur_sqld(monkeypatch):
    """Avoid depending on gaur_sqld/a live server for extractor-level tests."""
    monkeypatch.setattr(gaur, "_collect_traces", _stub_traces)
    monkeypatch.setattr(gaur, "_sql_keyword_re", lambda: re.compile(r"\bselect\b", re.IGNORECASE))


def test_extractor_output_shape_and_names():
    ext = GaurExtractor(mode="expert")
    df = _frame()
    matrix = ext.fit(df).transform(df)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, len(ext.get_feature_names_out()))
    assert matrix.dtype == np.float32
    assert list(ext.get_feature_names_out()) == list(gaur_feature_names("expert")) + list(LI_FEATURE_NAMES)


def test_extractor_preserves_row_count_and_column_count_across_modes():
    df = _frame()
    for mode in ["expert", "chatgpt", "claude", "llama", "mistral", "gpt-oss", "ruleid"]:
        matrix = GaurExtractor(mode=mode).transform(df)
        assert matrix.shape[0] == len(df)
        assert matrix.shape[1] == len(gaur_feature_names(mode)) + len(LI_FEATURE_NAMES)


def test_extractor_accepts_list_of_strings():
    matrix = GaurExtractor(mode="expert").transform(["select 1", "select 2"])
    assert matrix.shape[0] == 2


# ----- ruleid sparsity ---------------------------------------------------------


def test_sparse_block_matches_dense_construction():
    names = ("a", "b", "c")
    rows = [{"a": 0.0, "b": 2.0, "c": 0.0}, {"a": 0.0, "b": 0.0, "c": 5.0}]
    block = _sparse_block(rows, names)
    assert issparse(block)
    expected = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    np.testing.assert_array_equal(block.toarray(), expected)
    assert block.nnz == 2


def test_ruleid_extractor_output_is_sparse_and_matches_dense_counts(monkeypatch):
    # symbkind 2979 (used elsewhere to match the paper's worked example) falls
    # outside ruleid's [832, 1844] range, so use an in-range value here instead.
    in_range_trace = "1:1000:100:CREATE:USER:admin|2:1000:101::: ||-||edges"
    monkeypatch.setattr(
        gaur,
        "_collect_traces",
        lambda df, trace_type: pd.DataFrame(
            {
                "n_terminal": 3,
                "n_nonterminal": 2,
                "is_syntax_error": 0,
                "semantic_tree": in_range_trace,
                "depth": 4,
                "n_parser_invoc": 1,
            },
            index=df.index,
        ),
    )
    ext = GaurExtractor(mode="ruleid")
    df = _frame()
    matrix = ext.fit(df).transform(df)
    assert issparse(matrix)
    dense = matrix.toarray()
    names = ext.get_feature_names_out()
    assert dense.shape == (2, len(names))
    # Both sample rows share the same stubbed trace: symbkind 1000 appears twice.
    kind_col = list(names).index("kind_1000")
    assert (dense[:, kind_col] == 2).all()
    # No other kind_* column should fire.
    other_kind_cols = [i for i, n in enumerate(names) if n.startswith("kind_") and i != kind_col]
    assert dense[:, other_kind_cols].sum() == 0


def test_non_ruleid_extractors_stay_dense():
    matrix = GaurExtractor(mode="expert").transform(_frame())
    assert isinstance(matrix, np.ndarray)
    assert not issparse(matrix)


# ----- _collect_traces: reindexing rows get_traces_from_df drops ---------------


def test_collect_traces_reindexes_dropped_rows(monkeypatch):
    """get_traces_from_df drops failed rows; _collect_traces must reinstate them.

    Simulates gaur_sqld.utils.traces_collector.get_traces_from_df by injecting a
    fake gaur_sqld package into sys.modules (the real one isn't a project
    dependency), returning only the surviving row — indexed as
    get_traces_from_df itself does, on the original df's index, before it drops
    failures — so this exercises _collect_traces's own reindex, not a stub of it.
    """
    fake_gaur_sqld = types.ModuleType("gaur_sqld")
    fake_config = types.ModuleType("gaur_sqld.config")
    fake_config.update_location_mysqlfiles = lambda trace_type: None
    fake_utils = types.ModuleType("gaur_sqld.utils")
    fake_traces_collector = types.ModuleType("gaur_sqld.utils.traces_collector")

    df_in = pd.DataFrame({"full_query": ["select 1", "select 2", "select 3"]})

    def fake_get_traces_from_df(df: pd.DataFrame) -> pd.DataFrame:
        assert list(df.columns) == ["full_query"]
        # Row at original index 1 failed to collect a trace and was dropped;
        # the survivors keep their original index labels.
        surviving = df.index.difference([1])
        return pd.DataFrame(
            {
                "n_terminal": 3,
                "n_nonterminal": 2,
                "is_syntax_error": 0,
                "semantic_tree": _SAMPLE_TRACE,
                "depth": 4,
                "n_parser_invoc": 1,
            },
            index=surviving,
        )

    fake_traces_collector.get_traces_from_df = fake_get_traces_from_df
    fake_gaur_sqld.config = fake_config
    fake_gaur_sqld.utils = fake_utils
    fake_utils.traces_collector = fake_traces_collector

    monkeypatch.setitem(sys.modules, "gaur_sqld", fake_gaur_sqld)
    monkeypatch.setitem(sys.modules, "gaur_sqld.config", fake_config)
    monkeypatch.setitem(sys.modules, "gaur_sqld.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "gaur_sqld.utils.traces_collector", fake_traces_collector)

    result = _REAL_COLLECT_TRACES(df_in, "expert")

    # Row count matches the input despite the collection failure...
    assert list(result.index) == list(df_in.index)
    assert result.loc[0, "semantic_tree"] == _SAMPLE_TRACE
    assert result.loc[2, "semantic_tree"] == _SAMPLE_TRACE
    # ...and the dropped row comes back as NaN, not absent.
    assert pd.isna(result.loc[1, "semantic_tree"])
