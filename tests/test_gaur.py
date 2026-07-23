"""Unit tests for the GAUR feature extractor.

Trace collection needs a live GAUR-instrumented MySQL server (via ``gaur_sqld``),
so these tests exercise trace parsing, tag counting, and the extractor's output
shape/order by monkeypatching ``_collect_one_trace`` (the per-query collection
seam) and the keyword regex rather than talking to a real server or requiring
``gaur_sqld`` to be installed. The checkpoint/resume and never-drop-a-row
behavior of ``_collect_and_featurize`` itself is covered directly further down.
"""

from __future__ import annotations

import re

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


def _stub_trace_row() -> dict:
    return {
        "n_terminal": 3,
        "n_nonterminal": 2,
        "is_syntax_error": 0,
        "semantic_tree": _SAMPLE_TRACE,
        "depth": 4,
        "n_parser_invoc": 1,
    }


def _stub_collect_one(query: str, sqlc, gtc) -> dict:
    return _stub_trace_row()


@pytest.fixture(autouse=True)
def _stub_gaur_sqld(monkeypatch, tmp_path):
    """Avoid depending on gaur_sqld/a live server for extractor-level tests.

    Redirects checkpointing to a per-test tmp_path too, so tests never write into
    the real repo working directory.
    """
    monkeypatch.setattr(gaur, "_configure_trace_type", lambda trace_type: None)
    monkeypatch.setattr(gaur, "_ensure_server", lambda trace_type: None)
    monkeypatch.setattr(gaur, "_new_connector_and_collector", lambda: (None, None))
    monkeypatch.setattr(gaur, "_collect_one_trace", _stub_collect_one)
    monkeypatch.setattr(gaur, "_trace_checkpoint_dir", lambda: tmp_path)
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
        "_collect_one_trace",
        lambda query, sqlc, gtc: {
            "n_terminal": 3,
            "n_nonterminal": 2,
            "is_syntax_error": 0,
            "semantic_tree": in_range_trace,
            "depth": 4,
            "n_parser_invoc": 1,
        },
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


# ----- _collect_and_featurize: never drop a row, checkpoint, resume ------------


def test_collect_and_featurize_never_drops_a_failed_row(monkeypatch):
    """A per-query collection failure must not shrink the output below len(query_df).

    Mirrors what get_traces_from_query itself returns on failure (a row of NaNs)
    for one query; _collect_and_featurize must still emit one feature row per
    input row, not silently drop it the way get_traces_from_df does.
    """

    def fake_collect_one(query: str, sqlc, gtc) -> dict:
        if query == "select 2":
            return {
                "n_terminal": pd.NA,
                "n_nonterminal": pd.NA,
                "is_syntax_error": pd.NA,
                "semantic_tree": pd.NA,
                "depth": pd.NA,
                "n_parser_invoc": pd.NA,
            }
        return _stub_trace_row()

    monkeypatch.setattr(gaur, "_collect_one_trace", fake_collect_one)
    query_df = pd.DataFrame({"full_query": ["select 1", "select 2", "select 3"]})

    rows = gaur._collect_and_featurize("expert", query_df)

    assert len(rows) == len(query_df)
    # The failed row still yields a (degraded) feature row rather than being dropped.
    assert rows[1]["is_syntax_error"] == 1.0
    assert rows[1]["n_terminal"] == 0.0
    assert rows[0]["is_syntax_error"] == 0.0


def test_collect_and_featurize_checkpoints_and_resumes_after_failure(monkeypatch, tmp_path):
    """A mid-collection failure checkpoints progress; a rerun resumes from it,
    only re-collecting the rows not yet covered, and cleans up on success.
    """
    monkeypatch.setattr(gaur, "_trace_checkpoint_dir", lambda: tmp_path)
    query_df = pd.DataFrame({"full_query": [f"select {i}" for i in range(20)]})
    calls: list[str] = []

    def failing_collect_one(query: str, sqlc, gtc) -> dict:
        calls.append(query)
        if len(calls) == 12:
            raise RuntimeError("simulated connection drop")
        return _stub_trace_row()

    monkeypatch.setattr(gaur, "_collect_one_trace", failing_collect_one)
    with pytest.raises(RuntimeError, match="simulated connection drop"):
        gaur._collect_and_featurize("expert", query_df)

    ckpt_path = gaur._checkpoint_path("expert", query_df)
    assert ckpt_path.exists()
    rows_so_far, n_done = gaur._load_checkpoint(ckpt_path)
    assert 0 < n_done < len(query_df)
    assert len(rows_so_far) == n_done

    calls.clear()

    def resumed_collect_one(query: str, sqlc, gtc) -> dict:
        calls.append(query)
        return _stub_trace_row()

    monkeypatch.setattr(gaur, "_collect_one_trace", resumed_collect_one)
    result = gaur._collect_and_featurize("expert", query_df)

    assert len(result) == len(query_df)
    # Only the rows not covered by the checkpoint were re-collected.
    assert len(calls) == len(query_df) - n_done
    # Checkpoint is removed once the whole collection succeeds.
    assert not ckpt_path.exists()
