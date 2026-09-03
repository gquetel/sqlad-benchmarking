"""Unit tests for the GAUR feature extractor.

Stubs out ``_get_traces`` and the keyword regex so tests don't need a live
GAUR-instrumented MySQL server or ``gaur_sqld`` installed.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from sqlad_benchmarking.features import gaur
from sqlad_benchmarking.features.gaur import (
    GAUR_SYNT_NAMES,
    GaurExtractor,
    gaur_feature_names,
    parse_semantic_tree,
)
from sqlad_benchmarking.features.li import FEATURE_NAMES as LI_FEATURE_NAMES

# A trace with two nodes: one tagged CREATE/USER with a literal, one untagged.
_SAMPLE_TRACE = "1:2979:100:CREATE:USER:admin|2:15:101::: ||-||edges"

# What gaur_sqld writes for a query it could not trace.
_FAILED_TRACE = "||-||"


def test_parse_semantic_tree_extracts_nodes():
    nodes = parse_semantic_tree(_SAMPLE_TRACE)
    assert nodes == [
        ("2979", "CREATE", "USER", "admin"),
        ("15", "", "", " "),
    ]


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


def test_ruleid_reads_the_expert_server():
    assert gaur._trace_type("ruleid") == "expert"
    assert gaur._trace_type("mistral") == "mistral"


def test_part_edges_makes_ten_equal_parts():
    edges = gaur._part_edges(3_353_671)
    sizes = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    assert len(sizes) == gaur._N_PARTS
    assert sum(sizes) == 3_353_671
    assert max(sizes) - min(sizes) <= 1


def test_part_edges_never_makes_an_empty_part():
    assert gaur._part_edges(7) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert gaur._part_edges(1) == [0, 1]


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


def _stub_traces(part: pd.DataFrame) -> pd.DataFrame:
    """Stand in for gaur_sqld: one trace row for each query, in input order."""
    return pd.DataFrame([_stub_trace_row() for _ in range(len(part))], index=part.index)


@pytest.fixture(autouse=True)
def _stub_gaur_sqld(monkeypatch, tmp_path):
    """Avoid depending on gaur_sqld/a live server; checkpoints go to tmp_path instead of the repo."""
    monkeypatch.setattr(gaur, "_configure_trace_type", lambda trace_type: None)
    monkeypatch.setattr(gaur, "_get_traces", _stub_traces)
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


# ----- ruleid output -----------------------------------------------------------


def test_ruleid_extractor_output_is_dense_and_matches_counts(monkeypatch):
    # symbkind 2979 (used elsewhere to match the paper's worked example) falls
    # outside ruleid's [832, 1844] range, so use an in-range value here instead.
    in_range_trace = "1:1000:100:CREATE:USER:admin|2:1000:101::: ||-||edges"

    def in_range_traces(part: pd.DataFrame) -> pd.DataFrame:
        row = _stub_trace_row() | {"semantic_tree": in_range_trace}
        return pd.DataFrame([row for _ in range(len(part))], index=part.index)

    monkeypatch.setattr(gaur, "_get_traces", in_range_traces)
    ext = GaurExtractor(mode="ruleid")
    df = _frame()
    dense = ext.fit(df).transform(df)
    # ruleid is dense like every other mode so the reference's StandardScaler /
    # MaxAbsScaler (mean-centrable) can be applied downstream.
    assert isinstance(dense, np.ndarray)
    assert not issparse(dense)
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


# ----- _collect_and_featurize: keep every row, checkpoint, resume ---------------


def test_collect_and_featurize_keeps_a_failed_row(monkeypatch):
    """A query gaur_sqld could not trace keeps its row, with n_parser_invoc 0 and no tags."""

    def traces_with_one_failure(part: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for query in part["full_query"]:
            if query == "select 2":
                rows.append(
                    {
                        "n_terminal": 0,
                        "n_nonterminal": 0,
                        "is_syntax_error": 0,
                        "semantic_tree": _FAILED_TRACE,
                        "depth": 0,
                        "n_parser_invoc": 0,
                    }
                )
            else:
                rows.append(_stub_trace_row())
        return pd.DataFrame(rows, index=part.index)

    monkeypatch.setattr(gaur, "_get_traces", traces_with_one_failure)
    query_df = pd.DataFrame({"full_query": ["select 1", "select 2", "select 3"]})

    rows = gaur._collect_and_featurize("expert", query_df)

    assert len(rows) == len(query_df)
    assert rows[1]["n_parser_invoc"] == 0.0
    assert rows[1]["CREATE"] == 0.0
    assert rows[1]["avg_c_sqlkywds"] == 0.0
    assert rows[0]["n_parser_invoc"] == 1.0
    assert rows[0]["CREATE"] == 1.0


def test_collect_and_featurize_checkpoints_and_resumes_after_failure(monkeypatch, tmp_path):
    """A failure in one part checkpoints the complete parts; a rerun resumes and cleans up."""
    monkeypatch.setattr(gaur, "_trace_checkpoint_dir", lambda: tmp_path)
    query_df = pd.DataFrame({"full_query": [f"select {i}" for i in range(20)]})  # 10 parts of 2 rows
    parts: list[int] = []

    def failing_get_traces(part: pd.DataFrame) -> pd.DataFrame:
        parts.append(len(part))
        if len(parts) == 3:
            raise RuntimeError("simulated connection drop")
        return _stub_traces(part)

    monkeypatch.setattr(gaur, "_get_traces", failing_get_traces)
    with pytest.raises(RuntimeError, match="simulated connection drop"):
        gaur._collect_and_featurize("expert", query_df)

    ckpt_path = gaur._checkpoint_path("expert", query_df)
    assert ckpt_path.exists()
    # The two complete parts are on disk; the failed part is not.
    assert len(gaur._load_checkpoint(ckpt_path)) == 4

    parts.clear()

    def resumed_get_traces(part: pd.DataFrame) -> pd.DataFrame:
        parts.append(len(part))
        return _stub_traces(part)

    monkeypatch.setattr(gaur, "_get_traces", resumed_get_traces)
    result = gaur._collect_and_featurize("expert", query_df)

    assert len(result) == len(query_df)
    # Only the rows not covered by the checkpoint were collected again.
    assert sum(parts) == 16
    assert len(parts) == 8
    # Checkpoint is removed once the whole collection succeeds.
    assert not ckpt_path.exists()


def test_collect_and_featurize_discards_a_checkpoint_off_the_part_boundaries(monkeypatch, tmp_path, caplog):
    """A checkpoint whose row count is not a part boundary (stale format, different _N_PARTS) is
    discarded, not resumed from -- the whole input is collected again instead of crashing."""
    monkeypatch.setattr(gaur, "_trace_checkpoint_dir", lambda: tmp_path)
    query_df = pd.DataFrame({"full_query": [f"select {i}" for i in range(20)]})  # 10 parts of 2 rows

    ckpt_path = gaur._checkpoint_path("expert", query_df)
    gaur._save_checkpoint(ckpt_path, [{"n_terminal": 0.0}] * 3)  # 3 is not a part boundary

    parts: list[int] = []

    def get_traces(part: pd.DataFrame) -> pd.DataFrame:
        parts.append(len(part))
        return _stub_traces(part)

    monkeypatch.setattr(gaur, "_get_traces", get_traces)
    with caplog.at_level("WARNING"):
        result = gaur._collect_and_featurize("expert", query_df)

    assert len(result) == len(query_df)
    assert sum(parts) == len(query_df)  # everything collected again, not just 17 rows
    assert "discarding an incompatible checkpoint" in caplog.text
