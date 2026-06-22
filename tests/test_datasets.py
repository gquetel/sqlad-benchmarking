"""Unit tests for the dataset families and the split loader."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_sqldetect.datasets import FAMILIES
from mlops_sqldetect.datasets.superviz25 import Superviz25, load_split
from mlops_sqldetect.datasets.superviz26_drift import Superviz26Drift, load_drift


def _write_csv(path):
    rows = [
        ("select 1", 0, "train", ""),
        ("select 2", 0, "train", ""),
        ("select 3", 0, "test", ""),
        ("' or 1=1 -- ", 1, "test", "tautology"),
    ]
    pd.DataFrame(rows, columns=["full_query", "label", "split", "attack_technique"]).to_csv(path, index=False)


def _write_drift_csv(path):
    # (full_query, label, split, drift_set, attack_technique): two benign origin-train
    # rows, one origin-test pair, and one shifted-test pair (the held-out S2 split).
    rows = [
        ("select 1", 0, "train", "origin", ""),
        ("select 2", 0, "train", "origin", ""),
        ("select 3", 0, "test", "origin", ""),
        ("' or 1=1 -- ", 1, "test", "origin", "tautology"),
        ("select 4", 0, "test", "shifted", ""),
        ("' union select 1 -- ", 1, "test", "shifted", "union"),
    ]
    cols = ["full_query", "label", "split", "drift_set", "attack_technique"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_family_registry_resolves_all_datasets():
    assert set(FAMILIES) == {"superviz25", "superviz26", "superviz26-big", "superviz26-drift"}
    assert FAMILIES["superviz25"].suites == {"all": (Superviz25.MAIN,)}
    assert set(FAMILIES["superviz26"].suites) == {"in_domain", "lodo", "all"}


def test_drift_family_is_a_drift_protocol_with_four_domains():
    family = FAMILIES["superviz26-drift"]
    assert family.protocol == "drift"
    assert family.suites["all"] == Superviz26Drift.DOMAINS
    assert FAMILIES["superviz26"].protocol == "suite"


def test_load_drift_partitions_origin_train_origin_test_shifted_test(tmp_path):
    _write_drift_csv(tmp_path / "a.csv")
    origin_train, origin_test, shifted_test = load_drift(Superviz26Drift.A, root=tmp_path)
    assert len(origin_train) == 2
    assert set(origin_train["label"]) == {0}
    assert len(origin_test) == 2
    assert set(origin_test["label"]) == {0, 1}
    # S2 is the held-out shifted test set only — no train rows leak into it.
    assert len(shifted_test) == 2
    assert set(shifted_test["label"]) == {0, 1}
    assert "select 4" in set(shifted_test["full_query"])


def test_load_drift_missing_file_hints_at_builder(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_concept_drift"):
        load_drift(Superviz26Drift.A, root=tmp_path)


def test_superviz25_load_split_filters_by_split(tmp_path):
    _write_csv(tmp_path / "dataset.csv")
    train = load_split(Superviz25.MAIN, "train", root=tmp_path)
    test = load_split(Superviz25.MAIN, "test", root=tmp_path)
    assert len(train) == 2
    assert set(train["label"]) == {0}
    assert len(test) == 2
    assert set(test["label"]) == {0, 1}


def test_superviz25_load_split_keeps_requested_columns(tmp_path):
    _write_csv(tmp_path / "dataset.csv")
    test = load_split(Superviz25.MAIN, "test", root=tmp_path, columns=("full_query", "label", "attack_technique"))
    assert {"full_query", "label", "attack_technique", "split"} <= set(test.columns)


def test_superviz25_missing_file_hints_at_fetcher(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_superviz25"):
        load_split(Superviz25.MAIN, "train", root=tmp_path)
