"""Unit tests for the dataset families and the split loader."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_sqldetect.datasets import FAMILIES
from mlops_sqldetect.datasets.superviz25 import Superviz25, load_split


def _write_csv(path):
    rows = [
        ("select 1", 0, "train", ""),
        ("select 2", 0, "train", ""),
        ("select 3", 0, "test", ""),
        ("' or 1=1 -- ", 1, "test", "tautology"),
    ]
    pd.DataFrame(rows, columns=["full_query", "label", "split", "attack_technique"]).to_csv(path, index=False)


def test_family_registry_resolves_both_datasets():
    assert set(FAMILIES) == {"superviz25", "superviz26"}
    assert FAMILIES["superviz25"].suites == {"all": (Superviz25.MAIN,)}
    assert set(FAMILIES["superviz26"].suites) == {"in_domain", "lodo", "all"}


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
