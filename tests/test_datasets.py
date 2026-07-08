"""Unit tests for the dataset families and the split loader."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from mlops_sqldetect.datasets import FAMILIES
from mlops_sqldetect.datasets.integrity import file_digest, verify_file
from mlops_sqldetect.datasets.superviz25 import Superviz25, load_split
from mlops_sqldetect.datasets.superviz26 import Superviz26
from mlops_sqldetect.datasets.superviz26_drift import DOMAINS as DRIFT_DOMAINS
from mlops_sqldetect.datasets.superviz26_drift import Superviz26Drift, load_drift
from mlops_sqldetect.datasets.superviz26_fsl import DOMAINS as FSL_DOMAINS
from mlops_sqldetect.datasets.superviz26_fsl import Superviz26FSL, load_fsl, lodo_source


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
    assert set(FAMILIES) == {"superviz25", "superviz26", "superviz26-big", "superviz26-drift", "superviz26-fsl"}
    assert FAMILIES["superviz25"].suites == {"all": (Superviz25.MAIN,)}
    assert set(FAMILIES["superviz26"].suites) == {"in_domain", "lodo", "all"}


def test_drift_family_is_a_drift_protocol_with_four_domains():
    family = FAMILIES["superviz26-drift"]
    assert family.protocol == "drift"
    assert family.suites["all"] == DRIFT_DOMAINS
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
    # A custom root is caller-managed: no auto-fetch, just the hint.
    with pytest.raises(FileNotFoundError, match="build_concept_drift"):
        load_drift(Superviz26Drift.A, root=tmp_path)


def test_load_drift_auto_fetches_missing_default_root_file(tmp_path, monkeypatch):
    import tools.fetch_superviz26 as fetch_mod
    from mlops_sqldetect.datasets import superviz26_drift

    monkeypatch.setattr(superviz26_drift, "default_root", lambda: tmp_path)
    # tmp_path stands in as the default root, so the synthetic CSV would trip the
    # manifest integrity check; stub it out to exercise the auto-fetch path alone.
    monkeypatch.setattr(superviz26_drift, "verify_file", lambda *a, **k: None)
    calls = []

    def fake_ensure_group(group, **kwargs):
        calls.append(group)
        _write_drift_csv(tmp_path / "a.csv")

    monkeypatch.setattr(fetch_mod, "ensure_group", fake_ensure_group)

    origin_train, _, _ = load_drift(Superviz26Drift.A)
    assert calls == ["drift"]
    assert len(origin_train) == 2


def test_fsl_family_is_an_fsl_protocol_with_four_targets():
    family = FAMILIES["superviz26-fsl"]
    assert family.protocol == "fsl"
    assert family.suites["all"] == FSL_DOMAINS


def test_fsl_lodo_source_maps_each_target_to_its_held_out_lodo_split():
    assert lodo_source(Superviz26FSL.A) == Superviz26.BCD_A
    assert lodo_source(Superviz26FSL.D) == Superviz26.ABC_D


def test_load_fsl_returns_target_train_and_test_from_the_in_domain_file(tmp_path):
    # The few-shot loader reads the target's in-domain (x-x) CSV: train rows (both
    # labels, for the benign k-draw) and the test split (with attack_technique).
    _write_csv(tmp_path / "a-a.csv")
    train, test = load_fsl(Superviz26FSL.A, root=tmp_path)
    assert len(train) == 2
    assert set(train["label"]) == {0}
    assert len(test) == 2
    assert set(test["label"]) == {0, 1}
    assert "attack_technique" in test.columns


def test_load_fsl_missing_file_hints_at_fetcher(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_superviz26"):
        load_fsl(Superviz26FSL.A, root=tmp_path)


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


def _integrity_entry(path):
    """A manifest-style entry (bytes + sha256) matching ``path``'s current content."""
    return {"bytes": path.stat().st_size, "sha256": file_digest(path, "sha256")}


def test_verify_file_passes_on_matching_content(tmp_path):
    path = tmp_path / "d.csv"
    path.write_bytes(b"col\n1\n")
    verify_file(path, _integrity_entry(path))  # matching sha256 -> no raise


def test_verify_file_raises_on_modified_content(tmp_path):
    path = tmp_path / "d.csv"
    path.write_bytes(b"col\n1\n")
    entry = _integrity_entry(path)
    path.write_bytes(b"col\n1\n2\n")  # same name, tampered content
    with pytest.raises(ValueError, match="differs from the Zenodo"):
        verify_file(path, entry)


def test_verify_file_reports_size_mismatch_before_hashing(tmp_path):
    path = tmp_path / "d.csv"
    path.write_bytes(b"col\n1\n")
    entry = {"bytes": 999, "sha256": hashlib.sha256(b"col\n1\n").hexdigest()}
    with pytest.raises(ValueError, match="bytes"):
        verify_file(path, entry)


def test_verify_file_noop_without_checksum(tmp_path):
    path = tmp_path / "d.csv"
    path.write_bytes(b"col\n1\n")
    verify_file(path, {"kind": "in_domain"})  # no sha256/md5 -> nothing to check
