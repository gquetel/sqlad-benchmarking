"""Tests for log capture and curve artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sqlad_benchmarking import tracking
from sqlad_benchmarking.tracking import capture_cell_log, cell_log_path
from sqlad_benchmarking.visualize import curve_artifact_dir, plot_curves


@pytest.fixture
def cell_logger():
    """Return a DEBUG logger."""
    log = logging.getLogger("sqlad_benchmarking.test_capture")
    log.setLevel(logging.DEBUG)
    return log


def test_capture_cell_log_writes_the_cell_log_under_the_given_directory(tmp_path, cell_logger):
    with capture_cell_log(tmp_path / "logs", "ocsvm_li_superviz26_a") as cell_log:
        cell_logger.info("training started")
    assert cell_log.path == tmp_path / "logs" / "ocsvm_li_superviz26_a.log"
    assert "training started" in cell_log.path.read_text()


def test_capture_cell_log_keeps_records_the_console_handler_drops(tmp_path, cell_logger):
    with capture_cell_log(tmp_path / "logs", "cell") as cell_log:
        cell_logger.debug("threshold calibrated")
    assert "threshold calibrated" in cell_log.path.read_text()


def test_capture_cell_log_detaches_its_handler_on_the_way_out(tmp_path):
    before = list(logging.getLogger().handlers)
    with capture_cell_log(tmp_path / "logs", "cell"):
        assert len(logging.getLogger().handlers) == len(before) + 1
    assert logging.getLogger().handlers == before


def test_capture_cell_log_detaches_its_handler_after_a_failure(tmp_path):
    before = list(logging.getLogger().handlers)
    with pytest.raises(RuntimeError), capture_cell_log(tmp_path / "logs", "cell"):
        raise RuntimeError("cell died")
    assert logging.getLogger().handlers == before


def test_cell_log_exception_lands_in_the_file_not_on_the_console(tmp_path, capsys):
    with capture_cell_log(tmp_path / "logs", "cell") as cell_log:
        try:
            raise ValueError("no kernel image")
        except ValueError:
            cell_log.exception("Cell cell failed")
    assert "no kernel image" in cell_log.path.read_text()
    assert "no kernel image" not in capsys.readouterr().err


def test_curve_artifact_dir_files_points_apart_from_figures():
    assert curve_artifact_dir(Path("ae_li_superviz26_a_roc.png")) == "roc_curves"
    assert curve_artifact_dir(Path("ae_li_superviz26_a_auprc.png")) == "pr_curves"
    assert curve_artifact_dir(Path("ae_li_superviz26_a_roc.csv")) == "curve_data"
    assert curve_artifact_dir(Path("ae_li_superviz26_a_auprc.csv")) == "curve_data"


def test_plot_curves_keeps_the_points_when_the_browser_cannot_render(tmp_path, monkeypatch, caplog):
    def no_browser(*args, **kwargs):
        raise RuntimeError("The browser seemed to close immediately after starting.")

    monkeypatch.setattr("sqlad_benchmarking.visualize.plot_roc_curve", no_browser)
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    with caplog.at_level(logging.WARNING):
        written = plot_curves(labels, scores, "a", tmp_path, "ocsvm_li_superviz26_a")
    assert [p.name for p in written] == ["ocsvm_li_superviz26_a_roc.csv", "ocsvm_li_superviz26_a_auprc.csv"]
    assert all(p.exists() for p in written)
    assert "keeping the CSV points" in caplog.text


def test_plot_curves_returns_points_and_figures_when_rendering_works(tmp_path, monkeypatch):
    monkeypatch.setattr("sqlad_benchmarking.visualize.plot_roc_curve", lambda *a, **kw: a[3])
    monkeypatch.setattr("sqlad_benchmarking.visualize.plot_pr_curve", lambda *a, **kw: a[3])
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    written = plot_curves(labels, scores, "a", tmp_path, "stem")
    assert [p.name for p in written] == ["stem_roc.csv", "stem_auprc.csv", "stem_roc.png", "stem_auprc.png"]
    assert [curve_artifact_dir(p) for p in written] == ["curve_data", "curve_data", "roc_curves", "pr_curves"]


def test_cell_log_path_names_the_log_of_a_cell():
    expected = Path("reports/superviz26/logs/ocsvm_li_superviz26_a-a.log")
    assert cell_log_path("superviz26", "ocsvm", "li", "a-a") == expected


@pytest.fixture
def killed_cell(tmp_path, monkeypatch):
    """Create a manifest and cell log."""
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "cells_cpu.jsonl"
    manifest.write_text(
        json.dumps({"scenario": "x", "method": "ocsvm", "extractor": "li"})
        + "\n"
        + json.dumps({"scenario": "a-a", "method": "ae", "extractor": "cv"})
        + "\n"
    )
    log = cell_log_path("superviz26", "ae", "cv", "a-a")
    log.parent.mkdir(parents=True)
    log.write_text("CUDA out of memory\n")
    return SimpleNamespace(manifest=manifest, index=1, log=log)


@pytest.fixture
def fake_mlflow(monkeypatch):
    """Mock one RUNNING MLflow run."""
    calls = SimpleNamespace(filter=None, terminated=[], artifacts=[], upload_error=None)

    class FakeClient:
        def log_artifact(self, run_id, local_path, artifact_path):
            if calls.upload_error is not None:
                raise calls.upload_error
            calls.artifacts.append((run_id, local_path, artifact_path))

        def set_terminated(self, run_id, status):
            calls.terminated.append((run_id, status))

    def fake_search_runs(filter_string, **kwargs):
        calls.filter = filter_string
        return [SimpleNamespace(info=SimpleNamespace(run_id="r1"))]

    monkeypatch.setenv("SLURM_JOB_ID", "985129")
    monkeypatch.setattr(tracking, "setup_mlflow", lambda dataset: True)
    monkeypatch.setattr(tracking.mlflow, "search_runs", fake_search_runs)
    monkeypatch.setattr(tracking, "MlflowClient", FakeClient)
    return calls


def test_fail_killed_run_attaches_the_log_of_the_cell_that_died(killed_cell, fake_mlflow):
    tracking.fail_killed_run("superviz26", 137, str(killed_cell.manifest), killed_cell.index)
    assert "attributes.status = 'RUNNING'" in fake_mlflow.filter
    assert "slurm_job_id` = '985129'" in fake_mlflow.filter
    assert fake_mlflow.artifacts == [("r1", str(killed_cell.log), "logs")]
    assert fake_mlflow.terminated == [("r1", "FAILED")]


def test_fail_killed_run_closes_the_run_when_the_cell_wrote_no_log(killed_cell, fake_mlflow):
    killed_cell.log.unlink()
    tracking.fail_killed_run("superviz26", 137, str(killed_cell.manifest), killed_cell.index)
    assert fake_mlflow.artifacts == []
    assert fake_mlflow.terminated == [("r1", "FAILED")]


def test_fail_killed_run_closes_the_run_when_the_upload_fails(killed_cell, fake_mlflow):
    fake_mlflow.upload_error = OSError("no route")
    tracking.fail_killed_run("superviz26", 137, str(killed_cell.manifest), killed_cell.index)
    assert fake_mlflow.terminated == [("r1", "FAILED")]


def test_fail_killed_run_closes_the_run_when_the_manifest_is_invalid(killed_cell, fake_mlflow):
    killed_cell.manifest.write_text("[]\n")
    tracking.fail_killed_run("superviz26", 137, str(killed_cell.manifest), 0)
    assert fake_mlflow.terminated == [("r1", "FAILED")]


def test_fail_killed_run_never_raises_when_the_tracking_server_is_down(monkeypatch):
    monkeypatch.setattr(tracking, "setup_mlflow", lambda dataset: True)
    monkeypatch.setattr(tracking.mlflow, "search_runs", lambda **kwargs: (_ for _ in ()).throw(OSError("no route")))
    tracking.fail_killed_run("superviz26", 137)


def test_fail_killed_run_never_raises_when_mlflow_setup_fails(monkeypatch):
    monkeypatch.setattr(tracking, "setup_mlflow", lambda dataset: (_ for _ in ()).throw(OSError("bad config")))
    tracking.fail_killed_run("superviz26", 137)
