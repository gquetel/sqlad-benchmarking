"""Unit tests for per-cell log capture and the curve artifacts every protocol writes."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from sqlad_benchmarking.tracking import capture_cell_log
from sqlad_benchmarking.visualize import curve_artifact_dir, plot_curves


@pytest.fixture
def cell_logger():
    """A logger that emits at DEBUG, as the evaluators' loggers do under ``basicConfig``."""
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


def test_cell_log_upload_is_a_no_op_without_an_active_run(tmp_path):
    with capture_cell_log(tmp_path / "logs", "cell") as cell_log:
        cell_log.upload()


def test_curve_artifact_dir_files_points_apart_from_figures():
    from pathlib import Path

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
