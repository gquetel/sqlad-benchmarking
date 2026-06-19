"""Unit tests for grid enumeration, scenario resolution, and SLURM cell grouping."""

from __future__ import annotations

import pytest
import typer

from mlops_sqldetect.datasets import FAMILIES
from mlops_sqldetect.evaluate_suite import Cell, _validate_grid, enumerate_cells
from tools.slurm_submit import (
    _bucket,
    _check_venv,
    _eligible_partitions,
    _min_vram,
    _needs_gpu,
    _write_job_script,
)

_JOB_CFG = {
    "cpu": {"partition": "cpu", "cpus_per_task": 4, "mem": "8G", "time": "01:00:00"},
}

# GPU partitions (VRAM in GB) and per-cell minimums used by the bucketing tests.
_GPU_CFG = {
    "gpu": {
        "gres": "gpu:1",
        "cpus_per_task": 32,
        "mem": "64G",
        "time": "08:00:00",
        "partitions": {"V100-16GB": 16, "A30": 24, "V100-32GB": 32, "A100": 40, "A40": 46},
    },
    "min_vram_gb": {"default": 0, "codet5": 24},
}


def _write_cpu_script(tmp_path, *, limit):
    script = tmp_path / "job.sbatch"
    _write_job_script(
        script,
        bucket="cpu",
        res=_JOB_CFG["cpu"],
        cfg=_JOB_CFG,
        dataset="superviz25",
        manifest=tmp_path / "cells.jsonl",
        n=3,
        log_pattern=str(tmp_path / "logs" / "cpu-%A_%a.log"),
        target_fpr=0.001,
        seed=7,
        track=False,
        limit=limit,
    )
    return script.read_text()


def test_enumerate_cells_orders_pipeline_extractor_scenario():
    cells = enumerate_cells("superviz26", "all", "ocsvm,ae", "li")
    # 2 pipelines x 1 extractor x 8 scenarios, pipeline outermost, scenario innermost.
    assert len(cells) == 16
    assert cells[0] == Cell("a-a", "ocsvm", "li")
    assert cells[8] == Cell("a-a", "ae", "li")
    assert [c.pipeline for c in cells] == ["ocsvm"] * 8 + ["ae"] * 8


def test_enumerate_cells_rejects_unknown_names():
    with pytest.raises(typer.BadParameter):
        enumerate_cells("superviz26", "all", "bogus", "li")
    with pytest.raises(typer.BadParameter):
        enumerate_cells("superviz26", "all", "ocsvm", "bogus")


def test_validate_grid_scenario_overrides_suite():
    _, datasets, _, _ = _validate_grid("superviz26", "in_domain", "ocsvm", "li", scenario="bcd-a")
    assert [d.value for d in datasets] == ["bcd-a"]


def test_validate_grid_rejects_unknown_scenario():
    with pytest.raises(typer.BadParameter):
        _validate_grid("superviz26", "all", "ocsvm", "li", scenario="z-z")


def test_needs_gpu_rule():
    assert _needs_gpu(Cell("a-a", "ae", "li")) is True
    assert _needs_gpu(Cell("a-a", "ocsvm", "sbert")) is True
    assert _needs_gpu(Cell("a-a", "ocsvm", "codet5")) is True
    assert _needs_gpu(Cell("a-a", "ocsvm", "li")) is False


def test_min_vram_resolves_extractor_combo_and_default():
    assert _min_vram(Cell("a-a", "ocsvm", "codet5"), _GPU_CFG) == 24  # extractor match
    assert _min_vram(Cell("a-a", "ae", "li"), _GPU_CFG) == 0  # falls back to default
    cfg = {"min_vram_gb": {"sbert": 16, "ae:sbert": 32}}
    assert _min_vram(Cell("a-a", "ae", "sbert"), cfg) == 32  # pipeline:extractor overrides extractor
    assert _min_vram(Cell("a-a", "ocsvm", "sbert"), cfg) == 16


def test_eligible_partitions_filters_by_vram_largest_first():
    # codet5 (24 GB) drops the 16 GB V100; the rest come back largest-VRAM first.
    assert _eligible_partitions(_GPU_CFG, 24) == ["A40", "A100", "V100-32GB", "A30"]
    assert "V100-16GB" not in _eligible_partitions(_GPU_CFG, 24)


def test_eligible_partitions_raises_when_none_fit():
    with pytest.raises(typer.BadParameter, match="VRAM"):
        _eligible_partitions(_GPU_CFG, 64)


def test_bucket_assignment():
    assert _bucket(Cell("a-a", "ocsvm", "li"), _GPU_CFG) == "cpu"
    assert _bucket(Cell("a-a", "ae", "li"), _GPU_CFG) == "gpu"
    assert _bucket(Cell("a-a", "ocsvm", "codet5"), _GPU_CFG) == "gpu-24gb"


def test_family_scenario_count_matches_suite_all():
    assert len(FAMILIES["superviz26"].suites["all"]) == 8


def test_job_script_includes_limit_when_set(tmp_path):
    assert "--limit 50000" in _write_cpu_script(tmp_path, limit=50000)


def test_job_script_omits_limit_when_none(tmp_path):
    assert "--limit" not in _write_cpu_script(tmp_path, limit=None)


def test_check_venv_raises_when_missing(tmp_path):
    with pytest.raises(typer.BadParameter, match="uv sync"):
        _check_venv(tmp_path / ".venv" / "bin" / "activate")


def test_check_venv_passes_when_present(tmp_path):
    activate = tmp_path / "activate"
    activate.touch()
    _check_venv(activate)
