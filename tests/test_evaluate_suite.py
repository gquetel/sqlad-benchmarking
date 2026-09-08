"""Unit tests for grid enumeration, scenario resolution, and SLURM cell grouping."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
import yaml

from sqlad_benchmarking.datasets import FAMILIES
from sqlad_benchmarking.evaluate_drift import evaluate_drift
from sqlad_benchmarking.evaluate_suite import Cell, _validate_grid, enumerate_cells, evaluate_suite
from sqlad_benchmarking.features import GPU_EXTRACTORS
from tools.slurm_submit import (
    _bucket,
    _build_units,
    _eligible_partitions,
    _ensure_env,
    _gpu_section,
    _is_long_running,
    _min_vram,
    _needs_gpu,
    _target_arches,
    _tick,
    _venv_faults,
    _write_job_script,
    env_setup,
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
        "partitions": {"A100": 40, "A40": 46, "A30": 24, "V100-32GB": 32, "V100-16GB": 16},
    },
    "gpu-long": {
        "gres": "gpu:1",
        "cpus_per_task": 32,
        "mem": "64G",
        "time": "24:00:00",
        "partitions": {"A100": 40, "A40": 46, "A30": 24, "V100-32GB": 32, "V100-16GB": 16},
    },
    "min_vram_gb": {"default": 0, "codet5": 24},
    "long_running": ["ocsvm:sbert"],
}


def _write_cpu_script(tmp_path, *, limit):
    script = tmp_path / "job.sbatch"
    _write_job_script(
        script,
        job_name="id-li-ocsvm-superviz25",
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


def test_enumerate_cells_orders_method_extractor_scenario():
    cells = enumerate_cells("superviz26", "all", "ocsvm,ae", "li")
    # 2 methods x 1 extractor x 8 scenarios, method outermost, scenario innermost.
    assert len(cells) == 16
    assert cells[0] == Cell("a-a", "ocsvm", "li")
    assert cells[8] == Cell("a-a", "ae", "li")
    assert [c.method for c in cells] == ["ocsvm"] * 8 + ["ae"] * 8


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
    assert all(_needs_gpu(Cell("a-a", "ocsvm", extractor)) for extractor in GPU_EXTRACTORS)
    assert _needs_gpu(Cell("a-a", "ocsvm", "li")) is False


def test_min_vram_resolves_extractor_combo_and_default():
    assert _min_vram(Cell("a-a", "ocsvm", "codet5"), _GPU_CFG) == 24  # extractor match
    assert _min_vram(Cell("a-a", "ae", "li"), _GPU_CFG) == 0  # falls back to default
    cfg = {"min_vram_gb": {"sbert": 16, "ae:sbert": 32}}
    assert _min_vram(Cell("a-a", "ae", "sbert"), cfg) == 32  # engine:extractor overrides extractor
    assert _min_vram(Cell("a-a", "ocsvm", "sbert"), cfg) == 16


def test_eligible_partitions_filters_by_vram_preference_order():
    # codet5 (24 GB) drops the 16 GB V100; the rest come back in config (preference) order.
    assert _eligible_partitions(_GPU_CFG["gpu"], 24) == ["A100", "A40", "A30", "V100-32GB"]
    assert "V100-16GB" not in _eligible_partitions(_GPU_CFG["gpu"], 24)


def test_eligible_partitions_raises_when_none_fit():
    with pytest.raises(typer.BadParameter, match="VRAM"):
        _eligible_partitions(_GPU_CFG["gpu"], 64)


def test_bucket_assignment():
    assert _bucket(Cell("a-a", "ocsvm", "li"), _GPU_CFG, "gpu") == "cpu"
    assert _bucket(Cell("a-a", "ae", "li"), _GPU_CFG, "gpu") == "gpu"
    assert _bucket(Cell("a-a", "ocsvm", "codet5"), _GPU_CFG, "gpu") == "gpu-24gb"


def test_is_long_running_matches_combo_or_extractor():
    cfg = {"long_running": ["ocsvm:sbert", "codet5"]}
    assert _is_long_running(Cell("a-a", "ocsvm", "sbert"), cfg)  # engine:extractor match
    assert _is_long_running(Cell("a-a", "ae", "codet5"), cfg)  # bare extractor match
    assert not _is_long_running(Cell("a-a", "ae", "sbert"), cfg)  # sbert only long-running under ocsvm
    assert not _is_long_running(Cell("a-a", "ocsvm", "li"), {})  # empty/absent list


def test_long_running_cells_route_to_gpu_long_bucket():
    # ocsvm:sbert is flagged long-running, so it splits into its own gpu-long array...
    assert _bucket(Cell("a-a", "ocsvm", "sbert"), _GPU_CFG, "gpu") == "gpu-long"
    assert _gpu_section(Cell("a-a", "ocsvm", "sbert"), _GPU_CFG, "gpu") == "gpu-long"
    # ...while ae:sbert (same extractor, not flagged) stays on the default 12h gpu block.
    assert _bucket(Cell("a-a", "ae", "sbert"), _GPU_CFG, "gpu") == "gpu"
    assert _gpu_section(Cell("a-a", "ae", "sbert"), _GPU_CFG, "gpu") == "gpu"


def test_gpu_section_default_applies_to_non_long_running_cells():
    # A submission-wide --gpu-section is the default for cells not flagged long-running.
    assert _gpu_section(Cell("a-a", "ae", "li"), _GPU_CFG, "gpu-long") == "gpu-long"
    assert _bucket(Cell("a-a", "ocsvm", "codet5"), _GPU_CFG, "gpu-long") == "gpu-long-24gb"


def test_family_scenario_count_matches_suite_all():
    assert len(FAMILIES["superviz26"].suites["all"]) == 8


def test_enumerate_cells_works_for_drift_family():
    # The drift family reuses the generic Cell enumeration: 4 domains x 1 method x 1 extractor.
    cells = enumerate_cells("superviz26-drift", "all", "ocsvm", "li")
    assert len(cells) == 4
    assert [c.scenario for c in cells] == ["a", "b", "c", "d"]
    assert all(c.method == "ocsvm" and c.extractor == "li" for c in cells)


def test_drift_cells_bucket_like_the_suite():
    # GPU bucketing keys off method/extractor, so drift cells route identically.
    assert _bucket(Cell("a", "ocsvm", "li"), _GPU_CFG, "gpu") == "cpu"
    assert _bucket(Cell("a", "ae", "li"), _GPU_CFG, "gpu") == "gpu"
    assert _bucket(Cell("a", "ocsvm", "codet5"), _GPU_CFG, "gpu") == "gpu-24gb"


def test_evaluate_suite_rejects_drift_family():
    with pytest.raises(typer.BadParameter, match="evaluate_drift"):
        evaluate_suite(dataset="superviz26-drift", suite="all", methods="ocsvm", extractors="li")


def test_evaluate_drift_rejects_non_drift_family():
    with pytest.raises(typer.BadParameter, match="drift family"):
        evaluate_drift(dataset="superviz26", suite="all", methods="ocsvm", extractors="li")


def test_job_script_includes_limit_when_set(tmp_path):
    assert "--limit 50000" in _write_cpu_script(tmp_path, limit=50000)


def test_job_script_omits_limit_when_none(tmp_path):
    assert "--limit" not in _write_cpu_script(tmp_path, limit=None)


def test_venv_faults_reports_a_missing_venv(tmp_path):
    assert _venv_faults(tmp_path / ".venv-cluster", {}) == ["missing"]


def test_venv_faults_reports_the_architectures_the_partitions_need(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "activate").touch()
    monkeypatch.setattr("tools.slurm_submit._lock_matches", lambda venv: True)
    monkeypatch.setattr("tools.slurm_submit._torch_arch_flags", lambda venv: {"sm_75", "sm_80", "sm_86"})
    assert _venv_faults(tmp_path, {"A100": "sm_80", "V100-16GB": "sm_70"}) == ["torch has no sm_70 kernels"]


def test_venv_faults_is_empty_when_every_partition_is_covered(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "activate").touch()
    monkeypatch.setattr("tools.slurm_submit._lock_matches", lambda venv: True)
    monkeypatch.setattr("tools.slurm_submit._torch_arch_flags", lambda venv: {"sm_70", "sm_80"})
    assert _venv_faults(tmp_path, {"A100": "sm_80", "V100-16GB": "sm_70"}) == []


def test_venv_faults_does_not_probe_torch_for_a_cpu_only_grid(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "activate").touch()
    monkeypatch.setattr("tools.slurm_submit._lock_matches", lambda venv: True)
    monkeypatch.setattr("tools.slurm_submit._torch_arch_flags", lambda venv: set())
    assert _venv_faults(tmp_path, {}) == []


def test_ensure_env_syncs_a_faulty_venv_and_submits(monkeypatch):
    synced = []
    faults = iter([["torch has no sm_70 kernels"], []])
    monkeypatch.setattr("tools.slurm_submit._venv_faults", lambda venv, targets: next(faults))
    monkeypatch.setattr("tools.slurm_submit._sync_venv", lambda venv: synced.append(venv))
    _ensure_env(_GPU_CFG, [Cell("a", "ae", "li")], "gpu")
    assert [v.name for v in synced] == [".venv-cluster"]


def test_ensure_env_leaves_a_good_venv_alone(monkeypatch):
    monkeypatch.setattr("tools.slurm_submit._venv_faults", lambda venv, targets: [])
    monkeypatch.setattr("tools.slurm_submit._sync_venv", lambda venv: pytest.fail("must not sync"))
    _ensure_env(_GPU_CFG, [Cell("a", "ae", "li")], "gpu")


def test_ensure_env_gives_up_when_a_sync_does_not_help(monkeypatch):
    monkeypatch.setattr("tools.slurm_submit._venv_faults", lambda venv, targets: ["torch has no sm_70 kernels"])
    monkeypatch.setattr("tools.slurm_submit._sync_venv", lambda venv: None)
    with pytest.raises(typer.BadParameter, match="still wrong after a sync"):
        _ensure_env(_GPU_CFG, [Cell("a", "ae", "li")], "gpu")


def test_env_setup_activates_the_configured_venv():
    assert env_setup({}) == "source .venv-cluster/bin/activate"
    assert env_setup({"env": {"venv": ".venv-other"}}) == "source .venv-other/bin/activate"


def test_env_setup_purges_then_loads_the_site_modules_in_order():
    lines = env_setup({"env": {"modules": ["gcc/14.3.0", "cuda/12.9"]}}).splitlines()
    assert lines == [
        "module purge",
        "module load gcc/14.3.0",
        "module load cuda/12.9",
        "source .venv-cluster/bin/activate",
    ]


def test_tick_submits_each_cell_once_even_when_the_job_failed(monkeypatch):
    """A cell goes out on the first tick only: a failure must not put it back in the queue.

    The queue stays empty and nothing ever finishes, which is exactly what a crashing cell
    looks like. The second tick must find nothing left to do.
    """
    submitted = []
    monkeypatch.setattr("tools.slurm_submit._squeue", lambda *a, **kw: [])
    monkeypatch.setattr("tools.slurm_submit._submit_cells", lambda cells, **kw: submitted.append(list(cells)))

    units = _build_units("superviz26", "in_domain", "ocsvm", "li")
    done: set[Cell] = set()
    tick = lambda: _tick(units, done=done, max_jobs=24, user="tester", count_array_tasks=True, dry_run=True)  # noqa: E731

    assert tick() == 0  # the unit goes out, and nothing is left over to wait for
    assert tick() == 0  # a second pass resubmits nothing: the failed cells are not retried
    assert len(submitted) == 1
    assert submitted[0] == list(units[0].cells)


def test_tick_drops_a_unit_too_big_for_the_cap(monkeypatch):
    """A unit larger than the whole cap can never fit, thus it is dropped instead of waited on."""
    monkeypatch.setattr("tools.slurm_submit._squeue", lambda *a, **kw: [])
    monkeypatch.setattr("tools.slurm_submit._submit_cells", lambda cells, **kw: pytest.fail("must not submit"))

    units = _build_units("superviz26", "in_domain", "ocsvm", "li")
    done: set[Cell] = set()
    remaining = _tick(units, done=done, max_jobs=1, user="tester", count_array_tasks=True, dry_run=True)

    assert remaining == 0
    assert done == set(units[0].cells)


def test_target_arches_covers_every_partition_a_cpu_free_cell_can_land_on():
    cfg = {**_GPU_CFG, "gpu_arch": {"A100": "sm_80", "A30": "sm_80", "V100-16GB": "sm_70", "V100-32GB": "sm_70"}}
    targets = _target_arches([Cell("a", "ae", "li")], cfg, "gpu")
    assert targets == {"A100": "sm_80", "A30": "sm_80", "V100-16GB": "sm_70", "V100-32GB": "sm_70"}


def test_target_arches_skips_partitions_a_cell_has_too_little_vram_for():
    cfg = {**_GPU_CFG, "gpu_arch": {"A100": "sm_80", "A30": "sm_80", "V100-16GB": "sm_70", "V100-32GB": "sm_70"}}
    assert "V100-16GB" not in _target_arches([Cell("a", "ocsvm", "codet5")], cfg, "gpu")


def test_target_arches_is_empty_for_a_cpu_only_grid():
    cfg = {**_GPU_CFG, "gpu_arch": {"A100": "sm_80"}}
    assert _target_arches([Cell("a", "ocsvm", "li")], cfg, "gpu") == {}


def test_configured_partitions_all_declare_a_gpu_arch():
    cfg = yaml.safe_load(Path("configs/slurm.yaml").read_text())
    for section in ("gpu", "gpu-long"):
        assert set(cfg[section]["partitions"]) <= set(cfg["gpu_arch"]), section
