"""Unit tests for grid enumeration, scenario resolution, and SLURM cell grouping."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
    _check_cuda_builds,
    _eligible_partitions,
    _gpu_section,
    _is_long_running,
    _min_vram,
    _needs_gpu,
    _resolve_resources,
    _tick,
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
        "partitions": {"RTX6000PRO": 96, "A100": 40, "A40": 46, "A30": 24, "V100-32GB": 32, "V100-16GB": 16},
    },
    "gpu-long": {
        "gres": "gpu:1",
        "cpus_per_task": 32,
        "mem": "64G",
        "time": "24:00:00",
        "partitions": {"RTX6000PRO": 96, "A100": 40, "A40": 46, "A30": 24, "V100-32GB": 32, "V100-16GB": 16},
    },
    "min_vram_gb": {"default": 0, "codet5": 24},
    "long_running": ["ocsvm:sbert"],
}

# Both builds support sm_80 to test that the first matching build wins.
_CUDA_CFG = {
    **_GPU_CFG,
    "gpu_arch": {
        "RTX6000PRO": "sm_120",
        "A100": "sm_80",
        "A40": "sm_86",
        "A30": "sm_80",
        "V100-32GB": "sm_70",
        "V100-16GB": "sm_70",
    },
    "cuda_builds": {
        "cu126": {"venv": ".venv-cluster", "extra": "cu126", "arch": ["sm_70", "sm_80", "sm_86"]},
        "cu130": {"venv": ".venv-cluster-cu130", "extra": "cu130", "arch": ["sm_80", "sm_120"]},
    },
}


# Some nvidia-smi versions print this to stdout and exit with code 6 when no GPU is allocated.
_NO_DEVICES = "No devices were found"


def _run_venv_selection(tmp_path, compute_cap: str | None) -> str:
    """Select an environment using simulated nvidia-smi output.

    With ``None``, use the existing PATH without a simulated command.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    path = os.environ["PATH"]
    if compute_cap is not None:
        smi = bin_dir / "nvidia-smi"
        status = 6 if compute_cap == _NO_DEVICES else 0
        smi.write_text(f"#!/bin/sh\necho '{compute_cap}'\nexit {status}\n")
        smi.chmod(0o755)
        path = f"{bin_dir}{os.pathsep}{path}"
    script = tmp_path / "select.sh"
    body = env_setup(_CUDA_CFG).split('source "$venv/bin/activate"')[0] + 'echo "$venv"'
    script.write_text(f"set -euo pipefail\n{body}\n")
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(  # noqa: S603
        [bash, str(script)], capture_output=True, text=True, check=True, env={"PATH": path}
    )
    return result.stdout.strip()


def _write_cpu_script(tmp_path, *, limit, track=False):
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
        track=track,
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
    # Exclude the 16 GB V100 and preserve config order.
    assert _eligible_partitions(_GPU_CFG["gpu"], 24) == ["RTX6000PRO", "A100", "A40", "A30", "V100-32GB"]
    assert "V100-16GB" not in _eligible_partitions(_GPU_CFG["gpu"], 24)


def test_eligible_partitions_raises_when_none_fit():
    with pytest.raises(typer.BadParameter, match="VRAM"):
        _eligible_partitions(_GPU_CFG["gpu"], 128)


def test_llm2vec_uses_only_a100_and_rtx_partitions():
    cfg = yaml.safe_load(Path("configs/slurm.yaml").read_text())
    for section in ("gpu", "gpu-long"):
        for method in ("ocsvm", "lof", "ae"):
            cell = Cell("a-a", method, "llm2vec")
            assert _resolve_resources(cfg, cell, section)["partition"] == "RTX6000PRO,A100"

    same_duration_cfg = {**cfg, "long_running": []}
    llm2vec = Cell("a-a", "ae", "llm2vec")
    codet5 = Cell("a-a", "ae", "codet5")
    assert _bucket(llm2vec, same_duration_cfg, "gpu") != _bucket(codet5, same_duration_cfg, "gpu")


def test_llm2vec_benchmark_uses_only_a100_and_rtx_partitions():
    from tools.benchmark import _bucket as benchmark_bucket
    from tools.benchmark import _resolve_resources as benchmark_resources

    cfg = yaml.safe_load(Path("configs/slurm.yaml").read_text())
    same_duration_cfg = {**cfg, "long_running": []}
    llm2vec = Cell("a-a", "ae", "llm2vec")
    codet5 = Cell("a-a", "ae", "codet5")
    assert benchmark_resources(cfg, llm2vec, "gpu")["partition"] == "RTX6000PRO,A100"
    assert benchmark_bucket(llm2vec, same_duration_cfg, "gpu") != benchmark_bucket(codet5, same_duration_cfg, "gpu")


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
    assert _bucket(Cell("a-a", "ocsvm", "sbert"), _GPU_CFG, "gpu") == "gpu-long"
    assert _gpu_section(Cell("a-a", "ocsvm", "sbert"), _GPU_CFG, "gpu") == "gpu-long"
    assert _bucket(Cell("a-a", "ae", "sbert"), _GPU_CFG, "gpu") == "gpu"
    assert _gpu_section(Cell("a-a", "ae", "sbert"), _GPU_CFG, "gpu") == "gpu"


def test_gpu_section_default_applies_to_non_long_running_cells():
    # A submission-wide --gpu-section is the default for cells not flagged long-running.
    assert _gpu_section(Cell("a-a", "ae", "li"), _GPU_CFG, "gpu-long") == "gpu-long"
    assert _bucket(Cell("a-a", "ocsvm", "codet5"), _GPU_CFG, "gpu-long") == "gpu-long-24gb"


def test_family_scenario_count_matches_suite_all():
    assert len(FAMILIES["superviz26"].suites["all"]) == 8


def test_enumerate_cells_works_for_drift_family():
    # Four domains, one method, one extractor.
    cells = enumerate_cells("superviz26-drift", "all", "ocsvm", "li")
    assert len(cells) == 4
    assert [c.scenario for c in cells] == ["a", "b", "c", "d"]
    assert all(c.method == "ocsvm" and c.extractor == "li" for c in cells)


def test_drift_cells_bucket_like_the_suite():
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


def test_job_script_marks_the_mlflow_run_failed_when_the_task_dies(tmp_path):
    script = _write_cpu_script(tmp_path, limit=None, track=True)
    assert "trap close_dead_run EXIT" in script
    manifest = tmp_path / "cells.jsonl"
    assert f"fail_killed_run('superviz25', $status, '{manifest}', ${{SLURM_ARRAY_TASK_ID:--1}})" in script
    bash = shutil.which("bash")
    assert bash is not None
    assert subprocess.run([bash, "-n", str(tmp_path / "job.sbatch")], check=False).returncode == 0  # noqa: S603


def test_job_script_has_no_exit_trap_without_tracking(tmp_path):
    assert "fail_killed_run" not in _write_cpu_script(tmp_path, limit=None, track=False)


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


def test_env_setup_selects_the_venv_from_the_gpu_of_the_node():
    lines = env_setup(_CUDA_CFG).splitlines()
    assert lines[0].startswith('gpu_cc="$(nvidia-smi')
    assert '  sm_70|sm_80|sm_86) venv=".venv-cluster"; extra="cu126" ;;' in lines
    assert '  sm_120) venv=".venv-cluster-cu130"; extra="cu130" ;;' in lines
    assert 'source "$venv/bin/activate"' in lines


def test_env_setup_stops_a_task_whose_venv_does_not_match_the_lock():
    lines = env_setup(_CUDA_CFG).splitlines()
    activate = lines.index('source "$venv/bin/activate"')
    assert 'UV_PROJECT_ENVIRONMENT="$venv" uv sync --frozen --extra "$extra" --check ||' in lines[activate:]


def test_env_setup_still_loads_the_site_modules_before_the_selection():
    cfg = {**_CUDA_CFG, "env": {"modules": ["cuda/12.8"]}}
    lines = env_setup(cfg).splitlines()
    assert lines[:2] == ["module purge", "module load cuda/12.8"]


@pytest.mark.parametrize(
    ("compute_cap", "expected"),
    [("7.0", ".venv-cluster"), ("8.0", ".venv-cluster"), ("12.0", ".venv-cluster-cu130")],
)
def test_venv_selection_matches_the_architecture_of_the_gpu(tmp_path, compute_cap, expected):
    assert _run_venv_selection(tmp_path, compute_cap) == expected


@pytest.mark.parametrize("probe", [None, _NO_DEVICES])
def test_venv_selection_takes_the_default_build_without_a_gpu(tmp_path, probe):
    """Select the default build when nvidia-smi is absent or reports no GPU."""
    assert _run_venv_selection(tmp_path, probe) == ".venv-cluster"


def test_check_cuda_builds_accepts_every_partition_a_cell_can_get():
    _check_cuda_builds(_CUDA_CFG, [Cell("a-a", "ae", "li")], "gpu")


def test_check_cuda_builds_rejects_a_partition_no_build_serves():
    cfg = {**_CUDA_CFG, "gpu_arch": {**_CUDA_CFG["gpu_arch"], "A40": "sm_89"}}
    with pytest.raises(typer.BadParameter, match="A40"):
        _check_cuda_builds(cfg, [Cell("a-a", "ae", "li")], "gpu")


def test_check_cuda_builds_ignores_cells_that_need_no_gpu():
    cfg = {**_CUDA_CFG, "gpu_arch": {**_CUDA_CFG["gpu_arch"], "A40": "sm_89"}}
    _check_cuda_builds(cfg, [Cell("a-a", "ocsvm", "li")], "gpu")


def test_tick_submits_each_cell_once_even_when_the_job_failed(monkeypatch):
    """Do not resubmit cells that disappear from the queue without finishing."""
    submitted = []
    monkeypatch.setattr("tools.slurm_submit._squeue", lambda *a, **kw: [])
    monkeypatch.setattr("tools.slurm_submit._submit_cells", lambda cells, **kw: submitted.append(list(cells)))

    units = _build_units("superviz26", "in_domain", "ocsvm", "li")
    done: set[Cell] = set()
    tick = lambda: _tick(units, done=done, max_jobs=24, user="tester", count_array_tasks=True, dry_run=True)  # noqa: E731

    assert tick() == 0
    assert tick() == 0
    assert len(submitted) == 1
    assert submitted[0] == list(units[0].cells)


def test_tick_drops_a_unit_too_big_for_the_cap(monkeypatch):
    """Skip groups that exceed the job limit."""
    monkeypatch.setattr("tools.slurm_submit._squeue", lambda *a, **kw: [])
    monkeypatch.setattr("tools.slurm_submit._submit_cells", lambda cells, **kw: pytest.fail("must not submit"))

    units = _build_units("superviz26", "in_domain", "ocsvm", "li")
    done: set[Cell] = set()
    remaining = _tick(units, done=done, max_jobs=1, user="tester", count_array_tasks=True, dry_run=True)

    assert remaining == 0
    assert done == set(units[0].cells)


def test_configured_partitions_all_declare_a_gpu_arch():
    cfg = yaml.safe_load(Path("configs/slurm.yaml").read_text())
    for section in ("gpu", "gpu-long"):
        assert set(cfg[section]["partitions"]) <= set(cfg["gpu_arch"]), section


def test_every_configured_architecture_has_a_cuda_build():
    cfg = yaml.safe_load(Path("configs/slurm.yaml").read_text())
    served = {arch for build in cfg["cuda_builds"].values() for arch in build["arch"]}
    assert set(cfg["gpu_arch"].values()) <= served


def test_the_configured_site_submits_without_a_cuda_gap():
    cfg = yaml.safe_load(Path("configs/slurm.yaml").read_text())
    _check_cuda_builds(cfg, list(enumerate_cells("superviz26", "all", "ae", "li")), "gpu")


def test_slurm_submit_imports_without_the_training_stack():
    probe = "import tools.slurm_submit, sys; print(sorted({'torch', 'sklearn', 'transformers'} & sys.modules.keys()))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)  # noqa: S603
    assert result.stdout.strip() == "[]"
