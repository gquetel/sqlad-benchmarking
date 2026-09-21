"""Submit evaluation grids to SLURM."""

from __future__ import annotations

import getpass
import json
import logging
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, NamedTuple

import mlflow
import typer
import yaml

from sqlad_benchmarking.datasets import FAMILIES
from sqlad_benchmarking.grid import DEFAULT_KS, GPU_EXTRACTORS, Cell, enumerate_cells, parent_run_spec
from sqlad_benchmarking.tracking import ensure_parent_run, experiment_name, setup_mlflow

logger = logging.getLogger(__name__)

SCENARIO_TAG = {"suite": "scenario", "drift": "domain", "fsl": "target"}

REPO_ROOT = Path(__file__).resolve().parents[1]

SUBMIT_DIR = "reports/slurm"
DEFAULT_ENV = {"venv": ".venv-cluster", "modules": []}
# Convert a GPU version such as 12.0 to sm_120. Ignore nvidia-smi errors on nodes without a GPU.
ARCH_PROBE = [
    "gpu_cc=\"$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)\"",
    'case "$gpu_cc" in',
    '  [0-9]*.[0-9]*) gpu_arch="sm_${gpu_cc%.*}${gpu_cc#*.}" ;;',
    '  *) gpu_arch="" ;;',
    "esac",
]
# Find uv and stop if the environment does not match uv.lock.
VENV_CHECK = [
    'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi',
    'UV_PROJECT_ENVIRONMENT="$venv" uv sync --frozen --extra "$extra" --check ||',
    '  { echo "$venv does not match uv.lock; rebuild it with tools/setup-env.sh" >&2; exit 1; }',
]


def _env(cfg: dict) -> dict:
    """Return the configured task environment."""
    return {**DEFAULT_ENV, **cfg.get("env", {})}


def _cuda_builds(cfg: dict) -> dict:
    """Return the configured CUDA builds."""
    return cfg.get("cuda_builds") or {}


def _build_by_arch(cfg: dict) -> dict[str, dict]:
    """Map each GPU architecture to the first build that supports it."""
    mapping: dict[str, dict] = {}
    for build in _cuda_builds(cfg).values():
        for arch in build.get("arch", []):
            mapping.setdefault(arch, build)
    return mapping


def _default_build(cfg: dict) -> dict:
    """Return the build for a task with no GPU."""
    return next(iter(_cuda_builds(cfg).values()))


def _venv_select(cfg: dict) -> list[str]:
    """Generate shell commands to select and check the Python environment for this GPU."""
    by_arch = _build_by_arch(cfg)
    if not by_arch:
        return [f"source {_env(cfg)['venv']}/bin/activate"]
    by_build: dict[tuple[str, str], list[str]] = {}
    for arch, build in by_arch.items():
        by_build.setdefault((build["venv"], build["extra"]), []).append(arch)
    default = _default_build(cfg)
    return [
        *ARCH_PROBE,
        'case "$gpu_arch" in',
        *(f'  {"|".join(arches)}) venv="{venv}"; extra="{extra}" ;;' for (venv, extra), arches in by_build.items()),
        f'  "") venv="{default["venv"]}"; extra="{default["extra"]}" ;;',
        '  *) echo "no CUDA build for $gpu_arch; check configs/slurm.yaml" >&2; exit 1 ;;',
        "esac",
        'source "$venv/bin/activate"',
        *VENV_CHECK,
    ]


def env_setup(cfg: dict) -> str:
    """Build the task environment setup commands."""
    env = _env(cfg)
    lines = ["module purge", *(f"module load {m}" for m in env["modules"])] if env["modules"] else []
    return "\n".join(lines + _venv_select(cfg))


def _needs_gpu(cell: Cell) -> bool:
    """Return whether a cell needs a GPU."""
    return cell.method == "ae" or cell.extractor in GPU_EXTRACTORS


def _min_vram(cell: Cell, cfg: dict) -> int:
    """Return the required GPU memory in GB."""
    reqs = cfg.get("min_vram_gb", {})
    key = f"{cell.method}:{cell.extractor}"
    return int(reqs.get(key, reqs.get(cell.extractor, reqs.get("default", 0))))


def _allowed_partitions(cell: Cell, cfg: dict) -> list[str] | None:
    """Return the GPU partition allowlist for an extractor, if configured."""
    return cfg.get("allowed_partitions", {}).get(cell.extractor)


def _eligible_partitions(gpu_cfg: dict, req: int, allowed: list[str] | None = None) -> list[str]:
    """Return GPU partitions with enough memory and permission for the extractor."""
    eligible = [
        name for name, gb in gpu_cfg["partitions"].items() if gb >= req and (allowed is None or name in allowed)
    ]
    if not eligible:
        raise typer.BadParameter(f"no allowed GPU partition has >= {req} GB VRAM; check configs/slurm.yaml.")
    return eligible


def _is_long_running(cell: Cell, cfg: dict) -> bool:
    """Return whether a cell needs the longer time limit."""
    keys = set(cfg.get("long_running", []))
    return f"{cell.method}:{cell.extractor}" in keys or cell.extractor in keys


def _gpu_section(cell: Cell, cfg: dict, default: str) -> str:
    """Return a cell's GPU resource section."""
    return "gpu-long" if _is_long_running(cell, cfg) else default


def _bucket(cell: Cell, cfg: dict, gpu_section: str) -> str:
    """Return a cell's group based on GPU, memory, and partition needs."""
    if not _needs_gpu(cell):
        return "cpu"
    section = _gpu_section(cell, cfg, gpu_section)
    req = _min_vram(cell, cfg)
    bucket = section if req <= 0 else f"{section}-{req}gb"
    allowed = _allowed_partitions(cell, cfg)
    if allowed is not None:
        bucket = f"{bucket}-{'-'.join(allowed)}"
    return bucket


def _check_cuda_builds(cfg: dict, cells: list[Cell], gpu_section: str) -> None:
    """Reject submissions with a configured GPU architecture that no CUDA build supports."""
    by_arch = _build_by_arch(cfg)
    if not by_arch:
        return
    declared = cfg.get("gpu_arch", {})
    unserved: dict[str, str] = {}
    for cell in cells:
        if not _needs_gpu(cell):
            continue
        section = cfg[_gpu_section(cell, cfg, gpu_section)]
        for partition in _eligible_partitions(section, _min_vram(cell, cfg), _allowed_partitions(cell, cfg)):
            arch = declared.get(partition)
            if arch is None:
                logger.warning(f"partition {partition} has no gpu_arch entry; its tasks take the default build.")
            elif arch not in by_arch:
                unserved[partition] = arch
    if unserved:
        listed = ", ".join(f"{p} ({a})" for p, a in sorted(unserved.items()))
        raise typer.BadParameter(f"no CUDA build has kernels for {listed}; check configs/slurm.yaml.")


def _experiment_tag(dataset: str, suite: str) -> str:
    """Return the experiment tag used in job names."""
    family = FAMILIES.get(dataset)
    if family is not None and family.protocol == "fsl":
        return "fsl"
    if family is not None and family.protocol == "drift":
        return "cd"
    if dataset == "superviz26-big":
        return "big"
    return {"in_domain": "id"}.get(suite, suite)


def _job_name(dataset: str, suite: str, cells: list[Cell]) -> str:
    """Build a SLURM job name."""
    extractors = "+".join(dict.fromkeys(c.extractor for c in cells))
    methods = "+".join(dict.fromkeys(c.method for c in cells))
    return f"{_experiment_tag(dataset, suite)}-{extractors}-{methods}-{dataset}"


def _resolve_resources(cfg: dict, cell: Cell, gpu_section: str = "gpu", gpu_qos: str | None = None) -> dict:
    """Return a cell's SLURM resource settings."""
    if not _needs_gpu(cell):
        return dict(cfg["cpu"])
    if gpu_section not in cfg:
        raise typer.BadParameter(f"GPU section {gpu_section!r} not found in config; check configs/slurm.yaml.")
    gpu_cfg = cfg[gpu_section]
    res = {k: v for k, v in gpu_cfg.items() if k != "partitions"}
    res["partition"] = ",".join(_eligible_partitions(gpu_cfg, _min_vram(cell, cfg), _allowed_partitions(cell, cfg)))
    if gpu_qos:
        res["qos"] = gpu_qos
    return res


def _write_manifest(path: Path, cells: list[Cell]) -> None:
    """Write one cell per line as JSON."""
    path.write_text("".join(json.dumps(cell._asdict()) + "\n" for cell in cells))


def _reap_trap(dataset: str, manifest: Path, track: bool) -> str:
    """Generate a shell exit handler to mark failed MLflow runs."""
    if not track:
        return ""
    call = (
        "from sqlad_benchmarking.tracking import fail_killed_run; "
        f"fail_killed_run('{dataset}', $status, '{manifest}', ${{SLURM_ARRAY_TASK_ID:--1}})"
    )
    return f"""close_dead_run() {{
  status=$?
  if [ "$status" -ne 0 ]; then
    python -c "{call}"
  fi
}}
trap close_dead_run EXIT
"""


def _write_job_script(
    path: Path,
    *,
    job_name: str,
    res: dict,
    cfg: dict,
    dataset: str,
    manifest: Path,
    n: int,
    log_pattern: str,
    target_fpr: float,
    seed: int,
    track: bool,
    limit: int | None,
) -> None:
    """Write a SLURM array script."""
    directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_pattern}",
        f"#SBATCH --error={log_pattern}",
        f"#SBATCH --partition={res['partition']}",
    ]
    if res.get("gres"):
        directives.append(f"#SBATCH --gres={res['gres']}")
    if res.get("qos"):
        directives.append(f"#SBATCH --qos={res['qos']}")
    directives += [
        f"#SBATCH --cpus-per-task={res['cpus_per_task']}",
        f"#SBATCH --mem={res['mem']}",
        f"#SBATCH --time={res['time']}",
        f"#SBATCH --array=0-{n - 1}",
    ]
    if cfg.get("account"):
        directives.append(f"#SBATCH --account={cfg['account']}")
    register_flag = "--register" if track and cfg.get("register") else "--no-register"
    track_flag = "--track" if track else "--no-track"
    limit_flag = f" \\\n  --limit {limit}" if limit is not None else ""
    header = "\n".join(directives)
    path.write_text(
        f"""#!/bin/bash
{header}
set -euo pipefail
cd {REPO_ROOT}
{env_setup(cfg)}
{_reap_trap(dataset, manifest, track)}
python -m tools.slurm_run_cell \\
  --manifest {manifest} \\
  --index "$SLURM_ARRAY_TASK_ID" \\
  --dataset {dataset} \\
  --target-fpr {target_fpr} \\
  --seed {seed} \\
  {register_flag} {track_flag}{limit_flag}
"""
    )


def _submit_array(script: Path, dry_run: bool) -> str | None:
    """Submit an array and return its job ID."""
    cmd = ["sbatch", str(script)]
    if dry_run:
        logger.info("DRY-RUN: " + " ".join(cmd))
        return None
    sbatch = shutil.which("sbatch")
    if not sbatch:
        raise typer.BadParameter("sbatch not found on PATH; run on a SLURM submit node or use --dry-run.")
    cmd[0] = sbatch
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise typer.BadParameter(f"sbatch rejected {script}:\n{result.stderr.strip() or result.stdout.strip()}")
    logger.info(result.stdout.strip())
    return result.stdout.strip().split()[-1]


def _submit_cells(
    cells: list[Cell],
    *,
    dataset: str,
    suite: str,
    cfg: dict,
    gpu_section: str,
    gpu_qos: str | None,
    target_fpr: float,
    seed: int,
    track: bool,
    limit: int | None,
    run_id: str | None,
    dry_run: bool,
) -> None:
    """Submit cells in arrays grouped by resource needs."""
    explicit_run_id = run_id is not None
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    submit_dir = REPO_ROOT / SUBMIT_DIR / run_id
    (submit_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger.info(f"{len(cells)} cells -> {submit_dir}")

    if track and setup_mlflow(dataset):
        for method, extractor in {(c.method, c.extractor) for c in cells}:
            name, tags = parent_run_spec(FAMILIES[dataset], method, extractor)
            ensure_parent_run(tags, name)

    buckets: dict[str, list[Cell]] = {}
    for cell in cells:
        buckets.setdefault(_bucket(cell, cfg, gpu_section), []).append(cell)
    job_ids: list[str] = []
    for bucket, group_cells in sorted(buckets.items()):
        res = _resolve_resources(cfg, group_cells[0], _gpu_section(group_cells[0], cfg, gpu_section), gpu_qos)
        manifest = submit_dir / f"cells_{bucket}.jsonl"
        script = submit_dir / f"eval_cell_{bucket}.sbatch"
        _write_manifest(manifest, group_cells)
        _write_job_script(
            script,
            job_name=_job_name(dataset, suite, group_cells),
            res=res,
            cfg=cfg,
            dataset=dataset,
            manifest=manifest,
            n=len(group_cells),
            log_pattern=str(submit_dir / "logs" / f"{bucket}-%A_%a.log"),
            target_fpr=target_fpr,
            seed=seed,
            track=track,
            limit=limit,
        )
        logger.info(f"{bucket}: {len(group_cells)} cells (partition {res['partition']})")
        job_id = _submit_array(script, dry_run)
        if job_id:
            job_ids.append(job_id)

    if dry_run:
        logger.info("Dry run: nothing submitted.")
    elif job_ids:
        logger.info(f"Submitted job arrays: {', '.join(job_ids)}")
        if not explicit_run_id:
            final_dir = REPO_ROOT / SUBMIT_DIR / f"{run_id.split('-')[0]}-{job_ids[0]}"
            submit_dir.rename(final_dir)
            submit_dir.symlink_to(final_dir)
            logger.info(f"Submission dir: {final_dir}")


class Unit(NamedTuple):
    """Cells submitted together for one method and extractor."""

    method: str
    extractor: str
    cells: tuple[Cell, ...]
    job_name: str


def _build_units(dataset: str, suite: str, methods: str, extractors: str) -> list[Unit]:
    """Group cells by method and extractor for submission."""
    units = []
    for method in methods.split(","):
        for extractor in extractors.split(","):
            cells = tuple(enumerate_cells(dataset, suite, method, extractor))
            units.append(Unit(method, extractor, cells, _job_name(dataset, suite, list(cells))))
    return units


def _protocol(dataset: str) -> str:
    """Return a dataset's evaluation protocol."""
    return FAMILIES[dataset].protocol if dataset in FAMILIES else "suite"


def _finished_cells(dataset: str, run_type: str | None, ks: str) -> set[Cell]:
    """Return cells with complete tracked runs."""
    protocol = _protocol(dataset)
    filters = ["attributes.status = 'FINISHED'"]
    if run_type and protocol != "fsl":
        filters.append(f"tags.run_type = '{run_type}'")
    name = experiment_name(dataset)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        raise typer.BadParameter(f"experiment {name!r} does not exist on {mlflow.get_tracking_uri()}.")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=" and ".join(filters), output_format="list"
    )

    tag = SCENARIO_TAG[protocol]
    seen: dict[Cell, set[str]] = defaultdict(set)
    for run in runs:
        tags = run.data.tags
        scenario = tags.get(tag)
        if not scenario:
            continue
        cell = Cell(scenario, tags.get("decision_engine", ""), tags.get("feature_extractor", ""))
        seen[cell].add(run.data.params.get("k", "") if protocol == "fsl" else "")
    expected = set(ks.split(",")) if protocol == "fsl" else set()
    return {cell for cell, got in seen.items() if not expected - got}


def _outstanding(units: list[Unit], done: set[Cell]) -> dict[str, list[Cell]]:
    """Group outstanding cells by job name."""
    pending = {}
    for unit in units:
        cells = [cell for cell in unit.cells if cell not in done]
        if cells:
            pending[unit.job_name] = cells
    return pending


def _squeue(user: str, count_array_tasks: bool, dry_run: bool = False) -> list[str]:
    """Return a user's queued and running job names."""
    squeue = shutil.which("squeue")
    if not squeue:
        if dry_run:
            logger.info("DRY-RUN: squeue not found, assuming an empty queue.")
            return []
        raise typer.BadParameter("squeue not found on PATH; run this on the SLURM submit node or use --dry-run.")
    cmd = [squeue, "-h", "-u", user, "-O", "Name:200"]
    if count_array_tasks:
        cmd.append("-r")
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"squeue failed: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _tick(
    units: list[Unit], *, done: set[Cell], max_jobs: int, user: str, count_array_tasks: bool, **submit_kwargs
) -> int:
    """Fill available queue capacity and return the units left."""
    running = _squeue(user, count_array_tasks, dry_run=submit_kwargs["dry_run"])
    in_flight_names = set(running)
    headroom = max_jobs - len(running)

    pending = {name: cells for name, cells in _outstanding(units, done).items() if name not in in_flight_names}
    remaining = sum(len(cells) for cells in pending.values())
    logger.info(f"{len(running)} job(s) currently running, {remaining} job(s) remaining")

    by_name = {unit.job_name: unit for unit in units}
    for job_name, cells in pending.items():
        n = len(cells)
        if n > max_jobs:
            logger.warning(f"dropping {job_name}: {n} cells exceeds the {max_jobs}-job cap on its own")
            done.update(cells)
            continue
        if n > headroom:
            continue
        unit = by_name[job_name]
        logger.info(f"submitting {job_name} ({n} cells)")
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{unit.method}-{unit.extractor}"
        try:
            _submit_cells(cells, run_id=run_id, **submit_kwargs)
        except Exception as exc:
            logger.error(f"submit failed for {job_name}, not retrying: {exc}")
        done.update(cells)
        headroom -= n
    return len(_outstanding(units, done))


def submit(
    dataset: Annotated[str, typer.Option(help="Dataset family: superviz26 or superviz25.")] = "superviz26",
    suite: Annotated[str, typer.Option(help="Suite name (e.g. in_domain, lodo, all).")] = "all",
    methods: Annotated[str, typer.Option(help="Comma-separated decision-head names (ocsvm, lof, ae).")] = "ocsvm,ae",
    extractors: Annotated[str, typer.Option(help="Comma-separated registered feature-extractor names.")] = "li",
    config: Annotated[Path, typer.Option(help="SLURM site config.")] = Path("configs/slurm.yaml"),
    check_mlflow: Annotated[
        bool, typer.Option(help="Skip cells that already have a FINISHED run, looked up once at startup.")
    ] = False,
    run_type: Annotated[
        str, typer.Option(help="Only count runs tagged with this run_type ('full-run', 'smoke-run', '' for any).")
    ] = "full-run",
    ks: Annotated[str, typer.Option(help="Few-shot budgets a target needs before it counts as finished.")] = DEFAULT_KS,
    queue: Annotated[
        bool, typer.Option(help="Drip-feed under --max-jobs; off submits every outstanding cell at once.")
    ] = True,
    max_jobs: Annotated[int, typer.Option(help="Cap on jobs in flight (queued + running) at any time.")] = 24,
    interval: Annotated[int, typer.Option(help="Seconds between checks.")] = 300,
    count_array_tasks: Annotated[
        bool, typer.Option(help="Count each array task against the cap (off: count whole arrays as one job).")
    ] = True,
    gpu_section: Annotated[
        str,
        typer.Option(help="GPU resource block in the config to use (e.g. 'gpu', 'gpu-long' for a 24h reservation)."),
    ] = "gpu",
    gpu_qos: Annotated[
        str,
        typer.Option(help="QoS for GPU jobs; 'runfill' is preemptible, so many more can run. Empty for the default."),
    ] = "runfill",
    target_fpr: Annotated[float, typer.Option(help="Target false-positive rate for the calibrated threshold.")] = 0.001,
    seed: Annotated[int, typer.Option(help="Random state for the train/validation calibration split.")] = 7,
    register: Annotated[bool, typer.Option(help="Register each fitted model in the MLflow Model Registry.")] = False,
    no_track: Annotated[bool, typer.Option(help="Disable MLflow tracking for the submitted jobs.")] = False,
    limit: Annotated[int | None, typer.Option(help="Label-stratified subset size per cell for smoke runs.")] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Submission id naming the dir under reports/slurm (--no-queue only; else one per unit)."),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the manifests and sbatch commands without submitting.")] = False,
) -> None:
    """Submit outstanding grid cells."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(config.read_text())
    cfg["register"] = register

    units = _build_units(dataset, suite, methods, extractors)
    total_cells = sum(len(u.cells) for u in units)
    _check_cuda_builds(cfg, [cell for unit in units for cell in unit.cells], gpu_section)

    if check_mlflow and not setup_mlflow(dataset):
        raise typer.BadParameter("MLFLOW_TRACKING_URI is not set; --check-mlflow cannot look up what finished.")

    submit_kwargs = dict(
        dataset=dataset,
        suite=suite,
        cfg=cfg,
        gpu_section=gpu_section,
        gpu_qos=gpu_qos or None,
        target_fpr=target_fpr,
        seed=seed,
        track=not no_track,
        limit=limit,
        dry_run=dry_run,
    )
    done: set[Cell] = set()
    if check_mlflow:
        done = _finished_cells(dataset, run_type or None, ks)
        logger.info(f"{sum(1 for u in units for c in u.cells if c in done)}/{total_cells} cells already finished")

    if not queue:
        cells = [cell for unit in units for cell in unit.cells if cell not in done]
        if not cells:
            logger.info("nothing to submit.")
            return
        _submit_cells(cells, run_id=run_id, **submit_kwargs)
        return

    logger.info(f"{len(units)} units / {total_cells} cells; cap {max_jobs} jobs, checking every {interval}s")
    while True:
        remaining = _tick(
            units,
            done=done,
            max_jobs=max_jobs,
            user=getpass.getuser(),
            count_array_tasks=count_array_tasks,
            **submit_kwargs,
        )
        if remaining == 0:
            logger.info("all units submitted or done.")
            return
        time.sleep(interval)


if __name__ == "__main__":
    typer.run(submit)
