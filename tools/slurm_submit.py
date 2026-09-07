"""Fan an evaluation grid out to SLURM, drip-feeding it under the cluster's in-flight job cap.

Each cell ``(scenario, method, extractor)`` becomes one array task that runs through
:func:`evaluate_suite` and writes its own per-cell CSV under ``reports/{dataset}/cells/``
(no shared writer, no merge step). Cells needing a GPU (``method == "ae"`` or an
extractor in :data:`sqlad_benchmarking.features.GPU_EXTRACTORS`) go to a GPU array whose partition list is
the GPU partitions with enough VRAM for that cell (per-cell minimum from ``min_vram_gb``, so a
hungry model like CodeT5+ skips the 16 GB V100); the rest go to a CPU array. Cells listed in
``long_running`` (config, keyed like ``min_vram_gb``) run on the 24h ``gpu-long`` block instead
of the default 12h ``gpu`` block. Resources and the environment setup come from ``configs/slurm.yaml``.

The cluster caps in-flight jobs per user (~24) and a full grid is hundreds, so by default this
does not submit everything at once: it splits the grid into **units** (one ``(method,
extractor)``) and every ``--interval`` seconds submits as many outstanding cells as the current
headroom allows. **Every cell goes out exactly once.** A cell that fails is not resubmitted:
a failure is nearly always a bug or a bad resource request, and retrying it only burns compute
and fills the tracking server with dead runs. Fix the cause, then run the command again --
which is also how a preempted cell gets another chance.

- **done** -- already submitted in this session, or (with ``--check-mlflow``) it already had a
  FINISHED run when the command started. The lookup happens once, at startup.
- **in flight** -- a job named like the unit's :func:`_job_name` is in ``squeue``, e.g. from an
  earlier invocation that is still running.
- **pending** -- everything else; submitted while headroom remains. A unit submits only its
  outstanding cells, so a partly finished one costs a partial array.

``--check-mlflow`` skips cells that already finished, which is how you fill in the holes left
by a broken batch; ``--no-queue`` submits everything in one go, ignoring the cap.

Run it on the submit node under ``tmux``/``nohup`` so a dropped VPN does not kill it:

    nohup uv run --frozen --extra cu126 python -m tools.slurm_submit \\
      --dataset superviz26 --suite all --methods ocsvm,lof,ae --extractors li,cv,sbert,codet5 \\
      > reports/slurm/queue.log 2>&1 &

Always give ``uv run`` the extra: without it uv replaces the pinned CUDA build of torch in
the venv the compute nodes activate.

Usage:
    # See what it would do, without touching the cluster:
    python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --dry-run --once
    # One pass then exit (e.g. from cron):
    python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --once
    # Only the cells that have no FINISHED run, to fill in the holes left by a broken batch:
    python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --check-mlflow
    # Everything at once, ignoring the cap (the pre-queue behaviour):
    python -m tools.slurm_submit --dataset superviz26 --methods ae --no-queue
    # Full-split data is heavy; give the AE/CodeT5+ cells a 24h GPU reservation:
    python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --gpu-section gpu-long
"""

from __future__ import annotations

import getpass
import json
import logging
import os
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
from sqlad_benchmarking.evaluate_fsl import DEFAULT_KS
from sqlad_benchmarking.evaluate_suite import Cell, enumerate_cells, parent_run_spec
from sqlad_benchmarking.features import GPU_EXTRACTORS
from sqlad_benchmarking.tracking import ensure_parent_run, experiment_name, setup_mlflow

logger = logging.getLogger(__name__)

# The tag naming a cell's scenario differs per protocol: each evaluator names it after what
# it iterates over.
SCENARIO_TAG = {"suite": "scenario", "drift": "domain", "fsl": "target"}

REPO_ROOT = Path(__file__).resolve().parents[1]

# Manifests, generated job scripts and .out logs (git-ignored), relative to REPO_ROOT.
SUBMIT_DIR = "reports/slurm"
# Site environment, when configs/slurm.yaml has no `env` block.
DEFAULT_ENV = {"venv": ".venv-cluster", "modules": []}
# The extra the compute nodes need: they run the GPU cells.
VENV_EXTRA = "cu126"


def _env(cfg: dict) -> dict:
    """Site environment settings, with the defaults filled in."""
    return {**DEFAULT_ENV, **cfg.get("env", {})}


def env_setup(cfg: dict) -> str:
    """Shell lines each array task runs before its cell: the site modules, then the venv.

    The venv is built once on the submit node and reused from the shared filesystem; array
    tasks only activate it, because a concurrent `uv sync` would race on one directory.
    """
    env = _env(cfg)
    lines = ["module purge", *(f"module load {m}" for m in env["modules"])] if env["modules"] else []
    lines.append(f"source {env['venv']}/bin/activate")
    return "\n".join(lines)


def _needs_gpu(cell: Cell) -> bool:
    """A cell needs a GPU when it trains an autoencoder or uses a GPU embedding extractor."""
    return cell.method == "ae" or cell.extractor in GPU_EXTRACTORS


def _min_vram(cell: Cell, cfg: dict) -> int:
    """Per-GPU VRAM (GB) a cell needs: engine:extractor overrides extractor, else the default."""
    reqs = cfg.get("min_vram_gb", {})
    key = f"{cell.method}:{cell.extractor}"
    return int(reqs.get(key, reqs.get(cell.extractor, reqs.get("default", 0))))


def _eligible_partitions(gpu_cfg: dict, req: int) -> list[str]:
    """GPU partitions meeting req GB of VRAM, in config (preference) order so cells prefer the fastest GPU."""
    eligible = [name for name, gb in gpu_cfg["partitions"].items() if gb >= req]
    if not eligible:
        raise typer.BadParameter(f"no GPU partition has >= {req} GB VRAM; check configs/slurm.yaml.")
    return eligible


def _is_long_running(cell: Cell, cfg: dict) -> bool:
    """Whether a cell needs extended wall time, per config ``long_running`` (keyed by extractor or method)."""
    keys = set(cfg.get("long_running", []))
    return f"{cell.method}:{cell.extractor}" in keys or cell.extractor in keys


def _gpu_section(cell: Cell, cfg: dict, default: str) -> str:
    """GPU resource block for a cell: ``gpu-long`` when flagged long-running, else the submission default."""
    return "gpu-long" if _is_long_running(cell, cfg) else default


def _bucket(cell: Cell, cfg: dict, gpu_section: str) -> str:
    """Bucket label for a cell's array: cpu, the GPU section (gpu/gpu-long), or ``{section}-{req}gb`` for VRAM-pinned.

    The section is baked into the label so long-running cells land in their own array (each
    array carries one #SBATCH header) instead of sharing the default GPU block's wall time.
    """
    if not _needs_gpu(cell):
        return "cpu"
    section = _gpu_section(cell, cfg, gpu_section)
    req = _min_vram(cell, cfg)
    return section if req <= 0 else f"{section}-{req}gb"


def _experiment_tag(dataset: str, suite: str) -> str:
    """Short experiment-type tag for the job name: fsl/cd/big by family, else id/lodo by suite."""
    family = FAMILIES.get(dataset)
    if family is not None and family.protocol == "fsl":
        return "fsl"
    if family is not None and family.protocol == "drift":
        return "cd"
    if dataset == "superviz26-big":
        return "big"
    return {"in_domain": "id"}.get(suite, suite)


def _job_name(dataset: str, suite: str, cells: list[Cell]) -> str:
    """Informative SLURM job name: {tag}-{extractors}-{methods}-{dataset}.

    Buckets split by resource class, so one array can mix extractors/methods; each is
    listed once (in first-seen order) joined by '+' (e.g. id-li+cv-ocsvm+ae-superviz26).
    """
    extractors = "+".join(dict.fromkeys(c.extractor for c in cells))
    methods = "+".join(dict.fromkeys(c.method for c in cells))
    return f"{_experiment_tag(dataset, suite)}-{extractors}-{methods}-{dataset}"


def _resolve_resources(cfg: dict, cell: Cell, gpu_section: str = "gpu", gpu_qos: str | None = None) -> dict:
    """SBATCH resource block for a cell: the cpu block, or ``gpu_section`` with a VRAM-filtered partition list.

    ``gpu_qos`` (opt-in, GPU cells only) overrides the QoS, e.g. the preemptible ``runfill``.
    """
    if not _needs_gpu(cell):
        return dict(cfg["cpu"])
    if gpu_section not in cfg:
        raise typer.BadParameter(f"GPU section {gpu_section!r} not found in config; check configs/slurm.yaml.")
    gpu_cfg = cfg[gpu_section]
    res = {k: v for k, v in gpu_cfg.items() if k != "partitions"}
    res["partition"] = ",".join(_eligible_partitions(gpu_cfg, _min_vram(cell, cfg)))
    if gpu_qos:
        res["qos"] = gpu_qos
    return res


def _check_venv(activate: Path) -> None:
    """Fail fast if the shared venv is missing; every array task sources it on the compute node."""
    if not activate.exists():
        raise typer.BadParameter(f"{activate} not found; build it with `. tools/setup-env.sh` before submitting.")


def _check_lock(venv: Path) -> None:
    """Fail fast if the venv no longer matches uv.lock, e.g. after a pull.

    Array tasks never sync, thus a stale venv makes every one of them run the old packages.
    """
    uv = shutil.which("uv")
    if uv is None:
        logger.warning(f"uv is not on PATH; cannot check that {venv.name} matches uv.lock.")
        return
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}
    # Safe: shell=False, absolute uv path, fixed arguments.
    check = subprocess.run([uv, "sync", "--frozen", "--extra", VENV_EXTRA, "--check"], env=env, capture_output=True)  # noqa: S603
    if check.returncode != 0:
        raise typer.BadParameter(f"{venv.name} does not match uv.lock; run `. tools/setup-env.sh` before submitting.")


def _check_env(cfg: dict) -> None:
    """Check the shared venv on the submit node, once, before N array tasks activate it."""
    venv = REPO_ROOT / _env(cfg)["venv"]
    _check_venv(venv / "bin" / "activate")
    _check_lock(venv)


def _write_manifest(path: Path, cells: list[Cell]) -> None:
    path.write_text("".join(json.dumps(cell._asdict()) + "\n" for cell in cells))


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
    """Generate a self-contained array script: full #SBATCH header, then dispatch one cell per index."""
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
    """Submit (or, in dry-run, just print) one job array and return its job id.

    Resources live in the script's #SBATCH header, so submission is just ``sbatch <script>``.
    """
    cmd = ["sbatch", str(script)]
    if dry_run:
        logger.info("DRY-RUN: " + " ".join(cmd))
        return None
    sbatch = shutil.which("sbatch")
    if not sbatch:
        raise typer.BadParameter("sbatch not found on PATH; run on a SLURM submit node or use --dry-run.")
    cmd[0] = sbatch
    # Safe: shell=False, absolute sbatch path, script generated from the versioned config.
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        # Propagate sbatch errors
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
    """Pre-create the MLflow parents and submit ``cells`` as one job array per resource class."""
    # An explicit run_id is honoured as-is; an auto-generated one starts as a timestamp so we
    # have a directory to stage manifests/scripts in, then gets renamed to date-<jobid> once
    # sbatch hands back the first array's id (the id doesn't exist until after submission).
    explicit_run_id = run_id is not None
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    submit_dir = REPO_ROOT / SUBMIT_DIR / run_id
    (submit_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger.info(f"{len(cells)} cells -> {submit_dir}")

    # Pre-create the MLflow parents once, serially, so concurrent array tasks reuse them
    # instead of racing on find-or-create and spawning duplicate parents.
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
            # Rename the staging dir to date-<first job id>. The in-flight arrays carry the
            # original absolute --output/--error paths, so leave a symlink at the old path for
            # tasks that open their logs after the rename.
            final_dir = REPO_ROOT / SUBMIT_DIR / f"{run_id.split('-')[0]}-{job_ids[0]}"
            submit_dir.rename(final_dir)
            submit_dir.symlink_to(final_dir)
            logger.info(f"Submission dir: {final_dir}")


class Unit(NamedTuple):
    """One submission: a ``(method, extractor)`` pair across every scenario of the suite."""

    method: str
    extractor: str
    cells: tuple[Cell, ...]
    job_name: str


def _build_units(dataset: str, suite: str, methods: str, extractors: str) -> list[Unit]:
    """Split the grid into one unit per ``(method, extractor)``, in the order given on the CLI."""
    units = []
    for method in methods.split(","):
        for extractor in extractors.split(","):
            cells = tuple(enumerate_cells(dataset, suite, method, extractor))
            units.append(Unit(method, extractor, cells, _job_name(dataset, suite, list(cells))))
    return units


def _protocol(dataset: str) -> str:
    return FAMILIES[dataset].protocol if dataset in FAMILIES else "suite"


def _finished_cells(dataset: str, run_type: str | None, ks: str) -> set[Cell]:
    """Cells with a FINISHED run on the tracking server.

    Few-shot cells sweep several ``k`` under one target, so one counts as finished only
    once every ``k`` has its own run.
    """
    protocol = _protocol(dataset)
    filters = ["attributes.status = 'FINISHED'"]
    # The few-shot evaluator does not tag run_type, so filtering on it would drop every run.
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
        # Parent runs carry no scenario tag: they group children, they are not results.
        scenario = tags.get(tag)
        if not scenario:
            continue
        cell = Cell(scenario, tags.get("decision_engine", ""), tags.get("feature_extractor", ""))
        seen[cell].add(run.data.params.get("k", "") if protocol == "fsl" else "")
    expected = set(ks.split(",")) if protocol == "fsl" else set()
    return {cell for cell, got in seen.items() if not expected - got}


def _outstanding(units: list[Unit], done: set[Cell]) -> dict[str, list[Cell]]:
    """Cells still to run, per unit job name, dropping units with nothing left."""
    pending = {}
    for unit in units:
        cells = [cell for cell in unit.cells if cell not in done]
        if cells:
            pending[unit.job_name] = cells
    return pending


def _squeue(user: str, count_array_tasks: bool, dry_run: bool = False) -> list[str]:
    """Job names currently queued or running for ``user`` (``-r`` expands array tasks into one row each).

    Returns one entry per counted job, so ``len()`` is the number of jobs against the cap
    and the names identify which units are already in flight.
    """
    squeue = shutil.which("squeue")
    if not squeue:
        if dry_run:  # let a dry run work off the submit node: pretend the queue is empty
            logger.info("DRY-RUN: squeue not found, assuming an empty queue.")
            return []
        raise typer.BadParameter("squeue not found on PATH; run this on the SLURM submit node or use --dry-run.")
    cmd = [squeue, "-h", "-u", user, "-O", "Name:200"]
    if count_array_tasks:
        cmd.append("-r")
    # Safe: shell=False, absolute squeue path, no user-controlled arguments.
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"squeue failed: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _tick(
    units: list[Unit], *, done: set[Cell], max_jobs: int, user: str, count_array_tasks: bool, **submit_kwargs
) -> int:
    """Submit as many pending units as fit under ``max_jobs``. Returns the number of units still to do.

    Every cell this tick disposes of -- submitted, rejected, or too big for the cap -- is added
    to ``done``, so the caller's loop always makes progress and no cell goes out twice.
    """
    running = _squeue(user, count_array_tasks, dry_run=submit_kwargs["dry_run"])
    in_flight_names = set(running)
    headroom = max_jobs - len(running)

    pending = {name: cells for name, cells in _outstanding(units, done).items() if name not in in_flight_names}
    logger.info(f"{len(running)} job(s) in flight, headroom {headroom}, {len(pending)} unit(s) pending")

    by_name = {unit.job_name: unit for unit in units}
    for job_name, cells in pending.items():
        n = len(cells)
        if n > max_jobs:
            # It can never fit, thus waiting for headroom would spin forever: drop it.
            logger.warning(f"dropping {job_name}: {n} cells exceeds the {max_jobs}-job cap on its own")
            done.update(cells)
            continue
        if n > headroom:
            continue
        unit = by_name[job_name]
        logger.info(f"submitting {job_name} ({n} cells)")
        # Explicit per-unit run_id: several units can be submitted in the same tick, and the
        # auto-generated id is a whole-second timestamp they would collide on.
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{unit.method}-{unit.extractor}"
        try:
            _submit_cells(cells, run_id=run_id, **submit_kwargs)
        except Exception as exc:  # a rejected unit must not take the whole queue down
            # sbatch rejects on a bad resource request, which the next tick would hit again.
            logger.error(f"submit failed for {job_name}, not retrying: {exc}")
        done.update(cells)  # submitted is done: a cell goes out once, whatever becomes of it
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
    once: Annotated[bool, typer.Option(help="Do a single pass and exit instead of looping.")] = False,
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
    """Submit the grid's outstanding cells, by default drip-fed under the in-flight job cap."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(config.read_text())
    cfg["register"] = register

    # Fail fast on the submit node: a dry run only prints scripts, but a real submit needs
    # the shared venv the compute nodes will source.
    if not dry_run:
        _check_env(cfg)

    units = _build_units(dataset, suite, methods, extractors)
    total_cells = sum(len(u.cells) for u in units)
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
    # Doneness is decided once, here: the cells that already finished. From then on a cell is
    # marked done the moment it is submitted, so each goes out exactly once and a failing one
    # is never retried.
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
        if once:
            logger.info(f"single pass done; {remaining} unit(s) still pending.")
            return
        time.sleep(interval)


if __name__ == "__main__":
    typer.run(submit)
