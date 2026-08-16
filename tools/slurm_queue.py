"""Drip-feed the evaluation grid to SLURM, respecting a cap on in-flight jobs.

The cluster only accepts a limited number of jobs per user at a time, so the full grid
(all extractors x all methods x 8 scenarios) cannot be submitted in one go. This driver
splits the grid into **units** -- one ``(method, extractor)`` pair, i.e. one
:mod:`tools.slurm_submit` call over every scenario -- and every ``--interval`` seconds
submits as many pending units as the current headroom allows.

There is no state file: on each tick the driver rebuilds the picture from the cluster and
the filesystem, so it is safe to kill, resume, or run twice.

- **done** -- every per-cell CSV under ``reports/{dataset}/cells/`` already exists. With
  ``--cells-file`` the CSVs are ignored: the manifest already decides what is missing, so a
  unit counts as done once this session has submitted it.
- **in flight** -- a job named like the unit's :func:`tools.slurm_submit._job_name` is in
  ``squeue`` (queued or running).
- **pending** -- everything else; submitted oldest-first while headroom remains.

GPU cells go out under the preemptible ``runfill`` QoS by default (``--gpu-qos``), which lets
many more of them run; SLURM kills them when someone else claims the GPU, and since a killed
cell writes no CSV the next tick simply resubmits it.

Run it on the submit node under ``tmux``/``nohup`` so a dropped VPN does not kill it:

    nohup uv run python -m tools.slurm_queue \\
      --dataset superviz26 --suite all --methods ocsvm,lof,ae --extractors li,cv,sbert,codet5 \\
      > reports/slurm/queue.log 2>&1 &

Usage:
    # See what it would do, without touching the cluster:
    python -m tools.slurm_queue --extractors li,cv --methods ae --dry-run --once
    # One pass then exit (e.g. from cron):
    python -m tools.slurm_queue --extractors li,cv --methods ae --once
"""

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

import typer

from sqlad_benchmarking.evaluate_suite import Cell, enumerate_cells
from tools.slurm_submit import REPO_ROOT, SUBMIT_DIR, _job_name, submit

logger = logging.getLogger(__name__)


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


def _units_from_cells(dataset: str, suite: str, cells_file: Path) -> list[Unit]:
    """Group a JSONL manifest (e.g. from tools.mlflow_missing) into one unit per ``(method, extractor)``.

    Units hold only the manifest's cells, so a unit can cover a subset of the scenarios.
    """
    grouped: dict[tuple[str, str], list[Cell]] = defaultdict(list)
    for line in cells_file.read_text().splitlines():
        if line.strip():
            cell = Cell(**json.loads(line))
            grouped[(cell.method, cell.extractor)].append(cell)
    if not grouped:
        raise typer.BadParameter(f"{cells_file} contains no cells.")
    return [
        Unit(method, extractor, tuple(cells), _job_name(dataset, suite, cells))
        for (method, extractor), cells in grouped.items()
    ]


def _write_unit_manifest(run_id: str, unit: Unit) -> Path:
    """Stage a unit's cells so slurm_submit takes exactly them rather than re-expanding the grid."""
    path = REPO_ROOT / SUBMIT_DIR / run_id / "unit_cells.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(cell._asdict()) + "\n" for cell in unit.cells))
    return path


def _cell_csv(dataset: str, cell: Cell) -> Path:
    """Path of the per-cell CSV an array task writes; its existence is our 'done' marker."""
    return REPO_ROOT / "reports" / dataset / "cells" / f"{cell.method}_{cell.extractor}_{cell.scenario}.csv"


def _is_done(dataset: str, unit: Unit, submitted: set[str], from_cells_file: bool) -> bool:
    """A unit is done once every one of its cells has written its CSV.

    A manifest-driven queue ignores the CSVs: the manifest lists cells whose CSV may exist
    from a run that then died, so being resubmitted once in this session is the only marker.
    """
    if from_cells_file:
        return unit.job_name in submitted
    return all(_cell_csv(dataset, cell).exists() for cell in unit.cells)


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
    units: list[Unit],
    *,
    dataset: str,
    max_jobs: int,
    user: str,
    count_array_tasks: bool,
    submitted: set[str],
    from_cells_file: bool,
    **submit_kwargs,
) -> int:
    """Submit as many pending units as fit under ``max_jobs``. Returns the number of units left to do."""
    running = _squeue(user, count_array_tasks, dry_run=submit_kwargs.get("dry_run", False))
    in_flight_names = set(running)
    headroom = max_jobs - len(running)

    pending = [
        u for u in units if u.job_name not in in_flight_names and not _is_done(dataset, u, submitted, from_cells_file)
    ]
    logger.info(f"{len(running)} job(s) in flight, headroom {headroom}, {len(pending)} unit(s) pending")

    for unit in pending:
        n = len(unit.cells)
        if n > max_jobs:
            logger.warning(f"skipping {unit.job_name}: {n} cells exceeds the {max_jobs}-job cap on its own")
            continue
        if n > headroom:
            continue
        logger.info(f"submitting {unit.job_name} ({n} cells)")
        # Explicit per-unit run_id: several units can be submitted in the same tick, and the
        # auto-generated id is a whole-second timestamp they would collide on.
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{unit.method}-{unit.extractor}"
        try:
            submit(dataset=dataset, cells_file=_write_unit_manifest(run_id, unit), run_id=run_id, **submit_kwargs)
        except Exception as exc:  # a rejected unit must not take the whole queue down
            logger.error(f"submit failed for {unit.job_name}: {exc}")
            continue
        submitted.add(unit.job_name)
        headroom -= n
    return len(pending)


def run_queue(
    dataset: Annotated[str, typer.Option(help="Dataset family: superviz26 or superviz25.")] = "superviz26",
    suite: Annotated[str, typer.Option(help="Suite name (e.g. in_domain, lodo, all).")] = "all",
    methods: Annotated[str, typer.Option(help="Comma-separated decision-head names (ocsvm, lof, ae).")] = "ae",
    extractors: Annotated[str, typer.Option(help="Comma-separated feature-extractor names.")] = "li",
    cells_file: Annotated[
        Path | None,
        typer.Option(help="JSONL of cells to drip-feed instead of the full grid (e.g. from tools.mlflow_missing)."),
    ] = None,
    max_jobs: Annotated[int, typer.Option(help="Cap on jobs in flight (queued + running) at any time.")] = 24,
    interval: Annotated[int, typer.Option(help="Seconds between checks.")] = 300,
    once: Annotated[bool, typer.Option(help="Do a single pass and exit instead of looping.")] = False,
    count_array_tasks: Annotated[
        bool, typer.Option(help="Count each array task against the cap (off: count whole arrays as one job).")
    ] = True,
    gpu_section: Annotated[str, typer.Option(help="GPU resource block in configs/slurm.yaml (gpu, gpu-long).")] = "gpu",
    gpu_qos: Annotated[
        str,
        typer.Option(help="QoS for GPU jobs; 'runfill' is preemptible, so many more can run. Empty for the default."),
    ] = "runfill",
    target_fpr: Annotated[float, typer.Option(help="Target false-positive rate for the calibrated threshold.")] = 0.001,
    seed: Annotated[int, typer.Option(help="Random state for the train/validation calibration split.")] = 7,
    register: Annotated[bool, typer.Option(help="Register each fitted model in the MLflow Model Registry.")] = False,
    no_track: Annotated[bool, typer.Option(help="Disable MLflow tracking for the submitted jobs.")] = False,
    limit: Annotated[int | None, typer.Option(help="Label-stratified subset size per cell for smoke runs.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Print what would be submitted without submitting.")] = False,
) -> None:
    """Submit the grid unit by unit, keeping at most ``--max-jobs`` jobs in flight."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    user = getpass.getuser()
    from_cells_file = cells_file is not None
    units = (
        _units_from_cells(dataset, suite, cells_file)
        if cells_file is not None
        else _build_units(dataset, suite, methods, extractors)
    )
    submitted: set[str] = set()
    total_cells = sum(len(u.cells) for u in units)
    logger.info(f"{len(units)} units / {total_cells} cells; cap {max_jobs} jobs, checking every {interval}s")

    submit_kwargs = dict(
        suite=suite,
        gpu_section=gpu_section,
        gpu_qos=gpu_qos or None,
        target_fpr=target_fpr,
        seed=seed,
        register=register,
        no_track=no_track,
        limit=limit,
        dry_run=dry_run,
    )
    while True:
        remaining = _tick(
            units,
            dataset=dataset,
            max_jobs=max_jobs,
            user=user,
            count_array_tasks=count_array_tasks,
            submitted=submitted,
            from_cells_file=from_cells_file,
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
    typer.run(run_queue)
