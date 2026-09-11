"""Run a single evaluation cell, selected from a manifest by index.

This is the body each SLURM array task executes: it reads line ``--index`` of the
JSONL manifest written by :mod:`tools.slurm_submit` and runs that one
``(scenario, method, extractor)`` cell through :func:`evaluate_suite`, which writes
its own per-cell CSV under ``reports/{dataset}/cells/``.

Usage:
    python -m tools.slurm_run_cell --manifest <path> --index $SLURM_ARRAY_TASK_ID --dataset superviz26
"""

from __future__ import annotations

import json
import logging
import signal
from pathlib import Path
from typing import Annotated

import typer

from sqlad_benchmarking.datasets import FAMILIES
from sqlad_benchmarking.evaluate_drift import evaluate_drift
from sqlad_benchmarking.evaluate_fsl import evaluate_fsl
from sqlad_benchmarking.evaluate_suite import evaluate_suite
from sqlad_benchmarking.visualize import image_export

logger = logging.getLogger(__name__)


class SlurmPreempted(RuntimeError):
    """Raised when SLURM sends SIGTERM to an active cell."""


def _handle_sigterm(signum, frame) -> None:  # noqa: ARG001
    raise SlurmPreempted("received SIGTERM (SLURM preemption or time limit)")


def run_cell(
    manifest: Annotated[Path, typer.Option(help="JSONL manifest of cells, one per line.")],
    index: Annotated[int, typer.Option(help="0-based line index into the manifest (the SLURM array task id).")],
    dataset: Annotated[str, typer.Option(help="Dataset family: superviz26 or superviz25.")] = "superviz26",
    target_fpr: Annotated[float, typer.Option(help="Target false-positive rate for the calibrated threshold.")] = 0.001,
    seed: Annotated[int, typer.Option(help="Random state for the train/validation calibration split.")] = 7,
    register: Annotated[bool, typer.Option(help="Register the fitted model in the MLflow Model Registry.")] = False,
    track: Annotated[bool, typer.Option(help="Log the run to MLflow when MLFLOW_TRACKING_URI is set.")] = True,
    limit: Annotated[int | None, typer.Option(help="Label-stratified subset size for smoke runs.")] = None,
) -> None:
    """Run the manifest cell at ``index`` through the evaluation suite."""
    signal.signal(signal.SIGTERM, _handle_sigterm)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    if not 0 <= index < len(lines):
        raise typer.BadParameter(f"--index {index} out of range for {len(lines)} cells in {manifest}")
    cell = json.loads(lines[index])
    logger.info(f"Running cell {index}/{len(lines) - 1}: {cell}")
    with image_export():
        _dispatch(cell, dataset, target_fpr, seed, register, track, limit)


def _dispatch(
    cell: dict,
    dataset: str,
    target_fpr: float,
    seed: int,
    register: bool,
    track: bool,
    limit: int | None,
) -> None:
    """Run a cell with its dataset family's evaluator."""
    protocol = FAMILIES[dataset].protocol if dataset in FAMILIES else "suite"
    if protocol == "drift":
        evaluate_drift(
            dataset=dataset,
            scenario=cell["scenario"],
            methods=cell["method"],
            extractors=cell["extractor"],
            target_fpr=target_fpr,
            seed=seed,
            track=track,
            limit=limit,
        )
    elif protocol == "fsl":
        # The few-shot evaluator runs its own k/seed sweep, so the array's --seed and
        # --limit (calibration-split knobs) do not apply; it keeps its own defaults.
        evaluate_fsl(
            dataset=dataset,
            scenario=cell["scenario"],
            methods=cell["method"],
            extractors=cell["extractor"],
            target_fpr=target_fpr,
            track=track,
        )
    else:
        evaluate_suite(
            dataset=dataset,
            scenario=cell["scenario"],
            methods=cell["method"],
            extractors=cell["extractor"],
            target_fpr=target_fpr,
            seed=seed,
            register=register,
            track=track,
            limit=limit,
        )


if __name__ == "__main__":
    typer.run(run_cell)
