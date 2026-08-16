"""Find grid cells that have no finished MLflow run, and resubmit them.

A SLURM array task that crashed, was preempted, or hung leaves either no run at all or a
run stuck in ``RUNNING``/``FAILED`` on the tracking server. This tool rebuilds the expected
grid with :func:`enumerate_cells`, asks MLflow which cells have a **FINISHED** child run,
and reports the difference. The missing cells are written to a JSONL manifest that
``tools.slurm_queue --cells-file`` drip-feeds back to SLURM under the in-flight cap
(no need to re-run whole method x extractor blocks).

Matching is by tags, per protocol:
- suite (LODO / in-domain): ``decision_engine`` + ``feature_extractor`` + ``scenario``
- drift: same, with ``domain`` instead of ``scenario``
- fsl: same, with ``target``; a cell counts as done only when every ``k`` of the sweep
  has its own finished run.

Usage:
    # Just look:
    python -m tools.mlflow_missing --dataset superviz26 --suite all \\
      --methods ocsvm,lof,ae --extractors li,cv,sbert,codet5
    # Write the manifest and drip-feed it to the cluster:
    python -m tools.mlflow_missing ... --submit
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import mlflow
import typer

from sqlad_benchmarking.datasets import FAMILIES
from sqlad_benchmarking.evaluate_fsl import DEFAULT_KS
from sqlad_benchmarking.evaluate_suite import Cell, enumerate_cells
from sqlad_benchmarking.tracking import experiment_name, setup_mlflow
from tools.slurm_queue import run_queue
from tools.slurm_submit import REPO_ROOT, SUBMIT_DIR

logger = logging.getLogger(__name__)

SCENARIO_TAG = {"suite": "scenario", "drift": "domain", "fsl": "target"}


def _protocol(dataset: str) -> str:
    return FAMILIES[dataset].protocol if dataset in FAMILIES else "suite"


def _finished_runs(dataset: str, protocol: str, run_type: str | None) -> dict[tuple[str, str, str], set[str]]:
    """Map (method, extractor, scenario) to the set of finished runs' ``k`` params.

    ``k`` is only meaningful for the few-shot sweep; other protocols get ``{""}``, i.e.
    "one finished run exists".
    """
    filters = ["attributes.status = 'FINISHED'"]
    if run_type and protocol != "fsl":
        filters.append(f"tags.run_type = '{run_type}'")
    name = experiment_name(dataset)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        raise typer.BadParameter(f"experiment {name!r} does not exist on {mlflow.get_tracking_uri()}.")
    logger.info(f"Searching experiment {name} (id {experiment.experiment_id}).")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=" and ".join(filters), output_format="list"
    )

    tag = SCENARIO_TAG[protocol]
    done: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for run in runs:
        tags = run.data.tags
        scenario = tags.get(tag)
        if not scenario:
            continue
        key = (tags.get("decision_engine", ""), tags.get("feature_extractor", ""), scenario)
        done[key].add(run.data.params.get("k", "") if protocol == "fsl" else "")
    return done


def _missing(
    cells: list[Cell], done: dict[tuple[str, str, str], set[str]], expected: set[str]
) -> list[tuple[Cell, str]]:
    """Cells with no finished run (or, for fsl, an incomplete k sweep), with a reason each."""
    out = []
    for cell in cells:
        got = done.get((cell.method, cell.extractor, cell.scenario), set())
        if not got:
            out.append((cell, "no finished run"))
        elif expected - got:
            out.append((cell, f"missing k={','.join(sorted(expected - got, key=int))}"))
    return out


def check(
    dataset: Annotated[str, typer.Option(help="Dataset family: superviz26, superviz26-drift, ...")] = "superviz26",
    suite: Annotated[str, typer.Option(help="Suite name (e.g. in_domain, lodo, all).")] = "all",
    methods: Annotated[str, typer.Option(help="Comma-separated decision-head names (ocsvm, lof, ae).")] = "ocsvm,ae",
    extractors: Annotated[str, typer.Option(help="Comma-separated registered feature-extractor names.")] = "li",
    run_type: Annotated[
        str, typer.Option(help="Only count runs with this run_type tag ('full-run', 'smoke-run', or '' for any).")
    ] = "full-run",
    ks: Annotated[str, typer.Option(help="Few-shot budgets a target must have to count as complete.")] = DEFAULT_KS,
    out: Annotated[
        Path | None, typer.Option(help="Where to write the missing-cells manifest (default: reports/slurm/missing-*).")
    ] = None,
    submit_missing: Annotated[
        bool, typer.Option("--submit", help="Drip-feed the missing cells to SLURM via tools.slurm_queue.")
    ] = False,
    max_jobs: Annotated[int, typer.Option(help="Cap on jobs in flight (queued + running) at any time.")] = 24,
    interval: Annotated[int, typer.Option(help="Seconds between queue checks.")] = 300,
    once: Annotated[bool, typer.Option(help="Single queue pass then exit, instead of looping.")] = False,
    gpu_qos: Annotated[str, typer.Option(help="QoS for the resubmitted GPU jobs. Empty for the default.")] = "runfill",
    dry_run: Annotated[bool, typer.Option(help="With --submit, print the sbatch commands without submitting.")] = False,
) -> None:
    """Report the grid cells missing a finished MLflow run; optionally resubmit them."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow(dataset):
        raise typer.BadParameter("MLFLOW_TRACKING_URI is not set; nothing to check against.")

    protocol = _protocol(dataset)
    cells = enumerate_cells(dataset, suite, methods, extractors)
    done = _finished_runs(dataset, protocol, run_type or None)
    expected = set(ks.split(",")) if protocol == "fsl" else set()
    missing = _missing(cells, done, expected)

    logger.info(f"{len(cells) - len(missing)}/{len(cells)} cells have a finished run.")
    if not missing:
        logger.info("Nothing to resubmit.")
        return

    by_unit: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cell, reason in missing:
        by_unit[(cell.method, cell.extractor)].append(f"{cell.scenario} ({reason})")
    for (method, extractor), items in sorted(by_unit.items()):
        logger.info(f"  {method}/{extractor}: {len(items)} missing -> {', '.join(sorted(items))}")

    manifest = out or REPO_ROOT / SUBMIT_DIR / f"missing-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(cell._asdict()) + "\n" for cell, _ in missing))
    logger.info(f"{len(missing)} missing cells -> {manifest}")

    if not submit_missing:
        logger.info(f"Resubmit with:\n  python -m tools.slurm_queue --dataset {dataset} --cells-file {manifest}")
        return
    run_queue(
        dataset=dataset,
        suite=suite,
        cells_file=manifest,
        max_jobs=max_jobs,
        interval=interval,
        once=once,
        gpu_qos=gpu_qos,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    typer.run(check)
