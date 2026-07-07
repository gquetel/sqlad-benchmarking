r"""Overlay ROC/PR curves across the (pipeline, extractor) grid into combined figures.

Each suite cell uploads its raw curve points to MLflow as ``curve_data`` artifacts
(see :func:`evaluate_suite._run_one`). This command fetches those CSVs for one
dataset scenario, then renders a single combined ROC figure and a single combined
precision-recall figure — colour encodes the extractor, line style the pipeline.
Each is emitted both as a plotly PNG/PDF (``\includegraphics``) and as a native
pgfplots ``tikzpicture`` .tex the thesis can ``\input`` directly.

No refitting is needed: only the persisted curve points are read.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
import typer

from mlops_sqldetect.datasets import FAMILIES
from mlops_sqldetect.evaluate_suite import ALL_PIPELINES, _all_scenarios
from mlops_sqldetect.features import EXTRACTORS
from mlops_sqldetect.tracking import setup_mlflow
from mlops_sqldetect.visualize import (
    Curve,
    plot_combined_pr,
    plot_combined_pr_tikz,
    plot_combined_roc,
    plot_combined_roc_tikz,
)

logger = logging.getLogger(__name__)


def _find_cell_run_id(pipeline: str, extractor: str, scenario: str) -> str | None:
    """Latest full-run child run for a (pipeline, extractor, scenario), or None.

    The decision head is `decision_engine` on current runs and `pipeline` on legacy
    ones. MLflow filter strings are AND-only (no OR), so we can't match both tags in a
    single query — we search once per tag and keep whichever run started most recently.
    """
    common = f"tags.feature_extractor = '{extractor}' and tags.scenario = '{scenario}' and tags.run_type = 'full-run'"
    best = None
    for engine_tag in ("decision_engine", "pipeline"):
        runs = mlflow.search_runs(
            filter_string=f"tags.{engine_tag} = '{pipeline}' and {common}",
            order_by=["start_time DESC"],
            max_results=1,
            output_format="list",
        )
        if runs and (best is None or runs[0].info.start_time > best.info.start_time):
            best = runs[0]
    return best.info.run_id if best else None


def _fetch_curve(run_id: str, stem: str, pipeline: str, extractor: str, cache_dir: Path) -> Curve | None:
    """Download a cell's ``_roc.csv``/``_auprc.csv`` from MLflow into ``cache_dir``."""
    paths: dict[str, Path] = {}
    for suffix in ("roc", "auprc"):
        artifact = f"curve_data/{stem}_{suffix}.csv"
        try:
            local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact, dst_path=str(cache_dir))
        except Exception as exc:  # noqa: BLE001 - missing/partial artifacts shouldn't abort the grid
            logger.warning(f"  missing artifact {artifact} on run {run_id}: {exc}")
            return None
        paths[suffix] = Path(local)
    return Curve(pipeline=pipeline, extractor=extractor, roc=pd.read_csv(paths["roc"]), pr=pd.read_csv(paths["auprc"]))


def _load_cell(pipeline: str, extractor: str, scenario: str, stem_prefix: str, cache_dir: Path) -> Curve | None:
    """Resolve one grid cell's run and load its curve (None if the run or its artifacts are absent)."""
    run_id = _find_cell_run_id(pipeline, extractor, scenario)
    if run_id is None:
        logger.warning(f"  no run for {pipeline}+{extractor} on {stem_prefix}/{scenario}; skipping")
        return None
    stem = f"{pipeline}_{extractor}_{stem_prefix}_{scenario}"
    curve = _fetch_curve(run_id, stem, pipeline, extractor, cache_dir)
    if curve is not None:
        logger.info(f"  loaded {pipeline}+{extractor} from run {run_id}")
    return curve


def aggregate_curves(
    dataset: Annotated[str, typer.Option(help="Dataset family: superviz26 or superviz25.")] = "superviz25",
    scenario: Annotated[
        str | None, typer.Option(help="Scenario value (e.g. 'dataset', 'a-a'); defaults to the family's only/first.")
    ] = None,
    pipelines: Annotated[str, typer.Option(help="Comma-separated decision-head names.")] = "ae,lof,ocsvm",
    extractors: Annotated[str, typer.Option(help="Comma-separated feature-extractor names.")] = "li,cv,sbert",
    out_dir: Annotated[Path, typer.Option(help="Directory for the combined PNG/PDF figures.")] = Path(
        "reports/figures"
    ),
    cache_dir: Annotated[
        Path | None, typer.Option(help="Where to cache downloaded curve CSVs (default: <out_dir>/curve_data).")
    ] = None,
    max_workers: Annotated[
        int, typer.Option(help="Parallel MLflow artifact downloads (curve fetches are network-bound).")
    ] = 8,
) -> None:
    """Build combined ROC and PR figures from the MLflow-stored curve points."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if dataset not in FAMILIES:
        raise typer.BadParameter(f"--dataset must be one of {sorted(FAMILIES)}")
    family = FAMILIES[dataset]

    scenarios = _all_scenarios(family)
    if scenario is None:
        scenario = next(iter(scenarios))
    elif scenario not in scenarios:
        raise typer.BadParameter(f"--scenario for {dataset} must be one of {sorted(scenarios)}")

    requested_pipelines = tuple(p.strip() for p in pipelines.split(",") if p.strip())
    if unknown := set(requested_pipelines) - set(ALL_PIPELINES):
        raise typer.BadParameter(f"Unknown pipeline(s): {sorted(unknown)}")
    requested_extractors = tuple(e.strip() for e in extractors.split(",") if e.strip())
    if unknown := set(requested_extractors) - set(EXTRACTORS):
        raise typer.BadParameter(f"Unknown extractor(s): {sorted(unknown)}")

    if not setup_mlflow(dataset):
        logger.error("MLflow tracking unavailable (URI unset or server unreachable); cannot fetch curve artifacts.")
        raise typer.Exit(1)

    cache_dir = cache_dir or out_dir / "curve_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # The per-file tqdm bars MLflow prints interleave into noise once downloads run in
    # parallel; silence them and rely on the one-line-per-cell logging below instead.
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

    # Each cell is an independent, network-bound fetch (search + up to two CSV downloads),
    # so run the grid through a thread pool. ThreadPoolExecutor.map preserves input order,
    # keeping the curve list — and thus the figure's colour/legend order — deterministic.
    grid = [(p, e) for p in requested_pipelines for e in requested_extractors]
    workers = max(1, min(max_workers, len(grid)))
    logger.info(f"Fetching {len(grid)} cells with {workers} worker(s)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda pe: _load_cell(pe[0], pe[1], scenario, family.name, cache_dir), grid)
    curves = [curve for curve in results if curve is not None]

    if not curves:
        logger.error("No curves could be loaded; nothing to plot.")
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{family.name}_{scenario}_combined"
    # At the lowest threshold every sample is predicted positive (recall=1), so the
    # precision there equals the test-set prevalence — the PR random-classifier baseline.
    pr0 = curves[0].pr
    prevalence = float(pr0.loc[pr0["recall"].idxmax(), "precision"])
    roc_path = plot_combined_roc(curves, out_dir / f"{base}_roc.png")
    pr_path = plot_combined_pr(curves, out_dir / f"{base}_auprc.png", prevalence=prevalence)
    # Native pgfplots/TikZ alongside the plotly raster/PDF, for direct \input into the thesis.
    roc_tex = plot_combined_roc_tikz(curves, out_dir / f"{base}_roc.tex")
    pr_tex = plot_combined_pr_tikz(curves, out_dir / f"{base}_auprc.tex", prevalence=prevalence)
    logger.info(f"Wrote {len(curves)} curves to {roc_path} and {pr_path} (+ .pdf)")
    logger.info(f"Wrote pgfplots figures to {roc_tex} and {pr_tex}")


if __name__ == "__main__":
    typer.run(aggregate_curves)
