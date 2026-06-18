"""Overlay ROC/PR curves across the (pipeline, extractor) grid into combined figures.

Each suite cell uploads its raw curve points to MLflow as ``curve_data`` artifacts
(see :func:`evaluate_suite._run_one`). This command fetches those CSVs for one
dataset scenario, then renders a single combined ROC figure and a single combined
precision-recall figure — colour encodes the extractor, line style the pipeline.

No refitting is needed: only the persisted curve points are read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
import typer

from mlops_sqldetect.datasets import FAMILIES
from mlops_sqldetect.evaluate_suite import ALL_PIPELINES, _all_scenarios
from mlops_sqldetect.features import EXTRACTORS
from mlops_sqldetect.tracking import setup_mlflow
from mlops_sqldetect.visualize import Curve, plot_combined_pr, plot_combined_roc

logger = logging.getLogger(__name__)


def _find_cell_run_id(pipeline: str, extractor: str, scenario: str) -> str | None:
    """Latest full-run child run for a (pipeline, extractor, scenario), or None."""
    filter_string = (
        f"tags.pipeline = '{pipeline}' and tags.feature_extractor = '{extractor}' "
        f"and tags.dataset = '{scenario}' and tags.run_type = 'full-run'"
    )
    runs = mlflow.search_runs(
        filter_string=filter_string, order_by=["start_time DESC"], max_results=1, output_format="list"
    )
    return runs[0].info.run_id if runs else None


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

    curves: list[Curve] = []
    for pipeline in requested_pipelines:
        for extractor in requested_extractors:
            run_id = _find_cell_run_id(pipeline, extractor, scenario)
            if run_id is None:
                logger.warning(f"  no run for {pipeline}+{extractor} on {dataset}/{scenario}; skipping")
                continue
            stem = f"{pipeline}_{extractor}_{family.name}_{scenario}"
            if (curve := _fetch_curve(run_id, stem, pipeline, extractor, cache_dir)) is not None:
                curves.append(curve)
                logger.info(f"  loaded {pipeline}+{extractor} from run {run_id}")

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
    logger.info(f"Wrote {len(curves)} curves to {roc_path} and {pr_path} (+ .pdf)")


if __name__ == "__main__":
    typer.run(aggregate_curves)
