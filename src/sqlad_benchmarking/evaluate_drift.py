"""Run the same-domain concept-drift protocol over the Superviz26 domains.

For each (domain, method, extractor) cell the detector is trained *once* on the
benign origin (S1) train rows, the decision threshold is calibrated on a held-out
slice of those normals at ``--target-fpr``, and the fitted model is then scored on
two test sets: the origin S1 test set (reference) and the held-out shifted S2 test
set (post-drift). Drift robustness is the AUROC drop ``delta = auroc_s1 - auroc_s2``.
One row per cell is appended to ``reports/superviz26-drift_results.csv`` (or a
per-cell file under ``reports/superviz26-drift/cells/`` for single-scenario runs),
and the per-domain rows are averaged downstream to obtain the per-method table.

This mirrors :mod:`sqlad_benchmarking.evaluate_suite` but evaluates two test sets per
fit instead of one; the SLURM runner dispatches here when the dataset family's
``protocol`` is ``"drift"``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import mlflow
import numpy as np
import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from sqlad_benchmarking.data import split_normals
from sqlad_benchmarking.datasets import FAMILIES, DatasetFamily
from sqlad_benchmarking.datasets.superviz26_drift import Superviz26Drift, load_drift
from sqlad_benchmarking.evaluate_suite import (
    DEFAULT_SAVE_METHODS,
    VAL_FRACTION,
    _mlflow_key,
    _model_filename,
    _validate_grid,
    parent_run_spec,
    parse_save_methods,
)
from sqlad_benchmarking.features import EXTRACTOR_LABELS, extractor_observes_insider
from sqlad_benchmarking.metrics import compute_metrics, recall_per_attack, threshold_for_fpr
from sqlad_benchmarking.model import METHOD_LABELS, AEDetector, MethodName, build_method
from sqlad_benchmarking.tracking import ensure_parent_run, log_dataset_input, setup_mlflow
from sqlad_benchmarking.visualize import dump_curve_points, plot_pr_curve, plot_roc_curve

logger = logging.getLogger(__name__)

DATASET = "superviz26-drift"


@dataclass
class DriftResultRow:
    """One row of the concept-drift results table: a single (domain, method, extractor)."""

    dataset: str
    domain: str
    method: str
    extractor: str
    n_train: int
    n_test_s1: int
    n_attacks_s1: int
    n_test_s2: int
    n_attacks_s2: int
    auroc_s1: float
    auroc_s2: float
    delta_auroc: float
    auprc_s1: float
    auprc_s2: float
    f1_s1: float
    f1_s2: float
    recall_s1: float
    recall_s2: float
    auroc_s1_ci: float
    auroc_s2_ci: float
    threshold: float
    fit_seconds: float
    score_seconds: float
    model_path: str


def _score_with_insider_mask(model, df_test: pd.DataFrame, capture_insider: bool) -> np.ndarray:
    """Score ``df_test`` and, unless ``capture_insider``, force insider attacks to false negatives.

    Mirrors :mod:`evaluate_suite`: "insider" attacks are unobservable in some
    data-collection setups, so by default their scores are pinned to the minimum.
    """
    scores = model.score_samples(df_test)
    if not capture_insider and "attack_technique" in df_test.columns:
        insider_mask = (df_test["attack_technique"] == "insider").to_numpy()
        if insider_mask.any():
            scores[insider_mask] = scores.min()
            logger.info(f"  forced {int(insider_mask.sum())} insider samples to false negatives")
    return scores


def _evaluate_one_set(
    df_test: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    *,
    model_name: str,
    curve_stem: str,
    model_dir: Path,
) -> tuple[dict[str, float], dict[str, float], list[Path]]:
    """Compute the metric suite, per-attack recall, and persist the ROC/PR curves for one test set."""
    m = compute_metrics(df_test["label"], scores, threshold, model_name)
    preds = (scores > threshold).astype(int)
    rpa = recall_per_attack(df_test["label"], preds, df_test["attack_technique"])

    labels = df_test["label"].to_numpy()
    roc_path = plot_roc_curve(labels, scores, curve_stem, model_dir / "curves" / f"{curve_stem}_roc.png")
    pr_path = plot_pr_curve(labels, scores, curve_stem, model_dir / "curves" / f"{curve_stem}_auprc.png")
    roc_csv, pr_csv = dump_curve_points(labels, scores, model_dir / "curves", curve_stem)
    return m, rpa, [roc_path, pr_path, roc_csv, pr_csv]


def _run_one(
    family: DatasetFamily,
    domain: Superviz26Drift,
    method: MethodName,
    extractor: str,
    data_root: Path,
    model_dir: Path,
    limit: int | None = None,
    target_fpr: float = 0.001,
    capture_insider: bool = False,
    seed: int = 7,
    track: bool = False,
    cache: bool = True,
    cache_dir: Path | None = None,
    save_methods: frozenset[str] = frozenset({"ae"}),
) -> DriftResultRow:
    """Train one cell on the origin (S1) normals and evaluate it on S1 and the shifted (S2) test set."""
    logger.info(
        f"=== {METHOD_LABELS.get(method, method)} + {EXTRACTOR_LABELS.get(extractor, extractor)} "
        f"on concept-drift/{domain.value} ==="
    )
    origin_train, origin_test, shifted_test = load_drift(domain, root=data_root, limit=limit, seed=seed)
    df_train_normal = split_normals(origin_train)
    # Hold out a validation slice of the train normals to calibrate the threshold
    # out-of-sample; the model is fitted on df_fit only.
    df_fit, df_val = train_test_split(df_train_normal, test_size=VAL_FRACTION, random_state=seed)
    logger.info(
        f"  train: {len(origin_train)} rows ({len(df_train_normal)} normal = {len(df_fit)} fit + {len(df_val)} val); "
        f"S1 test: {len(origin_test)} ({int(origin_test['label'].sum())} attacks); "
        f"S2 test: {len(shifted_test)} ({int(shifted_test['label'].sum())} attacks)"
    )

    run_type = "smoke-run" if limit is not None else "full-run"
    model = build_method(method, extractor, cache=cache, cache_dir=cache_dir)
    run_name = f"{domain.value}#{time.strftime('%Y%m%d-%H%M%S')}"
    run_ctx = mlflow.start_run(run_name=run_name, nested=True) if track else nullcontext()
    with run_ctx:
        if track:
            mlflow.log_params(
                {
                    **asdict(model.config),
                    "limit": limit,
                    "domain": domain.value,
                    "target_fpr": target_fpr,
                    "capture_insider": capture_insider,
                }
            )
            mlflow.set_tags(
                {
                    "decision_engine": method,
                    "dataset": data_root.name,
                    "feature_extractor": extractor,
                    "domain": domain.value,
                    "setting": "concept_drift",
                    "run_type": run_type,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                }
            )
            # Sets the "Dataset" column to the specific per-domain drift CSV: the
            # reused Superviz26 manifest carries its sha256 under the "drift" group.
            mf = family.manifest()
            file_entry = mf["groups"]["drift"]["files"][family.resolve_path(domain).name]
            log_dataset_input(
                url=mf["record_url"],
                name=f"{family.name}-{domain.value}",
                digest=file_entry.get("sha256", ""),
                context="train+test",
            )

        epoch_callback = (
            (lambda epoch, loss: mlflow.log_metric("train_loss", loss, step=epoch))
            if track and isinstance(model, AEDetector)
            else None
        )

        t0 = time.perf_counter()
        if isinstance(model, AEDetector):
            model.fit(df_fit, epoch_callback=epoch_callback)
        else:
            model.fit(df_fit)
        fit_s = time.perf_counter() - t0

        # One threshold from the held-out origin normals, shared by both evaluations:
        # the same fitted model is judged on S1 and S2, so the operating point is fixed.
        threshold = threshold_for_fpr(model.score_samples(df_val), target_fpr)

        t0 = time.perf_counter()
        scores_s1 = _score_with_insider_mask(model, origin_test, capture_insider)
        scores_s2 = _score_with_insider_mask(model, shifted_test, capture_insider)
        score_s = time.perf_counter() - t0

        stem = f"{method}_{extractor}_{family.name}_{domain.value}"
        m_s1, rpa_s1, artifacts_s1 = _evaluate_one_set(
            origin_test,
            scores_s1,
            threshold,
            model_name=f"{stem}_s1",
            curve_stem=f"{stem}_s1",
            model_dir=model_dir,
        )
        m_s2, rpa_s2, artifacts_s2 = _evaluate_one_set(
            shifted_test,
            scores_s2,
            threshold,
            model_name=f"{stem}_s2",
            curve_stem=f"{stem}_s2",
            model_dir=model_dir,
        )

        if method in save_methods:
            model_path = model_dir / _model_filename(family.name, method, extractor, domain)
            model.save(model_path)
        else:
            model_path = None
            logger.info(f"  not saving {method} model (--save-models)")

        row = DriftResultRow(
            dataset=family.name,
            domain=domain.value,
            method=method,
            extractor=extractor,
            n_train=int(len(df_fit)),
            n_test_s1=int(len(origin_test)),
            n_attacks_s1=int(origin_test["label"].sum()),
            n_test_s2=int(len(shifted_test)),
            n_attacks_s2=int(shifted_test["label"].sum()),
            auroc_s1=m_s1["rocauc"],
            auroc_s2=m_s2["rocauc"],
            delta_auroc=m_s1["rocauc"] - m_s2["rocauc"],
            auprc_s1=m_s1["auprc"],
            auprc_s2=m_s2["auprc"],
            f1_s1=m_s1["f1"],
            f1_s2=m_s2["f1"],
            recall_s1=m_s1["recall"],
            recall_s2=m_s2["recall"],
            auroc_s1_ci=m_s1["auroc_ci"],
            auroc_s2_ci=m_s2["auroc_ci"],
            threshold=float(threshold),
            fit_seconds=round(fit_s, 3),
            score_seconds=round(score_s, 3),
            model_path=str(model_path) if model_path else "",
        )

        if track:
            mlflow.log_params({"n_train": row.n_train, "n_test_s1": row.n_test_s1, "n_test_s2": row.n_test_s2})
            mlflow.log_metrics(
                {
                    "auroc_s1": row.auroc_s1,
                    "auroc_s2": row.auroc_s2,
                    "delta_auroc": row.delta_auroc,
                    "auprc_s1": row.auprc_s1,
                    "auprc_s2": row.auprc_s2,
                    "f1_s1": row.f1_s1,
                    "f1_s2": row.f1_s2,
                    "recall_s1": row.recall_s1,
                    "recall_s2": row.recall_s2,
                    "auroc_s1_ci": row.auroc_s1_ci,
                    "auroc_s2_ci": row.auroc_s2_ci,
                    "threshold": row.threshold,
                    "fit_seconds": row.fit_seconds,
                    "score_seconds": row.score_seconds,
                }
            )
            if rpa_s1:
                mlflow.log_metrics({f"{_mlflow_key(t)}_s1": v for t, v in rpa_s1.items()})
            if rpa_s2:
                mlflow.log_metrics({f"{_mlflow_key(t)}_s2": v for t, v in rpa_s2.items()})
            for artifact in artifacts_s1 + artifacts_s2:
                sub = (
                    "roc_curves"
                    if "_roc" in artifact.name
                    else "pr_curves"
                    if "_auprc" in artifact.name
                    else "curve_data"
                )
                mlflow.log_artifact(str(artifact), artifact_path=sub)

        return row


def evaluate_drift(
    dataset: Annotated[str, typer.Option(help="Drift dataset family (only superviz26-drift).")] = DATASET,
    suite: Annotated[str, typer.Option(help="Suite name (only 'all' for drift: the four domains).")] = "all",
    scenario: Annotated[
        str | None, typer.Option(help="Run a single domain (a, b, c, d); overrides --suite. Used by the SLURM runner.")
    ] = None,
    methods: Annotated[str, typer.Option(help="Comma-separated decision-head names.")] = "ocsvm,lof,ae",
    extractors: Annotated[str, typer.Option(help="Comma-separated feature-extractor names.")] = "li",
    data_root: Annotated[
        Path | None,
        typer.Option(help="Directory holding the per-domain drift CSVs (default: ~/datasets/superviz26-cd)."),
    ] = None,
    model_dir: Annotated[Path, typer.Option(help="Where to save fitted models.")] = Path("models"),
    save_models: Annotated[
        str,
        typer.Option(help="Which fitted heads to write to disk: comma-separated names, 'all' or 'none'."),
    ] = DEFAULT_SAVE_METHODS,
    report: Annotated[
        Path | None, typer.Option(help="Output CSV (default: reports/superviz26-drift_results.csv).")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Label-stratified subset size per partition for smoke runs.")
    ] = None,
    target_fpr: Annotated[float, typer.Option(help="Target false-positive rate for the calibrated threshold.")] = 0.001,
    capture_insider: Annotated[
        bool, typer.Option(help="Treat 'insider' attacks as observable instead of forcing them to false negatives.")
    ] = False,
    seed: Annotated[int, typer.Option(help="Random state for the train/validation calibration split.")] = 7,
    track: Annotated[bool, typer.Option(help="Log runs to MLflow when MLFLOW_TRACKING_URI is set.")] = True,
    cache: Annotated[
        bool, typer.Option(help="Cache extractor features to disk so identical splits are re-used.")
    ] = True,
    cache_dir: Annotated[
        Path | None, typer.Option(help="Feature cache directory (default: $SQLDETECT_CACHE_DIR or data/processed).")
    ] = None,
) -> pd.DataFrame:
    """Run the concept-drift grid and append one (domain, method, extractor) row per cell to ``report``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if dataset not in FAMILIES or FAMILIES[dataset].protocol != "drift":
        raise typer.BadParameter(f"--dataset must be a drift family (protocol='drift'); got {dataset!r}")
    family, domains, requested_methods, requested_extractors = _validate_grid(
        dataset, suite, methods, extractors, scenario
    )
    save_methods = parse_save_methods(save_models)

    data_root = data_root or family.default_root()
    if report is None:
        report = (
            Path(f"reports/{dataset}/cells/{requested_methods[0]}_{requested_extractors[0]}_{scenario}.csv")
            if scenario is not None
            else Path(f"reports/{dataset}_results.csv")
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    if scenario is not None:
        report.unlink(missing_ok=True)
    write_header = not report.exists()

    track_enabled = track and setup_mlflow(dataset)

    rows: list[DriftResultRow] = []
    for method in requested_methods:
        for extractor in requested_extractors:
            # Insider observability is a declared property of the collector (see
            # features.extractor_observes_insider); --capture-insider is an explicit override.
            cell_capture_insider = capture_insider or extractor_observes_insider(extractor)
            parent_name, parent_tags = parent_run_spec(family, method, extractor)
            parent_ctx = (
                mlflow.start_run(run_id=ensure_parent_run(parent_tags, parent_name)) if track_enabled else nullcontext()
            )
            with parent_ctx:
                for domain in domains:
                    row = _run_one(
                        family,
                        domain,  # type: ignore[arg-type]
                        method,  # type: ignore[arg-type]
                        extractor,
                        data_root,
                        model_dir,
                        limit=limit,
                        target_fpr=target_fpr,
                        capture_insider=cell_capture_insider,
                        seed=seed,
                        track=track_enabled,
                        cache=cache,
                        cache_dir=cache_dir,
                        save_methods=save_methods,
                    )
                    rows.append(row)
                    pd.DataFrame([asdict(row)]).to_csv(report, mode="a", header=write_header, index=False)
                    write_header = False

    df = pd.DataFrame([asdict(r) for r in rows])
    logger.info(f"Wrote {len(df)} rows to {report}")
    return df


if __name__ == "__main__":
    typer.run(evaluate_drift)
