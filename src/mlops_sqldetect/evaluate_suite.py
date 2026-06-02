"""Run the in-domain + LODO evaluation suite over Superviz26-SQL.

For each (dataset, pipeline) pair: fit on ``split == "train"`` normal samples,
calibrate a decision threshold on the train-normal scores at ``--target-fpr``,
score the test split, record the full metric suite (ROC-AUC, AUPRC, f1, accuracy,
precision, recall, achieved FPR, AUROC/AUPRC CIs, per-attack recall), and persist
the fitted artifact under ``models/``. One row per cell is appended to
``reports/superviz26_results.csv`` so the report can be inspected during a long
run, not only at the end.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
import typer
from sklearn.metrics import average_precision_score, roc_auc_score

from mlops_sqldetect.data import split_normals
from mlops_sqldetect.datasets import IN_DOMAIN, LODO, Superviz26, load_split, manifest
from mlops_sqldetect.features import EXTRACTORS
from mlops_sqldetect.metrics import compute_metrics, recall_per_attack, threshold_for_fpr
from mlops_sqldetect.model import AEDetector, PipelineName, build_pipeline
from mlops_sqldetect.tracking import log_and_register_detector, log_dataset_input, setup_mlflow

logger = logging.getLogger(__name__)

ALL_PIPELINES: tuple[PipelineName, ...] = ("ocsvm", "ae")
# MLflow metric keys allow alnum and ``_-./`` plus space; anything else is replaced.
_MLFLOW_KEY_RE = re.compile(r"[^0-9A-Za-z_\-./ ]+")
SUITES = {
    "in_domain": IN_DOMAIN,
    "lodo": LODO,
    "all": IN_DOMAIN + LODO,
}


@dataclass
class ResultRow:
    """One row of the suite results table."""

    dataset: str
    kind: str
    pipeline: str
    extractor: str
    n_train: int
    n_test: int
    n_attacks: int
    roc_auc: float
    average_precision: float
    f1: float
    accuracy: float
    precision: float
    recall: float
    fpr: float
    auprc: float
    auroc_ci: float
    auprc_ci: float
    threshold: float
    # JSON-encoded {technique: recall} keeps the CSV schema fixed despite the
    # technique set varying per dataset (rows are appended one at a time).
    recall_per_attack: str
    fit_seconds: float
    score_seconds: float
    model_path: str


def _model_filename(pipeline: PipelineName, extractor: str, dataset: Superviz26) -> str:
    suffix = "joblib" if pipeline == "ocsvm" else "pt"
    return f"{pipeline}_{extractor}_{dataset.value}.{suffix}"


def _mlflow_key(name: str) -> str:
    """Sanitise an attack-technique name into a valid MLflow metric key."""
    return f"recall_{_MLFLOW_KEY_RE.sub('_', name).strip()}"


def _run_one(
    dataset: Superviz26,
    pipeline: PipelineName,
    extractor: str,
    data_root: Path,
    model_dir: Path,
    limit: int | None = None,
    suite: str = "",
    target_fpr: float = 0.01,
    track: bool = False,
    register: bool = False,
) -> ResultRow:
    """Train + evaluate one (dataset, pipeline, extractor) cell, optionally logging to MLflow."""
    logger.info(f"=== {pipeline} + {extractor} on {dataset.value} ===")
    df_train = load_split(dataset, "train", root=data_root, limit=limit)
    # attack_technique is needed for per-technique recall; it is NaN on normal rows.
    df_test = load_split(
        dataset, "test", root=data_root, columns=("full_query", "label", "attack_technique"), limit=limit
    )
    df_train_normal = split_normals(df_train)
    logger.info(
        f"  train: {len(df_train)} rows ({len(df_train_normal)} normal); "
        f"test: {len(df_test)} rows ({int(df_test['label'].sum())} attacks)"
    )
    mf = manifest()
    file_key = f"{dataset.value}.csv"
    file_entry = mf["files"][file_key]
    kind = file_entry["kind"]

    # A limited run trains on a stratified subset (smoke test); an unlimited run
    # uses the full dataset. The label distinguishes the two in the MLflow UI.
    run_type = "smoke-run" if limit is not None else "full-run"

    model = build_pipeline(pipeline, extractor)
    run_name = f"{pipeline}-{extractor}-{dataset.value}[{run_type}]"
    run_ctx = mlflow.start_run(run_name=run_name, nested=True) if track else nullcontext()
    with run_ctx:
        if track:
            # ``scenario`` is logged as a param so it can serve as the X-axis when
            # comparing metrics in mlflow UI.
            mlflow.log_params(
                {**asdict(model.config), "limit": limit, "scenario": dataset.value, "target_fpr": target_fpr}
            )
            mlflow.set_tags(
                {
                    "pipeline": pipeline,
                    "feature_extractor": extractor,
                    "dataset": dataset.value,
                    "kind": kind,
                    "suite": suite,
                    "run_type": run_type,
                }
            )
            # This sets the "Dataset" column for each child run.
            log_dataset_input(
                url=mf["url_pattern"].format(filename=file_key),
                name=f"superviz26-{dataset.value}",
                digest=file_entry["sha256"],
                context="train+test",
            )

        # The AE exposes a per-epoch loss hook; OCSVM has no iterative training loss.
        epoch_callback = (
            (lambda epoch, loss: mlflow.log_metric("train_loss", loss, step=epoch))
            if track and isinstance(model, AEDetector)
            else None
        )

        t0 = time.perf_counter()
        if isinstance(model, AEDetector):
            model.fit(df_train_normal, epoch_callback=epoch_callback)
        else:
            model.fit(df_train_normal)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = model.score_samples(df_test)
        score_s = time.perf_counter() - t0
        
        # TODO: Split train into train, validation to compute threshold on that validation split.
        # Calibrate the decision threshold on train-normal scores at the target FPR,
        # then binarise the test scores and compute the threshold-based metrics.
        threshold = threshold_for_fpr(model.score_samples(df_train_normal), target_fpr)
        m = compute_metrics(df_test["label"], scores, threshold)
        preds = (scores > threshold).astype(int)
        rpa = recall_per_attack(df_test["label"], preds, df_test["attack_technique"])

        model_path = model_dir / _model_filename(pipeline, extractor, dataset)
        model.save(model_path)

        row = ResultRow(
            dataset=dataset.value,
            kind=kind,
            pipeline=pipeline,
            extractor=extractor,
            n_train=int(len(df_train_normal)),
            n_test=int(len(df_test)),
            n_attacks=int(df_test["label"].sum()),
            roc_auc=float(roc_auc_score(df_test["label"], scores)),
            average_precision=float(average_precision_score(df_test["label"], scores)),
            f1=m["f1"],
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            fpr=m["fpr"],
            auprc=m["auprc"],
            auroc_ci=m["auroc_ci"],
            auprc_ci=m["auprc_ci"],
            threshold=float(threshold),
            recall_per_attack=json.dumps(rpa),
            fit_seconds=round(fit_s, 3),
            score_seconds=round(score_s, 3),
            model_path=str(model_path),
        )

        if track:
            mlflow.log_params(
                {
                    "n_train": row.n_train,
                    "n_test": row.n_test,
                    "n_attacks": row.n_attacks,
                }
            )
            mlflow.log_metrics(
                {
                    "roc_auc": row.roc_auc,
                    "average_precision": row.average_precision,
                    "f1": row.f1,
                    "accuracy": row.accuracy,
                    "precision": row.precision,
                    "recall": row.recall,
                    "fpr": row.fpr,
                    "auprc": row.auprc,
                    "auroc_ci": row.auroc_ci,
                    "auprc_ci": row.auprc_ci,
                    "threshold": row.threshold,
                    "fit_seconds": row.fit_seconds,
                    "score_seconds": row.score_seconds,
                }
            )
            if rpa:
                mlflow.log_metrics({_mlflow_key(t): v for t, v in rpa.items()})
            if register:
                registered_name = f"sqldetect-{pipeline}-{extractor}-{dataset.value}"
                log_and_register_detector(model_path, registered_name, df_test[["full_query"]].head(3))

        return row


def evaluate_suite(
    suite: Annotated[str, typer.Option(help="One of: in_domain, lodo, all.")] = "all",
    pipelines: Annotated[str, typer.Option(help="Comma-separated decision-head names.")] = "ocsvm,ae",
    extractors: Annotated[str, typer.Option(help="Comma-separated feature-extractor names.")] = "li",
    data_root: Annotated[Path, typer.Option(help="Directory holding the CSVs (default: data/raw/superviz26).")] = Path(
        "data/raw/superviz26"
    ),
    model_dir: Annotated[Path, typer.Option(help="Where to save fitted models.")] = Path("models"),
    report: Annotated[Path, typer.Option(help="Output CSV for results.")] = Path("reports/superviz26_results.csv"),
    limit: Annotated[int | None, typer.Option(help="Label-stratified subset size for smoke runs.")] = None,
    target_fpr: Annotated[
        float, typer.Option(help="Target false-positive rate for the train-normal calibrated threshold.")
    ] = 0.01,
    track: Annotated[bool, typer.Option(help="Log runs to MLflow when MLFLOW_TRACKING_URI is set.")] = True,
    register: Annotated[
        bool,
        typer.Option(help="Also log each fitted model and register a new version in the MLflow Model Registry."),
    ] = False,
) -> pd.DataFrame:
    """Run the full Superviz26 evaluation grid and append rows to ``report``.

    Returns the results table as a DataFrame for programmatic use.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if suite not in SUITES:
        raise typer.BadParameter(f"--suite must be one of {sorted(SUITES)}")
    requested_pipelines = tuple(p.strip() for p in pipelines.split(",") if p.strip())
    unknown = set(requested_pipelines) - set(ALL_PIPELINES)
    if unknown:
        raise typer.BadParameter(f"Unknown pipeline(s): {sorted(unknown)}")
    requested_extractors = tuple(e.strip() for e in extractors.split(",") if e.strip())
    unknown_extractors = set(requested_extractors) - set(EXTRACTORS)
    if unknown_extractors:
        raise typer.BadParameter(f"Unknown extractor(s): {sorted(unknown_extractors)}")

    datasets = SUITES[suite]
    model_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_header = not report.exists()

    track_enabled = track and setup_mlflow()

    # if we setup a sample limit, we consider this a smoke-run
    run_type = "smoke-run" if limit is not None else "full-run"

    rows: list[ResultRow] = []

    # This code is highly shaped around logging to mlflow. It will work without
    # logging, but this explains most of its structure.
    #
    # We create for each pipeline (decision engine + feature extractor) a parent run.
    # Then, for each dataset we create a child, where we log the metrics of interest.
    for pipeline in requested_pipelines:
        for extractor in requested_extractors:
            parent_name = f"{pipeline}-{extractor}[{run_type}]"
            parent_ctx = mlflow.start_run(run_name=parent_name) if track_enabled else nullcontext()
            with parent_ctx:
                if track_enabled:
                    mlflow.set_tags(
                        {
                            "pipeline": pipeline,
                            "feature_extractor": extractor,
                            "suite": suite,
                            "run_type": run_type,
                        }
                    )
                for dataset in datasets:
                    row = _run_one(
                        dataset,
                        pipeline,  # type: ignore[arg-type]
                        extractor,
                        data_root,
                        model_dir,
                        limit=limit,
                        suite=suite,
                        target_fpr=target_fpr,
                        track=track_enabled,
                        register=register,
                    )
                    rows.append(row)
                    pd.DataFrame([asdict(row)]).to_csv(report, mode="a", header=write_header, index=False)
                    write_header = False
                    logger.info(
                        f"Results: ROC-AUC={row.roc_auc:.4f}  AUPRC={row.auprc:.4f}  F1={row.f1:.4f}  "
                        f"recall={row.recall:.4f}  FPR={row.fpr:.4f}  "
                    )

    df = pd.DataFrame([asdict(r) for r in rows])
    logger.info(f"Wrote {len(df)} rows to {report}")
    return df


if __name__ == "__main__":
    typer.run(evaluate_suite)
