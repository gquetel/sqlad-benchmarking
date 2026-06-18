"""Run an evaluation suite over a chosen dataset family (Superviz25 or Superviz26).

For each (scenario, pipeline, extractor) cell: fit on a 90% slice of the
``split == "train"`` normal samples, calibrate a decision threshold on the
held-out 10% validation normals at ``--target-fpr``, score the test split,
record the full metric suite (ROC-AUC, AUPRC, f1, accuracy, precision, recall,
achieved FPR, AUROC/AUPRC CIs, per-attack recall), and persist the fitted
artifact under ``models/``. One row per cell is appended to
``reports/{dataset}_results.csv``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NamedTuple

import mlflow
import pandas as pd
import typer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from mlops_sqldetect.data import load_whole_sampled, split_normals
from mlops_sqldetect.datasets import FAMILIES, DatasetFamily
from mlops_sqldetect.features import EXTRACTORS
from mlops_sqldetect.metrics import compute_metrics, recall_per_attack, threshold_for_fpr
from mlops_sqldetect.model import AEDetector, PipelineName, build_pipeline
from mlops_sqldetect.tracking import (
    ensure_parent_run,
    log_and_register_detector,
    log_dataset_input,
    setup_mlflow,
)
from mlops_sqldetect.visualize import dump_curve_points, plot_pr_curve, plot_roc_curve

logger = logging.getLogger(__name__)

ALL_PIPELINES: tuple[PipelineName, ...] = ("ocsvm", "lof", "ae")

# Human-readable labels used to build MLflow parent run names. Acronyms stay
# uppercase; extractor labels mirror their reference works (Li et al., SecureBERT).
PIPELINE_LABELS: dict[str, str] = {"ocsvm": "OCSVM", "lof": "LOF", "ae": "Autoencoder"}
EXTRACTOR_LABELS: dict[str, str] = {"li": "Li", "cv": "CountVectorizer", "sbert": "SecureBERT"}

# MLflow metric keys allow alnum and ``_-./`` plus space; anything else is replaced.
_MLFLOW_KEY_RE = re.compile(r"[^0-9A-Za-z_\-./ ]+")

# Validation fraction of train samples held out to calibrate the decision threshold
VAL_FRACTION = 0.1


class Cell(NamedTuple):
    """One unit of the evaluation grid: a single (scenario, pipeline, extractor)."""

    scenario: str
    pipeline: str
    extractor: str


def _all_scenarios(family: DatasetFamily) -> dict[str, StrEnum]:
    """Map every scenario value the family exposes to its StrEnum member."""
    return {s.value: s for scenarios in family.suites.values() for s in scenarios}


def _validate_grid(
    dataset: str, suite: str, pipelines: str, extractors: str, scenario: str | None = None
) -> tuple[DatasetFamily, tuple[StrEnum, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate the requested grid and resolve it to (family, scenarios, pipelines, extractors).

    When ``scenario`` is given it overrides ``suite`` and selects that single scenario;
    otherwise the named suite is expanded. Raises ``typer.BadParameter`` on any unknown name.
    """
    if dataset not in FAMILIES:
        raise typer.BadParameter(f"--dataset must be one of {sorted(FAMILIES)}")
    family = FAMILIES[dataset]
    if scenario is not None:
        scenarios = _all_scenarios(family)
        if scenario not in scenarios:
            raise typer.BadParameter(f"--scenario for {dataset} must be one of {sorted(scenarios)}")
        datasets: tuple[StrEnum, ...] = (scenarios[scenario],)
    else:
        if suite not in family.suites:
            raise typer.BadParameter(f"--suite for {dataset} must be one of {sorted(family.suites)}")
        datasets = family.suites[suite]
    requested_pipelines = tuple(p.strip() for p in pipelines.split(",") if p.strip())
    unknown = set(requested_pipelines) - set(ALL_PIPELINES)
    if unknown:
        raise typer.BadParameter(f"Unknown pipeline(s): {sorted(unknown)}")
    requested_extractors = tuple(e.strip() for e in extractors.split(",") if e.strip())
    unknown_extractors = set(requested_extractors) - set(EXTRACTORS)
    if unknown_extractors:
        raise typer.BadParameter(f"Unknown extractor(s): {sorted(unknown_extractors)}")
    return family, datasets, requested_pipelines, requested_extractors


def parent_run_spec(family: DatasetFamily, pipeline: str, extractor: str) -> tuple[str, dict[str, str]]:
    """Build the (name, tags) of the MLflow parent run that groups a (pipeline, extractor)."""
    name = (
        f"{family.name.capitalize()}:"
        f"{PIPELINE_LABELS.get(pipeline, pipeline)} and "
        f"{EXTRACTOR_LABELS.get(extractor, extractor)}"
    )
    tags = {
        "pipeline": pipeline,
        "feature_extractor": extractor,
        "dataset_family": family.name,
        "run_role": "parent",
    }
    return name, tags


def enumerate_cells(dataset: str, suite: str, pipelines: str, extractors: str) -> list[Cell]:
    """Flatten the requested grid into ordered cells (pipeline -> extractor -> scenario).

    The ordering mirrors the nested loops in :func:`evaluate_suite`, so it is a stable
    single source of truth for callers that fan the grid out (e.g. the SLURM submitter).
    """
    _, datasets, requested_pipelines, requested_extractors = _validate_grid(dataset, suite, pipelines, extractors)
    return [
        Cell(scenario.value, pipeline, extractor)
        for pipeline in requested_pipelines
        for extractor in requested_extractors
        for scenario in datasets
    ]


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


def _model_filename(family: str, pipeline: PipelineName, extractor: str, dataset: StrEnum) -> str:
    # AE persists a torch checkpoint; OCSVM/LOF persist a joblib sklearn pipeline.
    suffix = "pt" if pipeline == "ae" else "joblib"
    return f"{pipeline}_{extractor}_{family}_{dataset.value}.{suffix}"


def _mlflow_key(name: str) -> str:
    """Sanitise an attack-technique name into a valid MLflow metric key."""
    return f"recall_{_MLFLOW_KEY_RE.sub('_', name).strip()}"


def _run_one(
    family: DatasetFamily,
    dataset: StrEnum,
    pipeline: PipelineName,
    extractor: str,
    data_root: Path,
    model_dir: Path,
    limit: int | None = None,
    suite: str = "",
    target_fpr: float = 0.001,
    capture_insider: bool = False,
    seed: int = 7,
    track: bool = False,
    register: bool = False,
    cache: bool = True,
    cache_dir: Path | None = None,
) -> ResultRow:
    """Train + evaluate one (dataset, pipeline, extractor) cell, optionally logging to MLflow."""
    logger.info(f"=== {pipeline} + {extractor} on {family.name}/{dataset.value} ===")
    # attack_technique is needed for per-technique recall; it is NaN on normal rows.
    if limit is not None:
        # Smoke run: sample `limit` rows from the whole file (all splits, both
        # labels) *before* splitting on `split`, rather than subsampling each
        # split independently. Split proportions follow the file's distribution.
        df_all = load_whole_sampled(
            family.resolve_path(dataset, data_root),
            columns=("full_query", "label", "split", "attack_technique"),
            limit=limit,
            seed=seed,
        )
        df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
        df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)
    else:
        df_train = family.load_split(dataset, "train", root=data_root, limit=limit)
        df_test = family.load_split(
            dataset, "test", root=data_root, columns=("full_query", "label", "attack_technique"), limit=limit
        )
    df_train_normal = split_normals(df_train)
    # Hold out a validation slice of the train normals to calibrate the threshold
    # out-of-sample. The model is fitted on df_fit only; df_val never informs the fit.
    df_fit, df_val = train_test_split(df_train_normal, test_size=VAL_FRACTION, random_state=seed)
    logger.info(
        f"  train: {len(df_train)} rows ({len(df_train_normal)} normal "
        f"= {len(df_fit)} fit + {len(df_val)} val); "
        f"test: {len(df_test)} rows ({int(df_test['label'].sum())} attacks)"
    )
    mf = family.manifest()
    file_key = f"{dataset.value}.csv"
    file_entry = mf["files"][file_key]
    kind = file_entry["kind"]
    # Zenodo records publish either a sha256 (superviz26) or an md5 (superviz25).
    digest = file_entry.get("sha256") or file_entry.get("md5", "")

    # A limited run trains on a stratified subset (smoke test); an unlimited run
    # uses the full dataset. The label distinguishes the two in the MLflow UI.
    run_type = "smoke-run" if limit is not None else "full-run"

    model = build_pipeline(pipeline, extractor, cache=cache, cache_dir=cache_dir)
    # Child name is self-describing and time-ordered: ``{dataset}@{run_type}#{ts}``.
    # The pipeline/extractor context is carried by the parent it nests under.
    run_name = f"{dataset.value}@{run_type}#{time.strftime('%Y%m%d-%H%M%S')}"
    run_ctx = mlflow.start_run(run_name=run_name, nested=True) if track else nullcontext()
    with run_ctx:
        if track:
            # ``scenario`` is logged as a param so it can serve as the X-axis when
            # comparing metrics in mlflow UI.
            mlflow.log_params(
                {
                    **asdict(model.config),
                    "limit": limit,
                    "scenario": dataset.value,
                    "target_fpr": target_fpr,
                    "capture_insider": capture_insider,
                }
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
                name=f"{family.name}-{dataset.value}",
                digest=digest,
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
            model.fit(df_fit, epoch_callback=epoch_callback)
        else:
            model.fit(df_fit)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = model.score_samples(df_test)
        score_s = time.perf_counter() - t0

        # "insider" attacks are unobservable in some data-collection setups; unless
        # capture_insider is set, force them to false negatives to model that blind spot.
        if not capture_insider and "attack_technique" in df_test.columns:
            insider_mask = (df_test["attack_technique"] == "insider").to_numpy()
            if insider_mask.any():
                scores[insider_mask] = scores.min()
                logger.info(f"  forced {int(insider_mask.sum())} insider samples to false negatives")

        # Calibrate the decision threshold on the held-out validation normals at the
        # target FPR (out-of-sample), then binarise the test scores for the metrics.
        threshold = threshold_for_fpr(model.score_samples(df_val), target_fpr)
        m = compute_metrics(df_test["label"], scores, threshold, f"{pipeline}_{extractor}_{family.name}_{dataset.value}")
        preds = (scores > threshold).astype(int)
        rpa = recall_per_attack(df_test["label"], preds, df_test["attack_technique"])

        model_path = model_dir / _model_filename(family.name, pipeline, extractor, dataset)
        model.save(model_path)

        # One ROC and one AUPRC curve per cell, written under models/curves/.
        labels = df_test["label"].to_numpy()
        curve_stem = f"{pipeline}_{extractor}_{family.name}_{dataset.value}"
        roc_path = plot_roc_curve(labels, scores, dataset.value, model_dir / "curves" / f"{curve_stem}_roc.png")
        pr_path = plot_pr_curve(labels, scores, dataset.value, model_dir / "curves" / f"{curve_stem}_auprc.png")
        # Persist the raw curve points so curves can be re-plotted offline without refitting.
        roc_csv, pr_csv = dump_curve_points(labels, scores, model_dir / "curves", curve_stem)

        row = ResultRow(
            dataset=dataset.value,
            kind=kind,
            pipeline=pipeline,
            extractor=extractor,
            n_train=int(len(df_fit)),
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
            mlflow.log_artifact(str(roc_path), artifact_path="roc_curves")
            mlflow.log_artifact(str(pr_path), artifact_path="pr_curves")
            mlflow.log_artifact(str(roc_csv), artifact_path="curve_data")
            mlflow.log_artifact(str(pr_csv), artifact_path="curve_data")
            if register:
                registered_name = f"sqldetect-{pipeline}-{extractor}-{family.name}-{dataset.value}"
                log_and_register_detector(model_path, registered_name, df_test[["full_query"]].head(3))

        return row


def evaluate_suite(
    dataset: Annotated[str, typer.Option(help="Dataset family: superviz26 or superviz25.")] = "superviz26",
    suite: Annotated[str, typer.Option(help="Suite name; depends on the dataset (e.g. in_domain, lodo, all).")] = "all",
    scenario: Annotated[
        str | None,
        typer.Option(help="Run a single scenario (e.g. a-a, bcd-a); overrides --suite. Used by the SLURM runner."),
    ] = None,
    pipelines: Annotated[str, typer.Option(help="Comma-separated decision-head names.")] = "ocsvm,ae",
    extractors: Annotated[str, typer.Option(help="Comma-separated feature-extractor names.")] = "li",
    data_root: Annotated[
        Path | None, typer.Option(help="Directory holding the CSVs (default: the family's data dir).")
    ] = None,
    model_dir: Annotated[Path, typer.Option(help="Where to save fitted models.")] = Path("models"),
    report: Annotated[
        Path | None, typer.Option(help="Output CSV for results (default: reports/{dataset}_results.csv).")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Label-stratified subset size for smoke runs.")] = None,
    target_fpr: Annotated[
        float, typer.Option(help="Target false-positive rate for the validation-calibrated threshold.")
    ] = 0.001,
    capture_insider: Annotated[
        bool,
        typer.Option(help="Treat 'insider' attacks as observable instead of forcing them to false negatives."),
    ] = False,
    seed: Annotated[int, typer.Option(help="Random state for the train/validation calibration split.")] = 7,
    track: Annotated[bool, typer.Option(help="Log runs to MLflow when MLFLOW_TRACKING_URI is set.")] = True,
    register: Annotated[
        bool,
        typer.Option(help="Also log each fitted model and register a new version in the MLflow Model Registry."),
    ] = False,
    cache: Annotated[
        bool, typer.Option(help="Cache extractor features to disk so identical splits are re-used across heads.")
    ] = True,
    cache_dir: Annotated[
        Path | None, typer.Option(help="Feature cache directory (default: $SQLDETECT_CACHE_DIR or data/processed).")
    ] = None,
) -> pd.DataFrame:
    """Run the chosen dataset's evaluation grid and append rows to ``report``.

    Returns the results table as a DataFrame for programmatic use.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    family, datasets, requested_pipelines, requested_extractors = _validate_grid(
        dataset, suite, pipelines, extractors, scenario
    )

    data_root = data_root or family.default_root()
    if report is None:
        # A single-scenario run writes its own per-cell file (race-free under SLURM and
        # idempotent on rerun); a full-grid run accumulates into one combined CSV.
        report = (
            Path(f"reports/{dataset}/cells/{requested_pipelines[0]}_{requested_extractors[0]}_{scenario}.csv")
            if scenario is not None
            else Path(f"reports/{dataset}_results.csv")
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    if scenario is not None:
        report.unlink(missing_ok=True)
    write_header = not report.exists()

    track_enabled = track and setup_mlflow(dataset)

    rows: list[ResultRow] = []

    # This code is highly shaped around logging to mlflow. It will work without
    # logging, but this explains most of its structure.
    for pipeline in requested_pipelines:
        for extractor in requested_extractors:
            parent_name, parent_tags = parent_run_spec(family, pipeline, extractor)
            if track_enabled:
                parent_ctx = mlflow.start_run(run_id=ensure_parent_run(parent_tags, parent_name))
            else:
                parent_ctx = nullcontext()
            with parent_ctx:
                for scenario in datasets:
                    row = _run_one(
                        family,
                        scenario,
                        pipeline,  # type: ignore[arg-type]
                        extractor,
                        data_root,
                        model_dir,
                        limit=limit,
                        suite=suite,
                        target_fpr=target_fpr,
                        capture_insider=capture_insider,
                        seed=seed,
                        track=track_enabled,
                        register=register,
                        cache=cache,
                        cache_dir=cache_dir,
                    )
                    rows.append(row)
                    pd.DataFrame([asdict(row)]).to_csv(report, mode="a", header=write_header, index=False)
                    write_header = False

    df = pd.DataFrame([asdict(r) for r in rows])
    logger.info(f"Wrote {len(df)} rows to {report}")
    return df


if __name__ == "__main__":
    typer.run(evaluate_suite)
