"""Run an evaluation suite over a chosen dataset family (Superviz25 or Superviz26).

For each (scenario, method, extractor) cell: fit on a 90% slice of the
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
import os
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from mlops_sqldetect.data import load_whole_sampled, split_normals
from mlops_sqldetect.datasets import FAMILIES, DatasetFamily
from mlops_sqldetect.features import EXTRACTOR_LABELS, EXTRACTORS, extractor_observes_insider
from mlops_sqldetect.metrics import compute_metrics, recall_per_attack, threshold_for_fpr
from mlops_sqldetect.model import METHOD_LABELS, AEDetector, MethodName, build_method
from mlops_sqldetect.tracking import (
    ensure_parent_run,
    log_and_register_detector,
    log_dataset_input,
    setup_mlflow,
)
from mlops_sqldetect.visualize import dump_curve_points, plot_pr_curve, plot_roc_curve

logger = logging.getLogger(__name__)
# Dedicated to writing a failed cell's traceback into its own log file. propagate=False
# keeps it off the root logger's console handler to prevent double print in stdout.
_cell_failure_logger = logging.getLogger(f"{__name__}.cell_failure")
_cell_failure_logger.propagate = False

ALL_METHODS: tuple[MethodName, ...] = ("ocsvm", "lof", "ae")

# MLflow metric keys allow alnum and ``_-./`` plus space; anything else is replaced.
_MLFLOW_KEY_RE = re.compile(r"[^0-9A-Za-z_\-./ ]+")

# Validation fraction of train samples held out to calibrate the decision threshold
VAL_FRACTION = 0.1


class Cell(NamedTuple):
    """One unit of the evaluation grid: a single (scenario, method, extractor)."""

    scenario: str
    method: str
    extractor: str


def _all_scenarios(family: DatasetFamily) -> dict[str, StrEnum]:
    """Map every scenario value the family exposes to its StrEnum member."""
    return {s.value: s for scenarios in family.suites.values() for s in scenarios}


def _validate_grid(
    dataset: str, suite: str, methods: str, extractors: str, scenario: str | None = None
) -> tuple[DatasetFamily, tuple[StrEnum, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate the requested grid and resolve it to (family, scenarios, methods, extractors).

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
    requested_methods = tuple(p.strip() for p in methods.split(",") if p.strip())
    unknown = set(requested_methods) - set(ALL_METHODS)
    if unknown:
        raise typer.BadParameter(f"Unknown method(s): {sorted(unknown)}")
    requested_extractors = tuple(e.strip() for e in extractors.split(",") if e.strip())
    unknown_extractors = set(requested_extractors) - set(EXTRACTORS)
    if unknown_extractors:
        raise typer.BadParameter(f"Unknown extractor(s): {sorted(unknown_extractors)}")
    return family, datasets, requested_methods, requested_extractors


def parent_run_spec(family: DatasetFamily, method: str, extractor: str) -> tuple[str, dict[str, str]]:
    """Build the (name, tags) of the MLflow parent run that groups a (method, extractor)."""
    name = (
        f"{family.name.capitalize()}:"
        f"{METHOD_LABELS.get(method, method)} and "
        f"{EXTRACTOR_LABELS.get(extractor, extractor)}"
    )
    # The family is not tagged: each family has its own MLflow experiment, so the
    # parent is uniquely identified within that experiment by (method, extractor).
    tags = {
        "decision_engine": method,
        "feature_extractor": extractor,
        "run_role": "parent",
    }
    return name, tags


def enumerate_cells(dataset: str, suite: str, methods: str, extractors: str) -> list[Cell]:
    """Flatten the requested grid into ordered cells (method -> extractor -> scenario).

    The ordering mirrors the nested loops in :func:`evaluate_suite`, so it is a stable
    single source of truth for callers that fan the grid out (e.g. the SLURM submitter).
    """
    _, datasets, requested_methods, requested_extractors = _validate_grid(dataset, suite, methods, extractors)
    return [
        Cell(scenario.value, method, extractor)
        for method in requested_methods
        for extractor in requested_extractors
        for scenario in datasets
    ]


@dataclass
class ResultRow:
    """One row of the suite results table."""

    dataset: str
    scenario: str
    kind: str
    method: str
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
    recall_per_attack: str
    fit_seconds: float
    score_seconds: float
    model_path: str


def _model_filename(family: str, method: MethodName, extractor: str, scenario: StrEnum) -> str:
    # AE saves a torch checkpoint. OCSVM/LOF saves a joblib sklearn pipeline.
    suffix = "pt" if method == "ae" else "joblib"
    return f"{method}_{extractor}_{family}_{scenario.value}.{suffix}"


def _mlflow_key(name: str) -> str:
    """Sanitise an attack-technique name into a valid MLflow metric key."""
    return f"recall_{_MLFLOW_KEY_RE.sub('_', name).strip()}"


def _run_one(
    family: DatasetFamily,
    scenario: StrEnum,
    method: MethodName,
    extractor: str,
    data_root: Path,
    model_dir: Path,
    log_dir: Path,
    limit: int | None = None,
    target_fpr: float = 0.001,
    capture_insider: bool = False,
    seed: int = 7,
    track: bool = False,
    register: bool = False,
    cache: bool = True,
    cache_dir: Path | None = None,
) -> ResultRow:
    """Train + evaluate one (scenario, method, extractor) cell, optionally logging to MLflow."""
    # Tee this cell's log output to its own file under reports/ and (when tracking)
    # attach it to the child run as an artifact, so a cell's diagnostics live next to
    # its metrics and curves in MLflow. SLURM's per-array .log interleaves nothing here
    # (one cell per array task) but isn't reachable from a run; this file is.
    stem = f"{method}_{extractor}_{family.name}_{scenario.value}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stem}.log"
    log_handler = logging.FileHandler(log_path, mode="w")
    # DEBUG so the artifact uploaded to MLflow captures the fullest diagnostics.
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(log_handler)
    _cell_failure_logger.addHandler(log_handler)
    try:
        return _run_one_tracked(
            family=family,
            scenario=scenario,
            method=method,
            extractor=extractor,
            data_root=data_root,
            model_dir=model_dir,
            stem=stem,
            log_path=log_path,
            log_handler=log_handler,
            limit=limit,
            target_fpr=target_fpr,
            capture_insider=capture_insider,
            seed=seed,
            track=track,
            register=register,
            cache=cache,
            cache_dir=cache_dir,
        )
    finally:
        logging.getLogger().removeHandler(log_handler)
        _cell_failure_logger.removeHandler(log_handler)
        log_handler.close()


def _run_one_tracked(
    *,
    family: DatasetFamily,
    scenario: StrEnum,
    method: MethodName,
    extractor: str,
    data_root: Path,
    model_dir: Path,
    stem: str,
    log_path: Path,
    log_handler: logging.Handler,
    limit: int | None,
    target_fpr: float,
    capture_insider: bool,
    seed: int,
    track: bool,
    register: bool,
    cache: bool,
    cache_dir: Path | None,
) -> ResultRow:
    """Body of :func:`_run_one`, run with a file handler already capturing this cell's log."""
    logger.info(
        f"=== {METHOD_LABELS.get(method, method)} + {EXTRACTOR_LABELS.get(extractor, extractor)} "
        f"on {family.name.capitalize()}/{scenario.value} ==="
    )
    # Whether insider attacks reach the collector's observation point is a property of the
    # collector, declared per extractor (see features.extractor_observes_insider): GAUR
    # instruments the parser and observes insider traffic, external collectors do not. The
    # CLI --capture-insider stays an explicit override on top of that declared capability.
    capture_insider = capture_insider or extractor_observes_insider(extractor)
    # attack_technique is needed for per-technique recall; it is NaN on normal rows.
    if limit is not None:
        # Smoke run: sample `limit` rows from the whole file (all splits, both
        # labels) *before* splitting on `split`, rather than subsampling each
        # split independently. Split proportions follow the file's distribution.
        df_all = load_whole_sampled(
            family.resolve_path(scenario, data_root),
            columns=("full_query", "label", "split", "attack_technique"),
            limit=limit,
            seed=seed,
        )
        df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
        df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)
    else:
        df_train = family.load_split(scenario, "train", root=data_root, limit=limit)
        df_test = family.load_split(
            scenario, "test", root=data_root, columns=("full_query", "label", "attack_technique"), limit=limit
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
    file_key = f"{scenario.value}.csv"
    # superviz26 bundles its scenarios under the "main" group; superviz25 keeps a
    # flat top-level ``files`` map.
    file_entry = mf["groups"]["main"]["files"][file_key] if "groups" in mf else mf["files"][file_key]
    kind = file_entry["kind"]
    # Zenodo records publish either a sha256 (superviz26) or an md5 (superviz25).
    digest = file_entry.get("sha256") or file_entry.get("md5", "")

    # A limited run trains on a stratified subset (smoke test); an unlimited run
    # uses the full dataset. The label distinguishes the two in the MLflow UI.
    run_type = "smoke-run" if limit is not None else "full-run"

    model = build_method(method, extractor, cache=cache, cache_dir=cache_dir)
    # Child name is self-describing and time-ordered: ``{scenario}#{ts}``. The run_type
    # (full-run/smoke-run) lives in its own tag; the method/extractor context is
    # carried by the parent it nests under.
    run_name = f"{scenario.value}#{time.strftime('%Y%m%d-%H%M%S')}"
    run_ctx = mlflow.start_run(run_name=run_name, nested=True) if track else nullcontext()
    with run_ctx:
        try:
            if track:
                # ``scenario`` is logged as a param so it can serve as the X-axis when
                # comparing metrics in mlflow UI.
                mlflow.log_params(
                    {
                        **asdict(model.config),
                        "limit": limit,
                        "scenario": scenario.value,
                        "target_fpr": target_fpr,
                        "capture_insider": capture_insider,
                    }
                )
                mlflow.set_tags(
                    {
                        "decision_engine": method,
                        # dataset is the on-disk family (superviz26-lodo, superviz25, ...);
                        # scenario is the split within it (a-a, bcd-a, ...).
                        "dataset": data_root.name,
                        "feature_extractor": extractor,
                        "scenario": scenario.value,
                        "setting": kind,
                        "run_type": run_type,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                    }
                )
                # Sets the "Dataset" column for each child run. Skipped for locally-generated
                # families (e.g. superviz26-big) that have no published Zenodo source to point to.
                if family.on_zenodo:
                    log_dataset_input(
                        # superviz26 ships as one archive (no per-file URL), so record the
                        # Zenodo landing page; superviz25 still has a per-file url_pattern.
                        url=mf["record_url"] if "groups" in mf else mf["url_pattern"].format(filename=file_key),
                        # Name after the on-disk directory the CSVs come from, not the family:
                        # the in-domain/LODO superviz26 files live under superviz26-lodo/.
                        name=f"{data_root.name}-{scenario.value}",
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
            m = compute_metrics(
                df_test["label"], scores, threshold, f"{method}_{extractor}_{family.name}_{scenario.value}"
            )
            preds = (scores > threshold).astype(int)
            rpa = recall_per_attack(df_test["label"], preds, df_test["attack_technique"])

            model_path = model_dir / _model_filename(family.name, method, extractor, scenario)
            model.save(model_path)

            # One ROC and one AUPRC curve per cell, written under models/curves/.
            labels = df_test["label"].to_numpy()
            curve_stem = stem
            roc_path = plot_roc_curve(labels, scores, scenario.value, model_dir / "curves" / f"{curve_stem}_roc.png")
            pr_path = plot_pr_curve(labels, scores, scenario.value, model_dir / "curves" / f"{curve_stem}_auprc.png")
            # Persist the raw curve points so curves can be re-plotted offline without refitting.
            roc_csv, pr_csv = dump_curve_points(labels, scores, model_dir / "curves", curve_stem)

            row = ResultRow(
                dataset=family.name,
                scenario=scenario.value,
                kind=kind,
                method=method,
                extractor=extractor,
                n_train=int(len(df_fit)),
                n_test=int(len(df_test)),
                n_attacks=int(df_test["label"].sum()),
                roc_auc=m["rocauc"],
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
                    registered_name = f"sqldetect-{method}-{extractor}-{family.name}-{scenario.value}"
                    log_and_register_detector(model_path, registered_name, df_test[["full_query"]].head(3))

            return row
        except Exception:
            _cell_failure_logger.exception(f"Cell {stem} failed")
            raise
        finally:
            # Attach the captured per-cell log even if the cell raised: a failed run
            # should still carry its diagnostics in MLflow.
            if track:
                try:
                    log_handler.flush()
                    mlflow.log_artifact(str(log_path), artifact_path="logs")
                except Exception as exc:
                    logger.warning(f"Could not upload log artifact for {stem}: {exc}")


def evaluate_suite(
    dataset: Annotated[str, typer.Option(help="Dataset family: superviz26 or superviz25.")] = "superviz26",
    suite: Annotated[str, typer.Option(help="Suite name; depends on the dataset (e.g. in_domain, lodo, all).")] = "all",
    scenario: Annotated[
        str | None,
        typer.Option(help="Run a single scenario (e.g. a-a, bcd-a); overrides --suite. Used by the SLURM runner."),
    ] = None,
    methods: Annotated[str, typer.Option(help="Comma-separated decision-head names.")] = "ocsvm,ae",
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
    # Root/console stay INFO (quiet stdout, no third-party debug); only our package
    # emits DEBUG, which the per-cell file handler captures for the MLflow artifact.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("mlops_sqldetect").setLevel(logging.DEBUG)
    if dataset in FAMILIES and FAMILIES[dataset].protocol != "suite":
        evaluator = {"drift": "evaluate_drift", "fsl": "evaluate_fsl"}.get(FAMILIES[dataset].protocol, "its evaluator")
        raise typer.BadParameter(
            f"{dataset!r} uses the {FAMILIES[dataset].protocol!r} protocol; "
            f"run it through `python -m mlops_sqldetect.{evaluator}`."
        )
    family, datasets, requested_methods, requested_extractors = _validate_grid(
        dataset, suite, methods, extractors, scenario
    )

    data_root = data_root or family.default_root()
    if report is None:
        # A single-scenario run writes its own per-cell file (race-free under SLURM and
        # idempotent on rerun); a full-grid run accumulates into one combined CSV.
        report = (
            Path(f"reports/{dataset}/cells/{requested_methods[0]}_{requested_extractors[0]}_{scenario}.csv")
            if scenario is not None
            else Path(f"reports/{dataset}_results.csv")
        )
    # Per-cell logs live under reports/{dataset}/logs/ (next to the per-cell CSVs) and
    # are also uploaded to each child run as an MLflow artifact.
    log_dir = Path(f"reports/{dataset}/logs")
    model_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    if scenario is not None:
        report.unlink(missing_ok=True)
    write_header = not report.exists()

    track_enabled = track and setup_mlflow(dataset)

    rows: list[ResultRow] = []

    # This code is highly shaped around logging to mlflow. It will work without
    # logging, but this explains most of its structure.
    for method in requested_methods:
        for extractor in requested_extractors:
            parent_name, parent_tags = parent_run_spec(family, method, extractor)
            if track_enabled:
                parent_ctx = mlflow.start_run(run_id=ensure_parent_run(parent_tags, parent_name))
            else:
                parent_ctx = nullcontext()
            with parent_ctx:
                for scenario in datasets:
                    row = _run_one(
                        family,
                        scenario,
                        method,  # type: ignore[arg-type]
                        extractor,
                        data_root,
                        model_dir,
                        log_dir,
                        limit=limit,
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
