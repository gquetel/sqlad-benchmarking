"""Run the few-shot domain-adaptation protocol over the Superviz26 domains.

For each target domain we start from the autoencoder **pretrained on the LODO split**
that excludes it (e.g. for target ``a`` the ``bcd-a`` model), and ask how many
unlabelled benign target-domain samples suffice to recover in-domain performance. We
sweep the adaptation budget ``k`` over ``{0, 5, 10, 50, 100, 500, 1000, 10000}`` and,
for each ``k > 0`` and each of several seeds:

  * draw ``k`` benign samples from the target's in-domain *train* split,
  * re-load the pretrained LODO autoencoder (so every repetition restarts from the
    same pretrained weights) and fine-tune **only** its weights on those ``k``
    samples — feature extractor and scaler frozen — at the pretraining learning rate
    divided by ten,
  * recompute the decision threshold from those same ``k`` samples, and
  * score the target's in-domain *test* split (down-sampled to keep the AUROC
    tractable).

``k = 0`` is the unmodified LODO model evaluated on the target domain; with no
adaptation samples there is nothing to calibrate a threshold from, so only the
threshold-free metrics (AUROC, AUPRC) are reported for it. One row per
``(target, k, seed)`` is appended to the report; the per-seed AUROCs are averaged
downstream and compared against the in-domain table (``reports/superviz26_results.csv``),
where recovery is defined as coming within 0.01 AUROC of the in-domain score.

This protocol is autoencoder-only (the cheaply-adaptable head) and re-uses the
Superviz26 data; the SLURM runner dispatches here when the family's ``protocol`` is
``"fsl"``.
"""

from __future__ import annotations

import logging
import math
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
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

from sqlad_benchmarking.data import split_normals
from sqlad_benchmarking.datasets import FAMILIES, DatasetFamily
from sqlad_benchmarking.datasets.superviz26_fsl import Superviz26, Superviz26FSL, load_fsl, lodo_source
from sqlad_benchmarking.evaluate_suite import _model_filename, _validate_grid, parent_run_spec
from sqlad_benchmarking.features import EXTRACTOR_LABELS, extractor_observes_insider
from sqlad_benchmarking.features.cache import memory_only
from sqlad_benchmarking.metrics import threshold_for_fpr, wilson_ci
from sqlad_benchmarking.model import AEDetector
from sqlad_benchmarking.tracking import delete_running_cell_runs, ensure_parent_run, log_dataset_input, setup_mlflow

logger = logging.getLogger(__name__)

DATASET = "superviz26-fsl"

# The base family that holds the data files and the pretrained LODO checkpoints the
# few-shot models adapt from. The few-shot family only adds the adaptation protocol.
BASE_FAMILY = "superviz26"

DEFAULT_KS = "0,5,10,50,100,500,1000,10000"
DEFAULT_SEEDS = "0,1,2,3,4"
# Target test sets are large; down-sampling keeps the AUROC computation tractable.
DEFAULT_TEST_LIMIT = 50000


@dataclass
class FSLResultRow:
    """One row of the few-shot table: a single (target, k, seed) repetition."""

    dataset: str
    target: str
    method: str
    extractor: str
    lodo_source: str
    k: int
    seed: int
    n_finetune: int
    n_test: int
    n_attacks: int
    auroc: float
    auprc: float
    f1: float
    recall: float
    auroc_ci: float
    threshold: float
    finetune_seconds: float
    score_seconds: float
    base_model_path: str


def _score_with_insider_mask(model: AEDetector, df_test: pd.DataFrame, capture_insider: bool) -> np.ndarray:
    """Score ``df_test`` and, unless ``capture_insider``, force insider attacks to false negatives.

    Mirrors :mod:`evaluate_suite`/:mod:`evaluate_drift`: "insider" attacks are
    unobservable in some data-collection setups, so by default their scores are
    pinned to the minimum.
    """
    scores = model.score_samples(df_test)
    if not capture_insider and "attack_technique" in df_test.columns:
        insider_mask = (df_test["attack_technique"] == "insider").to_numpy()
        if insider_mask.any():
            scores[insider_mask] = scores.min()
            logger.info(f"  forced {int(insider_mask.sum())} insider samples to false negatives")
    return scores


def _reps(ks: tuple[int, ...], seeds: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
    """Group the sweep into one ``(k, seeds)`` entry per adaptation budget.

    ``k = 0`` is the unmodified LODO model, so it is independent of the adaptation
    seed and uses a single seed; every ``k > 0`` is repeated over all seeds. The
    seeds for one ``k`` are merged into a single MLflow run downstream.
    """
    return [(k, (seeds[0],) if k == 0 else seeds) for k in ks]


def _run_repetition(
    model: AEDetector,
    *,
    family: DatasetFamily,
    target: Superviz26FSL,
    extractor: str,
    lodo: Superviz26,
    train_normal: pd.DataFrame,
    target_test: pd.DataFrame,
    labels: np.ndarray,
    n_attacks: int,
    base_model_path: Path,
    k: int,
    seed: int,
    lr_scale: float,
    target_fpr: float,
    capture_insider: bool,
) -> FSLResultRow:
    """Adapt one freshly-loaded checkpoint for a single (k, seed) and score the target."""
    n_finetune = 0
    threshold = math.nan
    ft_seconds = 0.0
    if k > 0:
        df_k = train_normal.sample(n=min(k, len(train_normal)), random_state=seed).reset_index(drop=True)
        n_finetune = len(df_k)
        ft_lr = model.config.learning_rate * lr_scale
        t0 = time.perf_counter()
        # The k samples are unique to this (k, seed), so their features are memoised in
        # memory only: writing them would leave one cache file per repetition.
        with memory_only(model.extractor):
            model.fine_tune(df_k, learning_rate=ft_lr, seed=seed)
            ft_seconds = time.perf_counter() - t0
            # Recompute the operating point from the same k benign samples.
            threshold = threshold_for_fpr(model.score_samples(df_k), target_fpr)

    t0 = time.perf_counter()
    scores = _score_with_insider_mask(model, target_test, capture_insider)
    score_seconds = time.perf_counter() - t0

    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    if math.isnan(threshold):
        f1 = recall = math.nan
    else:
        preds = (scores > threshold).astype(int)
        f1 = float(f1_score(labels, preds, zero_division=0))
        recall = float(recall_score(labels, preds, zero_division=0))

    logger.info(f"  k={k:>5} seed={seed}: AUROC {auroc:.4f}  AUPRC {auprc:.4f}")

    return FSLResultRow(
        dataset=family.name,
        target=target.value,
        method="ae",
        extractor=extractor,
        lodo_source=lodo.value,
        k=k,
        seed=seed,
        n_finetune=n_finetune,
        n_test=int(len(target_test)),
        n_attacks=n_attacks,
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        recall=recall,
        auroc_ci=wilson_ci(auroc, len(scores)),
        threshold=float(threshold),
        finetune_seconds=round(ft_seconds, 3),
        score_seconds=round(score_seconds, 3),
        base_model_path=str(base_model_path),
    )


def _log_merged_k(rows: list[FSLResultRow], *, dataset: str, target: str, lodo: str, extractor: str, **params) -> None:
    """Log one nested run for an adaptation budget, averaging the per-seed rows over seeds."""
    k = rows[0].k
    mlflow.log_params(
        {
            "target": target,
            "lodo_source": lodo,
            "k": k,
            "seeds": ",".join(str(r.seed) for r in rows),
            "n_seeds": len(rows),
            "n_finetune": rows[0].n_finetune,
            **params,
        }
    )
    mlflow.set_tags(
        {
            "decision_engine": "ae",
            "dataset": dataset,
            "feature_extractor": extractor,
            "target": target,
            "setting": "few_shot",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        }
    )
    metrics = {
        "auroc": float(np.mean([r.auroc for r in rows])),
        "auprc": float(np.mean([r.auprc for r in rows])),
        "auroc_ci": float(np.mean([r.auroc_ci for r in rows])),
        "auroc_std": float(np.std([r.auroc for r in rows])),
        "auprc_std": float(np.std([r.auprc for r in rows])),
    }
    # k=0 has no threshold to calibrate, so its threshold-dependent metrics are NaN.
    if not math.isnan(rows[0].threshold):
        metrics |= {
            "f1": float(np.mean([r.f1 for r in rows])),
            "recall": float(np.mean([r.recall for r in rows])),
            "threshold": float(np.mean([r.threshold for r in rows])),
        }
    mlflow.log_metrics(metrics)


def _run_target(
    family: DatasetFamily,
    target: Superviz26FSL,
    extractor: str,
    data_root: Path,
    model_dir: Path,
    ks: tuple[int, ...],
    seeds: tuple[int, ...],
    *,
    test_limit: int | None,
    lr_scale: float,
    target_fpr: float,
    capture_insider: bool,
    track: bool,
) -> list[FSLResultRow]:
    """Sweep the adaptation budget for one (target, extractor) cell and return its rows."""
    logger.info(
        f"=== Few-shot adaptation: Autoencoder + {EXTRACTOR_LABELS.get(extractor, extractor)} -> {target.value} ==="
    )
    lodo = lodo_source(target)
    base_model_path = model_dir / _model_filename(BASE_FAMILY, "ae", extractor, lodo)
    if not base_model_path.exists():
        raise FileNotFoundError(
            f"pretrained LODO model {base_model_path} not found. Train it first, e.g.:\n"
            f"  python -m sqlad_benchmarking.evaluate_suite --dataset {BASE_FAMILY} "
            f"--scenario {lodo.value} --methods ae --extractors {extractor}"
        )

    if track:
        delete_running_cell_runs(
            {
                "decision_engine": "ae",
                "feature_extractor": extractor,
                "target": target.value,
            }
        )

    # One fixed test set per cell (down-sampled once), shared by the whole sweep so
    # AUROCs at different k are read on the same samples.
    target_train, target_test = load_fsl(target, root=data_root, test_limit=test_limit, seed=7)
    train_normal = split_normals(target_train)
    labels = target_test["label"].to_numpy()
    n_attacks = int(labels.sum())
    logger.info(
        f"  target train normals: {len(train_normal)}; test: {len(target_test)} ({n_attacks} attacks); "
        f"adapting from {base_model_path.name}"
    )

    rows: list[FSLResultRow] = []
    # All seeds for a given k are merged into a single nested run (their metrics are
    # averaged); the per-seed rows are still returned individually for the CSV.
    for k, k_seeds in _reps(ks, seeds):
        k_rows = [
            _run_repetition(
                # Every repetition restarts from the pretrained weights: re-loading the
                # checkpoint also restores the frozen extractor and scaler.
                AEDetector.load(base_model_path),
                family=family,
                target=target,
                extractor=extractor,
                lodo=lodo,
                train_normal=train_normal,
                target_test=target_test,
                labels=labels,
                n_attacks=n_attacks,
                base_model_path=base_model_path,
                k=k,
                seed=seed,
                lr_scale=lr_scale,
                target_fpr=target_fpr,
                capture_insider=capture_insider,
            )
            for seed in k_seeds
        ]
        rows.extend(k_rows)
        if track:
            with mlflow.start_run(run_name=f"{target.value}@k{k}#{time.strftime('%Y%m%d-%H%M%S')}", nested=True):
                _log_merged_k(
                    k_rows,
                    dataset=data_root.name,
                    target=target.value,
                    lodo=lodo.value,
                    extractor=extractor,
                    lr_scale=lr_scale,
                    target_fpr=target_fpr,
                    test_limit=test_limit,
                )
                # Sets the "Dataset" column to the specific in-domain CSV few-shot
                # adapts on: the reused Superviz26 manifest carries its sha256 under
                # the "fsl" group.
                mf = family.manifest()
                file_entry = mf["groups"]["fsl"]["files"][family.resolve_path(target).name]
                log_dataset_input(
                    url=mf["record_url"],
                    name=f"{family.name}-{target.value}",
                    digest=file_entry.get("sha256", ""),
                    context="finetune+test",
                )

    return rows


def evaluate_fsl(
    dataset: Annotated[str, typer.Option(help="Few-shot dataset family (only superviz26-fsl).")] = DATASET,
    suite: Annotated[str, typer.Option(help="Suite name (only 'all' for fsl: the four domains).")] = "all",
    scenario: Annotated[
        str | None,
        typer.Option(help="Run a single target domain (a, b, c, d); overrides --suite. Used by the SLURM runner."),
    ] = None,
    methods: Annotated[str, typer.Option(help="Decision head; few-shot adaptation is autoencoder-only.")] = "ae",
    extractors: Annotated[str, typer.Option(help="Comma-separated feature-extractor names.")] = "li",
    ks: Annotated[str, typer.Option(help="Comma-separated adaptation budgets to sweep.")] = DEFAULT_KS,
    seeds: Annotated[
        str, typer.Option(help="Comma-separated seeds; each k>0 is repeated over all of them.")
    ] = DEFAULT_SEEDS,
    data_root: Annotated[
        Path | None, typer.Option(help="Directory holding the Superviz26 CSVs (default: the family's data dir).")
    ] = None,
    model_dir: Annotated[Path, typer.Option(help="Where the pretrained LODO models live.")] = Path("models"),
    report: Annotated[
        Path | None, typer.Option(help="Output CSV (default: reports/superviz26-fsl_results.csv).")
    ] = None,
    test_limit: Annotated[
        int | None, typer.Option(help="Label-stratified cap on the target test set (default: 50000).")
    ] = DEFAULT_TEST_LIMIT,
    lr_scale: Annotated[
        float, typer.Option(help="Adaptation learning rate as a fraction of the pretraining rate.")
    ] = 0.1,
    target_fpr: Annotated[
        float, typer.Option(help="Target false-positive rate for the recalibrated threshold.")
    ] = 0.001,
    capture_insider: Annotated[
        bool, typer.Option(help="Treat 'insider' attacks as observable instead of forcing them to false negatives.")
    ] = False,
    track: Annotated[bool, typer.Option(help="Log runs to MLflow when MLFLOW_TRACKING_URI is set.")] = True,
) -> pd.DataFrame:
    """Run the few-shot adaptation sweep and append one (target, k, seed) row per repetition."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if dataset not in FAMILIES or FAMILIES[dataset].protocol != "fsl":
        raise typer.BadParameter(f"--dataset must be a few-shot family (protocol='fsl'); got {dataset!r}")
    family, targets, requested_methods, requested_extractors = _validate_grid(
        dataset, suite, methods, extractors, scenario
    )
    if set(requested_methods) != {"ae"}:
        raise typer.BadParameter("few-shot adaptation is autoencoder-only; pass --methods ae")

    ks_parsed = tuple(sorted({int(x) for x in ks.split(",") if x.strip()}))
    seeds_parsed = tuple(int(x) for x in seeds.split(",") if x.strip())
    if not ks_parsed or not seeds_parsed:
        raise typer.BadParameter("--ks and --seeds must each list at least one integer")

    data_root = data_root or family.default_root()
    if report is None:
        report = (
            Path(f"reports/{dataset}/cells/ae_{requested_extractors[0]}_{scenario}.csv")
            if scenario is not None
            else Path(f"reports/{dataset}_results.csv")
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    if scenario is not None:
        report.unlink(missing_ok=True)
    write_header = not report.exists()

    track_enabled = track and setup_mlflow(dataset)

    rows: list[FSLResultRow] = []
    for extractor in requested_extractors:
        # Insider observability is a declared property of the collector (see
        # features.extractor_observes_insider); --capture-insider is an explicit override.
        cell_capture_insider = capture_insider or extractor_observes_insider(extractor)
        parent_name, parent_tags = parent_run_spec(family, "ae", extractor)
        parent_ctx = (
            mlflow.start_run(run_id=ensure_parent_run(parent_tags, parent_name)) if track_enabled else nullcontext()
        )
        with parent_ctx:
            for target in targets:
                cell_rows = _run_target(
                    family,
                    target,  # type: ignore[arg-type]
                    extractor,
                    data_root,
                    model_dir,
                    ks_parsed,
                    seeds_parsed,
                    test_limit=test_limit,
                    lr_scale=lr_scale,
                    target_fpr=target_fpr,
                    capture_insider=cell_capture_insider,
                    track=track_enabled,
                )
                rows.extend(cell_rows)
                pd.DataFrame([asdict(r) for r in cell_rows]).to_csv(report, mode="a", header=write_header, index=False)
                write_header = False

    df = pd.DataFrame([asdict(r) for r in rows])
    logger.info(f"Wrote {len(df)} rows to {report}")
    return df


if __name__ == "__main__":
    typer.run(evaluate_fsl)
