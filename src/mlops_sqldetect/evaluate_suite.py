"""Run the in-domain + LODO evaluation suite over Superviz26-SQL.

For each (dataset, pipeline) pair: fit on ``split == "train"`` normal samples,
score the test split, record ROC-AUC and average precision, and persist the
fitted artifact under ``models/``. One row per cell is appended to
``reports/superviz26_results.csv`` so the report can be inspected during a long
run, not only at the end.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from sklearn.metrics import average_precision_score, roc_auc_score

from mlops_sqldetect.data import split_normals
from mlops_sqldetect.datasets import IN_DOMAIN, LODO, Superviz26, load_split, manifest
from mlops_sqldetect.model import PipelineName, build_pipeline

logger = logging.getLogger(__name__)

ALL_PIPELINES: tuple[PipelineName, ...] = ("ocsvm", "ae")
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
    n_train: int
    n_test: int
    n_attacks: int
    roc_auc: float
    average_precision: float
    fit_seconds: float
    score_seconds: float
    model_path: str


def _model_filename(pipeline: PipelineName, dataset: Superviz26) -> str:
    suffix = "joblib" if pipeline == "ocsvm" else "pt"
    return f"{pipeline}_li_{dataset.value}.{suffix}"


def _run_one(
    dataset: Superviz26,
    pipeline: PipelineName,
    data_root: Path,
    model_dir: Path,
    limit: int | None = None,
) -> ResultRow:
    """Train + evaluate one (dataset, pipeline) cell."""
    logger.info(f"=== {pipeline} on {dataset.value} ===")
    df_train = load_split(dataset, "train", root=data_root, limit=limit)
    df_test = load_split(dataset, "test", root=data_root, limit=limit)
    df_train_normal = split_normals(df_train)
    logger.info(
        f"  train: {len(df_train)} rows ({len(df_train_normal)} normal); "
        f"test: {len(df_test)} rows ({int(df_test['label'].sum())} attacks)"
    )

    model = build_pipeline(pipeline)
    t0 = time.perf_counter()
    model.fit(df_train_normal)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = model.score_samples(df_test)
    score_s = time.perf_counter() - t0

    model_path = model_dir / _model_filename(pipeline, dataset)
    model.save(model_path)

    return ResultRow(
        dataset=dataset.value,
        kind=manifest()["files"][f"{dataset.value}.csv"]["kind"],
        pipeline=pipeline,
        n_train=int(len(df_train_normal)),
        n_test=int(len(df_test)),
        n_attacks=int(df_test["label"].sum()),
        roc_auc=float(roc_auc_score(df_test["label"], scores)),
        average_precision=float(average_precision_score(df_test["label"], scores)),
        fit_seconds=round(fit_s, 3),
        score_seconds=round(score_s, 3),
        model_path=str(model_path),
    )


def evaluate_suite(
    suite: Annotated[str, typer.Option(help="One of: in_domain, lodo, all.")] = "all",
    pipelines: Annotated[str, typer.Option(help="Comma-separated pipeline names.")] = "ocsvm,ae",
    data_root: Annotated[Path, typer.Option(help="Directory holding the CSVs (default: data/raw/superviz26).")] = Path("data/raw/superviz26"),
    model_dir: Annotated[Path, typer.Option(help="Where to save fitted models.")] = Path("models"),
    report: Annotated[Path, typer.Option(help="Output CSV for results.")] = Path("reports/superviz26_results.csv"),
    limit: Annotated[int | None, typer.Option(help="Label-stratified subset size for smoke runs.")] = None,
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

    datasets = SUITES[suite]
    model_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_header = not report.exists()

    rows: list[ResultRow] = []
    for dataset in datasets:
        for pipeline in requested_pipelines:
            row = _run_one(dataset, pipeline, data_root, model_dir, limit=limit)  # type: ignore[arg-type]
            rows.append(row)
            pd.DataFrame([asdict(row)]).to_csv(
                report, mode="a", header=write_header, index=False
            )
            write_header = False
            logger.info(
                f"  -> ROC-AUC={row.roc_auc:.4f}  AP={row.average_precision:.4f}  "
                f"fit={row.fit_seconds}s  score={row.score_seconds}s"
            )

    df = pd.DataFrame([asdict(r) for r in rows])
    logger.info(f"Wrote {len(df)} rows to {report}")
    return df


if __name__ == "__main__":
    typer.run(evaluate_suite)
