"""CLI to evaluate a fitted detection method on a labelled CSV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from sklearn.metrics import average_precision_score, roc_auc_score

from mlops_sqldetect.data import load_dataset
from mlops_sqldetect.model import MethodName, load_method

logger = logging.getLogger(__name__)


def evaluate(
    dataset: Annotated[Path, typer.Argument(help="Labelled CSV (full_query,label).")],
    model_path: Annotated[Path, typer.Argument(help="Path to a fitted method.")],
    method: Annotated[MethodName, typer.Option(help="Detector head used at training time.")] = "ocsvm",
    limit: Annotated[int | None, typer.Option(help="Label-stratified subset size for smoke runs.")] = None,
) -> dict[str, float]:
    """Score ``dataset`` and report ROC-AUC and average precision."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load_dataset(dataset, limit=limit)
    model = load_method(method, model_path)
    scores = model.score_samples(df)

    metrics = {
        "roc_auc": float(roc_auc_score(df["label"], scores)),
        "average_precision": float(average_precision_score(df["label"], scores)),
        "n_samples": int(len(df)),
        "n_attacks": int(df["label"].sum()),
    }
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    return metrics


if __name__ == "__main__":
    typer.run(evaluate)
