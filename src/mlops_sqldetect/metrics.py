"""Evaluation metrics for one-class SQLi detection."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def threshold_for_fpr(normal_scores: np.ndarray, target_fpr: float) -> float:
    """Score percentile of normal samples that admits at most ``target_fpr`` false positives."""
    return float(np.percentile(np.asarray(normal_scores, dtype=float), (1 - target_fpr) * 100))


def wilson_ci(score: float, n: int) -> float:
    """95% normal-approximation half-width for a metric in ``[0, 1]`` over ``n`` samples."""
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(score * (1 - score) / n)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, model_name: str = "") -> dict[str, float]:
    """Threshold-based + curve-based metrics from anomaly scores; values in ``[0, 1]``."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores > threshold).astype(int)

    p, r, _ = precision_recall_curve(labels, scores, pos_label=1)
    auprc = float(auc(r, p))
    fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
    rocauc = float(auc(fpr_curve, tpr_curve))

    tn, fp, _, _ = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    achieved_fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0

    m = {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "fpr": achieved_fpr,
        "auprc": auprc,
        "rocauc": rocauc,
        "auroc_ci": wilson_ci(rocauc, len(scores)),
        "auprc_ci": wilson_ci(auprc, len(scores)),
    }

    logger.info(f"Metrics for {model_name}.")
    logger.info(f"Accuracy: {m['accuracy'] * 100:.2f}%")
    logger.info(f"F1 Score: {m['f1'] * 100:.2f}%")
    logger.info(f"Precision: {m['precision'] * 100:.2f}%")
    logger.info(f"Recall: {m['recall'] * 100:.2f}%")
    logger.info(f"False Positive Rate: {m['fpr'] * 100:.5f}%")
    logger.info(f"ROC-AUC: {rocauc:.4f}, CI {m['auroc_ci']:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}, CI {m['auprc_ci']:.4f}")

    return m


def recall_per_attack(labels: np.ndarray, preds: np.ndarray, techniques: pd.Series) -> dict[str, float]:
    """Recall for each attack technique (rows where ``label == 1``), skipping empty/NaN names."""
    df = pd.DataFrame({"label": np.asarray(labels, dtype=int), "preds": np.asarray(preds, dtype=int)})
    df["technique"] = pd.Series(techniques).reset_index(drop=True)

    names = [
        t
        for t in df.loc[df["label"] == 1, "technique"].unique().tolist()
        if t and not (isinstance(t, float) and pd.isna(t))
    ]
    out: dict[str, float] = {}
    for technique in names:
        mask = df["technique"] == technique
        out[technique] = float(recall_score(df.loc[mask, "label"], df.loc[mask, "preds"], zero_division=0))
    return out
