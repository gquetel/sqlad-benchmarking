"""ROC and precision-recall curve plotting for the evaluation suite.

One ROC and one AUPRC curve are produced per evaluated cell and written as both
PNG and PDF (via plotly + kaleido); the PNG is uploaded as an MLflow artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PAPER_FONT = "CMU Serif"


def dump_curve_points(labels: np.ndarray, scores: np.ndarray, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Persist raw ROC and PR curve points as two CSVs for offline re-plotting."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_path = out_dir / f"{stem}_roc.csv"
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(roc_path, index=False)

    precision, recall, pr_thresholds = precision_recall_curve(labels, scores, pos_label=1)
    # sklearn returns one fewer threshold than precision/recall (the last point
    # precision=1, recall=0 has no threshold); pad with NaN to align columns.
    pr_thresholds = np.append(pr_thresholds, np.nan)
    pr_path = out_dir / f"{stem}_auprc.csv"
    pd.DataFrame({"precision": precision, "recall": recall, "threshold": pr_thresholds}).to_csv(pr_path, index=False)

    return roc_path, pr_path


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, name: str, out_path: Path) -> Path:
    """Plot the ROC curve for ``scores`` and save it as a PNG at ``out_path``."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUROC={auc(fpr, tpr):.4f})"))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random classifier", line=dict(dash="dash", color="gray"))
    )
    return _export(fig, out_path, "ROC curve", "FPR", "TPR")


def plot_pr_curve(labels: np.ndarray, scores: np.ndarray, name: str, out_path: Path) -> Path:
    """Plot the precision-recall curve for ``scores`` and save it as a PNG."""
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} (AUPRC={auc(recall, precision):.4f})"))
    # Prevalence baseline: the AUPRC a random classifier would achieve.
    prevalence = float(np.sum(labels)) / len(labels)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[prevalence, prevalence],
            mode="lines",
            name=f"Random classifier = {prevalence:.4f}",
            line=dict(dash="dash", color="gray"),
        )
    )
    return _export(fig, out_path, "Precision-Recall curve", "Recall", "Precision")


def _export(fig: go.Figure, out_path: Path, title: str, xlabel: str, ylabel: str) -> Path:
    fig.update_layout(
        title=title,
        font=dict(family=PAPER_FONT, size=18),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(family=PAPER_FONT, size=22)),
            tickfont=dict(family=PAPER_FONT, size=18),
            range=[0, 1],
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(family=PAPER_FONT, size=22)),
            tickfont=dict(family=PAPER_FONT, size=18),
            range=[0, 1.02],
        ),
        template="plotly_white",
        width=900,
        height=750,
        legend=dict(
            font=dict(family=PAPER_FONT, size=16),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=2)
    fig.write_image(str(out_path.with_suffix(".pdf")))
    return out_path
