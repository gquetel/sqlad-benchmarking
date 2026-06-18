"""ROC and precision-recall curve plotting for the evaluation suite.

One ROC and one AUPRC curve are produced per evaluated cell and written as both
PNG and PDF (via plotly + kaleido); the PNG is uploaded as an MLflow artifact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from mlops_sqldetect.features import EXTRACTOR_LABELS
from mlops_sqldetect.model import PIPELINE_LABELS

# Disable log clutter because of kaleido
for _name in ("kaleido", "choreographer", "logistro", "browser_proc"):
    logging.getLogger(_name).setLevel(logging.WARNING)

PAPER_FONT = "CMU Serif"

# Combined-figure styling: extractor sets the colour, pipeline sets the dash so a
# (pipeline, extractor) cell is identifiable from either axis of comparison.
EXTRACTOR_COLORS = {"li": "#ffd54f", "sbert": "#7E57C2", "cv": "#6490f6"}
PIPELINE_DASHES = {"ae": "solid", "lof": "dash", "ocsvm": "dot"}


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


@dataclass(frozen=True)
class Curve:
    """One cell's persisted ROC + PR points, paired with its (pipeline, extractor)."""

    pipeline: str
    extractor: str
    roc: pd.DataFrame  # columns: fpr, tpr, threshold
    pr: pd.DataFrame  # columns: precision, recall, threshold

    def label(self, metric: str, value: float) -> str:
        pipe = PIPELINE_LABELS.get(self.pipeline, self.pipeline)
        feat = EXTRACTOR_LABELS.get(self.extractor, self.extractor)
        return f"{pipe} + {feat} ({metric}={value:.4f})"

    def _line(self) -> dict:
        return dict(color=EXTRACTOR_COLORS.get(self.extractor), dash=PIPELINE_DASHES.get(self.pipeline))


def plot_combined_roc(curves: list[Curve], out_path: Path) -> Path:
    """Overlay every cell's ROC curve onto a single figure (colour=extractor, dash=pipeline)."""
    fig = go.Figure()
    for c in curves:
        fpr, tpr = c.roc["fpr"].to_numpy(), c.roc["tpr"].to_numpy()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=c.label("AUROC", auc(fpr, tpr)), line=c._line()))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random classifier", line=dict(dash="dash", color="gray"))
    )
    return _export(fig, out_path, "ROC curve", "FPR", "TPR")


def plot_combined_pr(curves: list[Curve], out_path: Path, prevalence: float | None = None) -> Path:
    """Overlay every cell's precision-recall curve onto a single figure."""
    fig = go.Figure()
    for c in curves:
        precision, recall = c.pr["precision"].to_numpy(), c.pr["recall"].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision, mode="lines", name=c.label("AUPRC", auc(recall, precision)), line=c._line()
            )
        )
    if prevalence is not None:
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
            # Hold the range at [0,1] and shrink the drawing area (not the range)
            # to satisfy the equal-aspect constraint below — keeps ticks sane.
            constrain="domain",
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(family=PAPER_FONT, size=22)),
            tickfont=dict(family=PAPER_FONT, size=18),
            range=[0, 1],
            # Lock one y-unit to one x-unit so the plotting area renders square.
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        template="plotly_white",
        width=900,
        height=900,
        # Legend horizontal, centred below the plot rather than alongside it.
        legend=dict(
            font=dict(family=PAPER_FONT, size=16),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=2)
    fig.write_image(str(out_path.with_suffix(".pdf")))
    return out_path
