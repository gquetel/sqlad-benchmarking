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
from mlops_sqldetect.model import METHOD_LABELS

# Disable log clutter because of kaleido
for _name in ("kaleido", "choreographer", "logistro", "browser_proc"):
    logging.getLogger(_name).setLevel(logging.WARNING)

PAPER_FONT = "CMU Serif"

# Combined-figure styling: extractor sets the colour, method sets the dash so a
# (method, extractor) cell is identifiable from either axis of comparison.
EXTRACTOR_COLORS = {"li": "#ffd54f", "sbert": "#7E57C2", "cv": "#6490f6", "loginov": "#26a69a", "codet5": "#ef5350"}
METHOD_DASHES = {"ae": "solid", "lof": "dash", "ocsvm": "dot"}


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
    """One cell's persisted ROC + PR points, paired with its (method, extractor)."""

    method: str
    extractor: str
    roc: pd.DataFrame  # columns: fpr, tpr, threshold
    pr: pd.DataFrame  # columns: precision, recall, threshold

    def label(self, metric: str, value: float) -> str:
        pipe = METHOD_LABELS.get(self.method, self.method)
        feat = EXTRACTOR_LABELS.get(self.extractor, self.extractor)
        return f"{pipe} + {feat} ({metric}={value:.4f})"

    def _line(self) -> dict:
        return dict(color=EXTRACTOR_COLORS.get(self.extractor), dash=METHOD_DASHES.get(self.method))


def plot_combined_roc(curves: list[Curve], out_path: Path) -> Path:
    """Overlay every cell's ROC curve onto a single figure (colour=extractor, dash=method)."""
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


# --- native pgfplots / TikZ export ------------------------------------------
#
# The plotly PNG/PDF above is a raster/vector image dropped in via \includegraphics;
# the .tex emitted here is a standalone tikzpicture the thesis \inputs directly, so
# fonts, line weights and colours match the surrounding LaTeX. The document preamble
# must load pgfplots (e.g. ``\usepackage{pgfplots}\pgfplotsset{compat=1.18}``).
#
# Decision engines become the *rows* of the combined figure (one PR + one ROC
# panel each): a curve's colour encodes the extractor, its row encodes the method,
# so no per-line dashing is needed. Fixed row order; unknown methods are appended.
METHOD_ROW_ORDER = ["ocsvm", "lof", "ae"]

# ROC/PR curves carry hundreds-to-thousands of points; inlining all of them per curve
# would bloat the .tex and slow LaTeX to a crawl. Thin each curve to this many points
# (endpoints always kept) — visually indistinguishable for smooth monotone curves.
TIKZ_MAX_POINTS = 400


def _tikz_color_name(extractor: str) -> str:
    """xcolor name for an extractor's curve colour (letters/digits only, tex-safe)."""
    return f"ext{extractor}"


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int = TIKZ_MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Evenly thin a dense curve to ``<= max_points`` points, always keeping the endpoints."""
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    return x[idx], y[idx]


def _pgf_coords(x: np.ndarray, y: np.ndarray) -> str:
    """Format ``(x,y)`` pairs for a pgfplots ``coordinates {...}`` list."""
    return " ".join(f"({xi:.5f},{yi:.5f})" for xi, yi in zip(x, y, strict=True))


def _curve_xy(c: Curve, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the (x, y) arrays for ``metric`` (``"roc"`` -> FPR/TPR, ``"pr"`` -> Recall/Precision)."""
    if metric == "roc":
        return c.roc["fpr"].to_numpy(), c.roc["tpr"].to_numpy()
    return c.pr["recall"].to_numpy(), c.pr["precision"].to_numpy()


def plot_combined_curves_tikz(
    curves: list[Curve],
    out_path: Path,
    prevalence: float | None = None,
    extractor_labels: dict[str, str] | None = None,
) -> Path:
    """Emit the combined performance figure as one pgfplots ``groupplot`` ``.tex`` file.

    Layout: a ``2`` (columns: PR | ROC) ``x N`` (rows: one decision engine each) grid.
    Every line is solid and coloured by feature extractor; the row (its bold y-label)
    encodes the decision engine. A single shared legend -- extractor names plus the
    random-classifier baseline -- is collected once (via ``legend to name``) and rendered
    below the grid with ``\\ref``. Per-cell AUC scores are intentionally dropped here (they
    live in the metrics table). The document preamble must load pgfplots *and*
    ``\\usepgfplotslibrary{groupplots}``.

    ``extractor_labels`` overrides the extractor legend text (defaults to the package's
    canonical :data:`EXTRACTOR_LABELS`); pass the thesis-facing labels so the legend matches
    the metrics table (e.g. ``"Secure-BERT"`` rather than the internal ``"SecureBERT"``).
    """
    labels = extractor_labels or EXTRACTOR_LABELS
    lookup = {(c.method, c.extractor): c for c in curves}

    # Rows: known engines first (fixed order), then any unexpected method as-seen.
    methods = [m for m in METHOD_ROW_ORDER if any(c.method == m for c in curves)]
    for c in curves:
        if c.method not in methods:
            methods.append(c.method)

    # Columns share the extractor colour order; first-seen keeps it deterministic.
    extractors: list[str] = []
    for c in curves:
        if c.extractor not in extractors:
            extractors.append(c.extractor)

    color_defs = "\n".join(
        f"\\definecolor{{{_tikz_color_name(e)}}}{{HTML}}{{{EXTRACTOR_COLORS[e].lstrip('#').upper()}}}"
        for e in extractors
    )

    # Collect the shared legend from whichever engine pairs with the most extractors,
    # so its PR panel is guaranteed to actually plot every entry the legend names.
    legend_method = max(methods, key=lambda m: sum((m, e) in lookup for e in extractors))

    roc_baseline = "\\addplot[gray, dashed, line width=0.8pt] coordinates {(0,0) (1,1)};"
    pr_baseline = (
        None
        if prevalence is None
        else f"\\addplot[gray, dashed, line width=0.8pt] coordinates {{(0,{prevalence:.5f}) (1,{prevalence:.5f})}};"
    )

    panels: list[str] = []
    for row, method in enumerate(methods):
        first_row, last_row = row == 0, row == len(methods) - 1
        for metric in ("pr", "roc"):
            opts: list[str] = []
            if metric == "pr":
                opts.append(f"ylabel={{\\textbf{{{METHOD_LABELS.get(method, method)}}}\\\\Precision}}")
                if first_row:
                    opts.append("title={PR curves}")
                if last_row:
                    opts.append("xlabel={Recall}")
            else:
                opts.append("ylabel={TPR}")
                if first_row:
                    opts.append("title={ROC curves}")
                if last_row:
                    opts.append("xlabel={FPR}")
            legend_here = metric == "pr" and method == legend_method
            if legend_here:
                opts.append("legend to name=perfleg")

            lines = [f"    \\nextgroupplot[{', '.join(opts)}]"]
            for e in extractors:
                c = lookup.get((method, e))
                if c is None:  # this engine never converged for extractor ``e``
                    continue
                xs, ys = _downsample(*_curve_xy(c, metric))
                lines.append(
                    f"      \\addplot[{_tikz_color_name(e)}, solid, line width=1pt] "
                    f"coordinates {{{_pgf_coords(xs, ys)}}};"
                )
                if legend_here:
                    lines.append(f"      \\addlegendentry{{{labels.get(e, e)}}}")
            baseline = roc_baseline if metric == "roc" else pr_baseline
            if baseline is not None:
                lines.append(f"      {baseline}")
                if legend_here:
                    lines.append("      \\addlegendentry{Random classifier}")
            panels.append("\n".join(lines))

    group_opts = (
        f"group style={{group size=2 by {len(methods)},\n"
        "          horizontal sep=1.6cm, vertical sep=1.2cm},\n"
        "        width=0.46\\linewidth, height=0.42\\linewidth,\n"
        "        xmin=0, xmax=1, ymin=0, ymax=1,\n"
        "        tick label style={font=\\scriptsize},\n"
        "        label style={font=\\small},\n"
        "        title style={font=\\small\\bfseries},\n"
        "        ylabel style={align=center},\n"
        "        grid=both, grid style={dotted},\n"
        "        legend cell align=left,\n"
        "        legend style={font=\\small, legend columns=3, draw=none,\n"
        "          /tikz/every even column/.append style={column sep=1em}},"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "% Auto-generated by mlops_sqldetect.visualize — do not edit by hand.\n"
        "% Requires \\usepackage{pgfplots}, \\pgfplotsset{compat=1.18}\n"
        "% and \\usepgfplotslibrary{groupplots} in the preamble.\n"
        f"{color_defs}\n"
        "\\begin{tikzpicture}\n"
        f"    \\begin{{groupplot}}[\n        {group_opts}\n      ]\n"
        + "\n".join(panels)
        + "\n    \\end{groupplot}\n"
        "\\end{tikzpicture}\n"
        "\\par\\vspace{0.4\\baselineskip}\n"
        "\\ref{perfleg}\n"
    )
    return out_path


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
