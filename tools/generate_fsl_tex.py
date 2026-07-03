"""Generate the SuperViz26 few-shot adaptation figure from MLflow.

The few-shot experiment (:mod:`mlops_sqldetect.evaluate_fsl`) takes each LODO
autoencoder and adapts it to its held-out target domain with a small budget ``k``
of benign samples, sweeping ``k`` over ``{0, 5, 10, 50, 100, 500, 1000, 10000}``
and averaging over five seeds. ``k = 0`` is the unmodified LODO model. The runs
live in the ``FSL-Superviz26-SQL`` experiment (see tracking._EXPERIMENT_NAMES),
one nested run per ``(feature_extractor, target, k)`` cell tagged ``kind=few_shot``.

This emits the AUROC-vs-``k`` figure: one curve per feature extractor (AE head),
averaging the per-target AUROC over the four target domains at each budget. A
vertical tick marks the smallest ``k`` that recovers in-domain performance (within
``0.01`` AUROC). The in-domain reference is the AE in-domain AUROC averaged over the
four in-domain scenarios, read from the standard ``Superviz26-SQL`` experiment.

MLflow is the single source of truth: the latest finished run per
``(extractor, target, k)`` cell wins, so re-running this refreshes the figure.
Extractors with no few-shot data are dropped.

Usage:
    python -m tools.generate_fsl_tex \
        --figure-out ~/repos/quetel_phd_latex/papers/superviz26/data/superviz26-fsl.tex
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Annotated

import mlflow
import typer

from mlops_sqldetect.features import EXTRACTOR_LABELS
from mlops_sqldetect.tracking import setup_mlflow

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# MLflow experiments: the few-shot sweep and the standard grid holding the in-domain
# reference (see tracking._EXPERIMENT_NAMES).
EXPERIMENT_FSL = "FSL-Superviz26-SQL"
EXPERIMENT_BASE = "Superviz26-SQL"

# The set of feature extractors is discovered from the experiment, not hard-coded; these
# only fix the *appearance* of each curve. Colours are the IFIPSEC/DIMVA palette
# (papers/IFIPSEC/colors.tex): ``li`` and ``securebert`` are taken verbatim; the other
# extractors reuse hues from the same file (codet5 keeps the ossgpt green). Any extractor
# not listed here falls back to the remaining palette colours. Values are HTML hex.
EXTRACTOR_STYLES: dict[str, tuple[str, str]] = {
    "li": ("FFD54F", "*"),  # IFIPSEC \definecolor{li}
    "sbert": ("7E57C2", "square*"),  # IFIPSEC \definecolor{securebert}
    "codet5": ("66BB6A", "triangle*"),  # IFIPSEC ossgpt (green)
    "cv": ("4FC3F7", "diamond*"),  # IFIPSEC chatgpt (blue)
    "loginov": ("E57373", "pentagon*"),  # IFIPSEC claude (red)
}
# Remaining IFIPSEC colours.tex hues, for any other extractor discovered in the experiment.
FALLBACK_HEX = ["0D47A1", "8b7e74", "9fa8da", "DCE775", "4FC3F7"]
FALLBACK_MARKS = ["oplus*", "otimes*", "halfcircle*", "star", "x"]

# Display order for known extractors; any extractor not listed here is appended, sorted by name.
EXTRACTOR_ORDER = ["cv", "li", "loginov", "sbert", "codet5"]

# Target domains the LODO models adapt to (tags.target values).
TARGETS = ["a", "b", "c", "d"]

# In-domain scenarios (tags.scenario values) whose AE AUROC defines the recovery threshold.
INDOMAIN = ["a-a", "b-b", "c-c", "d-d"]

# k = 0 is the unadapted LODO model; the log x-axis cannot show 0, so it is placed at x = 1.
ZERO_X = 1.0
# An adapted model "recovers" once its AUROC is within this margin of the in-domain reference.
RECOVERY_MARGIN = 0.01

CAPTION = (
    r"Few-shot adaptation of LODO autoencoders: AUROC vs.\ adaptation budget~$k$, averaged over "
    r"the four target domains and five seeds ($k{=}0$ is the LODO model). A black-bordered marker "
    r"indicates the smallest~$k$ recovering in-domain performance (within $0.01$ AUROC)."
)


def _num(x: object) -> float | None:
    """Coerce an MLflow metric/param cell to a float, mapping missing/NaN to None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _ordered(extractors: list[str]) -> list[str]:
    """Display order: known extractors first (per ``EXTRACTOR_ORDER``), then the rest by name."""
    rank = {e: i for i, e in enumerate(EXTRACTOR_ORDER)}
    return sorted(extractors, key=lambda e: (rank.get(e, len(rank)), e))


def _style(extractor: str, index: int) -> tuple[str, str, str, str]:
    """Return ``(label, color_name, rgb, mark)`` for ``extractor`` (``index`` picks the fallback)."""
    label = EXTRACTOR_LABELS.get(extractor, extractor)
    color = "col" + "".join(ch for ch in extractor.title() if ch.isalnum())
    rgb, mark = EXTRACTOR_STYLES.get(
        extractor,
        (FALLBACK_HEX[index % len(FALLBACK_HEX)], FALLBACK_MARKS[index % len(FALLBACK_MARKS)]),
    )
    return label, color, rgb, mark


def load_curves() -> dict[str, dict[int, float]]:
    """Per extractor, the AUROC averaged over the four targets at each adaptation budget ``k``.

    Reads the latest finished ``kind=few_shot`` run per ``(extractor, target, k)`` cell; a budget
    is kept only when all four targets are present, so every plotted point averages the same domains.
    """
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_FSL],
        filter_string=(
            "attributes.status = 'FINISHED' "
            "and (tags.setting = 'few_shot' or tags.kind = 'few_shot') "
            "and (tags.decision_engine = 'ae' or tags.pipeline = 'ae')"
        ),
        output_format="pandas",
    )
    if runs.empty:
        raise RuntimeError(f"No finished few-shot run found in MLflow experiment {EXPERIMENT_FSL!r}.")

    # Ascending start_time so the latest run of a repeated cell overwrites earlier ones.
    runs = runs.sort_values("start_time")
    # (extractor, target, k) -> AUROC.
    cells: dict[tuple[str, str, int], float] = {}
    for _, r in runs.iterrows():
        extractor = r.get("tags.feature_extractor")
        target = r.get("tags.target")
        k = _num(r.get("params.k"))
        auroc = _num(r.get("metrics.auroc"))
        if not (isinstance(extractor, str) and isinstance(target, str)) or k is None or auroc is None:
            continue
        cells[(extractor, target, int(k))] = auroc

    # Every feature extractor present in the experiment gets a curve (discovered, not hard-coded).
    curves: dict[str, dict[int, float]] = {}
    for extractor in _ordered(sorted({e for (e, _, _) in cells})):
        ks = sorted({k for (e, _, k) in cells if e == extractor})
        per_k: dict[int, float] = {}
        for k in ks:
            present = [cells[(extractor, t, k)] for t in TARGETS if (extractor, t, k) in cells]
            if len(present) == len(TARGETS):
                per_k[k] = sum(present) / len(present)
            else:
                logger.info(f"Skipping {extractor} k={k}: only {len(present)}/{len(TARGETS)} targets available.")
        if per_k:
            curves[extractor] = per_k
        else:
            logger.info(f"Dropping {extractor}: no complete few-shot budget.")
    if not curves:
        raise RuntimeError("No feature extractor has a complete (four-target) few-shot budget for the figure.")
    logger.info(f"Loaded {len(curves)} extractor curves: {', '.join(curves)}.")
    return curves


def load_indomain(extractors: list[str]) -> dict[str, float]:
    """Per extractor, the AE in-domain AUROC averaged over the four in-domain scenarios.

    This is the recovery target; an extractor missing any in-domain cell is simply omitted
    (its curve is still drawn, only without a recovery tick).
    """
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_BASE],
        filter_string=(
            "attributes.status = 'FINISHED' and tags.run_type = 'full-run' "
            "and (tags.decision_engine = 'ae' or tags.pipeline = 'ae')"
        ),
        output_format="pandas",
    )
    if runs.empty:
        logger.warning(f"No finished AE full-run in {EXPERIMENT_BASE!r}; recovery ticks will be omitted.")
        return {}
    runs = runs.sort_values("start_time")
    cells: dict[tuple[str, str], float] = {}
    for _, r in runs.iterrows():
        extractor, scenario, auroc = r.get("tags.feature_extractor"), r.get("tags.scenario"), _num(r.get("metrics.roc_auc"))
        if isinstance(extractor, str) and isinstance(scenario, str) and auroc is not None:
            cells[(extractor, scenario)] = auroc

    refs: dict[str, float] = {}
    for extractor in extractors:
        present = [cells[(extractor, sc)] for sc in INDOMAIN if (extractor, sc) in cells]
        if len(present) == len(INDOMAIN):
            refs[extractor] = sum(present) / len(present)
        else:
            logger.info(f"No complete in-domain reference for {extractor}; recovery tick omitted.")
    return refs


def _x(k: int) -> float:
    """X position of budget ``k`` on the log axis (``k = 0`` lives at ``ZERO_X``)."""
    return ZERO_X if k == 0 else float(k)


def _klabel(k: int) -> str:
    """Axis tick label for budget ``k`` (``0``, ``500``, ``{1\\,k}`` ...)."""
    if k == 0:
        return "0"
    if k >= 1000 and k % 1000 == 0:
        return rf"{{{k // 1000}\,k}}"
    return str(k)


def _recovery(curve: dict[int, float], ref: float | None) -> int | None:
    """Smallest budget ``k`` whose AUROC is within ``RECOVERY_MARGIN`` of the in-domain ``ref``."""
    if ref is None:
        return None
    for k in sorted(curve):
        if curve[k] >= ref - RECOVERY_MARGIN:
            return k
    return None


def render_figure(curves: dict[str, dict[int, float]], refs: dict[str, float]) -> str:
    """Render the AUROC-vs-``k`` few-shot adaptation figure."""
    # (extractor, label, color_name, rgb, mark) in display order.
    styled = [(e, *_style(e, i)) for i, e in enumerate(curves)]
    all_ks = sorted({k for curve in curves.values() for k in curve})
    positions = [_x(k) for k in all_ks]
    min_auroc = min(v for curve in curves.values() for v in curve.values())
    ymin = max(0.0, math.floor((min_auroc - 0.02) * 20) / 20)

    out: list[str] = [
        "% Generated from MLflow by mlops-sqldetect/tools/generate_fsl_tex.py. Do not edit by hand.",
        r"\begin{figure}[!htb]",
        r"  \centering",
    ]
    out += [rf"  \definecolor{{{color}}}{{HTML}}{{{rgb}}}" for _, _, color, rgb, _ in styled]
    out += [
        r"  \pgfplotsset{",
        r"    fslcurve/.style={",
        r"      width=0.82\linewidth, height=6cm,",
        rf"      xmin=0.7, xmax={max(positions) * 1.5:.0f},",
        rf"      xtick={{{', '.join(f'{p:.0f}' for p in positions)}}},",
        rf"      xticklabels={{{', '.join(_klabel(k) for k in all_ks)}}},",
        r"      xlabel={$k$ (adaptation samples)},",
        r"      tick label style={font=\scriptsize},",
        r"      label style={font=\small},",
        r"      grid=major, grid style={dotted, gray!40},",
        r"      mark size=1.8pt,",
        r"      line width=0.9pt,",
        r"      enlarge x limits=0.02,",
        r"      legend style={font=\tiny, at={(1.02,1)}, anchor=north west, draw=none,",
        r"                    fill=white, fill opacity=0.85, text opacity=1, row sep=1pt},",
        r"    }",
        r"  }",
        r"  \begin{tikzpicture}",
        rf"    \begin{{semilogxaxis}}[fslcurve, ylabel={{AUROC}}, ymin={ymin:.2f}, ymax=1.02]",
    ]
    for extractor, label, color, _, mark in styled:
        curve = curves[extractor]
        ref = refs.get(extractor)
        coords = " ".join(f"({_x(k):.0f},{curve[k]:.4f})" for k in sorted(curve))
        out += [
            f"      % {label} + AE",
            rf"      \addplot[{color}, mark={mark}] coordinates {{",
            f"        {coords}",
            r"      };",
            rf"      \addlegendentry{{{label}}}",
        ]
        rk = _recovery(curve, ref)
        if rk is not None:
            out += [
                f"      % Recovery marker (within {RECOVERY_MARGIN} of in-domain {ref:.4f}): k={rk}",
                rf"      \addplot[only marks, mark={mark}, mark size=2.6pt,",
                rf"               mark options={{fill={color}, draw=black, line width=0.9pt}},",
                rf"               forget plot] coordinates {{({_x(rk):.0f},{curve[rk]:.4f})}};",
            ]
        elif ref is not None:
            out.append(f"      % No k recovers in-domain {ref:.4f} within {RECOVERY_MARGIN}; no tick.")
    out += [
        r"    \end{semilogxaxis}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{CAPTION}}}",
        r"  \label{fig:superviz26-fsl}",
        r"\end{figure}",
    ]
    return "\n".join(out)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    figure_out: Annotated[Path, typer.Option(help="Path of the few-shot adaptation figure (.tex).")] = REPO_ROOT
    / "reports"
    / "superviz26"
    / "superviz26-fsl.tex",
) -> None:
    """Load the few-shot results from MLflow and write the AUROC-vs-k adaptation figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow("superviz26-fsl"):
        raise typer.Exit(code=1)
    curves = load_curves()
    refs = load_indomain(list(curves))
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    figure_out.write_text(render_figure(curves, refs) + "\n")
    logger.info(f"Wrote {figure_out}")


if __name__ == "__main__":
    app()
