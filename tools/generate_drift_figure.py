"""Generate the SuperViz26 concept-drift dumbbell figure from MLflow.

Loads every finished full-run cell of the ``Drift-Superviz26-SQL`` experiment and
emits the reference vs. post-drift dumbbell figure (one AUROC panel and one AUPRC
panel) for the benchmark paper. Each pipeline is trained once on a benign origin
(S1) and scored on the origin test set (reference) and the held-out shifted test
set (S2, post-drift); the figure visualises the per-pipeline performance drop
$\\Delta = \\text{S1} - \\text{S2}$ averaged over the four Superviz26 domains.

This mirrors the figure side of :mod:`tools.generate_baselines_tex`, but compares
the reference and post-drift regimes instead of in-domain and LODO, and writes
only the figure (no table).

MLflow is the single source of truth: the latest finished full-run per
``(feature_extractor, pipeline, domain)`` cell wins, so re-running this after new
runs refreshes the figure. Pipelines missing any of the four domains are dropped.

Usage:
    python -m tools.generate_drift_figure \
        --figure-out ~/repos/quetel_phd_latex/papers/superviz26/sections/superviz26-drift-dumbbell.tex
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Annotated

import mlflow
import typer

from mlops_sqldetect.tracking import setup_mlflow

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The MLflow experiment holding the concept-drift grid (see tracking._EXPERIMENT_NAMES).
EXPERIMENT = "Drift-Superviz26-SQL"

# Pipelines in display order: (feature_extractor, engine, paper label, figure short label).
# Mirrors tools.generate_baselines_tex so labels stay consistent across figures.
PIPELINES: list[tuple[str, str, str, str]] = [
    ("cv", "ae", "CountVectorizer + AE", "CV+AE"),
    ("cv", "lof", "CountVectorizer + LOF", "CV+LOF"),
    ("cv", "ocsvm", "CountVectorizer + OCSVM", "CV+OCSVM"),
    ("li", "ae", "Li + AE", "Li+AE"),
    ("li", "lof", "Li + LOF", "Li+LOF"),
    ("li", "ocsvm", "Li + OCSVM", "Li+OCSVM"),
    ("loginov", "ae", "Loginov + AE", "Loginov+AE"),
    ("loginov", "lof", "Loginov + LOF", "Loginov+LOF"),
    ("loginov", "ocsvm", "Loginov + OCSVM", "Loginov+OCSVM"),
    ("sbert", "ae", "SecureBERT + AE", "SecureBERT+AE"),
    ("sbert", "lof", "SecureBERT + LOF", "SecureBERT+LOF"),
    ("sbert", "ocsvm", "SecureBERT + OCSVM", "SecureBERT+OCSVM"),
    ("codet5", "ae", "CodeT5+ + AE", "CodeT5+AE"),
    ("codet5", "lof", "CodeT5+ + LOF", "CodeT5+LOF"),
    ("codet5", "ocsvm", "CodeT5+ + OCSVM", "CodeT5+OCSVM"),
]

# The four Superviz26 domains, each evaluated as its own concept-drift cell.
DOMAINS: list[str] = ["a", "b", "c", "d"]

Metrics = dict[str, float | None]
Results = dict[tuple[str, str, str], Metrics]

FIGURE_CAPTION = (
    r"Reference (\textbullet) vs.\ post-drift ($\blacksquare$) detection performance per pipeline, "
    r"averaged over the four Superviz26 domains; the connector length is the concept-drift drop "
    r"$\Delta = \text{S1} - \text{S2}$, shown in green when small ($< 0.05$) and red otherwise."
)


def _num(x: object) -> float | None:
    """Coerce an MLflow metric cell to a float, mapping missing/NaN to None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None
    return None if math.isnan(v) else v


def load_results() -> Results:
    """Load the latest finished full-run per cell, indexed by (extractor, engine, domain)."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string="attributes.status = 'FINISHED' and tags.run_type = 'full-run'",
        output_format="pandas",
    )
    if runs.empty:
        raise RuntimeError(f"No finished full-run found in MLflow experiment {EXPERIMENT!r}.")

    # Ascending start_time so the latest run of a repeated cell overwrites earlier ones.
    runs = runs.sort_values("start_time")
    data: Results = {}
    for _, r in runs.iterrows():
        extractor, engine, domain = r.get("tags.feature_extractor"), r.get("tags.pipeline"), r.get("tags.domain")
        if not (isinstance(extractor, str) and isinstance(engine, str) and isinstance(domain, str)):
            continue
        data[(extractor, engine, domain)] = {
            "auroc_s1": _num(r.get("metrics.auroc_s1")),
            "auroc_s2": _num(r.get("metrics.auroc_s2")),
            "auprc_s1": _num(r.get("metrics.auprc_s1")),
            "auprc_s2": _num(r.get("metrics.auprc_s2")),
        }
    logger.info(f"Loaded {len(data)} cells from {EXPERIMENT}.")
    return data


# --- dumbbell figure ---------------------------------------------------------

# Places the drop label by each connector; \dumbthresh/\dumbthreshR are set per subfigure.
DUMBDELTA_MACRO = r"""  % \dumbdelta{leftx}{rightx}{ycoord}{value}: place the delta label left of the left marker;
  % flip it right of the right marker if it would overflow (leftx<thresh), else onto the midpoint.
  \def\dumbdelta#1#2#3#4{%
    \pgfmathsetmacro{\dmid}{(#1+#2)/2}%
    \pgfmathparse{#1<\dumbthresh}%
    \ifdim\pgfmathresult pt>0.5pt
      \pgfmathparse{#2>\dumbthreshR}%
      \ifdim\pgfmathresult pt>0.5pt
        \node[font=\tiny, fill=white, fill opacity=0.9, text opacity=1, inner sep=0.4pt] at (axis cs:\dmid,#3) {$#4$};
      \else
        \node[font=\tiny, anchor=west, xshift=3pt] at (axis cs:#2,#3) {$#4$};
      \fi
    \else
      \node[font=\tiny, anchor=east, xshift=-3pt] at (axis cs:#1,#3) {$#4$};
    \fi}"""

# Decision-engine rows inside each per-feature-extractor mini-plot. ENGINE_ORDER is the
# top-to-bottom screen order; the symbolic y coords list them bottom-to-top (ENGINE_BTT).
ENGINE_ORDER = {"lof": 0, "ocsvm": 1, "ae": 2}
ENGINE_BTT = sorted(ENGINE_ORDER, key=lambda e: ENGINE_ORDER[e], reverse=True)  # ae, ocsvm, lof

# Paper-facing short labels: feature extractor names title each mini-plot, decision engines
# label the (shared) y ticks.
EXTRACTOR_SHORT = {"cv": "CV", "li": "Li", "loginov": "Loginov", "sbert": "SecureBERT", "codet5": "CodeT5+"}
ENGINE_SHORT = {"lof": "LOF", "ocsvm": "OCSVM", "ae": "AE"}
SHORT_BY_CELL = {(extractor, engine): short for extractor, engine, _, short in PIPELINES}

# x ticks shared by both panels; a panel renders only those within its x-range.
_XTICK_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _fmt_tick(v: float) -> str:
    """Format an x tick: a bare ``0`` for the origin, one decimal otherwise (e.g. ``0.2``, ``1.0``)."""
    return "0" if abs(v) < 1e-9 else f"{v:.1f}"


def _xticks(xmin: float) -> str:
    """Comma-separated ticks from ``_XTICK_GRID`` that fall within ``[xmin, 1]``."""
    return ",".join(_fmt_tick(v) for v in _XTICK_GRID if v >= xmin - 1e-9)


def _extractor_order() -> list[str]:
    """Feature extractors top-to-bottom in the figure: the reverse of their PIPELINES order."""
    seen: list[str] = []
    for extractor, *_ in PIPELINES:
        if extractor not in seen:
            seen.append(extractor)
    return list(reversed(seen))


def _averages(data: Results) -> dict[str, Metrics]:
    """Per pipeline, average each metric over the four domains."""

    def avg_over(extractor: str, engine: str, key: str) -> float | None:
        xs = []
        for domain in DOMAINS:
            cell = data.get((extractor, engine, domain))
            if cell is None or cell[key] is None:
                return None
            xs.append(cell[key])
        return sum(xs) / len(xs)

    out: dict[str, Metrics] = {}
    for extractor, engine, _, short in PIPELINES:
        out[short] = {
            "s1_auroc": avg_over(extractor, engine, "auroc_s1"),
            "s2_auroc": avg_over(extractor, engine, "auroc_s2"),
            "s1_auprc": avg_over(extractor, engine, "auprc_s1"),
            "s2_auprc": avg_over(extractor, engine, "auprc_s2"),
        }
    return out


def _pgfplotsset(auroc_xmin: float, ycoords: str, bottom_title: str) -> str:
    """The shared ``dumbstack``/``dumbstackp`` mini-plot styles (one x scale per panel)."""
    return (
        "  % shared style for the per-feature-extractor AUROC mini-plots: identical x scale,\n"
        f"  % x ticks/label hidden by default and re-enabled only on the bottom ({bottom_title}) plot.\n"
        "  \\pgfplotsset{\n"
        "    dumbstack/.style={\n"
        "      width=\\linewidth, height=2.4cm,\n"
        f"      xmin={auroc_xmin:.2f}, xmax=1.00,\n"
        f"      xtick={{{_xticks(auroc_xmin)}}},\n"
        f"      symbolic y coords={{{ycoords}}},\n"
        "      ytick=data,\n"
        f"      yticklabels={{{ycoords}}},\n"
        "      tick label style={font=\\scriptsize},\n"
        "      yticklabel style={font=\\tiny},\n"
        "      enlarge y limits=0.12,\n"
        "      xmajorgrids, grid style={dotted},\n"
        "      tick align=outside,\n"
        "      xticklabels={},\n"
        "      title style={font=\\scriptsize, at={(0.01,1)}, anchor=south west, yshift=-5pt},\n"
        "    },\n"
        "    % AUPRC variant: wider x range, no y labels (shared from the AUROC panel).\n"
        f"    dumbstackp/.style={{dumbstack, xmin=0.00, xtick={{{_xticks(0.0)}}}, yticklabels={{}}}},\n"
        "  }"
    )


def _fe_plot(
    extractor: str,
    avgs: dict[str, Metrics],
    s1_key: str,
    s2_key: str,
    style: str,
    *,
    is_bottom: bool,
    xticklabels: str,
    bottom_shows_label: bool,
) -> str:
    """Render one feature-extractor mini-plot (three decision-engine dumbbell rows)."""
    title = EXTRACTOR_SHORT[extractor]
    if is_bottom:
        extra = " + label" if bottom_shows_label else ""
        comment = f"    % --- {title} (bottom plot: x-axis ticks{extra} shown here only) ---"
    else:
        comment = f"    % --- {title} ---"
    axis_opts = f"{style}, title={title}"
    if is_bottom:
        axis_opts += f", xticklabels={{{xticklabels}}}"

    cells = {e: avgs[SHORT_BY_CELL[(extractor, e)]] for e in ENGINE_BTT}
    connectors = "\n".join(
        f"        \\addplot[gray, line width=0.8pt, forget plot] coordinates "
        f"{{({cells[e][s2_key]:.4f},{ENGINE_SHORT[e]}) ({cells[e][s1_key]:.4f},{ENGINE_SHORT[e]})}};"
        for e in ENGINE_BTT
    )
    s2_marks = " ".join(f"({cells[e][s2_key]:.4f},{ENGINE_SHORT[e]})" for e in ENGINE_BTT)
    s1_marks = " ".join(f"({cells[e][s1_key]:.4f},{ENGINE_SHORT[e]})" for e in ENGINE_BTT)
    # Drop label per engine: green when the drop is small (< 0.05), red otherwise.
    deltas = []
    for e in ENGINE_BTT:
        s1v, s2v = cells[e][s1_key], cells[e][s2_key]
        delta = s1v - s2v
        color = "green!50!black" if delta < 0.05 else "red!80!black"
        value = rf"\textcolor{{{color}}}{{{delta:+.2f}}}"
        deltas.append(
            f"        \\dumbdelta{{{min(s1v, s2v):.4f}}}{{{max(s1v, s2v):.4f}}}{{{ENGINE_SHORT[e]}}}{{{value}}}"
        )
    delta_block = "\n".join(deltas)
    suffix = "" if is_bottom else "\\\\[-6pt]"
    return (
        f"{comment}\n"
        "    \\begin{tikzpicture}\n"
        f"      \\begin{{axis}}[{axis_opts}]\n"
        f"{connectors}\n"
        f"        \\addplot[only marks, mark=square*, mark size=2pt, black] coordinates {{{s2_marks}}};\n"
        f"        \\addplot[only marks, mark=*, mark size=2pt, black] coordinates {{{s1_marks}}};\n"
        f"{delta_block}\n"
        "      \\end{axis}\n"
        f"    \\end{{tikzpicture}}{suffix}"
    )


def _panel(
    extractors: list[str],
    avgs: dict[str, Metrics],
    s1_key: str,
    s2_key: str,
    style: str,
    xticklabels: str,
    xmin: float,
    xmax: float,
    *,
    caption: str,
    bottom_shows_label: bool,
) -> str:
    """Render one dumbbell subfigure: a stack of per-feature-extractor mini-plots."""
    span = xmax - xmin
    lines = [
        "  \\begin{subfigure}{0.49\\linewidth}",
        "    \\centering",
        f"    \\def\\dumbthresh{{{xmin + 0.20 * span:.2f}}}",
        f"    \\def\\dumbthreshR{{{xmax - 0.13 * span:.2f}}}",
    ]
    for i, extractor in enumerate(extractors):
        lines.append(
            _fe_plot(
                extractor,
                avgs,
                s1_key,
                s2_key,
                style,
                is_bottom=i == len(extractors) - 1,
                xticklabels=xticklabels,
                bottom_shows_label=bottom_shows_label,
            )
        )
    lines.append(f"    \\caption{{{caption}}}")
    lines.append("  \\end{subfigure}")
    return "\n".join(lines)


def render_figure(data: Results) -> str:
    """Render the two-panel (AUROC, AUPRC) dumbbell figure for the fully-available extractors."""
    avgs = _averages(data)
    keys = ("s1_auroc", "s2_auroc", "s1_auprc", "s2_auprc")

    def complete(extractor: str) -> bool:
        return all(avgs[SHORT_BY_CELL[(extractor, e)]][k] is not None for e in ENGINE_BTT for k in keys)

    extractors = [e for e in _extractor_order() if complete(e)]
    if not extractors:
        raise RuntimeError("No feature extractor has all three engines' S1 and S2 averages required for the figure.")

    auroc_min = min(
        min(avgs[SHORT_BY_CELL[(extractor, e)]]["s1_auroc"], avgs[SHORT_BY_CELL[(extractor, e)]]["s2_auroc"])
        for extractor in extractors
        for e in ENGINE_BTT
    )
    auroc_xmin = max(0.0, math.floor(auroc_min * 20 - 1) / 20)

    ycoords = ",".join(ENGINE_SHORT[e] for e in ENGINE_BTT)
    pgfset = _pgfplotsset(auroc_xmin, ycoords, EXTRACTOR_SHORT[extractors[-1]])

    panels = "\n  \\hfill\n".join(
        [
            _panel(
                extractors,
                avgs,
                "s1_auroc",
                "s2_auroc",
                "dumbstack",
                _xticks(auroc_xmin),
                auroc_xmin,
                1.0,
                caption="AUROC",
                bottom_shows_label=True,
            ),
            _panel(
                extractors,
                avgs,
                "s1_auprc",
                "s2_auprc",
                "dumbstackp",
                _xticks(0.0),
                0.0,
                1.0,
                caption="AUPRC",
                bottom_shows_label=False,
            ),
        ]
    )
    return (
        "\\begin{figure}[!htb]\n"
        "  \\centering\n"
        f"{DUMBDELTA_MACRO}\n"
        f"{pgfset}\n"
        f"{panels}\n"
        f"  \\caption{{{FIGURE_CAPTION}}}\n"
        "  \\label{fig:superviz26-drift-dumbbell}\n"
        "\\end{figure}"
    )


# --- entry point -------------------------------------------------------------


app = typer.Typer(add_completion=False, help=__doc__)


def _write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    logger.info(f"Wrote {path}")


@app.command()
def main(
    figure_out: Annotated[Path, typer.Option(help="Path of the concept-drift dumbbell figure (.tex).")] = REPO_ROOT
    / "reports"
    / "superviz26"
    / "superviz26-drift-dumbbell.tex",
) -> None:
    """Load the SuperViz26 concept-drift results from MLflow and write the dumbbell figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow("superviz26-drift"):
        raise typer.Exit(code=1)
    data = load_results()
    _write(render_figure(data), figure_out)


if __name__ == "__main__":
    app()
