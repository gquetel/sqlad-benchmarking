"""Generate the feature-extractor study concept-drift dumbbell figure from MLflow.

Loads every finished full-run cell of the ``FE-study-CD`` experiment and
emits the reference vs. post-drift dumbbell figure (one AUROC panel and one AUPRC
panel) for the thesis. Each method is trained once on a benign origin
(S1) and scored on the origin test set (reference) and the held-out shifted test
set (S2, post-drift); the figure visualises the per-method performance drop
$\\Delta = \\text{S2} - \\text{S1}$ averaged over the four Superviz26 domains.

This mirrors the figure side of :mod:`tools.generate_baselines_tex`, but compares
the reference and post-drift settings instead of in-domain and LODO, and writes
only the figure (no table).

MLflow is the single source of truth: the latest finished full-run per
``(feature_extractor, engine, domain)`` cell wins, so re-running this after new
runs refreshes the figure. Methods missing any of the four domains are dropped.

Usage:
    python -m tools.generate_drift_figure \
        --figure-out ~/repos/quetel_phd_latex/thesis/chapters/05-generalization/data/concept-drift-dumbbell.tex
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Annotated

import mlflow
import typer

from sqlad_benchmarking.features import EXTRACTOR_LABELS
from sqlad_benchmarking.tracking import setup_mlflow

logger = logging.getLogger(__name__)

DEFAULT_FIGURE_OUT = (
    Path.home() / "repos/quetel_phd_latex/thesis/chapters/05-generalization/data/concept-drift-dumbbell.tex"
)

# The MLflow experiment holding the concept-drift grid (see tracking._EXPERIMENT_NAMES).
EXPERIMENT = "FE-study-CD"

# Stable thesis display order. Missing runs are omitted, so the same script can be rerun
# while the grid is still filling.
EXTRACTOR_ORDER = [
    "cv",
    "tfidf",
    "li",
    "loginov",
    "kakisim",
    "gaur-expert",
    "gaur-chatgpt",
    "gaur-claude",
    "gaur-llama",
    "gaur-mistral",
    "gaur-gpt-oss",
    "gaur-ruleid",
    "sbert",
    "sbert2",
    "roberta",
    "modernbert",
    "sentbert",
    "codebert",
    "codet5",
    "flan-t5",
    "qwen3-emb",
]
DRIFT_ENGINES = [("ae", "AE"), ("ocsvm", "OCSVM"), ("lof", "LOF")]
METHODS: list[tuple[str, str, str, str]] = [
    (extractor, engine, f"{EXTRACTOR_LABELS[extractor]} + {engine_label}", f"{extractor}+{engine}")
    for extractor in EXTRACTOR_ORDER
    for engine, engine_label in DRIFT_ENGINES
]

# The four Superviz26 domains, each evaluated as its own concept-drift cell.
DOMAINS: list[str] = ["a", "b", "c", "d"]

Metrics = dict[str, float | None]
Results = dict[tuple[str, str, str], Metrics]

FIGURE_CAPTION = (
    r"Reference (\textbullet) vs.\ post-drift ($\blacksquare$) detection performance per method, "
    r"averaged over the four \datasetdeux{} domains; the connector length is the concept-drift drop "
    r"$\Delta = \text{S2} - \text{S1}$, shown in green when small ($> -0.05$) and red otherwise."
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
        extractor, domain = r.get("tags.feature_extractor"), r.get("tags.domain")
        # Decision engine: current runs tag it as `decision_engine`; legacy runs used `pipeline`.
        engine = r.get("tags.decision_engine")
        if not isinstance(engine, str):
            engine = r.get("tags.pipeline")
        if not (isinstance(extractor, str) and isinstance(engine, str) and isinstance(domain, str)):
            continue
        data[(extractor, engine, domain)] = {
            "auroc_s1": _num(r.get("metrics.auroc_s1")),
            "auroc_s2": _num(r.get("metrics.auroc_s2")),
            "auprc_s1": _num(r.get("metrics.auprc_s1")),
            "auprc_s2": _num(r.get("metrics.auprc_s2")),
        }
    logger.info(f"Loaded {len(data)} cells from {EXPERIMENT}.")
    _warn_missing(data)
    return data


def _warn_missing(data: Results) -> None:
    """Warn for every expected grid cell that is absent from MLflow or lacks its S1/S2 metrics."""
    metric_keys = ("auroc_s1", "auroc_s2", "auprc_s1", "auprc_s2")
    for extractor, engine, label, _ in METHODS:
        for domain in DOMAINS:
            cell = data.get((extractor, engine, domain))
            if cell is None:
                logger.warning(f"Missing run: {label} / {domain}")
            elif any(cell[k] is None for k in metric_keys):
                logger.warning(f"Incomplete metrics (S1/S2 NaN): {label} / {domain}")


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

# Thesis-facing short labels: feature extractor names title each mini-plot, decision engines
# label the (shared) y ticks.
EXTRACTOR_SHORT = {extractor: EXTRACTOR_LABELS[extractor] for extractor in EXTRACTOR_ORDER}
ENGINE_SHORT = {"lof": "LOF", "ocsvm": "OCSVM", "ae": "AE"}
SHORT_BY_CELL = {(extractor, engine): short for extractor, engine, _, short in METHODS}
# A "method" is one (feature extractor, decision engine) cell -- one dumbbell row in the figure.
LABEL_BY_CELL = {(extractor, engine): label for extractor, engine, label, _ in METHODS}

# x ticks shared by both panels; a panel renders only those within its x-range.
_XTICK_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _fmt_tick(v: float) -> str:
    """Format an x tick: a bare ``0`` for the origin, one decimal otherwise (e.g. ``0.2``, ``1.0``)."""
    return "0" if abs(v) < 1e-9 else f"{v:.1f}"


def _xticks(xmin: float) -> str:
    """Comma-separated ticks from ``_XTICK_GRID`` that fall within ``[xmin, 1]``."""
    return ",".join(_fmt_tick(v) for v in _XTICK_GRID if v >= xmin - 1e-9)


def _extractor_order() -> list[str]:
    """Feature extractors top-to-bottom in the figure, following their METHODS order."""
    seen: list[str] = []
    for extractor, *_ in METHODS:
        if extractor not in seen:
            seen.append(extractor)
    return seen


def _averages(data: Results) -> dict[str, Metrics]:
    """Per method, average each metric over the four domains."""

    def avg_over(extractor: str, engine: str, key: str) -> float | None:
        xs = []
        for domain in DOMAINS:
            cell = data.get((extractor, engine, domain))
            if cell is None or cell[key] is None:
                return None
            xs.append(cell[key])
        return sum(xs) / len(xs)

    out: dict[str, Metrics] = {}
    for extractor, engine, _, short in METHODS:
        out[short] = {
            "s1_auroc": avg_over(extractor, engine, "auroc_s1"),
            "s2_auroc": avg_over(extractor, engine, "auroc_s2"),
            "s1_auprc": avg_over(extractor, engine, "auprc_s1"),
            "s2_auprc": avg_over(extractor, engine, "auprc_s2"),
        }
    return out


def _pgfplotsset(auroc_xmin: float, bottom_title: str, n_plots: int) -> str:
    """The shared ``dumbstack``/``dumbstackp`` mini-plot styles (one x scale per panel).

    ``symbolic y coords``/``yticklabels`` are set per mini-plot instead of here, because a
    feature extractor renders only the decision methods whose averages are complete, so the
    engine rows differ from plot to plot. Mini-plots shrink only when the complete study would
    otherwise exceed the usable thesis-page height.
    """
    plot_height = min(2.4, 22.0 / n_plots)
    return (
        "  % shared style for the per-feature-extractor AUROC mini-plots: identical x scale,\n"
        f"  % x ticks/label hidden by default and re-enabled only on the bottom ({bottom_title}) plot.\n"
        "  \\pgfplotsset{\n"
        "    dumbstack/.style={\n"
        f"      width=0.80\\linewidth, height={plot_height:.2f}cm,\n"
        "      scale only axis,\n"
        f"      xmin={auroc_xmin:.2f}, xmax=1.00,\n"
        f"      xtick={{{_xticks(auroc_xmin)}}},\n"
        "      ytick=data,\n"
        "      tick label style={font=\\scriptsize},\n"
        "      yticklabel style={font=\\tiny},\n"
        "      enlarge y limits=0.12,\n"
        "      xmajorgrids, grid style={dotted},\n"
        "      tick align=outside,\n"
        "      xticklabels={},\n"
        "      title style={font=\\scriptsize, at={(0.01,1)}, anchor=south west, yshift=-5pt},\n"
        "    },\n"
        "    % AUPRC variant: wider x range (y labels are set per mini-plot and hidden there).\n"
        f"    dumbstackp/.style={{dumbstack, xmin=0.00, xtick={{{_xticks(0.0)}}}}},\n"
        "  }"
    )


def _fe_plot(
    extractor: str,
    engines: list[str],
    avgs: dict[str, Metrics],
    s1_key: str,
    s2_key: str,
    style: str,
    *,
    is_bottom: bool,
    xticklabels: str,
    bottom_shows_label: bool,
    show_ylabels: bool,
) -> str:
    """Render one feature-extractor mini-plot (one dumbbell row per available decision method)."""
    title = EXTRACTOR_SHORT[extractor]
    if is_bottom:
        extra = " + label" if bottom_shows_label else ""
        comment = f"    % --- {title} (bottom plot: x-axis ticks{extra} shown here only) ---"
    else:
        comment = f"    % --- {title} ---"
    ycoords = ",".join(ENGINE_SHORT[e] for e in engines)
    axis_opts = f"{style}, title={title}, symbolic y coords={{{ycoords}}}"
    axis_opts += f", yticklabels={{{ycoords if show_ylabels else ''}}}"
    if is_bottom:
        axis_opts += f", xticklabels={{{xticklabels}}}"

    cells = {e: avgs[SHORT_BY_CELL[(extractor, e)]] for e in engines}
    connectors = "\n".join(
        f"        \\addplot[gray, line width=0.8pt, forget plot] coordinates "
        f"{{({cells[e][s2_key]:.4f},{ENGINE_SHORT[e]}) ({cells[e][s1_key]:.4f},{ENGINE_SHORT[e]})}};"
        for e in engines
    )
    s2_marks = " ".join(f"({cells[e][s2_key]:.4f},{ENGINE_SHORT[e]})" for e in engines)
    s1_marks = " ".join(f"({cells[e][s1_key]:.4f},{ENGINE_SHORT[e]})" for e in engines)
    # Drop label per engine: green when the drop is small (> -0.05), red otherwise.
    deltas = []
    for e in engines:
        s1v, s2v = cells[e][s1_key], cells[e][s2_key]
        delta = s2v - s1v
        color = "66BB6A" if delta > -0.05 else "E57373"
        value = rf"\textcolor[HTML]{{{color}}}{{{delta:+.2f}}}"
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
    plots: list[tuple[str, list[str]]],
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
    show_ylabels: bool,
) -> str:
    """Render one dumbbell subfigure: a stack of per-feature-extractor mini-plots."""
    span = xmax - xmin
    lines = [
        "  \\begin{subfigure}{0.49\\linewidth}",
        "    \\centering",
        f"    \\def\\dumbthresh{{{xmin + 0.20 * span:.2f}}}",
        f"    \\def\\dumbthreshR{{{xmax - 0.13 * span:.2f}}}",
    ]
    for i, (extractor, engines) in enumerate(plots):
        lines.append(
            _fe_plot(
                extractor,
                engines,
                avgs,
                s1_key,
                s2_key,
                style,
                is_bottom=i == len(plots) - 1,
                xticklabels=xticklabels,
                bottom_shows_label=bottom_shows_label,
                show_ylabels=show_ylabels,
            )
        )
    lines.append(f"    \\caption{{{caption}}}")
    lines.append("  \\end{subfigure}")
    return "\n".join(lines)


def render_figure(data: Results) -> str:
    """Render the two-panel (AUROC, AUPRC) dumbbell figure for the fully-available methods.

    A feature extractor keeps its mini-plot as long as at least one of its decision methods
    has complete S1/S2 averages; methods missing any domain are dropped from the plot (not
    the whole extractor), so the complete ones still render.
    """
    avgs = _averages(data)
    keys = ("s1_auroc", "s2_auroc", "s1_auprc", "s2_auprc")

    def available_engines(extractor: str) -> list[str]:
        return [e for e in ENGINE_BTT if all(avgs[SHORT_BY_CELL[(extractor, e)]][k] is not None for k in keys)]

    plots: list[tuple[str, list[str]]] = []
    for extractor in _extractor_order():
        engines = available_engines(extractor)
        for e in ENGINE_BTT:
            if e not in engines:
                label = LABEL_BY_CELL[(extractor, e)]
                logger.warning(f"Dropping {label} from dumbbell figure: incomplete S1/S2 averages.")
        if engines:
            plots.append((extractor, engines))
    if not plots:
        raise RuntimeError("No decision method has the S1 and S2 averages required for the figure.")

    auroc_min = min(
        min(avgs[SHORT_BY_CELL[(extractor, e)]]["s1_auroc"], avgs[SHORT_BY_CELL[(extractor, e)]]["s2_auroc"])
        for extractor, engines in plots
        for e in engines
    )
    auroc_xmin = max(0.0, math.floor(auroc_min * 20 - 1) / 20)

    pgfset = _pgfplotsset(auroc_xmin, EXTRACTOR_SHORT[plots[-1][0]], len(plots))

    panels = "\n  \\hfill\n".join(
        [
            _panel(
                plots,
                avgs,
                "s1_auroc",
                "s2_auroc",
                "dumbstack",
                _xticks(auroc_xmin),
                auroc_xmin,
                1.0,
                caption="AUROC",
                bottom_shows_label=True,
                show_ylabels=True,
            ),
            _panel(
                plots,
                avgs,
                "s1_auprc",
                "s2_auprc",
                "dumbstackp",
                _xticks(0.0),
                0.0,
                1.0,
                caption="AUPRC",
                bottom_shows_label=False,
                show_ylabels=False,
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
        "  \\label{fig:concept-drift}\n"
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
    figure_out: Annotated[
        Path, typer.Option(help="Path of the concept-drift dumbbell figure (.tex).")
    ] = DEFAULT_FIGURE_OUT,
) -> None:
    """Load the feature-study concept-drift results from MLflow and write the dumbbell figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow("superviz26-drift"):
        raise typer.Exit(code=1)
    data = load_results()
    _write(render_figure(data), figure_out)


if __name__ == "__main__":
    app()
