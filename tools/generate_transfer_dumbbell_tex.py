r"""Generate the zero-shot transfer-effect dumbbell figure from MLflow.

Combines two blocks into one figure, split by representation kind rather than by
source experiment:

  * the handcrafted extractors (CountVectorizer, TF-IDF, Li, Loginov, every GAUR
    variant, Kakisim), and
  * the pretrained extractors, ordered by architectural complexity (encoder-only
    RoBERTa/SecureBERT/CodeBERT/Sentence-BERT, then encoder-decoder
    Flan-T5-Small/CodeT5+, then LLM2Vec last as the most complex, decoder-derived
    family).

Runs for both blocks are pulled from either of the two MLflow experiments below
(see EXPERIMENTS); which block an extractor renders in is purely a display choice.

For each (feature extractor, decision engine) cell it averages AUROC over the
four in-domain scenarios and the four LODO scenarios, then plots both averages as
a dumbbell (in-domain dot vs. LODO square, connected by a gray stem), one mini-plot
per feature extractor stacked in a single column -- the same visual language as the
Chapter 3 in-domain-vs-LODO dumbbell figure (see ``generate_baselines_tex.py``) and
the Chapter 5 concept-drift dumbbell figure (see ``generate_drift_figure.py``).

The feature extractor list (HANDCRAFTED_EXTRACTORS / PRETRAINED_EXTRACTORS below)
is a plain constant -- edit it directly to add, drop, or reorder extractors.
A cell without a finished run (e.g. an extractor not yet implemented, or still
running) drops that decision method's row from its mini-plot; an extractor with no
complete decision method is dropped entirely, so the figure stays complete as new
runs land.

MLflow is the single source of truth: the latest finished full-run per
(feature_extractor, decision_engine, scenario) cell wins, across both
experiments, so re-running this after new FE-study runs finish refreshes the
figure.

Usage:
    python -m tools.generate_transfer_dumbbell_tex
    python -m tools.generate_transfer_dumbbell_tex --figure-out /some/other/path.tex
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Annotated

import mlflow
import typer

from sqlad_benchmarking.features import EXTRACTORS as REGISTERED_EXTRACTORS
from sqlad_benchmarking.tracking import setup_mlflow

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_OUT = Path(
    "/home/gquetel/repos/quetel_phd_latex/thesis/chapters/05-generalization/data/transfer-effect-dumbbell.tex"
)

# The two MLflow experiments this figure draws from (see tracking._EXPERIMENT_NAMES):
# the original SuperViz26 grid, and the newer, wider feature-extractor study.
EXPERIMENTS = ["Superviz26-SQL", "FE-study-LODO"]

# (scenario key as tagged in MLflow) -- same protocol/tags in both experiments.
INDOMAIN: list[str] = ["a-a", "b-b", "c-c", "d-d"]
LODO: list[str] = ["bcd-a", "acd-b", "abd-c", "abc-d"]

# Decision-engine rows inside each per-feature-extractor mini-plot. ENGINE_ORDER is the
# top-to-bottom screen order; the symbolic y coords list them bottom-to-top (ENGINE_BTT).
ENGINE_ORDER = {"lof": 0, "ocsvm": 1, "ae": 2}
ENGINE_BTT = sorted(ENGINE_ORDER, key=lambda e: ENGINE_ORDER[e], reverse=True)  # ae, ocsvm, lof
ENGINE_SHORT = {"lof": "LOF", "ocsvm": "OCSVM", "ae": "AE"}

# --- feature extractors: edit these two lists to add/drop/reorder rows -------

# Handcrafted, explicitly-defined representations; rendered as the top block.
HANDCRAFTED_EXTRACTORS: list[tuple[str, str]] = [
    ("cv", "CountVectorizer"),
    ("tfidf", "TF-IDF"),
    ("li", "Li et al."),
    ("loginov", "Loginov et al."),
    ("gaur-expert", r"\gaur{} (Expert)"),
    ("gaur-chatgpt", r"\gaur{} (ChatGPT)"),
    ("gaur-claude", r"\gaur{} (Claude)"),
    ("gaur-llama", r"\gaur{} (Llama)"),
    ("gaur-mistral", r"\gaur{} (Mistral)"),
    ("gaur-gpt-oss", r"\gaur{} (GPT-OSS)"),
    ("gaur-ruleid", r"\gaur{} (RuleID)"),
    ("kakisim", "Kakisim"),
]

# Pretrained representations, ordered by architectural complexity: encoder-only,
# then encoder-decoder, then LLM2Vec last as the most complex, decoder-derived
# family; rendered as the bottom block. LLM2Vec's runs are currently FAILED
# (LOF scenarios) or missing a full-run tag (AE/OCSVM), so its row renders
# without a stem until a clean full run lands.
PRETRAINED_EXTRACTORS: list[tuple[str, str]] = [
    ("roberta", "RoBERTa"),
    ("sbert", "SecureBERT"),
    ("codebert", "CodeBERT"),
    ("sentbert", "Sentence-BERT"),
    ("flan-t5", "Flan-T5-Small"),
    ("codet5", "CodeT5+"),
    ("llm2vec", "LLM2Vec"),
]

# Combined, top-to-bottom display order.
EXTRACTORS: list[tuple[str, str]] = HANDCRAFTED_EXTRACTORS + PRETRAINED_EXTRACTORS
EXTRACTOR_SHORT = dict(EXTRACTORS)
_HANDCRAFTED_KEYS = {key for key, _ in HANDCRAFTED_EXTRACTORS}

Metrics = dict[str, float | None]
Results = dict[tuple[str, str, str], Metrics]  # (extractor, engine, scenario) -> {"auroc": ...}
Averages = dict[tuple[str, str], Metrics]  # (extractor, engine) -> {"id_auroc": ..., "lodo_auroc": ...}

FIGURE_CAPTION = (
    r"In-domain (\textbullet) vs.\ LODO ($\blacksquare$) AUROC per feature extractor and decision "
    r"engine, averaged over the four scenarios of each setting; the connector length is the "
    r"cross-domain gap $\Delta = \text{LODO} - \text{ID}$, shown in green when small ($> -0.05$) "
    r"and red otherwise. Missing rows correspond to feature extractors without a finished run yet."
)


def _num(x: object) -> float | None:
    """Coerce an MLflow metric cell to a float, mapping missing/NaN to None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _warn_unregistered() -> None:
    """Info-log pretrained extractors that aren't in the features registry yet (typo guard)."""
    for key, label in PRETRAINED_EXTRACTORS:
        if key not in REGISTERED_EXTRACTORS:
            logger.info(f"{label!r} ({key}) is not a registered feature extractor yet; its row will render empty.")


def load_results() -> Results:
    """Load the latest finished full-run per cell, indexed by (extractor, engine, scenario).

    Searches both EXPERIMENTS at once: cells naturally partition by extractor (the
    Superviz26 five only ever ran in Superviz26-SQL, the rest only in FE-study-LODO),
    so a single query is enough and stays correct if a cell is later rerun elsewhere.
    """
    runs = mlflow.search_runs(
        experiment_names=EXPERIMENTS,
        filter_string="attributes.status = 'FINISHED' and tags.run_type = 'full-run'",
        output_format="pandas",
    )
    if runs.empty:
        raise RuntimeError(f"No finished full-run found in MLflow experiments {EXPERIMENTS!r}.")

    # Ascending start_time so the latest run of a repeated cell overwrites earlier ones.
    runs = runs.sort_values("start_time")
    data: Results = {}
    for _, r in runs.iterrows():
        extractor = r.get("tags.feature_extractor")
        # Decision engine: current runs tag it as `decision_engine`; legacy runs used `pipeline`.
        engine = r.get("tags.decision_engine")
        if not isinstance(engine, str):
            engine = r.get("tags.pipeline")
        # Scenario key (a-a, bcd-a, ...): current runs tag it as `scenario`; legacy runs
        # (before the dataset/scenario tag split) carried it under `dataset` instead.
        scenario = r.get("tags.scenario")
        if not isinstance(scenario, str):
            scenario = r.get("tags.dataset")
        if not (isinstance(extractor, str) and isinstance(engine, str) and isinstance(scenario, str)):
            continue
        data[(extractor, engine, scenario)] = {"auroc": _num(r.get("metrics.roc_auc"))}
    logger.info(f"Loaded {len(data)} cells from {EXPERIMENTS}.")
    _warn_missing(data)
    return data


def _warn_missing(data: Results) -> None:
    """Warn for every expected grid cell that is absent from MLflow or lacks AUROC."""
    for extractor, label in EXTRACTORS:
        for engine, elabel in ENGINE_SHORT.items():
            for sc in INDOMAIN + LODO:
                cell = data.get((extractor, engine, sc))
                if cell is None:
                    logger.warning(f"Missing run: {label} + {elabel} / {sc}")
                elif cell["auroc"] is None:
                    logger.warning(f"Incomplete metrics (AUROC NaN): {label} + {elabel} / {sc}")


# --- in-domain / LODO averages ------------------------------------------------


def _avg_auroc(data: Results, extractor: str, engine: str, scenarios: list[str]) -> float | None:
    xs = []
    for sc in scenarios:
        cell = data.get((extractor, engine, sc))
        if cell is None or cell["auroc"] is None:
            return None
        xs.append(cell["auroc"])
    return sum(xs) / len(xs)


def _averages(data: Results) -> Averages:
    """Per (extractor, engine): the in-domain and LODO AUROC averages."""
    out: Averages = {}
    for extractor, _ in EXTRACTORS:
        for engine in ENGINE_SHORT:
            out[(extractor, engine)] = {
                "id_auroc": _avg_auroc(data, extractor, engine, INDOMAIN),
                "lodo_auroc": _avg_auroc(data, extractor, engine, LODO),
            }
    return out


# --- dumbbell figure -----------------------------------------------------------

# Places the gap label by each connector; \dumbthresh/\dumbthreshR are set once for the figure.
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

# x ticks shared by every mini-plot; only those within the panel's x-range are kept.
_XTICK_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _fmt_tick(v: float) -> str:
    """Format an x tick: a bare ``0`` for the origin, one decimal otherwise (e.g. ``0.2``, ``1.0``)."""
    return "0" if abs(v) < 1e-9 else f"{v:.1f}"


def _xticks(xmin: float) -> str:
    """Comma-separated ticks from ``_XTICK_GRID`` that fall within ``[xmin, 1]``."""
    return ",".join(_fmt_tick(v) for v in _XTICK_GRID if v >= xmin - 1e-9)


def _pgfplotsset(xmin: float, n_plots: int) -> str:
    """The shared ``dumbstack`` mini-plot style (one x scale for the whole figure).

    ``symbolic y coords``/``yticklabels`` are set per mini-plot instead of here, because a
    feature extractor renders only the decision methods whose averages are complete, so the
    engine rows differ from plot to plot. Mini-plots shrink only when the complete study would
    otherwise exceed the usable thesis-page height.
    """
    plot_height = min(2.4, 22.0 / n_plots)
    return (
        "  % shared style for the per-feature-extractor AUROC mini-plots: identical x scale,\n"
        "  % x ticks/label hidden by default and re-enabled only on the bottom-most plot.\n"
        "  \\pgfplotsset{\n"
        "    dumbstack/.style={\n"
        f"      width=0.80\\linewidth, height={plot_height:.2f}cm,\n"
        "      scale only axis,\n"
        f"      xmin={xmin:.2f}, xmax=1.00,\n"
        f"      xtick={{{_xticks(xmin)}}},\n"
        "      ytick=data,\n"
        "      tick label style={font=\\scriptsize},\n"
        "      yticklabel style={font=\\tiny},\n"
        "      enlarge y limits=0.12,\n"
        "      xmajorgrids, grid style={dotted},\n"
        "      tick align=outside,\n"
        "      xticklabels={},\n"
        "      title style={font=\\scriptsize, at={(0.01,1)}, anchor=south west, yshift=-5pt},\n"
        "    },\n"
        "  }"
    )


def _fe_plot(
    extractor: str,
    engines: list[str],
    avgs: Averages,
    *,
    is_bottom: bool,
    xticklabels: str,
) -> str:
    """Render one feature-extractor mini-plot (one dumbbell row per available decision method)."""
    title = EXTRACTOR_SHORT[extractor]
    comment = (
        f"    % --- {title} (bottom plot: x-axis ticks + label shown here only) ---"
        if is_bottom
        else f"    % --- {title} ---"
    )
    ycoords = ",".join(ENGINE_SHORT[e] for e in engines)
    axis_opts = f"dumbstack, title={title}, symbolic y coords={{{ycoords}}}, yticklabels={{{ycoords}}}"
    if is_bottom:
        axis_opts += f", xticklabels={{{xticklabels}}}, xlabel={{AUROC}}"

    cells = {e: avgs[(extractor, e)] for e in engines}
    connectors = "\n".join(
        f"        \\addplot[gray, line width=0.8pt, forget plot] coordinates "
        f"{{({cells[e]['lodo_auroc']:.4f},{ENGINE_SHORT[e]}) ({cells[e]['id_auroc']:.4f},{ENGINE_SHORT[e]})}};"
        for e in engines
    )
    lodo_marks = " ".join(f"({cells[e]['lodo_auroc']:.4f},{ENGINE_SHORT[e]})" for e in engines)
    id_marks = " ".join(f"({cells[e]['id_auroc']:.4f},{ENGINE_SHORT[e]})" for e in engines)
    # Gap label per engine: green when the gap is small (> -0.05), red otherwise.
    deltas = []
    for e in engines:
        idv, lodov = cells[e]["id_auroc"], cells[e]["lodo_auroc"]
        delta = lodov - idv
        color = "66BB6A" if delta > -0.05 else "E57373"
        value = rf"\textcolor[HTML]{{{color}}}{{{delta:+.2f}}}"
        deltas.append(
            f"        \\dumbdelta{{{min(idv, lodov):.4f}}}{{{max(idv, lodov):.4f}}}{{{ENGINE_SHORT[e]}}}{{{value}}}"
        )
    delta_block = "\n".join(deltas)
    suffix = "" if is_bottom else "\\\\[-6pt]"
    return (
        f"{comment}\n"
        "    \\begin{tikzpicture}\n"
        f"      \\begin{{axis}}[{axis_opts}]\n"
        f"{connectors}\n"
        f"        \\addplot[only marks, mark=square*, mark size=2pt, black] coordinates {{{lodo_marks}}};\n"
        f"        \\addplot[only marks, mark=*, mark size=2pt, black] coordinates {{{id_marks}}};\n"
        f"{delta_block}\n"
        "      \\end{axis}\n"
        f"    \\end{{tikzpicture}}{suffix}"
    )


def render_figure(data: Results) -> str:
    """Render the single-column AUROC dumbbell figure for the fully-available methods.

    A feature extractor keeps its mini-plot as long as at least one of its decision methods
    has complete ID/LODO averages; methods missing any scenario are dropped from the plot (not
    the whole extractor), so the complete ones still render. A thin rule separates the
    handcrafted block (top) from the pretrained block (bottom).
    """
    _warn_unregistered()
    avgs = _averages(data)

    def available_engines(extractor: str) -> list[str]:
        return [
            e
            for e in ENGINE_BTT
            if avgs[(extractor, e)]["id_auroc"] is not None and avgs[(extractor, e)]["lodo_auroc"] is not None
        ]

    plots: list[tuple[str, list[str]]] = []
    for extractor, label in EXTRACTORS:
        engines = available_engines(extractor)
        for e in ENGINE_BTT:
            if e not in engines:
                logger.warning(f"Dropping {label} + {ENGINE_SHORT[e]} from dumbbell figure: incomplete ID/LODO averages.")
        if engines:
            plots.append((extractor, engines))
    if not plots:
        raise RuntimeError("No decision method has the in-domain and LODO averages required for the figure.")

    auroc_min = min(
        min(avgs[(extractor, e)]["id_auroc"], avgs[(extractor, e)]["lodo_auroc"])
        for extractor, engines in plots
        for e in engines
    )
    xmin = max(0.0, math.floor(auroc_min * 20 - 1) / 20)
    xticklabels = _xticks(xmin)

    lines = [
        "\\begin{figure}[p]",
        "  \\centering",
        DUMBDELTA_MACRO,
        _pgfplotsset(xmin, len(plots)),
        f"  \\def\\dumbthresh{{{xmin + 0.20 * (1.0 - xmin):.2f}}}",
        f"  \\def\\dumbthreshR{{{1.0 - 0.13 * (1.0 - xmin):.2f}}}",
    ]
    for i, (extractor, engines) in enumerate(plots):
        is_bottom = i == len(plots) - 1
        # A block separator between the handcrafted and pretrained rows, drawn once, right
        # before the first pretrained mini-plot that actually renders.
        if i > 0 and plots[i - 1][0] in _HANDCRAFTED_KEYS and extractor not in _HANDCRAFTED_KEYS:
            lines.append("  \\noindent\\rule{0.80\\linewidth}{0.3pt}\\\\[2pt]")
        lines.append(
            "  " + _fe_plot(extractor, engines, avgs, is_bottom=is_bottom, xticklabels=xticklabels).replace("\n", "\n  ")
        )
    lines += [
        f"  \\caption{{{FIGURE_CAPTION}}}",
        "  \\label{fig:transfer-dumbbell}",
        "\\end{figure}",
    ]
    return "\n".join(lines)


# --- entry point -----------------------------------------------------------------

app = typer.Typer(add_completion=False, help=__doc__)


def _write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    logger.info(f"Wrote {path}")


@app.command()
def main(
    figure_out: Annotated[
        Path, typer.Option(help="Path of the transfer-effect dumbbell figure (.tex).")
    ] = DEFAULT_FIGURE_OUT,
) -> None:
    """Load Superviz26-SQL + FE-study-LODO results from MLflow and write the dumbbell figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow("superviz26"):
        raise typer.Exit(code=1)
    data = load_results()
    _write(render_figure(data), figure_out)


if __name__ == "__main__":
    app()
