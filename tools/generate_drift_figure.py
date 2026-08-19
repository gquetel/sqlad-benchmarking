r"""Generate the concept-drift dumbbell figure from MLflow.

Same visual language and layout as the zero-shot transfer-effect dumbbell figure
(see ``generate_transfer_dumbbell_tex.py``): one groupplot panel per decision engine
(AE, OCSVM, LOF), one row per feature extractor, ordered handcrafted-then-pretrained
by architectural complexity, with a dashed separator between the two blocks. Each row
is drawn as a reference-vs-post-drift dumbbell instead of an in-domain-vs-LODO one.

Each method is trained once on a benign origin split (S1) and scored on the origin
test set (reference) and a held-out shifted test set (S2, post-drift); for each
(feature extractor, decision engine) cell, AUROC is averaged over the four
\datasetdeux{} domains for S1 and for S2.

The feature extractor list (HANDCRAFTED_EXTRACTORS / PRETRAINED_EXTRACTORS below) is
the same as in ``generate_transfer_dumbbell_tex.py`` -- edit it directly to add, drop,
or reorder extractors. A cell without a finished run still gets its y-tick row; it is
simply left without a dumbbell, so the figure stays complete as new runs land.

MLflow is the single source of truth: the latest finished full-run per
(feature_extractor, decision_engine, domain) cell wins, across both experiments, so
re-running this after new FE-study-CD (or legacy Drift-Superviz26-SQL) runs finish
refreshes the figure.

Usage:
    python -m tools.generate_drift_figure
    python -m tools.generate_drift_figure --figure-out /some/other/path.tex
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
    "/home/gquetel/repos/quetel_phd_latex/thesis/chapters/05-generalization/data/concept-drift-dumbbell.tex"
)

# The two MLflow experiments this figure draws from (see tracking._EXPERIMENT_NAMES):
# the legacy Chapter 3 concept-drift grid (cv/li/loginov/sbert/codet5), and the newer,
# wider feature-extractor study.
EXPERIMENTS = ["Drift-Superviz26-SQL", "FE-study-CD"]

# The four Superviz26 domains, each evaluated as its own concept-drift cell.
DOMAINS: list[str] = ["a", "b", "c", "d"]

# Decision engines, top-to-bottom groupplot panel order. Fixed.
ENGINES: list[tuple[str, str]] = [("ae", "AE"), ("ocsvm", "OCSVM"), ("lof", "LOF")]
ENGINE_SHORT = dict(ENGINES)

# --- feature extractors: edit these two lists to add/drop/reorder rows -------
# Kept identical to generate_transfer_dumbbell_tex.py so both Chapter 5 dumbbell
# figures show the same extractor rows in the same order.

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
# family; rendered as the bottom block.
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

Metrics = dict[str, float | None]
Results = dict[tuple[str, str, str], Metrics]  # (extractor, engine, domain) -> {"auroc": ...}
Averages = dict[tuple[str, str], Metrics]  # (extractor, engine) -> {"s1_auroc": ..., "s2_auroc": ...}

FIGURE_CAPTION = (
    r"Reference (\textbullet) and post-drift ($\blacksquare$) AUROC averaged over the four "
    r"\datasetdeux{} domains. Connectors show the concept-drift drop $\Delta = \text{S2} - \text{S1}$; "
    r"green indicates $\Delta > -0.05$."
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
    """Load the latest finished full-run per cell, indexed by (extractor, engine, domain).

    Searches both EXPERIMENTS at once: cells naturally partition by extractor (the
    Chapter 3 five only ever ran in Drift-Superviz26-SQL, the rest only in FE-study-CD),
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
        }
    logger.info(f"Loaded {len(data)} cells from {EXPERIMENTS}.")
    _warn_missing(data)
    return data


def _warn_missing(data: Results) -> None:
    """Warn for every expected grid cell that is absent from MLflow or lacks S1/S2 AUROC."""
    for extractor, label in EXTRACTORS:
        for engine, elabel in ENGINE_SHORT.items():
            for domain in DOMAINS:
                cell = data.get((extractor, engine, domain))
                if cell is None:
                    logger.warning(f"Missing run: {label} + {elabel} / {domain}")
                elif cell["auroc_s1"] is None or cell["auroc_s2"] is None:
                    logger.warning(f"Incomplete metrics (S1/S2 AUROC NaN): {label} + {elabel} / {domain}")


# --- reference / post-drift averages ------------------------------------------


def _avg_auroc(data: Results, extractor: str, engine: str, key: str) -> float | None:
    xs = []
    for domain in DOMAINS:
        cell = data.get((extractor, engine, domain))
        if cell is None or cell[key] is None:
            return None
        xs.append(cell[key])
    return sum(xs) / len(xs)


def _averages(data: Results) -> Averages:
    """Per (extractor, engine): the reference (S1) and post-drift (S2) AUROC averages."""
    out: Averages = {}
    for extractor, _ in EXTRACTORS:
        for engine in ENGINE_SHORT:
            out[(extractor, engine)] = {
                "s1_auroc": _avg_auroc(data, extractor, engine, "auroc_s1"),
                "s2_auroc": _avg_auroc(data, extractor, engine, "auroc_s2"),
            }
    return out


# --- dumbbell figure -----------------------------------------------------------

# Places the gap label by each connector; \dumbthresh/\dumbthreshR are set once for the figure.
# Defined with \def (not \newcommand) so it can be silently redefined when this figure is
# input alongside the Chapter 3 / transfer-effect dumbbell figures, which define the same macro.
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

DUMBBELL_MACRO = (
    "  \\def\\dumbrow#1#2#3{%\n"
    "    % #1=ycoord #2=s1-auroc #3=s2-auroc -- gray connector plus the reference dot / post-drift square markers.\n"
    "    \\addplot[gray, line width=0.55pt, forget plot] coordinates {(#2,#1) (#3,#1)};\n"
    "    \\addplot[only marks, mark=*, mark size=2.2pt, draw=black, fill=black, forget plot] coordinates {(#2,#1)};\n"
    "    \\addplot[only marks, mark=square*, mark size=2pt, draw=black, fill=black, forget plot] coordinates {(#3,#1)};\n"
    "  }"
)


def _fmt_tick(v: float) -> str:
    """Format an x tick: a bare ``0`` for the origin, one decimal otherwise."""
    return "0" if abs(v) < 1e-9 else f"{v:.1f}"


def _nice_bounds(vals: list[float]) -> tuple[float, float]:
    """x-axis bounds: the data range padded slightly and rounded out to the nearest 0.05."""
    lo, hi = min(vals), max(vals)
    xmin = max(0.0, math.floor((lo - 0.02) * 20) / 20)
    xmax = min(1.0, math.ceil((hi + 0.02) * 20) / 20)
    return round(xmin, 2), round(xmax, 2)


def _xticks(xmin: float, xmax: float) -> str:
    """Comma-separated ticks every 0.1 across ``[xmin, xmax]``."""
    n_steps = round((xmax - xmin) / 0.1)
    ticks = [round(xmin + i * 0.1, 2) for i in range(n_steps + 1)]
    return ",".join(_fmt_tick(t) for t in ticks)


def _pgfplotsset(xmin: float, xmax: float) -> str:
    """The shared ``dumbaxis`` style: x range/ticks from the data, y ticks/labels from EXTRACTORS."""
    n = len(EXTRACTORS)
    yticks = ",".join(str(n - i) for i in range(n))
    yticklabels = ",".join(label for _, label in EXTRACTORS)
    # Row spacing scales with the number of extractors, so the figure stays legible
    # as HANDCRAFTED_EXTRACTORS/PRETRAINED_EXTRACTORS grow or shrink.
    height = round(5.9 * n / 15, 1)
    return (
        "\\pgfplotsset{\n"
        "  dumbaxis/.style={\n"
        f"    xmin={xmin:.2f}, xmax={xmax:.2f},\n"
        f"    xtick={{{_xticks(xmin, xmax)}}},\n"
        "    xmajorgrids, grid style={dotted, gray},\n"
        "    tick label style={font=\\scriptsize},\n"
        "    label style={font=\\small},\n"
        f"    ytick={{{yticks}}},\n"
        f"    yticklabels={{{yticklabels}}},\n"
        "    yticklabel style={font=\\scriptsize, anchor=east},\n"
        "    enlarge y limits=0.02,\n"
        "    axis line style={black!65},\n"
        "    tick align=outside,\n"
        "    width=0.82\\linewidth,\n"
        f"    height={height}cm,\n"
        "  },\n"
        "}"
    )


def _panel(
    engine: str,
    elabel: str,
    avgs: Averages,
    sep_y: float,
    xmin: float,
    xmax: float,
    *,
    is_last: bool,
) -> list[str]:
    """Render one ``\\nextgroupplot`` panel: one dumbbell row per extractor (skipped if incomplete)."""
    n = len(EXTRACTORS)
    opts = f"ylabel={{{elabel}}}" + ("" if is_last else ", xlabel={}")
    lines = [f"      \\nextgroupplot[{opts}]"]
    for i, (extractor, _) in enumerate(EXTRACTORS):
        y = n - i
        cell = avgs[(extractor, engine)]
        s1v, s2v = cell["s1_auroc"], cell["s2_auroc"]
        if s1v is None or s2v is None:
            continue
        lines.append(f"        \\dumbrow{{{y}}}{{{s1v:.4f}}}{{{s2v:.4f}}}")
        delta = s2v - s1v
        color = "66BB6A" if delta > -0.05 else "E57373"
        value = rf"\textcolor[HTML]{{{color}}}{{{delta:+.2f}}}"
        lines.append(f"        \\dumbdelta{{{min(s1v, s2v):.4f}}}{{{max(s1v, s2v):.4f}}}{{{y}}}{{{value}}}")
    lines.append(
        f"        \\draw[black!55, dashed, line width=0.45pt] (axis cs:{xmin:.2f},{sep_y}) -- (axis cs:{xmax:.2f},{sep_y});"
    )
    return lines


def render_figure(data: Results) -> str:
    """Render the three-panel (AE, OCSVM, LOF) reference-vs-post-drift dumbbell figure.

    A row is drawn only for the (extractor, engine) cells with complete S1/S2 averages;
    others are simply left without a dumbbell, so the panel stays complete as new runs land.
    A dashed line separates the handcrafted block (top) from the pretrained block (bottom).
    """
    _warn_unregistered()
    avgs = _averages(data)
    present = [
        v
        for cell in avgs.values()
        for v in (cell["s1_auroc"], cell["s2_auroc"])
        if v is not None
    ]
    if not present:
        raise RuntimeError("No (extractor, engine) cell has both reference and post-drift averages; nothing to plot.")
    xmin, xmax = _nice_bounds(present)

    # Boundary between the handcrafted block (top) and the pretrained block (bottom).
    sep_y = len(PRETRAINED_EXTRACTORS) + 0.5
    span = xmax - xmin

    lines = [
        "\\begin{figure}[p]",
        "  \\centering",
        f"  \\def\\dumbthresh{{{xmin + 0.20 * span:.2f}}}",
        f"  \\def\\dumbthreshR{{{xmax - 0.13 * span:.2f}}}",
        "  \\begin{tikzpicture}",
        "    \\begin{groupplot}[",
        "      group style={group size=1 by 3, vertical sep=1.0cm},",
        "      dumbaxis,",
        "      xlabel={AUROC},",
        "    ]",
    ]
    for i, (engine, elabel) in enumerate(ENGINES):
        lines.extend(_panel(engine, elabel, avgs, sep_y, xmin, xmax, is_last=i == len(ENGINES) - 1))
    lines += [
        "    \\end{groupplot}",
        "  \\end{tikzpicture}",
        f"  \\caption{{{FIGURE_CAPTION}}}",
        "  \\label{fig:concept-drift-dumbbell}",
        "\\end{figure}",
    ]
    return f"{DUMBDELTA_MACRO}\n\n{DUMBBELL_MACRO}\n\n{_pgfplotsset(xmin, xmax)}\n\n" + "\n".join(lines)


# --- entry point -----------------------------------------------------------------

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
