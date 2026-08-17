r"""Generate the zero-shot transfer-effect lollipop figure from MLflow.

Combines two sources into one figure:

  * the five already-computed baseline extractors (CountVectorizer, Li, Loginov,
    SecureBERT, CodeT5+) from the "Superviz26-SQL" experiment, and
  * the wider feature-extractor study (every GAUR variant, Qwen3-Emb,
    Sentence-BERT, CodeBERT, RoBERTa, Flan-T5-Small, ...) from "FE-study-LODO".

For each (feature extractor, decision engine) cell it averages AUROC over the
four in-domain scenarios and the four LODO scenarios, then plots the gap
Delta = LODO - in-domain as a lollipop/stem, one groupplot panel per decision
engine (AE, OCSVM, LOF). This reproduces, with real numbers, the structure of
the validated draft at thesis/drafts/transfer-lollipop-preferred.tex.

The feature extractor list (SUPERVIZ26_EXTRACTORS / FE_STUDY_EXTRACTORS below)
is a plain constant -- edit it directly to add, drop, or reorder extractors.
A cell without a finished run (e.g. an extractor not yet implemented, or still
running) still gets its y-tick row; it is simply left without a stem, so the
figure stays complete as new runs land.

MLflow is the single source of truth: the latest finished full-run per
(feature_extractor, decision_engine, scenario) cell wins, across both
experiments, so re-running this after new FE-study runs finish refreshes the
figure.

Usage:
    python -m tools.generate_transfer_lollipop_tex
    python -m tools.generate_transfer_lollipop_tex --figure-out /some/other/path.tex
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
    "/home/gquetel/repos/quetel_phd_latex/thesis/chapters/05-generalization/data/transfer-effect-lollipop.tex"
)

# The two MLflow experiments this figure draws from (see tracking._EXPERIMENT_NAMES):
# the original SuperViz26 grid, and the newer, wider feature-extractor study.
EXPERIMENTS = ["Superviz26-SQL", "FE-study-LODO"]

# (scenario key as tagged in MLflow) -- same protocol/tags in both experiments.
INDOMAIN: list[str] = ["a-a", "b-b", "c-c", "d-d"]
LODO: list[str] = ["bcd-a", "acd-b", "abd-c", "abc-d"]

# Decision engines, top-to-bottom groupplot panel order. Fixed.
ENGINES: list[tuple[str, str]] = [("ae", "AE"), ("ocsvm", "OCSVM"), ("lof", "LOF")]

# --- feature extractors: edit these two lists to add/drop/reorder rows -------

# Already computed in the original Superviz26-SQL grid; rendered as the top block.
SUPERVIZ26_EXTRACTORS: list[tuple[str, str]] = [
    ("cv", "CountVectorizer"),
    ("li", "Li et al."),
    ("loginov", "Loginov et al."),
    ("sbert", "SecureBERT"),
    ("codet5", "CodeT5+"),
]

# Wider feature-extractor study, from FE-study-LODO; rendered as the bottom block.
# "llm2vec-mistral" is not a registered extractor yet (see features/__init__.py);
# it is kept here as a placeholder so its row renders (without a stem) once it
# exists and has finished runs.
FE_STUDY_EXTRACTORS: list[tuple[str, str]] = [
    ("gaur-expert", r"\gaur{} (Expert)"),
    ("gaur-chatgpt", r"\gaur{} (ChatGPT)"),
    ("gaur-claude", r"\gaur{} (Claude)"),
    ("gaur-llama", r"\gaur{} (Llama)"),
    ("gaur-mistral", r"\gaur{} (Mistral)"),
    ("gaur-gpt-oss", r"\gaur{} (GPT-OSS)"),
    ("gaur-ruleid", r"\gaur{} (RuleID)"),
    ("qwen3-emb", "Qwen3-Emb"),
    ("sentbert", "Sentence-BERT"),
    ("codebert", "CodeBERT"),
    ("roberta", "RoBERTa"),
    ("flan-t5", "Flan-T5-Small"),
    ("llm2vec-mistral", "LLM2Vec"),
]

# Combined, top-to-bottom display order.
EXTRACTORS: list[tuple[str, str]] = SUPERVIZ26_EXTRACTORS + FE_STUDY_EXTRACTORS

Metrics = dict[str, float | None]
Results = dict[tuple[str, str, str], Metrics]  # (extractor, engine, scenario) -> {"auroc": ...}

FIGURE_CAPTION = (
    r"Zero-shot transfer effect (LODO AUROC $-$ in-domain AUROC) per feature extractor and "
    r"decision engine, averaged over the four scenarios of each setting. Negative values "
    r"indicate a lower LODO than in-domain AUROC; missing rows correspond to feature "
    r"extractors without a finished run yet."
)


def _num(x: object) -> float | None:
    """Coerce an MLflow metric cell to a float, mapping missing/NaN to None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _warn_unregistered() -> None:
    """Info-log FE-study extractors that aren't in the features registry yet (typo guard)."""
    for key, label in FE_STUDY_EXTRACTORS:
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
        for engine, elabel in ENGINES:
            for sc in INDOMAIN + LODO:
                cell = data.get((extractor, engine, sc))
                if cell is None:
                    logger.warning(f"Missing run: {label} + {elabel} / {sc}")
                elif cell["auroc"] is None:
                    logger.warning(f"Incomplete metrics (AUROC NaN): {label} + {elabel} / {sc}")


# --- delta computation ---------------------------------------------------------


def _avg_auroc(data: Results, extractor: str, engine: str, scenarios: list[str]) -> float | None:
    xs = []
    for sc in scenarios:
        cell = data.get((extractor, engine, sc))
        if cell is None or cell["auroc"] is None:
            return None
        xs.append(cell["auroc"])
    return sum(xs) / len(xs)


def compute_deltas(data: Results) -> dict[tuple[str, str], float | None]:
    """Per (extractor, engine): LODO - in-domain AUROC gap, or None if either average is incomplete."""
    out: dict[tuple[str, str], float | None] = {}
    for extractor, _ in EXTRACTORS:
        for engine, _ in ENGINES:
            id_avg = _avg_auroc(data, extractor, engine, INDOMAIN)
            lodo_avg = _avg_auroc(data, extractor, engine, LODO)
            out[(extractor, engine)] = None if id_avg is None or lodo_avg is None else lodo_avg - id_avg
    return out


# --- lollipop figure -----------------------------------------------------------

DELTASTEM_MACRO = (
    "\\newcommand{\\deltastem}[2]{%\n"
    "  \\addplot[draw=black, line width=0.55pt, forget plot] coordinates {(0,#1) (#2,#1)};\n"
    "  \\addplot[only marks, mark=*, mark size=2.2pt, draw=black, fill=black,\n"
    "    forget plot] coordinates {(#2,#1)};\n"
    "}"
)


def _fmt_tick(v: float) -> str:
    """Format an x tick: a bare ``0`` for the origin, one decimal otherwise."""
    return "0" if abs(v) < 1e-9 else f"{v:.1f}"


def _nice_bounds(vals: list[float]) -> tuple[float, float]:
    """x-axis bounds: the data range padded slightly and rounded out to the nearest 0.1, including 0."""
    lo, hi = min(vals), max(vals)
    xmin = min(0.0, math.floor((lo - 0.02) * 10) / 10)
    xmax = max(0.0, math.ceil((hi + 0.02) * 10) / 10)
    return round(xmin, 2), round(xmax, 2)


def _xticks(xmin: float, xmax: float) -> str:
    """Comma-separated ticks every 0.1 across ``[xmin, xmax]``."""
    n_steps = round((xmax - xmin) / 0.1)
    ticks = [round(xmin + i * 0.1, 2) for i in range(n_steps + 1)]
    return ",".join(_fmt_tick(t) for t in ticks)


def _pgfplotsset(xmin: float, xmax: float) -> str:
    """The shared ``deltaaxis`` style: x range/ticks from the data, y ticks/labels from EXTRACTORS."""
    n = len(EXTRACTORS)
    yticks = ",".join(str(n - i) for i in range(n))
    yticklabels = ",".join(label for _, label in EXTRACTORS)
    # Row spacing scales with the number of extractors, so the figure stays legible
    # as SUPERVIZ26_EXTRACTORS/FE_STUDY_EXTRACTORS grow or shrink.
    height = round(5.9 * n / 15, 1)
    return (
        "\\pgfplotsset{\n"
        "  deltaaxis/.style={\n"
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
    deltas: dict[tuple[str, str], float | None],
    xmin: float,
    xmax: float,
    sep_y: float,
    *,
    is_last: bool,
) -> list[str]:
    """Render one ``\\nextgroupplot`` panel: one stem per extractor row (skipped if delta is None)."""
    n = len(EXTRACTORS)
    opts = f"ylabel={{{elabel}}}" + ("" if is_last else ", xlabel={}")
    lines = [f"      \\nextgroupplot[{opts}]"]
    for i, (extractor, _) in enumerate(EXTRACTORS):
        y = n - i
        d = deltas[(extractor, engine)]
        if d is not None:
            lines.append(f"        \\deltastem{{{y}}}{{{d:+.2f}}}")
    lines.append(f"        \\draw[black, line width=0.6pt] (axis cs:0,0.5) -- (axis cs:0,{n + 0.5});")
    lines.append(f"        \\draw[black!55, dashed, line width=0.45pt] (axis cs:{xmin:.2f},{sep_y}) -- (axis cs:{xmax:.2f},{sep_y});")
    return lines


def render_figure(data: Results) -> str:
    """Render the three-panel (AE, OCSVM, LOF) transfer-effect lollipop figure."""
    _warn_unregistered()
    deltas = compute_deltas(data)
    present = [d for d in deltas.values() if d is not None]
    if not present:
        raise RuntimeError("No (extractor, engine) cell has both in-domain and LODO averages; nothing to plot.")
    xmin, xmax = _nice_bounds(present)

    # Boundary between the Superviz26 block (top) and the FE-study block (bottom).
    sep_y = len(FE_STUDY_EXTRACTORS) + 0.5

    lines = [
        "\\begin{figure}[htb]",
        "  \\centering",
        "  \\begin{tikzpicture}",
        "    \\begin{groupplot}[",
        "      group style={group size=1 by 3, vertical sep=1.0cm},",
        "      deltaaxis,",
        "      xlabel={Transfer effect: LODO AUROC $-$ in-domain AUROC},",
        "    ]",
    ]
    for i, (engine, elabel) in enumerate(ENGINES):
        lines.extend(_panel(engine, elabel, deltas, xmin, xmax, sep_y, is_last=i == len(ENGINES) - 1))
    lines += [
        "    \\end{groupplot}",
        "  \\end{tikzpicture}",
        f"  \\caption{{{FIGURE_CAPTION}}}",
        "  \\label{fig:cross-domain-effect}",
        "\\end{figure}",
    ]
    return f"{DELTASTEM_MACRO}\n\n{_pgfplotsset(xmin, xmax)}\n\n" + "\n".join(lines)


# --- entry point -----------------------------------------------------------------

app = typer.Typer(add_completion=False, help=__doc__)


def _write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    logger.info(f"Wrote {path}")


@app.command()
def main(
    figure_out: Annotated[
        Path, typer.Option(help="Path of the transfer-effect lollipop figure (.tex).")
    ] = DEFAULT_FIGURE_OUT,
) -> None:
    """Load Superviz26-SQL + FE-study-LODO results from MLflow and write the lollipop figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow("superviz26"):
        raise typer.Exit(code=1)
    data = load_results()
    _write(render_figure(data), figure_out)


if __name__ == "__main__":
    app()
