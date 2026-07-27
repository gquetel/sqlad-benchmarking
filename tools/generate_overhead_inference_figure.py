"""Generate the IFIPSEC accuracy-vs-inference-latency scatter (RQ2.3) from MLflow.

Run (needs ``tools.benchmark_inference`` to have produced the latency runs first):

    python -m tools.generate_overhead_inference_figure \
        --figure-out ~/repos/quetel_phd_latex/papers/IFIPSEC/sections/overhead-inference.tex

The figure: one point per ``(decision_engine, feature_extractor)`` pipeline on
SuperViz25 -- AUROC (x) vs per-query latency (y, log scale). Colour = feature
extractor, marker = decision engine.

Data: AUROC from the eval experiment, ``infer_ms_per_query`` from the benchmark
experiment, joined per cell; the latest finished run per cell wins. Cells missing
either metric are dropped with a warning, so re-running after new runs or a fresh
benchmark refreshes the figure.

Output: a standalone pgfplots ``tikzpicture`` ``.tex``. The paper preamble must
load ``pgfplots`` (``\\pgfplotsset{compat=1.18}``).
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

# The eval experiment supplies AUROC; the benchmark experiment supplies the latency.
EXPERIMENT = "Superviz25-SQL"
BENCHMARK_EXPERIMENT = "Inference-Latency-Superviz25"

# Default output: the figure's home in the IFIPSEC paper sources.
DEFAULT_FIGURE_OUT = (
    Path.home() / "repos" / "quetel_phd_latex" / "papers" / "IFIPSEC" / "sections" / "overhead-inference.tex"
)

# Feature extractors in legend order: (extractor short name, paper label, HEX colour).
# Colours and labels are paper-facing and mirror the IFIPSEC overhead figure; they
# intentionally differ from the repo's canonical EXTRACTOR_LABELS.
EXTRACTORS: list[tuple[str, str, str]] = [
    ("gaur-expert", "Expert", "DCE775"),
    ("gaur-chatgpt", "ChatGPT", "4FC3F7"),
    ("gaur-claude", "Claude", "E57373"),
    ("gaur-llama", "Llama", "8B7E74"),
    ("gaur-mistral", "Mistral", "9FA8DA"),
    ("gaur-gpt-oss", "GPT-OSS", "66BB6A"),
    ("gaur-ruleid", "Naive", "0D47A1"),
    ("li", "Li et al.", "FFD54F"),
    ("sbert", "SecureBERT", "7E57C2"),
]

# Decision engines in legend order: (engine short name, paper label, pgfplots mark).
ENGINES: list[tuple[str, str, str]] = [
    ("ocsvm", "OCSVM", "square*"),
    ("ae", "AE", "triangle*"),
]

FIGURE_CAPTION = (
    r"AUROC versus per-query inference latency on \dataset{}, one point per pipeline "
    r"(colour: feature extractor; marker: decision engine). Latency is measured with the "
    r"feature cache disabled; \securebert[] runs on GPU, every other pipeline on CPU."
)


def _color_name(extractor: str) -> str:
    """xcolor name for an extractor's colour (letters/digits only, tex-safe)."""
    return "ext" + extractor.replace("-", "")


def _num(x: object) -> float | None:
    """Coerce an MLflow metric cell to a float, mapping missing/NaN to None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None
    return None if math.isnan(v) else v


Point = dict[tuple[str, str], tuple[float, float]]  # (engine, extractor) -> (auroc, ms_per_query)
Cells = dict[tuple[str, str], float]  # (engine, extractor) -> metric value

# Selects the finished detection full-runs whose AUROC feeds the figure.
EVAL_FILTER = "attributes.status = 'FINISHED' and tags.run_type = 'full-run'"


def latest_metric_per_cell(experiment: str, filter_string: str, metric: str) -> Cells:
    """Latest finished run's ``metric`` per ``(engine, extractor)`` cell in ``experiment``."""
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=filter_string,
        output_format="pandas",
    )
    if runs.empty:
        raise RuntimeError(f"No run matching {filter_string!r} in MLflow experiment {experiment!r}.")

    # Ascending start_time so the latest run of a repeated cell overwrites earlier ones.
    runs = runs.sort_values("start_time")
    out: Cells = {}
    for _, r in runs.iterrows():
        engine = r.get("tags.decision_engine")
        extractor = r.get("tags.feature_extractor")
        value = _num(r.get(f"metrics.{metric}"))
        if isinstance(engine, str) and isinstance(extractor, str) and value is not None:
            out[(engine, extractor)] = value
    return out


def eval_aurocs() -> Cells:
    """AUROC per cell from the existing detection full-runs in the eval experiment."""
    return latest_metric_per_cell(EXPERIMENT, EVAL_FILTER, "roc_auc")


def join_points(aurocs: Cells, latencies: Cells) -> Point:
    """Keep the cells present in both, pairing (AUROC, latency)."""
    data: Point = {cell: (aurocs[cell], latencies[cell]) for cell in aurocs.keys() & latencies.keys()}
    logger.info(f"Joined {len(data)} cells (AUROC + latency).")
    return data


def load_points() -> Point:
    """Join each cell's AUROC (eval experiment) with its latency (benchmark experiment)."""
    latencies = latest_metric_per_cell(
        BENCHMARK_EXPERIMENT,
        "attributes.status = 'FINISHED' and tags.run_role = 'benchmark'",
        "infer_ms_per_query",
    )
    return join_points(eval_aurocs(), latencies)


def _warn_missing(data: Point) -> None:
    """Warn for every (engine, extractor) cell absent from the loaded points."""
    for engine, engine_label, _ in ENGINES:
        for extractor, extractor_label, _ in EXTRACTORS:
            if (engine, extractor) not in data:
                logger.warning(f"Missing AUROC or latency: {engine_label} + {extractor_label}")


def render_figure(data: Point) -> str:
    """Render the accuracy-vs-latency scatter as a standalone pgfplots ``tikzpicture``."""
    _warn_missing(data)
    present_extractors = [e for e in EXTRACTORS if any((eng, e[0]) in data for eng, *_ in ENGINES)]
    if not present_extractors:
        raise RuntimeError("No cell has both an AUROC and an infer_ms_per_query metric.")

    aurocs = [xy[0] for xy in data.values()]
    xmin = math.floor(min(aurocs) * 50 - 1) / 50  # one 0.02 step below the smallest AUROC
    xmax = 1.0

    color_defs = "\n".join(f"\\definecolor{{{_color_name(e)}}}{{HTML}}{{{hexc}}}" for e, _, hexc in present_extractors)

    # One \addplot per point; forget plot keeps the auto legend empty, so the
    # hand-built legend below (marker=engine, colour=extractor) is the only one.
    point_lines: list[str] = []
    for engine, _, mark in ENGINES:
        for extractor, _, _ in present_extractors:
            xy = data.get((engine, extractor))
            if xy is None:
                continue
            auroc, latency = xy
            point_lines.append(
                f"      \\addplot[only marks, forget plot, mark={mark}, mark size=3pt, "
                f"color={_color_name(extractor)}, mark options={{draw=black, line width=0.4pt}}] "
                f"coordinates {{({auroc:.5f},{latency:.5f})}};"
            )

    # Legend built by hand: decision engines (black marks) first, then feature-extractor
    # colours (square swatches), so a reader maps marker->engine and colour->extractor.
    legend_lines: list[str] = []
    for _, label, mark in ENGINES:
        legend_lines.append(f"      \\addlegendimage{{only marks, mark={mark}, mark size=3pt, black}}")
        legend_lines.append(f"      \\addlegendentry{{{label}}}")
    for extractor, label, _ in present_extractors:
        legend_lines.append(
            f"      \\addlegendimage{{only marks, mark=square*, mark size=3pt, color={_color_name(extractor)}}}"
        )
        legend_lines.append(f"      \\addlegendentry{{{label}}}")

    axis_opts = (
        "        width=\\linewidth, height=0.55\\linewidth,\n"
        f"        xmin={xmin:.2f}, xmax={xmax:.2f},\n"
        "        ymode=log, log basis y=10,\n"
        "        xlabel={AUROC}, ylabel={Inference time per query (ms)},\n"
        "        xlabel style={font=\\small}, ylabel style={font=\\small},\n"
        "        tick label style={font=\\scriptsize},\n"
        "        grid=both, grid style={dotted},\n"
        "        legend cell align=left,\n"
        "        legend style={font=\\scriptsize, at={(0.02,0.98)}, anchor=north west,\n"
        "          draw=black, fill=white, fill opacity=0.85, text opacity=1, legend columns=2},"
    )
    return (
        "% Auto-generated by tools.generate_overhead_inference_figure — do not edit by hand.\n"
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18} in the preamble.\n"
        "\\begin{figure}[htb]\n"
        "  \\centering\n"
        f"{color_defs}\n"
        "  \\begin{tikzpicture}\n"
        f"    \\begin{{axis}}[\n{axis_opts}\n      ]\n"
        f"{chr(10).join(point_lines)}\n"
        f"{chr(10).join(legend_lines)}\n"
        "    \\end{axis}\n"
        "  \\end{tikzpicture}\n"
        f"  \\caption{{{FIGURE_CAPTION}}}\n"
        "  \\label{fig:overhead-inference}\n"
        "\\end{figure}"
    )


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    figure_out: Annotated[
        Path, typer.Option(help="Path of the overhead-inference figure (.tex).")
    ] = DEFAULT_FIGURE_OUT,
) -> None:
    """Load the SuperViz25 AUROC + latency results from MLflow and write the scatter figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow("superviz25"):
        raise typer.Exit(code=1)
    data = load_points()
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    figure_out.write_text(render_figure(data) + "\n")
    logger.info(f"Wrote {figure_out}")


if __name__ == "__main__":
    app()
