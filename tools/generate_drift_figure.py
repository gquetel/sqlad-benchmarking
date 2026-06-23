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
    ("sbert", "ae", "Secure-BERT + AE", "SBERT+AE"),
    ("sbert", "lof", "Secure-BERT + LOF", "SBERT+LOF"),
    ("sbert", "ocsvm", "Secure-BERT + OCSVM", "SBERT+OCSVM"),
    ("codet5", "ae", "CodeT5+ + AE", "CodeT5+AE"),
    ("codet5", "lof", "CodeT5+ + LOF", "CodeT5+LOF"),
    ("codet5", "ocsvm", "CodeT5+ + OCSVM", "CodeT5+OCSVM"),
]

# The four Superviz26 domains, each evaluated as its own concept-drift cell.
DOMAINS: list[str] = ["a", "b", "c", "d"]

Metrics = dict[str, float | None]
Results = dict[tuple[str, str, str], Metrics]

FIGURE_CAPTION = (
    r"Reference (\textcolor{blue!70!black}{\textbullet}) vs.\ post-drift "
    r"(\textcolor{red!70!black}{$\blacksquare$}) detection performance per pipeline, averaged "
    r"over the four Superviz26 domains; the connector length is the concept-drift drop "
    r"$\Delta = \text{S1} - \text{S2}$. CV~=~CountVectorizer, SBERT~=~Secure-BERT, "
    r"CodeT5~=~CodeT5+."
)


def _num(x: object) -> float | None:
    """Coerce an MLflow metric cell to a float, mapping missing/NaN to None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
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

# Decision-head order within each feature-extractor group (top-to-bottom in the figure).
ENGINE_ORDER = {"lof": 0, "ocsvm": 1, "ae": 2}


def _grouped_order() -> list[str]:
    """Figure short labels top-to-bottom: grouped by feature extractor, then LOF, OCSVM, AE."""
    extractor_pos: dict[str, int] = {}
    for extractor, _, _, _ in PIPELINES:
        extractor_pos.setdefault(extractor, len(extractor_pos))
    ordered = sorted(PIPELINES, key=lambda p: (extractor_pos[p[0]], ENGINE_ORDER.get(p[1], 99)))
    return [short for _, _, _, short in ordered]


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


def _panel(
    order: list[str],
    avgs: dict[str, Metrics],
    s1_key: str,
    s2_key: str,
    xmin: float,
    xmax: float,
    *,
    yticklabels: bool,
    caption: str,
) -> str:
    """Render one dumbbell subfigure (``order`` lists pipelines bottom-to-top)."""
    coords = ",".join(order)
    connector_style = r"\addplot[gray, line width=0.8pt, forget plot]"
    connectors = "\n".join(
        f"        {connector_style} coordinates {{({avgs[s][s2_key]:.4f},{s}) ({avgs[s][s1_key]:.4f},{s})}};"
        for s in order
    )
    s2_marks = " ".join(f"({avgs[s][s2_key]:.4f},{s})" for s in order)
    s1_marks = " ".join(f"({avgs[s][s1_key]:.4f},{s})" for s in order)
    ytick = "" if yticklabels else "\n        yticklabels={},"
    return (
        "  \\begin{subfigure}{0.49\\linewidth}\n"
        "    \\centering\n"
        "    \\begin{tikzpicture}\n"
        "      \\begin{axis}[\n"
        "        width=\\linewidth, height=5.2cm,\n"
        f"        xmin={xmin:.2f}, xmax={xmax:.2f},\n"
        f"        symbolic y coords={{{coords}}},\n"
        f"        ytick=data,{ytick}\n"
        "        tick label style={font=\\scriptsize},\n"
        "        enlarge y limits=0.12,\n"
        "        xmajorgrids, grid style={dotted},\n"
        "        tick align=outside,\n"
        "      ]\n"
        f"{connectors}\n"
        f"        \\addplot[only marks, mark=square*, mark size=2pt, red!70!black] coordinates {{{s2_marks}}};\n"
        f"        \\addplot[only marks, mark=*, mark size=2pt, blue!70!black] coordinates {{{s1_marks}}};\n"
        "      \\end{axis}\n"
        "    \\end{tikzpicture}\n"
        f"    \\caption{{{caption}}}\n"
        "  \\end{subfigure}"
    )


def render_figure(data: Results) -> str:
    """Render the two-panel (AUROC, AUPRC) dumbbell figure for the fully-available pipelines."""
    avgs = _averages(data)
    keys = ("s1_auroc", "s2_auroc", "s1_auprc", "s2_auprc")
    top_to_bottom = [s for s in _grouped_order() if all(avgs[s][k] is not None for k in keys)]
    if not top_to_bottom:
        raise RuntimeError("No pipeline has the full S1 and S2 averages required for the figure.")
    # Rows grouped by feature extractor (FE+LOF, FE+OCSVM, FE+AE); symbolic y coords
    # list bottom-to-top, so reverse the top-to-bottom display order.
    order = list(reversed(top_to_bottom))

    auroc_min = min(min(avgs[s]["s1_auroc"], avgs[s]["s2_auroc"]) for s in order)
    auroc_xmin = max(0.0, math.floor(auroc_min * 20 - 1) / 20)

    panels = "\n  \\hfill\n".join(
        [
            _panel(order, avgs, "s1_auroc", "s2_auroc", auroc_xmin, 1.0, yticklabels=True, caption="AUROC"),
            _panel(order, avgs, "s1_auprc", "s2_auprc", 0.0, 1.0, yticklabels=False, caption="AUPRC"),
        ]
    )
    return (
        "\\begin{figure}[!htb]\n"
        "  \\centering\n"
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
