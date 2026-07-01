"""Generate the SuperViz26 size-sufficiency LaTeX table from MLflow.

The size experiment (Section "Size evaluation" of the benchmark paper) asks whether
the full per-domain training sets are large enough, by retraining the most performant
AE pipelines on the Big pools (2x the size of full) and re-evaluating on the test splits.

The two training-set sizes live in two MLflow experiments that share the same tags:

  * Full       -> ``Superviz26-SQL``       (see tracking._EXPERIMENT_NAMES)
  * Big (2x)   -> ``Superviz26-SQL-Big``

For each pipeline (AE on top of the Li, CodeT5+ and SecureBERT extractors) and each
regime (in-domain, LODO), this emits the AUROC averaged over the four scenarios on the
full and Big sets, plus the big-minus-full difference Delta. MLflow is the
single source of truth: the latest finished full-run per ``(extractor, scenario)`` cell
wins, so re-running this refreshes the table.

Usage:
    python -m tools.generate_size_tex \
        --table-out ~/repos/quetel_phd_latex/papers/superviz26/data/superviz26-size.tex
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

# MLflow experiments holding the two training-set sizes (see tracking._EXPERIMENT_NAMES).
EXPERIMENT_FULL = "Superviz26-SQL"
EXPERIMENT_BIG = "Superviz26-SQL-Big"

# Pipelines in display order: (feature_extractor tag, paper label). All use the AE engine.
PIPELINES: list[tuple[str, str]] = [
    ("li", "Li + AE"),
    ("sbert", "SecureBERT + AE"),
    ("codet5", "CodeT5+ + AE"),
]

# (scenario key as tagged in MLflow) for each regime.
INDOMAIN = ["a-a", "b-b", "c-c", "d-d"]
LODO = ["bcd-a", "acd-b", "abd-c", "abc-d"]
REGIMES: list[tuple[str, list[str]]] = [("In-domain", INDOMAIN), ("LODO", LODO)]

def _caption() -> str:
    """Caption listing the pipelines in display order (e.g. 'A, B and C')."""
    labels = [label.replace(" ", "~") for _, label in PIPELINES]
    listed = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
    return (
        rf"Effect of doubling the training set beyond the full per-domain pools on the AUROC of the "
        rf"{listed} pipelines. For each pipeline and regime, the AUROC is averaged over the four "
        r"scenarios on the full and Big (2x) training sets; the $\Delta$ column reports the "
        r"big-minus-full difference."
    )


def load_aurocs(experiment: str) -> dict[tuple[str, str], float]:
    """Latest finished full-run AUROC per (extractor, scenario) for the AE pipeline."""
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="attributes.status = 'FINISHED' and tags.run_type = 'full-run' and tags.pipeline = 'ae'",
        output_format="pandas",
    )
    if runs.empty:
        raise RuntimeError(f"No finished AE full-run found in MLflow experiment {experiment!r}.")
    # Ascending start_time so the latest run of a repeated cell overwrites earlier ones.
    runs = runs.sort_values("start_time")
    out: dict[tuple[str, str], float] = {}
    for _, r in runs.iterrows():
        extractor, scenario, auroc = r.get("tags.feature_extractor"), r.get("tags.dataset"), r.get("metrics.roc_auc")
        if not (isinstance(extractor, str) and isinstance(scenario, str)):
            continue
        try:
            value = float(auroc)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            out[(extractor, scenario)] = value
    logger.info(f"Loaded {len(out)} AE cells from {experiment}.")
    return out


def _avg(aurocs: dict[tuple[str, str], float], extractor: str, scenarios: list[str]) -> float:
    """Mean AUROC over ``scenarios``; raises if any cell is missing (the table needs all four)."""
    try:
        return sum(aurocs[(extractor, sc)] for sc in scenarios) / len(scenarios)
    except KeyError as exc:
        raise RuntimeError(f"Missing AUROC for {extractor!r} scenario {exc.args[0][1]!r}.") from None


def render_table(full: dict[tuple[str, str], float], big: dict[tuple[str, str], float]) -> str:
    """Render the size-sufficiency table (pipeline x regime: Full, Big, Delta)."""
    out = [
        r"\begin{table}[htb]",
        r"  \centering",
        r"  \footnotesize",
        r"  \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} ll|ccc }",
        r"    \hline",
        r"    & & \multicolumn{3}{c}{\textbf{AUROC}} \\",
        r"    \cline{3-5}",
        r"    \textbf{Pipeline} & \textbf{Regime} & \textbf{Full} & \textbf{Big} & \textbf{$\Delta$} \\",
        r"    \hline",
    ]
    for extractor, label in PIPELINES:
        for i, (regime_label, scenarios) in enumerate(REGIMES):
            first = rf"\multirow{{{len(REGIMES)}}}{{*}}{{{label}}}" if i == 0 else ""
            a = _avg(full, extractor, scenarios)
            b = _avg(big, extractor, scenarios)
            out.append(rf"    {first} & {regime_label} & ${a:.4f}$ & ${b:.4f}$ & ${b - a:+.4f}$ \\")
        out.append(r"    \hline")
    out += [
        r"  \end{tabular*}",
        rf"  \caption{{{_caption()}}}",
        r"  \label{tab:superviz26-size}",
        r"\end{table}",
    ]
    return "\n".join(out)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    table_out: Annotated[Path, typer.Option(help="Path of the size-sufficiency table (.tex).")] = REPO_ROOT
    / "reports"
    / "superviz26"
    / "superviz26-size.tex",
) -> None:
    """Load the Full/Big AE results from MLflow and write the size-sufficiency table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow("superviz26"):
        raise typer.Exit(code=1)
    full = load_aurocs(EXPERIMENT_FULL)
    big = load_aurocs(EXPERIMENT_BIG)
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table_out.write_text(render_table(full, big) + "\n")
    logger.info(f"Wrote {table_out}")


if __name__ == "__main__":
    app()
