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

Superviz25 is the domain-A subset of Superviz26, so its size table reuses the same two
experiments, reading only the in-domain ``a-a`` cell (un-averaged); pass ``--dataset superviz25``.

Usage:
    python -m tools.generate_size_tex --dataset superviz26 \
        --table-out ~/repos/quetel_phd_latex/thesis/chapters/03-evaluation/data/superviz26-size.tex
    python -m tools.generate_size_tex --dataset superviz25 \
        --table-out ~/repos/quetel_phd_latex/thesis/chapters/03-evaluation/data/superviz25-size.tex
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import mlflow
import typer

from mlops_sqldetect.tracking import setup_mlflow

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Pipelines in display order: (feature_extractor tag, paper label). All use the AE engine.
# Shared across datasets: Superviz25 is Superviz26's domain A, so both need the same three.
PIPELINES: list[tuple[str, str]] = [
    ("li", "Li + AE"),
    ("sbert", "SecureBERT + AE"),
    ("codet5", "CodeT5+ + AE"),
]

# (scenario key as tagged in MLflow) for each regime.
INDOMAIN = ["a-a", "b-b", "c-c", "d-d"]
LODO = ["bcd-a", "acd-b", "abd-c", "abc-d"]


@dataclass(frozen=True)
class SizeSpec:
    """One dataset's size-sufficiency table: which experiments, scenarios, and captioning.

    ``regimes`` maps a row label to the scenarios averaged into it. A single regime with a
    ``None`` label collapses the Regime column (Superviz25: the lone un-averaged ``a-a`` cell).
    """

    experiment_full: str
    experiment_big: str
    regimes: list[tuple[str | None, list[str]]]
    caption: str
    label: str
    out_name: str


# Superviz25 is the domain-A subset of Superviz26, so its size test reuses Superviz26's
# in-domain a-a cells (Full vs Big) rather than a separate experiment.
SPECS: dict[str, SizeSpec] = {
    "superviz26": SizeSpec(
        experiment_full="Superviz26-SQL",
        experiment_big="Superviz26-SQL-Big",
        regimes=[("In-domain", INDOMAIN), ("LODO", LODO)],
        caption=(
            r"Effect of doubling the training set on AUROC for the three pipelines. AUROC is "
            r"averaged over the four scenarios on the full and Big (2x) sets. $\Delta$ is the "
            r"Big-minus-full difference."
        ),
        label="tab:superviz26-size",
        out_name="superviz26-size.tex",
    ),
    "superviz25": SizeSpec(
        experiment_full="Superviz26-SQL",
        experiment_big="Superviz26-SQL-Big",
        regimes=[(None, ["a-a"])],
        caption=(
            r"Effect of doubling the training set on the AUROC of the three detection pipelines "
            r"on \dataset{}, on the full training set and a Big (2x) variant. $\Delta$ is the "
            r"Big-minus-full difference."
        ),
        label="tab:perfs-big-super",
        out_name="superviz25-size.tex",
    ),
}


def load_aurocs(experiment: str) -> dict[tuple[str, str], float]:
    """Latest finished full-run AUROC per (extractor, scenario) for the AE pipeline."""
    # MLflow filter strings are AND-only (no OR), so the AE head can't be matched
    # server-side across the current `decision_engine` and legacy `pipeline` tags;
    # query the AND-only conditions and reconcile the engine tag client-side below.
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="attributes.status = 'FINISHED' and tags.run_type = 'full-run'",
        output_format="pandas",
    )
    if runs.empty:
        raise RuntimeError(f"No finished full-run found in MLflow experiment {experiment!r}.")
    # Ascending start_time so the latest run of a repeated cell overwrites earlier ones.
    runs = runs.sort_values("start_time")
    out: dict[tuple[str, str], float] = {}
    for _, r in runs.iterrows():
        # Decision engine: current runs tag it `decision_engine`; legacy runs used `pipeline`.
        engine = r.get("tags.decision_engine")
        if not isinstance(engine, str):
            engine = r.get("tags.pipeline")
        if engine != "ae":
            continue
        # Scenario: current runs tag it `scenario`; legacy runs logged it as a param.
        scenario = r.get("tags.scenario")
        if not isinstance(scenario, str):
            scenario = r.get("params.scenario")
        extractor, auroc = r.get("tags.feature_extractor"), r.get("metrics.roc_auc")
        if not (isinstance(extractor, str) and isinstance(scenario, str)):
            continue
        try:
            value = float(auroc)
        except TypeError, ValueError:
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


def render_table(spec: SizeSpec, full: dict[tuple[str, str], float], big: dict[tuple[str, str], float]) -> str:
    """Render the size-sufficiency table (pipeline x regime: Full, Big, Delta).

    A single ``None``-labelled regime drops the Regime column, yielding one row per pipeline.
    """
    has_regimes = any(rl is not None for rl, _ in spec.regimes)
    colspec = "ll|ccc" if has_regimes else "l|ccc"
    span_start = 3 if has_regimes else 2
    header_pad = "    & & " if has_regimes else "    & "
    regime_col = r"\textbf{Regime} & " if has_regimes else ""
    out = [
        r"\begin{table}[htb]",
        r"  \centering",
        r"  \footnotesize",
        rf"  \begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}} {colspec} }}",
        r"    \hline",
        rf"{header_pad}\multicolumn{{3}}{{c}}{{\textbf{{AUROC}}}} \\",
        rf"    \cline{{{span_start}-{span_start + 2}}}",
        rf"    \textbf{{Pipeline}} & {regime_col}\textbf{{Full}} & \textbf{{Big}} & \textbf{{$\Delta$}} \\",
        r"    \hline",
    ]
    for extractor, label in PIPELINES:
        for i, (regime_label, scenarios) in enumerate(spec.regimes):
            first = rf"\multirow{{{len(spec.regimes)}}}{{*}}{{{label}}}" if i == 0 else ""
            head = label if not has_regimes else first
            regime_cell = f"{regime_label} & " if has_regimes else ""
            a = _avg(full, extractor, scenarios)
            b = _avg(big, extractor, scenarios)
            out.append(rf"    {head} & {regime_cell}${a:.4f}$ & ${b:.4f}$ & ${b - a:+.4f}$ \\")
        if has_regimes:
            out.append(r"    \hline")
    if not has_regimes:
        out.append(r"    \hline")
    out += [
        r"  \end{tabular*}",
        rf"  \caption{{{spec.caption}}}",
        rf"  \label{{{spec.label}}}",
        r"\end{table}",
    ]
    return "\n".join(out)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    dataset: Annotated[str, typer.Option(help=f"Which size table to build: one of {sorted(SPECS)}.")] = "superviz26",
    table_out: Annotated[
        Path | None,
        typer.Option(help="Path of the size-sufficiency table (.tex); default: reports/<dataset>/<spec.out_name>."),
    ] = None,
) -> None:
    """Load the Full/Big AE results from MLflow and write the size-sufficiency table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec = SPECS.get(dataset)
    if spec is None:
        raise typer.BadParameter(f"--dataset must be one of {sorted(SPECS)}")
    # Both specs read the Superviz26 experiments; MLflow setup is keyed on the family only.
    if not setup_mlflow("superviz26"):
        raise typer.Exit(code=1)
    full = load_aurocs(spec.experiment_full)
    big = load_aurocs(spec.experiment_big)
    table_out = table_out or REPO_ROOT / "reports" / dataset / spec.out_name
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table_out.write_text(render_table(spec, full, big) + "\n")
    logger.info(f"Wrote {table_out}")


if __name__ == "__main__":
    app()
