"""Generate the SuperViz26 baseline LaTeX artefacts from MLflow.

Loads every finished full-run cell of the ``Superviz26-SQL`` experiment and emits
two LaTeX files for the benchmark paper:

  * the compact in-domain vs. LODO summary table (per pipeline: AUROC and AUPRC
    averaged over each regime, plus the cross-domain gap Delta = LODO - ID), and
  * the in-domain vs. LODO dumbbell figure (one AUROC panel and one AUPRC panel),
    which visualises the same gap per pipeline.

The detailed per-scenario tables (in-domain and LODO, with confidence intervals and
F1) can still be written via ``--detailed-out``.

MLflow is the single source of truth: the latest finished full-run per
``(feature_extractor, pipeline, scenario)`` cell wins, so re-running this after
new runs (e.g. when the Loginov / CodeT5+ cells finish) refreshes both files.
Missing cells are rendered as placeholders (--) in the table and dropped from the
figure.

Usage:
    python -m tools.generate_baselines_tex \
        --table-out ~/repos/quetel_phd_latex/papers/superviz26/sections/superviz26-baselines.tex \
        --figure-out ~/repos/quetel_phd_latex/papers/superviz26/sections/superviz26-dumbbell.tex
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

# The MLflow experiment holding the standard SuperViz26 grid (see tracking._EXPERIMENT_NAMES).
EXPERIMENT = "Superviz26-SQL"

# Pipelines in display order: (feature_extractor, engine, paper label, figure short label).
# Labels are paper-facing (AE, Secure-BERT) and intentionally differ from the repo's
# canonical PIPELINE_LABELS / EXTRACTOR_LABELS ("Autoencoder", "SecureBERT").
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

# (scenario key as tagged in MLflow, LaTeX label).
INDOMAIN: list[tuple[str, str]] = [
    ("a-a", r"$A\to A$"),
    ("b-b", r"$B\to B$"),
    ("c-c", r"$C\to C$"),
    ("d-d", r"$D\to D$"),
]
LODO: list[tuple[str, str]] = [
    ("bcd-a", r"$BCD\to A$"),
    ("acd-b", r"$ACD\to B$"),
    ("abd-c", r"$ABD\to C$"),
    ("abc-d", r"$ABC\to D$"),
]

Metrics = dict[str, float | None]
Results = dict[tuple[str, str, str], Metrics]

IN_CAPTION = (
    r"In-domain detection performance of the fifteen reference pipelines on \datasetlodo{}. "
    r"For each pipeline, the four in-domain scenarios and their average are reported. "
    r"Missing entries (--) correspond to runs not yet available."
)
LODO_CAPTION = (
    r"Leave-one-domain-out (\gls{lodo}) detection performance of the fifteen reference pipelines "
    r"on \datasetlodo{}. For each pipeline, the four held-out scenarios and their average are "
    r"reported. Missing entries (--) correspond to runs not yet available."
)
FIGURE_CAPTION = (
    r"In-domain (\textcolor{blue!70!black}{\textbullet}) vs.\ LODO "
    r"(\textcolor{red!70!black}{$\blacksquare$}) detection performance per pipeline, averaged "
    r"over the four scenarios of each regime; the connector length is the cross-domain gap "
    r"$\Delta = \text{LODO} - \text{ID}$. CV~=~CountVectorizer, SBERT~=~Secure-BERT, "
    r"CodeT5~=~CodeT5+."
)
SUMMARY_CAPTION = (
    r"In-domain (ID) vs.\ leave-one-domain-out (LODO) detection performance of the fifteen "
    r"reference pipelines on \datasetlodo{}, averaged over the four scenarios of each regime. "
    r"$\Delta = \text{LODO} - \text{ID}$ is the cross-domain gap. Missing entries (--) correspond "
    r"to runs not yet available."
)


def _num(x: object) -> float | None:
    """Coerce an MLflow metric cell to a float, mapping missing/NaN to None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_results() -> Results:
    """Load the latest finished full-run per cell, indexed by (extractor, engine, scenario)."""
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
        extractor, engine, scenario = r.get("tags.feature_extractor"), r.get("tags.pipeline"), r.get("tags.dataset")
        if not (isinstance(extractor, str) and isinstance(engine, str) and isinstance(scenario, str)):
            continue
        data[(extractor, engine, scenario)] = {
            "auroc": _num(r.get("metrics.roc_auc")),
            "auroc_ci": _num(r.get("metrics.auroc_ci")),
            "auprc": _num(r.get("metrics.auprc")),
            "auprc_ci": _num(r.get("metrics.auprc_ci")),
            "f1": _num(r.get("metrics.f1")),
        }
    logger.info(f"Loaded {len(data)} cells from {EXPERIMENT}.")
    return data


# --- detailed appendix tables ------------------------------------------------


def _fmt_ci(v: float | None, ci: float | None) -> str:
    if v is None:
        return "--"
    return f"{v:.4f}" if ci is None else rf"${v:.4f} \pm {ci:.4f}$"


def _fmt_f1(v: float | None) -> str:
    return "--" if v is None else rf"{v * 100:.2f}\%"


def _avg(values: list[Metrics], key: str) -> float | None:
    xs = [v[key] for v in values if v[key] is not None]
    return sum(xs) / len(xs) if xs else None


def _table_rows(data: Results, scenarios: list[tuple[str, str]]) -> list[tuple[str, str, str, str, str]]:
    """Build (label, scenario, auroc, auprc, f1) rows for one regime, including the average."""
    rows: list[tuple[str, str, str, str, str]] = []
    for extractor, engine, label, _ in PIPELINES:
        present = [data.get((extractor, engine, sc)) for sc, _ in scenarios]
        for (_, sclabel), cell in zip(scenarios, present, strict=True):
            if cell is None:
                rows.append((label, sclabel, "--", "--", "--"))
            else:
                rows.append(
                    (
                        label,
                        sclabel,
                        _fmt_ci(cell["auroc"], cell["auroc_ci"]),
                        _fmt_ci(cell["auprc"], cell["auprc_ci"]),
                        _fmt_f1(cell["f1"]),
                    )
                )
        full = [c for c in present if c is not None]
        if len(full) == len(scenarios):
            rows.append(
                (
                    label,
                    "Average",
                    _fmt_ci(_avg(full, "auroc"), _avg(full, "auroc_ci")),
                    _fmt_ci(_avg(full, "auprc"), _avg(full, "auprc_ci")),
                    _fmt_f1(_avg(full, "f1")),
                )
            )
        else:
            rows.append((label, "Average", "--", "--", "--"))
    return rows


def _render_table(data: Results, scenarios: list[tuple[str, str]], caption: str, label: str) -> str:
    rows = _table_rows(data, scenarios)
    n = len(scenarios) + 1
    out = [
        r"\begin{table}[!htb]",
        r"  \centering",
        r"  \footnotesize",
        r"  \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} ll|ccc }",
        r"    \hline",
        r"    \textbf{Pipeline} & \textbf{Scenario} & \textbf{AUROC} & \textbf{AUPRC} & \textbf{F1} \\",
        r"    \hline",
    ]
    for i, (lab, sclabel, auroc, auprc, f1) in enumerate(rows):
        pos = i % n
        first = rf"\multirow{{{n}}}{{*}}{{{lab}}}" if pos == 0 else ""
        if pos == n - 1:
            out.append(r"    \cline{2-5}")
        out.append(f"    {first} & {sclabel} & {auroc} & {auprc} & {f1} \\\\")
        if pos == n - 1:
            out.append(r"    \hline")
    out += [
        r"  \end{tabular*}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(out)


def render_tables(data: Results) -> str:
    """Render the in-domain and LODO per-scenario tables, separated by a blank line."""
    return "\n\n".join(
        [
            _render_table(data, INDOMAIN, IN_CAPTION, "tab:superviz26-indomain"),
            _render_table(data, LODO, LODO_CAPTION, "tab:superviz26-lodo"),
        ]
    )


# --- compact summary table ---------------------------------------------------


def _fmt_avg(v: float | None) -> str:
    return "--" if v is None else f"{v:.4f}"


def _fmt_delta(v: float | None) -> str:
    return "--" if v is None else rf"${v:+.4f}$"


def _gap(lodo: float | None, indom: float | None) -> float | None:
    return lodo - indom if (lodo is not None and indom is not None) else None


def render_summary_table(data: Results) -> str:
    """Render the compact in-domain vs. LODO summary table (averages and the gap per pipeline)."""
    avgs = _averages(data)
    out = [
        r"\begin{table}[!htb]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        rf"  \caption{{{SUMMARY_CAPTION}}}",
        r"  \label{tab:superviz26-summary}",
        r"  \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l ccc ccc }",
        r"    \toprule",
        r"    & \multicolumn{3}{c}{\textbf{AUROC}} & \multicolumn{3}{c}{\textbf{AUPRC}} \\",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"    \textbf{Pipeline} & ID & LODO & $\Delta$ & ID & LODO & $\Delta$ \\",
        r"    \midrule",
    ]
    prev_extractor = None
    for extractor, _, label, short in PIPELINES:
        if prev_extractor is not None and extractor != prev_extractor:
            out.append(r"    \midrule")
        m = avgs[short]
        out.append(
            f"    {label} & {_fmt_avg(m['id_auroc'])} & {_fmt_avg(m['lodo_auroc'])} "
            f"& {_fmt_delta(_gap(m['lodo_auroc'], m['id_auroc']))} "
            f"& {_fmt_avg(m['id_auprc'])} & {_fmt_avg(m['lodo_auprc'])} "
            f"& {_fmt_delta(_gap(m['lodo_auprc'], m['id_auprc']))} \\\\"
        )
        prev_extractor = extractor
    out += [
        r"    \bottomrule",
        r"  \end{tabular*}",
        r"\end{table}",
    ]
    return "\n".join(out)


# --- dumbbell figure ---------------------------------------------------------


def _averages(data: Results) -> dict[str, Metrics]:
    """Per pipeline, average each metric over the in-domain and LODO scenarios."""

    def avg_over(extractor: str, engine: str, scenarios: list[tuple[str, str]], key: str) -> float | None:
        xs = []
        for sc, _ in scenarios:
            cell = data.get((extractor, engine, sc))
            if cell is None or cell[key] is None:
                return None
            xs.append(cell[key])
        return sum(xs) / len(xs)

    out: dict[str, Metrics] = {}
    for extractor, engine, _, short in PIPELINES:
        out[short] = {
            "id_auroc": avg_over(extractor, engine, INDOMAIN, "auroc"),
            "lodo_auroc": avg_over(extractor, engine, LODO, "auroc"),
            "id_auprc": avg_over(extractor, engine, INDOMAIN, "auprc"),
            "lodo_auprc": avg_over(extractor, engine, LODO, "auprc"),
        }
    return out


def _panel(
    order: list[str],
    avgs: dict[str, Metrics],
    id_key: str,
    lodo_key: str,
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
        f"        {connector_style} coordinates {{({avgs[s][lodo_key]:.4f},{s}) ({avgs[s][id_key]:.4f},{s})}};"
        for s in order
    )
    lodo_marks = " ".join(f"({avgs[s][lodo_key]:.4f},{s})" for s in order)
    id_marks = " ".join(f"({avgs[s][id_key]:.4f},{s})" for s in order)
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
        f"        \\addplot[only marks, mark=square*, mark size=2pt, red!70!black] coordinates {{{lodo_marks}}};\n"
        f"        \\addplot[only marks, mark=*, mark size=2pt, blue!70!black] coordinates {{{id_marks}}};\n"
        "      \\end{axis}\n"
        "    \\end{tikzpicture}\n"
        f"    \\caption{{{caption}}}\n"
        "  \\end{subfigure}"
    )


def render_figure(data: Results) -> str:
    """Render the two-panel (AUROC, AUPRC) dumbbell figure for the fully-available pipelines."""
    avgs = _averages(data)
    order_desc = [
        s
        for s, m in sorted(
            avgs.items(), key=lambda kv: (kv[1]["lodo_auroc"] is not None, kv[1]["lodo_auroc"] or 0.0), reverse=True
        )
        if all(m[k] is not None for k in ("id_auroc", "lodo_auroc", "id_auprc", "lodo_auprc"))
    ]
    if not order_desc:
        raise RuntimeError("No pipeline has the full in-domain and LODO averages required for the figure.")
    # Plot best-generalising pipeline on top: symbolic y coords list bottom-to-top.
    order = list(reversed(order_desc))

    auroc_min = min(min(avgs[s]["id_auroc"], avgs[s]["lodo_auroc"]) for s in order)
    auroc_xmin = max(0.0, math.floor(auroc_min * 20 - 1) / 20)

    panels = "\n  \\hfill\n".join(
        [
            _panel(order, avgs, "id_auroc", "lodo_auroc", auroc_xmin, 1.0, yticklabels=True, caption="AUROC"),
            _panel(order, avgs, "id_auprc", "lodo_auprc", 0.0, 1.0, yticklabels=False, caption="AUPRC"),
        ]
    )
    return (
        "\\begin{figure}[!htb]\n"
        "  \\centering\n"
        f"{panels}\n"
        f"  \\caption{{{FIGURE_CAPTION}}}\n"
        "  \\label{fig:superviz26-dumbbell}\n"
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
    table_out: Annotated[Path, typer.Option(help="Path of the compact in-domain vs. LODO summary table (.tex).")] = REPO_ROOT
    / "reports"
    / "superviz26"
    / "superviz26-baselines.tex",
    figure_out: Annotated[Path, typer.Option(help="Path of the dumbbell figure (.tex).")] = REPO_ROOT
    / "reports"
    / "superviz26"
    / "superviz26-dumbbell.tex",
    detailed_out: Annotated[
        Path | None, typer.Option(help="If set, also write the detailed per-scenario tables (.tex) here.")
    ] = None,
) -> None:
    """Load the SuperViz26 results from MLflow and write the summary table, the figure, and (optionally) the detailed tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow("superviz26"):
        raise typer.Exit(code=1)
    data = load_results()
    _write(render_summary_table(data), table_out)
    _write(render_figure(data), figure_out)
    if detailed_out is not None:
        _write(render_tables(data), detailed_out)


if __name__ == "__main__":
    app()
