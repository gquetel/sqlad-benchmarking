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
# Labels are paper-facing (AE, SecureBERT) and intentionally differ from the repo's
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
    ("sbert", "ae", "SecureBERT + AE", "SecureBERT+AE"),
    ("sbert", "lof", "SecureBERT + LOF", "SecureBERT+LOF"),
    ("sbert", "ocsvm", "SecureBERT + OCSVM", "SecureBERT+OCSVM"),
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
    r"In-domain (\textbullet) vs.\ LODO ($\blacksquare$) detection performance per pipeline, "
    r"averaged over the four scenarios of each regime; the connector length is the cross-domain "
    r"gap $\Delta = \text{LODO} - \text{ID}$, shown in green when small ($> -0.05$) and red otherwise."
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
    except TypeError, ValueError:
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

# Places the gap label by each connector; \dumbthresh/\dumbthreshR are set per subfigure.
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
    id_key: str,
    lodo_key: str,
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
        f"{{({cells[e][lodo_key]:.4f},{ENGINE_SHORT[e]}) ({cells[e][id_key]:.4f},{ENGINE_SHORT[e]})}};"
        for e in ENGINE_BTT
    )
    lodo_marks = " ".join(f"({cells[e][lodo_key]:.4f},{ENGINE_SHORT[e]})" for e in ENGINE_BTT)
    id_marks = " ".join(f"({cells[e][id_key]:.4f},{ENGINE_SHORT[e]})" for e in ENGINE_BTT)
    # Gap label per engine: green when the gap is small (> -0.05), red otherwise.
    deltas = []
    for e in ENGINE_BTT:
        idv, lodov = cells[e][id_key], cells[e][lodo_key]
        delta = lodov - idv
        color = "green!50!black" if delta > -0.05 else "red!80!black"
        value = rf"\textcolor{{{color}}}{{{delta:+.2f}}}"
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


def _panel(
    extractors: list[str],
    avgs: dict[str, Metrics],
    id_key: str,
    lodo_key: str,
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
                id_key,
                lodo_key,
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
    keys = ("id_auroc", "lodo_auroc", "id_auprc", "lodo_auprc")

    def complete(extractor: str) -> bool:
        return all(avgs[SHORT_BY_CELL[(extractor, e)]][k] is not None for e in ENGINE_BTT for k in keys)

    extractors = [e for e in _extractor_order() if complete(e)]
    if not extractors:
        raise RuntimeError(
            "No feature extractor has all three engines' in-domain and LODO averages required for the figure."
        )

    auroc_min = min(
        min(avgs[SHORT_BY_CELL[(extractor, e)]]["id_auroc"], avgs[SHORT_BY_CELL[(extractor, e)]]["lodo_auroc"])
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
                "id_auroc",
                "lodo_auroc",
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
                "id_auprc",
                "lodo_auprc",
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
    table_out: Annotated[
        Path, typer.Option(help="Path of the compact in-domain vs. LODO summary table (.tex).")
    ] = REPO_ROOT / "reports" / "superviz26" / "superviz26-baselines.tex",
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
