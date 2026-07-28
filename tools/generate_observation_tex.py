r"""Generate every observation-chapter (Chapter 4) LaTeX artefact from MLflow in one call.

One invocation writes ``--out-dir`` (default the chapter's ``data/`` dir) with only the
``.tex`` the thesis observation chapter ``\input``s, all sourced from the ``Superviz25-SQL``
experiment (the same grid the Chapter 3 tables draw from, so the baseline block of
``perfs-vs-sota.tex`` reproduces ``tab:sup25-metrics`` exactly):

  * ``overhead-inference.tex`` (fig:overhead-inference-raw): AUROC versus per-query
    *inference-only* latency scatter -- latencies hardcoded (see RAW_INFERENCE), AUROC
    fetched live from MLflow;
  * ``performance-nov.tex`` (tab:performance-nov): P/R/F1/FPR/AUPRC/AUROC of \gaur{}
    across its seven semantic-model instantiations x OCSVM/LOF/AE; and
  * ``perfs-vs-sota.tex`` (tab:perfs-vs-sota): the three best \gaur{}-based methods by F1
    (auto-selected) above a double rule, over the full reference-baseline block of
    Chapter 3.

Both tables adopt the Chapter 3 ``tab:sup25-metrics`` theme (Model | Detector split via
``\multirow``, siunitx S columns, best value per outcome column in bold, FPR never
bolded). MLflow is the single source of truth: the latest finished full-run per
``(feature_extractor, decision_engine)`` cell wins, and expected cells with no finished
run yet render as ``--``.

Run repo python via ``nix-shell --run '.venv/bin/python -m tools.generate_observation_tex ...'``
(shell.nix sets the LD_LIBRARY_PATH the venv needs).

Usage:
    python -m tools.generate_observation_tex
    python -m tools.generate_observation_tex \
        --out-dir ~/repos/quetel_phd_latex/thesis/chapters/04-observation/data
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

# Default output: the observation chapter's data dir (\input{data/...} resolves here via
# the \import of chapters/04-observation/ in the thesis main).
DEFAULT_OUT_DIR = (
    Path.home() / "repos" / "quetel_phd_latex" / "thesis" / "chapters" / "04-observation" / "data"
)

# The eval experiment holding the full SuperViz25 grid (baselines + every GAUR cell).
EXPERIMENT = "Superviz25-SQL"
EVAL_FILTER = "attributes.status = 'FINISHED' and tags.run_type = 'full-run'"

Metrics = dict[str, float | None]
Results = dict[tuple[str, str], Metrics]  # (feature_extractor, decision_engine) -> metrics


# --- MLflow loading ----------------------------------------------------------


def _num(x: object) -> float | None:
    """Coerce an MLflow metric cell to a float, mapping missing/NaN to None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_results() -> Results:
    """Latest finished full-run per (feature_extractor, decision_engine) cell.

    Loads every cell of the grid (baselines and GAUR alike); each table below filters the
    rows it needs. Legacy runs tag the decision head ``pipeline`` instead of
    ``decision_engine``.
    """
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=EVAL_FILTER,
        output_format="pandas",
    )
    if runs.empty:
        raise RuntimeError(f"No finished full-run found in MLflow experiment {EXPERIMENT!r}.")

    # Ascending start_time so the latest run of a repeated cell overwrites earlier ones.
    runs = runs.sort_values("start_time")
    data: Results = {}
    for _, r in runs.iterrows():
        extractor = r.get("tags.feature_extractor")
        engine = r.get("tags.decision_engine")
        if not isinstance(engine, str):
            engine = r.get("tags.pipeline")
        if not (isinstance(extractor, str) and isinstance(engine, str)):
            continue
        data[(extractor, engine)] = {
            "precision": _num(r.get("metrics.precision")),
            "recall": _num(r.get("metrics.recall")),
            "f1": _num(r.get("metrics.f1")),
            "fpr": _num(r.get("metrics.fpr")),
            "auprc": _num(r.get("metrics.auprc")),
            "auroc": _num(r.get("metrics.roc_auc")),
        }
    logger.info(f"Loaded {len(data)} cells from {EXPERIMENT}.")
    return data


# --- shared metrics-table machinery (mirrors tools.generate_superviz25_tex theme) -----

# Metric columns: (internal key, siunitx table-format, header). Percentages carry one
# decimal (FPR two), curve scores three; the unit lives in the header so the S columns
# stay purely numeric and decimal-aligned.
METRIC_COLUMNS: list[tuple[str, str, str]] = [
    ("precision", "2.1", r"P (\%)"),
    ("recall", "2.1", r"R (\%)"),
    ("f1", "2.1", r"F1 (\%)"),
    ("fpr", "1.2", r"FPR (\%)"),
    ("auprc", "1.3", "AUPRC"),
    ("auroc", "1.3", "AUROC"),
]

# Outcome columns eligible for the per-column best-value bold. FPR is a controlled protocol
# target (~0.1% FPR), not an outcome to maximise, so it is deliberately never bolded.
BOLD_KEYS: tuple[str, ...] = ("precision", "recall", "f1", "auprc", "auroc")

# Decision engines in display order, matching tab:sup25-metrics (AE, LOF, OCSVM).
DETECTOR_ORDER: list[str] = ["ae", "lof", "ocsvm"]
DETECTOR_LABELS: dict[str, str] = {"ae": "AE", "lof": "LOF", "ocsvm": "OCSVM"}

# GAUR semantic models in display order: (feature_extractor tag, header/label macro,
# GAUR-prefixed row-label macro). The plain macro (\chatgpt[]) names the model as a
# Chapter-4 grouping label; the g-prefixed one (\gchatgpt[]) reads "GAUR (ChatGPT)".
GAUR_MODELS: list[tuple[str, str, str]] = [
    ("gaur-expert", r"\expert[]", r"\gexpert[]"),
    ("gaur-chatgpt", r"\chatgpt[]", r"\gchatgpt[]"),
    ("gaur-claude", r"\claude[]", r"\gclaude[]"),
    ("gaur-llama", r"\llama[]", r"\gllama[]"),
    ("gaur-mistral", r"\mistral[]", r"\gmistral[]"),
    ("gaur-gpt-oss", r"\ossgpt[]", r"\gossgpt[]"),
    ("gaur-ruleid", r"\ruleid[]", r"\gruleid[]"),
]
GAUR_GLABEL: dict[str, str] = {ext: glabel for ext, _, glabel in GAUR_MODELS}

# Reference baselines of Chapter 3 for the bottom block of tab:perfs-vs-sota, in the same
# order and spelling as tab:sup25-metrics.
BASELINES: list[tuple[str, str]] = [
    ("cv", "CountVectorizer"),
    ("li", "Li et al."),
    ("loginov", "Loginov"),
    ("sbert", "Secure-BERT"),
    ("codet5", "CodeT5+"),
]

# Source-padding widths so the generated .tex stays column-aligned and diff-friendly.
_METRIC_WIDTH = {"precision": 4, "recall": 4, "f1": 4, "fpr": 4, "auprc": 5, "auroc": 5}
_DETECTOR_WIDTH = 5

NOV_CAPTION = (
    r"Performance metrics (\%) for SQL attack detection with \gaur{} using different "
    r"semantic models. Best value per outcome column in bold."
)
SOTA_CAPTION = (
    r"Performance comparison of the best-performing \gaur{}-based methods (top, selected "
    r"by F1) against the reference baselines of Chapter~\ref{chap:eval} (bottom, above and "
    r"below the double rule). All metrics are percentages (\%) except AUPRC and AUROC. "
    r"Best value per outcome column in bold."
)


def _fmt_metric(v: float | None, key: str) -> str:
    """One metrics-table value: 1-decimal % (precision/recall/f1), 2-decimal % (fpr) or
    3-decimal score (auprc/auroc). ``None`` -> empty (rendered as ``--`` by the caller)."""
    if v is None:
        return ""
    if key in ("auprc", "auroc"):
        return f"{v:.3f}"
    if key == "fpr":
        return f"{v * 100:.2f}"
    return f"{v * 100:.1f}"


def _column_best(cells: list[Metrics], key: str) -> float | None:
    """Best (max) present value of ``key`` across ``cells``, or None if none present."""
    vals = [c[key] for c in cells if c.get(key) is not None]
    return max(vals) if vals else None


def _row_cells(cell: Metrics | None, best: dict[str, float | None]) -> str:
    """The metric cells (siunitx S columns) for one row: value, best-in-bold, ``{--}`` for
    a missing (expected but not-yet-available) cell. ``{--}`` is braced so siunitx treats
    it as text rather than a malformed number."""
    out: list[str] = []
    for key, _, _ in METRIC_COLUMNS:
        v = None if cell is None else cell[key]
        s = _fmt_metric(v, key)
        if not s:
            out.append("{--}".rjust(_METRIC_WIDTH[key]))
        elif key in best and best[key] is not None and v == best[key]:
            out.append(r"\bfseries " + s)  # best cells left unpadded, like the header
        else:
            out.append(s.rjust(_METRIC_WIDTH[key]))
    return " & ".join(out)


def _header_row(group_label: str) -> str:
    r"""``\toprule`` header row: the two bold left labels then the bold metric headers."""
    cols = " & ".join(rf"{{\bfseries {h}}}" for _, _, h in METRIC_COLUMNS)
    return rf"    {{\bfseries {group_label}}} & {{\bfseries Detector}} & {cols} \\"


COLSPEC = "@{}l l " + " ".join(f"S[table-format={tf}]" for _, tf, _ in METRIC_COLUMNS) + "@{}"


def _emit_block(groups: list[tuple[str, str, list[str]]], data: Results, best: dict[str, float | None]) -> list[str]:
    r"""Body lines for a block of ``(row_label, extractor_tag, [engine, ...])`` groups.

    One ``\multirow`` label per group, then one ``& Detector & metrics \\`` line per engine.
    Groups are separated by ``\midrule``; the caller places the block's outer rules and, for
    tab:perfs-vs-sota, the double rule between the two blocks.
    """
    lines: list[str] = []
    for gi, (label, extractor, engines) in enumerate(groups):
        if gi:
            lines.append(r"    \midrule")
        lines.append(rf"    \multirow{{{len(engines)}}}{{*}}{{{label}}}")
        for engine in engines:
            detector = DETECTOR_LABELS[engine].ljust(_DETECTOR_WIDTH)
            lines.append(rf"      & {detector} & {_row_cells(data.get((extractor, engine)), best)} \\")
    return lines


# --- Table 4.3: GAUR semantic models (tab:performance-nov) --------------------


def render_performance_nov(data: Results) -> str:
    """P/R/F1/FPR/AUPRC/AUROC per GAUR semantic model x OCSVM/LOF/AE (tab:performance-nov).

    The full 7x3 grid is always emitted; cells with no finished run render as ``--``. Best
    value per outcome column (across the GAUR grid) is bolded, FPR excluded.
    """
    cells = [c for (ext, _), c in data.items() if ext in GAUR_GLABEL]
    best = {k: _column_best(cells, k) for k in BOLD_KEYS}
    groups = [(label, ext, list(DETECTOR_ORDER)) for ext, label, _ in GAUR_MODELS]

    lines = [
        r"\begin{table}[!htb]",
        r"  \centering",
        r"  % Auto-generated by tools.generate_observation_tex -- do not edit by hand.",
        r"  % Model | Detector split via \multirow, siunitx S columns aligned on the decimal.",
        r"  % Bold marks the best value per outcome column; FPR (a controlled ~0.1\% protocol",
        r"  % target) is never bolded. Expected cells with no finished run render as --.",
        rf"  \begin{{tabular}}{{{COLSPEC}}}",
        r"    \toprule",
        _header_row("Semantic model"),
        r"    \midrule",
        *_emit_block(groups, data, best),
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{NOV_CAPTION}}}",
        r"  \label{tab:performance-nov}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# --- Table 4.4: best GAUR methods vs baselines (tab:perfs-vs-sota) ------------


def _top_gaur_by_f1(data: Results, k: int = 3) -> list[tuple[str, str, str]]:
    r"""The ``k`` GAUR ``(row_label, extractor, [engine])`` cells with the highest F1.

    Cells are grouped by extractor while preserving F1-descending order (so two engines of
    the same model share a ``\multirow`` label), then flattened to per-cell groups of one
    engine each -- the ordering, not the grouping, is what the selection needs.
    """
    scored = [
        (ext, eng, c["f1"])
        for (ext, eng), c in data.items()
        if ext in GAUR_GLABEL and c.get("f1") is not None
    ]
    scored.sort(key=lambda t: t[2], reverse=True)
    top = scored[:k]
    if len(top) < k:
        logger.warning(f"Only {len(top)} GAUR cell(s) with an F1 available; expected {k}.")

    # Collapse consecutive same-extractor selections into one multirow group.
    groups: list[tuple[str, str, list[str]]] = []
    for ext, eng, _ in top:
        if groups and groups[-1][1] == ext:
            groups[-1][2].append(eng)
        else:
            groups.append((GAUR_GLABEL[ext], ext, [eng]))
    return groups


def render_perfs_vs_sota(data: Results) -> str:
    r"""Best GAUR methods (by F1) over the full Chapter-3 baseline block (tab:perfs-vs-sota).

    Top block: the three GAUR cells with the highest F1 (auto-selected). A double rule
    (``\hline\hline``, as in the manual table it replaces) separates them from the bottom
    block, which reproduces every baseline row of tab:sup25-metrics. Best value per outcome
    column is bolded across both blocks; missing baselines render as ``--``.
    """
    top_groups = _top_gaur_by_f1(data)
    base_groups = [(label, ext, list(DETECTOR_ORDER)) for ext, label in BASELINES]

    # Best per column spans every row shown in the table (selected GAUR + all baselines).
    shown: list[Metrics] = []
    for _, ext, engines in top_groups + base_groups:
        shown += [data[(ext, eng)] for eng in engines if (ext, eng) in data]
    best = {k: _column_best(shown, k) for k in BOLD_KEYS}

    lines = [
        r"\begin{table}[!htb]",
        r"  \centering",
        r"  % Auto-generated by tools.generate_observation_tex -- do not edit by hand.",
        r"  % Top block: the 3 best GAUR methods by F1 (auto-selected). Double rule, then the",
        r"  % Chapter-3 baseline block (tab:sup25-metrics). Bold = best per outcome column",
        r"  % across both blocks (FPR excluded); missing cells render as --.",
        rf"  \begin{{tabular}}{{{COLSPEC}}}",
        r"    \toprule",
        _header_row("Method"),
        r"    \midrule",
        *_emit_block(top_groups, data, best),
        r"    \hline\hline",
        *_emit_block(base_groups, data, best),
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{SOTA_CAPTION}}}",
        r"  \label{tab:perfs-vs-sota}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# --- overhead-inference figure (inference-only, "fair" variant) ---------------

# Rows of the IFIP-SEC test split; the raw totals below are over this many queries.
N_SAMPLES = 3353671

# IFIP-SEC raw inference totals, inference-only, one per (engine, extractor):
# "HH:MM:SS.micro" wall-clock to score the whole split (see the provenance note in the repo
# memory). These reproduce the paper text (ruleid AE 1.41 ms, ruleid OCSVM 8.11 ms, sbert
# AE 6.33 ms).
RAW_INFERENCE: list[tuple[str, str, str]] = [
    ("ae", "gaur-expert", "00:16:32.330907"),
    ("ocsvm", "gaur-expert", "01:16:00.408862"),
    ("ae", "gaur-chatgpt", "00:17:23.462143"),
    ("ocsvm", "gaur-chatgpt", "01:20:50.952262"),
    ("ae", "gaur-claude", "00:19:08.950280"),
    ("ocsvm", "gaur-claude", "01:22:38.084346"),
    ("ae", "gaur-llama", "00:17:55.570022"),
    ("ocsvm", "gaur-llama", "01:21:17.337873"),
    ("ae", "gaur-mistral", "00:17:44.664827"),
    ("ocsvm", "gaur-mistral", "01:19:05.302183"),
    ("ae", "gaur-gpt-oss", "00:17:04.488690"),
    ("ocsvm", "gaur-gpt-oss", "01:22:36.662312"),
    ("ae", "gaur-ruleid", "01:19:04.137367"),
    ("ocsvm", "gaur-ruleid", "07:33:50.453943"),
    ("ae", "li", "00:10:38.816208"),
    ("ocsvm", "li", "00:41:01.847022"),
    ("ae", "sbert", "05:53:49.095117"),
    ("ocsvm", "sbert", "14:05:34.200465"),
]

# codet5/loginov infer_ms_per_query from a one-time MLflow benchmark query. Inference-only
# already (no collection step), so they mix cleanly with the raw totals above.
MLFLOW_LATENCY_MS: dict[tuple[str, str], float] = {
    ("ae", "codet5"): 6.742606724239886,
    ("ocsvm", "codet5"): 11.9822740662843,
    ("ae", "loginov"): 0.04221667516976595,
    ("ocsvm", "loginov"): 0.7687311675399542,
}

# Feature extractors in figure-legend order: (extractor tag, legend label, HEX colour).
# Paper-facing colours/labels, intentionally distinct from EXTRACTOR_LABELS.
FIG_EXTRACTORS: list[tuple[str, str, str]] = [
    ("gaur-expert", "Expert", "DCE775"),
    ("gaur-chatgpt", "ChatGPT", "4FC3F7"),
    ("gaur-claude", "Claude", "E57373"),
    ("gaur-llama", "Llama", "8B7E74"),
    ("gaur-mistral", "Mistral", "9FA8DA"),
    ("gaur-gpt-oss", "GPT-OSS", "66BB6A"),
    ("gaur-ruleid", "Naive", "0D47A1"),
    ("li", "Li et al.", "FFD54F"),
    ("sbert", "SecureBERT", "7E57C2"),
    ("codet5", "CodeT5+", "26A69A"),
    ("loginov", "Loginov et al.", "EC407A"),
]

# Decision engines in legend order: (engine tag, legend label, pgfplots mark).
FIG_ENGINES: list[tuple[str, str, str]] = [
    ("ocsvm", "OCSVM", "square*"),
    ("ae", "AE", "triangle*"),
]

FIG_CAPTION = (
    r"AUROC score versus average per-query \emph{inference-only} time (data collection "
    r"excluded), measured on the \dataset{} dataset."
)
FIG_LABEL = "fig:overhead-inference-raw"

# pgfplots mark size for the scatter points and legend swatches (raised from 3pt so the
# overlapping low-latency cluster stays legible).
FIG_MARK_SIZE = "4pt"

FigPoint = dict[tuple[str, str], tuple[float, float]]  # (engine, extractor) -> (auroc, ms)


def _parse_hms(hms: str) -> float:
    """Seconds for a ``HH:MM:SS.micro`` wall-clock string."""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _color_name(extractor: str) -> str:
    """xcolor name for an extractor's colour (letters/digits only, tex-safe)."""
    return "ext" + extractor.replace("-", "")


def figure_points(data: Results) -> FigPoint:
    """Join each cell's live AUROC with its inference-only latency (raw totals + MLflow ms)."""
    latencies: dict[tuple[str, str], float] = {
        (e, x): _parse_hms(t) / N_SAMPLES * 1000.0 for e, x, t in RAW_INFERENCE
    }
    latencies.update(MLFLOW_LATENCY_MS)
    points: FigPoint = {}
    for (extractor, engine), cell in data.items():
        auroc = cell["auroc"]
        latency = latencies.get((engine, extractor))
        if auroc is not None and latency is not None:
            points[(engine, extractor)] = (auroc, latency)
    # Every cell here has a hardcoded latency, so a dropped one is missing only its AUROC.
    for engine, extractor in sorted(latencies.keys() - points.keys()):
        logger.warning(f"No AUROC in MLflow for {engine}+{extractor}; point omitted.")
    return points


def render_figure(points: FigPoint) -> str:
    """Render the AUROC-vs-inference-latency scatter as a standalone pgfplots figure."""
    present = [e for e in FIG_EXTRACTORS if any((eng, e[0]) in points for eng, *_ in FIG_ENGINES)]
    if not present:
        raise RuntimeError("No cell has both an AUROC and an inference-only latency.")

    aurocs = [xy[0] for xy in points.values()]
    xmin = math.floor(min(aurocs) * 50 - 1) / 50  # one 0.02 step below the smallest AUROC
    color_defs = "\n".join(f"  \\definecolor{{{_color_name(e)}}}{{HTML}}{{{hexc}}}" for e, _, hexc in present)

    point_lines: list[str] = []
    for engine, _, mark in FIG_ENGINES:
        for extractor, _, _ in present:
            xy = points.get((engine, extractor))
            if xy is None:
                continue
            auroc, latency = xy
            point_lines.append(
                f"      \\addplot[only marks, forget plot, mark={mark}, mark size={FIG_MARK_SIZE}, "
                f"color={_color_name(extractor)}, mark options={{draw=black, line width=0.4pt}}] "
                f"coordinates {{({auroc:.5f},{latency:.5f})}};"
            )

    legend_lines: list[str] = []
    for _, label, mark in FIG_ENGINES:
        legend_lines.append(f"      \\addlegendimage{{only marks, mark={mark}, mark size={FIG_MARK_SIZE}, black}}")
        legend_lines.append(f"      \\addlegendentry{{{label}}}")
    for extractor, label, _ in present:
        legend_lines.append(
            f"      \\addlegendimage{{only marks, mark=square*, mark size={FIG_MARK_SIZE}, color={_color_name(extractor)}}}"
        )
        legend_lines.append(f"      \\addlegendentry{{{label}}}")

    axis_opts = (
        "        width=\\linewidth, height=0.55\\linewidth,\n"
        f"        xmin={xmin:.2f}, xmax=1.00,\n"
        "        ymode=log, log basis y=10,\n"
        "        xlabel={AUROC}, ylabel={Inference time per query (ms)},\n"
        "        xlabel style={font=\\small}, ylabel style={font=\\small},\n"
        "        tick label style={font=\\scriptsize},\n"
        "        grid=both, grid style={dotted},\n"
        "        legend cell align=left,\n"
        "        legend style={font=\\scriptsize, at={(0.98,0.02)}, anchor=south east,\n"
        "          draw=black, fill=white, fill opacity=0.85, text opacity=1, legend columns=2},"
    )
    return (
        "% Auto-generated by tools.generate_observation_tex -- do not edit by hand.\n"
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
        f"  \\caption{{{FIG_CAPTION}}}\n"
        f"  \\label{{{FIG_LABEL}}}\n"
        "\\end{figure}"
    )


# --- entry point -------------------------------------------------------------


def _write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    logger.info(f"Wrote {path}")


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    out_dir: Annotated[
        Path, typer.Option(help="Directory for the observation-chapter .tex artefacts.")
    ] = DEFAULT_OUT_DIR,
) -> None:
    """Load the SuperViz25 grid from MLflow and write the figure and both tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow("superviz25"):
        logger.error("MLflow tracking unavailable (URI unset or server unreachable).")
        raise typer.Exit(code=1)
    data = load_results()
    _write(render_performance_nov(data), out_dir / "performance-nov.tex")
    _write(render_perfs_vs_sota(data), out_dir / "perfs-vs-sota.tex")
    _write(render_figure(figure_points(data)), out_dir / "overhead-inference.tex")


if __name__ == "__main__":
    app()
