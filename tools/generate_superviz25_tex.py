r"""Generate every SuperViz25 LaTeX artefact from MLflow in a single call.

One invocation writes ``--out-dir`` (default ``reports/superviz25-tex/``) with only the
LaTeX ``.tex`` artefacts the thesis ``\input``s, all sourced from the ``Superviz25-SQL``
experiment:

  * the combined ROC and precision-recall figures as native pgfplots ``.tex`` -- rebuilt
    from every cell's ``curve_data`` artifacts. This folds in (and replaces) the former
    ``aggregate_curves`` command;
  * ``superviz25-metrics.tex`` (\label{tab:sup25-metrics}): precision, recall, F1, FPR,
    AUPRC and AUROC per method; and
  * ``superviz25-recall.tex`` (\label{tab:sup25-recall}): per-technique recall heatmap.

The downloaded curve CSVs (and, with ``--rasters``, the PNG/PDF figures) go to
``--cache-dir`` (a throwaway temp dir by default), so ``--out-dir`` stays ``.tex``-only.

MLflow is the single source of truth: the latest finished full-run per
``(feature_extractor, decision_engine)`` cell wins. Cells with no finished run yet render
as empty fields in the tables and are dropped from the figures, so re-running after new
runs finish (e.g. a pending Secure-BERT / LOF cell) refreshes every artefact.

Usage:
    python -m tools.generate_superviz25_tex
    python -m tools.generate_superviz25_tex --out-dir ~/repos/quetel_phd_latex/thesis/chapters/03-evaluation/data
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
import typer

from mlops_sqldetect.datasets import FAMILIES
from mlops_sqldetect.evaluate_suite import ALL_METHODS, _all_scenarios
from mlops_sqldetect.features import EXTRACTORS
from mlops_sqldetect.tracking import setup_mlflow
from mlops_sqldetect.visualize import (
    Curve,
    plot_combined_curves_tikz,
    plot_combined_pr,
    plot_combined_roc,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The MLflow experiment holding the SuperViz25 grid (see tracking._EXPERIMENT_NAMES);
# it carries a single scenario, tagged "dataset".
DATASET = "superviz25"
EXPERIMENT = "Superviz25-SQL"
SCENARIO = "dataset"

# Methods in display order: (feature_extractor tag, decision_engine tag, paper label).
# Labels are thesis-facing and intentionally differ from the repo's canonical extractor
# names ("Secure-BERT" vs "SecureBERT", "AE" vs "Autoencoder").
METHODS: list[tuple[str, str, str]] = [
    ("cv", "ae", "CountVectorizer + AE"),
    ("cv", "lof", "CountVectorizer + LOF"),
    ("cv", "ocsvm", "CountVectorizer + OCSVM"),
    ("li", "ae", "Li + AE"),
    ("li", "lof", "Li + LOF"),
    ("li", "ocsvm", "Li + OCSVM"),
    ("loginov", "ae", "Loginov + AE"),
    ("loginov", "lof", "Loginov + LOF"),
    ("loginov", "ocsvm", "Loginov + OCSVM"),
    ("sbert", "ae", "Secure-BERT + AE"),
    ("sbert", "lof", "Secure-BERT + LOF"),
    ("sbert", "ocsvm", "Secure-BERT + OCSVM"),
    ("codet5", "ae", "CodeT5+ + AE"),
    ("codet5", "lof", "CodeT5+ + LOF"),
    ("codet5", "ocsvm", "CodeT5+ + OCSVM"),
]

# Thesis-facing extractor labels for the combined figure legend, taken from the same
# paper labels as the tables (the "<Extractor> + <Engine>" prefix). This keeps the
# figure legend spelling ("Secure-BERT") consistent with the metrics table instead of
# the package's canonical EXTRACTOR_LABELS ("SecureBERT").
PAPER_EXTRACTOR_LABELS: dict[str, str] = {ext: label.split(" + ")[0] for ext, _, label in METHODS}

# Per-technique recall columns: (MLflow metric suffix, column header). The metric key is
# ``recall_<suffix>`` (see evaluate_suite._mlflow_key). Insider attacks are forced to false
# negatives during evaluation, so their recall is 0 wherever a run exists.
# Headers are abbreviated to keep the heatmap within \linewidth; the metric key stays the
# full technique name (recall_<suffix>).
TECHNIQUES: list[tuple[str, str]] = [
    ("union", "Union"),
    ("boolean", "Bool."),
    ("stacked", "Stack."),
    ("error", "Error"),
    ("time", "Time"),
    ("inline", "Inline"),
    ("insider", "Insider"),
]

# Metric columns of the performance table: (internal key, siunitx table-format, header).
# Percentages carry one decimal, curve scores three; the unit lives in the header so the
# S columns stay purely numeric and decimal-aligned.
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

# Paper-facing row labels for the split Embedding | Detector columns. Embeddings use the
# thesis spelling ("Li et al.", "Secure-BERT"); the combined-figure legend keeps the terser
# PAPER_EXTRACTOR_LABELS ("Li") since a plot legend has less room.
EMBEDDING_LABELS: dict[str, str] = {
    "cv": "CountVectorizer",
    "li": "Li et al.",
    "loginov": "Loginov",
    "sbert": "Secure-BERT",
    "codet5": "CodeT5+",
}
DETECTOR_LABELS: dict[str, str] = {"ae": "AE", "lof": "LOF", "ocsvm": "OCSVM"}

METRICS_CAPTION = "Performance metrics for studied novelty detection methods. Best value per outcome column in bold."
RECALL_CAPTION = (
    r"Per-technique recall (\%) heatmap for studied detection methods. Cell shading "
    r"encodes recall from 0\% (white) to 100\% (green)."
)

Metrics = dict[str, float | None]
Results = dict[tuple[str, str], Metrics]


# --- MLflow loading ----------------------------------------------------------


def _num(x: object) -> float | None:
    """Coerce an MLflow metric cell to a float, mapping missing/NaN to None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None
    return None if math.isnan(v) else v


def load_results() -> Results:
    """Load the latest finished full-run per cell, indexed by (extractor, engine)."""
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
        extractor = r.get("tags.feature_extractor")
        # Decision engine: current runs tag it as `decision_engine`; legacy runs used `pipeline`.
        engine = r.get("tags.decision_engine")
        if not isinstance(engine, str):
            engine = r.get("tags.pipeline")
        if not (isinstance(extractor, str) and isinstance(engine, str)):
            continue
        cell: Metrics = {
            "precision": _num(r.get("metrics.precision")),
            "recall": _num(r.get("metrics.recall")),
            "f1": _num(r.get("metrics.f1")),
            "fpr": _num(r.get("metrics.fpr")),
            "auprc": _num(r.get("metrics.auprc")),
            "auroc": _num(r.get("metrics.roc_auc")),
        }
        for suffix, _ in TECHNIQUES:
            cell[f"recall_{suffix}"] = _num(r.get(f"metrics.recall_{suffix}"))
        data[(extractor, engine)] = cell
    logger.info(f"Loaded {len(data)} cells from {EXPERIMENT}.")
    _warn_missing(data)
    return data


def _warn_missing(data: Results) -> None:
    """Warn for every expected grid cell that is absent from MLflow."""
    for extractor, engine, label in METHODS:
        if (extractor, engine) not in data:
            logger.warning(f"Missing run: {label}")


# --- table rendering ---------------------------------------------------------


def _fmt_metric(v: float | None, key: str) -> str:
    """One metrics-table value: 1-decimal % (precision/recall/f1), 2-decimal % (fpr) or
    3-decimal score (auprc/auroc). ``None`` -> empty cell (run not available yet)."""
    if v is None:
        return ""
    if key in ("auprc", "auroc"):
        return f"{v:.3f}"
    if key == "fpr":
        return f"{v * 100:.2f}"
    return f"{v * 100:.1f}"


def _fmt_heat(v: float | None) -> str:
    r"""Recall as a heatmap cell ``\hc{p}`` (integer percent). ``None`` -> empty cell."""
    return "" if v is None else rf"\hc{{{round(v * 100)}}}"


# Source-padding widths so the generated .tex stays column-aligned and diff-friendly.
_METRIC_WIDTH = {"precision": 4, "recall": 4, "f1": 4, "fpr": 4, "auprc": 5, "auroc": 5}
_HEAT_WIDTH = 7
_DETECTOR_WIDTH = 5


def _extractor_groups() -> list[tuple[str, list[str]]]:
    """Group METHODS into ``(extractor, [engine, ...])`` runs, preserving display order."""
    groups: list[tuple[str, list[str]]] = []
    for extractor, engine, _ in METHODS:
        if not groups or groups[-1][0] != extractor:
            groups.append((extractor, []))
        groups[-1][1].append(engine)
    return groups


def _column_best(data: Results, key: str) -> float | None:
    """Best (max) present value of ``key`` across all cells, or None if none present."""
    vals = [c[key] for c in data.values() if c.get(key) is not None]
    return max(vals) if vals else None


def _header_row(headers: list[str]) -> str:
    """`\\toprule` header row: bold Embedding/Detector labels then the bold column headers."""
    cols = " & ".join(rf"{{\bfseries {h}}}" for h in headers)
    return rf"    {{\bfseries Embedding}} & {{\bfseries Detector}} & {cols} \\"


def _group_rows(row_cells) -> list[str]:
    """Emit the multirow-grouped body: one ``\\multirow`` line per embedding then one line
    per detector. ``row_cells(extractor, engine)`` returns that row's already-formatted
    cells (from the Detector column onward)."""
    lines: list[str] = []
    for gi, (extractor, engines) in enumerate(_extractor_groups()):
        if gi:
            lines.append(r"    \midrule")
        lines.append(rf"    \multirow{{{len(engines)}}}{{*}}{{{EMBEDDING_LABELS[extractor]}}}")
        for engine in engines:
            detector = DETECTOR_LABELS[engine].ljust(_DETECTOR_WIDTH)
            lines.append(rf"      & {detector} & {row_cells(extractor, engine)} \\")
    return lines


def render_metrics_table(data: Results) -> str:
    """Precision/Recall/F1/FPR/AUPRC/AUROC per method (tab:sup25-metrics).

    Embedding | Detector split via \\multirow, siunitx S columns, best value per outcome
    column in bold (FPR excluded), missing runs left as empty cells.
    """
    best = {k: _column_best(data, k) for k in BOLD_KEYS}
    colspec = "@{}l l " + " ".join(f"S[table-format={tf}]" for _, tf, _ in METRIC_COLUMNS) + "@{}"

    def row_cells(extractor: str, engine: str) -> str:
        cell = data.get((extractor, engine))
        out: list[str] = []
        for key, _, _ in METRIC_COLUMNS:
            v = None if cell is None else cell[key]
            s = _fmt_metric(v, key)
            if not s:
                out.append("")
            elif key in best and best[key] is not None and v == best[key]:
                out.append(r"\bfseries " + s)  # best cells left unpadded, like the header
            else:
                out.append(s.rjust(_METRIC_WIDTH[key]))
        return " & ".join(out)

    lines = [
        r"\begin{table}[!htb]",
        r"  \centering",
        r"  % Embedding | Detector splits the former X + Y label; \multirow groups the",
        r"  % detectors under each embedding. S columns align on the decimal point. Bold marks",
        r"  % the best value per outcome column; FPR is a controlled protocol target (~0.1\%),",
        r"  % not an outcome, so it is left unbolded. Missing runs render as empty cells.",
        rf"  \begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        _header_row([h for _, _, h in METRIC_COLUMNS]),
        r"    \midrule",
        *_group_rows(row_cells),
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{METRICS_CAPTION}}}",
        r"  \label{tab:sup25-metrics}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def render_recall_table(data: Results) -> str:
    """Per-technique recall heatmap per method (tab:sup25-recall).

    Same Embedding | Detector layout; each cell is a \\hc{}-shaded recall value, missing
    runs left empty. The all-zero Insider column is kept intentionally (see module docs).
    """

    def row_cells(extractor: str, engine: str) -> str:
        cell = data.get((extractor, engine))
        cells = (
            _fmt_heat(None if cell is None else cell[f"recall_{suffix}"]).ljust(_HEAT_WIDTH) for suffix, _ in TECHNIQUES
        )
        return " & ".join(cells).rstrip()

    lines = [
        r"\begin{table}[!htb]",
        r"  \centering",
        r"  % Heatmap: \hc{v} shades the cell by recall v (integer %, 0=white..100=green) and",
        r"  % prints v (see \hc / heat in head/colors.tex). Missing runs render as empty cells.",
        r"  % The Insider column is structurally 0 -- insider traffic never reaches the",
        r"  % detector's observation point -- kept to make that explicit.",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{@{}l l *{7}{c}@{}}",
        r"    \toprule",
        _header_row([h for _, h in TECHNIQUES]),
        r"    \midrule",
        *_group_rows(row_cells),
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{RECALL_CAPTION}}}",
        r"  \label{tab:sup25-recall}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# --- combined curve figures (folded in from the former aggregate_curves) -----


def _find_cell_run_id(method: str, extractor: str) -> str | None:
    """Latest full-run child run for a (method, extractor) cell, or None.

    The decision head is `decision_engine` on current runs and `pipeline` on legacy ones.
    MLflow filter strings are AND-only, so we search once per tag and keep the newest run.
    """
    common = f"tags.feature_extractor = '{extractor}' and tags.scenario = '{SCENARIO}' and tags.run_type = 'full-run'"
    best = None
    for engine_tag in ("decision_engine", "pipeline"):
        runs = mlflow.search_runs(
            filter_string=f"tags.{engine_tag} = '{method}' and {common}",
            order_by=["start_time DESC"],
            max_results=1,
            output_format="list",
        )
        if runs and (best is None or runs[0].info.start_time > best.info.start_time):
            best = runs[0]
    return best.info.run_id if best else None


def _fetch_curve(run_id: str, stem: str, method: str, extractor: str, cache_dir: Path) -> Curve | None:
    """Download a cell's ``_roc.csv``/``_auprc.csv`` from MLflow into ``cache_dir``."""
    paths: dict[str, Path] = {}
    for suffix in ("roc", "auprc"):
        artifact = f"curve_data/{stem}_{suffix}.csv"
        try:
            local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact, dst_path=str(cache_dir))
        except Exception as exc:  # noqa: BLE001 - missing/partial artifacts shouldn't abort the grid
            logger.warning(f"  missing artifact {artifact} on run {run_id}: {exc}")
            return None
        paths[suffix] = Path(local)
    return Curve(method=method, extractor=extractor, roc=pd.read_csv(paths["roc"]), pr=pd.read_csv(paths["auprc"]))


def _load_cell(method: str, extractor: str, family_name: str, cache_dir: Path) -> Curve | None:
    """Resolve one grid cell's run and load its curve (None if the run or its artifacts are absent)."""
    run_id = _find_cell_run_id(method, extractor)
    if run_id is None:
        logger.warning(f"  no run for {method}+{extractor} on {family_name}/{SCENARIO}; skipping")
        return None
    stem = f"{method}_{extractor}_{family_name}_{SCENARIO}"
    curve = _fetch_curve(run_id, stem, method, extractor, cache_dir)
    if curve is not None:
        logger.info(f"  loaded {method}+{extractor} from run {run_id}")
    return curve


def generate_curves(out_dir: Path, cache_dir: Path, max_workers: int, rasters: bool) -> None:
    """Build the combined ROC and PR pgfplots ``.tex`` figures (into ``out_dir``) from MLflow.

    Downloaded curve CSVs -- and, when ``rasters`` is set, the PNG/PDF figures -- land in
    ``cache_dir`` instead, keeping ``out_dir`` free of anything but the ``.tex`` the thesis inputs.
    """
    family = FAMILIES[DATASET]
    if SCENARIO not in _all_scenarios(family):
        raise RuntimeError(f"{SCENARIO!r} is not a scenario of {DATASET}.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    # The per-file tqdm bars MLflow prints interleave into noise once downloads run in
    # parallel; silence them and rely on the one-line-per-cell logging below instead.
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

    # Each cell is an independent, network-bound fetch, so run the grid through a thread pool.
    # ThreadPoolExecutor.map preserves input order, keeping the figure's colour/legend order
    # deterministic. Extractor order matches METHODS; methods are the decision heads.
    extractors = list(dict.fromkeys(e for e, _, _ in METHODS))
    grid = [(p, e) for p in ALL_METHODS for e in extractors]
    if unknown := (set(extractors) - set(EXTRACTORS)):
        raise RuntimeError(f"Unknown extractor(s): {sorted(unknown)}")
    workers = max(1, min(max_workers, len(grid)))
    logger.info(f"Fetching {len(grid)} curve cells with {workers} worker(s)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda pe: _load_cell(pe[0], pe[1], family.name, cache_dir), grid)
    curves = [c for c in results if c is not None]
    if not curves:
        raise RuntimeError("No curves could be loaded; nothing to plot.")

    base = f"{family.name}_{SCENARIO}_combined"
    # At the lowest threshold every sample is predicted positive (recall=1), so the precision
    # there equals the test-set prevalence -- the PR random-classifier baseline.
    pr0 = curves[0].pr
    prevalence = float(pr0.loc[pr0["recall"].idxmax(), "precision"])
    curves_tex = plot_combined_curves_tikz(
        curves, out_dir / f"{base}_curves.tex", prevalence=prevalence, extractor_labels=PAPER_EXTRACTOR_LABELS
    )
    logger.info(f"Wrote {len(curves)} curves as a combined pgfplots figure to {curves_tex}")
    if rasters:
        roc_path = plot_combined_roc(curves, cache_dir / f"{base}_roc.png")
        pr_path = plot_combined_pr(curves, cache_dir / f"{base}_auprc.png", prevalence=prevalence)
        logger.info(f"Wrote raster figures to {roc_path} and {pr_path} (+ .pdf)")


# --- entry point -------------------------------------------------------------


def _write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    logger.info(f"Wrote {path}")


def main(
    out_dir: Annotated[
        Path, typer.Option(help="Directory for the .tex artefacts (curve figures + tables).")
    ] = REPO_ROOT / "reports" / "superviz25-tex",
    cache_dir: Annotated[
        Path | None,
        typer.Option(help="Where to cache curve CSVs (and rasters); default: a throwaway temp dir."),
    ] = None,
    rasters: Annotated[bool, typer.Option(help="Also render the PNG/PDF figures (into --cache-dir).")] = False,
    max_workers: Annotated[int, typer.Option(help="Parallel MLflow artifact downloads for the curve fetches.")] = 8,
) -> None:
    """Load SuperViz25 results from MLflow and write the combined figures and both tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not setup_mlflow(DATASET):
        logger.error("MLflow tracking unavailable (URI unset or server unreachable).")
        raise typer.Exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix="superviz25-curves-"))
    generate_curves(out_dir, cache_dir, max_workers, rasters)
    data = load_results()
    _write(render_metrics_table(data), out_dir / "superviz25-metrics.tex")
    _write(render_recall_table(data), out_dir / "superviz25-recall.tex")


if __name__ == "__main__":
    typer.run(main)
