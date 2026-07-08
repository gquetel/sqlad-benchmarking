r"""Generate every SuperViz25 LaTeX artefact from MLflow in a single call.

One invocation writes ``--out-dir`` (default ``reports/superviz25-tex/``) with only the
LaTeX ``.tex`` artefacts the thesis ``\input``s, all sourced from the ``Superviz25-SQL``
experiment:

  * the combined ROC and precision-recall figures as native pgfplots ``.tex`` -- rebuilt
    from every cell's ``curve_data`` artifacts. This folds in (and replaces) the former
    ``aggregate_curves`` command;
  * ``superviz25-metrics.tex`` (\label{tab:sup25-metrics}): precision, recall, F1, FPR,
    AUPRC and AUROC per pipeline; and
  * ``superviz25-recall.tex`` (\label{tab:sup25-recall}): recall per attack technique.

The downloaded curve CSVs (and, with ``--rasters``, the PNG/PDF figures) go to
``--cache-dir`` (a throwaway temp dir by default), so ``--out-dir`` stays ``.tex``-only.

MLflow is the single source of truth: the latest finished full-run per
``(feature_extractor, decision_engine)`` cell wins. Missing cells render as ``--`` in the
tables and are dropped from the figures, so re-running after new runs finish (e.g. the
Loginov / CodeT5+ cells) refreshes every artefact.

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
from mlops_sqldetect.evaluate_suite import ALL_PIPELINES, _all_scenarios
from mlops_sqldetect.features import EXTRACTORS
from mlops_sqldetect.tracking import setup_mlflow
from mlops_sqldetect.visualize import (
    Curve,
    plot_combined_pr,
    plot_combined_pr_tikz,
    plot_combined_roc,
    plot_combined_roc_tikz,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The MLflow experiment holding the SuperViz25 grid (see tracking._EXPERIMENT_NAMES);
# it carries a single scenario, tagged "dataset".
DATASET = "superviz25"
EXPERIMENT = "Superviz25-SQL"
SCENARIO = "dataset"

# Pipelines in display order: (feature_extractor tag, decision_engine tag, paper label).
# Labels are thesis-facing and intentionally differ from the repo's canonical extractor
# names ("Secure-BERT" vs "SecureBERT", "AE" vs "Autoencoder").
PIPELINES: list[tuple[str, str, str]] = [
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

# Per-technique recall columns: (MLflow metric suffix, column header). The metric key is
# ``recall_<suffix>`` (see evaluate_suite._mlflow_key). Insider attacks are forced to false
# negatives during evaluation, so their recall is 0 wherever a run exists.
TECHNIQUES: list[tuple[str, str]] = [
    ("union", "Union"),
    ("boolean", "Boolean"),
    ("stacked", "Stacked"),
    ("error", "Error"),
    ("time", "Time"),
    ("inline", "Inline"),
    ("insider", "Insider"),
]

# Metric columns of the performance table: (internal key, column header).
METRIC_COLUMNS: list[tuple[str, str]] = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("fpr", "FPR"),
    ("auprc", "AUPRC"),
    ("auroc", "AUROC"),
]

METRICS_CAPTION = "Performance metrics for studied novelty detection pipelines."
RECALL_CAPTION = "Recall score per technique for studied detection pipelines."

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
    for extractor, engine, label in PIPELINES:
        if (extractor, engine) not in data:
            logger.warning(f"Missing run: {label}")


# --- table rendering ---------------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    return "--" if v is None else rf"{v * 100:.2f}\%"


def _fmt_score(v: float | None) -> str:
    return "--" if v is None else f"{v:.4f}"


# Rows share the PIPELINES order, so a group ends wherever the next pipeline switches
# feature extractor; those rows get 20% extra vertical space to separate the groups.
GROUP_END_ROWS = frozenset(i for i in range(len(PIPELINES) - 1) if PIPELINES[i][0] != PIPELINES[i + 1][0])
GROUP_VSPACE = r"0.2\normalbaselineskip"


def _render_table(header: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    """Render a ``Model | <cols>`` table, space-padding cells so the source stays readable."""
    ncols = len(header) - 1
    label_w = max(len(r[0]) for r in rows)
    cell_w = max(len(c) for r in rows for c in r[1:])

    def body_line(cells: list[str], i: int) -> str:
        first = cells[0].ljust(label_w)
        rest = " & ".join(c.rjust(cell_w) for c in cells[1:])
        end = rf" \\[{GROUP_VSPACE}]" if i in GROUP_END_ROWS else r" \\"
        return f"    {first} & {rest}{end}"

    # Header is left unpadded (like the hand-written tables); only body cells are aligned.
    header_line = "    " + " & ".join(rf"\bfseries {h}" for h in header) + r" \\"
    out = [
        r"\begin{table}[!htb]",
        r"  \centering",
        rf"  \begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}} c|{'c' * ncols} }}",
        r"    \hline",
        header_line,
        r"    \hline",
    ]
    out += [body_line(r, i) for i, r in enumerate(rows)]
    out += [
        r"    \hline",
        r"  \end{tabular*}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(out)


def render_metrics_table(data: Results) -> str:
    """Precision/Recall/F1/FPR/AUPRC/AUROC per pipeline (tab:sup25-metrics)."""
    header = ["Model", *(h for _, h in METRIC_COLUMNS)]
    rows: list[list[str]] = []
    for extractor, engine, label in PIPELINES:
        cell = data.get((extractor, engine))
        row = [label]
        for key, _ in METRIC_COLUMNS:
            v = None if cell is None else cell[key]
            row.append(_fmt_score(v) if key in ("auprc", "auroc") else _fmt_pct(v))
        rows.append(row)
    return _render_table(header, rows, METRICS_CAPTION, "tab:sup25-metrics")


def render_recall_table(data: Results) -> str:
    """Per-technique recall per pipeline (tab:sup25-recall)."""
    header = ["Model", *(h for _, h in TECHNIQUES)]
    rows: list[list[str]] = []
    for extractor, engine, label in PIPELINES:
        cell = data.get((extractor, engine))
        row = [label]
        for suffix, _ in TECHNIQUES:
            row.append(_fmt_pct(None if cell is None else cell[f"recall_{suffix}"]))
        rows.append(row)
    return _render_table(header, rows, RECALL_CAPTION, "tab:sup25-recall")


# --- combined curve figures (folded in from the former aggregate_curves) -----


def _find_cell_run_id(pipeline: str, extractor: str) -> str | None:
    """Latest full-run child run for a (pipeline, extractor) cell, or None.

    The decision head is `decision_engine` on current runs and `pipeline` on legacy ones.
    MLflow filter strings are AND-only, so we search once per tag and keep the newest run.
    """
    common = f"tags.feature_extractor = '{extractor}' and tags.scenario = '{SCENARIO}' and tags.run_type = 'full-run'"
    best = None
    for engine_tag in ("decision_engine", "pipeline"):
        runs = mlflow.search_runs(
            filter_string=f"tags.{engine_tag} = '{pipeline}' and {common}",
            order_by=["start_time DESC"],
            max_results=1,
            output_format="list",
        )
        if runs and (best is None or runs[0].info.start_time > best.info.start_time):
            best = runs[0]
    return best.info.run_id if best else None


def _fetch_curve(run_id: str, stem: str, pipeline: str, extractor: str, cache_dir: Path) -> Curve | None:
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
    return Curve(pipeline=pipeline, extractor=extractor, roc=pd.read_csv(paths["roc"]), pr=pd.read_csv(paths["auprc"]))


def _load_cell(pipeline: str, extractor: str, family_name: str, cache_dir: Path) -> Curve | None:
    """Resolve one grid cell's run and load its curve (None if the run or its artifacts are absent)."""
    run_id = _find_cell_run_id(pipeline, extractor)
    if run_id is None:
        logger.warning(f"  no run for {pipeline}+{extractor} on {family_name}/{SCENARIO}; skipping")
        return None
    stem = f"{pipeline}_{extractor}_{family_name}_{SCENARIO}"
    curve = _fetch_curve(run_id, stem, pipeline, extractor, cache_dir)
    if curve is not None:
        logger.info(f"  loaded {pipeline}+{extractor} from run {run_id}")
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
    # deterministic. Extractor order matches PIPELINES; pipelines are the decision heads.
    extractors = list(dict.fromkeys(e for e, _, _ in PIPELINES))
    grid = [(p, e) for p in ALL_PIPELINES for e in extractors]
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
    roc_tex = plot_combined_roc_tikz(curves, out_dir / f"{base}_roc.tex")
    pr_tex = plot_combined_pr_tikz(curves, out_dir / f"{base}_auprc.tex", prevalence=prevalence)
    logger.info(f"Wrote {len(curves)} curves as pgfplots figures to {roc_tex} and {pr_tex}")
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
