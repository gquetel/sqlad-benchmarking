"""Inference-latency benchmark for the IFIPSEC RQ2.3 figure: fan-out + per-cell task.

The single command to reproduce the figure, from a SLURM submit node:

    python -m tools.benchmark submit

It fans each ``(method, extractor)`` cell out as one array task (``run-cell``
below), then schedules a dependent CPU job that renders the pgfplots figure once
every cell is done. Each task load-or-trains the fitted model, times scoring with
the feature cache off, and logs ``infer_ms_per_query`` to the dedicated
``Inference-Latency-Superviz25`` experiment. Detection results are reused, never
recomputed.

Why the cache must be off: the feature cache
(:class:`~mlops_sqldetect.features.cache.CachingExtractor`) is shared across
decision engines, so per extractor only the first run computes features cold and
the rest read the matrix back -- the suite's ``score_seconds`` is therefore not a
usable latency. Device policy matches the paper: only embedding extractors
(SecureBERT, CodeT5+) time on GPU; every other cell -- including the autoencoders
-- times on CPU. Resources come from ``configs/slurm.yaml`` (same machinery as
:mod:`tools.slurm_submit`).
"""

from __future__ import annotations

import json
import logging
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
import typer
import yaml
from sklearn.model_selection import train_test_split

from mlops_sqldetect.data import split_normals
from mlops_sqldetect.datasets import FAMILIES
from mlops_sqldetect.evaluate_suite import VAL_FRACTION, Cell, _model_filename, enumerate_cells
from mlops_sqldetect.features.cache import CachingExtractor
from mlops_sqldetect.model import AEDetector, Detector, MethodName, build_method, load_method
from mlops_sqldetect.tracking import setup_mlflow
from tools.slurm_submit import (
    ENV_SETUP,
    REPO_ROOT,
    SUBMIT_DIR,
    _check_venv,
    _eligible_partitions,
    _gpu_section,
    _min_vram,
    _write_manifest,
)

logger = logging.getLogger(__name__)

# Only SuperViz25 (single scenario "dataset") backs the figure.
DATASET = "superviz25"
# Latency runs go to their own experiment so they never touch the eval runs.
BENCHMARK_EXPERIMENT = "Inference-Latency-Superviz25"
DEFAULT_EXTRACTORS = "gaur-expert,gaur-chatgpt,gaur-claude,gaur-llama,gaur-mistral,gaur-gpt-oss,gaur-ruleid,li,sbert"
# Figure default: repo-local (git-ignored). Point --figure-out at the paper to write there.
DEFAULT_FIGURE_OUT = REPO_ROOT / "reports" / "superviz25" / "overhead-inference.tex"
# Per-cell latency CSVs (one file per array task, no shared writer).
CELLS_DIR = REPO_ROOT / "reports" / "superviz25" / "cells-inference"

try:
    import torch

    _CUDA = torch.cuda.is_available()
except ImportError:  # torch is always installed, but keep the module importable without it.
    torch = None  # type: ignore[assignment]
    _CUDA = False


# --- per-cell benchmark ------------------------------------------------------


def _extractor_step(model: Detector) -> object:
    """Return the model's feature-extractor step (pipeline head, or AE's attribute)."""
    if isinstance(model, AEDetector):
        return model.extractor
    return model.pipeline.named_steps["features"]


def _disable_feature_cache(model: Detector) -> bool:
    """Turn the feature cache off so ``transform`` recomputes features every call.

    Returns True when the model carried a cache wrapper (i.e. was trained with
    caching on); a model trained cache-off already recomputes and needs no change.
    """
    ext = _extractor_step(model)
    if isinstance(ext, CachingExtractor):
        ext.cache_dir = None
        return True
    return False


def _is_cuda_model(model: Detector) -> bool:
    return _CUDA and isinstance(model, AEDetector) and model.device.type == "cuda"


def _time_score(model: Detector, df: pd.DataFrame, repeats: int, warmup: int) -> list[float]:
    """Time ``score_samples(df)`` ``repeats`` times after ``warmup`` untimed runs.

    Returns the per-run wall-clock seconds. CUDA work is synchronised around each
    timed run so GPU latency is not undercounted by asynchronous kernel launches.
    """
    sync = _is_cuda_model(model)
    for _ in range(warmup):
        model.score_samples(df)
    if sync:
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.score_samples(df)
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def _train_model(method: MethodName, extractor: str, df_fit: pd.DataFrame, model_path: Path, cache: bool) -> Detector:
    """Fit a model on ``df_fit`` and save it, mirroring the eval suite's training.

    ``df_fit`` is the seed-split 90% of the train-normal rows, so the fitted model
    matches the one behind the detection results already in MLflow.
    """
    logger.info(f"Training missing model {method}+{extractor} on {len(df_fit)} normals -> {model_path}")
    model = build_method(method, extractor, cache=cache)
    model.fit(df_fit)
    model.save(model_path)
    return model


def _load_test_df(data_root: Path | None) -> pd.DataFrame:
    """Load the SuperViz25 test split used for timing (carries attack_technique)."""
    family = FAMILIES[DATASET]
    scenario = family.suites["all"][0]
    return family.load_split(scenario, "test", root=data_root, columns=("full_query", "label", "attack_technique"))


def _load_fit_df(data_root: Path | None, seed: int) -> pd.DataFrame:
    """Load the eval suite's fit split (seed-split 90% of train normals) for training missing models."""
    family = FAMILIES[DATASET]
    scenario = family.suites["all"][0]
    df_train_normal = split_normals(family.load_split(scenario, "train", root=data_root))
    df_fit, _ = train_test_split(df_train_normal, test_size=VAL_FRACTION, random_state=seed)
    return df_fit


def _benchmark_cell(
    method: MethodName,
    extractor: str,
    df_test: pd.DataFrame,
    df_fit: pd.DataFrame | None,
    model_dir: Path,
    repeats: int,
    warmup: int,
    cache: bool,
    train_missing: bool,
    track: bool,
) -> dict | None:
    """Benchmark one cell; returns a result dict, or None when its model is missing and not trained."""
    scenario = FAMILIES[DATASET].suites["all"][0]
    model_path = model_dir / _model_filename(DATASET, method, extractor, scenario)
    if model_path.exists():
        model = load_method(method, model_path)
        trained = False
    elif train_missing and df_fit is not None:
        model = _train_model(method, extractor, df_fit, model_path, cache)
        trained = True
    else:
        logger.warning(f"Skipping {method}+{extractor}: no model at {model_path} (pass --train-missing to fit it)")
        return None

    cached = _disable_feature_cache(model)
    times = _time_score(model, df_test, repeats, warmup)

    n = len(df_test)
    median_s = statistics.median(times)
    ms_per_query = median_s / n * 1000.0
    device = "cuda" if _is_cuda_model(model) else "cpu"
    logger.info(
        f"{method}+{extractor}: {ms_per_query:.4f} ms/query "
        f"(median {median_s:.3f}s over {n} queries, {repeats} runs, {device}{', trained' if trained else ''})"
    )

    if track:
        # Fresh run in the benchmark experiment; eval runs untouched.
        with mlflow.start_run(run_name=f"{method}+{extractor}#{time.strftime('%Y%m%d-%H%M%S')}"):
            mlflow.set_tags(
                {
                    "decision_engine": method,
                    "feature_extractor": extractor,
                    "run_role": "benchmark",
                    "infer_device": device,
                    "infer_cache_disabled": str(cached),
                    "model_trained_here": str(trained),
                }
            )
            mlflow.log_params({"n_test": n, "repeats": repeats, "warmup": warmup})
            mlflow.log_metric("infer_ms_per_query", ms_per_query)
            mlflow.log_metric("infer_median_seconds", median_s)

    return {
        "method": method,
        "extractor": extractor,
        "n_test": n,
        "repeats": repeats,
        "device": device,
        "trained": trained,
        "median_seconds": round(median_s, 6),
        "ms_per_query": round(ms_per_query, 6),
    }


# --- SLURM resource routing --------------------------------------------------


def _needs_gpu(cell: Cell) -> bool:
    """Paper device policy: only embedding extractors time on GPU; AEs stay on CPU."""
    return cell.extractor in ("sbert", "codet5")


def _bucket(cell: Cell, cfg: dict, gpu_section: str) -> str:
    """Array bucket for a cell: cpu, the GPU section, or ``{section}-{req}gb`` for VRAM-pinned."""
    if not _needs_gpu(cell):
        return "cpu"
    section = _gpu_section(cell, cfg, gpu_section)
    req = _min_vram(cell, cfg)
    return section if req <= 0 else f"{section}-{req}gb"


def _resolve_resources(cfg: dict, cell: Cell, gpu_section: str) -> dict:
    """SBATCH resource block for a cell: the cpu block, or a VRAM-filtered GPU block."""
    if not _needs_gpu(cell):
        return dict(cfg["cpu"])
    if gpu_section not in cfg:
        raise typer.BadParameter(f"GPU section {gpu_section!r} not found in config; check configs/slurm.yaml.")
    gpu_cfg = cfg[gpu_section]
    res = {k: v for k, v in gpu_cfg.items() if k != "partitions"}
    res["partition"] = ",".join(_eligible_partitions(gpu_cfg, _min_vram(cell, cfg)))
    return res


# --- sbatch script generation ------------------------------------------------


def _header(job_name: str, cfg: dict, log_pattern: str, extra: list[str]) -> str:
    """Common #SBATCH directives (name, logs, account) plus the caller's resource lines."""
    directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_pattern}",
        f"#SBATCH --error={log_pattern}",
        *extra,
    ]
    if cfg.get("account"):
        directives.append(f"#SBATCH --account={cfg['account']}")
    return "\n".join(directives)


def _write_array_script(
    path: Path, *, job_name: str, res: dict, cfg: dict, manifest: Path, n: int, log_pattern: str, seed: int, track: bool
) -> None:
    """One array script: full #SBATCH header, then benchmark one cell per array index."""
    extra = [f"#SBATCH --partition={res['partition']}"]
    if res.get("gres"):
        extra.append(f"#SBATCH --gres={res['gres']}")
    extra += [
        f"#SBATCH --cpus-per-task={res['cpus_per_task']}",
        f"#SBATCH --mem={res['mem']}",
        f"#SBATCH --time={res['time']}",
        f"#SBATCH --array=0-{n - 1}",
    ]
    track_flag = "--track" if track else "--no-track"
    path.write_text(
        f"""#!/bin/bash
{_header(job_name, cfg, log_pattern, extra)}
set -euo pipefail
cd {REPO_ROOT}
{ENV_SETUP}
python -m tools.benchmark run-cell \\
  --manifest {manifest} \\
  --index "$SLURM_ARRAY_TASK_ID" \\
  --seed {seed} \\
  {track_flag}
"""
    )


def _write_figure_script(path: Path, *, cfg: dict, log_pattern: str, figure_out: Path) -> None:
    """A light CPU job that renders the figure once the arrays have logged their latencies."""
    cpu = cfg["cpu"]
    extra = [
        f"#SBATCH --partition={cpu['partition']}",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=8G",
        "#SBATCH --time=00:20:00",
    ]
    path.write_text(
        f"""#!/bin/bash
{_header("lat-figure-superviz25", cfg, log_pattern, extra)}
set -euo pipefail
cd {REPO_ROOT}
{ENV_SETUP}
python -m tools.generate_overhead_inference_figure --figure-out {figure_out}
"""
    )


def _sbatch(script: Path, dry_run: bool, dependency: str | None = None) -> str | None:
    """Submit (or, in dry-run, print) one job; return its job id. ``dependency`` is an afterok id list."""
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(str(script))
    if dry_run:
        logger.info("DRY-RUN: " + " ".join(cmd))
        return None
    sbatch = shutil.which("sbatch")
    if not sbatch:
        raise typer.BadParameter("sbatch not found on PATH; run on a SLURM submit node or use --dry-run.")
    cmd[0] = sbatch
    # Safe: shell=False, absolute sbatch path, script generated from the versioned config.
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise typer.BadParameter(f"sbatch rejected {script}:\n{result.stderr.strip() or result.stdout.strip()}")
    logger.info(result.stdout.strip())
    return result.stdout.strip().split()[-1]


# --- CLI ---------------------------------------------------------------------

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def submit(
    methods: Annotated[str, typer.Option(help="Comma-separated decision engines.")] = "ocsvm,ae",
    extractors: Annotated[str, typer.Option(help="Comma-separated feature extractors.")] = DEFAULT_EXTRACTORS,
    config: Annotated[Path, typer.Option(help="SLURM site config.")] = Path("configs/slurm.yaml"),
    gpu_section: Annotated[str, typer.Option(help="GPU resource block (e.g. 'gpu', 'gpu-long').")] = "gpu",
    seed: Annotated[int, typer.Option(help="Train/val split seed for the train-missing fallback.")] = 7,
    no_track: Annotated[bool, typer.Option(help="Disable MLflow logging for the submitted jobs.")] = False,
    figure: Annotated[bool, typer.Option(help="Schedule the dependent figure job after the arrays.")] = True,
    figure_out: Annotated[Path, typer.Option(help="Where the figure job writes the .tex.")] = DEFAULT_FIGURE_OUT,
    run_id: Annotated[str | None, typer.Option(help="Submission id (names the dir under submit_dir).")] = None,
    dry_run: Annotated[bool, typer.Option(help="Print manifests and sbatch commands without submitting.")] = False,
) -> None:
    """Enumerate the grid and submit one benchmark array per resource class, plus the figure job."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = yaml.safe_load(config.read_text())
    track = not no_track

    # A dry run only prints scripts; a real submit needs the shared .venv the compute nodes source.
    if not dry_run:
        _check_venv()

    cells = enumerate_cells(DATASET, "all", methods, extractors)
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    submit_dir = REPO_ROOT / SUBMIT_DIR / f"latency-{run_id}"
    (submit_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger.info(f"{len(cells)} cells -> {submit_dir}")

    buckets: dict[str, list[Cell]] = {}
    for cell in cells:
        buckets.setdefault(_bucket(cell, cfg, gpu_section), []).append(cell)

    job_ids: list[str] = []
    for bucket, group_cells in sorted(buckets.items()):
        res = _resolve_resources(cfg, group_cells[0], _gpu_section(group_cells[0], cfg, gpu_section))
        manifest = submit_dir / f"cells_{bucket}.jsonl"
        script = submit_dir / f"bench_cell_{bucket}.sbatch"
        _write_manifest(manifest, group_cells)
        _write_array_script(
            script,
            job_name=f"lat-{bucket}-superviz25",
            res=res,
            cfg=cfg,
            manifest=manifest,
            n=len(group_cells),
            log_pattern=str(submit_dir / "logs" / f"{bucket}-%A_%a.log"),
            seed=seed,
            track=track,
        )
        logger.info(f"{bucket}: {len(group_cells)} cells (partition {res['partition']})")
        job_id = _sbatch(script, dry_run)
        if job_id:
            job_ids.append(job_id)

    if figure:
        if not track:
            logger.warning("Figure job skipped: it needs MLflow (AUROC + latency); pass without --no-track.")
        else:
            script = submit_dir / "bench_figure.sbatch"
            _write_figure_script(
                script, cfg=cfg, log_pattern=str(submit_dir / "logs" / "figure-%j.log"), figure_out=figure_out
            )
            logger.info(f"figure: renders after arrays complete -> {figure_out}")
            _sbatch(script, dry_run, dependency=":".join(job_ids) or None)

    if dry_run:
        logger.info("Dry run: nothing submitted.")
    elif job_ids:
        logger.info(f"Submitted job arrays: {', '.join(job_ids)}")


@app.command("run-cell")
def run_cell(
    manifest: Annotated[Path, typer.Option(help="JSONL manifest of cells, one per line.")],
    index: Annotated[int, typer.Option(help="0-based line index into the manifest (the SLURM array task id).")],
    model_dir: Annotated[Path, typer.Option(help="Directory holding the fitted models.")] = REPO_ROOT / "models",
    data_root: Annotated[Path | None, typer.Option(help="SuperViz25 CSV directory (default: repo data dir).")] = None,
    repeats: Annotated[int, typer.Option(help="Timed runs (median taken).")] = 5,
    warmup: Annotated[int, typer.Option(help="Untimed warm-up runs before timing.")] = 1,
    seed: Annotated[int, typer.Option(help="Train/val split seed for the train-missing fallback.")] = 7,
    train_missing: Annotated[bool, typer.Option(help="Fit and save the model if absent, then time it.")] = True,
    cache: Annotated[bool, typer.Option(help="Cache features during the train-missing fit.")] = True,
    track: Annotated[bool, typer.Option(help="Log the latency run to MLflow when configured.")] = True,
) -> None:
    """Benchmark the manifest cell at ``index`` (one SLURM array task) and log/write its latency."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    if not 0 <= index < len(lines):
        raise typer.BadParameter(f"--index {index} out of range for {len(lines)} cells in {manifest}")
    cell = json.loads(lines[index])
    method, extractor = cell["method"], cell["extractor"]
    logger.info(f"Benchmarking cell {index}/{len(lines) - 1}: {method}+{extractor}")

    # setup_mlflow selects the eval experiment; redirect logging to the benchmark one.
    if track and setup_mlflow(DATASET):
        mlflow.set_experiment(BENCHMARK_EXPERIMENT)
    elif track:
        logger.warning("MLflow unavailable; writing CSV only.")
        track = False

    df_test = _load_test_df(data_root)
    df_fit = _load_fit_df(data_root, seed) if train_missing else None
    row = _benchmark_cell(method, extractor, df_test, df_fit, model_dir, repeats, warmup, cache, train_missing, track)
    if row is None:
        raise typer.Exit(code=1)

    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    out = CELLS_DIR / f"{method}_{extractor}.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    logger.info(f"Wrote {out}")


if __name__ == "__main__":
    app()
