"""Measure prediction time for the IFIPSEC RQ2.3 figure.

Run GAUR locally with the instrumented MySQL server; submit other extractors to SLURM:

    python -m tools.benchmark cluster
    python -m tools.benchmark lames

Both commands load or train models and time predictions with feature caching disabled.
They log ``infer_ms_per_query`` to the ``Inference-Latency-Superviz25`` MLflow experiment.
Embedding extractors use GPUs; other extractors use CPUs.
Render the figure with :mod:`tools.generate_observation_tex`.
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

from sqlad_benchmarking.data import split_normals
from sqlad_benchmarking.datasets import FAMILIES
from sqlad_benchmarking.evaluate_suite import VAL_FRACTION, Cell, _model_filename, enumerate_cells
from sqlad_benchmarking.features import GPU_EXTRACTORS
from sqlad_benchmarking.features.cache import CachingExtractor
from sqlad_benchmarking.model import AEDetector, Detector, MethodName, build_method, load_method
from sqlad_benchmarking.tracking import setup_mlflow
from tools.slurm_submit import (
    REPO_ROOT,
    SUBMIT_DIR,
    _allowed_partitions,
    _check_cuda_builds,
    _eligible_partitions,
    _gpu_section,
    _min_vram,
    _write_manifest,
    env_setup,
)

logger = logging.getLogger(__name__)

DATASET = "superviz25"
BENCHMARK_EXPERIMENT = "Inference-Latency-Superviz25"
GAUR_EXTRACTORS = "gaur-expert,gaur-chatgpt,gaur-claude,gaur-llama,gaur-mistral,gaur-gpt-oss,gaur-ruleid"
CLUSTER_EXTRACTORS = "cv,li,loginov,sbert,codet5"
DEFAULT_METHODS = "ocsvm,ae,lof"
# One CSV per task avoids concurrent writes to the same file.
CELLS_DIR = REPO_ROOT / "reports" / "superviz25" / "cells-inference"
# Sample rows to limit tracing time; 0 uses the full test set.
TIMING_SAMPLE = 5000

try:
    import torch

    _CUDA = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    _CUDA = False


# --- per-cell benchmark ------------------------------------------------------


def _extractor_step(model: Detector) -> object:
    """Return the model's feature extractor."""
    if isinstance(model, AEDetector):
        return model.extractor
    return model.pipeline.named_steps["features"]


def _disable_feature_cache(model: Detector) -> bool:
    """Disable feature caching and return whether the extractor supports it."""
    ext = _extractor_step(model)
    if isinstance(ext, CachingExtractor):
        ext.cache_dir = None
        return True
    return False


def _is_cuda_model(model: Detector) -> bool:
    """Return whether the model is an autoencoder running on a GPU."""
    return _CUDA and isinstance(model, AEDetector) and model.device.type == "cuda"


def _force_cpu(model: Detector) -> None:
    """Move an autoencoder to the CPU for timing."""
    if isinstance(model, AEDetector) and model.net is not None:
        model.device = torch.device("cpu")
        model.net.to(model.device)


def _enable_benchmark_tracking() -> bool:
    """Select the benchmark MLflow experiment; return False if tracking is unavailable."""
    if not setup_mlflow(DATASET):
        return False
    mlflow.set_experiment(BENCHMARK_EXPERIMENT)
    return True


def _time_score(model: Detector, df: pd.DataFrame, repeats: int, warmup: int) -> list[float]:
    """Return prediction times in seconds after untimed warm-up runs.

    Wait for GPU autoencoders to finish before recording each time.
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
    """Train a model on df_fit and save it."""
    logger.info(f"Training missing model {method}+{extractor} on {len(df_fit)} normals -> {model_path}")
    model = build_method(method, extractor, cache=cache)
    model.fit(df_fit)
    model.save(model_path)
    return model


def _load_test_df(data_root: Path | None, sample: int = TIMING_SAMPLE, seed: int = 7) -> pd.DataFrame:
    """Load a reproducible SuperViz25 test sample; sample <= 0 loads all rows."""
    family = FAMILIES[DATASET]
    scenario = family.suites["all"][0]
    df = family.load_split(scenario, "test", root=data_root, columns=("full_query", "label", "attack_technique"))
    if 0 < sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    return df


def _load_fit_df(data_root: Path | None, seed: int) -> pd.DataFrame:
    """Load the same 90% of benign training rows used by the evaluation suite."""
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
    force_cpu: bool = False,
) -> dict | None:
    """Measure one model's prediction time; return None if no model can be loaded or trained."""
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

    if force_cpu:
        _force_cpu(model)
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
    """Return whether the extractor needs a GPU for timing."""
    return cell.extractor in GPU_EXTRACTORS


def _bucket(cell: Cell, cfg: dict, gpu_section: str) -> str:
    """Group a cell by its CPU, GPU, memory, and partition needs."""
    if not _needs_gpu(cell):
        return "cpu"
    section = _gpu_section(cell, cfg, gpu_section)
    req = _min_vram(cell, cfg)
    bucket = section if req <= 0 else f"{section}-{req}gb"
    allowed = _allowed_partitions(cell, cfg)
    if allowed is not None:
        bucket = f"{bucket}-{'-'.join(allowed)}"
    return bucket


def _resolve_resources(cfg: dict, cell: Cell, gpu_section: str) -> dict:
    """Return a cell's SLURM settings, selecting GPUs with enough memory."""
    if not _needs_gpu(cell):
        return dict(cfg["cpu"])
    if gpu_section not in cfg:
        raise typer.BadParameter(f"GPU section {gpu_section!r} not found in config; check configs/slurm.yaml.")
    gpu_cfg = cfg[gpu_section]
    res = {k: v for k, v in gpu_cfg.items() if k != "partitions"}
    res["partition"] = ",".join(_eligible_partitions(gpu_cfg, _min_vram(cell, cfg), _allowed_partitions(cell, cfg)))
    return res


# --- sbatch script generation ------------------------------------------------


def _header(job_name: str, cfg: dict, log_pattern: str, extra: list[str]) -> str:
    """Build SLURM directives for the job name, logs, account, and resources."""
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
    path: Path,
    *,
    job_name: str,
    res: dict,
    cfg: dict,
    manifest: Path,
    n: int,
    log_pattern: str,
    seed: int,
    sample: int,
    track: bool,
) -> None:
    """Write a SLURM script that benchmarks one cell per array task."""
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
{env_setup(cfg)}
python -m tools.benchmark run-cell \\
  --manifest {manifest} \\
  --index "$SLURM_ARRAY_TASK_ID" \\
  --seed {seed} \\
  --sample {sample} \\
  {track_flag}
"""
    )


def _sbatch(script: Path, dry_run: bool) -> str | None:
    """Submit an array and return its job ID, or print the command for a dry run."""
    cmd = ["sbatch", str(script)]
    if dry_run:
        logger.info("DRY-RUN: " + " ".join(cmd))
        return None
    sbatch = shutil.which("sbatch")
    if not sbatch:
        raise typer.BadParameter("sbatch not found on PATH; run on a SLURM submit node or use --dry-run.")
    cmd[0] = sbatch
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise typer.BadParameter(f"sbatch rejected {script}:\n{result.stderr.strip() or result.stdout.strip()}")
    logger.info(result.stdout.strip())
    return result.stdout.strip().split()[-1]


# --- CLI ---------------------------------------------------------------------

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def cluster(
    methods: Annotated[str, typer.Option(help="Comma-separated decision engines.")] = DEFAULT_METHODS,
    extractors: Annotated[str, typer.Option(help="Comma-separated feature extractors (no GAUR).")] = CLUSTER_EXTRACTORS,
    config: Annotated[Path, typer.Option(help="SLURM site config.")] = Path("configs/slurm.yaml"),
    gpu_section: Annotated[str, typer.Option(help="GPU resource block (e.g. 'gpu', 'gpu-long').")] = "gpu",
    sample: Annotated[
        int, typer.Option(help="Random test-set rows to time per cell (0 = full split).")
    ] = TIMING_SAMPLE,
    seed: Annotated[int, typer.Option(help="Seed for the timing sample and the train-missing split.")] = 7,
    no_track: Annotated[bool, typer.Option(help="Disable MLflow logging for the submitted jobs.")] = False,
    run_id: Annotated[str | None, typer.Option(help="Submission id (names the dir under submit_dir).")] = None,
    dry_run: Annotated[bool, typer.Option(help="Print manifests and sbatch commands without submitting.")] = False,
) -> None:
    """Submit non-GAUR benchmarks from the submit node, grouped by resource needs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if any(e.strip().startswith("gaur") for e in extractors.split(",")):
        raise typer.BadParameter("GAUR extractors need the MySQL server; run them with the 'lames' command.")
    cfg = yaml.safe_load(config.read_text())
    track = not no_track

    cells = enumerate_cells(DATASET, "all", methods, extractors)
    _check_cuda_builds(cfg, cells, gpu_section)
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
            sample=sample,
            track=track,
        )
        logger.info(f"{bucket}: {len(group_cells)} cells (partition {res['partition']})")
        job_id = _sbatch(script, dry_run)
        if job_id:
            job_ids.append(job_id)

    if dry_run:
        logger.info("Dry run: nothing submitted.")
    elif job_ids:
        logger.info(f"Submitted job arrays: {', '.join(job_ids)}")


@app.command()
def lames(
    methods: Annotated[str, typer.Option(help="Comma-separated decision engines.")] = DEFAULT_METHODS,
    extractors: Annotated[str, typer.Option(help="Comma-separated GAUR extractors.")] = GAUR_EXTRACTORS,
    model_dir: Annotated[Path, typer.Option(help="Directory holding the fitted models.")] = REPO_ROOT / "models",
    data_root: Annotated[Path | None, typer.Option(help="SuperViz25 CSV directory (default: repo data dir).")] = None,
    repeats: Annotated[int, typer.Option(help="Timed runs per cell (median taken).")] = 5,
    warmup: Annotated[int, typer.Option(help="Untimed warm-up runs before timing.")] = 1,
    sample: Annotated[
        int, typer.Option(help="Random test-set rows to time per cell (0 = full split).")
    ] = TIMING_SAMPLE,
    seed: Annotated[int, typer.Option(help="Seed for the timing sample and the train-missing split.")] = 7,
    train_missing: Annotated[bool, typer.Option(help="Fit and save any missing model, then time it.")] = True,
    cache: Annotated[bool, typer.Option(help="Cache features during the train-missing fit.")] = True,
    track: Annotated[bool, typer.Option(help="Log a latency run per cell to MLflow.")] = True,
) -> None:
    """Benchmark GAUR on the local CPU; requires the instrumented MySQL server."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not all(e.strip().startswith("gaur") for e in extractors.split(",") if e.strip()):
        raise typer.BadParameter("The 'lames' command is for GAUR extractors only; use 'cluster' for the rest.")

    mlflow_ok = _enable_benchmark_tracking() if track else False
    if track and not mlflow_ok:
        logger.warning("MLflow unavailable; writing CSV only.")
        track = False

    cells = enumerate_cells(DATASET, "all", methods, extractors)
    df_test = _load_test_df(data_root, sample, seed)
    df_fit = _load_fit_df(data_root, seed) if train_missing else None

    rows = []
    for cell in cells:
        # Time GAUR on the CPU, including autoencoders.
        row = _benchmark_cell(
            cell.method, cell.extractor, df_test, df_fit, model_dir, repeats, warmup, cache, train_missing, track, True
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise typer.Exit(code=1)

    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        pd.DataFrame([row]).to_csv(CELLS_DIR / f"{row['method']}_{row['extractor']}.csv", index=False)
    logger.info(f"Wrote {len(rows)} per-cell CSVs to {CELLS_DIR}")


@app.command("run-cell")
def run_cell(
    manifest: Annotated[Path, typer.Option(help="JSONL manifest of cells, one per line.")],
    index: Annotated[int, typer.Option(help="0-based line index into the manifest (the SLURM array task id).")],
    model_dir: Annotated[Path, typer.Option(help="Directory holding the fitted models.")] = REPO_ROOT / "models",
    data_root: Annotated[Path | None, typer.Option(help="SuperViz25 CSV directory (default: repo data dir).")] = None,
    repeats: Annotated[int, typer.Option(help="Timed runs (median taken).")] = 5,
    warmup: Annotated[int, typer.Option(help="Untimed warm-up runs before timing.")] = 1,
    sample: Annotated[int, typer.Option(help="Random test-set rows to time (0 = full split).")] = TIMING_SAMPLE,
    seed: Annotated[int, typer.Option(help="Seed for the timing sample and the train-missing split.")] = 7,
    train_missing: Annotated[bool, typer.Option(help="Fit and save the model if absent, then time it.")] = True,
    cache: Annotated[bool, typer.Option(help="Cache features during the train-missing fit.")] = True,
    track: Annotated[bool, typer.Option(help="Log the latency run to MLflow when configured.")] = True,
) -> None:
    """Benchmark the cell at index in the task list and save its prediction time."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    if not 0 <= index < len(lines):
        raise typer.BadParameter(f"--index {index} out of range for {len(lines)} cells in {manifest}")
    cell = json.loads(lines[index])
    method, extractor = cell["method"], cell["extractor"]
    logger.info(f"Benchmarking cell {index}/{len(lines) - 1}: {method}+{extractor}")

    if track and not _enable_benchmark_tracking():
        logger.warning("MLflow unavailable; writing CSV only.")
        track = False

    df_test = _load_test_df(data_root, sample, seed)
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
