"""Fan an evaluation grid out to SLURM as one job array per resource class.

Each cell ``(scenario, pipeline, extractor)`` becomes one array task that runs through
:func:`evaluate_suite` and writes its own per-cell CSV under ``reports/{dataset}/cells/``
(no shared writer, no merge step). Cells needing a GPU (``pipeline == "ae"`` or
``extractor == "sbert"``) go to a GPU array on the config's partition list (A100 first,
V100 fallback); cells whose extractor or ``pipeline:extractor`` is listed in ``force_a100``
get an A100-only array; the rest go to a CPU array. Resources and the environment setup
come from ``configs/slurm.yaml``.

Usage:
    python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ae --extractors li
    python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ocsvm,ae --dry-run
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml

from mlops_sqldetect.datasets import FAMILIES
from mlops_sqldetect.evaluate_suite import Cell, enumerate_cells, parent_run_spec
from mlops_sqldetect.tracking import ensure_parent_run, setup_mlflow

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Manifests, generated job scripts and .out logs (git-ignored), relative to REPO_ROOT.
SUBMIT_DIR = "reports/slurm"
# Run on the compute node (after cd-ing into the repo) before each cell. The uv .venv is
# built once on the login node (`uv sync --frozen`) and reused from the shared filesystem;
# array tasks only activate it (no concurrent `uv sync`, which would race).
ENV_SETUP = "source .venv/bin/activate"
# The activate script every array task sources on the compute node. Checked on the submit
# node before sbatch so a missing .venv fails once here, not silently in N array tasks.
VENV_ACTIVATE = REPO_ROOT / ".venv" / "bin" / "activate"


# GPU cells try A100 first then fall back to V100 via the config's partition list; cells
# whose extractor or pipeline:extractor combo is pinned in force_a100 (VRAM) get A100 only.
BUCKETS = ("gpu_a100", "gpu", "cpu")


def _needs_gpu(cell: Cell) -> bool:
    """A cell needs a GPU when it trains an autoencoder or uses the SecureBERT extractor."""
    return cell.pipeline == "ae" or cell.extractor == "sbert"


def _is_forced_a100(cell: Cell, cfg: dict) -> bool:
    """A cell is pinned to A100 when its extractor or pipeline:extractor combo is in force_a100."""
    forced = cfg.get("force_a100", [])
    return cell.extractor in forced or f"{cell.pipeline}:{cell.extractor}" in forced


def _bucket(cell: Cell, cfg: dict) -> str:
    """Resource bucket for a cell: cpu, gpu (A100/V100 fallback), or gpu_a100 (pinned)."""
    if not _needs_gpu(cell):
        return "cpu"
    return "gpu_a100" if _is_forced_a100(cell, cfg) else "gpu"


def _resolve_resources(cfg: dict, bucket: str) -> dict:
    """Resource block for a bucket; gpu_a100 reuses the gpu block but pins the partition to A100."""
    res = dict(cfg["cpu"] if bucket == "cpu" else cfg["gpu"])
    if bucket == "gpu_a100":
        res["partition"] = "A100"
    return res


def _check_venv(activate: Path = VENV_ACTIVATE) -> None:
    """Fail fast if the shared .venv is missing; every array task sources it on the compute node."""
    if not activate.exists():
        raise typer.BadParameter(
            f"{activate} not found; build it on the login node with `uv sync --frozen` before submitting."
        )


def _write_manifest(path: Path, cells: list[Cell]) -> None:
    path.write_text("".join(json.dumps(cell._asdict()) + "\n" for cell in cells))


def _write_job_script(
    path: Path,
    *,
    bucket: str,
    cfg: dict,
    dataset: str,
    manifest: Path,
    n: int,
    log_pattern: str,
    target_fpr: float,
    seed: int,
    track: bool,
    limit: int | None,
) -> None:
    """Generate a self-contained array script: full #SBATCH header, then dispatch one cell per index."""
    res = _resolve_resources(cfg, bucket)
    directives = [
        f"#SBATCH --job-name=sqldetect-{bucket}",
        f"#SBATCH --output={log_pattern}",
        f"#SBATCH --error={log_pattern}",
        f"#SBATCH --partition={res['partition']}",
    ]
    if res.get("gres"):
        directives.append(f"#SBATCH --gres={res['gres']}")
    directives += [
        f"#SBATCH --cpus-per-task={res['cpus_per_task']}",
        f"#SBATCH --mem={res['mem']}",
        f"#SBATCH --time={res['time']}",
        f"#SBATCH --array=0-{n - 1}",
    ]
    if cfg.get("account"):
        directives.append(f"#SBATCH --account={cfg['account']}")
    register_flag = "--register" if track and cfg.get("register") else "--no-register"
    track_flag = "--track" if track else "--no-track"
    limit_flag = f" \\\n  --limit {limit}" if limit is not None else ""
    header = "\n".join(directives)
    path.write_text(
        f"""#!/bin/bash
{header}
set -euo pipefail
cd {REPO_ROOT}
{ENV_SETUP}
python -m tools.slurm_run_cell \\
  --manifest {manifest} \\
  --index "$SLURM_ARRAY_TASK_ID" \\
  --dataset {dataset} \\
  --target-fpr {target_fpr} \\
  --seed {seed} \\
  {register_flag} {track_flag}{limit_flag}
"""
    )


def _submit_array(script: Path, dry_run: bool) -> str | None:
    """Submit (or, in dry-run, just print) one job array and return its job id.

    Resources live in the script's #SBATCH header, so submission is just ``sbatch <script>``.
    """
    cmd = ["sbatch", str(script)]
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
        # Propagate sbatch errors
        raise typer.BadParameter(f"sbatch rejected {script}:\n{result.stderr.strip() or result.stdout.strip()}")
    logger.info(result.stdout.strip())
    return result.stdout.strip().split()[-1]


def submit(
    dataset: Annotated[str, typer.Option(help="Dataset family: superviz26 or superviz25.")] = "superviz26",
    suite: Annotated[str, typer.Option(help="Suite name (e.g. in_domain, lodo, all).")] = "all",
    pipelines: Annotated[str, typer.Option(help="Comma-separated decision-head names (ocsvm, lof, ae).")] = "ocsvm,ae",
    extractors: Annotated[str, typer.Option(help="Comma-separated feature-extractor names (li, loginov, cv, sbert).")] = "li",
    config: Annotated[Path, typer.Option(help="SLURM site config.")] = Path("configs/slurm.yaml"),
    target_fpr: Annotated[float, typer.Option(help="Target false-positive rate for the calibrated threshold.")] = 0.001,
    seed: Annotated[int, typer.Option(help="Random state for the train/validation calibration split.")] = 7,
    register: Annotated[bool, typer.Option(help="Register each fitted model in the MLflow Model Registry.")] = False,
    no_track: Annotated[bool, typer.Option(help="Disable MLflow tracking for the submitted jobs.")] = False,
    limit: Annotated[int | None, typer.Option(help="Label-stratified subset size per cell for smoke runs.")] = None,
    run_id: Annotated[
        str | None, typer.Option(help="Submission id; names the dir under submit_dir (default: timestamp).")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the manifests and sbatch commands without submitting.")] = False,
) -> None:
    """Enumerate the grid, pre-create MLflow parents, and submit one job array per resource class."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = yaml.safe_load(config.read_text())
    cfg["register"] = register
    track = not no_track

    # Fail fast on the submit node: a dry run only prints scripts, but a real submit needs
    # the shared .venv the compute nodes will source.
    if not dry_run:
        _check_venv()

    cells = enumerate_cells(dataset, suite, pipelines, extractors)
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    submit_dir = REPO_ROOT / SUBMIT_DIR / run_id
    (submit_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger.info(f"{len(cells)} cells -> {submit_dir}")

    # Pre-create the MLflow parents once, serially, so concurrent array tasks reuse them
    # instead of racing on find-or-create and spawning duplicate parents.
    if track and setup_mlflow(dataset):
        for pipeline, extractor in {(c.pipeline, c.extractor) for c in cells}:
            name, tags = parent_run_spec(FAMILIES[dataset], pipeline, extractor)
            ensure_parent_run(tags, name)

    buckets: dict[str, list[Cell]] = {}
    for cell in cells:
        buckets.setdefault(_bucket(cell, cfg), []).append(cell)
    job_ids: list[str] = []
    for bucket in BUCKETS:
        group_cells = buckets.get(bucket, [])
        if not group_cells:
            continue
        manifest = submit_dir / f"cells_{bucket}.jsonl"
        script = submit_dir / f"eval_cell_{bucket}.sbatch"
        _write_manifest(manifest, group_cells)
        _write_job_script(
            script,
            bucket=bucket,
            cfg=cfg,
            dataset=dataset,
            manifest=manifest,
            n=len(group_cells),
            log_pattern=str(submit_dir / "logs" / f"{bucket}-%A_%a.log"),
            target_fpr=target_fpr,
            seed=seed,
            track=track,
            limit=limit,
        )
        logger.info(f"{bucket}: {len(group_cells)} cells (partition {_resolve_resources(cfg, bucket)['partition']})")
        job_id = _submit_array(script, dry_run)
        if job_id:
            job_ids.append(job_id)

    if dry_run:
        logger.info("Dry run: nothing submitted.")
    elif job_ids:
        logger.info(f"Submitted job arrays: {', '.join(job_ids)}")


if __name__ == "__main__":
    typer.run(submit)
