# Running the suite on SLURM

`evaluate_suite` runs the grid (pipelines × extractors × scenarios) serially in one
process. On a cluster you usually want each **cell** — one `(scenario, pipeline,
extractor)` — to run as its own job so they execute in parallel (e.g. the 8 Superviz26
scenarios become 8 jobs).

## How it works

`tools.slurm_submit` enumerates the grid and submits **one job array per resource
class**. Cells are bucketed by their needs:

- **CPU** — cells that need no GPU (e.g. `ocsvm/li`) → the `cpu` partition.
- **GPU (fallback)** — cells that train an autoencoder (`pipeline == "ae"`) or use
  SecureBERT (`extractor == "sbert"`) → the `gpu` partition, which may be a list such as
  `A100,V100` so SLURM places the job wherever it can start first.
- **GPU (A100-only)** — cells whose extractor or `pipeline:extractor` is listed in
  `force_a100` (pinned for VRAM) → the `A100` partition only.

Each array task runs one cell through `evaluate_suite` and writes its **own** per-cell CSV
to `reports/{dataset}/cells/{pipeline}_{extractor}_{scenario}.csv` — so the parallel jobs
never share a writer and a rerun simply overwrites its own file.

Site settings (partitions, resources, `force_a100`) live in
[`configs/slurm.yaml`](https://github.com/gquetel/mlops-sqldetect/blob/main/configs/slurm.yaml);
the full `#SBATCH` header is baked into each generated script. Each job `cd`s into the repo
and activates the uv `.venv` with `source .venv/bin/activate` on the compute node. That
`.venv` is built once on the login node (`uv sync --frozen`) and reused from the shared
filesystem; array tasks only activate it and never run `uv sync` concurrently (which would
race on the shared `.venv`).

!!! note "A100-first preference"
    `--partition=A100,V100` lets SLURM pick whichever frees up first. A *strict* A100-first
    preference requires the admin to give A100 a higher `PriorityTier`
    (`scontrol show partition A100 | grep PriorityTier`); list order alone is not a guarantee.

## Submitting

```bash
# Preview the plan (manifests + sbatch commands) without submitting:
python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ae --extractors li --dry-run

# Submit for real:
python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ocsvm,ae --extractors li

# Smoke run: cap each cell to a label-stratified subset (passed through to evaluate_suite):
python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ae --extractors sbert --limit 50000
```

Equivalent invoke alias:

```bash
invoke slurm-suite --dataset superviz26 --suite all --pipelines ocsvm,ae --dry-run
```

Manifests, generated job scripts, and `.out` logs are written under
`reports/slurm/<run-id>/` (git-ignored).

## Results

MLflow is the canonical store: each cell logs an independent run, nested under a parent
that `slurm_submit` pre-creates once (so concurrent jobs don't spawn duplicate parents).
Each cell also writes its row to `reports/{dataset}/cells/*.csv` on disk; read that
directory directly if you need a flat table.
