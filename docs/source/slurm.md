# Running the suite on SLURM

`evaluate_suite` runs the grid (pipelines × extractors × scenarios) serially in one
process. On a cluster you usually want each **cell** — one `(scenario, pipeline,
extractor)` — to run as its own job so they execute in parallel (e.g. the 8 Superviz26
scenarios become 8 jobs).

## How it works

`tools.slurm_submit` enumerates the grid and submits **one job array per resource
class**. Cells are bucketed by their needs:

- **CPU** — cells that need no GPU (e.g. `ocsvm/li`) → the `cpu` partition.
- **GPU** — cells that train an autoencoder (`pipeline == "ae"`) or use an embedding
  extractor (`extractor in {"sbert", "codet5"}`) → a GPU array whose partition list is the
  GPU partitions with **enough VRAM** for that cell. Each cell's minimum comes from
  `min_vram_gb` (keyed by `extractor` or `pipeline:extractor`, else `default`); partitions
  are tried largest-VRAM first so cells prefer the fastest GPU. A hungry model like CodeT5+
  (`codet5: 24`) is thus never scheduled on the 16 GB V100.

Each array task runs one cell through `evaluate_suite` and writes its **own** per-cell CSV
to `reports/{dataset}/cells/{pipeline}_{extractor}_{scenario}.csv` — so the parallel jobs
never share a writer and a rerun simply overwrites its own file.

Site settings (partitions, resources, `min_vram_gb`) live in
[`configs/slurm.yaml`](https://github.com/gquetel/mlops-sqldetect/blob/main/configs/slurm.yaml);
the full `#SBATCH` header is baked into each generated script. Each job `cd`s into the repo
and activates the uv `.venv` with `source .venv/bin/activate` on the compute node. That
`.venv` is built once on the login node (`uv sync --frozen`) and reused from the shared
filesystem; array tasks only activate it and never run `uv sync` concurrently (which would
race on the shared `.venv`).

!!! note "Largest-VRAM-first preference"
    The generated `--partition=A40,A100,V100-32GB,...` list lets SLURM pick whichever frees up
    first. A *strict* largest-first preference requires the admin to give the bigger-GPU
    partitions a higher `PriorityTier` (`scontrol show partition A40 | grep PriorityTier`);
    list order alone is not a guarantee.

## Submitting

```bash
# Preview the plan (manifests + sbatch commands) without submitting:
python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ae --extractors li --dry-run

# Submit for real:
python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ocsvm,ae --extractors li

# Smoke run: cap each cell to a label-stratified subset (passed through to evaluate_suite):
python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ae --extractors sbert --limit 50000
```

Manifests, generated job scripts, and `.out` logs are written under
`reports/slurm/<run-id>/` (git-ignored).

### Concept drift

The concept-drift family (`superviz26-drift`) fans out the same way — its four domains
are the scenarios — but each cell trains once and evaluates two test sets (S1/S2).
`slurm_run_cell` dispatches such cells to `evaluate_drift` automatically (it keys off the
family's `protocol == "drift"`), so the submit command is identical:

```bash
python -m tools.slurm_submit --dataset superviz26-drift --suite all --pipelines ocsvm,lof,ae --extractors li
```

Each cell writes one row (`auroc_s1`, `auroc_s2`, `delta_auroc`, …) to
`reports/superviz26-drift/cells/{pipeline}_{extractor}_{domain}.csv`.

## Results

MLflow is the canonical store: each cell logs an independent run, nested under a parent
that `slurm_submit` pre-creates once (so concurrent jobs don't spawn duplicate parents).
Each cell also writes its row to `reports/{dataset}/cells/*.csv` on disk; read that
directory directly if you need a flat table.
