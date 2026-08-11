# Running the suite on SLURM

`evaluate_suite` runs the grid (methods × extractors × scenarios) serially in one
process. On a cluster you usually want each **cell** — one `(scenario, method,
extractor)` — to run as its own job so they execute in parallel (e.g. the 8 Superviz26
scenarios become 8 jobs).

## How it works

`tools.slurm_submit` enumerates the grid and submits **one job array per resource
class**. Cells are bucketed by their needs:

- **CPU** — cells that need no GPU (e.g. `ocsvm/li`) → the `cpu` partition.
- **GPU** — cells that train an autoencoder (`method == "ae"`) or use an embedding
  extractor (`sbert`, `sbert2`, `codet5`, `roberta`, `modernbert`, `codebert`,
  `flan-t5`, `sentbert`, or `qwen3-emb`) → a GPU array whose partition list is the GPU
  partitions with **enough VRAM** for that cell. Each cell's minimum comes from
  `min_vram_gb` (keyed by `extractor` or `engine:extractor`, else `default`); partitions
  are tried largest-VRAM first so cells prefer the fastest GPU. A hungry model like CodeT5+
  (`codet5: 24`) is thus never scheduled on the 16 GB V100.

Each array task runs one cell through `evaluate_suite` and writes its **own** per-cell CSV
to `reports/{dataset}/cells/{method}_{extractor}_{scenario}.csv` — so the parallel jobs
never share a writer and a rerun simply overwrites its own file.

Site settings (partitions, resources, `min_vram_gb`) live in
[`configs/slurm.yaml`](https://github.com/gquetel/sqlad-benchmarking/blob/main/configs/slurm.yaml);
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
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --extractors li --dry-run

# Submit for real:
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ocsvm,ae --extractors li

# Smoke run: cap each cell to a label-stratified subset (passed through to evaluate_suite):
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --extractors sbert --limit 50000
```

Manifests, generated job scripts, and `.out` logs are written under
`reports/slurm/<run-id>/` (git-ignored).

### Queueing a grid larger than the job cap

Run this on the submit node, detached, when the grid is too big to submit in one go:

```bash
nohup uv run python -m tools.slurm_queue \
  --dataset superviz26 --suite all --methods ae \
  --extractors roberta,modernbert,codebert,flan-t5,sentbert,qwen3-emb \
  --max-jobs 24 --interval 300 > reports/slurm/queue.log 2>&1 &
```

Preview it anywhere first (off the submit node it assumes an empty queue):

```bash
python -m tools.slurm_queue --methods ae --extractors li,cv,sbert --dry-run --once
```

**Why.** The cluster caps in-flight jobs per user (~24). The full grid (every extractor ×
method × 8 scenarios) is hundreds of jobs, so `slurm_submit` alone gets rejected.

**What it does.**
[`tools.slurm_queue`](https://github.com/gquetel/sqlad-benchmarking/blob/main/tools/slurm_queue.py)
splits the grid into **units** — one `(method, extractor)` over all scenarios, i.e. one
`slurm_submit` call — and every `--interval` seconds submits as many as the headroom allows.

**No state file.** Each tick it rebuilds the picture from `squeue` and from disk:

- *done* — every `reports/{dataset}/cells/*.csv` for the unit exists
- *in flight* — a job with the unit's job name is in `squeue`
- *pending* — everything else

Kill it, lose the VPN, start it twice: it never resubmits finished work.

**Preemptible GPU jobs.** The queue submits GPU cells under `--gpu-qos runfill` by default —
a preemptible QoS, so many more GPU jobs are allowed to run, but SLURM kills them when
someone else needs the GPU. A killed cell writes no CSV, so the next tick sees it as pending
and resubmits it. Pass `--gpu-qos ""` for the default (non-preemptible) QoS. `slurm_submit`
has the same flag, off by default.

**Counting mode.** `--max-jobs` counts **array tasks** by default (`squeue -r`) — correct when
the cap is on *submitted* jobs. If your cap is on *concurrently running* jobs, pass
`--no-count-array-tasks`, or drop the queue entirely and throttle natively with `--array=0-N%24`.
Check which you have: `sacctmgr show assoc user=$USER format=maxsubmit,maxjobs`.

### Concept drift

The concept-drift family (`superviz26-drift`) fans out the same way — its four domains
are the scenarios — but each cell trains once and evaluates two test sets (S1/S2).
`slurm_run_cell` dispatches such cells to `evaluate_drift` automatically (it keys off the
family's `protocol == "drift"`), so the submit command is identical:

```bash
python -m tools.slurm_submit --dataset superviz26-drift --suite all --methods ocsvm,lof,ae --extractors li
```

Each cell writes one row (`auroc_s1`, `auroc_s2`, `delta_auroc`, …) to
`reports/superviz26-drift/cells/{method}_{extractor}_{domain}.csv`.

### Few-shot adaptation

The few-shot family (`superviz26-fsl`) fans out the same way — its four target domains
are the scenarios — but each cell *adapts a pretrained LODO autoencoder* rather than
training from scratch. For a target domain it loads the matching LODO checkpoint (e.g.
`bcd-a` for target `a`), fine-tunes only the autoencoder on `k` benign target-domain
samples (frozen extractor, learning rate ÷ 10), recalibrates the threshold from those
`k` samples, and scores the target test set — sweeping `k ∈ {0, 5, …, 10000}` over
several seeds. It is **autoencoder-only**, and the pretrained LODO models must already
exist under `models/` (run the `superviz26` `lodo` AE grid first). `slurm_run_cell`
dispatches such cells to `evaluate_fsl` (it keys off `protocol == "fsl"`):

```bash
# 1. Pretrain the LODO autoencoders (produces models/ae_<extractor>_superviz26_<lodo>.pt):
python -m tools.slurm_submit --dataset superviz26 --suite lodo --methods ae --extractors li

# 2. Run the few-shot sweep on top of them:
python -m tools.slurm_submit --dataset superviz26-fsl --suite all --methods ae --extractors li
```

Each cell writes one row per `(target, k, seed)` (`auroc`, `auprc`, `n_finetune`, …) to
`reports/superviz26-fsl/cells/ae_{extractor}_{target}.csv`; the per-seed AUROCs are
averaged downstream and compared against the in-domain table
(`reports/superviz26_results.csv`), where recovery is "within 0.01 AUROC of in-domain".

## Results

MLflow is the canonical store: each cell logs an independent run, nested under a parent
that `slurm_submit` pre-creates once (so concurrent jobs don't spawn duplicate parents).
Each cell also writes its row to `reports/{dataset}/cells/*.csv` on disk; read that
directory directly if you need a flat table.
