# Running experiments on SLURM

SLURM is optional: all methods can be trained and evaluated on any compatible machine. The provided submission tool was used to run most experiments in parallel, assigning CPU or GPU resources as needed and submitting jobs gradually to respect cluster limits. Therefore some part of this code is highly opinionated to my preferences / to the machines of the cluster and their environement. GAUR experiments were run locally because they require an instrumented MySQL server (requiring nix), but they can run on any machine where that dependency is available.

## How it works

`tools.slurm_submit` creates one job for each combination of scenario, method, and feature extractor. It groups these jobs into arrays according to their resource needs:

- **CPU** for methods and extractors that do not use a GPU.
- **GPU** for autoencoders and embedding-based extractors. Models with higher memory needs are
  limited to suitable GPUs.

Each array task runs one cell through `evaluate_suite` and writes its **own** per-cell CSV to `reports/{dataset}/cells/{method}_{extractor}_{scenario}.csv` — so the parallel jobs never share a writer and a rerun simply overwrites its own file.

Cluster-specific settings, including partitions, memory limits, and time limits, live in [`configs/slurm.yaml`](https://github.com/gquetel/sqlad-benchmarking/blob/main/configs/slurm.yaml). Adapt this file before using the tool on another cluster. Jobs use the uv `.venv` created on the shared filesystem with `uv sync --frozen`.

## Submitting

By default, the command remains active and submits more work as cluster capacity becomes available:

```bash
# Preview the plan (what is missing + the sbatch commands) without submitting:
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --extractors li --dry-run --once

# Submit for real:
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ocsvm,ae --extractors li

# Submit everything at once and ignore previous MLflow runs:
python -m tools.slurm_submit --dataset superviz26 --methods ae --extractors li --no-queue --no-check-mlflow

# Smoke run: cap each cell to a label-stratified subset (passed through to evaluate_suite):
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --extractors sbert --limit 50000
```

Manifests, generated job scripts, and `.out` logs are written under `reports/slurm/<run-id>/` (git-ignored).

### Gradual submission

Run it on the submit node, detached, so a dropped VPN does not kill it:

```bash
nohup uv run python -m tools.slurm_submit \
  --dataset superviz26 --suite all --methods ae \
  --extractors roberta,modernbert,codebert,flan-t5,sentbert,qwen3-emb \
  --max-jobs 24 --interval 300 > reports/slurm/queue.log 2>&1 &
```

Preview it anywhere first (off the submit node it assumes an empty queue):

```bash
python -m tools.slurm_submit --methods ae --extractors li,cv,sbert --dry-run --once
```

The gradual mode keeps the number of submitted jobs below `--max-jobs` and checks for available capacity every `--interval` seconds. Before submitting work, it checks MLflow for completed runs and the SLURM queue for active jobs. Restarting the command therefore skips completed experiments and makes interrupted experiments eligible to run again. No separate state file is required.

Pass `--no-queue` to submit all selected experiments at once.

`--no-check-mlflow` skips the lookup and submits every cell once — for a fresh grid, or when the tracking server is unreachable.

**Preemptible GPU jobs.** GPU jobs use `--gpu-qos runfill` by default. If the cluster preempts one of these jobs, gradual submission makes it eligible to run again. Pass `--gpu-qos ""` to use the cluster's default QoS.

**Counting mode.** `--max-jobs` counts **array tasks** by default (`squeue -r`) — correct when the cap is on *submitted* jobs. If your cap is on *concurrently running* jobs, pass `--no-count-array-tasks`, or drop the queue entirely and throttle natively with `--array=0-N%24`. Check which you have: `sacctmgr show assoc user=$USER format=maxsubmit,maxjobs`.

### Concept drift

The concept-drift family (`superviz26-drift`) fans out the same way — its four domains are the scenarios — but each cell trains once and evaluates two test sets (S1/S2). `slurm_run_cell` dispatches such cells to `evaluate_drift` automatically (it keys off the family's `protocol == "drift"`), so the submit command is identical:

```bash
python -m tools.slurm_submit --dataset superviz26-drift --suite all --methods ocsvm,lof,ae --extractors li
```

Each cell writes one row (`auroc_s1`, `auroc_s2`, `delta_auroc`, …) to `reports/superviz26-drift/cells/{method}_{extractor}_{domain}.csv`.

### Few-shot adaptation

The few-shot family (`superviz26-fsl`) fans out the same way — its four target domains are the scenarios — but each cell *adapts a pretrained LODO autoencoder* rather than training from scratch. For a target domain it loads the matching LODO checkpoint (e.g. `bcd-a` for target `a`), fine-tunes only the autoencoder on `k` benign target-domain samples (frozen extractor, learning rate ÷ 10), recalibrates the threshold from those `k` samples, and scores the target test set — sweeping `k ∈ {0, 5, …, 10000}` over several seeds. It is **autoencoder-only**, and the pretrained LODO models must already exist under `models/` (run the `superviz26` `lodo` AE grid first). `slurm_run_cell` dispatches such cells to `evaluate_fsl` (it keys off `protocol == "fsl"`):

```bash
# 1. Pretrain the LODO autoencoders (produces models/ae_<extractor>_superviz26_<lodo>.pt):
python -m tools.slurm_submit --dataset superviz26 --suite lodo --methods ae --extractors li

# 2. Run the few-shot sweep on top of them:
python -m tools.slurm_submit --dataset superviz26-fsl --suite all --methods ae --extractors li
```

Each cell writes one row per `(target, k, seed)` (`auroc`, `auprc`, `n_finetune`, …) to `reports/superviz26-fsl/cells/ae_{extractor}_{target}.csv`; the per-seed AUROCs are averaged downstream and compared against the in-domain table (`reports/superviz26_results.csv`), where recovery is "within 0.01 AUROC of in-domain".

## Results

MLflow is the canonical store: each cell logs an independent run, nested under a parent that `slurm_submit` pre-creates once (so concurrent jobs don't spawn duplicate parents). Each cell also writes its row to `reports/{dataset}/cells/*.csv` on disk; read that directory directly if you need a flat table.

When a cell is retried, any `RUNNING` run for the same cell is soft-deleted before the new run starts. Runs with any other status and runs for other cells are preserved.
