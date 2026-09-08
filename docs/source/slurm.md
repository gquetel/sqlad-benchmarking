# Running experiments on SLURM

SLURM is optional: all methods can be trained and evaluated on any compatible machine. The provided submission tool was used to run most experiments in parallel, assigning CPU or GPU resources as needed and submitting jobs gradually to respect cluster limits. Therefore some part of this code is highly opinionated to my preferences / to the machines of the cluster and their environement. GAUR experiments were run locally because they require an instrumented MySQL server (requiring nix), but they can run on any machine where that dependency is available.

## How it works

`tools.slurm_submit` creates one job for each combination of scenario, method, and feature extractor. It groups these jobs into arrays according to their resource needs:

- **CPU** for methods and extractors that do not use a GPU.
- **GPU** for autoencoders and embedding-based extractors. Models with higher memory needs are
  limited to suitable GPUs.

Each array task runs one cell through `evaluate_suite` and writes its **own** per-cell CSV to `reports/{dataset}/cells/{method}_{extractor}_{scenario}.csv` — so the parallel jobs never share a writer and a rerun simply overwrites its own file. It also writes that cell's log to `reports/{dataset}/logs/{stem}.log` and uploads it to the cell's MLflow run, so a failure is readable from the run itself rather than only from the array's `.log` on the submit node.

Cluster-specific settings, including partitions, memory limits, and time limits, live in [`configs/slurm.yaml`](https://github.com/gquetel/sqlad-benchmarking/blob/main/configs/slurm.yaml). Adapt this file before using the tool on another cluster. Its `env` block names the venv the array tasks activate, and the modules to load first.

## Cluster environment

The cluster has no Nix, and no module supplies Python 3.14. Thus `uv` installs itself and its own interpreter, both in your home directory, and builds the venv from the same `uv.lock` as the dev machine:

```sh
. tools/setup-env.sh   # -> .venv-cluster
```

Run this once on the submit node. The home directory is shared, so every node sees the result. No module is needed: the wheels find the system libraries, and torch carries its own CUDA runtime. If a node ever lacks one, list it in `env.modules` (or in `SQLAD_MODULES`), and the job scripts load it after `module purge`.

The CUDA build stays pinned to `cu126` because the V100 partitions are Volta (compute capability 7.0), which newer CUDA versions drop. The same build also runs on the A100 partitions, which keeps every cell of a table on one stack.

Array tasks only activate the venv, because a concurrent `uv sync` would race on one shared directory. `slurm_submit` therefore checks the venv on the submit node, and refuses to submit when either check fails:

- it matches `uv.lock`;
- its torch carries kernels for every GPU partition the cells can land on, listed per partition under `gpu_arch` in `configs/slurm.yaml`. A wheel built without one, which is what a `uv` command that omits `--extra cu126` installs, otherwise reaches the node and dies there with `CUDA error: no kernel image is available for execution on the device` — once per cell, after the array is already queued.

Re-run the script after every `git pull`, or enable the hook once with `git config core.hooksPath .githooks`. Always pass `--extra cu126` to `uv run` in this checkout: without it `uv` replaces the pinned torch in the venv the compute nodes activate.

## Submitting

By default, the command remains active and submits more work as cluster capacity becomes available:

```bash
# Preview the plan (what is missing + the sbatch commands) without submitting:
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --extractors li --dry-run

# Submit for real:
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ocsvm,ae --extractors li

# Submit everything at once, ignoring the in-flight cap:
python -m tools.slurm_submit --dataset superviz26 --methods ae --extractors li --no-queue

# Smoke run: cap each cell to a label-stratified subset (passed through to evaluate_suite):
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --extractors sbert --limit 50000
```

Manifests, generated job scripts, and `.out` logs are written under `reports/slurm/<run-id>/` (git-ignored).

### Gradual submission

Run it on the submit node, detached, so a dropped VPN does not kill it:

```bash
nohup uv run --frozen --extra cu126 python -m tools.slurm_submit \
  --dataset superviz26 --suite all --methods ae \
  --extractors roberta,modernbert,codebert,flan-t5,sentbert,qwen3-emb,llm2vec \
  --max-jobs 24 --interval 300 > reports/slurm/queue.log 2>&1 &
```

Preview it anywhere first (off the submit node it assumes an empty queue):

```bash
python -m tools.slurm_submit --methods ae --extractors li,cv,sbert --dry-run
```

The gradual mode keeps the number of submitted jobs below `--max-jobs` and checks for available capacity every `--interval` seconds. It reads the SLURM queue each tick, so it never submits a unit that an earlier invocation still has in flight. No separate state file is required.

**Each cell is submitted exactly once.** A cell that fails is not resubmitted: a failure is nearly always a bug or a wrong resource request, and a retry only burns compute and fills the tracking server with dead runs. Read the log under `reports/slurm/<run-id>/logs/`, fix the cause, then run the command again.

Pass `--no-queue` to submit all selected experiments at once.

`--check-mlflow` looks up, once at startup, which cells already have a FINISHED run and drops them. That is how a rerun fills in the holes left by a broken batch, instead of repeating the whole grid.

**Preemptible GPU jobs.** GPU jobs use `--gpu-qos runfill` by default, so the cluster can kill one when someone else claims the GPU. A preempted cell is not resubmitted either; run the command again with `--check-mlflow` to pick up what it left behind. Pass `--gpu-qos ""` to use the cluster's default QoS.

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
