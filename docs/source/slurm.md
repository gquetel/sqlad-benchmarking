# Running experiments on SLURM

SLURM runs experiments in parallel with the required CPU or GPU resources. Any compatible machine can also
run them directly. The settings here match the cluster used for the experiments; adapt them for your cluster.
GAUR experiments run locally because they need an instrumented MySQL server set up through Nix.

## How it works

`tools.slurm_submit` creates one task per scenario, method, and feature extractor combination (a **cell**).
It groups tasks with similar resource needs into SLURM job arrays:

- **CPU** for methods and extractors that do not use a GPU.
- **GPU** for autoencoders and embedding-based extractors. Models with higher memory needs are
  limited to suitable GPUs.

SLURM can assign GPU tasks to configured partitions (groups of nodes) with enough GPU memory and
permission for the selected extractor.
Each task selects the Python environment for its assigned GPU (see
[Two CUDA builds](#two-cuda-builds)).

Each task runs the dataset's evaluation and writes a CSV under `reports/{dataset}/cells/`.
It writes `reports/{dataset}/logs/{stem}.log` and uploads the log to MLflow, including on failure.

Edit [`configs/slurm.yaml`](https://github.com/gquetel/sqlad-benchmarking/blob/main/configs/slurm.yaml)
to set partitions, memory, and time limits. `cuda_builds` lists the Python environments;
`env.modules` lists system modules to load before Python. `allowed_partitions` limits an extractor
to named GPU partitions; LLM2Vec uses only `RTX6000PRO` and `A100`.

## Cluster environment

On this cluster, setup uses uv 0.11.13 and installs Python in the home directory. It creates these
environments from `uv.lock`:

| Python environment | Used for | Setup command |
|---|---|---|
| `.venv-cluster-cpu` | CPU array tasks | `. tools/setup-env.sh` |
| `.venv-cluster-cu126` | array tasks on a V100, A100, A40 or A30 | `SQLAD_EXTRA=cu126 . tools/setup-env.sh` |
| `.venv-cluster-cu130` | array tasks on an `RTX6000PRO` (Blackwell) GPU | `SQLAD_EXTRA=cu130 . tools/setup-env.sh` |
| `.venv-submit` | `slurm_submit`, on the submit node | `. tools/setup-env.sh submit` |

Installing GPU packages exceeds the submit node's 3.5 GiB memory limit. Build each GPU environment
on a compute node, and build the CPU environment if you will submit CPU tasks:

```sh
srun --partition=CPU --mem=16G --pty bash -lc '. tools/setup-env.sh'
srun --partition=CPU --mem=16G --pty bash -lc 'SQLAD_EXTRA=cu126 . tools/setup-env.sh'
srun --partition=CPU --mem=16G --pty bash -lc 'SQLAD_EXTRA=cu130 . tools/setup-env.sh'
```

The shared home directory makes the result available to every node. If a system library is missing, add its module to `env.modules` or `SQLAD_MODULES`.

`.venv-submit` omits torch and fits within the submit node's limit. Setup checks that
`tools.slurm_submit --help` runs in this environment:

```sh
. tools/setup-env.sh submit          # once
source .venv-submit/bin/activate
python -m tools.slurm_submit ...
```

Without `UV_PROJECT_ENVIRONMENT`, `uv run` manages a separate `.venv`. Activate the required environment or call its Python directly.

A task stops if its Python environment is missing or does not match `uv.lock` (checked with `uv sync --check`).
Rebuild each profile you use after the lock file changes. The setup script stores each profile in
`.venv-cluster-<profile>`, so adding or removing a CUDA build does not replace the CPU environment.

### Two CUDA builds

The configured PyTorch builds support different GPUs: `cu126` supports V100 but not Blackwell;
`cu130` supports Blackwell but not V100. Both environments are needed.

Each `cuda_builds` entry lists an environment path (`venv`), a package selection (`extra`), and supported
GPU architectures (`arch`). At startup, a task checks its GPU with `nvidia-smi` and uses the first matching
build. With the current order, V100, A100, A40, and A30 use `cu126`; RTX6000PRO uses `cu130`.
Tasks with no detected GPU use the CPU environment from `env`.

Before submission, `slurm_submit` checks the architectures listed in `gpu_arch` for the selected partitions.
It rejects architectures with no compatible build and warns about partitions missing a `gpu_arch` entry.

## Submitting

By default, the command remains active and submits more work as cluster capacity becomes available:

```bash
# Preview without submitting:
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --extractors li --dry-run

# Submit:
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ocsvm,ae --extractors li

# Submit everything at once, ignoring the job limit:
python -m tools.slurm_submit --dataset superviz26 --methods ae --extractors li --no-queue

# Quick check: sample up to 50,000 random rows from each scenario CSV before train/test splitting:
python -m tools.slurm_submit --dataset superviz26 --suite all --methods ae --extractors sbert --limit 50000
```

Task lists, job scripts, and SLURM logs are written under `reports/slurm/<run-id>/` (excluded from Git).

### Gradual submission

Run it on the submit node. The command stays active until every cell is submitted, so
use a terminal that survives a dropped VPN (`tmux` or `screen`):

```bash
.venv-submit/bin/python -m tools.slurm_submit \
  --dataset superviz26 --suite all --methods ae \
  --extractors roberta,modernbert,codebert,flan-t5,sentbert,qwen3-emb,llm2vec \
  --max-jobs 24 --interval 300
```

Preview it anywhere first (off the submit node it assumes an empty queue):

```bash
python -m tools.slurm_submit --methods ae --extractors li,cv,sbert --dry-run
```

The command checks the SLURM queue every `--interval` seconds and submits groups of cells while staying
within `--max-jobs`. Each group shares a method and extractor. Groups already queued or running are skipped.

**Each cell is submitted at most once per invocation.** Failed cells are not retried automatically.
Read `reports/slurm/<run-id>/logs/`, fix the cause, and rerun the command with `--check-mlflow`.

Pass `--no-queue` to submit all selected experiments at once.

`--check-mlflow` checks MLflow once at startup and skips cells with a FINISHED run.

**Interrupted GPU jobs.** The default `--gpu-qos runfill` lets the cluster stop a job when another user
needs its GPU. Rerun with `--check-mlflow` to finish interrupted cells.
Use `--gpu-qos ""` for the cluster's default scheduling policy.

**Counting jobs.** `--max-jobs` counts each queued or running array task by default (`squeue -r`).
`--no-count-array-tasks` counts each array as one job. Check your cluster's limits with
`sacctmgr show assoc user=$USER format=maxsubmit,maxjobs`.

### Concept drift

`superviz26-drift` measures performance before and after a change in the data.
Each cell trains once for one of four domains and evaluates the original (S1) and shifted (S2) test sets.
The task runs `evaluate_drift` automatically:

```bash
python -m tools.slurm_submit --dataset superviz26-drift --suite all --methods ocsvm,lof,ae --extractors li
```

Each cell writes one row (`auroc_s1`, `auroc_s2`, `delta_auroc`, …) to `reports/superviz26-drift/cells/{method}_{extractor}_{domain}.csv`.

### Few-shot adaptation

`superviz26-fsl` adapts an autoencoder to one of four target domains using a small number of benign examples.
It requires models trained on the other domains (leave-one-domain-out, or LODO) under `models/`.
For example, target `a` uses the `bcd-a` model, trained on domains `b`, `c`, and `d`.

Each task runs `evaluate_fsl` across the configured sample counts `k` (0 to 10,000) and random seeds.
It updates only the autoencoder, at one-tenth the training learning rate, then uses the same examples
to set the detection threshold. The feature extractor stays fixed; `k=0` measures the unadapted model.

```bash
# Train the source models (models/ae_<extractor>_superviz26_<lodo>.pt):
python -m tools.slurm_submit --dataset superviz26 --suite lodo --methods ae --extractors li

# Adapt and evaluate them on each target domain:
python -m tools.slurm_submit --dataset superviz26-fsl --suite all --methods ae --extractors li
```

Each cell writes one row per `(target, k, seed)` to `reports/superviz26-fsl/cells/ae_{extractor}_{target}.csv`.
Reports average AUROC across seeds and compare it with training on the target domain
(`reports/superviz26_results.csv`). Recovery means being within 0.01 AUROC of that result.

## Results

MLflow is the main results store. Each cell logs a run under a shared parent created before submission.
CSV results are also available under `reports/{dataset}/cells/`.

When you rerun a cell, its previous `RUNNING` runs are marked as deleted in MLflow.
Other statuses and cells are unaffected.

### Killed jobs

Failed cells mark their MLflow run as `FAILED` and upload their log. If a cell cannot do so, the job script's
exit trap makes the same attempt and attaches any log in `reports/{dataset}/logs/`. A run can remain `RUNNING`
if the trap or MLflow is unavailable.
