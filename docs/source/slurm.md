# Running experiments on SLURM

Run these commands from the repository root on the submit node. The site-specific resource settings are in
`configs/slurm.yaml`; set its partitions, GPU architectures, and limits for your cluster. Fetch the required CSVs
as described in [Datasets](datasets.md) before submitting jobs.

## Set up the environments

The setup script installs uv 0.11.13 and Python in the home directory on this cluster, then runs `uv sync` for
the selected profile. Build the CPU environment and the CUDA profiles needed by your partitions on a compute node:

```sh
srun --partition=CPU --mem=16G --pty bash -lc '. tools/setup-env.sh'
srun --partition=CPU --mem=16G --pty bash -lc 'SQLAD_EXTRA=cu126 . tools/setup-env.sh'
srun --partition=CPU --mem=16G --pty bash -lc 'SQLAD_EXTRA=cu130 . tools/setup-env.sh'
```

`cu126` supports V100, A100, A40, and A30; `cu130` supports Blackwell GPUs. The environments are
`.venv-cluster-cpu`, `.venv-cluster-cu126`, and `.venv-cluster-cu130` in the shared checkout. Build the small
submit environment on the submit node:

```sh
. tools/setup-env.sh submit
source .venv-submit/bin/activate
python -m tools.slurm_submit --help
```

The submit environment has no PyTorch. Keep it active for the submission commands below. Jobs select a compatible
cluster environment for their assigned GPU and stop if it is missing or does not match `uv.lock`. Rebuild
environments after a lock file change.

## Track and submit results

Copy the example configuration:

```sh
cp .env.example .env
```

Set `MLFLOW_TRACKING_URI` in `.env` to your MLflow server or shared tracking store. Set the
certificate and login variables from the example if your server requires them. The tools read `.env` from the
repository root. MLflow stores the runs and the finished-cell status used by `--check-mlflow`. Without a working
tracking URI, omit `--check-mlflow`; use `--no-track` if you want CSV results only.

Preview and submit a grid:

```sh
python -m tools.slurm_submit \
  --dataset superviz26 --suite all --methods ocsvm,ae --extractors li --dry-run
python -m tools.slurm_submit \
  --dataset superviz26 --suite all --methods ocsvm,ae --extractors li
```

Each scenario, method, and extractor combination is one task in a SLURM array. The command stays active while it
submits work under the default 24-job cap. A submission does not wait for the experiments to finish. Each cell
writes `reports/{dataset}/cells/*.csv` and `reports/{dataset}/logs/*.log`; submission files are in
`reports/slurm/<run-id>/`. Direct runs of `evaluate_suite` write a combined `reports/{dataset}_results.csv`.

A failed or interrupted cell is submitted at most once per invocation. After fixing the cause, rerun the same
command with `--check-mlflow` to skip cells whose MLflow runs are `FINISHED`. This lookup happens once at startup.

## Few-shot adaptation

Few-shot evaluation needs four pretrained leave-one-domain-out (LODO) autoencoders and the few-shot CSVs. Its
in-domain reference comes from four standard Superviz26 autoencoder runs with the same extractor. Submit both
standard suites with MLflow tracking enabled:

```sh
python -m tools.slurm_submit \
  --dataset superviz26 --suite lodo --methods ae --extractors li
python -m tools.slurm_submit \
  --dataset superviz26 --suite in_domain --methods ae --extractors li
```

After those jobs finish, submit the few-shot jobs:

```sh
python -m tools.slurm_submit \
  --dataset superviz26-fsl --suite all --methods ae --extractors li
```

The LODO runs save source models under `models/`. Few-shot results are written under
`reports/superviz26-fsl/cells/`. Once all runs finish, generate the figure with the project CPU environment:

```sh
source .venv-cluster-cpu/bin/activate
python -m tools.generate_fsl_tex \
  --figure-out reports/superviz26-fsl/few-shot-adaptation.tex
```

The figure reads few-shot AUROC and the four in-domain baseline runs from MLflow. It marks the smallest sample
count whose mean AUROC is at least the in-domain mean minus 0.01. A combined `reports/superviz26_results.csv` is only
written by a direct `evaluate_suite` run; SLURM writes per-cell CSVs.

## Concept drift

The drift family trains on each domain's origin data and scores its origin and shifted test sets:

```sh
python -m tools.slurm_submit \
  --dataset superviz26-drift --suite all --methods ocsvm,lof,ae --extractors li
```

Each cell CSV includes `auroc_s1`, `auroc_s2`, and `delta_auroc`.
