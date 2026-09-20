# SQL attack detection benchmarks

This repository compares anomaly-detection methods for SQL attacks across several feature extractors and Superviz datasets.

## Running experiments

All methods can be trained and evaluated on any compatible machine. Most experiments were run in parallel on a SLURM cluster using the provided submission tool, which assigns CPU or GPU resources as needed and submits jobs gradually to respect cluster limits. GAUR experiments were run locally because they require an instrumented MySQL server, but they can run on any machine where that dependency is available. See [Running experiments on SLURM](docs/source/slurm.md) and [Datasets](docs/source/datasets.md) for the commands and data requirements.

The `gaur-*` extractors collect one parser trace per query, in ten equal parts. Each part writes its feature rows to disk and logs the rows that are done, so a failed collection continues at the last complete part. A query that GAUR cannot trace keeps its row, with `n_parser_invoc = 0`. `gaur_sqld` sets the number of processes with its own `n_workers` setting, and it stops the collection with a `GaurServerError` if the server writes a single shared `gaur.log`.

## Development environment

Nix is the recommended setup and the one used in CI. It provides the expected Python interpreter and system libraries, synchronizes the packages from `uv.lock`, and activates the project virtual environment:

```sh
nix-shell
```

Without Nix, for example on the SLURM cluster, the same lock file builds a second environment. `uv` installs itself and its own interpreter, so nothing else is needed:

```sh
. tools/setup-env.sh
```

The script installs packages from `uv.lock` into `.venv-nix` under Nix or `.venv-cluster` elsewhere.
It defaults to CUDA 12.6. Set `SQLAD_EXTRA=cpu` without a GPU, or `SQLAD_EXTRA=cu130` for Blackwell GPUs
(`.venv-cluster-cu130` outside Nix).

Run commands with `uv run --frozen --extra cu126 <command>`, using `cpu` or `cu130` to match your setup.
Always pass `--extra` to keep the chosen PyTorch build. After `git pull`, source the setup script again,
or enable automatic setup with `git config core.hooksPath .githooks`.

For pip, use `requirements.txt`, exported from the lock file, and choose a PyTorch build for your hardware.

## Project structure

The main directories are:

```txt
├── configs/                  # Experiment and cluster configuration
├── data/                     # Dataset metadata
├── docs/                     # Documentation
├── models/                   # Trained models
├── reports/                  # Evaluation results
├── src/sqlad_benchmarking/   # Training and evaluation code
├── tests/                    # Test suite
└── tools/                    # Dataset, SLURM, and report utilities
```

### Credits

This repository structure is based on [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for Machine Learning Operations (MLOps).
