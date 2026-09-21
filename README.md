# SQL attack detection benchmarks

This repository compares anomaly-detection methods for SQL attacks across several feature extractors and Superviz datasets.

## Running experiments

All methods can be trained and evaluated on any compatible machine. Most experiments were run in parallel on a SLURM cluster using the provided submission tool, which assigns CPU or GPU resources as needed and submits jobs gradually to respect cluster limits. GAUR experiments were run locally because they require an instrumented MySQL server, but they can run on any machine where that dependency is available. See [Running experiments on SLURM](docs/source/slurm.md) and [Datasets](docs/source/datasets.md) for the commands and data requirements.

The `gaur-*` extractors collect one parser trace per query, in ten equal parts. Each part writes its feature rows to disk and logs the rows that are done, so a failed collection continues at the last complete part. A query that GAUR cannot trace keeps its row, with `n_parser_invoc = 0`. `gaur_sqld` sets the number of processes with its own `n_workers` setting, and it stops the collection with a `GaurServerError` if the server writes a single shared `gaur.log`.

## Development environment

Nix is the recommended local setup and the one used in CI. It provides the expected Python interpreter, uv, and system libraries. Entering the shell does not install Python packages:

```sh
nix-shell
uv run --frozen --extra cpu python -m sqlad_benchmarking.evaluate_suite --help
```

The shell uses the CPU profile by default. For a GPU, select its profile before entering the shell and use the matching extra in each command:

```sh
SQLAD_EXTRA=cu126 nix-shell
uv run --frozen --extra cu126 python -m sqlad_benchmarking.evaluate_suite --help
```

Use `SQLAD_EXTRA=cu130` for Blackwell GPUs. Each profile has its own `.venv-nix-<profile>` directory. The first `uv run` installs the locked packages for that profile; later runs reuse them. Always pass `--extra` so uv keeps the selected PyTorch build.

Without Nix, for example on the SLURM cluster, build each environment you need explicitly with `. tools/setup-env.sh` (CPU), `SQLAD_EXTRA=cu126 . tools/setup-env.sh`, or `SQLAD_EXTRA=cu130 . tools/setup-env.sh`. These use separate `.venv-cluster-<profile>` directories. Build GPU profiles on a compute node; see the [SLURM guide](docs/source/slurm.md). `. tools/setup-env.sh submit` builds the smaller `.venv-submit` environment on the submit node.

After a merge changes dependencies, local `uv run` syncs its profile on the next use. Cluster environments need an explicit rebuild. To get a reminder after merges, enable `git config core.hooksPath .githooks`; the hook only prints a notice.

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
