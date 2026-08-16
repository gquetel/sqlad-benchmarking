# SQL attack detection benchmarks

This repository compares anomaly-detection methods for SQL attacks across several feature extractors and Superviz datasets.

## Running experiments

All methods can be trained and evaluated on any compatible machine. Most experiments were run in parallel on a SLURM cluster using the provided submission tool, which assigns CPU or GPU resources as needed and submits jobs gradually to respect cluster limits. GAUR experiments were run locally because they require an instrumented MySQL server, but they can run on any machine where that dependency is available. See [Running experiments on SLURM](docs/source/slurm.md) and [Datasets](docs/source/datasets.md) for the commands and data requirements.

## Development environment

Nix is the recommended setup and the one used in CI. It provides the expected Python interpreter and system libraries, synchronizes the packages from `uv.lock`, and activates the project virtual environment:

```sh
nix-shell
```

Without Nix, the Python packages can be installed with [uv](https://docs.astral.sh/uv/) when a compatible Python interpreter is already available:

```sh
uv sync --frozen
source .venv/bin/activate
```

`requirements.txt` provides a pip-compatible export of the locked packages for environments that do not use uv.

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