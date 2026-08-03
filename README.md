# Baselines sqldetect

Codebase for SQL attack detection: anomaly-detection heads (One-Class SVM, LOF, AutoEncoder) over pluggable SQL feature extractors (Li hand-crafted features, Loginov hand-crafted features, CountVectorizer, SecureBERT, CodeT5+...), evaluated on the Superviz SQL datasets.

## Development environment

There are three ways to get a working environment, depending on what your machine already has installed. Yet, they should provide the same environment: same Python version, the same pinned packages (either through uv.lock or requirements.txt)

### With Nix (NixOS, or any machine with Nix installed)

This is the recommended path and the one used in CI. It provides a CUDA-compatible FHS environment and pins both the Python interpreter and the system libraries:

```sh
nix-shell
```

This builds (or refreshes) the `.venv` from `uv.lock` via `uv sync --frozen` and drops you into a shell with the virtualenv already activated. You don't need uv installed on the host: Nix provides it.

### Without Nix, but with uv installed

If you don't have Nix but do have [uv](https://docs.astral.sh/uv/), you can recreate the exact locked environment directly:

```sh
uv sync --frozen
source .venv/bin/activate
```

### Without Nix and without uv

If you have neither, fall back to the pip-compatible `requirements.txt`. It is automatically generated from `uv.lock`. Use it in a virtualenv of your choice.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # GitHub Actions and dependabot
│   ├── dependabot.yml
│   └── workflows/
│       └── tests.yaml
├── data/                     # Data directory
│   └── raw/
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yaml
│   └── source/
│       ├── datasets.md
│       └── index.md
├── models/                   # Trained models
├── nix/                      # Shared Nix environments
│   ├── ci-env.nix            # CI environment (Python + system libs for native extensions)
│   └── python-env.nix        # Pinned Python interpreter (used by Docker and CI)
├── npins/                    # Nix pin sources
├── reports/                  # Evaluation reports (CSV)
├── src/                      # Source code
│   └── sqlad_benchmarking/
│       ├── __init__.py
│       ├── api.py
│       ├── data.py            # CSV loading and split/subsample helpers
│       ├── datasets/          # Dataset families (Superviz25, Superviz26) + registry
│       ├── detector_model.py  # MLflow model-from-code wrapper
│       ├── evaluate.py        # CLI: score a saved method on a CSV
│       ├── evaluate_suite.py  # CLI: run an evaluation grid + MLflow logging
│       ├── features/          # Feature extractors (li, loginov, countvect, securebert, codet5)
│       ├── metrics.py         # Threshold calibration and metric computation
│       ├── model.py           # Detector heads (OCSVM, LOF, AutoEncoder) + factory
│       ├── tracking.py        # MLflow configuration and helpers
│       ├── train.py           # CLI: fit and save a method
│       └── visualize.py       # ROC / AUPRC curve helpers
├── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_datasets.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_model.py
├── tools/                    # Dataset fetch, SLURM dispatch, figure/table generation
│   ├── fetch_superviz25.py
│   ├── fetch_superviz26.py     # Superviz26 archive: main + drift + fsl groups (opt-in)
│   ├── generate_baselines_tex.py
│   ├── generate_drift_figure.py
│   ├── generate_fsl_tex.py
│   ├── slurm_run_cell.py
│   └── slurm_submit.py
├── .env.example              # Example MLflow tracking configuration
├── .gitignore
├── .pre-commit-config.yaml
├── AGENTS.md                 # Guidance for coding agents
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Pinned runtime deps, exported from uv.lock for pip users
├── shell.nix                 # Nix development environment
├── tasks.py                  # Invoke task definitions
├── treefmt.toml              # Formatter configuration
└── uv.lock                   # Canonical, hashed dependency lock (uv)
```

### Credits
This opinionated repository structure is based on [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).
