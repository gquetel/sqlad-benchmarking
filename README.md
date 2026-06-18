# mlops_sqldetect

An MLOps pipeline for one-class SQL injection detection: anomaly-detection heads (One-Class SVM, LOF, AutoEncoder) over pluggable SQL feature extractors (Li hand-crafted features, CountVectorizer, SecureBERT), evaluated on the Superviz SQL datasets with MLflow experiment tracking.

## Opinionated 

- Using `nix` and its ecosystem to manage development environments and foster reproducibility.
  - `nix/python-env.nix` is the single source of truth for the Python version: shared by Docker builds and CI, so the interpreter is fully pinned and reproducible across environments.
  - `shell.nix` wraps `nix/python-env.nix` with `buildFHSEnv` for local development, which provides a CUDA-compatible FHS environment on both NixOS and non-NixOS machines.
  - Runtime and dev dependencies are managed by [uv](https://docs.astral.sh/uv/) on top of the nix-provided interpreter (uv never downloads its own Python — `UV_PYTHON_DOWNLOADS=never`). uv is preferred over nix packages because it can pull pre-built wheels (including GPU-specific CUDA wheels) from PyPI caches. Using nix for those would require local rebuilds due to the absence of binary caches for GPU packages.
  - Dependencies are declared in `pyproject.toml` and locked in `uv.lock` (universal, hashed). A pip-compatible `requirements.txt` is exported from the lock (`uv export`) so contributors who don't use uv can still `pip install -r requirements.txt`.
- MLflow for experiment tracking: `evaluate_suite` logs params, metrics, the per-epoch AE training loss, and the fitted model artifact when `MLFLOW_TRACKING_URI` is set (opt out with `--no-track`). Tracking is configured via `.env` (see `.env.example`).
- Delete dependabot updates PR for two reasons: (1) notification fatigue, (2) bumping package version can lead to different experiments results or breaking changes that needs to be tested. It is time consuming to be done properly, hence packages updates are done at my discretion. The exception to this is dependabot security updates which can be activated through GitHub's interface that warns me about potential vulnerabilities in my dependencies. 

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
│   └── mlops_sqldetect/
│       ├── __init__.py
│       ├── api.py
│       ├── data.py            # CSV loading and split/subsample helpers
│       ├── datasets/          # Dataset families (Superviz25, Superviz26) + registry
│       ├── detector_model.py  # MLflow model-from-code wrapper
│       ├── evaluate.py        # CLI: score a saved pipeline on a CSV
│       ├── evaluate_suite.py  # CLI: run an evaluation grid + MLflow logging
│       ├── features/          # Feature extractors (li, countvect, securebert)
│       ├── metrics.py         # Threshold calibration and metric computation
│       ├── model.py           # Detector heads (OCSVM, LOF, AutoEncoder) + factory
│       ├── tracking.py        # MLflow configuration and helpers
│       ├── train.py           # CLI: fit and save a pipeline
│       └── visualize.py       # ROC / AUPRC curve helpers
├── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_datasets.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_model.py
├── tools/                    # Dataset download scripts
│   ├── fetch_superviz25.py
│   └── fetch_superviz26.py
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
