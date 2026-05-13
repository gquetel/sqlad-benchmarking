# mlops_sqldetect

A short description of the project.

## Opinionated 

- Using `nix` and its ecosystem to manage development environments and foster reproducibility.
  - `nix/python-env.nix` is the single source of truth for the Python version: shared by
    Docker builds and CI, so the interpreter is fully pinned and reproducible across environments.
  - `shell.nix` wraps `nix/python-env.nix` with `buildFHSEnv` for local development, which
    provides a CUDA-compatible FHS environment on both NixOS and non-NixOS machines.
  - Runtime and dev dependencies are managed by pip on top of the nix-provided interpreter.
    Pip is preferred over nix packages because pip can pull pre-built wheels (including GPU-specific
    CUDA wheels) from PyPI caches. Using nix for those would require local rebuilds due to
    the absence of binary caches for GPU packages.
- TODO: MLFLOW
- Delete dependabot updates PR for two reasons: (1) notification fatigue, (2) bumping package version 
can lead to different experiments results or breaking changes that needs to be tested. 
It is time consuming to be done properly, hence packages updates are done at my discretion.
The exception to this is dependabot security updates which can be activated through GitHub's interface
that warns me about potential vulnerabilites in my dependencies. 

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # GitHub Actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       ├── linting.yaml
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed/
│   └── raw/
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yaml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── nix/                      # Shared Nix environments
│   ├── ci-env.nix            # CI environment (Python + system libs for native extensions)
│   └── python-env.nix        # Pinned Python interpreter (used by Docker and CI)
├── npins/                    # Nix pin sources
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   └── mlops_sqldetect/
│       ├── __init__.py
│       ├── api.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── train.py
│       └── visualize.py
├── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── AGENTS.md                 # Guidance for coding agents
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Runtime dependencies (pinned)
├── shell.nix                 # Nix development environment
├── tasks.py                  # Invoke task definitions
└── treefmt.toml              # Formatter configuration
```

### Credits
This opinionated repository structure is based on [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
