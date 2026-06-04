> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

By default, you don't have access to any python environment or any other tools. When you need something, use `nix-shell`.

# Reproducibility 

This repo uses Nix for reproducible environments. **Never run unversioned upgrade commands** such as:
- `pip install --upgrade pip`
- `pip install -U <package>`
- `pip install <package>` without an exact version pin

Every `pip install` must use an exact version (`==`). Python tooling versions are pinned in `shell.nix`. Dependency versions are pinned in `requirements.txt`. Do not change pinned versions without explicit instruction.

# Relevant commands

* The project uses `pip` for Python package installation inside the Nix-managed venv.
  * To install a package: `pip install <package>==<exact-version>`.
  * To run Python scripts: `python <script-name>.py`.
* The project uses `pytest` for testing: `pytest tests/`.
* The project uses `treefmt` + `ruff` for formatting and linting:
    * To format code: `treefmt`.
    * To check formatting without writing: `treefmt --fail-on-change`.
* The project uses `invoke` for task management. Refer to `tasks.py` for available tasks.
* To parallelize the evaluation grid on a SLURM cluster, `tools.slurm_submit` fans each
  `(scenario, pipeline, extractor)` cell out as a job array (one array per resource class:
  `cpu`, `gpu` for `ae`/`sbert` cells on a partition list like `A100,V100`, and an A100-only
  array for cells matched by `force_a100`). Site settings live in `configs/slurm.yaml`; jobs
  activate the env via `conda`. Each cell writes its row to `reports/{dataset}/cells/*.csv`;
  MLflow is the canonical store.
    * Preview: `python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ae --dry-run`.
    * Submit: `invoke slurm-suite --dataset superviz26 --suite all --pipelines ocsvm,ae`.

# Code style
* DO NOT ADD EXCEPTIONS TO RUFF BY YOURSELF. ASK ME FIRST. 
* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints.
* Do not add inline comments unless absolutely necessary.

# Experiment tracking

* `evaluate_suite` logs params, metrics, the per-epoch AE training loss, and the fitted model artifact to MLflow when `MLFLOW_TRACKING_URI` is set (opt out with `--no-track`). Configuration is environment-driven via `.env` (see  `.env.example`).

# Documentation

* Use inline comments to document the **why** and not the **what**. Keep them short.
* If the project has a `docs/` folder, update documentation there as needed.
* The project uses `mkdocs` for documentation. To build the docs locally: `mkdocs serve --config-file docs/mkdocs.yaml`.
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
