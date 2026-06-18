> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

By default, you don't have access to any python environment or any other tools. When you need something, use `nix-shell`.

# Reproducibility 

This repo uses Nix for reproducible environments and [uv](https://docs.astral.sh/uv/) for
Python packages. **Never run unversioned upgrade commands** such as:
- `uv lock --upgrade` / `uv lock --upgrade-package <pkg>`
- `uv add <package>` without an exact version pin

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`; `requirements.txt` is a
pip-compatible export of the lock (do not hand-edit it). Nix provides the interpreter and uv is
configured to never download its own Python (`UV_PYTHON_DOWNLOADS=never`). Do not change pinned
versions or regenerate the lock without explicit instruction.

# Relevant commands

* The project uses `uv` for Python package management on top of the Nix-provided interpreter.
  * To sync the environment from the lock: `uv sync --frozen`.
  * To add a package: `uv add <package>==<exact-version>` (then commit `uv.lock` + `requirements.txt`).
  * To regenerate the lock and the `requirements.txt` export: `invoke lock`.
  * To run a command in the project env: `uv run <command>` (e.g. `uv run python <script>.py`).
* The project uses `pytest` for testing: `pytest tests/`.
* The project uses `treefmt` + `ruff` for formatting and linting:
    * To format code: `treefmt`.
    * To check formatting without writing: `treefmt --fail-on-change`.
* The project uses `invoke` for setup/orchestration tasks (e.g. `sync`, `lock`, `test`, docs,
  docker, dataset fetch). Refer to `tasks.py` for available tasks. The train/eval/SLURM entry
  points are Typer CLIs run directly via `python -m mlops_sqldetect.<module>` (e.g. `train`,
  `evaluate`, `evaluate_suite`) or `python -m tools.<module>` (e.g. `slurm_submit`); use `--help`
  on any of them to see options.
* To parallelize the evaluation grid on a SLURM cluster, `tools.slurm_submit` fans each
  `(scenario, pipeline, extractor)` cell out as a job array (one array per resource class:
  `cpu`, `gpu` for `ae`/`sbert` cells on a partition list like `A100,V100`, and an A100-only
  array for cells matched by `force_a100`). Site settings live in `configs/slurm.yaml`; jobs
  activate the uv `.venv` (built once on the login node). Each cell writes its row to `reports/{dataset}/cells/*.csv`;
  MLflow is the canonical store.
    * Preview: `python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ae --dry-run`.
    * Submit: `python -m tools.slurm_submit --dataset superviz26 --suite all --pipelines ocsvm,ae --extractors li`.

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
