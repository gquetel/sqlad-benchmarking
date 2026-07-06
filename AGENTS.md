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
  docker, dataset fetch). Refer to `tasks.py` for available tasks. Every Superviz26 build
  ships in one `superviz26.zip` on Zenodo record 21068333; `python -m tools.fetch_superviz26`
  downloads it once and extracts a named group (`main`/`drift`/`fsl`) to that group's loader
  root, verifying checksums against `data/raw/superviz26/MANIFEST.json`. `fetch-data` pulls
  the standard Superviz25 CSVs plus the `superviz26` `main` group
  (`~/datasets/superviz26-lodo/`); the `main` CSVs can alternatively be generated locally by
  the dataset generator's `generate_splits.py --full`. `superviz26-big` trains on an
  even-larger training set read from `~/datasets/superviz26-big/` (generated locally, not on
  Zenodo). The heavy concept-drift and few-shot CSVs are fetched opt-in via
  `invoke fetch-supplementary` / `python -m tools.fetch_superviz26 --groups drift,fsl`
  (excluded from `fetch-data` because they are several GB); the `superviz26-drift` and
  `superviz26-fsl` loaders also auto-download their group on first use when a file is missing
  from the default root.
  The train/eval/SLURM entry
  points are Typer CLIs run directly via `python -m mlops_sqldetect.<module>` (e.g. `train`,
  `evaluate`, `evaluate_suite`, `evaluate_drift`) or `python -m tools.<module>` (e.g.
  `slurm_submit`); use `--help` on any of them to see options.
  * `evaluate_suite` runs the standard train-once/evaluate-once grid (in-domain + LODO).
    `evaluate_drift` runs the same-domain concept-drift protocol (`superviz26-drift` family):
    it trains once per `(domain, pipeline, extractor)` cell and scores the origin (S1) and
    shifted (S2) test sets, writing `auroc_s1`/`auroc_s2`/`delta_auroc` rows. The two
    protocols are selected by a `DatasetFamily.protocol` field; `slurm_run_cell` dispatches
    on it, so `slurm_submit --dataset superviz26-drift` fans the drift grid out the same way.
* To parallelize the evaluation grid on a SLURM cluster, `tools.slurm_submit` fans each
  `(scenario, pipeline, extractor)` cell out as a job array (one array per resource class:
  `cpu`, and a `gpu` array per VRAM tier — each GPU cell runs on the partitions with enough
  VRAM for it, from `min_vram_gb` in the config, so e.g. CodeT5+ skips the 16 GB V100). Site
  settings live in `configs/slurm.yaml`; jobs
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
* Comments and docstrings must describe the **current** state of the code only. Never
  reference our conversation/review, the code change itself ("now", "no longer", "instead
  of X, which we removed"), or code that no longer exists. A reader with no history should
  not be able to tell the code was ever written differently.
  Exception: describing external realities that live code still handles — e.g. legacy data
  formats or older MLflow runs read by a fallback path — is present-tense behaviour and is fine.
* If the project has a `docs/` folder, update documentation there as needed.
* The project uses `mkdocs` for documentation. To build the docs locally: `mkdocs serve --config-file docs/mkdocs.yaml`.
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
