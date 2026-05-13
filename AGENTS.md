> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

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
* The project uses `treefmt` + `black` for formatting:
    * To format code: `treefmt`.
    * To check formatting without writing: `treefmt --fail-on-change`.
* The project uses `invoke` for task management. Refer to `tasks.py` for available tasks.

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints.
* Do not add inline comments unless absolutely necessary.

# Documentation

* Use inline comments to document the **why** and not the **what**.
* If the project has a `docs/` folder, update documentation there as needed.
* The project uses `mkdocs` for documentation. To build the docs locally: `mkdocs serve --config-file docs/mkdocs.yaml`.
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
