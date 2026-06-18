import os
import sys

from invoke import Context, task

# Disable pty when it is unavailable or unsupported: on Windows, and on Python
# 3.14+ where invoke's pty handling is broken (this project requires 3.14+).
NO_PTY = os.name == "nt" or sys.version_info >= (3, 14)
PROJECT_NAME = "mlops_sqldetect"


# Setup commands
@task
def sync(ctx: Context) -> None:
    """Create/refresh the .venv from uv.lock (deps + dev group + project)."""
    ctx.run("uv sync --frozen", echo=True, pty=not NO_PTY)


@task
def lock(ctx: Context) -> None:
    """Regenerate uv.lock and the pip-compatible requirements.txt export."""
    ctx.run("uv lock", echo=True, pty=not NO_PTY)
    ctx.run("uv export --no-dev --no-emit-project -o requirements.txt", echo=True, pty=not NO_PTY)


# Project commands
@task(help={"datasets": "Comma-separated Superviz26 short names (e.g. 'a-a,bcd-a'); empty for all"})
def fetch_superviz26(ctx: Context, datasets: str = "", force: bool = False, check: bool = False) -> None:
    """Download Superviz26-SQL CSVs from Zenodo with checksum verification."""
    cmd = "python -m tools.fetch_superviz26"
    if datasets:
        cmd += f" --datasets {datasets}"
    if force:
        cmd += " --force"
    if check:
        cmd += " --check"
    ctx.run(cmd, echo=True, pty=not NO_PTY)


@task
def fetch_superviz25(ctx: Context, force: bool = False, check: bool = False) -> None:
    """Download the Superviz25-SQL dataset from Zenodo with checksum verification."""
    cmd = "python -m tools.fetch_superviz25"
    if force:
        cmd += " --force"
    if check:
        cmd += " --check"
    ctx.run(cmd, echo=True, pty=not NO_PTY)


@task
def fetch_data(ctx: Context, force: bool = False, check: bool = False) -> None:
    """Download every dataset (Superviz25 + Superviz26) from Zenodo."""
    fetch_superviz26(ctx, force=force, check=check)
    fetch_superviz25(ctx, force=force, check=check)


@task(
    help={
        "limit": "Total stratified samples per split (default 1000).",
        "no_track": "Disable MLflow tracking.",
        "register": "Register fitted models in the MLflow Model Registry.",
    }
)
def smoke(ctx: Context, limit: int = 1000, no_track: bool = False, register: bool = False) -> None:
    """Fast end-to-end smoke run on a small subset (in-domain, OCSVM only)."""
    cmd = (
        f"python -m {PROJECT_NAME}.evaluate_suite --suite in_domain --pipelines ocsvm "
        f"--limit {limit} --report reports/smoke.csv"
    )
    if no_track:
        cmd += " --no-track"
    if register:
        cmd += " --register"
    ctx.run(cmd, echo=True, pty=not NO_PTY)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not NO_PTY)
    ctx.run("coverage report -m -i", echo=True, pty=not NO_PTY)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not NO_PTY,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not NO_PTY
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not NO_PTY)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not NO_PTY)
