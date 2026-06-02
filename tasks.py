import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_sqldetect"


# Setup commands
@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task(help={"dataset": "CSV dataset path", "pipeline": "ocsvm or ae", "output": "Output model path"})
def train(
    ctx: Context,
    dataset: str = "data/raw/dataset.csv",
    pipeline: str = "ocsvm",
    output: str = "models/li.joblib",
) -> None:
    """Train a Li SQL injection detection pipeline."""
    ctx.run(
        f"python -m {PROJECT_NAME}.train {dataset} --output {output} --pipeline {pipeline}",
        echo=True,
        pty=not WINDOWS,
    )


@task(help={"datasets": "Comma-separated short names (e.g. 'a-a,bcd-a'); empty for all"})
def fetch_data(ctx: Context, datasets: str = "", force: bool = False, check: bool = False) -> None:
    """Download Superviz26-SQL CSVs from Zenodo with MD5 verification."""
    cmd = "python -m tools.fetch_superviz26"
    if datasets:
        cmd += f" --datasets {datasets}"
    if force:
        cmd += " --force"
    if check:
        cmd += " --check"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task(help={"suite": "in_domain, lodo, or all", "pipelines": "Comma-separated pipeline names"})
def evaluate_suite(
    ctx: Context,
    suite: str = "all",
    pipelines: str = "ocsvm,ae",
) -> None:
    """Run the Superviz26 in-domain + LODO evaluation grid."""
    ctx.run(
        f"python -m {PROJECT_NAME}.evaluate_suite --suite {suite} --pipelines {pipelines}",
        echo=True,
        pty=not WINDOWS,
    )


@task(help={"dataset": "CSV dataset path", "model_path": "Saved model path", "pipeline": "ocsvm or ae"})
def evaluate(
    ctx: Context,
    dataset: str,
    model_path: str,
    pipeline: str = "ocsvm",
) -> None:
    """Evaluate a Li SQL injection detection pipeline."""
    ctx.run(
        f"python -m {PROJECT_NAME}.evaluate {dataset} {model_path} --pipeline {pipeline}",
        echo=True,
        pty=not WINDOWS,
    )


@task(help={"limit": "Total stratified samples per split (default 1000).", "no_track": "Disable MLflow tracking."})
def smoke(ctx: Context, limit: int = 1000, no_track: bool = False) -> None:
    """Fast end-to-end smoke run on a small subset (in-domain, OCSVM only)."""
    cmd = (
        f"python -m {PROJECT_NAME}.evaluate_suite --suite in_domain --pipelines ocsvm "
        f"--limit {limit} --report reports/smoke.csv"
    )
    if no_track:
        cmd += " --no-track"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
