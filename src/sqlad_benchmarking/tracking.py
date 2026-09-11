"""MLflow experiment tracking setup.

Some specificity of this setup:
- My mlflow server requires mTLS, we provide our certificate using "MLFLOW_TRACKING_CLIENT_CERT_PATH".
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)
_cell_failure_logger = logging.getLogger(f"{__name__}.cell_failure")
_cell_failure_logger.propagate = False


_EXPERIMENT_NAMES = {
    "superviz25": "Superviz25-SQL",
    "superviz26-big": "Superviz26-SQL-Big",
    "superviz26-drift": "FE-study-CD",
    "superviz26-fsl": "FE-study-FSL",
}


def experiment_name(dataset: str) -> str:
    """Map a dataset family to its MLflow experiment name.

    The feature-extractor study owns three experiments: the default superviz26 (the
    full LODO/in-domain dataset) → "FE-study-LODO"; concept-drift → "FE-study-CD";
    few-shot → "FE-study-FSL". The older families keep their own experiments: the
    larger-train variant superviz26-big → "Superviz26-SQL-Big"; Superviz25 →
    "Superviz25-SQL".

    The previous SuperViz26 experiments ("Superviz26-SQL", "Drift-Superviz26-SQL",
    "FSL-Superviz26-SQL") still hold the earlier runs; new runs no longer land there.
    """
    return _EXPERIMENT_NAMES.get(dataset, "FE-study-LODO")


# Path to the importable sqlad_benchmarking package, bundled into pyfunc models
# via ``code_paths`` so the detector can be unpickled and scored anywhere.
# See https://mlflow.org/docs/latest/ml/model/models-from-code/#using-the-model
_PACKAGE_ROOT = Path(__file__).resolve().parent

# Model-from-code script: passed as a path so MLflow loads it instead of
# CloudPickling a DetectorModel instance.
_MODEL_SCRIPT = _PACKAGE_ROOT / "detector_model.py"


class _UrlRewriteStream:
    """Wraps a stream and rewrites a URL in every write() call.

    MLflow prints run/experiment links via sys.stdout.write(), so a logging
    filter cannot intercept them — a stream wrapper is needed instead.
    """

    def __init__(self, stream, from_url: str, to_url: str) -> None:
        self._stream = stream
        self._from = from_url
        self._to = to_url

    def write(self, text: str) -> int:
        return self._stream.write(text.replace(self._from, self._to))

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def setup_mlflow(dataset: str) -> bool:
    """Configure MLflow from the environment.

    The experiment is derived from ``dataset`` via :func:`experiment_name`: the
    SuperViz26 families go to the "FE-study-*" experiments, Superviz25 to
    "Superviz25-SQL".

    Returns:
        True if a tracking server is configured and tracking should proceed
        False to disable tracking (no ``MLFLOW_TRACKING_URI`` in the environment).
    """
    # This load the env variables defined in .env file.
    load_dotenv()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.info("MLFLOW_TRACKING_URI not set. Experiment tracking disabled.")
        return False

    # Making sure that home-relative paths in .env still resolve
    cert_path = os.environ.get("MLFLOW_TRACKING_CLIENT_CERT_PATH")
    if cert_path:
        os.environ["MLFLOW_TRACKING_CLIENT_CERT_PATH"] = os.path.expanduser(cert_path)
    server_cert_path = os.environ.get("MLFLOW_TRACKING_SERVER_CERT_PATH")
    if server_cert_path:
        os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"] = os.path.expanduser(server_cert_path)

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(experiment_name(dataset))
    except Exception as exc:
        logger.warning(f"MLflow tracking disabled: could not reach {tracking_uri} ({exc})")
        return False

    display_uri = os.environ.get("MLFLOW_DISPLAY_URI")
    if display_uri:
        sys.stdout = _UrlRewriteStream(sys.stdout, tracking_uri, display_uri)

    logger.info(f"MLflow tracking enabled: {tracking_uri}")
    return True


def log_and_register_detector(
    model_path: Path,
    registered_name: str,
    input_example: pd.DataFrame,
):
    """Log the fitted detector as a pyfunc model and register a new version.

    Registering the same ``registered_name`` again creates version 2, 3, ... The
    source run id is recorded automatically, so each version traces back to the
    run that produced it.

    Returns the :class:`~mlflow.models.model.ModelInfo` of the logged model.
    """
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=str(_MODEL_SCRIPT),
        artifacts={"detector": str(model_path)},
        input_example=input_example,
        code_paths=[str(_PACKAGE_ROOT)],
        registered_model_name=registered_name,
    )


class CellLog:
    """A per-cell log uploadable to the active MLflow run."""

    def __init__(self, path: Path, handler: logging.Handler) -> None:
        self.path = path
        self._handler = handler

    def exception(self, message: str) -> None:
        """Record the current exception without console propagation."""
        _cell_failure_logger.exception(message)

    def upload(self, artifact_path: str = "logs") -> None:
        """Upload the current log to the active run, if any."""
        if mlflow.active_run() is None:
            return
        try:
            self._handler.flush()
            mlflow.log_artifact(str(self.path), artifact_path=artifact_path)
        except Exception as exc:
            logger.warning(f"Could not upload log artifact {self.path.name}: {exc}")


def cell_log_path(dataset: str, method: str, extractor: str, scenario: str) -> Path:
    """Return an evaluation cell's log path."""
    return Path("reports") / dataset / "logs" / f"{method}_{extractor}_{dataset}_{scenario}.log"


@contextmanager
def capture_cell_log(log_dir: Path, stem: str) -> Iterator[CellLog]:
    """Capture log records in ``log_dir/<stem>.log``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stem}.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    _cell_failure_logger.addHandler(handler)
    try:
        yield CellLog(log_path, handler)
    finally:
        logging.getLogger().removeHandler(handler)
        _cell_failure_logger.removeHandler(handler)
        handler.close()


def find_parent_run_id(tags: dict[str, str]) -> str | None:
    """Return the id of an existing root run whose tags match ``tags``, or None.

    Searches the active experiment for a single run carrying every key/value in
    ``tags``. Parents are marked with ``run_role == "parent"`` (see the caller),
    which distinguishes them from their nested children, so reusing the returned
    id keeps every child of a given (method, extractor) accumulating under one
    stable, comparable parent across repeated suite invocations.
    """
    filter_string = " and ".join(f"tags.`{key}` = '{value}'" for key, value in tags.items())
    runs = mlflow.search_runs(filter_string=filter_string, max_results=1, output_format="list")
    return runs[0].info.run_id if runs else None


def ensure_parent_run(tags: dict[str, str], name: str) -> str:
    """Return the id of the parent run matching ``tags``, creating it if absent.

    The run is created and immediately ended so callers can reopen it by id and nest
    children under it. Pre-creating it once (e.g. before fanning a grid out to many
    concurrent SLURM jobs) avoids duplicate parents racing on find-or-create.
    """
    run_id = find_parent_run_id(tags)
    if run_id:
        return run_id
    with mlflow.start_run(run_name=name, tags=tags) as run:
        return run.info.run_id


def delete_running_cell_runs(tags: dict[str, str]) -> int:
    """Soft-delete running child runs in the active parent that match ``tags``.

    Args:
        tags: Tags identifying one evaluation cell.

    Returns:
        Number of runs deleted.
    """
    parent = mlflow.active_run()
    if parent is None:
        raise RuntimeError("an active parent run is required to replace a running cell")

    filters = ["attributes.status = 'RUNNING'", f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"]
    filters.extend(f"tags.`{key}` = '{value}'" for key, value in tags.items())
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=" and ".join(filters),
    )

    deleted = 0
    for run in runs:
        if client.get_run(run.info.run_id).info.status != "RUNNING":
            continue
        client.delete_run(run.info.run_id)
        deleted += 1
        logger.info(f"Deleted stale RUNNING MLflow run {run.info.run_id}")
    return deleted


def log_dataset_input(*, url: str, name: str, digest: str, context: str) -> None:
    """Record the source dataset on the active run as metadata only.

    Uses :class:`MetaDataset` so MLflow stores the dataset's origin (its Zenodo
    URL), name, and content ``digest`` without re-hashing here. The loader has
    already verified the on-disk CSV against this manifest ``digest``, so it
    faithfully identifies the bytes trained on (see :mod:`datasets.integrity`).
    """
    # mlflow does not accept digest > 36 chars
    dataset = MetaDataset(source=HTTPDatasetSource(url=url), name=name, digest=digest[:36])
    mlflow.log_input(dataset, context=context)


def _killed_cell_log(dataset: str, manifest: str, index: int) -> Path | None:
    """Return a manifest cell's log path if it exists."""
    if not manifest or index < 0:
        return None
    try:
        lines = [line for line in Path(manifest).read_text().splitlines() if line.strip()]
        cell = json.loads(lines[index])
        path = cell_log_path(dataset, cell["method"], cell["extractor"], cell["scenario"])
        return path if path.exists() else None
    except (OSError, ValueError, IndexError, KeyError, TypeError) as exc:
        logger.warning(f"Could not find the log of cell {index} of {manifest}: {exc}")
        return None


def fail_killed_run(dataset: str, status: int, manifest: str = "", index: int = -1) -> None:
    """Attach a killed cell's log and mark its MLflow run FAILED."""
    job_id = os.environ.get("SLURM_JOB_ID", "")
    try:
        if not setup_mlflow(dataset):
            return
        log_path = _killed_cell_log(dataset, manifest, index)
        filter_string = f"attributes.status = 'RUNNING' and tags.`slurm_job_id` = '{job_id}'"
        runs = mlflow.search_runs(filter_string=filter_string, output_format="list")
        client = MlflowClient()
        for run in runs:
            if log_path is not None:
                try:
                    client.log_artifact(run.info.run_id, str(log_path), artifact_path="logs")
                except Exception as exc:
                    logger.warning(f"Could not upload log artifact {log_path.name}: {exc}")
            client.set_terminated(run.info.run_id, status="FAILED")
            logger.info(f"Marked MLflow run {run.info.run_id} FAILED: task exited with status {status}")
    except Exception as exc:
        logger.warning(f"Could not close the MLflow run of SLURM job {job_id}: {exc}")
