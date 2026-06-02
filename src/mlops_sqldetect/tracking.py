"""MLflow experiment tracking setup.

Some specificity of this setup:
- My mlflow server requires mTLS, we provide our certificate using "MLFLOW_TRACKING_CLIENT_CERT_PATH".
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "mlops-sqldetect"

# Path to the importable mlops_sqldetect package, bundled into pyfunc models
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


def setup_mlflow() -> bool:
    """Configure MLflow from the environment.

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

    # Making sure that home-relative path in .env still resolves
    cert_path = os.environ.get("MLFLOW_TRACKING_CLIENT_CERT_PATH")
    if cert_path:
        os.environ["MLFLOW_TRACKING_CLIENT_CERT_PATH"] = os.path.expanduser(cert_path)

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT)
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


def log_dataset_input(*, url: str, name: str, digest: str, context: str) -> None:
    """Record the source dataset on the active run as metadata only.

    Uses :class:`MetaDataset` so MLflow stores the dataset's origin (its Zenodo
    URL), name, and content ``digest`` without downloading or hashing the file.
    """
    # mlflow does not accept digest > 36 chars
    dataset = MetaDataset(source=HTTPDatasetSource(url=url), name=name, digest=digest[:36])
    mlflow.log_input(dataset, context=context)
