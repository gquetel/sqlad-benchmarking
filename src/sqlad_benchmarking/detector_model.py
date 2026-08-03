"""Model-from-code definition for the pyfunc detector.

This module is loaded by MLflow via a file path (not pickled), avoiding the
CloudPickle serialization of a Python object. The trailing ``set_model`` call
designates the instance MLflow should use.
See https://mlflow.org/docs/latest/ml/model/models-from-code/
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd


class DetectorModel(mlflow.pyfunc.PythonModel):
    """Pyfunc adapter exposing any detector's anomaly score through ``predict``.

    This wrapper allows to have model artifacts as real MLflow models of a raw pickle.
    """

    def load_context(self, context) -> None:
        from sqlad_benchmarking.model import load_method

        path = Path(context.artifacts["detector"])
        # Infer the decision engine from the file suffix. ``.pt`` is the
        # AutoEncoder; ``.joblib`` is a scikit-learn pipeline head (OCSVM or LOF),
        # both of which load and score identically, so "ocsvm" covers either.
        name = "ae" if path.suffix == ".pt" else "ocsvm"
        self.detector = load_method(name, path)

    def predict(self, context, model_input: pd.DataFrame):
        return self.detector.score_samples(model_input)


mlflow.models.set_model(DetectorModel())
