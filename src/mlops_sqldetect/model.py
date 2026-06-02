"""SQL injection detection pipelines: OCSVM and AutoEncoder.

Each pipeline pairs a *feature extractor* with a one-class decision head and produces
a scalar anomaly score per query (higher = more anomalous). The extractor is
injected, not hardcoded, so any extractor can be combined with any head. Both
expose a uniform ``fit``/``score_samples``/``save``/``load`` interface so callers
can swap them without branching.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch import nn

from mlops_sqldetect.features import DEFAULT_EXTRACTOR, build_extractor

PipelineName = Literal["ocsvm", "ae"]


# ----- OCSVM -----------------------------------------------------------------


@dataclass
class OCSVMConfig:
    """Hyper-parameters for the One-Class SVM head."""

    nu: float = 0.05
    kernel: str = "rbf"
    gamma: str = "scale"
    max_iter: int = 1000


class OCSVMDetector:
    """features → StandardScaler → OneClassSVM, exposed as a single object.

    The feature extractor is the first pipeline step and is pickled with the rest
    of the pipeline, so it travels with the saved artifact and is restored on
    :meth:`load` without the caller needing to name it again.
    """

    def __init__(
        self,
        config: OCSVMConfig | None = None,
        extractor: TransformerMixin | None = None,
    ) -> None:
        self.config = config or OCSVMConfig()
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("features", extractor if extractor is not None else build_extractor()),
                ("scaler", StandardScaler()),
                (
                    "ocsvm",
                    OneClassSVM(
                        nu=self.config.nu,
                        kernel=self.config.kernel,
                        gamma=self.config.gamma,
                        max_iter=self.config.max_iter,
                    ),
                ),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "OCSVMDetector":
        """Fit on normal samples only (caller's responsibility)."""
        self.pipeline.fit(df)
        return self

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        return -self.pipeline.decision_function(df)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "OCSVMDetector":
        obj = cls()
        obj.pipeline = joblib.load(Path(path))
        return obj


# ----- AutoEncoder -----------------------------------------------------------


@dataclass
class AEConfig:
    """Hyper-parameters for the AutoEncoder head."""

    learning_rate: float = 5e-3
    epochs: int = 100
    batch_size: int = 8192
    hidden_dims: tuple[int, ...] = (16, 8)
    seed: int = 7


class _AutoEncoderNet(nn.Module):
    """Symmetric MLP autoencoder with a sigmoid output for [0, 1]-scaled inputs."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev = hidden_dims[-1]
        for h in reversed(hidden_dims[:-1]):
            decoder_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        decoder_layers.extend([nn.Linear(prev, input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder_layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AEDetector:
    """features → StandardScaler → MLP autoencoder; score = reconstruction MSE.

    The feature extractor is injected and persisted in the checkpoint alongside
    the scaler and weights, so :meth:`load` restores the exact extractor that was
    used at training time regardless of which one it was.
    """

    def __init__(
        self,
        config: AEConfig | None = None,
        extractor: TransformerMixin | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or AEConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = extractor if extractor is not None else build_extractor()
        self.scaler = StandardScaler()
        self.net: _AutoEncoderNet | None = None

    def _featurize(self, df: pd.DataFrame, *, fit_scaler: bool) -> torch.Tensor:
        x = self.extractor.transform(df)
        x = self.scaler.fit_transform(x) if fit_scaler else self.scaler.transform(x)
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)

    def fit(
        self,
        df: pd.DataFrame,
        epoch_callback: Callable[[int, float], None] | None = None,
    ) -> "AEDetector":
        """Fit scaler + network on normal samples (caller's responsibility).

        Args:
            df: Normal-class training samples.
            epoch_callback: Optional ``(epoch, mean_loss)`` hook called once per
                epoch, e.g. to log a training-loss curve to MLflow.
        """
        torch.manual_seed(self.config.seed)
        x = self._featurize(df, fit_scaler=True)

        self.net = _AutoEncoderNet(input_dim=x.shape[1], hidden_dims=self.config.hidden_dims).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        self.net.train()
        for epoch in range(self.config.epochs):
            perm = torch.randperm(x.size(0), device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, x.size(0), self.config.batch_size):
                idx = perm[start : start + self.config.batch_size]
                batch = x[idx]
                optimizer.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if epoch_callback is not None:
                epoch_callback(epoch, epoch_loss / n_batches)
        self.net.eval()
        return self

    @torch.no_grad()
    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model has not been fitted")
        x = self._featurize(df, fit_scaler=False)
        recon = self.net(x)
        # Per-sample MSE — higher means worse reconstruction, i.e. more anomalous.
        return torch.mean((recon - x) ** 2, dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        if self.net is None:
            raise RuntimeError("Model has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "input_dim": self.net.input_dim,
                "config": self.config,
                "scaler": self.scaler,
                "extractor": self.extractor,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> "AEDetector":
        # weights_only=False is required because the checkpoint stores the fitted
        # sklearn StandardScaler, the feature extractor, and the AEConfig
        # dataclass alongside the weights.
        bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(config=bundle["config"], extractor=bundle.get("extractor"), device=device)
        obj.scaler = bundle["scaler"]
        obj.net = _AutoEncoderNet(input_dim=bundle["input_dim"], hidden_dims=obj.config.hidden_dims).to(obj.device)
        obj.net.load_state_dict(bundle["state_dict"])
        obj.net.eval()
        return obj


# ----- Factory ---------------------------------------------------------------


Detector = OCSVMDetector | AEDetector


def build_pipeline(name: PipelineName, extractor: str = DEFAULT_EXTRACTOR) -> Detector:
    """Return a fresh, unfitted pipeline by decision-head and extractor short name.

    Args:
        name: Decision head, ``"ocsvm"`` or ``"ae"``.
        extractor: Feature extractor short name (see :data:`EXTRACTORS`).
    """
    extractor_instance = build_extractor(extractor)
    if name == "ocsvm":
        return OCSVMDetector(extractor=extractor_instance)
    if name == "ae":
        return AEDetector(extractor=extractor_instance)
    raise ValueError(f"Unknown pipeline: {name!r} (expected 'ocsvm' or 'ae')")


def load_pipeline(name: PipelineName, path: Path) -> Detector:
    """Load a previously-saved pipeline by decision-head short name.

    The feature extractor travels with the saved artifact, so it does not need to
    be named again here.
    """
    if name == "ocsvm":
        return OCSVMDetector.load(path)
    if name == "ae":
        return AEDetector.load(path)
    raise ValueError(f"Unknown pipeline: {name!r} (expected 'ocsvm' or 'ae')")
