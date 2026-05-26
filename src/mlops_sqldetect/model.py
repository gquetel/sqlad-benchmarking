"""Li SQL injection detection pipelines: OCSVM and AutoEncoder.

Both pipelines share :class:`LiExtractor` for feature extraction and produce a
scalar anomaly score per query (higher = more anomalous). They expose a uniform
``fit``/``score_samples``/``save``/``load`` interface so callers can swap them
without branching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch import nn

from mlops_sqldetect.features import LiExtractor

PipelineName = Literal["ocsvm", "ae"]


# ----- OCSVM -----------------------------------------------------------------


@dataclass
class OCSVMConfig:
    """Hyper-parameters for the One-Class SVM head."""

    nu: float = 0.05
    kernel: str = "rbf"
    gamma: str = "scale"
    max_iter: int = 1000


class LiOCSVM:
    """Li features → StandardScaler → OneClassSVM, exposed as a single object."""

    def __init__(self, config: OCSVMConfig | None = None) -> None:
        self.config = config or OCSVMConfig()
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("features", LiExtractor()),
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

    def fit(self, df: pd.DataFrame) -> "LiOCSVM":
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
    def load(cls, path: Path) -> "LiOCSVM":
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


class LiAutoEncoder:
    """Li features → StandardScaler → MLP autoencoder; score = reconstruction MSE."""

    def __init__(
        self,
        config: AEConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or AEConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = LiExtractor()
        self.scaler = StandardScaler()
        self.net: _AutoEncoderNet | None = None

    def _featurize(self, df: pd.DataFrame, *, fit_scaler: bool) -> torch.Tensor:
        x = self.extractor.transform(df)
        x = self.scaler.fit_transform(x) if fit_scaler else self.scaler.transform(x)
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)

    def fit(self, df: pd.DataFrame) -> "LiAutoEncoder":
        """Fit scaler + network on normal samples (caller's responsibility)."""
        torch.manual_seed(self.config.seed)
        x = self._featurize(df, fit_scaler=True)

        self.net = _AutoEncoderNet(
            input_dim=x.shape[1], hidden_dims=self.config.hidden_dims
        ).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(self.config.epochs):
            perm = torch.randperm(x.size(0), device=self.device)
            for start in range(0, x.size(0), self.config.batch_size):
                idx = perm[start : start + self.config.batch_size]
                batch = x[idx]
                optimizer.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                optimizer.step()
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
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> "LiAutoEncoder":
        # weights_only=False is required because the checkpoint stores the fitted
        # sklearn StandardScaler and the AEConfig dataclass alongside the weights.
        bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(config=bundle["config"], device=device)
        obj.scaler = bundle["scaler"]
        obj.net = _AutoEncoderNet(
            input_dim=bundle["input_dim"], hidden_dims=obj.config.hidden_dims
        ).to(obj.device)
        obj.net.load_state_dict(bundle["state_dict"])
        obj.net.eval()
        return obj


# ----- Factory ---------------------------------------------------------------


def build_pipeline(name: PipelineName) -> LiOCSVM | LiAutoEncoder:
    """Return a fresh, unfitted Li pipeline by short name (``ocsvm`` or ``ae``)."""
    if name == "ocsvm":
        return LiOCSVM()
    if name == "ae":
        return LiAutoEncoder()
    raise ValueError(f"Unknown pipeline: {name!r} (expected 'ocsvm' or 'ae')")


def load_pipeline(name: PipelineName, path: Path) -> LiOCSVM | LiAutoEncoder:
    """Load a previously-saved pipeline by short name."""
    if name == "ocsvm":
        return LiOCSVM.load(path)
    if name == "ae":
        return LiAutoEncoder.load(path)
    raise ValueError(f"Unknown pipeline: {name!r} (expected 'ocsvm' or 'ae')")
