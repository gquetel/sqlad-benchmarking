"""SQL injection detection methods: OCSVM and AutoEncoder.

Each method pairs a *feature extractor* with a one-class decision head and produces
a scalar anomaly score per query (higher = more anomalous). The extractor is
injected, not hardcoded, so any extractor can be combined with any head. Both
expose a uniform ``fit``/``score_samples``/``save``/``load`` interface so callers
can swap them without branching.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from sklearn.base import TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, StandardScaler
from sklearn.svm import OneClassSVM
from torch import nn

from sqlad_benchmarking.determinism import enable_determinism
from sqlad_benchmarking.features import DEFAULT_EXTRACTOR, build_extractor
from sqlad_benchmarking.features.cache import maybe_wrap, resolve_cache_dir

logger = logging.getLogger(__name__)

MethodName = Literal["ocsvm", "lof", "ae"]

# Human-readable labels for logs and MLflow run names; acronyms stay uppercase.
METHOD_LABELS: dict[str, str] = {"ocsvm": "OCSVM", "lof": "LOF", "ae": "Autoencoder"}


def _scaler_for(extractor: str) -> TransformerMixin:
    """Pick a scaler compatible with the extractor's output.

    cv/sbert/codet5 stay unscaled (cv is sparse; the embeddings are already bounded).
    Li and every GAUR mode use StandardScaler, matching gaur-sql-detect.
    """
    return FunctionTransformer() if extractor in ("cv", "sbert", "codet5") else StandardScaler()


# ----- OCSVM -----------------------------------------------------------------


@dataclass
class OCSVMConfig:
    """Hyper-parameters for the One-Class SVM head."""

    nu: float = 0.05
    kernel: str = "rbf"
    gamma: str = "scale"
    max_iter: int = 1000
    # libsvm's decision_function is single-threaded; scoring is parallelised by splitting
    # the test rows across this many processes (-1 = all cores). See score_samples.
    n_jobs: int = -1


# Below this many rows, pickling the fitted SVM to each worker outweighs the parallel
# speedup, so score in the calling process instead.
_MIN_ROWS_FOR_PARALLEL_SCORE = 10_000


# ``X``: sklearn feature-matrix convention (kept uppercase).
def _parallel_decision_function(estimator, X, n_jobs: int) -> np.ndarray:  # noqa: N803
    """``estimator.decision_function(X)`` split row-wise across processes.

    libsvm's ``decision_function`` is single-threaded and holds the GIL, so the only way
    to use more than one core is to score disjoint contiguous row blocks in separate
    processes and concatenate; the result is identical to the serial call. Works for
    dense and sparse ``X`` (both support contiguous row slicing).
    """
    workers = joblib.effective_n_jobs(n_jobs)
    if workers == 1 or X.shape[0] < _MIN_ROWS_FOR_PARALLEL_SCORE:
        return estimator.decision_function(X)
    edges = np.linspace(0, X.shape[0], workers + 1, dtype=int)
    blocks = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]
    parts = joblib.Parallel(n_jobs=workers)(joblib.delayed(estimator.decision_function)(X[b]) for b in blocks)
    return np.concatenate(parts)


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
        scaler: TransformerMixin | None = None,
    ) -> None:
        self.config = config or OCSVMConfig()
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("features", extractor if extractor is not None else build_extractor()),
                ("scaler", scaler if scaler is not None else StandardScaler()),
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
        """Return anomaly scores (higher = more anomalous).

        Feature extraction (SecureBERT/CodeT5+ on GPU, disk-cached) and scaling run once,
        then the single-threaded OneClassSVM decision is parallelised over the test rows.
        """
        features = self.pipeline[:-1].transform(df)
        return -_parallel_decision_function(self.pipeline[-1], features, self.config.n_jobs)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "OCSVMDetector":
        obj = cls()
        obj.pipeline = joblib.load(Path(path))
        return obj


# ----- LOF -------------------------------------------------------------------


@dataclass
class LOFConfig:
    """Hyper-parameters for the Local Outlier Factor head."""

    n_neighbors: int = 20
    n_jobs: int = -1


class LOFDetector:
    """features → scaler → LocalOutlierFactor(novelty=True), as a single object.

    ``novelty=True`` is required so the model can be fitted on normal samples and
    then score unseen queries via ``decision_function``. Mirrors
    :class:`OCSVMDetector`: the extractor is pickled with the pipeline and restored
    on :meth:`load`.
    """

    def __init__(
        self,
        config: LOFConfig | None = None,
        extractor: TransformerMixin | None = None,
        scaler: TransformerMixin | None = None,
    ) -> None:
        self.config = config or LOFConfig()
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("features", extractor if extractor is not None else build_extractor()),
                ("scaler", scaler if scaler is not None else StandardScaler()),
                (
                    "lof",
                    LocalOutlierFactor(
                        n_neighbors=self.config.n_neighbors,
                        novelty=True,
                        n_jobs=self.config.n_jobs,
                    ),
                ),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "LOFDetector":
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
    def load(cls, path: Path) -> "LOFDetector":
        obj = cls()
        obj.pipeline = joblib.load(Path(path))
        return obj


# ----- AutoEncoder -----------------------------------------------------------

# Caps the CountVectorizer vocabulary feeding the AE: its input width equals the
# feature count and the matrix is densified, so an unbounded vocabulary would blow
# up the network size and memory. Other heads keep the full vocabulary.
AE_CV_MAX_FEATURES = 20000


@dataclass
class AEConfig:
    """Hyper-parameters for the AutoEncoder head."""

    learning_rate: float = 5e-3
    epochs: int = 100
    batch_size: int = 8192
    seed: int = 7


def _hidden_dims(input_dim: int) -> tuple[int, int]:
    """Two-layer bottleneck derived from the input width.

    Returns ``(int(0.67 * X), int(0.33 * X))`` where ``X`` is the number of input
    features, so the topology depends only on the feature count and not on which
    extractor produced the matrix.
    """
    return (int(0.67 * input_dim), int(0.33 * input_dim))


OutputActivation = Literal["sigmoid", "relu", "tanh"]

_FINAL_ACTIVATIONS: dict[OutputActivation, type[nn.Module]] = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class _AutoEncoderNet(nn.Module):
    """Symmetric MLP autoencoder; the output activation matches the input range.

    The two hidden widths are always derived from ``input_dim`` via
    :func:`_hidden_dims`, so the topology depends only on the feature count. The
    final activation is ``sigmoid`` for [0, 1]-scaled inputs, ``relu`` for raw
    non-negative inputs (e.g. unscaled word counts), and ``tanh`` for [-1, 1]
    inputs (e.g. SecureBERT embeddings), so the reconstruction can span the
    actual target range.
    """

    def __init__(self, input_dim: int, output_activation: OutputActivation = "sigmoid") -> None:
        super().__init__()
        hidden_dims = _hidden_dims(input_dim)
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
        final_activation = _FINAL_ACTIVATIONS[output_activation]()
        decoder_layers.extend([nn.Linear(prev, input_dim), final_activation])
        self.decoder = nn.Sequential(*decoder_layers)
        self.input_dim = input_dim
        self.output_activation = output_activation

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
        scaler: TransformerMixin | None = None,
        device: torch.device | None = None,
        output_activation: OutputActivation = "sigmoid",
    ) -> None:
        self.config = config or AEConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("AEDetector using device: %s", self.device)
        self.extractor = extractor if extractor is not None else build_extractor()
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.output_activation = output_activation
        self.net: _AutoEncoderNet | None = None

    def _featurize(self, df: pd.DataFrame, *, fit: bool):
        """Return the scaled feature matrix (dense ndarray or sparse), not on device.

        When ``fit`` is set, the (possibly stateful) extractor and the scaler are
        fitted first. Keeping the matrix sparse here lets ``fit``/``score_samples``
        densify one batch at a time, so a wide CountVectorizer matrix never
        materialises in full.
        """
        if fit:
            self.extractor.fit(df)
        x = self.extractor.transform(df)
        return self.scaler.fit_transform(x) if fit else self.scaler.transform(x)

    def _batch_tensor(self, rows) -> torch.Tensor:
        """Densify a row slice (if sparse) and move it to the compute device."""
        arr = rows.toarray() if issparse(rows) else np.asarray(rows)
        return torch.from_numpy(arr.astype(np.float32)).to(self.device)

    def _train(
        self,
        x,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        epoch_callback: Callable[[int, float], None] | None,
    ) -> None:
        """Run ``epochs`` of MSE reconstruction training over the feature matrix ``x``.

        Shared by :meth:`fit` (fresh network) and :meth:`fine_tune` (pretrained
        network at a reduced learning rate); the caller owns network creation and the
        optimizer so this only drives the batch loop.
        """
        if self.net is None:
            raise RuntimeError("network must be created before training")
        n_samples = x.shape[0]
        loss_fn = nn.MSELoss()
        self.net.train()
        for epoch in range(epochs):
            # Sequential (un-shuffled) batching: batches follow the row order of x.
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, self.config.batch_size):
                batch = self._batch_tensor(x[start : start + self.config.batch_size])
                optimizer.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if epoch_callback is not None and n_batches:
                epoch_callback(epoch, epoch_loss / n_batches)
        self.net.eval()

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
        enable_determinism()
        torch.manual_seed(self.config.seed)
        x = self._featurize(df, fit=True)
        input_dim = x.shape[1]

        self.net = _AutoEncoderNet(input_dim=input_dim, output_activation=self.output_activation).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        self._train(x, optimizer, self.config.epochs, epoch_callback)
        return self

    def fine_tune(
        self,
        df: pd.DataFrame,
        *,
        learning_rate: float,
        epochs: int | None = None,
        seed: int | None = None,
        epoch_callback: Callable[[int, float], None] | None = None,
    ) -> "AEDetector":
        """Continue training the pretrained network on ``df`` with a frozen front-end.

        Used by the few-shot adaptation protocol: the autoencoder keeps the weights it
        was loaded with and only they are updated, while the already-fitted feature
        extractor and scaler are reused unchanged (``df`` is featurised with
        ``fit=False``). The caller passes the reduced ``learning_rate`` (the
        pretraining rate divided by ten) and the per-repetition ``seed``; every other
        setting follows the loaded :class:`AEConfig`.

        Args:
            df: Benign target-domain samples to adapt on.
            learning_rate: Optimizer learning rate for the adaptation phase.
            epochs: Number of epochs (defaults to the config's ``epochs``).
            seed: Torch seed for this run (defaults to the config's ``seed``).
            epoch_callback: Optional ``(epoch, mean_loss)`` hook called once per epoch.

        Raises:
            RuntimeError: If no network has been fitted or loaded yet.
        """
        if self.net is None:
            raise RuntimeError("fine_tune requires a fitted or loaded model")
        enable_determinism()
        torch.manual_seed(self.config.seed if seed is None else seed)
        x = self._featurize(df, fit=False)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self._train(x, optimizer, self.config.epochs if epochs is None else epochs, epoch_callback)
        return self

    @torch.no_grad()
    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model has not been fitted")
        x = self._featurize(df, fit=False)
        n_samples = x.shape[0]
        scores = np.empty(n_samples, dtype=np.float32)
        for start in range(0, n_samples, self.config.batch_size):
            batch = self._batch_tensor(x[start : start + self.config.batch_size])
            recon = self.net(batch)
            # Per-sample MSE — higher means worse reconstruction, i.e. more anomalous.
            s = torch.mean((recon - batch) ** 2, dim=1).cpu().numpy()
            scores[start : start + len(s)] = s
        return scores

    def save(self, path: Path) -> None:
        if self.net is None:
            raise RuntimeError("Model has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "input_dim": self.net.input_dim,
                "output_activation": self.net.output_activation,
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
        output_activation = bundle.get("output_activation", "sigmoid")
        obj = cls(
            config=bundle["config"],
            extractor=bundle.get("extractor"),
            device=device,
            output_activation=output_activation,
        )
        obj.scaler = bundle["scaler"]
        obj.net = _AutoEncoderNet(input_dim=bundle["input_dim"], output_activation=output_activation).to(obj.device)
        obj.net.load_state_dict(bundle["state_dict"])
        obj.net.eval()
        return obj


# ----- Factory ---------------------------------------------------------------


Detector = OCSVMDetector | LOFDetector | AEDetector


def build_method(
    name: MethodName,
    extractor: str = DEFAULT_EXTRACTOR,
    cache: bool = True,
    cache_dir: Path | None = None,
) -> Detector:
    """Return a fresh, unfitted method by decision-head and extractor short name.

    Args:
        name: Decision head, ``"ocsvm"``, ``"lof"`` or ``"ae"``.
        extractor: Feature extractor short name (see :data:`EXTRACTORS`).
        cache: Memoise the extractor's transform output to disk (see
            :func:`~sqlad_benchmarking.features.cache.resolve_cache_dir`).
        cache_dir: Override the cache directory; ignored when ``cache`` is False.
    """
    # Build the raw extractor first so the cv-AE branch can set_params on it,
    # then wrap with the cache once any extractor-specific tuning is applied.
    extractor_instance = build_extractor(extractor)
    cdir = resolve_cache_dir(cache, cache_dir)
    if name == "ocsvm":
        # cv/sbert/codet5/gaur-* need a higher QP iteration budget to converge
        # (wider spread than Li); Li keeps the default.
        config = (
            OCSVMConfig(max_iter=10000)
            if extractor in ("cv", "sbert", "codet5") or extractor.startswith("gaur-")
            else OCSVMConfig()
        )
        return OCSVMDetector(
            config=config, extractor=maybe_wrap(extractor_instance, cdir), scaler=_scaler_for(extractor)
        )
    if name == "lof":
        return LOFDetector(extractor=maybe_wrap(extractor_instance, cdir), scaler=_scaler_for(extractor))
    if name == "ae":
        if extractor == "cv":
            # cv AE trains directly on raw word counts: no scaler (an identity
            # transform) and a ReLU output to reconstruct the non-negative,
            # unbounded counts. The vocabulary is capped because the AE densifies
            # its input and its width equals the feature count.
            extractor_instance.set_params(max_features=AE_CV_MAX_FEATURES)
            return AEDetector(
                config=AEConfig(learning_rate=1e-3, epochs=100, batch_size=4096),
                extractor=maybe_wrap(extractor_instance, cdir),
                scaler=FunctionTransformer(),
                output_activation="relu",
            )
        if extractor in ("sbert", "codet5"):
            # SecureBERT pooler_output is tanh-bounded and CodeT5+ embeddings are
            # L2-normalized, both within [-1, 1]: train directly on the raw
            # embeddings (no scaler) with a tanh output so reconstruction can span
            # the negative range, mirroring the reference MyAutoEncoderTanh.
            return AEDetector(
                config=AEConfig(learning_rate=1e-3, epochs=100, batch_size=512),
                extractor=maybe_wrap(extractor_instance, cdir),
                scaler=FunctionTransformer(),
                output_activation="tanh",
            )
        if extractor.startswith("gaur-"):
            # Sigmoid output over MaxAbsScaler's [0, 1] features; lr/batch size match
            # the reference gaur-sql-detect AE.
            return AEDetector(
                config=AEConfig(learning_rate=1e-3, epochs=100, batch_size=4096),
                extractor=maybe_wrap(extractor_instance, cdir),
                scaler=MaxAbsScaler(),
            )
        # Li dense features are non-negative, so a sigmoid output needs inputs in
        # [0, 1]: MaxAbsScaler maps the feature matrix into that range.
        return AEDetector(extractor=maybe_wrap(extractor_instance, cdir), scaler=MaxAbsScaler())
    raise ValueError(f"Unknown method: {name!r} (expected 'ocsvm', 'lof' or 'ae')")


def load_method(name: MethodName, path: Path) -> Detector:
    """Load a previously-saved method by decision-head short name.

    The feature extractor travels with the saved artifact, so it does not need to
    be named again here.
    """
    if name == "ocsvm":
        return OCSVMDetector.load(path)
    if name == "lof":
        return LOFDetector.load(path)
    if name == "ae":
        return AEDetector.load(path)
    raise ValueError(f"Unknown method: {name!r} (expected 'ocsvm', 'lof' or 'ae')")
