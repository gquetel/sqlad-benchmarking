"""SentenceBERT (mpnet) embedding feature extractor.

Maps SQL queries to 768-d dense embeddings using ``sentence-transformers/all-mpnet-base-v2``
via the ``sentence-transformers`` library (mirrors the reference ``sentbert.py``).
The model is pretrained, so ``fit`` is a no-op; the model is lazy-loaded on first
use to keep ``build_extractor`` cheap and avoid the download in tests that never
touch this extractor. Disk caching is layered on externally via
:class:`~sqlad_benchmarking.features.cache.CachingExtractor`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from sqlad_benchmarking.determinism import enable_determinism

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_BATCH_SIZE = 512
EMBED_DIM = 768


def _as_queries(X) -> list[str]:  # noqa: N803
    if isinstance(X, pd.DataFrame):
        return X["full_query"].astype(str).tolist()
    return [str(q) for q in X]


class SentBertExtractor(BaseEstimator, TransformerMixin):
    """Sklearn transformer producing SentenceBERT (mpnet) embeddings.

    ``transform`` returns a ``(n_samples, 768)`` float32 ndarray. Inference runs on
    GPU when available. :func:`~sqlad_benchmarking.determinism.enable_determinism`
    pins the CUDA kernels so runs are reproducible.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: torch.device | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_model(self) -> None:
        """Lazy-load model on first use (heavy import + download)."""
        if getattr(self, "model_", None) is not None:
            return
        # Imported lazily so the sentence-transformers dependency is only needed
        # when sentbert is actually used.
        from sentence_transformers import SentenceTransformer

        enable_determinism()
        torch.manual_seed(2)
        self.model_ = SentenceTransformer(self.model_name, device=str(self.device))

    def fit(self, X, y=None) -> "SentBertExtractor":  # noqa: N803
        return self

    def transform(self, X) -> np.ndarray:  # noqa: N803
        self._ensure_model()
        queries = _as_queries(X)
        if not queries:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        embeddings = self.model_.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
