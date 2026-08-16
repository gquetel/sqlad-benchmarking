"""Qwen3-Embedding-0.6B embedding feature extractor.

Maps SQL queries to 1024-d dense embeddings using ``Qwen/Qwen3-Embedding-0.6B``
via the ``sentence-transformers`` library, loaded in bfloat16 (mirrors the
reference ``qwen3_emb.py``). The model is pretrained, so ``fit`` is a no-op; the
model is lazy-loaded on first use to keep ``build_extractor`` cheap and avoid the
download in tests that never touch this extractor. Disk caching is layered on
externally via :class:`~sqlad_benchmarking.features.cache.CachingExtractor`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from sqlad_benchmarking.determinism import enable_determinism

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BATCH_SIZE = 16
EMBED_DIM = 1024


def _as_queries(X) -> list[str]:  # noqa: N803
    if isinstance(X, pd.DataFrame):
        return X["full_query"].astype(str).tolist()
    return [str(q) for q in X]


class Qwen3EmbExtractor(BaseEstimator, TransformerMixin):
    """Sklearn transformer producing L2-normalized Qwen3-Embedding-0.6B embeddings.

    ``transform`` returns a ``(n_samples, 1024)`` float32 ndarray. Inference runs on
    GPU when available, with the model loaded in bfloat16 (its native dtype).
    :func:`~sqlad_benchmarking.determinism.enable_determinism` pins the CUDA kernels
    so runs are reproducible.
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
        # when qwen3-emb is actually used.
        from sentence_transformers import SentenceTransformer

        enable_determinism()
        torch.manual_seed(2)
        self.model_ = SentenceTransformer(
            self.model_name,
            device=str(self.device),
            model_kwargs={"torch_dtype": torch.bfloat16},
            config_kwargs={"use_cache": False},
        )

    def fit(self, X, y=None) -> "Qwen3EmbExtractor":  # noqa: N803
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
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)
