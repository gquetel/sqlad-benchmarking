"""Flan-T5-Small embedding feature extractor.

Maps SQL queries to 512-d dense embeddings by mean-pooling the encoder hidden
states of ``google/flan-t5-small`` (mirrors the reference ``flan_t5.py``). The
model is pretrained, so ``fit`` is a no-op; the tokenizer/model are lazy-loaded
on first use to keep ``build_extractor`` cheap and avoid the download in tests
that never touch this extractor. Disk caching is layered on externally via
:class:`~sqlad_benchmarking.features.cache.CachingExtractor`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from sqlad_benchmarking.determinism import enable_determinism

DEFAULT_MODEL = "google/flan-t5-small"
DEFAULT_BATCH_SIZE = 128
MAX_LENGTH = 512
EMBED_DIM = 512


def _as_queries(X) -> list[str]:  # noqa: N803
    if isinstance(X, pd.DataFrame):
        return X["full_query"].astype(str).tolist()
    return [str(q) for q in X]


class FlanT5Extractor(BaseEstimator, TransformerMixin):
    """Sklearn transformer producing mean-pooled Flan-T5-Small encoder embeddings.

    ``transform`` returns a ``(n_samples, 512)`` float32 ndarray. Inference runs on
    GPU when available, in ``torch.no_grad`` eval mode.
    :func:`~sqlad_benchmarking.determinism.enable_determinism` pins the CUDA kernels so
    runs are reproducible.
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
        """Lazy-load tokenizer + model on first use (heavy import + download)."""
        if getattr(self, "model_", None) is not None:
            return
        # Imported lazily so the transformers dependency is only needed when
        # flan_t5 is actually used. Only the encoder half is needed for embeddings.
        from transformers import T5EncoderModel, T5TokenizerFast

        enable_determinism()
        torch.manual_seed(2)
        self.tokenizer_ = T5TokenizerFast.from_pretrained(self.model_name)
        self.model_ = T5EncoderModel.from_pretrained(self.model_name).to(self.device)
        self.model_.eval()

    def fit(self, X, y=None) -> "FlanT5Extractor":  # noqa: N803
        return self

    @torch.no_grad()
    def transform(self, X) -> np.ndarray:  # noqa: N803
        self._ensure_model()
        # Follow the model's actual device rather than self.device: a checkpoint
        # loaded with map_location (see AEDetector.load) can leave the unpickled
        # model on CPU while self.device still reads "cuda" from when it was saved.
        model_device = next(self.model_.parameters()).device
        queries = _as_queries(X)
        out: list[np.ndarray] = []
        for start in range(0, len(queries), self.batch_size):
            batch = queries[start : start + self.batch_size]
            inputs = self.tokenizer_(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            hidden = self.model_(**inputs).last_hidden_state
            # Mean pool over token dimension, excluding padding tokens.
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(pooled.cpu().numpy())
        if not out:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)
