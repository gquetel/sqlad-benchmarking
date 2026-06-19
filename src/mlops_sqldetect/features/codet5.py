"""CodeT5+ embedding feature extractor.

Maps SQL queries to 256-d dense embeddings using the code-domain model
``Salesforce/codet5p-110m-embedding`` (mirrors the reference ``codet5.py``).
The model is pretrained, so ``fit`` is a no-op; the tokenizer/model are lazy-loaded
on first use to keep ``build_extractor`` cheap and avoid the download in tests that
never touch this extractor. Like SecureBERT it is GPU-only in practice, and OOMs on a 16 GB
V100, so the SLURM submitter pins it to GPUs with >16 GB VRAM (``min_vram_gb`` in
``configs/slurm.yaml``); disk caching is layered on externally via
:class:`~mlops_sqldetect.features.cache.CachingExtractor`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_MODEL = "Salesforce/codet5p-110m-embedding"
DEFAULT_BATCH_SIZE = 512
MAX_LENGTH = 512
EMBED_DIM = 256


def _as_queries(X) -> list[str]:  # noqa: N803
    if isinstance(X, pd.DataFrame):
        return X["full_query"].astype(str).tolist()
    return [str(q) for q in X]


class CodeT5Extractor(BaseEstimator, TransformerMixin):
    """Sklearn transformer producing CodeT5+ 256-d embeddings.

    ``transform`` returns a ``(n_samples, 256)`` float32 ndarray. Inference runs on
    GPU when available, in ``torch.no_grad`` eval mode, with a fixed seed so the
    embeddings are deterministic across runs. The model returns L2-normalized
    embeddings directly, so no scaler is applied downstream (see ``_scaler_for``).
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
        # Imported lazily so the transformers dependency is only needed when codet5
        # is actually used. trust_remote_code is required by the embedding model.
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        torch.manual_seed(2)
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # The model's remote config predates current transformers and omits the
        # T5 encoder flags T5Stack now reads; default them so loading doesn't raise.
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        if not hasattr(config, "is_decoder"):
            config.is_decoder = False
        self.model_ = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, config=config).to(self.device)
        self.model_.eval()

    def fit(self, X, y=None) -> "CodeT5Extractor":  # noqa: N803
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
            input_ids = inputs["input_ids"].to(model_device)
            # Model returns (batch_size, 256) L2-normalized embeddings directly.
            embeddings = self.model_(input_ids)
            out.append(embeddings.cpu().numpy())
        if not out:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)
