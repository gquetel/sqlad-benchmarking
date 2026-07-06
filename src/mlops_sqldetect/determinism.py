"""Best-effort determinism controls for the torch-backed pipelines.

Calling :func:`enable_determinism` before model load /
training pins those so AE weights and CodeT5+/SecureBERT embeddings are
reproducible across runs on the same hardware + library versions.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_configured = False


def enable_determinism() -> None:
    """Pin torch/CUDA to deterministic kernels (idempotent, safe to call often).

    Uses ``warn_only=True`` so an op without a deterministic implementation warns
    instead of raising, keeping the pipelines runnable on any backend.
    """
    global _configured
    if _configured:
        return
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _configured = True
    logger.debug("Deterministic torch backend enabled")
