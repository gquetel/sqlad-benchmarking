"""Generic on-disk caching wrapper for feature extractors.

Wraps any sklearn-compatible extractor and memoises its ``transform`` output to
a compressed ``.npz`` file, keyed by the extractor's fingerprint plus the exact
ordered input queries. The expensive case is SecureBERT (re-embedding identical
splits across the ocsvm/lof/ae heads), but nothing here is model-specific: the
wrapper plugs onto ``li``, ``cv`` or ``sbert`` alike.

Granularity is the *whole feature matrix* (one file per distinct input), not one
file per query, so a full Superviz26 suite produces only a few dozen files.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin

# Default cache location: a subfolder of the (gitignored) data/processed dir.
DEFAULT_CACHE_DIR = Path("data/processed/feature_cache")
# Env var overriding the default dir when caching is enabled.
CACHE_DIR_ENV = "SQLDETECT_CACHE_DIR"
# Dense float matrices are stored at this dtype to halve the footprint; restored
# to float32 on load so downstream scalers see today's dtype.
STORE_DTYPE = np.float16

logger = logging.getLogger(__name__)


def resolve_cache_dir(cache: bool = True, cache_dir: str | os.PathLike | None = None) -> Path | None:
    """Resolve the cache directory, or ``None`` when caching is disabled.

    Explicit ``cache_dir`` wins, then the ``SQLDETECT_CACHE_DIR`` env var, then
    :data:`DEFAULT_CACHE_DIR`.
    """
    if not cache:
        return None
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get(CACHE_DIR_ENV)
    return Path(env) if env else DEFAULT_CACHE_DIR


def _as_queries(X) -> list[str]:  # noqa: N803
    if isinstance(X, pd.DataFrame):
        return X["full_query"].astype(str).tolist()
    return [str(q) for q in X]


class CachingExtractor(BaseEstimator, TransformerMixin):
    """Transformer that memoises ``base.transform`` output to disk.

    ``fit`` delegates to the wrapped extractor so stateful ones (e.g. CountVect)
    still learn their vocabulary. ``transform`` returns the cached matrix when the
    fingerprint + input match, otherwise computes, stores and returns it. With
    ``cache_dir=None`` it is a transparent pass-through.
    """

    def __init__(self, base: TransformerMixin, cache_dir: str | os.PathLike | None = None) -> None:
        self.base = base
        self.cache_dir = cache_dir

    def fit(self, X, y=None) -> "CachingExtractor":  # noqa: N803
        self.base.fit(X, y)
        return self

    def transform(self, X):  # noqa: N803
        if self.cache_dir is None:
            return self.base.transform(X)
        path = Path(self.cache_dir) / f"{self._key(X)}.npz"
        cached = self._load(path)
        if cached is not None:
            logger.info("Feature cache hit: %s", path.name)
            return cached
        result = self.base.transform(X)
        self._save(path, result)
        logger.info("Feature cache saved: %s", path.name)
        return result

    def get_feature_names_out(self, input_features=None):
        return self.base.get_feature_names_out(input_features)

    # ----- internals ---------------------------------------------------------

    def _fingerprint(self) -> str:
        """Identity of the wrapped extractor: class + params + optional state.

        ``cache_key_state`` lets a stateful extractor (e.g. CountVect's fitted
        vocabulary) fold its state into the key so a cache hit can't return
        features computed from a different state.
        """
        parts = [type(self.base).__name__, repr(sorted(self.base.get_params().items()))]
        state = getattr(self.base, "cache_key_state", None)
        if callable(state):
            parts.append(str(state()))
        return "|".join(parts)

    def _key(self, X) -> str:  # noqa: N803
        """blake2b over the fingerprint and the exact ordered query list."""
        h = hashlib.blake2b(digest_size=16)
        h.update(self._fingerprint().encode())
        for q in _as_queries(X):
            h.update(b"\0")
            h.update(q.encode("utf-8", "surrogatepass"))
        return h.hexdigest()

    @staticmethod
    def _load(path: Path):
        if not path.exists():
            return None
        with np.load(path) as npz:
            if str(npz["kind"]) == "sparse":
                return csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=tuple(npz["shape"]))
            return npz["data"].astype(np.float32)

    def _save(self, path: Path, result) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        # Write to a pid-suffixed temp file then atomically rename, so parallel
        # SLURM jobs computing the same cell never see a half-written file.
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with open(tmp, "wb") as f:
            if issparse(result):
                csr = result.tocsr()
                np.savez_compressed(
                    f,
                    kind="sparse",
                    data=csr.data,
                    indices=csr.indices,
                    indptr=csr.indptr,
                    shape=np.asarray(csr.shape),
                )
            else:
                np.savez_compressed(f, kind="dense", data=np.asarray(result).astype(STORE_DTYPE))
        os.replace(tmp, path)


def maybe_wrap(extractor: TransformerMixin, cache_dir: str | os.PathLike | None) -> TransformerMixin:
    """Wrap ``extractor`` in a :class:`CachingExtractor` when ``cache_dir`` is set."""
    if cache_dir is None:
        return extractor
    return CachingExtractor(extractor, cache_dir)
