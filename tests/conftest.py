"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from mlops_sqldetect.features.cache import CACHE_DIR_ENV


@pytest.fixture(autouse=True)
def _isolated_feature_cache(tmp_path, monkeypatch):
    """Point the feature cache at a per-test temp dir.

    Keeps the suite hermetic: tests never read/write the persistent
    ``data/processed`` cache, so a stale entry can't leak a false hit.
    """
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "feature_cache"))
