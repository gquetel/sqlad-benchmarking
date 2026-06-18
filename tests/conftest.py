"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from mlops_sqldetect.features.cache import CACHE_DIR_ENV


def pytest_addoption(parser):
    """Add ``--skip-slow`` to opt out of the slow tests.

    Slow tests exercise heavy embedding models (large download + inference) and
    run by default, including in CI; pass ``--skip-slow`` to deselect them in
    environments where that cost is undesirable.
    """
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip tests marked 'slow' (those that download/run a heavy embedding model).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip-slow"):
        return
    skip_slow = pytest.mark.skip(reason="deselected by --skip-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _isolated_feature_cache(tmp_path, monkeypatch):
    """Point the feature cache at a per-test temp dir.

    Keeps the suite hermetic: tests never read/write the persistent
    ``data/processed`` cache, so a stale entry can't leak a false hit.
    """
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "feature_cache"))
