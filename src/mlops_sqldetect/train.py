"""CLI to fit a SQL injection detection method (OCSVM or AutoEncoder).

Trains on normal samples only (one-class novelty detection) and persists the
fitted method. Run via ``python -m mlops_sqldetect.train`` or ``invoke train``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from mlops_sqldetect.data import load_dataset, split_normals
from mlops_sqldetect.features import DEFAULT_EXTRACTOR, EXTRACTOR_LABELS
from mlops_sqldetect.model import METHOD_LABELS, MethodName, build_method

logger = logging.getLogger(__name__)


def train(
    dataset: Annotated[Path, typer.Argument(help="CSV with full_query,label columns.")],
    output: Annotated[Path, typer.Option(help="Path to save the fitted method.")] = Path("models/li.joblib"),
    method: Annotated[MethodName, typer.Option(help="Detector head.")] = "ocsvm",
    extractor: Annotated[str, typer.Option(help="Feature extractor short name.")] = DEFAULT_EXTRACTOR,
    limit: Annotated[int | None, typer.Option(help="Label-stratified subset size for smoke runs.")] = None,
    cache: Annotated[bool, typer.Option(help="Cache extractor features to disk to skip recomputation.")] = True,
) -> None:
    """Fit a method on the normal-class samples of ``dataset``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load_dataset(dataset, limit=limit)
    df_normal = split_normals(df)
    logger.info(f"Loaded {len(df)} rows ({len(df_normal)} normal) from {dataset}")

    model = build_method(method, extractor, cache=cache)
    model.fit(df_normal)
    model.save(output)
    logger.info(
        f"Saved {METHOD_LABELS.get(method, method)} method "
        f"({EXTRACTOR_LABELS.get(extractor, extractor)} features) to {output}"
    )


if __name__ == "__main__":
    typer.run(train)
