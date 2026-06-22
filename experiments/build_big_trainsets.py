"""Build the 200k "Big" Superviz26 train sets for the ANUBIS size-sufficiency test.


Usage:
    python -m experiments.build_big_trainsets                  # all 8 scenarios
    python -m experiments.build_big_trainsets --scenario d-d   # one scenario
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from mlops_sqldetect.datasets import superviz26

logger = logging.getLogger(__name__)

NORMAL_TRAIN_SIZE = 100_000
EXTRA_TOTAL = 100_000
BIG_TRAIN_SIZE = NORMAL_TRAIN_SIZE + EXTRA_TOTAL

# Domain letter → full-pool filename under ``--full-root`` (manifest "domains" names
# don't match the file names, so the map is explicit).
DOMAIN_POOL = {
    "a": "OurAirports.csv",
    "b": "sakila.csv",
    "c": "AdventureWorks.csv",
    "d": "OHR.csv",
}


def _per_source_counts(n: int, n_sources: int) -> list[int]:
    """Split ``n`` Extra samples into per-source thirds (remainder to the first sources)."""
    base, rem = divmod(n, n_sources)
    return [base + (1 if i < rem else 0) for i in range(n_sources)]


def _load_pool_extra(pool_path: Path, n: int, seed: int, chunksize: int) -> pd.DataFrame:
    """Sample ``n`` random benign train rows from a full pool.

    Drawn from the pool's ``train`` split (so naturally row-disjoint from the 1M test
    set, which comes from the ``test`` split). No de-duplication against Normal: the
    benign data is template-generated with heavy repetition, so duplicate draws are
    expected and acceptable — we only need 100k more samples from the same population.
    """
    candidates: list[pd.DataFrame] = []
    for chunk in pd.read_csv(pool_path, chunksize=chunksize, low_memory=False):
        keep = chunk[(chunk["split"] == "train") & (chunk["label"] == 0)]
        if not keep.empty:
            candidates.append(keep)
    pool = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    if len(pool) < n:
        raise RuntimeError(f"{pool_path.name}: only {len(pool)} benign train rows available (need {n}).")
    return pool.sample(n=n, random_state=seed).reset_index(drop=True)


def _build_extra(sources: list[str], full_root: Path, seed: int, chunksize: int) -> pd.DataFrame:
    """Draw the 100k Extra for one scenario, one-third per source domain in LODO."""
    counts = _per_source_counts(EXTRA_TOTAL, len(sources))
    parts: list[pd.DataFrame] = []
    for i, (domain, n) in enumerate(zip(sources, counts, strict=True)):
        pool_path = full_root / DOMAIN_POOL[domain]
        part = _load_pool_extra(pool_path, n, seed=seed + i, chunksize=chunksize)
        logger.info(f"  extra[{domain}]: {len(part)} from {pool_path.name}")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _build_scenario(
    scenario: str, sources: list[str], columns: list[str], source_root: Path, full_root: Path, seed: int, chunksize: int
) -> pd.DataFrame:
    """Assemble the full Big CSV (200k train + 1M test) for one scenario."""
    src = pd.read_csv(source_root / f"{scenario}.csv", low_memory=False)
    normal = src[src["split"] == "train"].reset_index(drop=True)
    test = src[src["split"] == "test"].reset_index(drop=True)
    logger.info(f"{scenario}: normal={len(normal)} test={len(test)} sources={sources}")

    extra = _build_extra(sources, full_root, seed, chunksize)
    big_train = pd.concat([normal, extra], ignore_index=True)
    big_train["split"] = "train"

    # Fail-fast invariants: exact size, Normal kept verbatim (superset), Extra benign.
    if len(big_train) != BIG_TRAIN_SIZE:
        raise RuntimeError(f"{scenario}: big train is {len(big_train)}, expected {BIG_TRAIN_SIZE}")
    if len(extra) != EXTRA_TOTAL:
        raise RuntimeError(f"{scenario}: extra is {len(extra)}, expected {EXTRA_TOTAL}")
    # Compare as strings: all-NaN attack columns infer different dtypes per source but
    # write identically to CSV, so value (not dtype) equality is the right invariant.
    front = big_train.iloc[: len(normal)][columns].reset_index(drop=True).astype(str)
    if not front.equals(normal[columns].reset_index(drop=True).astype(str)):
        raise RuntimeError(f"{scenario}: Normal not preserved in Big")
    if not (extra["label"] == 0).all():
        raise RuntimeError(f"{scenario}: Extra contains non-benign rows")

    out = pd.concat([big_train, test], ignore_index=True)
    return out[columns]


def build(
    scenario: Annotated[str, typer.Option(help="Scenario name (a-a … abc-d) or 'all'.")] = "all",
    seed: Annotated[int, typer.Option(help="Base random state for sampling Extra (recorded in logs).")] = 7,
    full_root: Annotated[Path, typer.Option(help="Directory of the full per-domain pools.")] = Path("~/datasets/full"),
    out_root: Annotated[Path, typer.Option(help="Output directory for the Big CSVs.")] = Path(
        "~/datasets/200k-training"
    ),
    source_root: Annotated[Path | None, typer.Option(help="Directory of the Normal Superviz26 CSVs.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Rebuild scenarios whose output CSV already exists.")] = False,
    chunksize: Annotated[int, typer.Option(help="Rows per chunk when streaming the full pools.")] = 500_000,
) -> None:
    """Build the 200k Big train CSVs for the size-sufficiency experiment."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    full_root = full_root.expanduser()
    out_root = out_root.expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    src_root = source_root.expanduser() if source_root else superviz26.default_root()

    files = superviz26.manifest()["files"]
    columns = superviz26.manifest()["columns"]
    scenarios = [s.value for s in superviz26.IN_DOMAIN + superviz26.LODO] if scenario == "all" else [scenario]

    logger.info(f"Building Big train sets (seed={seed}) → {out_root}")
    for name in scenarios:
        out_path = out_root / f"{name}.csv"
        if out_path.exists() and not overwrite:
            logger.info(f"{name}: {out_path} exists, skipping (use --overwrite).")
            continue
        sources = files[f"{name}.csv"]["train_domains"]
        out = _build_scenario(name, sources, columns, src_root, full_root, seed, chunksize)
        out.to_csv(out_path, index=False)
        logger.info(f"{name}: wrote {len(out)} rows → {out_path}")


if __name__ == "__main__":
    typer.run(build)
