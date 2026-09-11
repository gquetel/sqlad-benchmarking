"""Evaluation grid definitions without training-stack imports."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, NamedTuple

import typer

from sqlad_benchmarking.datasets import FAMILIES, DatasetFamily

MethodName = Literal["ocsvm", "lof", "ae"]

ALL_METHODS: tuple[MethodName, ...] = ("ocsvm", "lof", "ae")

METHOD_LABELS: dict[str, str] = {"ocsvm": "OCSVM", "lof": "LOF", "ae": "Autoencoder"}

EXTRACTOR_LABELS: dict[str, str] = {
    "li": "Li",
    "cv": "CountVectorizer",
    "tfidf": "TF-IDF",
    "sbert": "SecureBERT",
    "sbert2": "SecureBERT2",
    "loginov": "Loginov",
    "kakisim": "Kakisim",
    "codet5": "CodeT5+",
    "roberta": "RoBERTa-base",
    "modernbert": "ModernBERT-base",
    "codebert": "CodeBERT",
    "flan-t5": "Flan-T5-Small",
    "sentbert": "SentenceBERT-mpnet",
    "qwen3-emb": "Qwen3-Emb-0.6B",
    "llm2vec": "LLM2Vec-Mistral-7B",
    "gaur-expert": "GAUR (Expert)",
    "gaur-chatgpt": "GAUR (ChatGPT)",
    "gaur-claude": "GAUR (Claude)",
    "gaur-llama": "GAUR (Llama)",
    "gaur-mistral": "GAUR (Mistral)",
    "gaur-gpt-oss": "GAUR (GPT-OSS)",
    "gaur-ruleid": "GAUR (RuleID)",
}

# Extractors assigned to GPU partitions.
GPU_EXTRACTORS = frozenset(
    {
        "sbert",
        "sbert2",
        "codet5",
        "roberta",
        "modernbert",
        "codebert",
        "flan-t5",
        "sentbert",
        "qwen3-emb",
        "llm2vec",
    }
)

DEFAULT_EXTRACTOR = "li"

DEFAULT_KS = "0,5,10,50,100,500,1000,10000"


class Cell(NamedTuple):
    """An evaluation grid cell."""

    scenario: str
    method: str
    extractor: str


def _all_scenarios(family: DatasetFamily) -> dict[str, StrEnum]:
    """Map scenario values to enum members."""
    return {s.value: s for scenarios in family.suites.values() for s in scenarios}


def _validate_grid(
    dataset: str, suite: str, methods: str, extractors: str, scenario: str | None = None
) -> tuple[DatasetFamily, tuple[StrEnum, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate and resolve an evaluation grid."""
    if dataset not in FAMILIES:
        raise typer.BadParameter(f"--dataset must be one of {sorted(FAMILIES)}")
    family = FAMILIES[dataset]
    if scenario is not None:
        scenarios = _all_scenarios(family)
        if scenario not in scenarios:
            raise typer.BadParameter(f"--scenario for {dataset} must be one of {sorted(scenarios)}")
        datasets: tuple[StrEnum, ...] = (scenarios[scenario],)
    else:
        if suite not in family.suites:
            raise typer.BadParameter(f"--suite for {dataset} must be one of {sorted(family.suites)}")
        datasets = family.suites[suite]
    requested_methods = tuple(p.strip() for p in methods.split(",") if p.strip())
    unknown = set(requested_methods) - set(ALL_METHODS)
    if unknown:
        raise typer.BadParameter(f"Unknown method(s): {sorted(unknown)}")
    requested_extractors = tuple(e.strip() for e in extractors.split(",") if e.strip())
    unknown_extractors = set(requested_extractors) - set(EXTRACTOR_LABELS)
    if unknown_extractors:
        raise typer.BadParameter(f"Unknown extractor(s): {sorted(unknown_extractors)}")
    return family, datasets, requested_methods, requested_extractors


def parent_run_spec(family: DatasetFamily, method: str, extractor: str) -> tuple[str, dict[str, str]]:
    """Build an MLflow parent-run specification."""
    name = (
        f"{family.name.capitalize()}:"
        f"{METHOD_LABELS.get(method, method)} and "
        f"{EXTRACTOR_LABELS.get(extractor, extractor)}"
    )
    tags = {
        "decision_engine": method,
        "feature_extractor": extractor,
        "run_role": "parent",
    }
    return name, tags


def enumerate_cells(dataset: str, suite: str, methods: str, extractors: str) -> list[Cell]:
    """List cells in method-extractor-scenario order."""
    _, datasets, requested_methods, requested_extractors = _validate_grid(dataset, suite, methods, extractors)
    return [
        Cell(scenario.value, method, extractor)
        for method in requested_methods
        for extractor in requested_extractors
        for scenario in datasets
    ]
