"""Tests for feature-study concept-drift and few-shot figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools import generate_drift_figure as drift
from tools import generate_fsl_tex as fsl

THESIS_DATA = Path.home() / "repos/quetel_phd_latex/thesis/chapters/05-generalization/data"


def test_generators_use_feature_study_experiments_and_thesis_outputs():
    assert drift.EXPERIMENT == "FE-study-CD"
    assert fsl.EXPERIMENT_FSL == "FE-study-FSL"
    assert fsl.EXPERIMENT_BASE == "FE-study-LODO"
    assert drift.DEFAULT_FIGURE_OUT == THESIS_DATA / "concept-drift-dumbbell.tex"
    assert fsl.DEFAULT_FIGURE_OUT == THESIS_DATA / "few-shot-adaptation.tex"


def test_drift_method_inventory_covers_new_feature_study_extractors():
    extractors = {extractor for extractor, _, _, _ in drift.METHODS}
    assert {"tfidf", "gaur-claude", "qwen3-emb", "modernbert"} <= extractors


def test_drift_render_keeps_style_and_uses_thesis_label():
    data: drift.Results = {}
    for domain in drift.DOMAINS:
        data[("qwen3-emb", "ae", domain)] = {
            "auroc_s1": 0.95,
            "auroc_s2": 0.90,
            "auprc_s1": 0.85,
            "auprc_s2": 0.80,
        }

    figure = drift.render_figure(data)

    assert "Qwen3-Emb-0.6B" in figure
    assert r"\begin{subfigure}" in figure
    assert r"\label{fig:concept-drift}" in figure


def test_drift_full_inventory_scales_to_a_thesis_page():
    data: drift.Results = {}
    for extractor in drift.EXTRACTOR_ORDER:
        for domain in drift.DOMAINS:
            data[(extractor, "ae", domain)] = {
                "auroc_s1": 0.95,
                "auroc_s2": 0.90,
                "auprc_s1": 0.85,
                "auprc_s2": 0.80,
            }

    figure = drift.render_figure(data)

    assert "height=1.05cm" in figure


def test_fsl_styles_remain_distinguishable_for_complete_feature_study():
    styles = [fsl._style(extractor, index)[2:] for index, extractor in enumerate(drift.EXTRACTOR_ORDER)]

    assert len(styles) == len(set(styles))
    assert fsl._ordered(list(reversed(drift.EXTRACTOR_ORDER))) == drift.EXTRACTOR_ORDER


def test_fsl_render_keeps_superviz26_style_and_uses_thesis_label():
    curves = {"qwen3-emb": {0: 0.70, 5: 0.80, 10: 0.91}}
    refs = {"qwen3-emb": 0.92}

    figure = fsl.render_figure(curves, refs)

    assert r"\begin{semilogxaxis}" in figure
    assert "Qwen3-Emb-0.6B" in figure
    assert r"\label{fig:fine-tune}" in figure
    assert "k=10" in figure
    assert "no more than $0.01$ below, or exceeding" in figure


def test_drift_loader_accepts_legacy_engine_tag_and_keeps_latest_run(monkeypatch):
    runs = pd.DataFrame(
        [
            {
                "start_time": 1,
                "tags.feature_extractor": "cv",
                "tags.domain": "a",
                "tags.pipeline": "ae",
                "metrics.auroc_s1": 0.1,
                "metrics.auroc_s2": 0.2,
                "metrics.auprc_s1": 0.3,
                "metrics.auprc_s2": 0.4,
            },
            {
                "start_time": 2,
                "tags.feature_extractor": "cv",
                "tags.domain": "a",
                "tags.decision_engine": "ae",
                "metrics.auroc_s1": 0.9,
                "metrics.auroc_s2": 0.8,
                "metrics.auprc_s1": 0.7,
                "metrics.auprc_s2": 0.6,
            },
            {
                "start_time": 3,
                "tags.feature_extractor": "li",
                "tags.domain": "b",
                "tags.pipeline": "lof",
                "metrics.auroc_s1": 0.5,
                "metrics.auroc_s2": 0.4,
                "metrics.auprc_s1": 0.3,
                "metrics.auprc_s2": 0.2,
            },
        ]
    )
    monkeypatch.setattr(drift.mlflow, "search_runs", lambda **_: runs)
    monkeypatch.setattr(drift, "_warn_missing", lambda _: None)

    data = drift.load_results()

    assert data[("cv", "ae", "a")] == {
        "auroc_s1": 0.9,
        "auroc_s2": 0.8,
        "auprc_s1": 0.7,
        "auprc_s2": 0.6,
    }
    assert data[("li", "lof", "b")]["auroc_s1"] == 0.5


def test_drift_loader_rejects_an_empty_experiment(monkeypatch):
    monkeypatch.setattr(drift.mlflow, "search_runs", lambda **_: pd.DataFrame())

    with pytest.raises(RuntimeError, match="No finished full-run"):
        drift.load_results()


def test_fsl_curve_loader_reconciles_tags_and_drops_incomplete_budgets(monkeypatch):
    rows = []
    for index, target in enumerate(fsl.TARGETS):
        rows.append(
            {
                "start_time": index,
                "tags.feature_extractor": "cv",
                "tags.target": target,
                "tags.setting": "few_shot" if index % 2 == 0 else None,
                "tags.kind": "few_shot" if index % 2 else None,
                "tags.decision_engine": "ae" if index % 2 == 0 else None,
                "tags.pipeline": "ae" if index % 2 else None,
                "params.k": "0",
                "metrics.auroc": 0.7 + index * 0.1,
            }
        )
    rows.append(
        {
            "start_time": 10,
            "tags.feature_extractor": "cv",
            "tags.target": "a",
            "tags.setting": "few_shot",
            "tags.decision_engine": "ae",
            "params.k": "5",
            "metrics.auroc": 0.99,
        }
    )
    runs = pd.DataFrame(rows)
    monkeypatch.setattr(fsl.mlflow, "search_runs", lambda **_: runs)

    curves = fsl.load_curves()

    assert curves == {"cv": {0: 0.85}}


def test_fsl_indomain_loader_requires_all_four_scenarios(monkeypatch):
    rows = []
    for index, scenario in enumerate(fsl.INDOMAIN):
        rows.append(
            {
                "start_time": index,
                "tags.feature_extractor": "cv",
                "tags.scenario": scenario,
                "tags.decision_engine": "ae" if index % 2 == 0 else None,
                "tags.pipeline": "ae" if index % 2 else None,
                "metrics.roc_auc": 0.8 + index * 0.05,
            }
        )
    rows.append(
        {
            "start_time": 20,
            "tags.feature_extractor": "li",
            "tags.scenario": "a-a",
            "tags.decision_engine": "ae",
            "metrics.roc_auc": 0.99,
        }
    )
    runs = pd.DataFrame(rows)
    monkeypatch.setattr(fsl.mlflow, "search_runs", lambda **_: runs)

    references = fsl.load_indomain(["cv", "li"])

    assert references == {"cv": 0.875}


def test_fsl_curve_loader_rejects_an_empty_experiment(monkeypatch):
    monkeypatch.setattr(fsl.mlflow, "search_runs", lambda **_: pd.DataFrame())

    with pytest.raises(RuntimeError, match="No finished run"):
        fsl.load_curves()


def test_fsl_indomain_loader_omits_ticks_for_an_empty_experiment(monkeypatch):
    monkeypatch.setattr(fsl.mlflow, "search_runs", lambda **_: pd.DataFrame())

    assert fsl.load_indomain(["cv"]) == {}
