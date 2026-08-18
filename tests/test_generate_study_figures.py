"""Tests for feature-study concept-drift and few-shot figures."""

from __future__ import annotations

from pathlib import Path

from tools import generate_drift_figure as drift
from tools import generate_fsl_tex as fsl

THESIS_DATA = Path("/home/gquetel/repos/quetel_phd_latex/thesis/chapters/05-generalization/data")


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
