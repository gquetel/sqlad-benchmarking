"""Unit tests for the metric helpers in sqlad_benchmarking.metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sqlad_benchmarking.metrics import (
    compute_metrics,
    recall_per_attack,
    threshold_for_fpr,
    wilson_ci,
)


def test_threshold_for_fpr_is_percentile():
    normal = np.arange(0, 1001, dtype=float)  # 0..1000
    # 1% FPR 99th percentile.
    assert threshold_for_fpr(normal, 0.01) == pytest.approx(np.percentile(normal, 99.0))


def test_wilson_ci_matches_formula():
    assert wilson_ci(0.5, 100) == pytest.approx(1.96 * math.sqrt(0.25 / 100))
    assert wilson_ci(1.0, 100) == 0.0  # degenerate score zero width
    assert wilson_ci(0.9, 0) == 0.0  # guard against n == 0


def test_perfect_separation():
    # Normals score 0, attacks score 1; threshold 0.5 separates them cleanly.
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    m = compute_metrics(labels, scores, threshold=0.5)
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["fpr"] == 0.0
    assert m["auprc"] == pytest.approx(1.0)
    assert m["rocauc"] == pytest.approx(1.0)


def test_achieved_fpr_and_recall():
    # One normal mislabelled (score above threshold), one attack missed.
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.9, 0.2, 0.8])
    m = compute_metrics(labels, scores, threshold=0.5)
    assert m["fpr"] == pytest.approx(0.5)  # 1 of 2 normals above threshold
    assert m["recall"] == pytest.approx(0.5)  # 1 of 2 attacks caught


def test_recall_per_attack_skips_nan_and_empty():
    labels = np.array([0, 1, 1, 1, 1])
    preds = np.array([0, 1, 0, 1, 1])
    techniques = pd.Series([np.nan, "tautology", "tautology", "union", ""])
    out = recall_per_attack(labels, preds, techniques)
    assert set(out) == {"tautology", "union"}  # NaN (normal) and "" excluded
    assert out["tautology"] == pytest.approx(0.5)  # 1 of 2 caught
    assert out["union"] == pytest.approx(1.0)  # 1 of 1 caught
