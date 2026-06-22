"""Unit tests for the detector heads and the build/load factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, StandardScaler

from mlops_sqldetect.features import build_extractor
from mlops_sqldetect.model import (
    AEConfig,
    AEDetector,
    _scaler_for,
    build_pipeline,
    load_pipeline,
)

# Query templates held in variables so the fixtures below build strings by
# concatenating a name (not a SQL-keyword literal), keeping the SQL out of the
# dynamic expression. The varying numeric suffix gives distinct rows.
_NORMAL_TMPL = "select name from airports where id = "
_ATTACK_TMPL = "select * from t union select password from admin -- "


def _train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Distinct, benign-looking queries to train on; the test normals are held-out
    # (unseen) rows and the attacks vary so LOF densities are not degenerate.
    train_normal = [_NORMAL_TMPL + str(i) for i in range(160)]
    test_normal = [_NORMAL_TMPL + str(i) for i in range(200, 250)]
    attack = [_ATTACK_TMPL + str(i) for i in range(30)]
    df_train = pd.DataFrame({"full_query": train_normal, "label": 0})
    df_test = pd.DataFrame({"full_query": test_normal + attack, "label": [0] * len(test_normal) + [1] * len(attack)})
    return df_train, df_test


def _fast_ae(extractor: str) -> AEDetector:
    """An AE with a tiny training budget so tests stay quick."""
    return AEDetector(
        config=AEConfig(epochs=5, batch_size=32),
        extractor=build_extractor(extractor),
        scaler=MaxAbsScaler(),
    )


def test_scaler_for_routes_per_extractor():
    """Each extractor must be paired with the scaler its feature space needs."""
    # cv (raw counts) and the sbert/codet5 embeddings go to OCSVM/LOF unscaled;
    # the Li and Loginov dense features are standardised.
    assert isinstance(_scaler_for("cv"), FunctionTransformer)
    assert isinstance(_scaler_for("sbert"), FunctionTransformer)
    assert isinstance(_scaler_for("codet5"), FunctionTransformer)
    assert isinstance(_scaler_for("li"), StandardScaler)
    assert isinstance(_scaler_for("loginov"), StandardScaler)


@pytest.mark.parametrize("head", ["ocsvm", "lof"])
@pytest.mark.parametrize(
    "extractor",
    [
        "li",
        "loginov",
        "cv",
        pytest.param("sbert", marks=pytest.mark.slow),
        pytest.param("codet5", marks=pytest.mark.slow),
    ],
)
def test_sklearn_heads_rank_attacks_above_normals(head, extractor):
    """End-to-end smoke test of the OCSVM/LOF heads across every extractor.

    This checks that the pipeline runs correctly, we expect a ROC superior to a
    threshold, when not the case, it might exhibit a ranking issue.
    """
    df_train, df_test = _train_test()
    model = build_pipeline(head, extractor).fit(df_train)
    scores = model.score_samples(df_test)
    # Shape + finiteness first: a NaN/inf or mis-shaped score array would poison
    # every downstream metric, so fail loudly here rather than in the harness.
    assert scores.shape == (len(df_test),)
    assert np.isfinite(scores).all()
    # AUC checks score orientation, not model quality: it catches a head that
    # ranks attacks below normals (e.g. a sign error). Maybe the threshold should be adjusted.
    assert roc_auc_score(df_test["label"], scores) >= 0.9


@pytest.mark.parametrize(
    "extractor",
    [
        "li",
        "loginov",
        "cv",
        pytest.param("sbert", marks=pytest.mark.slow),
        pytest.param("codet5", marks=pytest.mark.slow),
    ],
)
def test_ae_sparse_safe_path_runs(extractor):
    """The autoencoder must train and score over each extractor's matrix type.

    This runs a tiny budget over each to prove the fit/score loop and the batched
    densification works.
    """
    df_train, df_test = _train_test()
    scores = _fast_ae(extractor).fit(df_train).score_samples(df_test)
    assert scores.shape == (len(df_test),)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    ("head", "extractor"),
    [
        ("ocsvm", "li"),
        ("lof", "li"),
        ("ae", "li"),
        pytest.param("ocsvm", "sbert", marks=pytest.mark.slow),
        pytest.param("lof", "sbert", marks=pytest.mark.slow),
        pytest.param("ae", "sbert", marks=pytest.mark.slow),
        pytest.param("ocsvm", "codet5", marks=pytest.mark.slow),
        pytest.param("lof", "codet5", marks=pytest.mark.slow),
        pytest.param("ae", "codet5", marks=pytest.mark.slow),
    ],
)
def test_save_load_roundtrip(head, extractor, tmp_path):
    """Checks that a persisted model scores identically after being reloaded."""
    df_train, df_test = _train_test()
    model = _fast_ae(extractor) if head == "ae" else build_pipeline(head, extractor)
    model.fit(df_train)
    suffix = "pt" if head == "ae" else "joblib"
    path = tmp_path / f"model.{suffix}"
    model.save(path)
    reloaded = load_pipeline(head, path)
    np.testing.assert_allclose(model.score_samples(df_test), reloaded.score_samples(df_test), rtol=1e-5)


def test_build_pipeline_unknown_raises():
    """An unknown head name must fail fast with a clear error."""
    with pytest.raises(ValueError, match="Unknown pipeline"):
        build_pipeline("nope", "li")  # type: ignore[arg-type]
