"""Unit tests for the feature extractors and their registry."""

from __future__ import annotations

import pickle
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin

from sqlad_benchmarking.features import EXTRACTORS, GPU_EXTRACTORS, build_extractor
from sqlad_benchmarking.features.cache import CachingExtractor
from sqlad_benchmarking.features.countvect import CountVectorizerExtractor
from sqlad_benchmarking.features.kakisim import KakisimExtractor
from sqlad_benchmarking.features.loginov import FEATURE_NAMES, LoginovExtractor, extract_loginov_features
from sqlad_benchmarking.features.qwen3_emb import Qwen3EmbExtractor
from sqlad_benchmarking.features.tfidf import TfidfExtractor


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "full_query": [
                "select * from users",
                "select name from airports where id = 1",
                "select * from t where 1=1 union select password from admin",
            ]
        }
    )


def test_registry_exposes_all_extractors():
    assert set(EXTRACTORS) == {
        "li",
        "cv",
        "tfidf",
        "sbert",
        "sbert2",
        "loginov",
        "kakisim",
        "codet5",
        "roberta",
        "modernbert",
        "codebert",
        "flan-t5",
        "sentbert",
        "qwen3-emb",
        "gaur-expert",
        "gaur-chatgpt",
        "gaur-claude",
        "gaur-llama",
        "gaur-mistral",
        "gaur-gpt-oss",
        "gaur-ruleid",
    }


def test_gpu_extractors_are_registered():
    assert GPU_EXTRACTORS == {
        "sbert",
        "sbert2",
        "codet5",
        "roberta",
        "modernbert",
        "codebert",
        "flan-t5",
        "sentbert",
        "qwen3-emb",
    }
    assert GPU_EXTRACTORS <= EXTRACTORS.keys()


def test_build_extractor_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown feature extractor"):
        build_extractor("does-not-exist")


def test_qwen_loads_in_bfloat16(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", constructor)

    extractor = Qwen3EmbExtractor(device=torch.device("cpu"))
    extractor._ensure_model()

    assert extractor.batch_size == 16
    assert constructor.call_args.kwargs["model_kwargs"] == {
        "torch_dtype": torch.bfloat16,
    }


def test_countvect_is_stateful_and_sparse():
    df = _frame()
    ext = CountVectorizerExtractor(max_features=50)
    # transform before fit must fail: the vocabulary is learned at fit time.
    with pytest.raises(Exception):  # noqa: B017 - sklearn raises NotFittedError
        ext.transform(df)
    ext.fit(df)
    matrix = ext.transform(df)
    assert issparse(matrix)
    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] == len(ext.get_feature_names_out())


def test_countvect_accepts_list_of_strings():
    ext = CountVectorizerExtractor().fit(["select 1", "select 2"])
    assert ext.transform(["select 1"]).shape == (1, ext.transform(["select 1"]).shape[1])


def test_tfidf_is_stateful_and_sparse():
    df = _frame()
    ext = TfidfExtractor(max_features=50)
    # transform before fit must fail: the vocabulary is learned at fit time.
    with pytest.raises(Exception):  # noqa: B017 - sklearn raises NotFittedError
        ext.transform(df)
    ext.fit(df)
    matrix = ext.transform(df)
    assert issparse(matrix)
    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] == len(ext.get_feature_names_out())


def test_tfidf_caps_vocabulary_at_200_by_default():
    ext = TfidfExtractor()
    assert ext.max_features == 200
    queries = [f"select col{i} from t{i}" for i in range(300)]
    ext.fit(queries)
    assert ext.transform(queries).shape[1] == 200


def test_li_extractor_is_dense_fixed_width():
    ext = build_extractor("li")
    matrix = ext.fit(_frame()).transform(_frame())
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == 3
    assert matrix.dtype == np.float32


# ----- Loginov ---------------------------------------------------------------


def test_loginov_features_keys_and_token_counts():
    # With nothing whitelisted, every special-char group counts as anomalous.
    feats = extract_loginov_features("select * from t where id = 1", frozenset())
    assert set(feats) == set(FEATURE_NAMES)
    # '*' and '=' are the two special-char groups.
    assert feats["n_anomalous_schars"] == 2
    # SELECT / FROM / WHERE are keywords; t / id are alpha; 1 is numeric.
    assert feats["s1_n_keywords"] == 3
    assert feats["s1_n_alpha"] == 2
    assert feats["s1_n_numeric"] == 1
    assert feats["s1_n_mixed"] == 0


def test_loginov_is_stateful_and_dense():
    df = _frame()
    ext = LoginovExtractor()
    # transform before fit must fail: valid_schars is learned at fit time.
    with pytest.raises(Exception):  # noqa: B017 - sklearn raises NotFittedError
        ext.transform(df)
    matrix = ext.fit(df).transform(df)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(df), len(FEATURE_NAMES))
    assert matrix.dtype == np.float32


def test_loginov_flags_special_chars_unseen_in_training():
    # Training queries only ever use '=' as punctuation; a '#' in the test query
    # is then an anomalous special-char group.
    ext = LoginovExtractor().fit(pd.DataFrame({"full_query": ["select a from t where id = 1"]}))
    out = ext.transform(pd.DataFrame({"full_query": ["select a from t where id = 1 or 1=1#"]}))
    assert out[0, 0] == 1  # n_anomalous_schars is the first column


def test_loginov_cache_key_tracks_learned_state():
    a = LoginovExtractor().fit(pd.DataFrame({"full_query": ["select a from t where id = 1"]}))
    b = LoginovExtractor().fit(pd.DataFrame({"full_query": ["select a from t where id like '%x%'"]}))
    # Different training punctuation -> different state -> different cache key.
    assert a.cache_key_state() != b.cache_key_state()


# ----- Kakisim -----------------------------------------------------------------


def test_kakisim_is_stateful_and_sparse():
    df = _frame()
    ext = KakisimExtractor()
    # transform before fit must fail: per-view vocabularies are learned at fit time.
    with pytest.raises(Exception):  # noqa: B017 - AttributeError, no vectorizers_ yet
        ext.transform(df)
    ext.fit(df)
    matrix = ext.transform(df)
    assert issparse(matrix)
    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] == len(ext.get_feature_names_out())


# ----- CachingExtractor ------------------------------------------------------


class _CountingStub(BaseEstimator, TransformerMixin):
    """Deterministic dense extractor that records how often transform ran.

    Avoids the 500 MB SecureBERT download while exercising the cache machinery.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.n_calls = 0

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        self.n_calls += 1
        return np.array([[len(q) * self.scale] for q in X["full_query"]], dtype=np.float32)


def test_cache_hit_skips_base_transform(tmp_path):
    base = _CountingStub()
    ext = CachingExtractor(base, cache_dir=tmp_path)
    first = ext.transform(_frame())
    second = ext.transform(_frame())
    # Second call served from disk: base.transform ran exactly once.
    assert base.n_calls == 1
    np.testing.assert_array_equal(first, second)


def test_cache_persists_across_instances(tmp_path):
    CachingExtractor(_CountingStub(), cache_dir=tmp_path).transform(_frame())
    fresh_base = _CountingStub()
    CachingExtractor(fresh_base, cache_dir=tmp_path).transform(_frame())
    # The fresh instance hit the on-disk cache without invoking its base.
    assert fresh_base.n_calls == 0


def test_cache_key_depends_on_params(tmp_path):
    # Different params (here, scale) must not collide on the same input.
    a = _CountingStub(scale=1.0)
    b = _CountingStub(scale=2.0)
    out_a = CachingExtractor(a, cache_dir=tmp_path).transform(_frame())
    out_b = CachingExtractor(b, cache_dir=tmp_path).transform(_frame())
    assert b.n_calls == 1  # not a stale hit on a's entry
    np.testing.assert_array_equal(out_b, out_a * 2.0)


def test_cache_disabled_is_passthrough(tmp_path):
    base = _CountingStub()
    ext = CachingExtractor(base, cache_dir=None)
    ext.transform(_frame())
    ext.transform(_frame())
    assert base.n_calls == 2
    assert not list(tmp_path.iterdir())


def test_cache_roundtrips_sparse(tmp_path):
    base = CountVectorizerExtractor().fit(_frame())
    ext = CachingExtractor(base, cache_dir=tmp_path)
    first = ext.transform(_frame())
    second = ext.transform(_frame())
    assert issparse(first) and issparse(second)
    assert (first != second).nnz == 0


def test_cache_invalidates_on_vocabulary_change(tmp_path):
    # A refitted CountVect with a different vocabulary must not reuse stale features.
    ext = CountVectorizerExtractor().fit(_frame())
    cached = CachingExtractor(ext, cache_dir=tmp_path)
    first = cached.transform(_frame())
    ext.fit(pd.DataFrame({"full_query": ["drop table students", "delete from logs"]}))
    second = cached.transform(_frame())
    assert isinstance(second, csr_matrix)
    assert first.shape[1] != second.shape[1] or (first != second).nnz != 0


def test_cache_pickle_roundtrip(tmp_path):
    ext = CachingExtractor(_CountingStub(), cache_dir=tmp_path)
    ext.transform(_frame())
    restored = pickle.loads(pickle.dumps(ext))  # noqa: S301 - round-tripping our own object
    before = restored.base.n_calls
    out = restored.transform(_frame())
    # Restored extractor still reads the same cache dir, so the base is not re-invoked.
    assert restored.base.n_calls == before
    np.testing.assert_array_equal(out, np.array([[len(q)] for q in _frame()["full_query"]], dtype=np.float32))
