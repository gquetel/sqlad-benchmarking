"""Tests for the in-repo port of llm2vec inference.

The port was validated once against the real ``llm2vec`` 0.2.3 package (transformers
4.44.2) on a tiny fixture model: prepared text, ``input_ids``, ``attention_mask`` and
``embed_mask`` matched exactly, and the embeddings matched bit-for-bit at batch sizes 3
and 16 (1.2e-07 at batch size 2, i.e. below the reference's own batch-to-batch spread).
That comparison is the real correctness evidence and cannot run here -- it needs the
pinned transformers 4.x. The tests below are regression guards for the handful of
non-obvious things the port had to replicate: the instruction template, the separate
``embed_mask``, the clipped token count driving word truncation, order-preserving
batched encoding, and bidirectional attention.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import MistralConfig, MistralModel

from sqlad_benchmarking.features.llm2vec import (
    DOC_MAX_LENGTH,
    EMBED_DIM,
    MAX_LENGTH,
    LLM2VecExtractor,
    MistralBiModel,
)

BOS_ID = 1
PAD_ID = 0


class FakeTokenizer:
    """Word-level stand-in with the tokenizer behaviour the port depends on.

    Real sentencepiece splits ``"[/INST]"`` into several tokens; what matters for the
    port is the *geometry* (BOS placement, left padding, truncation, and how a standalone
    re-tokenization lines up with the suffix of the full sequence), which this reproduces.
    """

    padding_side = "left"
    pad_token = "<pad>"  # noqa: S105 - a token string, not a credential
    eos_token = "<pad>"  # noqa: S105

    def _ids(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = [(abs(hash(w)) % 30000) + 100 for w in text.split()]
        return ([BOS_ID] if add_special_tokens else []) + ids

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        batch = [self._ids(t, add_special_tokens) for t in texts]
        if truncation and max_length is not None:
            batch = [ids[:max_length] for ids in batch]
        width = max(len(ids) for ids in batch)
        input_ids = [[PAD_ID] * (width - len(ids)) + ids for ids in batch]
        attention_mask = [[0] * (width - len(ids)) + [1] * len(ids) for ids in batch]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _extractor() -> LLM2VecExtractor:
    ext = LLM2VecExtractor(device=torch.device("cpu"))
    ext.tokenizer_ = FakeTokenizer()
    return ext


# ---- text preparation ------------------------------------------------------------


def test_query_is_wrapped_in_instruction_template():
    """The MNTP config resolves to Mistral-7B-Instruct-v0.2, which selects this template.

    Also covers the whitespace quirk: the reference strips the sentinel-joined string,
    whose leading "!" makes the strip right-side only, and an empty query must still
    leave a non-empty suffix to pool over.
    """
    _, full, suffix = _extractor()._split_query("SELECT 1")
    assert full == "[INST] SELECT 1 [/INST]"
    # Only the suffix is pooled over: the "[INST]" prefix is excluded.
    assert suffix == "SELECT 1 [/INST]"

    _, full, _ = _extractor()._split_query("  SELECT 1  ")
    assert full == "[INST]   SELECT 1 [/INST]"

    for query in ("", "   ", "\n\t"):
        _, full, suffix = _extractor()._split_query(query)
        assert full == "[INST]  [/INST]"
        assert suffix == " [/INST]"


def test_long_query_is_truncated_by_words_to_doc_max_length():
    ext = _extractor()
    query = " ".join(f"col_{i}" for i in range(900))
    truncated, _, _ = ext._split_query(query)
    assert ext._n_tokens(truncated) <= DOC_MAX_LENGTH
    # Word-level truncation, so the kept part is a word prefix of the original.
    assert query.startswith(truncated)


def test_token_count_for_truncation_is_clipped_to_max_length():
    """The reference measures length with truncation=True, so the word-drop ratio is
    computed from a count already capped at MAX_LENGTH. Removing the cap would cut far
    more words than the reference does."""
    ext = _extractor()
    assert ext._n_tokens(" ".join("w" for _ in range(5000))) == MAX_LENGTH


# ---- tokenization / embed_mask ---------------------------------------------------


def test_embed_mask_is_a_suffix_that_excludes_bos_and_the_instruction_prefix():
    ext = _extractor()
    queries = ["SELECT 1", "SELECT a b c d e FROM t"]
    features = ext._tokenize([(f, s) for _, f, s in (ext._split_query(q) for q in queries)])
    attention_mask, embed_mask = features["attention_mask"], features["embed_mask"]

    for i in range(len(queries)):
        n_attend, n_embed = int(attention_mask[i].sum()), int(embed_mask[i].sum())
        # "<s>" plus the "[INST]" prefix (one word-token for the fake tokenizer).
        assert n_embed == n_attend - 2
        # Contiguous suffix, and never covering a padded position.
        assert embed_mask[i, -n_embed:].all()
        assert not embed_mask[i, :-n_embed].any()
        assert (embed_mask[i] <= attention_mask[i]).all()


# ---- pooling and batching --------------------------------------------------------


def test_pool_averages_only_the_masked_suffix():
    ext = _extractor()
    # Position p of every row is the constant p, so the mean reveals which positions were used.
    last_hidden = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1).repeat(2, 1, 1)
    embed_mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]])
    pooled = ext._pool(last_hidden, embed_mask)
    assert pooled.flatten().tolist() == [4.0, 4.5]  # mean(3,4,5) and mean(4,5)


def test_encode_restores_input_order_across_batch_sizes():
    """Encoding is length-sorted for padding efficiency; the output order must not change."""
    ext = _extractor()
    queries = ["SELECT 1", "SELECT a b c d e f g h FROM t", "SELECT x", "SELECT p q r FROM u"]

    # Stub model: each row's hidden state is a constant equal to its number of real
    # tokens, so a mis-ordered result is immediately visible.
    class StubModel:
        def __call__(self, input_ids, attention_mask):
            lengths = attention_mask.sum(dim=-1).to(torch.float32)
            hidden = lengths.reshape(-1, 1, 1).repeat(1, input_ids.shape[1], 4)
            return type("Out", (), {"last_hidden_state": hidden})()

    ext.model_ = StubModel()
    ext.batch_size = 16
    reference = ext._encode(queries)
    for batch_size in (1, 2, 3):
        ext.batch_size = batch_size
        assert np.array_equal(ext._encode(queries), reference)
    # Longer queries really do produce larger values, i.e. rows were not shuffled.
    assert reference[1, 0] > reference[0, 0]
    assert reference[3, 0] > reference[2, 0]


def test_transform_on_empty_input_returns_correct_shape():
    # Must not touch the model: an empty frame should never trigger a 14 GB download.
    assert LLM2VecExtractor().transform([]).shape == (0, EMBED_DIM)


# ---- the bidirectional model ------------------------------------------------------


def _tiny_models() -> tuple[MistralBiModel, MistralModel]:
    """A bidirectional and a causal model sharing identical weights."""
    torch.manual_seed(0)
    config = MistralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=None,
    )
    bi = MistralBiModel(config).eval()
    causal = MistralModel(config).eval()
    causal.load_state_dict(bi.state_dict())
    return bi, causal


def test_unpadded_batch_stays_bidirectional():
    """Regression guard for the is_causal flag.

    With no padding, transformers skips mask construction and SDPA falls back to the
    attention module's own is_causal, so leaving it at its default True would silently
    make exactly these batches causal -- and unpadded batches are common once queries
    are length-sorted.
    """
    bi, causal = _tiny_models()
    assert not any(layer.self_attn.is_causal for layer in bi.layers)
    input_ids = torch.randint(0, 64, (3, 7))
    unpadded = torch.ones_like(input_ids)
    with torch.no_grad():
        h_bi = bi(input_ids=input_ids, attention_mask=unpadded).last_hidden_state
        h_causal = causal(input_ids=input_ids, attention_mask=unpadded).last_hidden_state
    # Under causal attention the first token cannot see the rest of the sequence.
    assert (h_bi[:, 0] - h_causal[:, 0]).abs().max() > 1e-3


def test_padded_batch_stays_bidirectional():
    """Same check on the other branch: here create_bidirectional_mask returns a real mask."""
    bi, causal = _tiny_models()
    input_ids = torch.randint(2, 64, (2, 7))
    padded = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1] * 7])
    with torch.no_grad():
        h_bi = bi(input_ids=input_ids, attention_mask=padded).last_hidden_state
        h_causal = causal(input_ids=input_ids, attention_mask=padded).last_hidden_state
    assert (h_bi[:, 2] - h_causal[:, 2]).abs().max() > 1e-3


# ---- end-to-end with the real checkpoint -----------------------------------------


@pytest.mark.slow
def test_real_checkpoint_embeddings():
    """Smoke test against the published model (~14 GB download, 32 GB+ GPU)."""
    ext = LLM2VecExtractor(batch_size=2)
    embeddings = ext.transform(["SELECT * FROM users WHERE id = 1", "admin' OR 1=1 --", ""])
    assert embeddings.shape == (3, EMBED_DIM)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert ext.tokenizer_.padding_side == "left"
