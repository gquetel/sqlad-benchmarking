"""LLM2Vec-Mistral-7B embedding feature extractor.

Maps SQL queries to 4096-d dense embeddings using
``McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp`` (Behnamghader et al., 2024).

The upstream ``llm2vec`` package pins ``transformers>=4.43.1,<=4.44.2`` because it
subclasses attention classes (``MistralSdpaAttention``, ``MistralFlashAttention2``)
and re-implements ``_update_causal_mask``, none of which survive the transformers 5.x
attention refactor. Rather than keep a second environment, this module reimplements
the *inference* path of ``llm2vec`` 0.2.3 against today's transformers. It is a port,
not a redesign: every step below mirrors the reference, quirks included. It was diffed
against the real package and produced bit-identical embeddings -- the harness that
proves it is at the bottom of this file, and ``tests/test_llm2vec.py`` guards the
invariants it established.

The pipeline, as the reference performs it:

1. Truncate the query to ``doc_max_length`` tokens by dropping *words*
   (``LLM2Vec._convert_to_str``).
2. Wrap it as ``"[INST] <query> [/INST]"``. The reference does this because the model
   config's ``_name_or_path`` resolves to ``mistralai/Mistral-7B-Instruct-v0.2`` --
   the MNTP repo ships no base weights, so transformers redirects to the base repo
   named in ``adapter_config.json`` and that name selects the ``[INST]`` branch of
   ``LLM2Vec.prepare_for_tokenization``.
3. Tokenize the wrapped text with left padding, plus a second ``embed_mask`` marking
   only the ``"<query> [/INST]"`` suffix; ``skip_instruction`` then pools over that
   mask, so ``<s>`` and the ``"[INST]"`` prefix are excluded from the mean.
4. Run Mistral with *bidirectional* attention (the LLM2Vec contribution) and the MNTP
   LoRA merged in, then mean-pool the last ``embed_mask.sum()`` positions.

Embeddings are **not** L2-normalized, matching the reference.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer, MistralModel
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from sqlad_benchmarking.determinism import enable_determinism

logger = logging.getLogger(__name__)

# The MNTP repo holds only the LoRA adapter; the base weights live in the repo its
# adapter_config.json points at. We load both explicitly instead of relying on the
# transformers-side adapter redirect the reference happened to trigger.
DEFAULT_MODEL = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_BATCH_SIZE = 16
EMBED_DIM = 4096

# llm2vec 0.2.3 defaults (the MNTP repo ships no llm2vec_config.json to override them).
MAX_LENGTH = 512
DOC_MAX_LENGTH = 400

# Instruction template applied by LLM2Vec.prepare_for_tokenization for
# mistralai/Mistral-7B-Instruct-v0.2. Only the suffix is pooled over.
INSTRUCT_PREFIX = "[INST] "
INSTRUCT_SUFFIX = " [/INST]"


def _as_queries(X) -> list[str]:  # noqa: N803
    if isinstance(X, pd.DataFrame):
        return X["full_query"].astype(str).tolist()
    return [str(q) for q in X]


class MistralBiModel(MistralModel):
    """Mistral with the causal mask removed, i.e. every token attends to every token.

    Replaces the reference's ``MistralBiModel``, which overrode ``_update_causal_mask``
    to emit an all-zero (padding-only) mask. Transformers 5.x builds the same thing via
    :func:`~transformers.masking_utils.create_bidirectional_mask`, so the override here
    is just swapping which mask factory feeds the decoder layers.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # create_bidirectional_mask returns None when nothing is padded, and SDPA then
        # falls back to the module's own is_causal flag -- which would silently make the
        # unpadded batches causal. The reference clears the same flag for this reason.
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Passing a ready-made 4D mask makes the parent's own mask creation a no-op.
        bidirectional_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=bidirectional_mask,
            use_cache=False,
            **kwargs,
        )


class LLM2VecExtractor(BaseEstimator, TransformerMixin):
    """Sklearn transformer producing 4096-d LLM2Vec-Mistral embeddings.

    ``transform`` returns an ``(n_samples, 4096)`` float32 ndarray. The model is
    pretrained so ``fit`` is a no-op, and loading is deferred to first use to keep
    ``build_extractor`` cheap (the checkpoint is ~14 GB in bfloat16 and needs a 32 GB+
    GPU). Disk caching is layered on externally via
    :class:`~sqlad_benchmarking.features.cache.CachingExtractor`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_model_name: str = DEFAULT_BASE_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # bfloat16 is the checkpoint's native dtype and what the reference loads; float32
        # exists so the parity test can compare without bfloat16 rounding noise.
        self.dtype = dtype

    # ---- loading ----------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load tokenizer + base model with the MNTP LoRA merged in."""
        if getattr(self, "model_", None) is not None:
            return
        # Imported lazily so peft is only needed when llm2vec is actually used.
        from peft import PeftModel

        enable_determinism()
        torch.manual_seed(2)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Mistral has no pad token, and every pooling mode below assumes left padding.
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer_ = tokenizer

        model = MistralBiModel.from_pretrained(self.base_model_name, dtype=self.dtype)
        # Merging (rather than keeping the adapter live) matches the reference and drops
        # the LoRA branches from the forward pass.
        model = PeftModel.from_pretrained(model, self.model_name)
        model = model.merge_and_unload()
        self.model_ = model.to(self.device).eval()

    # ---- text preparation --------------------------------------------------------

    def _n_tokens(self, text: str) -> int:
        """Token count as the reference measures it: no special tokens, capped at ``MAX_LENGTH``."""
        return len(
            self.tokenizer_(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                add_special_tokens=False,
            )["input_ids"][0]
        )

    def _truncate_to_doc_max_length(self, text: str) -> str:
        """Drop trailing *words* until the text fits ``DOC_MAX_LENGTH`` tokens.

        Ported verbatim from ``LLM2Vec._convert_to_str``: it estimates how many words to
        keep from the token-count ratio and repeats until under budget, so the result
        depends on the loop, not on a single tokenizer truncation. Note the ratio is
        computed from a token count that ``_n_tokens`` already clipped to ``MAX_LENGTH``,
        which makes the loop cut far less aggressively than the raw length would.
        """
        n_tokens = self._n_tokens(text)
        while n_tokens > DOC_MAX_LENGTH:
            reduction_ratio = DOC_MAX_LENGTH / n_tokens
            reduced_length = int(len(text.split()) * reduction_ratio)
            text = " ".join(text.split()[:reduced_length])
            n_tokens = self._n_tokens(text)
        return text

    def _split_query(self, query: str) -> tuple[str, str, str]:
        """Return ``(truncated_query, full_text, pooled_suffix)`` for one query.

        The reference joins instruction and text with a ``"!@#$%^&*()"`` sentinel purely
        to remember where the pooled part starts, then splits it back out. With an empty
        instruction that reduces to the two strings returned here; the ``rstrip`` is the
        reference's ``.strip()`` on the sentinel-joined string, whose leading ``"!"``
        makes it right-side only.
        """
        truncated = self._truncate_to_doc_max_length(query)
        body = truncated.rstrip() + INSTRUCT_SUFFIX
        return truncated, INSTRUCT_PREFIX + body, body

    def _tokenize(self, prepared: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of ``(full_text, pooled_suffix)`` pairs and build the ``embed_mask``."""
        full_texts, suffixes = zip(*prepared, strict=True)
        features = self.tokenizer_(
            list(full_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # The suffix is re-tokenized standalone (as the reference does) and its length
        # taken from the right; with left padding that lands on the suffix positions.
        embed_mask = torch.zeros_like(features["attention_mask"])
        for i, suffix in enumerate(suffixes):
            n_suffix = self._n_tokens(suffix)
            if n_suffix > 0:
                embed_mask[i, -n_suffix:] = 1
        features["embed_mask"] = embed_mask
        return features

    # ---- encoding ----------------------------------------------------------------

    def _pool(self, last_hidden_state: torch.Tensor, embed_mask: torch.Tensor) -> torch.Tensor:
        """Mean over the pooled suffix positions, in the model dtype (as the reference does)."""
        seq_lengths = embed_mask.sum(dim=-1)
        return torch.stack(
            [last_hidden_state[i, -length:, :].mean(dim=0) for i, length in enumerate(seq_lengths)],
            dim=0,
        )

    def _encode_batch(self, prepared: list[tuple[str, str]]) -> torch.Tensor:
        features = self._tokenize(prepared)
        embed_mask = features.pop("embed_mask")
        features = {k: v.to(self.device) for k, v in features.items()}
        with torch.no_grad():
            out = self.model_(**features)
        return self._pool(out.last_hidden_state, embed_mask.to(self.device)).detach().cpu()

    def _encode(self, queries: list[str]) -> np.ndarray:
        split = [self._split_query(q) for q in queries]

        # Length-sorted batching (reference behaviour): groups similar lengths together
        # so batches carry less padding. The reference sorts on the truncated query with
        # the sentinel prepended; that is a constant offset, so it sorts identically.
        # Order is restored before returning.
        order = np.argsort([-len(truncated) for truncated, _, _ in split])
        prepared = [(split[i][1], split[i][2]) for i in order]

        chunks = [
            self._encode_batch(prepared[start : start + self.batch_size])
            for start in range(0, len(prepared), self.batch_size)
        ]
        embeddings = torch.cat(chunks, dim=0)[np.argsort(order)]
        return embeddings.to(torch.float32).numpy()

    # ---- sklearn API -------------------------------------------------------------

    def fit(self, X, y=None) -> "LLM2VecExtractor":  # noqa: N803
        return self

    def transform(self, X) -> np.ndarray:  # noqa: N803
        queries = _as_queries(X)
        if not queries:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        self._ensure_model()
        return self._encode(queries).astype(np.float32)


# ---------------------------------------------------------------------------------
# Parity harness: proof this port matches the real llm2vec, bit for bit.
#
# Kept commented out because running it needs a second environment -- upstream llm2vec
# pins transformers<=4.44.2 and this project runs 5.x, which is the whole reason the
# port exists. Uncomment into a scratch file to re-run after touching anything above.
#
# It compares against a *tiny* fixture model (2 layers, 64 hidden) laid out exactly like
# the real MNTP repo: a config, a LoRA adapter and the real tokenizer, but no base
# weights. That covers every code path -- the adapter redirect, the "[INST]" wrapping it
# triggers, tokenisation, embed_mask, bidirectional attention, LoRA merge, pooling,
# length-sorted batching -- with no 14 GB download and no 32 GB GPU.
#
# Last result: prepared text, input_ids, attention_mask and embed_mask identical on all
# probe queries; embeddings bit-identical at batch sizes 3 and 16, max abs diff 1.2e-07
# at batch size 2 (below the reference's own 4.8e-07 spread across batch sizes, which
# comes from how much padding each batch carries). A zero-padding batch -- where
# transformers skips mask construction and SDPA falls back to is_causal -- also matched
# bit for bit.
#
# --- setup ------------------------------------------------------------------------
# uv venv --python 3.12 /tmp/l2v-ref/.venv   # llm2vec does not support 3.14
# uv pip install --python /tmp/l2v-ref/.venv/bin/python \
#     torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
# uv pip install --python /tmp/l2v-ref/.venv/bin/python \
#     transformers==4.44.2 peft==0.12.0 llm2vec==0.2.3 accelerate==0.33.0 "numpy<2"
#
# --- probe queries (shared by both sides) -----------------------------------------
# Deliberately nasty: empty and whitespace-only input, text long enough to hit both the
# doc_max_length word truncation and the tokenizer's max_length, duplicates, unicode,
# and leading/trailing whitespace.
#
# QUERIES = [
#     "SELECT * FROM users WHERE id = 1",
#     "SELECT name, email FROM customers WHERE country = 'FR' ORDER BY name",
#     "admin' UNION SELECT username, password FROM users--",
#     "INSERT INTO logs (msg) VALUES ('héllo wörld — ünïcode')",
#     "", "   ", "\n\t", "SELECT 1",
#     "SELECT * FROM users WHERE id = 1",        # duplicate of the first
#     "  SELECT * FROM t WHERE a = 'b'  ",       # leading/trailing whitespace
#     "SELECT * FROM t WHERE " + " ".join(f"col_{i} = {i} AND" for i in range(600)) + " 1=1",
#     "SELECT " + ", ".join(f"t.field_{i}" for i in range(900)) + " FROM t",
#     "DROP TABLE students; --",
# ]
#
# --- step 1: build the fixture (reference env) ------------------------------------
# import json, os, torch
# from peft import LoraConfig, get_peft_model
# from transformers import AutoTokenizer, MistralConfig, MistralModel
#
# base_dir, mntp_dir = os.path.abspath("tiny-base"), os.path.abspath("tiny-mntp")
# tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
# cfg = MistralConfig(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128,
#                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
#                     max_position_embeddings=32768, rms_norm_eps=1e-5, rope_theta=1e6,
#                     sliding_window=None, tie_word_embeddings=False, bos_token_id=1,
#                     eos_token_id=2, attention_dropout=0.0)
# torch.manual_seed(0)
# base = MistralModel(cfg)
# for name, p in base.named_parameters():          # default init leaves norms at 1.0; a
#     if "norm" in name:                           # dropped RMSNorm would slip through
#         torch.nn.init.normal_(p, mean=1.0, std=0.05)
# base.eval().save_pretrained(base_dir); tok.save_pretrained(base_dir)
#
# peft_model = get_peft_model(base, LoraConfig(          # same shape as the real adapter
#     r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=None,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
# torch.manual_seed(1)
# for name, p in peft_model.named_parameters():    # lora_B inits to zero -> merging would
#     if "lora_B" in name:                         # be a no-op and test nothing
#         torch.nn.init.normal_(p, mean=0.0, std=0.02)
# peft_model.save_pretrained(mntp_dir); tok.save_pretrained(mntp_dir)
#
# cfg_dict = json.loads(open(f"{base_dir}/config.json").read())
# cfg_dict["_name_or_path"] = DEFAULT_BASE_MODEL   # as the real repo's config.json does;
# cfg_dict["architectures"] = ["MistralEncoderModel"]   # this is what selects "[INST]"
# json.dump(cfg_dict, open(f"{mntp_dir}/config.json", "w"), indent=2)
# adapter_cfg = json.loads(open(f"{mntp_dir}/adapter_config.json").read())
# adapter_cfg["base_model_name_or_path"] = base_dir
# json.dump(adapter_cfg, open(f"{mntp_dir}/adapter_config.json", "w"), indent=2)
#
# --- step 2: golden outputs from the real package (reference env) ------------------
# import numpy as np, torch
# from llm2vec import LLM2Vec
#
# l2v = LLM2Vec.from_pretrained("tiny-mntp", device_map="cpu", torch_dtype=torch.float32)
# assert l2v.model.config._name_or_path == DEFAULT_BASE_MODEL   # i.e. "[INST]" is applied
# assert (l2v.pooling_mode, l2v.max_length, l2v.doc_max_length, l2v.skip_instruction) \
#     == ("mean", 512, 400, True)
# feats = l2v.tokenize([l2v.prepare_for_tokenization(l2v._convert_to_str("", q)) for q in QUERIES])
# diag = {"prepared": [l2v.prepare_for_tokenization(l2v._convert_to_str("", q)) for q in QUERIES],
#         "input_ids": feats["input_ids"].tolist(),
#         "attention_mask": feats["attention_mask"].tolist(),
#         "embed_mask": feats["embed_mask"].tolist()}
# json.dump(diag, open("golden.json", "w"))
# np.savez("golden.npz", **{f"bs{bs}": l2v.encode(QUERIES, batch_size=bs, show_progress_bar=False,
#                                                 convert_to_numpy=True) for bs in (2, 3, 16)})
#
# --- step 3: diff the port against them (project env) ------------------------------
# float32, not the checkpoint's bfloat16: comparing in bfloat16 would hide real
# differences under rounding noise.
#
# import json, numpy as np, torch
#
# ref, gold = json.load(open("golden.json")), np.load("golden.npz")
# ext = LLM2VecExtractor(model_name="tiny-mntp", base_model_name="tiny-base",
#                        device=torch.device("cpu"), dtype=torch.float32)
# ext._ensure_model()
#
# split = [ext._split_query(q) for q in QUERIES]
# # The reference keeps a "!@#$%^&*()" sentinel in its prepared text purely to mark where
# # the pooled part starts; stripping it gives the text actually tokenised.
# assert [f for _, f, _ in split] == ["".join(p.split("!@#$%^&*()")) for p in ref["prepared"]]
# features = ext._tokenize([(f, s) for _, f, s in split])
# for key in ("input_ids", "attention_mask", "embed_mask"):
#     assert features[key].tolist() == ref[key], key
# for bs in (2, 3, 16):
#     ext.batch_size = bs
#     assert np.abs(ext.transform(QUERIES) - gold[f"bs{bs}"]).max() < 1e-5
