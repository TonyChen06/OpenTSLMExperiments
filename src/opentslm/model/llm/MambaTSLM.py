# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""
MambaTSLM — a fully-SSM time-series language model.

Contribution vs OpenTSLM-SP / OpenTSLM-Flamingo (which wrap a *transformer* LLM):
  * The backbone is a pretrained State-Space Model (Mamba), so the whole model is
    linear-time and processes one interleaved ``[text | signal | text | ...]`` stream with a
    constant-size recurrent state — no attention, no perceiver, no cross-attention.
  * The time series enters (and leaves) as TOKENS in the same stream as the text: each scalar
    value is quantized to one of ``n_bins`` dedicated value-bin vocab tokens (Chronos-style),
    with numeracy-grounded embedding init. No encoder, no projector, no regression head —
    "pure token in / pure token out". The SSM does all the temporal modelling.

Implements the same :class:`TimeSeriesLLM` interface (``compute_loss`` / ``generate`` /
``eval_prompt``) so it drops into the existing OpenTSLM ``CurriculumTrainer`` + F1 eval for an
apples-to-apples comparison. The ``forecast`` / ``forecast_loss`` methods add the "signal OUT"
half: the future is just more value-bin tokens the SSM continues the stream with, dequantized
back to data units.
"""
import os
from typing import Dict, List

import torch
import torch.nn as nn  # noqa: F401  (kept for parity with other model files / future submodules)
from transformers import AutoModelForCausalLM, AutoTokenizer

from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

from .TimeSeriesLLM import TimeSeriesLLM


class MambaTSLM(TimeSeriesLLM):
    """Fully-SSM time-series LM with a quantized value-bin signal representation.

    Args:
        llm_id: HF id of the Mamba backbone (e.g. ``state-spaces/mamba-370m-hf``).
        lora_r: LoRA rank for the backbone (0 = full fine-tune). When > 0 the backbone is
            frozen and LoRA adapters + the (untied) value-bin embeddings are trained.
        n_bins, vrange: number of value-bin tokens and the z-space range they span.
        scaling: per-series normalization, ``"zscore"`` (default) or ``"meanabs"``.
        bin_mode: ``"uniform"`` (equal-width, default) or ``"quantile"`` (equal-mass under N(0,1)).
        init_mode: ``"numeracy"`` (seed each bin embedding from the LLM's embedding of its
            center value's text, default) or ``"random"``.
    """

    def __init__(
        self,
        llm_id: str = "state-spaces/mamba-370m-hf",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        lora_r: int = 0,
        n_bins: int = 256,
        vrange: float = 5.0,
        scaling: str = "zscore",
        bin_mode: str = "uniform",
        init_mode: str = "numeracy",
    ):
        super().__init__(device)
        self.dtype = dtype
        self.scaling = scaling
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # sdpa (flash/mem-efficient self-attention) for ATTENTION backbones (e.g. Llama → the
        # attn+tokenized cell). Mamba/SSM backbones have no attention and reject the kwarg, so
        # only pass it for non-mamba. TSLM_ATTN_IMPL overrides.
        _kw = {}
        if "mamba" not in llm_id.lower():
            _kw["attn_implementation"] = os.environ.get("TSLM_ATTN_IMPL", "sdpa")
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id, dtype=dtype, **_kw).to(device)
        # Chronos-style scalar value-bin tokens (pure token in/out, head-free).
        self._init_value_bins(n_bins=n_bins, vrange=vrange, bin_mode=bin_mode, init_mode=init_mode)

        if lora_r > 0:
            # Match the OpenTSLM-SP protocol: freeze the backbone, train LoRA adapters.
            # Mamba block linear layers are in_proj / x_proj / dt_proj.
            from peft import LoraConfig, get_peft_model

            for p in self.llm.parameters():
                p.requires_grad = False
            mt = getattr(self.llm.config, "model_type", "")
            if mt in ("mamba", "falcon_mamba"):
                # pure Mamba-1 (state-spaces / Falcon-Mamba); peft rejects out_proj / conv1d.
                targets = ["in_proj", "x_proj", "dt_proj"]
            else:
                # hybrid / other backbones (e.g. falcon_h1): attn + mamba + MLP linears.
                cand = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "gate_proj", "up_proj", "down_proj"]
                present = {n.split(".")[-1] for n, _ in self.llm.named_modules()}
                targets = [c for c in cand if c in present]
            self.llm = get_peft_model(
                self.llm,
                LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=targets),
            )
            # The vocab was expanded with new value-bin tokens; they live in the (now-frozen)
            # embedding table, so plain LoRA would leave them stuck at init forever. Make the
            # input embeddings (+ untied lm_head) trainable so the new tokens can learn.
            emb = self.llm.get_input_embeddings()
            emb.weight.requires_grad = True
            out = self.llm.get_output_embeddings()
            if out is not None and out.weight is not emb.weight:
                out.weight.requires_grad = True

    # ---- value-bin tokenizer (the signal representation) -------------------------------
    def _init_value_bins(self, n_bins: int = 256, vrange: float = 5.0, bin_mode: str = "uniform", init_mode: str = "numeracy"):
        """Add ``n_bins`` scalar value-bin tokens to the vocab (one token per quantized value).

        bin_mode: ``"uniform"`` (Chronos-style, equal-WIDTH over [-vrange, vrange]) or
          ``"quantile"`` (equal-MASS under N(0,1) -> finer resolution near 0 where z-scored
          data is dense). init_mode: ``"numeracy"`` (seed each bin's embedding from the LLM's
          embedding of its center value's text, e.g. 1.23 <- embed of "1.23") or ``"random"``.
        """
        self.n_bins = n_bins
        if bin_mode == "quantile":  # equal-mass bins under N(0,1)
            q = torch.linspace(0.0, 1.0, n_bins + 1, device=self.device)
            self.bin_edges = (2.0 ** 0.5 * torch.erfinv(2.0 * q - 1.0)).clamp(-vrange, vrange)
        else:  # uniform (Chronos-style, default)
            self.bin_edges = torch.linspace(-vrange, vrange, n_bins + 1, device=self.device)
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        toks = [f"<v{i}>" for i in range(n_bins)]
        self.tokenizer.add_tokens(toks)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.bin_token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(toks), device=self.device)
        emb = self.llm.get_input_embeddings()
        if init_mode == "numeracy":  # "random" = keep the default resize_token_embeddings init
            with torch.no_grad():
                for i, c in enumerate(centers.tolist()):
                    ids = self.tokenizer(f"{c:.2f}", add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.device)
                    emb.weight[self.bin_token_ids[i]] = emb(ids).mean(0).to(emb.weight.dtype)
        # For signal OUTPUT (forecasting): map any generated bin-token id back to its center value.
        self.bin_centers = centers  # [n_bins] (z-space)
        self.deq_lookup = torch.full((emb.weight.shape[0],), float("nan"), device=self.device)
        self.deq_lookup[self.bin_token_ids] = centers  # vocab id -> center, nan if not a bin
        self.tok2binidx = torch.full((emb.weight.shape[0],), -1, dtype=torch.long, device=self.device)
        self.tok2binidx[self.bin_token_ids] = torch.arange(n_bins, device=self.device)  # vocab id -> bin index

    def _quantize_ids(self, ts: torch.Tensor) -> torch.Tensor:
        """z-scored signal -> bin-token ids [T] (one token per value)."""
        v = ts.clamp(self.bin_edges[0], self.bin_edges[-1] - 1e-4)
        idx = (torch.bucketize(v, self.bin_edges) - 1).clamp(0, self.n_bins - 1)
        return self.bin_token_ids[idx]

    def _quantize(self, ts: torch.Tensor) -> torch.Tensor:
        """z-scored signal -> bin-token embeddings [T, H] (one token per value)."""
        return self.llm.get_input_embeddings()(self._quantize_ids(ts))

    def _dequantize(self, ids: torch.Tensor) -> torch.Tensor:
        """bin-token ids -> z-space center values (nan for any non-bin token)."""
        return self.deq_lookup[ids]

    def _normalize(self, ts: torch.Tensor):
        """Per-series scaling -> (normalized, mu, sd); inverse is ``z * sd + mu``.

        ``"zscore"``  = (x - mean) / std  (centered, our default).
        ``"meanabs"`` = x / mean(|x|)     (Chronos-style mean-absolute scaling, no centering => mu=0).
        """
        if self.scaling == "meanabs":
            mu = ts.new_zeros(())
            sd = ts.abs().mean() + 1e-5
        else:  # zscore (default)
            mu = ts.mean()
            sd = ts.std() + 1e-5
        return (ts - mu) / sd, mu, sd

    # ---- one interleaved [text | signal | ...] embedding sequence per sample ----------
    def _embed_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        return self.llm.get_input_embeddings()(ids)[0]  # [L, H]

    def _prompt_embed(self, s: Dict) -> torch.Tensor:
        """One sample's interleaved prompt sequence [L, H] (no padding):
        ``pre_prompt | (text_i + signal_i)* | post_prompt``. Signals are scaled per series."""
        parts = [self._embed_text(s["pre_prompt"])]
        for tstext, ts in zip(s["time_series_text"], s["time_series"]):
            parts.append(self._embed_text(tstext))
            ts = torch.as_tensor(ts, dtype=torch.float32, device=self.device).flatten()
            ts, _, _ = self._normalize(ts)  # per-series scaling (zscore | meanabs)
            parts.append(self._quantize(ts))
        parts.append(self._embed_text(s["post_prompt"]))
        return torch.cat(parts, dim=0)  # [L, H]

    def _answer_ids(self, answer: str) -> torch.Tensor:
        return self.tokenizer([answer], add_special_tokens=False, truncation=True, return_tensors="pt").input_ids[0].to(self.device)  # [A]

    # ---- TimeSeriesLLM interface -------------------------------------------------------
    def compute_loss(self, batch: List[Dict]) -> torch.Tensor:
        """Batched, RIGHT-padded, no attention_mask. The fused mamba-ssm kernel produces nan
        gradients over LEFT-padded positions; with right-padding the (causal) answer never
        attends to the trailing pad and the backward never reaches it, so the loss and grads
        are exact while keeping full-batch throughput."""
        seqs, labels = [], []
        for s in batch:
            p = self._prompt_embed(s)  # [L, H]
            a_ids = self._answer_ids(s["answer"])  # [A]
            a_emb = self.llm.get_input_embeddings()(a_ids)  # [A, H]
            seqs.append(torch.cat([p, a_emb], dim=0))  # [L+A, H]
            labels.append(torch.cat([p.new_full((p.shape[0],), -100, dtype=torch.long), a_ids]))
        Lmax, H = max(x.shape[0] for x in seqs), seqs[0].shape[1]
        emb = torch.stack([torch.cat([x, x.new_zeros(Lmax - x.shape[0], H)]) for x in seqs])
        lab = torch.stack([torch.cat([y, y.new_full((Lmax - y.shape[0],), -100)]) for y in labels])
        return self.llm(inputs_embeds=emb, labels=lab, return_dict=True).loss

    @torch.no_grad()
    def generate(self, batch: List[Dict], max_new_tokens: int = 50, **kw) -> List[str]:
        """Batched, LEFT-padded prompts + attention_mask. Generation is forward-only (no
        backward), so the padding-backward nan can't occur."""
        embs = [self._prompt_embed(s) for s in batch]  # list of [L_i, H]
        Lmax, H = max(x.shape[0] for x in embs), embs[0].shape[1]
        emb = torch.stack([torch.cat([x.new_zeros(Lmax - x.shape[0], H), x]) for x in embs])
        mask = torch.stack([torch.cat([
            torch.zeros(Lmax - x.shape[0], device=self.device, dtype=torch.long),
            torch.ones(x.shape[0], device=self.device, dtype=torch.long)]) for x in embs])
        out = self.llm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new_tokens, **kw)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token

    def eval_prompt(self, prompt: FullPrompt, max_new_tokens: int = 1024, normalize: bool = False) -> str:
        self.eval()
        batch = extend_time_series_to_match_patch_size_and_aggregate([prompt.to_dict()], normalize=normalize)
        return self.generate(batch, max_new_tokens=max_new_tokens)[0]

    # ---- signal OUTPUT: forecast by generating value-bin tokens, then dequantize -------
    # The "signal out" half of pure-token in/out: the future is just more bin tokens the SSM
    # continues the stream with. History and future share the HISTORY's scaling stats.
    def _forecast_seq(self, s: Dict):
        """-> (input_embeds [L,H], future_bin_ids [F] or None, mu, sd) for one sample.
        Expects ``s['history']`` (1-D) and optionally ``s['future']`` (1-D), + pre/post_prompt text."""
        h = torch.as_tensor(s["history"], dtype=torch.float32, device=self.device).flatten()
        _, mu, sd = self._normalize(h)
        inp = torch.cat([
            self._embed_text(s.get("pre_prompt", "")),
            self._quantize((h - mu) / sd),
            self._embed_text(s.get("post_prompt", "")),
        ], dim=0)
        f_ids = None
        if s.get("future") is not None:
            f = torch.as_tensor(s["future"], dtype=torch.float32, device=self.device).flatten()
            f_ids = self._quantize_ids((f - mu) / sd)
        return inp, f_ids, mu, sd

    def forecast_loss(self, batch: List[Dict], soft_sigma: float = 0.0) -> torch.Tensor:
        """Teacher-forced loss on the FUTURE value-bin tokens (right-pad trick, as compute_loss).
        ``soft_sigma > 0`` => distance-aware soft-ordinal target: spread the target mass over
        neighbouring bins with a Gaussian of width ``soft_sigma`` bins (CRPS-friendlier than
        one-hot CE)."""
        seqs, labels = [], []
        for s in batch:
            inp, f_ids, _, _ = self._forecast_seq(s)
            f_emb = self.llm.get_input_embeddings()(f_ids)
            seqs.append(torch.cat([inp, f_emb], dim=0))
            labels.append(torch.cat([inp.new_full((inp.shape[0],), -100, dtype=torch.long), f_ids]))
        Lmax, H = max(x.shape[0] for x in seqs), seqs[0].shape[1]
        emb = torch.stack([torch.cat([x, x.new_zeros(Lmax - x.shape[0], H)]) for x in seqs])
        lab = torch.stack([torch.cat([y, y.new_full((Lmax - y.shape[0],), -100)]) for y in labels])
        if soft_sigma <= 0:
            return self.llm(inputs_embeds=emb, labels=lab, return_dict=True).loss
        logits = self.llm(inputs_embeds=emb, return_dict=True).logits[:, :-1, :]  # predict next token
        tgt = lab[:, 1:]
        pos = tgt != -100  # future-token positions
        if pos.sum() == 0:
            return logits.sum() * 0.0
        binidx = self.tok2binidx[tgt[pos]].float()  # [N] true bin index per future token
        lp_bins = torch.log_softmax(logits[pos].float(), dim=-1)[:, self.bin_token_ids]  # [N, n_bins]
        j = torch.arange(self.n_bins, device=self.device).float()
        w = torch.exp(-((j[None, :] - binidx[:, None]) ** 2) / (2 * soft_sigma ** 2))  # [N, n_bins]
        w = w / w.sum(-1, keepdim=True)
        return -(w * lp_bins).sum(-1).mean()

    @torch.no_grad()
    def forecast(self, batch: List[Dict], horizon: int, n_samples: int = 1, temperature: float = 1.0, caps=None) -> List[torch.Tensor]:
        """Generate ``horizon`` future values per sample (output constrained to bin tokens),
        dequantized + un-scaled back to data units. ``n_samples=1`` -> greedy point forecast,
        returns [horizon]; ``n_samples>1`` -> sampled (for CRPS), returns [n_samples, horizon].
        ``caps``: optional per-sample RAW upper bound (or None) -> mask out bins above it, so no
        sampled trajectory can violate a stated cap (structural constraint satisfaction)."""
        from transformers import LogitsProcessor

        embs, stats = [], []
        for s in batch:
            inp, _, mu, sd = self._forecast_seq(s)
            embs.append(inp)
            stats.append((mu, sd))
        Lmax, H = max(x.shape[0] for x in embs), embs[0].shape[1]
        emb = torch.stack([torch.cat([x.new_zeros(Lmax - x.shape[0], H), x]) for x in embs])  # left-pad
        mask = torch.stack([torch.cat([
            torch.zeros(Lmax - x.shape[0], device=self.device, dtype=torch.long),
            torch.ones(x.shape[0], device=self.device, dtype=torch.long)]) for x in embs])
        V = self.llm.get_input_embeddings().weight.shape[0]
        base = torch.full((len(batch), V), float("-inf"), device=self.device)  # per-sample allowed-bin mask
        for i, (mu, sd) in enumerate(stats):
            ok = torch.ones(self.n_bins, dtype=torch.bool, device=self.device)
            if caps is not None and caps[i] is not None:
                ok = ok & (self.bin_centers <= (caps[i] - mu) / sd)  # mask bins above the cap (z-space)
            if not ok.any():
                ok[0] = True
            base[i, self.bin_token_ids[ok]] = 0.0
        allow = base.repeat_interleave(n_samples, dim=0)  # [B*n_samples, V]

        class _Allow(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return scores + allow.to(scores.dtype)

        gk = dict(max_new_tokens=horizon, min_new_tokens=horizon, logits_processor=[_Allow()])
        if n_samples > 1:
            gk.update(do_sample=True, temperature=temperature, num_return_sequences=n_samples)
        else:
            gk.update(do_sample=False)
        out = self.llm.generate(inputs_embeds=emb, attention_mask=mask, **gk)  # [B*n_samples, horizon]
        res = []
        for i, (mu, sd) in enumerate(stats):
            rows = out[i * n_samples:(i + 1) * n_samples, -horizon:]  # [n_samples, horizon]
            vals = self._dequantize(rows) * sd + mu
            res.append(vals[0] if n_samples == 1 else vals)
        return res
