"""
PhysicsTSLM — the encoder-free TSLM used for the Ahri paper.

Architecture (paper Section 4.1):
  - Patch embedding: input signal (N=1024) split into 32 non-overlapping
    patches of size P=32. Each patch is linearly projected to d_model.
  - Learned positional encoding added to each projected patch token.
  - The resulting 32 patch tokens are inserted in place of the <|signal|>
    placeholder tokens in the prompt and the prompt is run through the LLM
    via `inputs_embeds`. Standard self-attention does the rest.
  - LLM backbone: any HF causal LM (paper uses Pythia-160M / 410M / 1.4B).
    Fully fine-tuned.
  - Two-signal tasks: same projection applied to both signals; outputs are
    concatenated with a learned separator embedding (Section 4.2).

This follows the ELF pattern from ELM/elms/llm_encoders/base_elf.py: we
embed text, embed signals, and replace the signal placeholder positions in
the text embeddings with the projected patch embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from opentslm.ahri.prompts import N_PATCHES, PATCH_SIZE, SEP_TOKEN, SIGNAL_TOKEN
from opentslm.ahri.waveforms import N as SIGNAL_LEN


@dataclass
class PhysicsTSLMConfig:
    llm_id: str = "EleutherAI/pythia-410m"
    patch_size: int = PATCH_SIZE   # 32
    n_patches: int = N_PATCHES     # 32
    signal_len: int = SIGNAL_LEN   # 1024
    max_signals_per_example: int = 2
    use_sep_token: bool = True
    pad_token_id: int | None = None  # filled in from tokenizer


class PatchProjection(nn.Module):
    """Single learned Linear(P, d_model). One projection, shared across all
    signals and patches — exactly the 'single linear projection' described
    in the paper."""
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # signal: (B, S, L) where S = num signals, L = signal_len
        B, S, L = signal.shape
        P = self.proj.in_features
        assert L % P == 0, f"signal length {L} not divisible by patch size {P}"
        n_patches = L // P
        patches = signal.reshape(B, S, n_patches, P)         # (B, S, N, P)
        embeds = self.proj(patches)                          # (B, S, N, d)
        return embeds


class PhysicsTSLM(nn.Module):
    """Encoder-free TSLM. Takes (signals, prompt strings) and runs the LLM
    over a sequence whose <|signal|> placeholder tokens have had their
    embeddings replaced by the projected patch embeddings."""

    def __init__(self, config: PhysicsTSLMConfig):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_id)
        # Add a single signal placeholder token + optional separator.
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [SIGNAL_TOKEN] + ([SEP_TOKEN] if config.use_sep_token else [])}
        )
        # Pythia/GPT-NeoX use <|endoftext|> as both bos and eos; ensure pad set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.signal_token_id = self.tokenizer.convert_tokens_to_ids(SIGNAL_TOKEN)
        self.sep_token_id = (
            self.tokenizer.convert_tokens_to_ids(SEP_TOKEN) if config.use_sep_token else None
        )

        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_id)
        # resize embedding table to cover the new special tokens
        if added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        d_model = self.llm.config.hidden_size
        self.d_model = d_model

        self.patch_proj = PatchProjection(config.patch_size, d_model)
        # Learned positional embedding for patch index inside a signal block.
        # Each signal independently spans positions 0..n_patches-1. The text
        # "Signal 1:" / "Signal 2:" / <|signal_sep|> tokens do the cross-signal
        # ordering; the LLM's own positional encoding handles absolute position.
        self.patch_pos_emb = nn.Parameter(torch.zeros(config.n_patches, d_model))
        nn.init.normal_(self.patch_pos_emb, std=0.02)

        config.pad_token_id = self.tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # tokenizing
    # ------------------------------------------------------------------

    def tokenize(
        self,
        prompts: list[str],
        answers: list[str] | None = None,
        device: torch.device | str = "cpu",
    ) -> dict:
        """Tokenize a batch. If `answers` is provided, returns a labels tensor
        with prompt positions masked to -100 so the loss is over answer tokens
        only (standard prefix-LM setup)."""
        if answers is None:
            # eval: just tokenize prompts, left-pad so generation works
            self.tokenizer.padding_side = "left"
            tok = self.tokenizer(prompts, return_tensors="pt", padding=True)
            return {k: v.to(device) for k, v in tok.items()}

        # training: concat prompt + answer + eos, right-pad
        full_texts = [f"{p} {a}{self.tokenizer.eos_token}" for p, a in zip(prompts, answers)]
        self.tokenizer.padding_side = "right"
        tok = self.tokenizer(full_texts, return_tensors="pt", padding=True)
        input_ids = tok["input_ids"]
        attn = tok["attention_mask"]

        # mask labels for prompt-only tokens (and pad)
        prompt_ids = self.tokenizer(prompts, padding=False)["input_ids"]
        labels = input_ids.clone()
        for i, pid in enumerate(prompt_ids):
            labels[i, : len(pid)] = -100
        labels[attn == 0] = -100
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attn.to(device),
            "labels": labels.to(device),
        }

    # ------------------------------------------------------------------
    # signal injection
    # ------------------------------------------------------------------

    def _build_signal_tokens(self, signals: torch.Tensor) -> torch.Tensor:
        """Project + position-embed signal patches.

        signals: (B, S, L) -> (B, S * n_patches, d_model). The separator
        between two signals is a normal SEP_TOKEN in the prompt; its embed
        comes from the LLM's input embedding table.
        """
        B, S, L = signals.shape
        embeds = self.patch_proj(signals)               # (B, S, N, d)
        n = embeds.shape[2]
        pos = self.patch_pos_emb[:n].unsqueeze(0).unsqueeze(0)  # (1, 1, N, d)
        embeds = embeds + pos
        return embeds.reshape(B, S * n, -1)             # (B, S*N, d)

    def _inject_signal_embeds(
        self,
        input_ids: torch.Tensor,
        token_embeds: torch.Tensor,
        signal_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Replace embeddings at signal-placeholder positions with the
        projected patch embeddings. We use a per-row scatter rather than a
        global mask so it works regardless of where in the prompt the
        <|signal|> tokens appear.

        input_ids:     (B, T)
        token_embeds:  (B, T, d)
        signal_embeds: (B, K, d)   K = number of signal placeholders per row
        """
        B = input_ids.shape[0]
        out = token_embeds.clone()
        for b in range(B):
            mask = input_ids[b] == self.signal_token_id
            n_placeholders = int(mask.sum().item())
            n_provided = signal_embeds.shape[1]
            # Skip the separator slot when fewer placeholders exist than patch
            # tokens (i.e. tokenizer didn't represent the sep token specially).
            if n_placeholders != n_provided:
                # Should only happen if the prompt template and config disagree.
                raise ValueError(
                    f"Row {b}: {n_placeholders} <|signal|> placeholders but "
                    f"{n_provided} patch tokens to inject"
                )
            out[b, mask] = signal_embeds[b].to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # forward / generate
    # ------------------------------------------------------------------

    def forward(
        self,
        signals: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        signal_embeds = self._build_signal_tokens(signals)
        token_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = self._inject_signal_embeds(input_ids, token_embeds, signal_embeds)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        signals: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        signal_embeds = self._build_signal_tokens(signals)
        token_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = self._inject_signal_embeds(input_ids, token_embeds, signal_embeds)
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # housekeeping
    # ------------------------------------------------------------------

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
