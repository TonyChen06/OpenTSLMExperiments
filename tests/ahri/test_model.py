"""PhysicsTSLM forward / backward / generate / DDP-compatibility."""

from __future__ import annotations

import os

import pytest
import torch

from opentslm.ahri.dataset import AhriParquetDataset
from opentslm.model.llm.PhysicsTSLM import PhysicsTSLM, PhysicsTSLMConfig


@pytest.fixture(scope="module")
def model():
    cfg = PhysicsTSLMConfig(llm_id="EleutherAI/pythia-70m")
    return PhysicsTSLM(cfg)


def test_special_tokens_added(model):
    assert model.signal_token_id is not None
    assert model.sep_token_id is not None
    assert model.signal_token_id != model.sep_token_id


def test_tokenize_train(model):
    prompts = ["Question: foo?\nAnswer:"]
    answers = ["bar"]
    tok = model.tokenize(prompts, answers)
    # labels should mask the prompt portion
    assert (tok["labels"] == -100).any()
    # at least one non-masked label (the answer + eos)
    assert (tok["labels"] != -100).any()


def test_forward_backward_single_signal(model, tiny_data_root):
    ds = AhriParquetDataset(tiny_data_root, "1.2", "train")
    sigs = torch.stack([ds[i]["signals"] for i in range(2)])
    prompts = [ds[i]["prompt"] for i in range(2)]
    answers = [ds[i]["answer"] for i in range(2)]
    tok = model.tokenize(prompts, answers)
    out = model(sigs, **tok)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    # at least one gradient was computed
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
    model.zero_grad()


def test_forward_backward_two_signal(model, tiny_data_root):
    ds = AhriParquetDataset(tiny_data_root, "3.1", "train")
    sigs = torch.stack([ds[i]["signals"] for i in range(2)])
    prompts = [ds[i]["prompt"] for i in range(2)]
    answers = [ds[i]["answer"] for i in range(2)]
    # placeholder count = 64 for two-signal prompts
    tok = model.tokenize(prompts, answers)
    n_placeholders = (tok["input_ids"] == model.signal_token_id).sum(dim=1)
    assert (n_placeholders == 64).all()
    out = model(sigs, **tok)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    model.zero_grad()


def test_generate(model, tiny_data_root):
    ds = AhriParquetDataset(tiny_data_root, "1.2", "test")
    sigs = torch.stack([ds[i]["signals"] for i in range(2)])
    prompts = [ds[i]["prompt"] for i in range(2)]
    tok = model.tokenize(prompts, None)
    gen = model.generate(sigs, **tok, max_new_tokens=4)
    assert gen.shape == (2, 4)
    decoded = model.tokenizer.batch_decode(gen, skip_special_tokens=True)
    assert len(decoded) == 2


def test_patch_projection_is_linear():
    """Single Linear(P, d_model), no nonlinearity per paper."""
    cfg = PhysicsTSLMConfig(llm_id="EleutherAI/pythia-70m")
    m = PhysicsTSLM(cfg)
    # patch projection should have exactly one weight + bias
    params = list(m.patch_proj.parameters())
    assert len(params) == 2  # weight + bias
    # shape: (d_model, P) and (d_model,)
    assert params[0].shape == (m.d_model, 32)
    assert params[1].shape == (m.d_model,)
