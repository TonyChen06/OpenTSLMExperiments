"""Eval harness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from opentslm.ahri.dataset import AhriParquetDataset
from opentslm.ahri.eval import bootstrap_ci, evaluate
from opentslm.model.llm.PhysicsTSLM import PhysicsTSLM, PhysicsTSLMConfig


def test_bootstrap_ci_all_correct_is_one():
    lo, hi = bootstrap_ci(np.ones(100))
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_all_wrong_is_zero():
    lo, hi = bootstrap_ci(np.zeros(100))
    assert lo == 0.0 and hi == 0.0


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    vals = rng.binomial(1, 0.7, size=200).astype(float)
    lo, hi = bootstrap_ci(vals, n_resamples=1000, rng=rng)
    assert lo < vals.mean() < hi


def test_bootstrap_ci_empty_returns_zeros():
    assert bootstrap_ci(np.array([])) == (0.0, 0.0)


@pytest.fixture(scope="module")
def model():
    cfg = PhysicsTSLMConfig(llm_id="EleutherAI/pythia-70m")
    return PhysicsTSLM(cfg)


def test_evaluate_returns_well_formed_result(model, tiny_data_root):
    ds = AhriParquetDataset(tiny_data_root, "1.2", "test")
    res = evaluate(model, ds, batch_size=2, max_new_tokens=4, device="cpu")
    assert res.n == len(ds)
    assert 0.0 <= res.accuracy <= 1.0
    assert res.accuracy_ci[0] <= res.accuracy_ci[1]
    assert len(res.per_example) == res.n
    # in/held split should cover all examples
    assert res.n_in + res.n_held == res.n
    for r in res.per_example:
        assert "correct" in r
        assert "prediction" in r
        assert "held" in r
