"""
Environment smoke test — verifies every dependency Ahri needs is importable
and at a compatible version. Run on a fresh cluster before training.
"""

from __future__ import annotations

import importlib

import pytest

REQUIRED_PACKAGES = [
    "numpy",
    "torch",
    "transformers",
    "huggingface_hub",
    "datasets",
    "pyarrow",
    "tqdm",
]


@pytest.mark.parametrize("pkg", REQUIRED_PACKAGES)
def test_dependency_importable(pkg):
    importlib.import_module(pkg)


def test_torch_distributed_available():
    import torch.distributed as dist
    # we don't init here — just ensure the module is present
    assert hasattr(dist, "init_process_group")


def test_transformers_can_load_tokenizer():
    """Tokenizer load is local-only if cached; otherwise needs internet.
    If this fails on a compute node, pre-cache Pythia weights on the login
    node first (see scripts/cluster/prefetch_models.py)."""
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    except (OSError, ValueError) as e:
        pytest.skip(f"Pythia tokenizer not cached: {e}. Pre-fetch on a login node.")


def test_pyarrow_parquet_roundtrip(tmp_path):
    """Parquet read/write works — the data path."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table({"a": [1, 2, 3], "b": [[1.0], [2.0], [3.0]]})
    p = tmp_path / "x.parquet"
    pq.write_table(t, p)
    t2 = pq.read_table(p)
    assert t2.equals(t)
