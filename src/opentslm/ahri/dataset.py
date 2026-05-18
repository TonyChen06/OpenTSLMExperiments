"""
PyTorch Dataset that reads pre-generated Ahri parquet shards.

The trainer / evaluator never instantiates a task to sample; it loads the
parquet produced by `ahri.generate`. This is the "treat dataset generation
as a one-time thing" contract.

Three loading modes:
  - local path:           AhriParquetDataset("data/ahri", "1.2", "train")
  - HuggingFace Hub repo: AhriParquetDataset.from_hub("org/ahri-v1", "1.2", "train")
  - manually pre-cached:  set HF_DATASETS_CACHE / AHRI_DATA_ROOT env vars
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]


class AhriParquetDataset(Dataset):
    """Loads one (task_id, split) parquet into memory.

    Memory footprint: 6k examples * 1024 floats * 4 bytes ~= 25 MB per
    single-signal task. Easily fits for the training set; two-signal tasks
    are ~50 MB. We pre-materialize signal tensors at load time.
    """

    def __init__(self, root: str | Path, task_id: str, split: Split):
        self.root = Path(root)
        self.task_id = task_id
        self.split = split
        self.task_dir = self.root / task_id
        self.manifest = json.loads((self.task_dir / "manifest.json").read_text())
        path = self.task_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} — run `python -m opentslm.ahri.generate` first")
        table = pq.read_table(path)
        # signals: list[list[list[float]]] — (N, num_signals, 1024)
        sigs = table.column("signals").to_pylist()
        self.signals = [
            torch.tensor(np.asarray(s, dtype=np.float32)) for s in sigs
        ]
        self.prompts = table.column("prompt").to_pylist()
        self.answers = table.column("answer").to_pylist()
        self.held = table.column("held").to_pylist()
        self.params_json = table.column("params").to_pylist()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "signals": self.signals[idx],          # (num_signals, 1024) float32
            "prompt": self.prompts[idx],
            "answer": self.answers[idx],
            "held": self.held[idx],
            "task_id": self.task_id,
        }


def load_all_splits(root: str | Path, task_id: str) -> dict[str, AhriParquetDataset]:
    return {s: AhriParquetDataset(root, task_id, s) for s in ("train", "val", "test")}


# ----------------------------------------------------------------------
# HuggingFace Hub support
# ----------------------------------------------------------------------

def _hub_download_task(repo_id: str, task_id: str, cache_dir: str | None = None, token: str | None = None) -> Path:
    """Materialize a single task's four parquet files + manifest.json from
    a HF dataset repo to local disk, returning the local task dir.

    Uses snapshot_download for atomic per-file caching. On offline nodes,
    set HF_HUB_OFFLINE=1 and the local cache is used.
    """
    from huggingface_hub import snapshot_download
    repo_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=cache_dir,
        token=token,
        allow_patterns=[f"data/{task_id}/*"],
    )
    return Path(repo_path) / "data" / task_id


def from_hub(repo_id: str, task_id: str, split: Split, cache_dir: str | None = None, token: str | None = None) -> "AhriParquetDataset":
    """Convenience wrapper: download (or hit cache) and construct the
    Dataset rooted at the cache. Sets AHRI_DATA_ROOT in os.environ for any
    downstream code that wants the resolved root path."""
    task_dir = _hub_download_task(repo_id, task_id, cache_dir=cache_dir, token=token)
    # task_dir is .../<cache>/data/<task_id>; AhriParquetDataset wants root such that root/<task_id>/...
    root = task_dir.parent
    os.environ.setdefault("AHRI_DATA_ROOT", str(root))
    return AhriParquetDataset(root, task_id, split)


# attach as a classmethod for ergonomic discovery
AhriParquetDataset.from_hub = staticmethod(from_hub)


def resolve_root(maybe_path_or_repo: str, cache_dir: str | None = None, token: str | None = None) -> str:
    """Disambiguate between a local path and a HF repo id.

    - If `maybe_path_or_repo` exists on disk, return it.
    - If it looks like an HF repo id (`org/name`), download it and return
      the resolved cache root.
    - If $AHRI_DATA_ROOT is set, use that.
    """
    p = Path(maybe_path_or_repo)
    if p.exists():
        return str(p)
    if os.environ.get("AHRI_DATA_ROOT"):
        return os.environ["AHRI_DATA_ROOT"]
    if "/" in maybe_path_or_repo and not maybe_path_or_repo.startswith("/"):
        # treat as HF repo id; fetch ALL tasks
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=maybe_path_or_repo,
            repo_type="dataset",
            cache_dir=cache_dir,
            token=token,
            allow_patterns=["data/*/*"],
        )
        return str(Path(local) / "data")
    raise FileNotFoundError(f"Could not resolve dataset root: {maybe_path_or_repo}")
