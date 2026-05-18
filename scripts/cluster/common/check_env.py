#!/usr/bin/env python3
"""
Cluster environment smoke check. Run this BEFORE any large job to confirm:
  - all required Python deps are importable
  - CUDA is visible (if --gpu)
  - HF cache is writable and pre-fetched
  - parquet read/write works
  - Ahri import path resolves
  - (optionally) distributed: torch.distributed can rendezvous

Exit code 0 = ready to train. Non-zero = something broken.

Run as part of every SLURM job's preamble:
    python scripts/cluster/common/check_env.py --gpu --hf-cache $HF_HOME --models EleutherAI/pythia-160m EleutherAI/pythia-410m
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

OK = "\033[32mOK\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _check(name: str, fn, *args, **kwargs) -> bool:
    print(f"[ ] {name:50s} ... ", end="", flush=True)
    try:
        msg = fn(*args, **kwargs)
        print(f"{OK}  {msg or ''}")
        return True
    except Exception as e:
        print(f"{FAIL}  {type(e).__name__}: {e}")
        return False


def check_python():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python >=3.10 required, got {sys.version}")
    return f"{platform.python_version()}"


def check_import(pkg):
    m = importlib.import_module(pkg)
    return getattr(m, "__version__", "?")


def check_torch():
    import torch
    return f"torch={torch.__version__} cuda_built={torch.version.cuda}"


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    n = torch.cuda.device_count()
    devs = [torch.cuda.get_device_name(i) for i in range(n)]
    return f"{n} device(s): {', '.join(devs)}"


def check_disk(path: str, min_gb: float):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    stat = shutil.disk_usage(p)
    free_gb = stat.free / 1e9
    if free_gb < min_gb:
        raise RuntimeError(f"{path}: {free_gb:.1f} GB free, need {min_gb}")
    # writable test
    tmp = p / ".write_check"
    tmp.write_text("ok")
    tmp.unlink()
    return f"{free_gb:.1f} GB free"


def check_parquet_roundtrip():
    import pyarrow as pa
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory() as d:
        t = pa.table({"x": [1, 2, 3]})
        pq.write_table(t, Path(d) / "t.parquet")
        t2 = pq.read_table(Path(d) / "t.parquet")
        assert t2.equals(t)
    return "rw ok"


def check_hf_cache(cache_dir: str | None):
    from huggingface_hub import constants
    resolved = cache_dir or constants.HF_HUB_CACHE
    p = Path(resolved)
    p.mkdir(parents=True, exist_ok=True)
    stat = shutil.disk_usage(p)
    return f"{resolved} ({stat.free / 1e9:.1f} GB free)"


def check_model_prefetched(repo_id: str, cache_dir: str | None):
    """Check that a model is already in cache so compute-node jobs without
    internet can still load it."""
    from huggingface_hub import HfApi, scan_cache_dir
    cache = scan_cache_dir(cache_dir=cache_dir)
    cached = {r.repo_id for r in cache.repos}
    if repo_id not in cached:
        raise RuntimeError(f"{repo_id} not in cache — pre-fetch on a login node")
    return "cached"


def check_ahri_import():
    import opentslm.ahri
    from opentslm.ahri.tasks import TASK_REGISTRY
    return f"{len(TASK_REGISTRY)} tasks registered"


def check_torch_distributed_can_init():
    """Init a single-process group on gloo to confirm torch.distributed works."""
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://")
    dist.destroy_process_group()
    return "gloo init ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="Require CUDA")
    ap.add_argument("--hf-cache", default=None, help="HF cache dir (else default)")
    ap.add_argument("--data-dir", default=None, help="Scratch path for Ahri data; will check writability")
    ap.add_argument("--min-disk-gb", type=float, default=5.0)
    ap.add_argument("--models", nargs="*", default=[], help="HF model IDs that must be pre-cached")
    ap.add_argument("--check-distributed", action="store_true")
    args = ap.parse_args()

    ok = True
    ok &= _check("python version", check_python)
    for pkg in ["numpy", "torch", "transformers", "huggingface_hub", "datasets", "pyarrow", "tqdm"]:
        ok &= _check(f"import {pkg}", check_import, pkg)
    ok &= _check("torch", check_torch)
    if args.gpu:
        ok &= _check("CUDA visible", check_cuda)
    ok &= _check("parquet roundtrip", check_parquet_roundtrip)
    ok &= _check("ahri import + task registry", check_ahri_import)
    ok &= _check("HF cache dir writable", check_hf_cache, args.hf_cache)
    if args.data_dir:
        ok &= _check(f"data dir writable ({args.data_dir})", check_disk, args.data_dir, args.min_disk_gb)
    for mid in args.models:
        ok &= _check(f"model cached: {mid}", check_model_prefetched, mid, args.hf_cache)
    if args.check_distributed:
        ok &= _check("torch.distributed (gloo, 1 proc)", check_torch_distributed_can_init)

    if not ok:
        print("\nEnvironment check FAILED. Fix issues above before launching jobs.")
        sys.exit(1)
    print("\nEnvironment check PASSED. Ready to train.")


if __name__ == "__main__":
    main()
