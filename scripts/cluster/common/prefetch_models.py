#!/usr/bin/env python3
"""
Pre-fetch HuggingFace models to a local cache. Run this once on a node with
internet (login node) so compute nodes that may lack network can read the
weights from cache.

Usage:
    HF_HOME=$SCRATCH/hf_cache python scripts/cluster/common/prefetch_models.py
    # or with specific models:
    python scripts/cluster/common/prefetch_models.py --models EleutherAI/pythia-160m EleutherAI/pythia-1.4b
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--include-weights", action="store_true", default=True)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if cache_dir:
        print(f"Cache: {cache_dir}")
    for mid in args.models:
        print(f"\n[prefetch] {mid}")
        try:
            snapshot_download(
                repo_id=mid,
                cache_dir=cache_dir,
                allow_patterns=["*.json", "*.txt", "tokenizer*", "*.model"] + (["*.bin", "*.safetensors"] if args.include_weights else []),
            )
            print(f"  ok")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            sys.exit(1)

    print("\nAll models pre-fetched. Compute nodes can now load them offline.")


if __name__ == "__main__":
    main()
