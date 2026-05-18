#!/usr/bin/env python3
"""
Push the locally-generated Ahri parquet shards to a HuggingFace Hub dataset.

Layout on the Hub:
    {repo_id}/data/{task_id}/{split}.parquet
    {repo_id}/data/{task_id}/manifest.json
    {repo_id}/README.md   (auto-generated index)

This treats the dataset as a static drop. Re-running with a new local
generation will overwrite. We always upload all four splits for every task
we find on disk.

Auth: requires `huggingface-cli login` or HF_TOKEN env var.

Usage:
    PYTHONPATH=src python scripts/ahri/push_to_hub.py \\
        --local data/ahri --repo your-org/ahri-v1 --private
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from opentslm.ahri.tasks import TASK_REGISTRY


_README_TEMPLATE = """---
license: mit
language:
- en
size_categories:
- 100K<n<1M
---

# Ahri: Ascending Harmonic Reasoning Instruction (v{version})

A controlled evaluation framework for Time-Series Language Models.
21 synthetic tasks across 5 tiers, parametric signals at fs=200 Hz, N=1024.

## Layout

```
data/
  {{task_id}}/
    manifest.json
    train.parquet
    val.parquet
    test_in.parquet
    test_held.parquet
```

`test_held` covers the held-out parameter regions defined in the paper
(Table 1); `test_in` covers the rest of the parameter space.

## Tasks

{task_table}

## Loading

```python
from opentslm.ahri.dataset import AhriParquetDataset
ds = AhriParquetDataset.from_hub("{repo_id}", task_id="1.2", split="train")
```

See the [Ahri package](https://github.com/StanfordBDHG/OpenTSLM) for the
companion code (PhysicsTSLM model, trainers, eval harness).
"""


def _task_table() -> str:
    rows = ["| Task | Tier | Format | Question |", "|---|---|---|---|"]
    for tid in sorted(TASK_REGISTRY.keys(), key=lambda s: tuple(int(x) for x in s.split("."))):
        t = TASK_REGISTRY[tid]()
        rows.append(f"| {tid} | {t.tier} | {t.output_format} | {t.question} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="data/ahri", help="Local dataset root")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. org/ahri-v1")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--token", default=None, help="HF token (else uses cached login)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    local = Path(args.local).resolve()
    if not local.exists():
        raise FileNotFoundError(f"{local} does not exist — generate first.")

    api = HfApi(token=args.token)
    if not args.dry_run:
        create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True, token=args.token)

    # README
    readme = _README_TEMPLATE.format(
        version=args.version,
        repo_id=args.repo,
        task_table=_task_table(),
    )
    (local / "README.md").write_text(readme)

    files = []
    for task_dir in sorted(local.iterdir()):
        if not task_dir.is_dir():
            continue
        for f in sorted(task_dir.glob("*")):
            if f.suffix in (".parquet", ".json"):
                rel = f.relative_to(local)
                files.append((f, f"data/{rel.as_posix()}"))

    files.append((local / "README.md", "README.md"))

    print(f"Uploading {len(files)} files to {args.repo} (private={args.private})")
    for src, dest in files:
        size_mb = src.stat().st_size / 1e6
        print(f"  {dest:60s}  {size_mb:8.2f} MB")
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=dest,
                repo_id=args.repo,
                repo_type="dataset",
                token=args.token,
            )
    print(f"[done] https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
