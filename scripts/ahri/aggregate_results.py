#!/usr/bin/env python3
"""
Aggregate single-task summary.json files into one table.

Walks a results directory, finds every `summary.json`, and prints a
markdown table with one row per task showing:
    accuracy / in-dist accuracy / held-out accuracy / generalisation gap.

Usage:
    PYTHONPATH=src python scripts/ahri/aggregate_results.py results/ahri/pythia-160m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_summaries(root: Path) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for p in sorted(root.rglob("summary.json")):
        try:
            s = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        task = s.get("task") or p.parent.name
        out.append((task, s))
    return sorted(out, key=lambda x: tuple(int(n) for n in x[0].split(".")))


def format_table(rows: list[tuple[str, dict]]) -> str:
    out = [
        "| Task | Tier | n_in | acc_in | n_held | acc_held | gap |",
        "|------|------|------|--------|--------|----------|-----|",
    ]
    for task, s in rows:
        res = s.get("results", {}).get("test", {})
        tier = task.split(".")[0]
        n_in = res.get("n_in", 0)
        n_held = res.get("n_held", 0)
        acc_in = res.get("accuracy_in", 0.0)
        acc_held = res.get("accuracy_held", 0.0)
        gap = acc_in - acc_held
        out.append(
            f"| {task} | {tier} | {n_in:>4d} | {acc_in:6.1%} | {n_held:>4d} | {acc_held:6.1%} | {gap:+6.1%} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Results directory containing per-task subdirs")
    ap.add_argument("--out", default=None, help="Optional path to write the table to")
    args = ap.parse_args()

    root = Path(args.root)
    rows = find_summaries(root)
    if not rows:
        print(f"No summary.json files found under {root}")
        return

    print(f"Results for {root} ({len(rows)} tasks)\n")
    table = format_table(rows)
    print(table)

    # Headline stats
    accs_in = [s.get("results", {}).get("test", {}).get("accuracy_in", 0.0) for _, s in rows]
    accs_held = [s.get("results", {}).get("test", {}).get("accuracy_held", 0.0) for _, s in rows]
    solved_in = sum(1 for a in accs_in if a >= 0.95)
    solved_held = sum(1 for a in accs_held if a >= 0.95)
    print(f"\nMean in-dist accuracy:  {sum(accs_in)/len(accs_in):6.1%}")
    print(f"Mean held-out accuracy: {sum(accs_held)/len(accs_held):6.1%}")
    print(f"Tasks solved (>=95%):   in-dist {solved_in}/{len(rows)}, held-out {solved_held}/{len(rows)}")

    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
