#!/usr/bin/env python3
"""
Compare control vs treatment (AHRI-pretrained) test accuracy across the
three downstream tasks.

Reads `{task}-control/summary.json` and `{task}-treatment/summary.json`
under the given results root and prints a markdown table:

    | task | control | treatment | Δ (pp) |

Plus headline stats: mean delta, whether the transfer benefit exists.

Usage:
    PYTHONPATH=src python scripts/transfer/compare_transfer.py results/transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ["tsqa", "har_cot", "sleep"]


def _load_summary(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _acc(summary: dict | None) -> float | None:
    if summary is None:
        return None
    tm = summary.get("test_metrics") or {}
    return tm.get("accuracy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Results root (contains pretrain-ahri/, {task}-control/, {task}-treatment/)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for task in TASKS:
        ctrl = _load_summary(root / f"{task}-control" / "summary.json")
        treat = _load_summary(root / f"{task}-treatment" / "summary.json")
        ctrl_acc = _acc(ctrl)
        treat_acc = _acc(treat)
        rows.append((task, ctrl_acc, treat_acc))

    table = ["| Task | Control | Treatment | Δ (pp) | Verdict |",
             "|------|---------|-----------|--------|---------|"]
    deltas = []
    for task, c, t in rows:
        cs = f"{c*100:5.1f}%" if c is not None else "  ?  "
        ts = f"{t*100:5.1f}%" if t is not None else "  ?  "
        if c is None or t is None:
            ds, v = "  ?  ", "missing summary"
        else:
            d = (t - c) * 100
            deltas.append(d)
            ds = f"{d:+5.1f}"
            v = ("⬆ helps" if d > 0.5 else "⬇ hurts" if d < -0.5 else "≈ flat")
        table.append(f"| {task} | {cs} | {ts} | {ds} | {v} |")

    table_str = "\n".join(table)
    print(table_str)
    if deltas:
        mean_d = sum(deltas) / len(deltas)
        n_help = sum(1 for d in deltas if d > 0.5)
        print()
        print(f"Mean Δ: {mean_d:+.2f} pp across {len(deltas)} task(s)")
        print(f"Helps on {n_help}/{len(deltas)} task(s)")
        verdict = "PROMISING — write the paper" if mean_d > 1.0 and n_help >= 2 else \
                  "NULL — transfer benefit not detected; redesign or shelve"
        print(f"\nHeadline verdict: {verdict}")

    if args.out:
        Path(args.out).write_text(table_str + "\n")


if __name__ == "__main__":
    main()
