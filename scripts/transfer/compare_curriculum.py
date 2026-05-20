#!/usr/bin/env python3
"""
Per-stage comparison between control and treatment curricula.

Reads:
  {root}/control/curriculum_summary.json
  {root}/treatment/curriculum_summary.json

Prints, for each downstream stage (HAR / Sleep / ECG), the test accuracy
in each run side by side plus the delta in percentage points. Verdict line
at the bottom.

Usage:
    PYTHONPATH=src python scripts/transfer/compare_curriculum.py results/transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DOWNSTREAM_STAGES = ["tsqa", "har_cot", "sleep"]


def _stage_acc(summary: dict, stage_name: str) -> tuple[float | None, int | None]:
    for s in summary.get("stages", []):
        if s.get("stage") == stage_name:
            tm = s.get("test_metrics") or {}
            return tm.get("accuracy"), tm.get("n")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Root containing control/ and treatment/ subdirs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)

    try:
        ctrl = json.loads((root / "control" / "curriculum_summary.json").read_text())
    except FileNotFoundError:
        print("[error] control/curriculum_summary.json not found — has the control run finished?")
        return
    try:
        treat = json.loads((root / "treatment" / "curriculum_summary.json").read_text())
    except FileNotFoundError:
        print("[error] treatment/curriculum_summary.json not found — has the treatment run finished?")
        return

    rows = ["| Stage   | Control | Treatment | Δ (pp) | Verdict |",
            "|---------|---------|-----------|--------|---------|"]
    deltas = []
    for stage in DOWNSTREAM_STAGES:
        ca, cn = _stage_acc(ctrl, stage)
        ta, tn = _stage_acc(treat, stage)
        ca_s = f"{ca*100:5.1f}% (n={cn})" if ca is not None else "  ?  "
        ta_s = f"{ta*100:5.1f}% (n={tn})" if ta is not None else "  ?  "
        if ca is None or ta is None:
            d_s, v = "  ?  ", "missing"
        else:
            d = (ta - ca) * 100
            deltas.append(d)
            d_s = f"{d:+5.1f}"
            v = "⬆ helps" if d > 0.5 else ("⬇ hurts" if d < -0.5 else "≈ flat")
        rows.append(f"| {stage:7s} | {ca_s} | {ta_s} | {d_s} | {v} |")
    table = "\n".join(rows)
    print(table)

    if deltas:
        mean_d = sum(deltas) / len(deltas)
        n_help = sum(1 for d in deltas if d > 0.5)
        print()
        print(f"Mean Δ across {len(deltas)} downstream stage(s): {mean_d:+.2f} pp")
        print(f"Helps on {n_help}/{len(deltas)} stage(s)")
        if mean_d > 1.0 and n_help >= 2:
            verdict = "PROMISING — write the paper"
        elif mean_d > 0.5 or n_help >= 2:
            verdict = "MILD POSITIVE — replicate with 3 seeds before claiming"
        elif mean_d < -0.5:
            verdict = "HURTS — AHRI pretraining is net negative; investigate"
        else:
            verdict = "NULL — no detectable effect; rethink framing or scope"
        print(f"\nHeadline verdict: {verdict}")

    # AHRI pretrain stage info (treatment only) — useful sanity check
    for s in treat.get("stages", []):
        if s.get("stage") == "ahri":
            print(f"\nAHRI pretrain (treatment): final epoch train_loss="
                  f"{s['history'][-1].get('train_loss', '?')} "
                  f"val_loss={s['history'][-1].get('val_loss', '?')} "
                  f"wall={s.get('stage_wall_minutes', '?'):.1f}m")

    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
