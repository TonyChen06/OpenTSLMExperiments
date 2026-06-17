"""Compute acc + macro-F1 the way the OpenTSLM codebase's eval scripts do (paper-consistent):
  tsqa  -> first-3-chars label, per-class macro-F1 (evaluation/opentslm/tsqa)
  har   -> evaluation/opentslm/parse_predictions.py  (HARCoTQADataset labels)
  sleep -> evaluation/opentslm/sleep/parse_sleep_cot_data.py (canonicalized sleep stages, 4->3 merge)
Usage: codebase_f1.py <tsqa|har|sleep> <test_predictions.jsonl>  ->  prints 'RESULT acc=.. macroF1=..'"""
import sys, json, io, contextlib, re
task, path = sys.argv[1], sys.argv[2]
def grab(out):
    a = re.search(r"Accuracy: ([0-9.]+)%", out); f = re.search(r"Macro-F1 Score: ([0-9.]+)", out)
    return (float(a.group(1)) / 100 if a else float("nan"), float(f.group(1)) if f else float("nan"))
if task == "tsqa":
    # The tsqa parser runs a hardcoded-file script at import (no __main__ guard, exit(1)s),
    # so we can't import it. Exec ONLY its function-def prefix to get the GENUINE
    # calculate_f1_score / calculate_f1_stats, then run its exact main() harness:
    # first-3-char labels, allowed_labels = the set of gold classes.
    SRC = "evaluation/opentslm/tsqa/parse_predictions.py"
    pre = open(SRC).read().split("# Path to your JSONL file")[0]  # imports + the two funcs only
    ns = {}; exec(compile(pre, SRC, "exec"), ns)
    correct, data_points = 0, []
    for l in open(path):
        l = l.strip()
        if not l: continue
        e = json.loads(l)
        gen = e.get("generated", "").strip()[:3]; gold = e.get("gold", "").strip()[:3]
        if gen == gold: correct += 1            # canonical acc = raw first-3 equality
        data_points.append(ns["calculate_f1_score"](gen, gold))
    allowed = {d["ground_truth_normalized"] for d in data_points}
    stats = ns["calculate_f1_stats"](data_points, allowed_labels=allowed)
    print(f"RESULT acc={correct/len(data_points):.4f} macroF1={stats['macro_f1']:.4f}")
elif task == "har":
    sys.path.insert(0, "evaluation/opentslm"); import parse_predictions as P
    with contextlib.redirect_stdout(io.StringIO()) as b: P.parse_rtf_jsonl(path)
    acc, f1 = grab(b.getvalue()); print(f"RESULT acc={acc:.4f} macroF1={f1:.4f}")
elif task == "sleep":
    sys.path.insert(0, "evaluation/opentslm/sleep"); import parse_sleep_cot_data as P
    with contextlib.redirect_stdout(io.StringIO()) as b: P.parse_sleep_cot_jsonl(path)
    acc, f1 = grab(b.getvalue()); print(f"RESULT acc={acc:.4f} macroF1={f1:.4f}")
