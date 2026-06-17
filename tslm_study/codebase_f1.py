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
    g, p = [], []
    for l in open(path):
        e = json.loads(l); p.append(e["generated"].strip()[:3].lower()); g.append(e["gold"].strip()[:3].lower())
    n = len(g); acc = sum(a == b for a, b in zip(p, g)) / n
    cls = {}
    for a, b in zip(p, g):
        cls.setdefault(b, {"tp": 0, "fp": 0, "fn": 0}); cls.setdefault(a, {"tp": 0, "fp": 0, "fn": 0})
        if a == b: cls[b]["tp"] += 1
        else: cls[b]["fn"] += 1; cls[a]["fp"] += 1
    f1s = []
    for c, d in cls.items():
        pr = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0
        rc = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    print(f"RESULT acc={acc:.4f} macroF1={sum(f1s)/len(f1s):.4f}")
elif task == "har":
    sys.path.insert(0, "evaluation/opentslm"); import parse_predictions as P
    with contextlib.redirect_stdout(io.StringIO()) as b: P.parse_rtf_jsonl(path)
    acc, f1 = grab(b.getvalue()); print(f"RESULT acc={acc:.4f} macroF1={f1:.4f}")
elif task == "sleep":
    sys.path.insert(0, "evaluation/opentslm/sleep"); import parse_sleep_cot_data as P
    with contextlib.redirect_stdout(io.StringIO()) as b: P.parse_sleep_cot_jsonl(path)
    acc, f1 = grab(b.getvalue()); print(f"RESULT acc={acc:.4f} macroF1={f1:.4f}")
