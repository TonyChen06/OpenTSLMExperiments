"""
TSQA (OpenTSLM multiple-choice trend QA: volatility / trend / seasonality / outliers) train+eval.
Same model code path as run_har (build_model / collate / checkpointing) — swaps the dataset and uses a
letter-choice accuracy metric. A cleaner, lower-noise task than HAR-CoT: used to check whether the
signal representation matters when the task isn't fuzzy. Random baseline = 25% (4 choices).
"""
import argparse, os, sys, time, random, re
import torch
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate as collate,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_har import build_model   # reuse the exact model factory


def parse_choice(text: str) -> str:
    m = re.search(r"\(([a-dA-D])\)", text)
    return m.group(1).lower() if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba", choices=["mamba", "sp"])
    ap.add_argument("--llm-id", default="state-spaces/mamba-370m-hf")
    ap.add_argument("--train-n", type=int, default=8000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0, help="controls data subsample/order + init (for multi-seed variance)")
    ap.add_argument("--n-bins", type=int, default=256, help="quant: number of value-bin tokens")
    ap.add_argument("--vrange", type=float, default=5.0, help="quant: bins span [-vrange, vrange] in normalized space")
    ap.add_argument("--scaling", default="zscore", choices=["zscore", "meanabs"], help="signal normalization (zscore | Chronos-style meanabs)")
    ap.add_argument("--bin-mode", default="uniform", choices=["uniform", "quantile"], help="bin spacing: uniform (equal-width) | quantile (equal-mass under N(0,1))")
    ap.add_argument("--init-mode", default="numeracy", choices=["numeracy", "random"], help="bin-token embedding init: numeracy-grounded | random")
    args = ap.parse_args()
    device = "cuda"
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    torch.manual_seed(args.seed)

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device); c = ck["cfg"]
        m, params = build_model(c["model"], c["llm_id"], device, lora_r=c["lora_r"], dtype=dtype)
        m.load_state_dict(ck["state"], strict=False)
        print(f"[ckpt {args.ckpt}] eval-only test-n={args.test_n}", flush=True)
    else:
        m, params = build_model(args.model, args.llm_id, device, lora_r=args.lora_r, dtype=dtype,
                                n_bins=args.n_bins, vrange=args.vrange, scaling=args.scaling,
                                bin_mode=args.bin_mode, init_mode=args.init_mode)
        try:                                                 # frees activation mem; harmless to math
            m.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.llm.config.use_cache = False
        except Exception as e:
            print(f"  gradient checkpointing unsupported: {str(e)[:80]}", flush=True)
    eos = m.get_eos_token() or "</s>"
    te = TSQADataset("test", eos)
    random.seed(0)
    test = [te[i] for i in random.sample(range(len(te)), min(args.test_n, len(te)))]

    t0 = time.time()
    if not args.ckpt:
        random.seed(args.seed)                              # vary train subsample + shuffle per seed (test set stays fixed at seed 0)
        tr = TSQADataset("train", eos)
        train = [tr[i] for i in random.sample(range(len(tr)), min(args.train_n, len(tr)))]
        print(f"[{args.model} {args.llm_id}] TSQA scaling={args.scaling} bins={args.n_bins} "
              f"trainable={sum(p.numel() for p in params)/1e6:.1f}M train={len(train)} test={len(test)} "
              f"bs={args.bs} ep={args.epochs} lr={args.lr}", flush=True)
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2, fused=True)
        m.train(); step = 0
        for ep in range(args.epochs):
            random.shuffle(train)
            for i in range(0, len(train), args.bs):
                loss = m.compute_loss(collate(train[i:i + args.bs]))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
                if step % 100 == 0:
                    print(f"  ep{ep} step{step} loss {loss.item():.4f} ({(time.time()-t0)/step:.2f}s/it)", flush=True)
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            torch.save({"cfg": {"model": args.model, "llm_id": args.llm_id, "lora_r": args.lora_r},
                        "state": m.state_dict()}, args.save)
            print(f"saved -> {args.save}", flush=True)

    try:                                                     # re-enable cache + drop checkpointing for fast eval generation
        m.llm.gradient_checkpointing_disable(); m.llm.config.use_cache = True
    except Exception:
        pass
    m.eval()
    correct, per_task = 0, {}
    with torch.no_grad():
        for i in range(0, len(test), args.bs):
            raw = test[i:i + args.bs]
            gold = [parse_choice(s["answer"]) for s in raw]
            tasks = [s["post_prompt"].replace("Predict the ", "").replace(" Answer:", "") for s in raw]
            gen = m.generate(collate(raw), max_new_tokens=args.max_new)
            for g, gl, tk in zip(gen, gold, tasks):
                ok = int(parse_choice(g) == gl)
                correct += ok
                d = per_task.setdefault(tk, [0, 0]); d[0] += ok; d[1] += 1
    acc = correct / max(len(test), 1)
    print(f"\n=== RESULT [{args.model} {args.llm_id}] TSQA scaling={args.scaling} bins={args.n_bins} binmode={args.bin_mode} init={args.init_mode} vr={args.vrange}  "
          f"acc={acc:.4f}  (n_test={len(test)}, t={time.time()-t0:.0f}s) ===", flush=True)
    for tk, (c, n) in sorted(per_task.items()):
        print(f"   {tk}: {c/n:.3f} ({n})", flush=True)

    from datetime import datetime
    now = datetime.now()
    runs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(runs, exist_ok=True)
    short = args.llm_id.split("/")[-1]
    rec = {"timestamp": now.isoformat(timespec="seconds"), "task": "TSQA", "model": args.model,
           "llm_id": args.llm_id, "scaling": args.scaling, "n_bins": args.n_bins, "vrange": args.vrange,
           "bin_mode": args.bin_mode, "init_mode": args.init_mode, "seed": args.seed,
           "n_test": len(test), "acc": round(acc, 4), "ckpt": args.ckpt or args.save}
    path = os.path.join(runs, f"{now.strftime('%Y%m%d_%H%M%S')}_TSQA_{args.model}_{short}_n{len(test)}.yaml")
    with open(path, "w") as f:
        for k, v in rec.items():
            f.write(f"{k}: {'null' if v is None else v}\n")
    print(f"run record -> {path}", flush=True)


if __name__ == "__main__":
    main()
