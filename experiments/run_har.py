"""
Neutral single-task HAR-CoT train+eval harness for the SSM-vs-transformer TS-LLM study.
Same code path / data / budget / metric for every model (swap --model). Fair by construction.

This is experiment tooling, not part of the OpenTSLM core: it builds a MambaTSLM (or OpenTSLM-SP)
directly and runs a single HAR-CoT stage so representation/training knobs can be A/B'd quickly,
outside the full curriculum. The shared `build_model` factory here is reused by run_tsqa.py /
run_pretrain.py.
"""
import argparse, os, time, random
import torch
from sklearn.metrics import f1_score
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate as collate,  # OpenTSLM's own collate; identical preprocessing for every model = fair
)

LABELS = HARCoTQADataset.get_labels()


def parse_label(text: str) -> str:
    t = text.lower()
    if "answer:" in t:
        t = t.rsplit("answer:", 1)[1]
    for lab in sorted(LABELS, key=len, reverse=True):       # longest first: walking_down before walking
        if lab in t or lab.replace("_", " ") in t:
            return lab
    return "UNK"


def build_model(kind, llm_id, device, lora_r=0, dtype=torch.bfloat16,
                n_bins=256, vrange=5.0, scaling="zscore", bin_mode="uniform", init_mode="numeracy"):
    """Shared model factory for the experiment harnesses. MambaTSLM uses the quant value-bin
    representation (the rep-study's chosen default); scaling/bins/init stay configurable for the
    representation ablations."""
    if kind == "mamba":
        from opentslm.model.llm.MambaTSLM import MambaTSLM
        m = MambaTSLM(llm_id=llm_id, device=device, dtype=dtype, lora_r=lora_r,
                      n_bins=n_bins, vrange=vrange, scaling=scaling, bin_mode=bin_mode, init_mode=init_mode)
    elif kind == "sp":
        from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
        m = OpenTSLMSP(llm_id=llm_id, device=device)
        m.enable_lora()
    else:
        raise ValueError(kind)
    params = [p for p in m.parameters() if p.requires_grad]
    return m, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba", choices=["mamba", "sp"])
    ap.add_argument("--llm-id", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank for mamba (0=full finetune)")
    ap.add_argument("--fp32", action="store_true", help="use fp32 (stable for mamba-ssm fused scan)")
    ap.add_argument("--save", default=None, help="save trained model (state+cfg) here for reuse (train once, eval many)")
    ap.add_argument("--ckpt", default=None, help="load a saved model and EVAL ONLY (skip training) -> re-eval on any test-n cheaply")
    ap.add_argument("--seed", type=int, default=0, help="paired: fixed test subsample (seed 0) + train subsample/init at this seed")
    ap.add_argument("--n-bins", type=int, default=256)
    ap.add_argument("--vrange", type=float, default=5.0)
    ap.add_argument("--scaling", default="zscore", choices=["zscore", "meanabs"])
    ap.add_argument("--bin-mode", default="uniform", choices=["uniform", "quantile"])
    ap.add_argument("--init-mode", default="numeracy", choices=["numeracy", "random"])
    args = ap.parse_args()
    device = "cuda"
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    torch.manual_seed(args.seed)

    if args.ckpt:                                              # EVAL-ONLY: load a saved model, skip training
        ck = torch.load(args.ckpt, map_location=device)
        c = ck["cfg"]
        m, params = build_model(c["model"], c["llm_id"], device, lora_r=c["lora_r"], dtype=dtype)
        m.load_state_dict(ck["state"], strict=False)
        print(f"[ckpt {args.ckpt}] {c['model']} {c['llm_id']} -> eval-only on test-n={args.test_n}", flush=True)
    else:
        m, params = build_model(args.model, args.llm_id, device, lora_r=args.lora_r, dtype=dtype,
                                n_bins=args.n_bins, vrange=args.vrange, scaling=args.scaling,
                                bin_mode=args.bin_mode, init_mode=args.init_mode)
    eos = m.get_eos_token() or "</s>"
    te = HARCoTQADataset(split="test", EOS_TOKEN=eos)
    random.seed(0)
    test = [te[i] for i in random.sample(range(len(te)), min(args.test_n, len(te)))]

    t0 = time.time()
    if not args.ckpt:
        random.seed(args.seed)                                 # vary train subsample/order per seed (test stays fixed at seed 0)
        tr = HARCoTQADataset(split="train", EOS_TOKEN=eos)
        train = [tr[i] for i in random.sample(range(len(tr)), min(args.train_n, len(tr)))]
        print(f"[{args.model} {args.llm_id}] trainable={sum(p.numel() for p in params)/1e6:.1f}M | "
              f"train={len(train)} test={len(test)} bs={args.bs} ep={args.epochs} lr={args.lr} scaling={args.scaling}", flush=True)
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
        m.train()
        step = 0
        for ep in range(args.epochs):
            random.shuffle(train)
            for i in range(0, len(train), args.bs):
                loss = m.compute_loss(collate(train[i:i + args.bs]))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
                if step % 50 == 0:
                    print(f"  ep{ep} step{step} loss {loss.item():.4f} ({(time.time()-t0)/step:.2f}s/it)", flush=True)
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            torch.save({"cfg": {"model": args.model, "llm_id": args.llm_id, "lora_r": args.lora_r},
                        "state": m.state_dict()}, args.save)
            print(f"saved -> {args.save}", flush=True)

    m.eval()
    preds, gts = [], []
    with torch.no_grad():
        for i in range(0, len(test), args.bs):
            raw = test[i:i + args.bs]
            gt_batch = [str(s["label"]).lower() for s in raw]   # label from raw sample
            gen = m.generate(collate(raw), max_new_tokens=args.max_new)
            for g, gt in zip(gen, gt_batch):
                preds.append(parse_label(g)); gts.append(gt)
    macro = f1_score(gts, preds, labels=LABELS, average="macro", zero_division=0)
    micro = f1_score(gts, preds, labels=LABELS, average="micro", zero_division=0)
    acc = sum(p == g for p, g in zip(preds, gts)) / max(len(gts), 1)
    print(f"\n=== RESULT [{args.model} {args.llm_id}] HAR-CoT scaling={args.scaling} bins={args.n_bins} vr={args.vrange} seed={args.seed}  macroF1={macro:.4f}  microF1={micro:.4f}  "
          f"acc={acc:.4f}  (n_test={len(gts)}, train_t={time.time()-t0:.0f}s) ===", flush=True)

    # --- persist a YAML run record (config + metrics) so no result is ever lost ---
    from datetime import datetime
    now = datetime.now()
    rec = {
        "timestamp": now.isoformat(timespec="seconds"), "task": "HAR-CoT",
        "mode": "eval-only" if args.ckpt else "train",
        "model": args.model, "llm_id": args.llm_id, "scaling": args.scaling,
        "n_bins": args.n_bins, "lora_r": args.lora_r, "fp32": args.fp32,
        "train_n": 0 if args.ckpt else args.train_n, "epochs": 0 if args.ckpt else args.epochs,
        "bs": args.bs, "lr": args.lr, "n_test": len(gts),
        "macro_f1": round(macro, 4), "micro_f1": round(micro, 4), "acc": round(acc, 4),
        "train_t_s": round(time.time() - t0), "ckpt": args.ckpt or args.save,
    }
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    short = args.llm_id.split("/")[-1]
    path = os.path.join(runs_dir, f"{now.strftime('%Y%m%d_%H%M%S')}_{args.model}_{short}_n{len(gts)}.yaml")
    with open(path, "w") as f:
        for k, v in rec.items():
            f.write(f"{k}: {'null' if v is None else v}\n")
    print(f"run record -> {path}", flush=True)


if __name__ == "__main__":
    main()
