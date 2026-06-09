"""
NUMERIC-COMPETENCE PRETRAINING (#5 — the dominant lever). Our sines-only synthetic produced a
WORSE-THAN-NAIVE real forecaster (CiK 0.454 > naive 0.379). Fix per Chronos (2403.07815) +
TSMamba (2411.02941): pretrain context-FREE (history->future) on KERNELSYNTH-style diverse
synthetic — GP samples from random compositions of {RBF, periodic, linear, constant} kernels.
Leakage-safe vs CiK by construction; far more diverse than sines. Then context-fine-tune on top.
"""
import argparse, os, sys, time, random, math, statistics as st
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_har import build_model
from dist_utils import dist_init, broadcast_params, sync_grads, is_main, cleanup


def kernelsynth(rng, T, jmax=4):
    """Sample one series ~ GP(0, K) where K is a random +/* composition of base kernels."""
    t = torch.linspace(0.0, 1.0, T)
    dd = (t[:, None] - t[None, :])
    def rbf(l): return torch.exp(-(dd * dd) / (2 * l * l))
    def per(p, l): return torch.exp(-2 * torch.sin(math.pi * dd.abs() / p) ** 2 / (l * l))
    Ks = []
    for _ in range(rng.randint(1, jmax)):
        c = rng.random()
        if c < 0.40: Ks.append(rbf(rng.uniform(0.05, 0.4)))
        elif c < 0.75: Ks.append(per(rng.uniform(0.08, 0.5), rng.uniform(0.3, 1.2)))
        elif c < 0.90: Ks.append((t[:, None] * t[None, :]) * rng.uniform(0.5, 2.0))   # linear
        else: Ks.append(torch.ones(T, T) * rng.uniform(0.3, 1.5))                      # constant
    K = Ks[0]
    for k in Ks[1:]:
        K = K + k if rng.random() < 0.6 else K * k
    K = K / (K.diagonal().mean() + 1e-6) + 1e-3 * torch.eye(T)
    try:
        L = torch.linalg.cholesky(K)
    except Exception:
        L = torch.linalg.cholesky(K + 1e-1 * torch.eye(T))
    return L @ torch.randn(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-id", default="state-spaces/mamba-1.4b-hf")
    ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pool", type=int, default=20000); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--serlen", type=int, default=256)
    ap.add_argument("--hist", type=int, default=96); ap.add_argument("--fut", type=int, default=32)
    ap.add_argument("--save", default="checkpoints/pretrain_ks_14.pt")
    ap.add_argument("--n-bins", type=int, default=256)
    ap.add_argument("--vrange", type=float, default=5.0)
    ap.add_argument("--scaling", default="zscore", choices=["zscore", "meanabs"])
    ap.add_argument("--soft-sigma", type=float, default=0.0, help="distance-aware CE width in bins (0=hard CE)")
    ap.add_argument("--seed", type=int, default=0, help="PAIRED seed: identical train pool+sampling AND test set across configs; +1 per replicate")
    args = ap.parse_args()
    rank, world, device = dist_init()
    torch.manual_seed(args.seed)                                  # PAIRED: identical train pool + window-sampling across configs at this seed

    if is_main():
        print(f"[pretrain {args.llm_id}] world={world} generating {args.pool} KernelSynth series/rank (len {args.serlen})...", flush=True)
    pool = [kernelsynth(random.Random(rank * 10_000_000 + i), args.serlen) for i in range(args.pool)]  # distinct data per rank
    pool_t = torch.stack(pool).to(device)                        # [N, serlen] on GPU -> fully-batched sampling
    W = args.hist + args.fut
    _off = torch.arange(W, device=device)

    m, params = build_model("mamba", args.llm_id, device, lora_r=args.lora_r, dtype=torch.bfloat16,
                            n_bins=args.n_bins, vrange=args.vrange, scaling=args.scaling)
    broadcast_params(params)                                      # identical start across ranks
    try:                                                         # checkpointing frees activation mem for big batch (deep models); harmless to math
        m.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.llm.config.use_cache = False
    except Exception as e:
        if is_main(): print(f"  [ckpt] gradient checkpointing unsupported, skipping: {str(e)[:80]}", flush=True)
    if is_main():
        print(f"  trainable={sum(p.numel() for p in params)/1e6:.1f}M steps={args.steps} bs={args.bs} "
              f"(eff bs={args.bs*world}) hist={args.hist} fut={args.fut}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2, fused=True); m.train(); t0 = time.time()
    for step in range(1, args.steps + 1):
        si = torch.randint(0, pool_t.shape[0], (args.bs,), device=device)
        s0 = torch.randint(0, args.serlen - W + 1, (args.bs,), device=device)
        w = pool_t[si].gather(1, s0[:, None] + _off[None, :])    # [bs, W] history+future windows (all on GPU)
        if args.scaling == "meanabs":                            # Chronos-style mean-abs scaling, no centering (matches eval)
            mu = w.new_zeros(w.shape[0], 1); sd = w[:, :args.hist].abs().mean(1, keepdim=True) + 1e-5
        else:                                                    # zscore (default)
            mu = w[:, :args.hist].mean(1, keepdim=True); sd = w[:, :args.hist].std(1, keepdim=True) + 1e-5
        ids = m._quantize_ids((w - mu) / sd)                     # [bs, W] bin-token ids (history-normalized = matches inference)
        lab = ids.clone(); lab[:, :args.hist] = -100             # teacher-forced loss only on future tokens
        emb = m.llm.get_input_embeddings()(ids)
        if args.soft_sigma > 0:                                  # distance-aware soft-ordinal CE on future bin tokens
            lg = m.llm(inputs_embeds=emb, return_dict=True).logits[:, :-1]      # logits at t predict token t+1
            tgt = ids[:, 1:]; msk = lab[:, 1:] != -100                           # future positions only
            lp = torch.log_softmax(lg.float(), dim=-1)[..., m.bin_token_ids]     # [bs, W-1, n_bins]
            bidx = m.tok2binidx[tgt].float()                                     # true bin index per target
            jj = torch.arange(m.n_bins, device=device).float()
            wgt = torch.exp(-((jj - bidx[..., None]) ** 2) / (2 * args.soft_sigma ** 2)); wgt = wgt / wgt.sum(-1, keepdim=True)
            loss = -(wgt * lp).sum(-1)[msk].mean()
        else:
            loss = m.llm(inputs_embeds=emb, labels=lab, return_dict=True).loss
        opt.zero_grad(); loss.backward(); sync_grads(params)     # all-reduce(AVG) grads across ranks
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 50 == 0 and is_main():
            print(f"  step{step}/{args.steps} loss {loss.item():.4f} ({(time.time()-t0)/step:.2f}s/it)", flush=True)
        if args.save and args.save != "none" and is_main() and step % 1500 == 0:  # periodic ckpt: crash never costs >1500 steps (flaky NCCL box)
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            torch.save({"cfg": {"model": "mamba", "llm_id": args.llm_id, "lora_r": args.lora_r}, "state": m.state_dict()}, args.save)
            print(f"  [ckpt] saved @ step{step} -> {args.save}", flush=True)
    if args.save and args.save != "none" and is_main():
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save({"cfg": {"model": "mamba", "llm_id": args.llm_id, "lora_r": args.lora_r},
                    "state": m.state_dict()}, args.save)
        print(f"saved -> {args.save}", flush=True)

    cleanup()                                                    # destroy PG FIRST: rank-0's solo eval below must not leave rank-1 waiting at a barrier (-> NCCL timeout/SIGABRT at run end)
    if is_main():
        m.eval(); torch.manual_seed(999000 + args.seed); mse, nv = [], []   # PAIRED test set: identical held-out KS across configs (disjoint from train seed)
        with torch.no_grad():
            for i in range(200):
                y = kernelsynth(random.Random(900000 + i), args.serlen)
                s = {"pre_prompt": "", "history": y[:args.hist], "future": y[args.hist:args.hist + args.fut], "post_prompt": ""}
                fc = m.forecast([s], horizon=args.fut)[0].float().cpu(); f = y[args.hist:args.hist + args.fut]
                mse.append(float(((fc - f) ** 2).mean()))
                nv.append(float(((y[args.hist - 1].repeat(args.fut) - f) ** 2).mean()))
        print(f"\n=== PRETRAIN [{args.llm_id}] held-out KernelSynth: MODEL MSE {st.mean(mse):.4f} vs naive {st.mean(nv):.4f} "
              f"(beats naive on {sum(a<b for a,b in zip(mse,nv))/len(mse)*100:.0f}%) ===", flush=True)


if __name__ == "__main__":
    main()
