"""
Validation-loss curves for ALL THREE tasks (TSQA stage-1, HAR-CoT stage-3, Sleep-CoT stage-4),
all 5 Table-1 models, identical 50-epoch protocol. 3 panels, LINEAR y from 0 so the flat tails read
honestly. Data: my box from results/<model>/<stage>/checkpoints/loss_history.txt; the other box's
TSQA from imports/, its HAR/Sleep from results_logs/boxA_*.txt (same epoch/train/val format).
Dots mark each model's best epoch. Outputs: assets/tslm_valloss.png + paper/figs/tslm_valloss.pdf
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN = "#1b7837"; DGREEN = "#66bb6a"; GRAY = "#546e7a"; PURPLE = "#8e24aa"; ORANGE = "#e65100"
DARK = "#263238"

# (label, color, linewidth)
MODELS = [
    ("Mamba-1.4b", GREEN, 2.4),
    ("Mamba-370m", DGREEN, 2.0),
    ("SP", GRAY, 2.0),
    ("Llama+bins", PURPLE, 2.0),
    ("Flamingo", ORANGE, 2.0),
]

# panel title, {model: loss_history path}, random-baseline (or None)
PANELS = [
    ("Stage-1 TSQA", {
        "Mamba-1.4b": "imports/TSQA_Mamba1.4/loss_history.txt",
        "Mamba-370m": "results/mamba_370m_hf/MambaTSLM/stage1_mcq/checkpoints/loss_history.txt",
        "SP":         "imports/TSQA_SP/loss_history.txt",
        "Llama+bins": "results/Llama_3_2_1B/MambaTSLM/stage1_mcq/checkpoints/loss_history.txt",
        "Flamingo":   "results/Llama_3_2_1B/OpenTSLMFlamingo/stage1_mcq/checkpoints/loss_history.txt",
    }, math.log(3) / 3.0),
    ("Stage-3 HAR-CoT", {
        "Mamba-1.4b": "results_logs/boxA_mamba1p4b_har.txt",
        "Mamba-370m": "results/mamba_370m_hf/MambaTSLM/stage3_cot/checkpoints/loss_history.txt",
        "SP":         "results_logs/boxA_sp_har.txt",
        "Llama+bins": "results/Llama_3_2_1B/MambaTSLM/stage3_cot/checkpoints/loss_history.txt",
        "Flamingo":   "results/Llama_3_2_1B/OpenTSLMFlamingo/stage3_cot/checkpoints/loss_history.txt",
    }, None),
    ("Stage-4 Sleep-CoT", {
        "Mamba-1.4b": "results_logs/boxA_mamba1p4b_sleep.txt",
        "Mamba-370m": "results/mamba_370m_hf/MambaTSLM/stage4_sleep_cot/checkpoints/loss_history.txt",
        "SP":         "results_logs/boxA_sp_sleep.txt",
        "Llama+bins": "results/Llama_3_2_1B/MambaTSLM/stage4_sleep_cot/checkpoints/loss_history.txt",
        "Flamingo":   "results/Llama_3_2_1B/OpenTSLMFlamingo/stage4_sleep_cot/checkpoints/loss_history.txt",
    }, None),
]


def load(path):
    eps, val = [], []
    for line in open(path):
        p = line.strip().split("\t")
        if len(p) >= 3 and p[0].isdigit():
            eps.append(int(p[0])); val.append(float(p[2]))
    return eps, val


COLOR = {m: c for m, c, _ in MODELS}
LW = {m: w for m, _, w in MODELS}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=200)
for ax, (title, paths, rand) in zip(axes, PANELS):
    for label, color, lw in MODELS:
        path = paths.get(label)
        if not path:
            continue
        try:
            eps, val = load(path)
        except FileNotFoundError:
            continue
        if not eps:
            continue
        ax.plot(eps, val, "-", color=color, lw=lw, label=label)
        bi = val.index(min(val))
        ax.scatter([eps[bi]], [val[bi]], s=36, color=color, zorder=5, edgecolor="white", lw=0.8)
    if rand is not None:
        ax.axhline(rand, ls="--", lw=1.4, color=DARK, alpha=0.7,
                   label=f"random = {rand:.2f}")
    ax.set_title(title, fontsize=11, fontweight="bold", color=DARK)
    ax.set_xlabel("epoch", fontsize=9.5)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=1)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("validation loss", fontsize=10)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.06),
           frameon=False, fontsize=9)
fig.suptitle("Validation loss across all tasks — identical 50-epoch protocol (patience-5)",
             fontsize=12.5, fontweight="bold", color=DARK, y=1.02)
fig.text(0.5, -0.15, "Dots mark each model's best epoch; TSQA dashed line = random 3-way chance. On "
         "TSQA the SSM+token models converge far lowest. For the CoT tasks (HAR/Sleep), validation "
         "loss is teacher-forced cross-entropy over the full reasoning text and DIVERGES from answer "
         "macro-F1 (Table 1): a model can match the reference reasoning closely yet pick the wrong "
         "final label, so SP's lower CoT text-loss does not translate to higher F1 — the Mamba models "
         "still lead the answer metric.",
         ha="center", fontsize=7.4, color=GRAY, wrap=True)
fig.tight_layout()
fig.savefig("assets/tslm_valloss.png", bbox_inches="tight", facecolor="white")
fig.savefig("paper/figs/tslm_valloss.pdf", bbox_inches="tight", facecolor="white")
print("written: assets/tslm_valloss.png + paper/figs/tslm_valloss.pdf")
