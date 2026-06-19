"""
2x2 panel of representative CoT examples (OpenTSLM-Mamba-370m, the SSM model) — success + failure
for each CoT stage (HAR top, Sleep bottom). Failures are plausible adjacent-class confusions.
Outputs: assets/tslm_cot_examples.png + paper/figs/tslm_cot_examples.pdf
"""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREEN = "#1b7837"; GREEN_BG = "#e8f5e9"; RED = "#c62828"; RED_BG = "#ffebee"; DARK = "#263238"

# (task, ok?, pred, gold, CoT text, why-note-or-None)
CELLS = [
    ("HAR-CoT", True, "sitting", "sitting",
     "The accelerometer data over the 2.56 s window shows relatively low variability and "
     "consistent patterns across the X, Y, Z axes. The lack of significant peaks or troughs "
     "indicates no rapid or intense motion — characteristic of a stationary, low-intensity "
     "activity where the body stays mostly fixed with minor adjustments.  →  Answer: sitting.",
     None),
    ("HAR-CoT", False, "sitting", "lying",
     "Stable readings across all three axes after an initial transient. The low variability "
     "and stability indicate a lack of intense or repetitive motion, aligning with a "
     "stationary, low-intensity activity. Therefore the activity is sitting.  →  Answer: sitting.",
     "sitting & lying are both stationary, low-variability signals — the reasoning is sound "
     "but accelerometer magnitude can't separate the two postures (lying had the lowest HAR per-class F1)."),
    ("Sleep-CoT", True, "Wake", "Wake",
     "The signal shows frequent fluctuations with noticeable amplitude variability, suggesting "
     "heightened brain activity. The irregularity and complexity are associated with a more "
     "active brain state — consistent with an alert, conscious condition rather than deep "
     "sleep.  →  Answer: Wake",
     None),
    ("Sleep-CoT", False, "Non-REM stage 2", "Non-REM stage 1",
     "Rhythmic oscillations with moderate amplitude and frequency, associated with reduced "
     "consciousness and not as high-amplitude as alert states. The regularity and stability "
     "of the brain's electrical activity fit a sleep stage of reduced consciousness.  →  "
     "Answer: Non-REM stage 2.",
     "N1 & N2 are adjacent NREM stages with overlapping EEG — correct 'reduced consciousness' "
     "read, but landed one stage off (the classic sleep-scoring error)."),
]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), dpi=200)
for ax, (task, ok, pred, gold, cot, why) in zip(axes.flat, CELLS):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    acc, bg = (GREEN, GREEN_BG) if ok else (RED, RED_BG)
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.01,rounding_size=0.03",
                                linewidth=1.6, edgecolor=acc, facecolor=bg, mutation_aspect=0.6))
    tag = "✓ SUCCESS" if ok else "✗ FAILURE"
    ax.text(0.04, 0.90, f"{task}  ·  {tag}", fontsize=12.5, fontweight="bold", color=acc, va="center")
    ax.text(0.96, 0.90, f"pred: {pred}   |   gold: {gold}", fontsize=9.5, color=DARK, va="center", ha="right")
    ax.plot([0.04, 0.96], [0.83, 0.83], color=acc, lw=0.8, alpha=0.5)
    body = "\n".join(textwrap.fill(line, 74) for line in cot.split("\n"))
    ax.text(0.04, 0.78, body, fontsize=9.2, color="#1c1c1c", va="top", ha="left", wrap=True)
    if why:
        wy = "Why: " + textwrap.fill(why, 84).replace("\n", "\n     ")
        ax.text(0.04, 0.17, wy, fontsize=8.2, color=RED, va="top", ha="left", style="italic")

fig.suptitle("Mamba-370m chain-of-thought — success & failure per CoT task",
             fontsize=14, fontweight="bold", color=DARK, y=0.99)
fig.text(0.5, 0.005, "Failures are plausible adjacent-class confusions (sitting↔lying; NREM stage 1↔2), "
         "not nonsense reasoning.", ha="center", fontsize=8, color="#546e7a")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig("assets/tslm_cot_examples.png", bbox_inches="tight", facecolor="white")
fig.savefig("paper/figs/tslm_cot_examples.pdf", bbox_inches="tight", facecolor="white")
print("written: assets/tslm_cot_examples.png + paper/figs/tslm_cot_examples.pdf")
