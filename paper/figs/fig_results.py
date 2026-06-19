"""
Table-1 results as medal tables (codebase_f1, paper-consistent): accuracy table + macro-F1 table,
each with gold / silver / bronze shading for the top-3 models in every task column. Makes the
"SSM (Mamba) sweeps gold+silver on every task" pattern read at a glance.
Outputs: assets/tslm_acc.png/.pdf, assets/tslm_f1.png/.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GREEN = "#1b7837"; DARK = "#263238"
GOLD = "#ffd54f"; SILVER = "#e0e0e0"; BRONZE = "#e0b080"
MEDAL = [GOLD, SILVER, BRONZE]

TASKS = ["TSQA", "HAR", "Sleep"]
# (label, arch, [tsqa,har,sleep] acc, [...] macro-F1)
MODELS = [
    ("Mamba-1.4b",  "SSM+tok",  [0.9981, 0.7132, 0.7940], [0.9981, 0.6709, 0.6265]),
    ("Mamba-370m",  "SSM+tok",  [0.9967, 0.7018, 0.7479], [0.9967, 0.6626, 0.6075]),
    ("SP",          "attn+enc", [0.9240, 0.6982, 0.7328], [0.9249, 0.6500, 0.5844]),
    ("Llama+bins",  "attn+tok", [0.9419, 0.6773, 0.6620], [0.9425, 0.6239, 0.4863]),
    ("Flamingo",    "attn+enc", [0.9190, 0.6733, 0.6942], [0.9198, 0.5701, 0.4335]),
]


def medal_table(metric_idx, metric_name, outname):
    vals = [MODELS[r][2 + metric_idx] for r in range(len(MODELS))]   # rows x 3 tasks
    # rank per task column -> medal color (or None)
    medal = [[None] * 3 for _ in MODELS]
    for t in range(3):
        order = sorted(range(len(MODELS)), key=lambda r: vals[r][t], reverse=True)
        for rank, r in enumerate(order[:3]):
            medal[r][t] = MEDAL[rank]

    cols = ["Model", "Arch x Rep"] + TASKS
    cell_text = [[m[0], m[1]] + [f"{vals[r][t]:.3f}" for t in range(3)] for r, m in enumerate(MODELS)]
    cell_colors = [["white", "white"] + [medal[r][t] or "white" for t in range(3)] for r in range(len(MODELS))]

    fig, ax = plt.subplots(figsize=(7.8, 2.9), dpi=200); ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colors,
                   cellLoc="center", loc="center",
                   colWidths=[0.22, 0.20, 0.19, 0.19, 0.19])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfd8dc")
        if r == 0:
            cell.set_facecolor(DARK); cell.set_text_props(color="white", fontweight="bold")
        elif c >= 2 and medal[r - 1][c - 2] == GOLD:
            cell.set_text_props(fontweight="bold")
    ax.set_title(f"Table 1 — {metric_name} (codebase_f1, paper-consistent)",
                 fontsize=11, fontweight="bold", color=DARK, pad=14)
    ax.legend(handles=[Patch(facecolor=GOLD, edgecolor="#cfd8dc", label="1st"),
                       Patch(facecolor=SILVER, edgecolor="#cfd8dc", label="2nd"),
                       Patch(facecolor=BRONZE, edgecolor="#cfd8dc", label="3rd")],
              loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=8.5)
    fig.text(0.5, -0.10, "Medals = rank within each task. Mamba (SSM) takes gold + silver on every task.",
             ha="center", fontsize=7.4, color="#546e7a")
    fig.savefig(f"assets/{outname}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"paper/figs/{outname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"written: assets/{outname}.png")


medal_table(0, "accuracy", "tslm_acc")
medal_table(1, "macro-F1", "tslm_f1")
