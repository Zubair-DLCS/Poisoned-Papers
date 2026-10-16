import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import RES, FIG

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8f8e89"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False})
ATTACK_LABELS = {"naive": "Plain command\n(include)", "authority": "Fake note to\nscreening tools",
                 "polite": "Polite author\nstatement", "varied": "Ten other\nphrasings",
                 "relevance": "Relevance claim\n(no instruction)", "naive_out": "Plain command\n(exclude)",
                 "authority_out": "Fake 'duplicate'\nnote (exclude)"}
DEF_LABELS = {"none": "No defence", "warning": "Warning in\nthe prompt", "spotlight": "Warning +\nspotlighting"}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_guard():
    g = json.load(open(os.path.join(RES, "guard_summary.json")))
    order = ["naive", "authority", "polite", "naive_out", "authority_out", "varied", "relevance"]
    vals = [g["detection_rate"][a] for a in order]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    cols = [BLUE] * 5 + [ORANGE, ORANGE]
    ax.bar(range(len(order)), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    ax.set_xticks(range(len(order)), [ATTACK_LABELS[a] for a in order], fontsize=8.5)
    ax.set_ylim(0, 1.12); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylabel("Injected abstracts flagged")
    ax.axvline(4.5, color=GREY, lw=1, ls=":")
    ax.text(2, 1.07, "the attacks it was written for", ha="center", fontsize=9, color=INK2)
    ax.text(5.5, 1.07, "anything else", ha="center", fontsize=9, color=INK2)
    ax.set_title("A pattern filter catches the attacks it was written for, and nothing else")
    save(fig, "fig_guard.png")


def fig_attack_success():
    p = os.path.join(RES, "attack_success.csv")
    if not os.path.exists(p):
        return
    a = pd.read_csv(p)
    models = sorted(a.model.unique())
    order = ["naive", "authority", "polite", "varied", "relevance", "naive_out", "authority_out"]
    order = [o for o in order if o in set(a.attack)]
    defs = [d for d in ("none", "warning", "spotlight") if d in set(a.defence)]
    cmap = LinearSegmentedColormap.from_list("o", ["#fdf1ec", "#f6b79c", ORANGE, "#9c3a12"])
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models) + 1.6, 4.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        M = a[a.model == m].pivot(index="attack", columns="defence", values="success").reindex(index=order, columns=defs)
        ax.imshow(M.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                ax.text(j, i, "-" if np.isnan(v) else f"{v:.0%}", ha="center", va="center",
                        fontsize=10.5, fontweight="bold", color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(defs)), [DEF_LABELS[d] for d in defs], fontsize=8.5)
        ax.set_yticks(range(len(order)), [ATTACK_LABELS[o].replace("\n", " ") for o in order], fontsize=8.5)
        ax.grid(False); ax.set_title(m.split(":", 1)[-1], fontsize=10.5)
        for s in ax.spines.values():
            s.set_visible(False)
        if ax is not axes[0][0]:
            ax.set_yticklabels([])
    fig.suptitle("How often a hidden sentence flips the screening decision (attack success rate)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig_attack_success.png")


def fig_clean_cost():
    p = os.path.join(RES, "clean_performance.csv")
    if not os.path.exists(p):
        return
    c = pd.read_csv(p)
    models = sorted(c.model.unique())
    defs = [d for d in ("none", "warning", "spotlight") if d in set(c.defence)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    w = 0.8 / len(models)
    cols = [BLUE, AQUA, ORANGE, GREY]
    for ax, metric, title in ((axes[0], "recall", "Relevant papers kept (recall)"),
                              (axes[1], "specificity", "Irrelevant papers dropped (specificity)")):
        for k, m in enumerate(models):
            s = c[c.model == m].set_index("defence").reindex(defs)[metric]
            ax.bar(np.arange(len(defs)) + (k - (len(models) - 1) / 2) * w, s, width=w,
                   color=cols[k % 4], label=m.split(":", 1)[-1])
        ax.set_xticks(range(len(defs)), [DEF_LABELS[d] for d in defs], fontsize=8.5)
        ax.set_ylim(0, 1.05); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.set_title(title, fontsize=10.5)
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("What each defence costs on clean records", x=0.01, ha="left", fontweight="bold",
                 fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig_clean_cost.png")


if __name__ == "__main__":
    fig_guard(); fig_attack_success(); fig_clean_cost()
