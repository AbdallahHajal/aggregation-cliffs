"""Methods Figure 1: the compound set, by evidence tier and measured CAC."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import figstyle as F

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    t = pd.read_csv(ROOT / "results/filters/TRUE_compounds.csv")
    t["tierletter"] = t["tier"].str[0]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1),
                             gridspec_kw={"width_ratios": [0.75, 0.85, 1.5], "wspace": 0.34})

    # a) class balance
    ax = axes[0]
    n_a = (t["class"] == "aggregator").sum(); n_n = len(t) - n_a
    ax.bar([0, 1], [n_a, n_n], color=[F.AGG, F.NON], width=0.62, edgecolor="none")
    for x, v in zip([0, 1], [n_a, n_n]):
        ax.text(x, v + 1, str(v), ha="center", fontsize=11, fontweight="bold",
                color=F.AGG if x == 0 else F.NON)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["aggregator", "non-\naggregator"], fontsize=8)
    ax.set_ylim(0, 44); ax.set_yticks([])
    ax.set_title("a   65 compounds", fontsize=9.5, loc="left", fontweight="bold")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)

    # b) evidence tiers
    ax = axes[1]
    order = ["A", "B", "C"]
    counts = [(t["tierletter"] == k).sum() for k in order]
    ax.barh(range(3)[::-1], counts, color=["#1F6F5C", F.NON, "#A9D9CD"],
            height=0.6, edgecolor="none")
    for i, c in enumerate(counts):
        ax.text(c + 0.8, 2 - i, str(c), va="center", fontsize=10, fontweight="bold",
                color=F.INK)
    ax.set_yticks(range(3)[::-1])
    ax.set_yticklabels(["A  measured CAC", "B  sources agree", "C  single source"],
                       fontsize=8)
    ax.set_xlim(0, 34); ax.set_xticks([])
    ax.set_title("b   Evidence tier", fontsize=9.5, loc="left", fontweight="bold")
    for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)

    # c) measured CAC values
    ax = axes[2]
    cac = t.dropna(subset=["CAC_uM"]).sort_values("CAC_uM")
    y = np.arange(len(cac))
    ax.hlines(y, 0.2, cac["CAC_uM"], color="#E4E7E8", lw=1.1)
    ax.scatter(cac["CAC_uM"], y, s=26, color=F.AGG, zorder=3, edgecolors="white", lw=0.6)
    ax.set_xscale("log"); ax.set_yticks(y)
    ax.set_yticklabels([n.strip()[:17] for n in cac["name"]], fontsize=6.3)
    ax.set_xlabel("critical aggregation concentration (µM, log scale)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_title("c   Measured CAC, 0.3 to 123 µM", fontsize=9.5, loc="left",
                 fontweight="bold")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#EFF1F1", lw=0.7); ax.set_axisbelow(True)

    F.save(fig, "M1_compound_sets")

if __name__ == "__main__":
    main()
