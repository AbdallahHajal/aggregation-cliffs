"""Methods Figure 4: drug-likeness, stability, and drift away from the parent drug."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import figstyle as F

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    a = pd.read_csv(ROOT / "results/beam/all_analogues_annotated.csv")
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9),
                             gridspec_kw={"wspace": 0.34})

    # a) QED
    ax = axes[0]
    ax.hist(a["qed"], bins=22, color=F.NON, alpha=0.85, edgecolor="white", lw=0.5)
    ax.axvline(0.5, color=F.INK, lw=1.4, ls="--")
    ax.text(0.52, ax.get_ylim()[1] * 0.88, "0.5", fontsize=7.6, color=F.INK)
    ax.set_xlabel("QED", fontsize=8); ax.set_yticks([]); ax.tick_params(labelsize=7.2)
    ax.set_title("a   Drug-likeness", fontsize=9.5, loc="left", fontweight="bold")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)

    # b) SA score
    ax = axes[1]
    ax.hist(a["sa"], bins=22, color=F.NON, alpha=0.85, edgecolor="white", lw=0.5)
    ax.axvline(4.5, color=F.INK, lw=1.4, ls="--")
    ax.text(4.6, ax.get_ylim()[1] * 0.88, "4.5", fontsize=7.6, color=F.INK)
    ax.set_xlabel("synthetic accessibility", fontsize=8); ax.set_yticks([])
    ax.tick_params(labelsize=7.2)
    ax.set_title("b   Ease of synthesis", fontsize=9.5, loc="left", fontweight="bold")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)

    # c) drift
    ax = axes[2]
    g = a.groupby("round")["tc_to_parent_drug"]
    xs = sorted(a["round"].unique())
    ax.plot(xs, [g.get_group(r).median() for r in xs], "-o", color=F.AGG, lw=2, ms=5)
    ax.fill_between(xs, [g.get_group(r).quantile(.25) for r in xs],
                    [g.get_group(r).quantile(.75) for r in xs],
                    color=F.AGG, alpha=0.18, lw=0)
    ax.axhline(0.6, color=F.NON, lw=1.4, ls="--")
    ax.text(1.1, 0.625, "matched-pair threshold", fontsize=7.0, color=F.NON)
    ax.set_xlabel("modification round", fontsize=8)
    ax.set_ylabel("similarity to the drug", fontsize=8)
    ax.set_xticks(xs); ax.set_ylim(0, 1.0); ax.tick_params(labelsize=7.2)
    ax.set_title("c   Drift", fontsize=9.5, loc="left", fontweight="bold")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # d) what survives filtering
    ax = axes[3]
    n = len(a)
    dl = ((a["qed"] >= 0.5) & (a["sa"] <= 4.5)).sum()
    both = ((a["qed"] >= 0.5) & (a["sa"] <= 4.5) & a["stable"]).sum()
    # a horizontal funnel: the two retention steps are very different in size,
    # and the near-equal last two bars are themselves the point.
    vals = [n, dl, both]
    labs = ["generated", "drug-like", "+ stable"]
    cols = ["#CFD5D7", F.NON, "#1F6F5C"]
    ys = [2, 1, 0]
    ax.barh(ys, vals, color=cols, height=0.56, edgecolor="none")
    for y, v in zip(ys, vals):
        ax.text(v + n * 0.03, y, f"{v:,}", va="center", fontsize=9.5,
                fontweight="bold", color=F.INK)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=8.4)
    ax.set_xlim(0, n * 1.30); ax.set_xticks([])
    ax.annotate("$-$1: a carbamic acid", xy=(dl * 0.62, 0.36),
                fontsize=7.4, color=F.GREY, ha="center")
    ax.set_title("d   Analogues retained", fontsize=9.5, loc="left", fontweight="bold")
    for sp in ("top", "right", "bottom"): ax.spines[sp].set_visible(False)

    F.save(fig, "M4_quality_filters")
    print(f"   {n} analogues, {dl} drug-like, {both} also chemically stable")

if __name__ == "__main__":
    main()
