"""Methods Figure 5: escalating modification of twelve confirmed non-aggregators.

a  best ChemAGG probability per lineage against modification round
b  the measured flip distance, in heavy atoms added
c  what ChemAGG called an aggregator each round, against the clogP null
d  how many of the twelve flips survive a similarity floor to the parent drug

"Rung" is internal jargon and is written as "modification round" throughout.
"""
import glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import pandas as pd
import figstyle as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLOORS = [0.3, 0.4, 0.5, 0.6, 0.7]


def main():
    st = json.loads((ROOT / "results/beam/state.json").read_text())
    rounds = sorted(int(p.split("rung")[1].split("_")[0])
                    for p in glob.glob(str(ROOT / "results/beam/rung*_scored.csv")))
    scored = {r: pd.read_csv(ROOT / f"results/beam/rung{r}_scored.csv") for r in rounds}
    ann = pd.read_csv(ROOT / "results/beam/all_analogues_annotated.csv")

    fig = plt.figure(figsize=(11.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.30, 1.0], height_ratios=[1, 0.92],
                          wspace=0.30, hspace=0.42)

    # ---- a  trajectories -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axhspan(0.5, 1.0, color=F.AGG, alpha=0.08)
    ax.axhline(0.5, color=F.AGG, lw=1.1, ls="--")
    for nm, L in sorted(st["lineages"].items()):
        h = L["history"]
        if not h:
            continue
        x = [0] + [e["rung"] for e in h]
        y = [0.0] + [e["best_prob"] for e in h]
        ax.plot(x, y, "-o", ms=3.2, lw=1.4, color=F.AGG, alpha=0.9)
        ax.annotate(nm, (x[-1], y[-1]), fontsize=6.4, color=F.INK,
                    textcoords="offset points", xytext=(4, -1.5))
    ax.set_xlabel("modification round", fontsize=8.5)
    ax.set_ylabel("best ChemAGG probability", fontsize=8.5)
    ax.set_title("a   Twelve confirmed non-aggregators, pushed", fontsize=9.5,
                 loc="left", fontweight="bold")
    ax.set_xlim(-0.15, max(rounds) + 1.9); ax.set_ylim(0, 1.0)
    ax.set_xticks(range(0, max(rounds) + 1))
    ax.text(0.03, 0.93, "ChemAGG calls aggregator", transform=ax.transAxes,
            fontsize=7.5, color=F.AGG, fontweight="bold")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # ---- b  measured flip distance --------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    fl = sorted(((nm, L["flip"]["d_heavy_total"], L["flip"]["prob"])
                 for nm, L in st["lineages"].items() if L.get("flip")),
                key=lambda t: t[1])
    ax.barh([f[0] for f in fl], [f[1] for f in fl], color=F.AGG, height=0.62,
            edgecolor="none")
    for i, f in enumerate(fl):
        ax.text(f[1] + 0.22, i, f"p={f[2]:.2f}", va="center", fontsize=7,
                color=F.GREY)
    ax.axvline(1, color=F.NON, lw=1.6, ls="--")
    ax.annotate("the one validated cliff: +1", xy=(1, 11.3), xytext=(3.4, 11.9),
                fontsize=7.4, color=F.NON, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=F.NON, lw=1.0))
    ax.set_xlabel("heavy atoms added before the verdict changed", fontsize=8.5)
    ax.set_title("b   Measured flip distance", fontsize=9.5, loc="left",
                 fontweight="bold")
    ax.set_xlim(0, max(f[1] for f in fl) + 3.2)
    ax.tick_params(labelsize=7.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # ---- c  against the trivial null ------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    xs = list(scored)
    ca = [100 * (scored[r]["chemagg_cat"] == 1).mean() for r in xs]
    nl = [100 * scored[r]["null_clogp_call"].mean() for r in xs]
    w = 0.36
    ax.bar([x - w / 2 for x in xs], ca, w, color=F.AGG, label="ChemAGG")
    ax.bar([x + w / 2 for x in xs], nl, w, color="#CFD5D7",
           label="clogP > 3.5 (null)")
    for x, v in zip(xs, ca):
        ax.text(x - w / 2, v + 2.0, f"{v:.1f}%", ha="center", fontsize=7,
                color=F.AGG, fontweight="bold")
    ax.set_xlabel("modification round", fontsize=8.5)
    ax.set_ylabel("analogues called aggregator (%)", fontsize=8.5)
    ax.set_title("c   Against a trivial baseline", fontsize=9.5, loc="left",
                 fontweight="bold")
    ax.set_xticks(xs); ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # ---- d  flips surviving a similarity floor ---------------------------
    ax = fig.add_subplot(gs[1, 1])
    n = [ann[(ann.chemagg_cat == 1) & (ann.tc_to_parent_drug >= f)].lineage.nunique()
         for f in FLOORS]
    cols = [F.AGG if f < 0.6 else F.NON for f in FLOORS]
    ax.bar(range(len(FLOORS)), n, color=cols, width=0.62, edgecolor="none")
    for i, v in enumerate(n):
        ax.text(i, v + 0.25, str(v), ha="center", fontsize=9, fontweight="bold",
                color=F.INK)
    ax.axvline(2.5, color=F.NON, lw=1.4, ls="--")
    ax.text(2.62, 11.6, "matched-pair\nthreshold", ha="left", fontsize=7.4,
            color=F.NON, fontweight="bold", linespacing=1.3, va="top")
    ax.set_xticks(range(len(FLOORS)))
    ax.set_xticklabels([f"{f:.1f}" for f in FLOORS], fontsize=8)
    ax.set_xlabel("similarity floor to the parent drug (ECFP4 $T_c$)", fontsize=8.5)
    ax.set_ylabel("compounds that still flip (of 12)", fontsize=8.5)
    ax.set_ylim(0, 13.6); ax.set_yticks([0, 4, 8, 12])
    ax.set_title("d   Almost no flip is a cliff", fontsize=9.5, loc="left",
                 fontweight="bold")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    F.save(fig, "M5_escalation")
    print(f"  {sum(len(d) for d in scored.values())} analogues, {len(fl)}/12 flipped; "
          f"surviving a floor: {dict(zip(FLOORS, n))}")


if __name__ == "__main__":
    main()
