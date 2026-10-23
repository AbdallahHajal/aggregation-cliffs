"""Figure: how far each non-aggregator had to be pushed before ChemAGG flipped it.

Panel A  trajectory of the best ChemAGG probability per lineage against rung.
Panel B  the flip distance actually measured, in heavy atoms added.
Panel C  what ChemAGG called an aggregator at each rung, against the trivial clogP null.

Palette follows Sturm et al. (Angew Chem 2025): muted teal/green for non-aggregating,
warm orange/red for aggregating, grey for everything structural.
"""
import glob, json, pathlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[1]
GREEN, RED, GREY, BLUE = "#2E8B74", "#D1495B", "#7A7A7A", "#2F6690"
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5})


def main():
    st = json.loads((ROOT / "results/beam/state.json").read_text())
    rungs = sorted(int(p.split("rung")[1].split("_")[0])
                   for p in glob.glob(str(ROOT / "results/beam/rung*_scored.csv")))
    scored = {r: pd.read_csv(ROOT / f"results/beam/rung{r}_scored.csv") for r in rungs}

    fig = plt.figure(figsize=(11.6, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.32)

    # ---- A: trajectories -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axhspan(0.5, 1.0, color=RED, alpha=0.07)
    ax.axhline(0.5, color=RED, lw=1.1, ls="--")
    for nm, L in sorted(st["lineages"].items()):
        h = L["history"]
        if not h:
            continue
        x = [0] + [e["rung"] for e in h]
        y = [0.0] + [e["best_prob"] for e in h]
        flip = L["status"] == "flipped"
        ax.plot(x, y, "-o", ms=3.4, lw=1.5,
                color=RED if flip else GREEN, alpha=0.95 if flip else 0.55)
        ax.annotate(nm[:13], (x[-1], y[-1]), fontsize=6.6,
                    color=RED if flip else GREY,
                    textcoords="offset points", xytext=(4, -1))
    ax.set_xlabel("escalation rung"); ax.set_ylabel("best ChemAGG probability")
    ax.set_title("A   Pushing confirmed non-aggregators toward 'aggregator'",
                 fontsize=9, loc="left", fontweight="bold")
    ax.set_xlim(-0.15, max(rungs) + 1.35); ax.set_ylim(0, 1.0)
    ax.set_xticks(range(0, max(rungs) + 1))
    ax.text(0.03, 0.93, "ChemAGG calls aggregator", transform=ax.transAxes,
            fontsize=7.5, color=RED)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=RED, marker="o", ms=4, label="flipped"),
                       Line2D([], [], color=GREEN, marker="o", ms=4, label="did not flip")],
              fontsize=7.5, frameon=False, loc="lower right")

    # ---- B: measured flip distance --------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    fl = [(nm, L["flip"]["d_heavy_total"], L["flip"]["prob"])
          for nm, L in st["lineages"].items() if L.get("flip")]
    fl.sort(key=lambda t: t[1])
    if fl:
        ax.barh([f[0][:15] for f in fl], [f[1] for f in fl], color=RED, alpha=0.85, height=0.6)
        for i, f in enumerate(fl):
            ax.text(f[1] + 0.12, i, f"p={f[2]:.2f}", va="center", fontsize=7, color=GREY)
    ax.set_xlabel("heavy atoms added before the verdict flipped")
    ax.set_title("B   Measured flip distance", fontsize=9, loc="left", fontweight="bold")
    ax.set_xlim(0, (max(f[1] for f in fl) + 2) if fl else 6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)

    # ---- C: ChemAGG vs the trivial null ---------------------------------
    ax = fig.add_subplot(gs[0, 2])
    xs = list(scored)
    ca = [100 * (scored[r]["chemagg_cat"] == 1).mean() for r in xs]
    nl = [100 * scored[r]["null_clogp_call"].mean() for r in xs]
    w = 0.36
    ax.bar([x - w/2 for x in xs], ca, w, color=RED, label="ChemAGG")
    ax.bar([x + w/2 for x in xs], nl, w, color=GREY, alpha=0.65, label="clogP > 3.5 (null)")
    for x, v in zip(xs, ca):
        ax.text(x - w/2, v + 1.6, f"{v:.1f}%", ha="center", fontsize=7, color=RED)
    ax.set_xlabel("escalation rung"); ax.set_ylabel("% of analogues called aggregator")
    ax.set_title("C   Against a trivial baseline", fontsize=9, loc="left", fontweight="bold")
    ax.set_xticks(xs); ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    n_tot = sum(len(d) for d in scored.values())
    n_fl = len(fl)
    fig.suptitle(
        f"Twelve DLS-confirmed non-aggregators, escalating structural modification, "
        f"{n_tot} analogues scored by ChemAGG.  {n_fl} of 12 crossed its decision boundary.",
        fontsize=10, y=1.03)
    for e in ("png", "pdf", "svg"):
        fig.savefig(ROOT / f"Figures/beam_progress.{e}", dpi=600, bbox_inches="tight")
    print(f"wrote Figures/beam_progress.png   rungs {xs}, {n_tot} analogues, {n_fl} flipped")
    for nm, d, p in fl:
        print(f"    {nm:18s} flipped at +{d} heavy atoms (p={p:.3f})")


if __name__ == "__main__":
    main()
