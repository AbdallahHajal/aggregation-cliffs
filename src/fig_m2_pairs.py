"""Methods Figure 2: matched pairs with opposite labels, and the isomer subset."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import rdFMCS
import figstyle as F

ROOT = pathlib.Path(__file__).resolve().parents[1]

def diff(a, b):
    try:
        r = rdFMCS.FindMCS([a, b], timeout=6)
        p = Chem.MolFromSmarts(r.smartsString)
        ma = a.GetSubstructMatch(p)
        return [x.GetIdx() for x in a.GetAtoms() if x.GetIdx() not in set(ma)]
    except Exception:
        return []

def main():
    cru = pd.read_csv(ROOT / "results/cruzain_full_real_tight.csv")
    amp = pd.read_csv(ROOT / "results/ampc_full_real_tight.csv")
    iso = pd.read_csv(ROOT / "results/cliffs_isomer_hardest.csv")

    fig = plt.figure(figsize=(10.2, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.30,
                          top=0.92, bottom=0.55, left=0.085, right=0.975)

    # a) how many
    ax = fig.add_subplot(gs[0, 0])
    ax.bar([0, 1], [len(cru), len(amp)], color=F.AGG, width=0.6, edgecolor="none")
    for x, v in zip([0, 1], [len(cru), len(amp)]):
        ax.text(x, v + 40, f"{v:,}", ha="center", fontsize=10.5, fontweight="bold",
                color=F.AGG)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Cruzain", "AmpC"], fontsize=8.5)
    ax.set_ylim(0, 2200); ax.set_yticks([])
    ax.set_title("a   3,457 matched pairs", fontsize=9.5, loc="left", fontweight="bold")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)

    # b) which transformation separates the pair.  The mass differences are
    # discrete, so a histogram misrepresents them; the named changes are shown.
    ax = fig.add_subplot(gs[0, 1])
    both = pd.concat([cru, amp]).drop_duplicates(subset=["agg_ik", "non_ik"])
    d = both["d_mw"].abs().round(2)
    N = len(both)
    named = [("CH$_2$", 14.03), ("isomer", 0.00), ("H$\\to$F", 17.99),
             ("CH$_3$/Cl", 20.42), ("+O", 16.00)]
    counts = [int((d == v).sum()) for _, v in named]
    labels = [l for l, _ in named] + ["other"]
    counts.append(N - sum(counts))
    cols = [F.AGG] * 5 + ["#CFD5D7"]
    cols[1] = F.NON
    y = range(len(labels))[::-1]
    ax.barh(list(y), counts, color=cols, height=0.62, edgecolor="none")
    for yy, c in zip(y, counts):
        ax.text(c + N * 0.02, yy, f"{100 * c / N:.0f}%", va="center", fontsize=8,
                fontweight="bold", color=F.INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(0, N * 1.18); ax.set_xticks([])
    ax.set_title("b   What separates the pair", fontsize=9.5, loc="left",
                 fontweight="bold")
    for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)

    # c) descriptor differences are exactly zero for the isomer subset
    ax = fig.add_subplot(gs[0, 2])
    labs = ["Δ clogP", "Δ TPSA", "Δ MW", "Δ heavy"]
    ax.barh(range(4)[::-1], [0, 0, 0, 0], color=F.NON, height=0.5)
    for i, l in enumerate(labs):
        ax.text(0.04, 3 - i, "0.00", va="center", fontsize=10, fontweight="bold",
                color=F.NON)
    ax.set_yticks(range(4)[::-1]); ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.text(0.04, -0.85, "ECFP4 similarity  1.000", fontsize=9, fontweight="bold",
            color=F.NON)
    ax.set_title("c   386 isomer pairs", fontsize=9.5, loc="left", fontweight="bold")
    for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)

    # d) three worked isomer pairs
    TOP, BOT = 0.395, 0.105
    fig.text(0.085, TOP + 0.012, "d   Same descriptors, opposite labels",
             fontsize=9.5, fontweight="bold", color=F.INK, ha="left")
    for k in range(3):
        r = iso.iloc[k]
        a, n = Chem.MolFromSmiles(r["agg_smiles"]), Chem.MolFromSmiles(r["non_smiles"])
        sub = fig.add_gridspec(1, 2, left=0.085 + k * 0.302, right=0.345 + k * 0.302,
                               bottom=BOT, top=TOP, wspace=0.02)
        for j, (m, other, col, lab) in enumerate([
                (a, n, F.AGG, "aggregator"), (n, a, F.NON, "non-aggregator")]):
            s = fig.add_subplot(sub[0, j]); s.axis("off")
            s.imshow(F.mol_png(m, diff(m, other), col, (420, 340), lw=1.7))
            s.text(0.5, -0.02, lab, transform=s.transAxes, ha="center", va="top",
                   fontsize=7.2, fontweight="bold", color=col)

    F.save(fig, "M2_matched_pairs")

if __name__ == "__main__":
    main()
