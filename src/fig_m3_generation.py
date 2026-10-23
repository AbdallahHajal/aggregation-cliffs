"""Methods Figure 3: why both CReM modes are needed.

mutate_mol replaces fragments and so cannot methylate a free hydroxyl; the validated
reference transformation is therefore unreachable with it. grow_mol adds at hydrogen
positions and recovers it.
"""
import collections, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from crem.crem import grow_mol, mutate_mol
from rdkit import Chem
from rdkit.Chem import rdFMCS
import figstyle as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = str(ROOT / "vendor/crem/chembl33_sa2_f5.db")
CLIO = "Oc1c(I)cc(Cl)c2cccnc12"
MCLIO = "COc1c(I)cc(Cl)c2cccnc12"

def diff(a, b):
    try:
        r = rdFMCS.FindMCS([a, b], timeout=6)
        p = Chem.MolFromSmarts(r.smartsString)
        ma = a.GetSubstructMatch(p)
        return [x.GetIdx() for x in a.GetAtoms() if x.GetIdx() not in set(ma)]
    except Exception:
        return []

def main():
    m = Chem.MolFromSmiles(CLIO)
    tgt = Chem.MolToInchiKey(Chem.MolFromSmiles(MCLIO))
    res = {}
    for lab, gen in [("mutate_mol", lambda: mutate_mol(m, db_name=DB, max_size=1,
                                                       min_size=0, radius=3)),
                     ("grow_mol", lambda: grow_mol(m, db_name=DB, min_atoms=1,
                                                   max_atoms=1, radius=3))]:
        s = set(gen()); d = []
        hit = False
        for x in s:
            q = Chem.MolFromSmiles(x)
            if q is None: continue
            d.append(q.GetNumHeavyAtoms() - m.GetNumHeavyAtoms())
            if Chem.MolToInchiKey(q) == tgt: hit = True
        res[lab] = (collections.Counter(d), hit, len(d))

    fig = plt.figure(figsize=(10.2, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.0, 1.25], wspace=0.30)

    # a) the reference transformation
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("a   The validated transformation", fontsize=9.5, loc="left",
                 fontweight="bold")
    for j, (smi, col, lab) in enumerate([
            (CLIO, F.AGG, "clioquinol\naggregates"),
            (MCLIO, F.NON, "methylclioquinol\ndoes not")]):
        s = fig.add_axes([0.055 + j * 0.125, 0.30, 0.118, 0.44]); s.axis("off")
        other = MCLIO if j == 0 else CLIO
        s.imshow(F.mol_png(smi, diff(Chem.MolFromSmiles(smi),
                                     Chem.MolFromSmiles(other)), col, (420, 360), lw=1.9))
        s.text(0.5, -0.03, lab, transform=s.transAxes, ha="center", va="top",
               fontsize=7.6, fontweight="bold", color=col, linespacing=1.3)
    fig.text(0.172, 0.155, "one methyl, on the oxygen", ha="center", fontsize=8,
             color=F.INK, style="italic")

    # b) net change actually produced
    ax = fig.add_subplot(gs[0, 1])
    w = 0.38
    for i, lab in enumerate(["mutate_mol", "grow_mol"]):
        c, _, n = res[lab]
        for dh in (0, 1, 2):
            frac = 100 * c.get(dh, 0) / n if n else 0
            ax.bar(dh + (i - 0.5) * w, frac, width=w,
                   color=F.AGG if i == 0 else F.NON, edgecolor="white", lw=0.6)
            if frac > 3:
                ax.text(dh + (i - 0.5) * w, frac + 2, f"{frac:.0f}", ha="center",
                        fontsize=7.4, color=F.AGG if i == 0 else F.NON,
                        fontweight="bold")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["0", "+1", "+2"], fontsize=8.5)
    ax.set_xlabel("net heavy atoms changed", fontsize=8)
    ax.set_ylabel("% of analogues", fontsize=8); ax.set_ylim(0, 118)
    ax.tick_params(labelsize=7.5)
    ax.set_title("b   What each mode produces", fontsize=9.5, loc="left",
                 fontweight="bold")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=F.AGG),
                       plt.Rectangle((0, 0), 1, 1, color=F.NON)],
              labels=["mutate_mol", "grow_mol"], fontsize=7.4, frameon=False,
              loc="upper left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # c) reachability
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.set_title("c   Can it reach the reference?", fontsize=9.5, loc="left",
                 fontweight="bold")
    for i, lab in enumerate(["mutate_mol", "grow_mol"]):
        c, hit, n = res[lab]
        col = F.NON if hit else "#C24B4B"
        ax.add_patch(FancyBboxPatch((0.04, 0.50 - i * 0.34), 0.92, 0.28,
                     transform=ax.transAxes, boxstyle="round,pad=0,rounding_size=0.05",
                     facecolor="white", edgecolor=col, linewidth=1.6))
        ax.text(0.10, 0.64 - i * 0.34, lab, transform=ax.transAxes, fontsize=9,
                fontweight="bold", color=F.INK, va="center")
        ax.text(0.10, 0.565 - i * 0.34, f"{n} analogues", transform=ax.transAxes,
                fontsize=7.4, color=F.GREY, va="center")
        ax.text(0.92, 0.605 - i * 0.34, "reaches it" if hit else "cannot reach it",
                transform=ax.transAxes, fontsize=9.2, fontweight="bold",
                color=col, ha="right", va="center")
    ax.text(0.5, 0.075, "replacing a fragment cannot methylate a free OH",
            transform=ax.transAxes, ha="center", fontsize=7.8, color=F.GREY,
            style="italic")

    F.save(fig, "M3_analogue_generation")
    for k, (c, hit, n) in res.items():
        print(f"   {k}: {n} analogues, deltas {dict(sorted(c.items()))}, reaches target {hit}")

if __name__ == "__main__":
    main()
