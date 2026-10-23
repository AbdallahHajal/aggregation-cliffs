"""Figure: parent vs best rung-1 analogue for each lineage, change highlighted.

Left = the experimentally confirmed non-aggregator. Right = the single modification,
out of everything CReM proposed, that ChemAGG scored highest. Orange marks what changed.
"""
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLUE, VERM, GREY, AMBER = "#0072B2", "#D55E00", "#666666", "#B8860B"
plt.rcParams.update({"font.family": "Arial", "font.size": 8})


def changed(prev, cur):
    try:
        r = rdFMCS.FindMCS([prev, cur], timeout=8)
        if r.canceled or r.numAtoms == 0:
            return [], []
        patt = Chem.MolFromSmarts(r.smartsString)
        mc, mp = cur.GetSubstructMatch(patt), prev.GetSubstructMatch(patt)
        add = [a.GetIdx() for a in cur.GetAtoms() if a.GetIdx() not in set(mc)]
        rem = [a.GetIdx() for a in prev.GetAtoms() if a.GetIdx() not in set(mp)]
        for ci, pi in zip(mc, mp):
            if cur.GetAtomWithIdx(ci).GetSymbol() != prev.GetAtomWithIdx(pi).GetSymbol():
                add.append(ci); rem.append(pi)
        return rem, add
    except Exception:
        return [], []


def png(mol, hl, size=(340, 250)):
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions(); o.bondLineWidth = 2; o.padding = 0.10
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=hl,
        highlightAtomColors={i: (0.98, 0.65, 0.35) for i in hl})
    d.FinishDrawing()
    import io
    from PIL import Image
    return Image.open(io.BytesIO(d.GetDrawingText()))


def main():
    s = pd.read_csv(ROOT / "results/beam/rung1_scored.csv")
    best = s.sort_values("chemagg_prob", ascending=False).groupby("lineage").first().reset_index()
    best = best.sort_values("chemagg_prob", ascending=False).head(6)

    n = len(best)
    fig = plt.figure(figsize=(7.4, 2.35 * n))
    gs = fig.add_gridspec(n, 3, width_ratios=[1, 0.34, 1], hspace=0.45, wspace=0.04)

    for i, (_, r) in enumerate(best.iterrows()):
        p = Chem.MolFromSmiles(r["from_smiles"]); c = Chem.MolFromSmiles(r["smiles"])
        rem, add = changed(p, c)

        ax = fig.add_subplot(gs[i, 0]); ax.axis("off"); ax.imshow(png(p, rem))
        ax.set_title(f"{r['lineage']}", fontsize=9.5, fontweight="bold", pad=2)
        ax.text(0.5, -0.02, "experimentally a NON-AGGREGATOR (DLS)", transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=GREY)

        mid = fig.add_subplot(gs[i, 1]); mid.axis("off")
        mid.annotate("", xy=(0.92, 0.55), xytext=(0.08, 0.55), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", lw=1.8, color=BLUE))
        dh = int(r["d_heavy"])
        mid.text(0.5, 0.63, f"{dh:+d} heavy atom" + ("s" if abs(dh) != 1 else ""),
                 transform=mid.transAxes, ha="center", fontsize=7.5, color=BLUE)
        mid.text(0.5, 0.40, f"$\\Delta$clogP {Crippen.MolLogP(c)-Crippen.MolLogP(p):+.2f}",
                 transform=mid.transAxes, ha="center", fontsize=7, color=GREY)

        ax = fig.add_subplot(gs[i, 2]); ax.axis("off"); ax.imshow(png(c, add))
        col = AMBER if r["chemagg_prob"] >= 0.3 else GREY
        ax.set_title(f"best of {int((s['lineage']==r['lineage']).sum())} analogues",
                     fontsize=8, color=GREY, pad=2)
        ax.text(0.5, -0.02, f"ChemAGG p = {r['chemagg_prob']:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=10, fontweight="bold", color=col)
        ax.text(0.5, -0.12, "still called NON-AGGREGATOR", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color=col)

    n_tot = len(s)
    n_null = int(s["null_clogp_call"].sum())
    fig.suptitle(
        "Rung 1: one structural change to twelve confirmed non-aggregators\n"
        f"ChemAGG called 0 of {n_tot} analogues an aggregator. A clogP > 3.5 cut-off called {n_null} of {n_tot}.",
        fontsize=10, y=0.995)
    out = ROOT / "Figures"
    for e in ("png", "pdf", "svg"):
        fig.savefig(out / f"rung1_pairs.{e}", dpi=600, bbox_inches="tight")
    print(f"wrote Figures/rung1_pairs.png  ({n} lineages shown of {s['lineage'].nunique()})")


if __name__ == "__main__":
    main()
