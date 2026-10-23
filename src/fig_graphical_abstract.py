"""Graphical abstract in the layout of Sturm et al. Angew Chem 2025, Figure 5.

Two rows, each inside a dashed rounded container with a vertical coloured tab on the
left, a small boxed sub-header, molecules carrying coloured halos on what changed, and
values underneath. Molecules are the content; text is kept to labels.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import rdFMCS
import figstyle as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
DASH, EDGE, HDR = "#9AA0A3", "#7E858A", "#F2F3F1"


def changed(prev, cur):
    if prev is None:
        return []
    try:
        r = rdFMCS.FindMCS([prev, cur], timeout=8)
        p = Chem.MolFromSmarts(r.smartsString)
        mc, mp = cur.GetSubstructMatch(p), prev.GetSubstructMatch(p)
        out = [a.GetIdx() for a in cur.GetAtoms() if a.GetIdx() not in set(mc)]
        for ci, pi in zip(mc, mp):
            if cur.GetAtomWithIdx(ci).GetSymbol() != prev.GetAtomWithIdx(pi).GetSymbol():
                out.append(ci)
        return out
    except Exception:
        return []


def container(fig, rect):
    ax = fig.add_axes(rect); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                 boxstyle="round,pad=0,rounding_size=0.025",
                 facecolor="white", edgecolor=DASH, linewidth=1.2,
                 linestyle=(0, (4, 3)), zorder=0))
    return ax


def side_tab(fig, x, y, h, text, colour, w=0.020):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                 boxstyle="round,pad=0,rounding_size=0.25",
                 facecolor=colour, edgecolor="none", alpha=0.85))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            rotation=90, fontsize=7.4, fontweight="bold", color="white")


def sub_header(fig, x, y, w, text, h=0.042):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                 boxstyle="round,pad=0,rounding_size=0.10",
                 facecolor=HDR, edgecolor=EDGE, linewidth=0.9))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=F.INK)


def mol_cell(fig, x, y, w, h, smi, prev, colour, tag, value, verdict, vcol):
    """colour is the glow tint, set by the predicted class of THIS molecule."""
    ax = fig.add_axes([x, y, w, h]); ax.axis("off")
    m = Chem.MolFromSmiles(smi)
    ax.imshow(F.mol_png(m, changed(prev, m), colour, (520, 420), lw=1.9))
    ax.text(0.02, 1.045, tag, transform=ax.transAxes, fontsize=8.2,
            fontweight="bold", color=F.INK, va="bottom")
    ax.text(0.5, -0.03, value, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.4, fontweight="bold", color=vcol)
    ax.text(0.5, -0.135, verdict, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.0, color=vcol)
    return m


def main():
    st = __import__("json").loads((ROOT / "results/beam/state.json").read_text())
    L = st["lineages"]["promethazine"]
    path = [L["root"]] + [h["best_smiles"] for h in L["history"]]
    probs = [None] + [h["best_prob"] for h in L["history"]]

    m58 = pd.read_csv(ROOT / "results/filters/merged_58.csv")
    aggs = m58[m58["class"] == "aggregator"].sort_values("chemagg_prob").head(4)

    fig = plt.figure(figsize=(9.8, 5.9)); fig.patch.set_facecolor("white")

    # ================= ROW 1 : a confirmed non-aggregator, modified =================
    container(fig, [0.035, 0.535, 0.945, 0.400])
    side_tab(fig, 0.047, 0.565, 0.340, "NON-AGGREGATOR", F.NON)
    sub_header(fig, 0.092, 0.876, 0.200, "Experimental")
    sub_header(fig, 0.314, 0.876, 0.644, "Stepwise modification, best-scoring analogue retained")

    prev = None
    keep = [0, 1, 3, 4]
    for i, k in enumerate(keep):
        smi, p = path[k], probs[k]
        x = 0.092 + i * 0.222
        w = 0.200
        val = "not an aggregator" if p is None else f"p = {p:.3f}"
        vcol = F.NON if (p is None or p < 0.5) else F.AGG
        verd = "measured by DLS" if p is None else (
            "ChemAGG: AGGREGATOR" if p >= 0.5 else "ChemAGG: non-aggregator")
        glow_col = F.NON if (p is None or p < 0.5) else F.AGG
        prev = mol_cell(fig, x, 0.638, w, 0.185, smi, prev, glow_col,
                        "promethazine" if i == 0 else f"round {k}", val, verd, vcol)
    fig.text(0.5075, 0.556, "eight heavy atoms are required before the verdict changes; "
                            "the analogue is then only 0.40 similar to the drug",
             ha="center", fontsize=7.4, color=F.GREY, style="italic")

    # ================= ROW 2 : confirmed aggregators, three filters =================
    container(fig, [0.035, 0.048, 0.945, 0.455])
    side_tab(fig, 0.047, 0.078, 0.392, "AGGREGATOR", F.AGG)
    sub_header(fig, 0.085, 0.450, 0.873, "Confirmed aggregators (DLS), and what three published filters say")

    for i, (_, r) in enumerate(aggs.iterrows()):
        x = 0.088 + i * 0.220
        ax = fig.add_axes([x, 0.220, 0.195, 0.180]); ax.axis("off")
        ax.imshow(F.mol_png(r["smiles"], [], F.AGG, (520, 420), lw=1.9))
        ax.text(0.02, 1.055, r["name"].strip()[:18], transform=ax.transAxes,
                fontsize=8.0, fontweight="bold", color=F.INK, va="bottom")
        for j, (lab, txt, ok) in enumerate([
                ("BAD", f"{r['bad_prob']:.2f}", r["bad_prob"] > 0.6),
                ("ChemAGG", f"{r['chemagg_prob']:.3f}", r["chemagg_cat"] == 1),
                ("Advisor", str(r["advisor_verdict"]).replace("_", " ").title(),
                 r["advisor_flagged"] == 1)]):
            yy = 0.186 - j * 0.042
            col = F.AGG if ok else F.GREY
            fig.text(x + 0.012, yy, lab, fontsize=7.0, color=F.GREY, va="center")
            fig.text(x + 0.183, yy, txt, fontsize=7.4, color=col, va="center",
                     ha="right", fontweight="bold" if ok else "normal")
            fig.add_artist(plt.Line2D([x + 0.012, x + 0.183], [yy - 0.020, yy - 0.020],
                           transform=fig.transFigure, color="#EAECEC", lw=0.7))
    fig.text(0.5075, 0.062, "all four form colloids in the laboratory.  BAD and ChemAGG call none of "
                            "them an aggregator; Advisor flags all four",
             ha="center", fontsize=7.4, color=F.GREY, style="italic")

    F.save(fig, "graphical_abstract")


if __name__ == "__main__":
    main()
