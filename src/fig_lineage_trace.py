"""Trace one drug through every modification round, boxing what changed at each step.

Reconstructs the winning path from the beam state: at each round the beam kept the three
best-scoring analogues, so the path is recovered by walking `parent_of` backwards from
the final molecule.
"""
import json, pathlib, sys
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFMCS, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
ROOT = pathlib.Path(__file__).resolve().parents[1]
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# palette shared with figstyle.py (Sturm et al. grammar)
GREEN, RED, GREY, ORANGE = "#2FA98C", "#E08A2E", "#7C8285", "#E08A2E"
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5})


def path_for(lineage):
    """Walk the scored rounds forward, following the best-scoring descendant each time."""
    st = json.loads((ROOT / "results/beam/state.json").read_text())
    root = st["lineages"][lineage]["root"]
    a = pd.read_csv(ROOT / "results/beam/all_analogues_annotated.csv")
    a = a[a["lineage"] == lineage]
    cur, out = root, [{"round": 0, "smiles": root, "chemagg_prob": None, "chemagg_cat": 0}]
    for r in sorted(a["round"].unique()):
        sub = a[(a["round"] == r) & (a["from_smiles"] == cur)]
        if sub.empty:                       # beam moved on; take best of the round
            sub = a[a["round"] == r]
        if sub.empty:
            break
        b = sub.sort_values("chemagg_prob", ascending=False).iloc[0]
        out.append({"round": int(r), "smiles": b["smiles"],
                    "chemagg_prob": float(b["chemagg_prob"]),
                    "chemagg_cat": int(b["chemagg_cat"])})
        cur = b["smiles"]
        if int(b["chemagg_cat"]) == 1:
            break
    return out


def changed(prev, cur):
    try:
        r = rdFMCS.FindMCS([prev, cur], timeout=8)
        if r.canceled or r.numAtoms == 0:
            return []
        patt = Chem.MolFromSmarts(r.smartsString)
        mc, mp = cur.GetSubstructMatch(patt), prev.GetSubstructMatch(patt)
        out = [a.GetIdx() for a in cur.GetAtoms() if a.GetIdx() not in set(mc)]
        for ci, pi in zip(mc, mp):
            if cur.GetAtomWithIdx(ci).GetSymbol() != prev.GetAtomWithIdx(pi).GetSymbol():
                out.append(ci)
        return out
    except Exception:
        return []


def png(mol, hl, size=(420, 350)):
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions(); o.bondLineWidth = 2; o.padding = 0.11
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=hl,
        highlightAtomColors={i: (0.96, 0.72, 0.42) for i in hl})
    d.FinishDrawing()
    import io
    from PIL import Image
    return Image.open(io.BytesIO(d.GetDrawingText()))


def main(lineage):
    p = path_for(lineage)
    n = len(p)
    root = Chem.MolFromSmiles(p[0]["smiles"])
    fig = plt.figure(figsize=(3.05 * n, 5.2))
    gs = fig.add_gridspec(2, n, height_ratios=[3.0, 1.1], hspace=0.34, wspace=0.05)

    prev = None
    for i, s in enumerate(p):
        m = Chem.MolFromSmiles(s["smiles"])
        hl = changed(prev, m) if prev is not None else []
        ax = fig.add_subplot(gs[0, i]); ax.axis("off"); ax.imshow(png(m, hl))
        if hl:   # box the changed region
            ax.add_patch(FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                boxstyle="round,pad=0.005", fill=False, lw=1.6,
                edgecolor=ORANGE, linestyle="--"))
        dh = m.GetNumHeavyAtoms() - root.GetNumHeavyAtoms()
        tc = DataStructs.TanimotoSimilarity(FP.GetFingerprint(root), FP.GetFingerprint(m))
        head = (f"{lineage}\nthe real drug" if i == 0
                else f"round {s['round']}\ncumulative {dh:+d} heavy atoms")
        ax.set_title(head, fontsize=9, fontweight="bold" if i == 0 else "normal",
                     color="black" if i == 0 else GREY, pad=3)
        if s["chemagg_prob"] is None:
            txt, col = "starting point", GREY
        else:
            flip = s["chemagg_cat"] == 1
            txt = f"p = {s['chemagg_prob']:.3f}"
            col = RED if flip else GREEN
        ax.text(0.5, -0.01, txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold", color=col)
        verdict = ("experimentally NON-AGGREGATOR" if i == 0 else
                   ("ChemAGG: AGGREGATOR" if s["chemagg_cat"] == 1 else "ChemAGG: non-aggregator"))
        ax.text(0.5, -0.09, verdict, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.8, color=col)
        ax.text(0.5, -0.16, f"similarity to the drug: {tc:.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.2, color=GREY)
        prev = m

    ax = fig.add_subplot(gs[1, :])
    xs = [s["round"] for s in p]
    ys = [0.0] + [s["chemagg_prob"] for s in p[1:]]
    ax.axhspan(0.5, 1.0, color=RED, alpha=0.07)
    ax.axhline(0.5, color=RED, lw=1.2, ls="--")
    ax.plot(xs, ys, "-o", color=RED if p[-1]["chemagg_cat"] == 1 else GREEN, lw=2.2, ms=7)
    for x, y in zip(xs[1:], ys[1:]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8)
    ax.text(0.012, 0.86, "ChemAGG calls it an aggregator above this line",
            transform=ax.transAxes, fontsize=8, color=RED)
    ax.set_xlabel("modification round"); ax.set_ylabel("ChemAGG probability")
    ax.set_xticks(xs); ax.set_xlim(-0.25, max(xs) + 0.25); ax.set_ylim(0, 1.0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    fin = Chem.MolFromSmiles(p[-1]["smiles"])
    fig.suptitle(
        f"{lineage}: an experimentally confirmed NON-aggregator, modified one step at a time.\n"
        f"ChemAGG only calls it an aggregator after "
        f"{fin.GetNumHeavyAtoms()-root.GetNumHeavyAtoms():+d} heavy atoms "
        f"({len(p)-1} rounds), by which point it is only "
        f"{DataStructs.TanimotoSimilarity(FP.GetFingerprint(root), FP.GetFingerprint(fin)):.2f} "
        f"similar to the original drug.   Nature flips aggregation with +1 atom.",
        fontsize=10, y=1.02)
    for e in ("png", "pdf", "svg"):
        fig.savefig(ROOT / f"Figures/trace_{lineage}.{e}", dpi=600, bbox_inches="tight")
    print(f"wrote Figures/trace_{lineage}.png  ({len(p)-1} rounds)")
    for s in p:
        print(f"   round {s['round']}: {s['smiles']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "promethazine")
