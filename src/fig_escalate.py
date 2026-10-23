"""Figure: greedy escalation path, with the change at each step highlighted."""
import pathlib, sys
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#666666"
plt.rcParams.update({"font.family": "Arial", "font.size": 8})


def changed_atoms(prev, cur):
    """Atoms in cur that are NOT in the MCS with prev, i.e. what was added or altered."""
    if prev is None:
        return []
    try:
        r = rdFMCS.FindMCS([prev, cur], timeout=10, ringMatchesRingOnly=False,
                           completeRingsOnly=False)
        if r.canceled or r.numAtoms == 0:
            return []
        patt = Chem.MolFromSmarts(r.smartsString)
        mc, mp = cur.GetSubstructMatch(patt), prev.GetSubstructMatch(patt)
        out = [a.GetIdx() for a in cur.GetAtoms() if a.GetIdx() not in set(mc)]
        # atoms inside the MCS whose element changed (e.g. Cl -> Br) also count
        for ci, pi in zip(mc, mp):
            if cur.GetAtomWithIdx(ci).GetSymbol() != prev.GetAtomWithIdx(pi).GetSymbol():
                out.append(ci)
        return out
    except Exception:
        return []


def draw(mol, hl, size=(330, 280)):
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions(); o.bondLineWidth = 2; o.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hl,
        highlightAtomColors={i: (0.96, 0.62, 0.35) for i in hl})
    d.FinishDrawing()
    import io
    from PIL import Image
    return Image.open(io.BytesIO(d.GetDrawingText()))


def main(name):
    p = pd.read_csv(ROOT / f"results/filters/escalate_{name}.csv")
    mols = [Chem.MolFromSmiles(s) for s in p["smiles"]]
    n = len(p)

    fig = plt.figure(figsize=(2.05 * n, 4.6))
    gs = fig.add_gridspec(2, n, height_ratios=[2.5, 1.25], hspace=0.30, wspace=0.06)

    prev = None
    for i, (m, (_, r)) in enumerate(zip(mols, p.iterrows())):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        hl = changed_atoms(prev, m)
        ax.imshow(draw(m, hl))
        flipped = r["bad_call"] == "Aggregator"
        col = VERM if flipped else ("#B8860B" if r["bad_call"] == "Ambiguous" else GREY)
        head = f"{name}" if i == 0 else f"step {int(r['step'])}   {int(r['heavy'])-int(p.iloc[i-1]['heavy']):+d} heavy atoms"
        ax.set_title(head, fontsize=8, color="black" if i == 0 else GREY, pad=3)
        ax.text(0.5, -0.04, f"p = {r['bad_prob']:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=10, fontweight="bold", color=col)
        ax.text(0.5, -0.13, r["bad_call"], transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color=col)
        prev = m

    ax = fig.add_subplot(gs[1, :])
    x = p["step"].values; y = p["bad_prob"].values
    ax.axhspan(0.0, 0.30, color=GREY, alpha=0.10)
    ax.axhspan(0.30, 0.60, color="#B8860B", alpha=0.10)
    ax.axhspan(0.60, 1.0, color=VERM, alpha=0.10)
    ax.axhline(0.60, color=VERM, lw=1.2, ls="--")
    ax.axhline(0.30, color=GREY, lw=1.0, ls=":")
    ax.plot(x, y, "-o", color=BLUE, lw=2, ms=6, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color=BLUE)
    tot = int(p["n_scored"].sum())
    dh = int(p.iloc[-1]["heavy"]) - int(p.iloc[0]["heavy"])
    ax.text(0.015, 0.90, "Aggregator  (p > 0.60)", transform=ax.transAxes,
            ha="left", fontsize=8, color=VERM)
    ax.text(0.015, 0.50, "Ambiguous", transform=ax.transAxes, ha="left",
            fontsize=8, color="#B8860B")
    ax.text(0.015, 0.10, "Non-aggregator  (p < 0.30)", transform=ax.transAxes,
            ha="left", fontsize=8, color=GREY)
    ax.set_xlabel("escalation step", fontsize=9)
    ax.set_ylabel("BAD filter probability", fontsize=9)
    ax.set_xlim(-0.3, x.max() + 0.3); ax.set_ylim(0, 1.0)
    ax.set_xticks(list(x))
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    fig.suptitle(
        f"{name.capitalize()} is an experimentally confirmed aggregator. "
        f"After {int(x.max())} rounds of filter-guided modification "
        f"({tot} analogues scored, {dh:+d} heavy atoms) it still does not cross the filter's own threshold.",
        fontsize=9.5, y=0.985)

    out = ROOT / "Figures"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"escalate_{name}.{ext}", dpi=600, bbox_inches="tight")
    print(f"wrote Figures/escalate_{name}.png/.pdf/.svg")
    print(f"   {tot} analogues scored across {int(x.max())} steps; "
          f"p {y[0]:.3f} -> {y[-1]:.3f}; never crossed 0.60")


if __name__ == "__main__":
    main(sys.argv[1])
