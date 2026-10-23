#!/usr/bin/env python
"""Gallery of mined cliff pairs, drawn side by side with the difference highlighted.

Each row is one matched pair with opposing experimental labels. The atoms outside the
maximum common substructure are glow-highlighted on both members, so the transformation
is visible. The right-hand strip carries what each filter said about each member, which
is the point: the structures differ, the labels differ, the verdicts do not.

House style from figstyle.py, following Sturm et al. Angew Chem 2025.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figstyle import AGG, AGG_SOFT, GREY, INK, NON, NON_SOFT, PANEL, badge, mol_png, panel, save  # noqa: E402

RDLogger.DisableLog("rdApp.*")
SEED = 20260928
N_ISO, N_CH2, N_OTHER = 2, 3, 1
MAX_HEAVY = 34


def diff_atoms(a, b):
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return []
    res = rdFMCS.FindMCS([ma, mb], timeout=10, ringMatchesRingOnly=True)
    if not res.smartsString:
        return []
    patt = Chem.MolFromSmarts(res.smartsString)
    hit = set(ma.GetSubstructMatch(patt))
    return [i for i in range(ma.GetNumAtoms()) if i not in hit]


def load():
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    bad = pd.read_csv(ROOT / "results/pairs_sample_500_bad.csv")
    adv = pd.read_csv(ROOT / "results/pairs_sample_500_advisor.csv")
    cag = pd.read_csv(ROOT / "results/pairs_sample_500_chemagg.csv")
    return (s,
            dict(zip(bad.inchikey, bad.bad_prob)),
            dict(zip(adv.inchikey, adv.advisor_verdict)),
            dict(zip(cag.inchikey, cag.chemagg_cat)))


def pick(s, badp, cagc):
    """Draw at random under the study seed, not by smallest response.

    Selecting the pairs a filter responded to least would make the figure argue its own
    conclusion, so each category is sampled rather than ranked. Only the constraints that
    make a pair drawable at figure scale are applied.
    """
    s = s[(s.agg_heavy <= MAX_HEAVY) & (s.non_heavy <= MAX_HEAVY)].copy()
    s["bad_d"] = [abs(badp.get(a, 0) - badp.get(n, 0)) for a, n in zip(s.agg_ik, s.non_ik)]
    iso = s[(s.d_heavy == 0) & (s.d_clogp.abs() < 1e-9)]
    ch2 = s[(s.d_heavy == 1) & (s.d_mw.abs().between(13.9, 14.1))]
    oth = s[(s.d_heavy == 1) & (~s.d_mw.abs().between(13.9, 14.1))]
    parts = [d.sample(min(n, len(d)), random_state=SEED)
             for d, n in ((iso, N_ISO), (ch2, N_CH2), (oth, N_OTHER))]
    return pd.concat(parts).reset_index(drop=True)


def label_for(row):
    if row.d_heavy == 0 and abs(row.d_clogp) < 1e-9:
        return "same formula, same clogP"
    if abs(abs(row.d_mw) - 14.03) < 0.1:
        return "one methylene"
    return f"Δ {row.d_heavy:+.0f} heavy, ΔMW {row.d_mw:+.1f}"


def main():
    s, badp, advv, cagc = load()
    ex = pick(s, badp, cagc)
    n = len(ex)

    fig = plt.figure(figsize=(7.4, 1.72 * n + 0.45))
    gs = fig.add_gridspec(n, 3, width_ratios=[1, 1, 0.82], wspace=0.045, hspace=0.30)

    for r, row in ex.iterrows():
        d_agg = diff_atoms(row.agg_smiles, row.non_smiles)
        d_non = diff_atoms(row.non_smiles, row.agg_smiles)
        for c, (smi, col, soft, cls, hl, ik) in enumerate([
                (row.agg_smiles, AGG, AGG_SOFT, "aggregator", d_agg, row.agg_ik),
                (row.non_smiles, NON, NON_SOFT, "non-aggregator", d_non, row.non_ik)]):
            ax = fig.add_subplot(gs[r, c])
            ax.set_axis_off()
            panel(ax, col, face=soft)
            ax.imshow(mol_png(smi, highlight=hl, colour=col, size=(470, 290)),
                      extent=(0.05, 0.95, 0.15, 0.88), aspect="auto", zorder=3)
            ax.text(0.5, 0.075, cls, transform=ax.transAxes, ha="center",
                    fontsize=8.2, fontweight="bold", color=col)
            if c == 0:
                ax.text(0.062, 0.945, chr(97 + r), transform=ax.transAxes, ha="center",
                        va="center", fontsize=8.5, fontweight="bold", color="white",
                        zorder=7, bbox=dict(boxstyle="circle,pad=0.32", fc=INK, ec="none"))
                ax.text(0.60, 0.955, f"{row.dataset}  ·  Tc {row.tanimoto:.2f}  ·  "
                        f"{label_for(row)}", transform=ax.transAxes, ha="center",
                        fontsize=7.2, color=GREY)

        # verdict strip
        ax = fig.add_subplot(gs[r, 2])
        ax.set_axis_off()
        panel(ax, GREY, face=PANEL)
        ax.text(0.50, 0.90, "what the filters said", transform=ax.transAxes,
                ha="center", fontsize=7.8, fontweight="bold", color=INK)
        ax.text(0.615, 0.795, "agg", transform=ax.transAxes, fontsize=6.8,
                color=AGG, ha="center", style="italic", fontweight="bold")
        ax.text(0.855, 0.795, "non", transform=ax.transAxes, fontsize=6.8,
                color=NON, ha="center", style="italic", fontweight="bold")
        ax.plot([0.06, 0.95], [0.755, 0.755], transform=ax.transAxes,
                color="#D8D8D4", lw=0.8)

        pa, pn = badp.get(row.agg_ik), badp.get(row.non_ik)
        ca, cn = cagc.get(row.agg_ik), cagc.get(row.non_ik)
        va, vn = advv.get(row.agg_ik), advv.get(row.non_ik)
        cw = {0: "non", 1: "agg", None: "—"}
        lines = [
            ("BAD  p", f"{pa:.3f}" if pa is not None else "—",
             f"{pn:.3f}" if pn is not None else "—"),
            ("ChemAGG", cw.get(ca, "—"), cw.get(cn, "—")),
            ("Advisor", (va or "—").lower()[:8], (vn or "—").lower()[:8]),
            ("clogP", f"{row.agg_clogp:.2f}", f"{row.non_clogp:.2f}"),
        ]
        for i, (lab, la, ln) in enumerate(lines):
            y = 0.665 - i * 0.125
            same = la == ln
            ax.text(0.075, y, lab, transform=ax.transAxes, fontsize=7.0,
                    color=INK, va="center")
            for x, v in ((0.615, la), (0.855, ln)):
                ax.text(x, y, v, transform=ax.transAxes, fontsize=7.0,
                        color=(AGG if same else INK), ha="center", va="center",
                        fontweight="bold" if same else "normal")
        dp = abs((pa or 0) - (pn or 0))
        ax.text(0.5, 0.135, f"BAD moves by {dp:.3f}", transform=ax.transAxes,
                ha="center", fontsize=7.4, color=AGG, fontweight="bold")
        ax.text(0.5, 0.045, "sampled, not selected for flatness", transform=ax.transAxes,
                ha="center", fontsize=6.8, color=GREY)

    save(fig, "M7_cliff_gallery")
    print("\npairs drawn:")
    print(ex[["dataset", "tanimoto", "d_heavy", "d_mw", "d_clogp", "bad_d"]]
          .round(4).to_string())


if __name__ == "__main__":
    main()
