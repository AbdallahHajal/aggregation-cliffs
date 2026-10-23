#!/usr/bin/env python
"""Figure 6. Published filters do not separate the members of a cliff pair.

(a) The anchor pair, clioquinol and methylclioquinol, the only aggregation cliff with a
    direct light-scattering measurement in both directions, with every filter verdict.
(b) Three census pairs the filters could not separate, one descriptor-identical isomer
    pair and two single-methylene pairs.
(c) Miss rate per filter over the 500 sampled pairs.

House style from figstyle.py, following Sturm et al. Angew Chem 2025.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figstyle import AGG, AGG_SOFT, GREY, INK, NON, NON_SOFT, PANEL, badge, mol_png, panel, panel_title, save  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ANCHOR = [
    ("Clioquinol", "Oc1c(I)cc(Cl)c2cccnc12", AGG, "aggregator",
     "CAC 15 µM, radius 1818 nm"),
    ("Methylclioquinol", "COc1c(I)cc(Cl)c2cccnc12", NON, "non-aggregator",
     "no scattering, 0.5 to 100 µM"),
]

ANCHOR_VERDICTS = [
    ("Aggregator Advisor", "known", "—", "lookup"),
    ("BAD Molecule Filter", "0.132", "—", "wrong"),
    ("ChemAGG", "0.041", "0.038", "wrong"),
    ("clogP > 3.5", "3.20", "3.50", "wrong"),
]


def diff_atoms(a, b):
    """Atoms of a that are outside the maximum common substructure with b."""
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return []
    res = rdFMCS.FindMCS([ma, mb], timeout=10, ringMatchesRingOnly=True,
                         completeRingsOnly=False)
    if not res.smartsString:
        return []
    patt = Chem.MolFromSmarts(res.smartsString)
    match = ma.GetSubstructMatch(patt)
    return [i for i in range(ma.GetNumAtoms()) if i not in set(match)]


def pick_examples(n_iso=1, n_ch2=2):
    """Pairs from the sample that every scored filter called the same way."""
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    adv = pd.read_csv(ROOT / "results/pairs_sample_500_advisor.csv")
    av = dict(zip(adv["inchikey"], adv["advisor_flagged"]))
    s["adv_same"] = [av.get(a) == av.get(b) and av.get(a) is not None
                     for a, b in zip(s["agg_ik"], s["non_ik"])]
    s = s[s["adv_same"]]
    iso = s[(s["d_heavy"] == 0) & (s["d_clogp"].abs() < 1e-9)].head(n_iso)
    ch2 = s[(s["d_heavy"] == 1) & (s["d_mw"].abs().between(13.9, 14.1))].head(n_ch2)
    return pd.concat([iso, ch2])


def verdict_line(ax, y, label, left, right, kind):
    col = {"wrong": AGG, "lookup": GREY, "right": NON}[kind]
    ax.text(0.055, y, label, transform=ax.transAxes, fontsize=7.2,
            color=INK, va="center", ha="left")
    ax.text(0.715, y, left, transform=ax.transAxes, fontsize=7.4,
            color=col, va="center", ha="center", fontweight="bold")
    ax.text(0.895, y, right, transform=ax.transAxes, fontsize=7.4,
            color=col, va="center", ha="center", fontweight="bold")


def main():
    ex = pick_examples()
    fig = plt.figure(figsize=(7.3, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.28, 1.42, 0.72], hspace=0.16)

    # ---------------- (a) the anchor pair ----------------
    gsa = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.06)
    d_a = diff_atoms(ANCHOR[1][1], ANCHOR[0][1])
    for k, (name, smi, col, cls, note) in enumerate(ANCHOR):
        ax = fig.add_subplot(gsa[0, k])
        ax.set_axis_off()
        panel(ax, col, face=(AGG_SOFT if col == AGG else NON_SOFT))
        hl = d_a if k == 1 else []
        ax.imshow(mol_png(smi, highlight=hl, colour=col, size=(430, 300)),
                  extent=(0.05, 0.95, 0.14, 0.80), aspect="auto", zorder=3)
        panel_title(ax, name, col)
        ax.text(0.5, 0.085, cls, transform=ax.transAxes, ha="center",
                fontsize=8.4, fontweight="bold", color=col)
        ax.text(0.5, 0.030, note, transform=ax.transAxes, ha="center",
                fontsize=7.2, color=GREY)
        if k == 0:
            badge(ax, 0.072, 0.915, "a", INK)

    axv = fig.add_subplot(gsa[0, 2])
    axv.set_axis_off()
    panel(axv, GREY, face=PANEL)
    axv.text(0.5, 0.93, "one added heavy atom", transform=axv.transAxes,
             ha="center", fontsize=8.6, fontweight="bold", color=INK)
    axv.text(0.715, 0.825, "clioq.", transform=axv.transAxes, fontsize=7.0,
             color=GREY, ha="center", style="italic")
    axv.text(0.895, 0.825, "methyl", transform=axv.transAxes, fontsize=7.0,
             color=GREY, ha="center", style="italic")
    axv.plot([0.055, 0.955], [0.775, 0.775], transform=axv.transAxes,
             color="#D8D8D4", lw=0.9)
    for i, (lab, left, right, kind) in enumerate(ANCHOR_VERDICTS):
        verdict_line(axv, 0.695 - i * 0.125, lab, left, right, kind)
    axv.text(0.5, 0.155, "ChemAGG calls both non-aggregators.\nThe clogP null separates them, backwards.",
             transform=axv.transAxes, ha="center", fontsize=7.4,
             color=AGG, fontweight="bold")
    axv.text(0.5, 0.045, "Advisor is right on clioquinol only by exact-structure\n"
             "lookup. The null flips class on 0.001 log units.",
             transform=axv.transAxes, ha="center", fontsize=6.6, color=GREY)

    # ---------------- (b) census pairs ----------------
    gsb = gs[1].subgridspec(len(ex), 2, wspace=0.05, hspace=0.12)
    for r, (_, row) in enumerate(ex.iterrows()):
        d_non = diff_atoms(row["non_smiles"], row["agg_smiles"])
        d_agg = diff_atoms(row["agg_smiles"], row["non_smiles"])
        for c, (smi, col, cls, hl) in enumerate([
                (row["agg_smiles"], AGG, "aggregator", d_agg),
                (row["non_smiles"], NON, "non-aggregator", d_non)]):
            ax = fig.add_subplot(gsb[r, c])
            ax.set_axis_off()
            panel(ax, col, face=(AGG_SOFT if col == AGG else NON_SOFT))
            ax.imshow(mol_png(smi, highlight=hl, colour=col, size=(430, 250)),
                      extent=(0.06, 0.94, 0.16, 0.90), aspect="auto", zorder=3)
            ax.text(0.5, 0.075, cls, transform=ax.transAxes, ha="center",
                    fontsize=7.8, fontweight="bold", color=col)
            if c == 0:
                if r == 0:
                    badge(ax, 0.072, 0.90, "b", INK)
                tag = ("identical formula, identical clogP"
                       if row["d_heavy"] == 0 else "one methylene")
                ax.text(0.58, 0.955, f"{row['dataset']} · Tc {row['tanimoto']:.2f} "
                        f"· {tag}", transform=ax.transAxes, ha="center",
                        fontsize=7.0, color=GREY)

    # ---------------- (c) miss rates ----------------
    axc = fig.add_subplot(gs[2])
    mr = pd.read_csv(ROOT / "results/RESULT2_miss_rates.csv")
    mr = mr.sort_values("missed_pct")
    y = np.arange(len(mr))
    axc.barh(y, mr["missed_pct"], color=AGG, height=0.56, zorder=3)
    axc.errorbar(mr["missed_pct"], y,
                 xerr=[mr["missed_pct"] - mr["ci_lo"], mr["ci_hi"] - mr["missed_pct"]],
                 fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=4)
    for i, (_, r) in enumerate(mr.iterrows()):
        axc.text(r["missed_pct"] - 1.6, i, f"{r['missed_pct']:.1f}%", va="center",
                 ha="right", fontsize=8.2, color="white", fontweight="bold", zorder=5)
        axc.text(101, i, f"n = {int(r['n'])}", va="center", ha="left",
                 fontsize=7.4, color=GREY)
    axc.set_yticks(y)
    axc.set_yticklabels(mr["filter"], fontsize=8.4)
    axc.set_xlim(0, 108)
    axc.set_xlabel("pairs given the same verdict for both members (%)", fontsize=8.4)
    axc.set_title("c   filters do not separate the members of a cliff pair",
                  fontsize=9.2, fontweight="bold", loc="left", color=INK)
    axc.spines[["top", "right"]].set_visible(False)
    axc.grid(axis="x", color="#E6E6E2", zorder=0)
    axc.set_axisbelow(True)

    save(fig, "M6_filters_on_cliffs")
    print("\nexample pairs drawn in panel b:")
    print(ex[["dataset", "tanimoto", "d_heavy", "d_mw", "d_clogp"]].to_string(index=False))


if __name__ == "__main__":
    main()
