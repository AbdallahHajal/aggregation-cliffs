#!/usr/bin/env python
"""The two structure figures.

F4  the anchor pair, the only aggregation cliff measured by light scattering in both
    directions, with what each filter said about each member
F6  six mined pairs, sampled under the study seed rather than chosen for effect

Both members of a pair are laid out against their shared maximum common substructure, so
the common core is drawn identically and only the differing atoms move. Atoms and the bonds
that carry them are shaded together, because an added methyl highlighted as a lone atom
reads as a dot rather than as a substituent.
"""
import pathlib
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figstyle import AGG, AGG_SOFT, GREY, INK, NON, NON_SOFT, PANEL, panel, save  # noqa: E402
from pairdraw import draw_pair  # noqa: E402

NAVY = "#1F3A5F"
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5})
SEED = 20260928


def canvas_for(fig, gs_cell, extent, height=420):
    """Pixel size for a drawing canvas that fills `extent` of `gs_cell` without stretching.

    imshow with an extent and aspect="auto" will squash whatever it is given to the shape
    of the box, which changes bond angles. Sizing the RDKit canvas to the box first means
    the mapping is one to one.
    """
    ax = fig.add_subplot(gs_cell)
    pos = ax.get_position()
    ax.remove()
    fw, fh = fig.get_size_inches()
    box_w = pos.width * fw * (extent[1] - extent[0])
    box_h = pos.height * fh * (extent[3] - extent[2])
    return int(round(height * box_w / box_h)), height


def show(ax, img, extent):
    ax.imshow(np.asarray(img), extent=extent, aspect="auto", zorder=3)
    ax.set_xlim(0, 1)      # imshow otherwise autoscales the axes to the extent, which
    ax.set_ylim(0, 1)      # puts the drawing over the captions placed in axes coordinates


def render(smi_a, smi_b, size=(620, 380), stereo=False, width=1400):
    """Return two PIL images of an MCS-aligned pair."""
    sa, sb = draw_pair(smi_a, smi_b, size=size, stereo=stereo)
    out = []
    with tempfile.TemporaryDirectory() as td:
        for i, svg in enumerate((sa, sb)):
            p = pathlib.Path(td) / f"m{i}.svg"
            p.write_text(svg)
            png = p.with_suffix(".png")
            subprocess.run(["rsvg-convert", "-w", str(width), "-o", str(png), str(p)],
                           check=True)
            out.append(Image.open(png).copy())
    return out


EXT = (0.06, 0.94, 0.16, 0.86)


def fig_anchor():
    A, B = "Oc1c(I)cc(Cl)c2cccnc12", "COc1c(I)cc(Cl)c2cccnc12"

    fig = plt.figure(figsize=(7.3, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.12], wspace=0.07)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)
    ia, ib = render(A, B, size=canvas_for(fig, gs[0, 0], EXT))

    for k, (img, col, soft, name, cls, note) in enumerate([
        (ia, AGG, AGG_SOFT, "Clioquinol", "aggregator", "CAC 15 µM   radius 1818 nm"),
        (ib, NON, NON_SOFT, "Methylclioquinol", "non-aggregator",
         "no scattering, 0.5 to 100 µM"),
    ]):
        ax = fig.add_subplot(gs[0, k])
        ax.set_axis_off()
        panel(ax, col, face=soft)
        show(ax, img, EXT)
        ax.text(0.5, 0.925, name, transform=ax.transAxes, ha="center", fontsize=10,
                fontweight="bold", color=col)
        ax.text(0.5, 0.095, cls, transform=ax.transAxes, ha="center", fontsize=9,
                fontweight="bold", color=col)
        ax.text(0.5, 0.035, note, transform=ax.transAxes, ha="center", fontsize=7.6,
                color=GREY)

    ax = fig.add_subplot(gs[0, 2])
    ax.set_axis_off()
    panel(ax, GREY, face=PANEL)
    ax.text(0.5, 0.925, "one added heavy atom", transform=ax.transAxes, ha="center",
            fontsize=9.6, fontweight="bold", color=INK)
    ax.text(0.66, 0.835, "clioq.", transform=ax.transAxes, ha="center", fontsize=7.4,
            color=AGG, style="italic", fontweight="bold")
    ax.text(0.88, 0.835, "methyl", transform=ax.transAxes, ha="center", fontsize=7.4,
            color=NON, style="italic", fontweight="bold")
    ax.plot([0.06, 0.95], [0.79, 0.79], transform=ax.transAxes, color="#D8D8D4", lw=0.9)
    rows = [("Aggregator Advisor", "known", "—", GREY),
            ("BAD Molecule Filter", "0.132", "—", AGG),
            ("ChemAGG", "0.041", "0.038", AGG),
            ("DeepSCAMs", "0.339", "0.249", AGG),   # results/anchor_deepscams.csv
            ("clogP > 3.5", "3.20", "3.50", AGG)]
    for i, (lab, x1, x2, col) in enumerate(rows):
        y = 0.715 - i * 0.105
        ax.text(0.06, y, lab, transform=ax.transAxes, fontsize=7.4, color=INK, va="center")
        ax.text(0.66, y, x1, transform=ax.transAxes, fontsize=7.4, color=col,
                ha="center", va="center", fontweight="bold")
        ax.text(0.88, y, x2, transform=ax.transAxes, fontsize=7.4, color=col,
                ha="center", va="center", fontweight="bold")
    ax.text(0.5, 0.145, "every filter places both members\nin the same class",
            transform=ax.transAxes, ha="center", fontsize=7.8, color=AGG,
            fontweight="bold")
    ax.text(0.5, 0.045, "the null model separates them, and\nassigns both to the wrong one",
            transform=ax.transAxes, ha="center", fontsize=7.0, color=GREY)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)
    save(fig, "Fig05_anchor_pair")
    plt.close(fig)


def fig_gallery():
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    bad = pd.read_csv(ROOT / "results/pairs_sample_500_bad.csv")
    cag = pd.read_csv(ROOT / "results/pairs_sample_500_chemagg.csv")
    bp = dict(zip(bad.inchikey, bad.bad_prob))
    cp = dict(zip(cag.inchikey, cag.chemagg_cat))

    def pick(df, **n_by_dataset):
        """Sample within each screening set, so the gallery is not drawn from one assay."""
        return pd.concat([df[df.dataset == ds].sample(k, random_state=SEED)
                          for ds, k in n_by_dataset.items()])

    s = s[(s.agg_heavy <= 32) & (s.non_heavy <= 32)]
    iso = pick(s[(s.d_heavy == 0) & (s.d_clogp.abs() < 1e-9)], cruzain=1, ampc=1)
    ch2 = pick(s[(s.d_heavy == 1) & (s.d_mw.abs().between(13.9, 14.1))], cruzain=1, ampc=2)
    oth = pick(s[(s.d_heavy == 1) & (~s.d_mw.abs().between(13.9, 14.1))], cruzain=1)
    ex = pd.concat([iso, ch2, oth]).reset_index(drop=True)

    # Three blocks by two rather than six rows: with two structures across a 7.3 inch page,
    # a six-row stack leaves each molecule a box twice as wide as it is tall, and the
    # structures shrink to fit the height. Blocks give each one a near-square box.
    ncol, nrow = 2, 3
    fig = plt.figure(figsize=(7.3, 8.3))
    outer = fig.add_gridspec(nrow, ncol, wspace=0.09, hspace=0.30)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.935, bottom=0.010)
    gext = (0.05, 0.95, 0.15, 0.94)
    inner0 = outer[0, 0].subgridspec(2, 2, height_ratios=[1, 0.40], hspace=0.10,
                                     wspace=0.045)
    csize = canvas_for(fig, inner0[0, 0], gext)
    cw = {0: "non-agg", 1: "agg"}

    for r, row in ex.iterrows():
        inner = outer[r // ncol, r % ncol].subgridspec(
            2, 2, height_ratios=[1, 0.40], hspace=0.10, wspace=0.045)
        ia, ib = render(row.agg_smiles, row.non_smiles, size=csize,
                        stereo=bool(row.d_heavy == 0))
        for c, (img, col, soft, cls) in enumerate([
                (ia, AGG, AGG_SOFT, "aggregator"),
                (ib, NON, NON_SOFT, "non-aggregator")]):
            ax = fig.add_subplot(inner[0, c])
            ax.set_axis_off()
            panel(ax, col, face=soft)
            show(ax, img, gext)
            ax.text(0.5, 0.068, cls, transform=ax.transAxes, ha="center", fontsize=8,
                    fontweight="bold", color=col)
            if c == 0:
                tag = ("same formula, same clogP" if row.d_heavy == 0
                       else ("one methylene" if abs(abs(row.d_mw) - 14.03) < 0.1
                             else f"+1 heavy, ΔMW {row.d_mw:+.1f}"))
                ax.text(0.015, 1.10, chr(97 + r), transform=ax.transAxes, fontsize=9.5,
                        fontweight="bold", color="white", ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.3", fc=INK, ec="none"), zorder=6)
                ax.text(0.10, 1.10, f"{row.dataset}   ·   Tc {row.tanimoto:.2f}   "
                        f"·   {tag}", transform=ax.transAxes, ha="left", va="center",
                        fontsize=7.4, color=GREY)

        ax = fig.add_subplot(inner[1, :])
        ax.set_axis_off()
        pa, pn = bp.get(row.agg_ik), bp.get(row.non_ik)
        ca, cn = cp.get(row.agg_ik), cp.get(row.non_ik)
        lines = [("BAD  p", f"{pa:.3f}" if pa is not None else "—",
                  f"{pn:.3f}" if pn is not None else "—"),
                 ("ChemAGG", cw.get(ca, "—"), cw.get(cn, "—")),
                 ("clogP", f"{row.agg_clogp:.2f}", f"{row.non_clogp:.2f}")]
        for i, (lab, v1, v2) in enumerate(lines):
            y = 0.80 - i * 0.30
            same = v1 == v2
            col = AGG if same else INK
            ax.text(0.15, y, lab, transform=ax.transAxes, fontsize=7.2, color=GREY,
                    va="center")
            ax.text(0.52, y, v1, transform=ax.transAxes, fontsize=7.2, color=col,
                    ha="center", va="center", fontweight="bold" if same else "normal")
            ax.text(0.80, y, v2, transform=ax.transAxes, fontsize=7.2, color=col,
                    ha="center", va="center", fontweight="bold" if same else "normal")

    fig.text(0.012, 0.995, "Six pairs from the census, drawn against their shared core, "
             "with what three filters said about each member",
             ha="left", va="top", fontsize=9.6, fontweight="bold", color=INK)
    save(fig, "Fig04_cliff_gallery")
    plt.close(fig)
    print("\npairs drawn:")
    print(ex[["dataset", "tanimoto", "d_heavy", "d_mw"]].round(3).to_string(index=False))


if __name__ == "__main__":
    fig_anchor()
    fig_gallery()
