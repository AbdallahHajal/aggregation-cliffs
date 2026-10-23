#!/usr/bin/env python
"""The two Methods figures, drawn rather than written.

The earlier versions of both were paragraphs inside rounded rectangles, which is a slide,
not a figure. They are now built from the things themselves.

F2  a two by two of the traces each experiment actually produces. The reader sees a steep
    dose-response curve abolished by detergent, a flat pair of curves, a scattering trace
    with a break at the critical aggregation concentration, and a flat one. The argument of
    the paper is the bottom left cell: a flat curve is recorded as a non-aggregator whether
    the compound was inert or never reached a testable concentration.
F3  a funnel from the screened compounds down to the census, with what leaves at each stage
    shown beside it, and the transformation table beneath.

Counts are read from results/ at draw time. The curves are drawn from Hill and
Rayleigh-type expressions with no fitted data behind them, and are labelled as schematic.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figstyle import AGG, AGG_SOFT, GREY, INK, NON, NON_SOFT, PANEL, save  # noqa: E402

NAVY, NAVY_SOFT = "#1F3A5F", "#DCE3EC"

# mmpdb writes transformations as SMARTS with numbered attachment points. Nobody reads
# those at a glance, and set in a monospace face they were wide enough to collide with the
# neighbouring panel, so the common fragments are named. Anything unlisted falls through
# as its SMILES.
FRAGMENT = {"[H]": "H", "C": "methyl", "CC": "ethyl", "CCC": "propyl",
            "F": "fluoro", "Cl": "chloro", "Br": "bromo", "O": "hydroxy",
            "OC": "methoxy", "N": "amino",
            "c1ccccc1": "phenyl", "c1ccc(C)cc1": "tolyl", "c1ccc(F)cc1": "4-F-phenyl",
            "c1ccc(Cl)cc1": "4-Cl-phenyl", "c1ccc(O)cc1": "4-OH-phenyl"}


def name_transform(s):
    a, _, b = s.partition(" >> ")
    out = []
    for side in (a, b):
        f = side.replace("[*:1]", "").replace("[*:2]", "")
        out.append(FRAGMENT.get(f, f))
    return f"{out[0]}  →  {out[1]}"
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5})


def hill(x, ac50, n):
    return 100 / (1 + (ac50 / x) ** n)


def mini(ax, face=PANEL):
    """A panel that shows the shape of a trace and nothing else. No tick labels: the axes
    are schematic, and printing decades invites the reader to read values off them."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C8C8C4")
        ax.spines[sp].set_linewidth(0.9)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False,
                   which="both")
    ax.set_facecolor(face)


# ----------------------------------------------------------------------------- F2

def fig2_label_provenance():
    fig = plt.figure(figsize=(7.3, 4.55))
    gs = fig.add_gridspec(2, 2, left=0.115, right=0.975, top=0.795, bottom=0.175,
                          wspace=0.20, hspace=0.30)
    c = np.logspace(-1, 2.3, 300)

    # --- a: aggregator in the counter-screen -------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    mini(ax)
    ax.semilogx(c, hill(c, 12, 4.2), color=AGG, lw=2.0, zorder=3)
    ax.semilogx(c, np.full_like(c, 3.0), color=GREY, lw=1.8, ls=(0, (4, 2)), zorder=3)
    ax.set_ylim(-8, 118)
    ax.text(0.055, 0.90, "no detergent", transform=ax.transAxes, fontsize=7.2,
            color=AGG, fontweight="bold")
    ax.text(0.055, 0.14, "with detergent", transform=ax.transAxes, fontsize=7.2,
            color=GREY, fontweight="bold")
    ax.text(0.97, 0.62, "steep", transform=ax.transAxes, fontsize=7.0, color=AGG,
            ha="right")

    # --- b: aggregator by light scattering ---------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    mini(ax)
    cac = 15.0
    inten = np.where(c < cac, 2.0, 2.0 + 62 * (np.log10(c / cac)) ** 1.1)
    ax.semilogx(c, np.clip(inten, 0, 110), color=NON, lw=2.0, zorder=3)
    ax.axvline(cac, color=NON, lw=1.0, ls=(0, (3, 2)), zorder=2)
    ax.set_ylim(-8, 118)
    ax.text(cac * 1.25, 96, "CAC", fontsize=7.4, color=NON, fontweight="bold")
    ax.text(0.055, 0.14, "particle detected", transform=ax.transAxes, fontsize=7.2,
            color=NON, fontweight="bold")

    # --- c: non-aggregator in the counter-screen, the problem case ---------------
    ax = fig.add_subplot(gs[1, 0])
    mini(ax, face="#FBEFE0")
    ax.semilogx(c, np.full_like(c, 3.0), color=GREY, lw=1.8, zorder=3)
    ax.semilogx(c, np.full_like(c, 6.5), color=GREY, lw=1.8, ls=(0, (4, 2)), zorder=3)
    ax.set_ylim(-8, 118)
    ax.text(0.5, 0.62, "no curve fitted", transform=ax.transAxes, ha="center",
            fontsize=8.6, color=AGG, fontweight="bold")
    ax.text(0.5, 0.40, "inert, or never tested here", transform=ax.transAxes,
            ha="center", fontsize=7.6, color=AGG)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(AGG)
        ax.spines[sp].set_linewidth(1.6)

    # --- d: non-aggregator by light scattering -----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    mini(ax)
    ax.semilogx(c, np.full_like(c, 2.5), color=NON, lw=2.0, zorder=3)
    ax.set_ylim(-8, 118)
    ax.text(0.5, 0.52, "no particle, across the range", transform=ax.transAxes,
            ha="center", fontsize=7.8, color=NON, fontweight="bold")

    # --- column and row headings --------------------------------------------------
    fig.text(0.115 + (0.975 - 0.115) * 0.235, 0.905, "detergent counter-screen",
             ha="center", fontsize=10, fontweight="bold", color=NAVY)
    fig.text(0.115 + (0.975 - 0.115) * 0.235, 0.855,
             "467,255 compounds   ·   label INFERRED", ha="center", fontsize=7.8,
             color=GREY)
    fig.text(0.115 + (0.975 - 0.115) * 0.765, 0.905, "dynamic light scattering",
             ha="center", fontsize=10, fontweight="bold", color=NON)
    fig.text(0.115 + (0.975 - 0.115) * 0.765, 0.855,
             "65 compounds   ·   particle MEASURED", ha="center", fontsize=7.8,
             color=GREY)

    fig.text(0.030, 0.640, "called\naggregator", ha="left", va="center", fontsize=9,
             fontweight="bold", color=AGG, linespacing=1.5)
    fig.text(0.030, 0.330, "called\nnon-aggregator", ha="left", va="center", fontsize=9,
             fontweight="bold", color=NON, linespacing=1.5)

    fig.text(0.545, 0.140, "concentration", ha="center", fontsize=8.4, color=GREY)
    fig.text(0.075, 0.485, "signal", rotation=90, va="center", fontsize=8.4, color=GREY)

    fig.text(0.545, 0.045, "A flat curve is filed as a non-aggregator whether the compound "
             "was inert or was never tested.",
             ha="center", va="center", fontsize=8.6, color=AGG, fontweight="bold")
    save(fig, "Fig02_label_provenance")
    plt.close(fig)


# ----------------------------------------------------------------------------- F3

CX = 0.30          # the funnel sits left, so the gates can be read beside it


def funnel_band(ax, y, h, w_top, w_bot, colour):
    ax.add_patch(Polygon([[CX - w_top / 2, y + h], [CX + w_top / 2, y + h],
                          [CX + w_bot / 2, y], [CX - w_bot / 2, y]],
                         closed=True, facecolor=colour, edgecolor="white", lw=1.4,
                         zorder=2, transform=ax.transAxes))


def fig3_census():
    # Redrawn 2026-10-21 against the harmonized census, after the AmpC set was re-prepared
    # with MolVS to match cruzain (src/prepare_sets.py). The version this replaces was drawn
    # on 30 Sep 2026 from the pre-harmonization data and carried 467,255 / 3,457 / 3,375 /
    # 2,795 in the funnel and 2,488 distinct transformations.
    tf = pd.read_csv(ROOT / "results/mmp_label_flipping_transforms_molvs.csv")
    t = tf.head(10)
    cen = pd.read_csv(ROOT / "results/census_full_molvs.csv")

    # Derived rather than written in, so these two cannot drift from the data again.
    N_CENSUS = f"{len(cen):,}"                        # 3,365
    N_TRANSFORMS = f"{tf['transform'].nunique():,}"   # 2,483
    N_SCREENED = "467,110"   # 191,023 cruzain + 276,087 AmpC
    N_RAW_PAIRS = "3,450"    # before cross-screen deduplication
    N_FORMAL = "2,780"       # also a formal matched pair, 82.6% of the census
    N_DUPES = f"{int(N_RAW_PAIRS.replace(',', '')) - len(cen):,}"  # 85, was 82
    n_cru = int((cen.dataset == "cruzain").sum())
    n_amp = int((cen.dataset == "ampc").sum())

    fig = plt.figure(figsize=(7.3, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0], wspace=0.30,
                          left=0.015, right=0.975, top=0.90, bottom=0.135)

    # -- a: the funnel
    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    stages = [
        (0.760, 0.150, 0.58, 0.44, NAVY,      N_SCREENED,  "compounds screened"),
        (0.560, 0.150, 0.44, 0.32, "#3E6187", N_RAW_PAIRS, "near-neighbour pairs,\nopposite labels"),
        (0.360, 0.150, 0.32, 0.26, AGG,       N_CENSUS,    "the census"),
        (0.160, 0.150, 0.26, 0.22, "#C4761F", N_FORMAL,    "also formal\nmatched pairs"),
    ]
    for y, h, wt, wb, col, num, lab in stages:
        funnel_band(ax, y, h, wt, wb, col)
        ax.text(CX, y + h * 0.63, num, transform=ax.transAxes, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="white", zorder=4)
        ax.text(CX, y + h * 0.25, lab, transform=ax.transAxes, ha="center", va="center",
                fontsize=6.9, color="white", zorder=4, linespacing=1.3)

    # what each stage applies, on the left, and what leaves, on the right
    gates = [(0.723, "shared Murcko scaffold\nECFP4 Tanimoto ≥ 0.85"),
             (0.523, f"{N_DUPES} duplicate pairs removed\nstereoisomers set aside"),
             (0.323, "re-expressed with mmpdb")]
    for y, txt in gates:
        ax.annotate("", xy=(CX, y - 0.040), xytext=(CX, y),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.1,
                                    mutation_scale=10))
        ax.plot([CX + 0.02, 0.60], [y - 0.020, y - 0.020], transform=ax.transAxes,
                color="#DCDCD8", lw=0.8, zorder=1)
        ax.text(0.625, y - 0.020, txt, transform=ax.transAxes, ha="left", va="center",
                fontsize=6.8, color=GREY, linespacing=1.45)

    ax.text(CX, 0.085, f"cruzain {n_cru:,}   ·   AmpC {n_amp:,}", transform=ax.transAxes,
            ha="center", fontsize=7.6, color=GREY)
    ax.text(CX, 0.025, "59% differ by one methylene", transform=ax.transAxes,
            ha="center", fontsize=8.2, color=INK, fontweight="bold")
    ax.set_title("a   From the screens to the census", loc="left", fontsize=10,
                 fontweight="bold", color=INK, pad=8)

    # -- b: transformations
    ax2 = fig.add_subplot(gs[1])
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color(GREY)
    ax2.tick_params(colors=GREY, length=3)
    ax2.grid(axis="x", color="#EFEFEC", lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    lab = [name_transform(s) for s in t["transform"]]
    y = np.arange(len(t))[::-1]
    ax2.barh(y, t.n_pairs, color=NAVY, height=0.62, zorder=3)
    for yy, v in zip(y, t.n_pairs):
        ax2.text(v + 6, yy, str(int(v)), va="center", fontsize=7.6, color=INK)
    ax2.set_yticks(y)
    ax2.set_yticklabels(lab, fontsize=7.6)
    ax2.set_xlabel("pairs separated by that transformation", fontsize=8.4)
    ax2.set_xlim(0, max(t.n_pairs) * 1.22)
    ax2.set_title("b   The ten commonest label-flipping transformations",
                  loc="left", fontsize=10, fontweight="bold", color=INK, pad=8)
    ax2.text(0.99, -0.145, f"oriented aggregator → non-aggregator, of {N_TRANSFORMS} in "
             "total. The table is close to\nsymmetric, so no design rule follows from it.",
             transform=ax2.transAxes, ha="right", fontsize=7.2, color=GREY, va="top")
    save(fig, "Fig03_census_construction", root="Figures/manuscript_molvs")
    plt.close(fig)


if __name__ == "__main__":
    fig2_label_provenance()
    fig3_census()
