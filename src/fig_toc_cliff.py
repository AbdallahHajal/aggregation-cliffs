#!/usr/bin/env python
"""The table of contents graphic, drawn from the real anchor pair.

The generated version of this made up its molecules, which is fine for the mechanism
schematic and not fine here, because the whole point is that these two specific compounds
were measured. Clioquinol and its O-methyl ether are drawn from SMILES and aligned to their
shared core, the colloid is a glyph rather than a depiction of anything measured, and the
filter row is read from the same results the paper reports.
"""
import pathlib
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, RegularPolygon
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figstyle import AGG, AGG_SOFT, GREY, INK, NON, NON_SOFT, save  # noqa: E402
from pairdraw import draw_pair  # noqa: E402

NAVY = "#1F3A5F"
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5})

CLIOQUINOL = "Oc1c(I)cc(Cl)c2cccnc12"
METHYL = "COc1c(I)cc(Cl)c2cccnc12"
SEED = 20260930


def render(a, b, size, width=1600, bond_px=34):
    # transparent, so the tinted panel shows through instead of a white slab
    sa, sb = draw_pair(a, b, size=size, opaque=False, bond_px=bond_px)
    out = []
    with tempfile.TemporaryDirectory() as td:
        for i, svg in enumerate((sa, sb)):
            p = pathlib.Path(td) / f"m{i}.svg"
            p.write_text(svg)
            png = p.with_suffix(".png")
            subprocess.run(["rsvg-convert", "-w", str(width), "-b", "none",
                            "-o", str(png), str(p)],
                           check=True)
            out.append(Image.open(png).copy())
    return out


def hexes(ax, xy, n, r, spread, colour, rng, alpha=1.0):
    """Scatter n small hexagons around xy. Used for both the colloid and the free compound,
    so the only difference the reader sees is how tightly they are packed."""
    for _ in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        rad = spread * np.sqrt(rng.uniform(0, 1))
        ax.add_patch(RegularPolygon((xy[0] + rad * np.cos(ang), xy[1] + rad * np.sin(ang)),
                                    6, radius=r, orientation=rng.uniform(0, 1),
                                    facecolor=colour, edgecolor="white", linewidth=0.5,
                                    alpha=alpha, zorder=4))


def main():
    # one pair on a whole graphic, so the structures can be drawn larger than in
    # the multi-panel figures
    ia, ib = render(CLIOQUINOL, METHYL, size=(520, 520), bond_px=62)
    rng = np.random.default_rng(SEED)

    fig = plt.figure(figsize=(7.0, 2.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    for x0, w, col, soft in ((0.015, 0.455, AGG, AGG_SOFT), (0.530, 0.455, NON, NON_SOFT)):
        ax.add_patch(FancyBboxPatch((x0, 0.245), w, 0.735,
                                    boxstyle="round,pad=0,rounding_size=0.022",
                                    facecolor=soft, edgecolor=col, linewidth=1.6, zorder=0))

    # --- structures, on one scale, with the shared core drawn identically ---
    for img, x0 in ((ia, 0.035), (ib, 0.550)):
        ax.imshow(np.asarray(img), extent=(x0, x0 + 0.250, 0.290, 0.895), aspect="auto",
                  zorder=3)

    ax.text(0.157, 0.925, "clioquinol", ha="center", fontsize=10.5, fontweight="bold",
            color=AGG)
    ax.text(0.672, 0.925, "its O-methyl ether", ha="center", fontsize=10.5,
            fontweight="bold", color=NON)

    # --- what each one does in water ---
    hexes(ax, (0.378, 0.680), 30, 0.0105, 0.050, AGG, rng)
    ax.add_patch(Circle((0.378, 0.680), 0.064, facecolor="none", edgecolor=AGG,
                        linewidth=1.1, linestyle=(0, (3, 2)), zorder=5))
    ax.text(0.378, 0.545, "colloid", ha="center", fontsize=9, fontweight="bold", color=AGG)
    ax.text(0.378, 0.482, "CAC 15 µM", ha="center", fontsize=7.6, color=INK)
    ax.text(0.378, 0.430, "radius 1818 nm", ha="center", fontsize=7.6, color=INK)

    hexes(ax, (0.888, 0.680), 30, 0.0105, 0.080, NON, rng)
    ax.text(0.888, 0.545, "dispersed", ha="center", fontsize=9, fontweight="bold", color=NON)
    ax.text(0.888, 0.482, "no scattering", ha="center", fontsize=7.6, color=INK)
    ax.text(0.888, 0.430, "0.5 to 100 µM", ha="center", fontsize=7.6, color=INK)

    # --- the transformation ---
    ax.add_patch(FancyArrowPatch((0.472, 0.575), (0.528, 0.575), arrowstyle="-|>",
                                 mutation_scale=13, color=INK, lw=1.4, zorder=6))
    ax.text(0.500, 0.655, "one added", ha="center", fontsize=8.2, fontweight="bold",
            color=INK)
    ax.text(0.500, 0.605, "carbon", ha="center", fontsize=8.2, fontweight="bold", color=INK)

    # --- what the filters return, from results/RESULT2_miss_rates_all.csv ---
    ax.add_patch(FancyBboxPatch((0.015, 0.020), 0.970, 0.190,
                                boxstyle="round,pad=0,rounding_size=0.018",
                                facecolor="#F2F2EF", edgecolor="#D8D8D4", linewidth=1.0,
                                zorder=0))
    ax.text(0.500, 0.152, "Aggregator Advisor   ·   BAD Molecule Filter   ·   ChemAGG   ·   "
            "DeepSCAMs   ·   CAGE-Fusion   ·   ChemFH",
            ha="center", va="center", fontsize=8.0, color=INK)
    ax.text(0.500, 0.073, "every one returns the same class for both compounds",
            ha="center", va="center", fontsize=9.0, fontweight="bold", color=AGG)

    save(fig, "TOC_aggregation_cliff")
    plt.close(fig)


if __name__ == "__main__":
    main()
