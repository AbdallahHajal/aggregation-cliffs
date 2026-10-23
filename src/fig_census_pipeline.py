#!/usr/bin/env python
"""Figure S1, the census construction flowchart, rebuilt as reproducible source.

The version in the manuscript ("figure_2_pipeline.png") was produced outside the project and
no source for it survives anywhere in the tree, so it could not be corrected when the AmpC set
was re-prepared with MolVS (src/prepare_sets.py). This redraws the same flowchart, with the
same boxes, order and colouring, and fixes what was wrong with it:

  * cruzain was given as 191,499, the as-deposited count, beside AmpC's cleaned count and a
    cleaned aggregator percentage. Both sets are now the cleaned counts the Methods states,
    191,023 and 276,087.
  * "merge & standardize (MolVS /" was clipped mid-sentence.
  * "82.8% also satisfy the formal transformation definition" overflowed its box and the last
    word was hidden behind the box below it.
  * six numbers moved with the harmonization: 276,232, 3,457, 3,375, 21.4%, 82.8%.

Counts are module constants rather than read at draw time, because several of them (the raw
pair count before cross-screen deduplication, the label-error rates) are produced by three
different scripts and there is no single file holding them all. Each is sourced in a comment.

    python src/fig_census_pipeline.py

Writes Figures/manuscript_molvs/FigS1_census_pipeline.{png,pdf,svg} at 600 dpi.
"""
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "Figures/manuscript_molvs"

# ---------------------------------------------------------------- the numbers
N_CRUZAIN = "191,023"     # Methods, cleaned; _outputs_molvs_baselines/cruzain_cleaned.csv
PCT_CRUZAIN = "13.2%"
N_AMPC = "276,087"        # data/AmpC/ampc_cleaned_molvs.csv, after MolVS and dedup
PCT_AMPC = "10.2%"
N_RAW = "3,450"           # cruzain 1,659 + AmpC 1,791 tight pairs
N_CENSUS = "3,365"        # after removing the 85 cross-screen duplicates
ERR_AMPC = "21.5%"        # 239 of 1,110 stereoisomer pairs, logs/base_rate_ampc_molvs.log
ERR_CRUZAIN = "49.6%"     # 164 of 331, unchanged; logs/base_rate_cruzain_fixed.log
PCT_FORMAL = "82.6%"      # 2,780 of 3,365; src/rebuild_mmp.py
PCT_METHYLENE = "59%"     # |dMW - 14.03| < 0.1 over the census

# ---------------------------------------------------------------- palette
BLUE, BLUE_F = "#2272B0", "#E8F1F8"
BLUE_D = "#1A5party"  # placeholder replaced below
BLUE_D = "#1A5A8A"
ORANGE, ORANGE_F = "#D97A2B", "#FBEBDC"
GREY, GREY_F = "#8E8E88", "#F1F0EC"
GOLD, GOLD_F = "#DDA02C", "#FCF4E0"
GREEN, GREEN_F = "#2E9E6B", "#E7F4EE"
INK = "#1A1A1A"
SOFT = "#555555"

fig = plt.figure(figsize=(8.5, 10.65))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_axis_off()


def box(x, y, w, h, edge, face, lw=1.6, r=1.1, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, edgecolor=edge, facecolor=face, zorder=z))


def txt(x, y, s, size=9, weight="normal", color=INK, ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, zorder=z)


def arrow(x, y0, y1, label=None, lx=None, size=8.4, bold=False):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color=INK, zorder=4,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx if lx is not None else x + 1.6, (y0 + y1) / 2, label,
                fontsize=size, color=INK, ha="left", va="center", zorder=5,
                fontweight="bold" if bold else "normal")


# ---------------------------------------------------------------- outer frame
box(1.2, 1.0, 97.6, 98.2, "#CFCFC8", "white", lw=1.3, r=0.8, z=1)

# ---------------------------------------------------------------- 1. the screens
box(8, 86.0, 84, 11.4, BLUE, BLUE_F)
txt(50, 95.6, "TWO DETERGENT COUNTER-SCREENS", 11, "bold")
box(10.5, 87.0, 39, 7.4, BLUE, "white", lw=1.3)
txt(30, 92.6, "Cruzain set", 10, "bold")
txt(30, 90.3, f"{N_CRUZAIN} compounds", 9)
txt(30, 88.3, f"{PCT_CRUZAIN} aggregators", 9)
box(50.5, 87.0, 39, 7.4, ORANGE, ORANGE_F, lw=1.3)
txt(70, 92.6, "AmpC beta-lactamase set", 10, "bold")
txt(70, 90.3, f"{N_AMPC} compounds", 9)
txt(70, 88.3, f"{PCT_AMPC} aggregators", 9)

arrow(50, 86.0, 82.6, "merge & standardize\n(MolVS 0.1.1)", lx=52.0)

# ---------------------------------------------------------------- 2. mining criteria
box(10, 70.0, 80, 12.4, GREY, GREY_F)
txt(50, 80.4, "PAIR MINING CRITERIA", 11, "bold")
for i, (n, s) in enumerate((("1", "Shared Murcko scaffold"),
                            ("2", "ECFP4 Tanimoto ≥ 0.85"),
                            ("3", "Opposite aggregation labels"))):
    yy = 77.3 - i * 2.9
    ax.add_patch(Circle((17.5, yy), 0.95, facecolor=INK, edgecolor="none", zorder=5))
    txt(17.5, yy, n, 7.6, "bold", "white", z=6)
    txt(20.0, yy, s, 9.4, ha="left")

arrow(50, 70.0, 64.4, f"{N_RAW} raw pairs", lx=52.0, bold=True)

# ---------------------------------------------------------------- 3. classification
dx, dy, dw, dh = 50, 57.6, 15.5, 6.6
ax.add_patch(Polygon([[dx, dy + dh], [dx + dw, dy], [dx, dy - dh], [dx - dw, dy]],
                     closed=True, facecolor=BLUE_F, edgecolor=BLUE, linewidth=1.6, zorder=2))
txt(50, 58.9, "PAIR", 10, "bold")
txt(50, 56.5, "CLASSIFICATION", 10, "bold")

# three branches
for x_end, lab, lx, la in ((21.5, "duplicates", 27.0, "right"),
                           (50.0, "stereoisomers", 48.4, "right"),
                           (78.5, "genuine structural pairs", 75.0, "right")):
    ax.add_patch(FancyArrowPatch((50, 51.0) if x_end == 50 else
                                 (dx - dw + 1.0 if x_end < 50 else dx + dw - 1.0, dy - 2.4),
                                 (x_end, 46.6), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color=INK, zorder=4,
                                 connectionstyle="arc3,rad=0"))
    ax.text(lx, 49.0, lab, fontsize=8.6, color=INK, ha=la, va="center", zorder=5)

box(8.5, 36.0, 26, 10.4, GREY, GREY_F)
txt(21.5, 43.3, "DUPLICATE", 9.6, "bold")
txt(21.5, 41.3, "PAIRS", 9.6, "bold")
txt(21.5, 38.6, "excluded from the census", 8.6, color=SOFT)

box(37.0, 36.0, 26, 10.4, GOLD, GOLD_F)
txt(50, 43.3, "STEREOISOMER", 9.6, "bold")
txt(50, 41.3, "PAIRS", 9.6, "bold")
txt(50, 39.2, "label error estimate", 8.6, color=SOFT)
txt(50, 37.3, f"AmpC {ERR_AMPC} · cruzain {ERR_CRUZAIN}", 8.6, color=SOFT)

box(65.5, 36.0, 26, 10.4, BLUE, BLUE_F)
txt(78.5, 43.3, "GENUINE", 9.6, "bold")
txt(78.5, 41.3, "STRUCTURAL PAIRS", 9.6, "bold")
txt(78.5, 38.6, f"census of {N_CENSUS} pairs", 8.6, color=SOFT)

# ---------------------------------------------------------------- 4. formal MMPs
arrow(78.5, 36.0, 29.4)
ax.text(76.9, 32.7, "formal MMP re-expression\n(mmpdb 3.1.4)", fontsize=8.4, color=INK,
        ha="right", va="center", zorder=5)

box(10, 17.6, 80, 11.4, GREEN, GREEN_F)
txt(50, 26.8, "FORMAL MATCHED MOLECULAR PAIRS", 11, "bold")
txt(50, 24.2, f"{PCT_FORMAL} also satisfy the formal transformation definition", 9)
box(27, 18.8, 46, 4.0, GREEN, "white", lw=1.2)
txt(50, 20.8, f"single methylene separates {PCT_METHYLENE} of pairs", 9)

arrow(50, 17.6, 14.0)

# ---------------------------------------------------------------- 5. release
box(10, 3.0, 80, 10.8, BLUE, BLUE_F)
txt(50, 11.6, "CENSUS RELEASED", 11, "bold")
box(12.5, 4.0, 36, 5.6, BLUE, "white", lw=1.2)
txt(30.5, 6.8, "filter verdicts", 9)
box(51.5, 4.0, 36, 5.6, BLUE, "white", lw=1.2)
txt(69.5, 7.6, "candidate structures for", 9)
txt(69.5, 5.6, "biophysical measurement", 9)

# ---------------------------------------------------------------- side bracket
bx = 93.6
ax.plot([bx, bx], [4.0, 45.0], color=INK, lw=1.3, zorder=4)
ax.plot([bx - 1.4, bx], [45.0, 45.0], color=INK, lw=1.3, zorder=4)
ax.plot([bx - 1.4, bx], [4.0, 4.0], color=INK, lw=1.3, zorder=4)
ax.text(bx + 1.0, 24.5, "molecular\nrepresentation\nscoring\n(12 fingerprint\ntypes)",
        fontsize=8.4, color=INK, ha="left", va="center", zorder=5)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"FigS1_census_pipeline.{ext}", dpi=600, facecolor="white",
                bbox_inches="tight")
print(f"wrote {OUT.name}/FigS1_census_pipeline.png / .pdf / .svg")
plt.close(fig)
