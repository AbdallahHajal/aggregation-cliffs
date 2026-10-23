#!/usr/bin/env python
"""Figure S1, the census construction flowchart. Version 2, 2026-10-23.

Supersedes src/fig_census_pipeline.py, which is left in place unchanged. Two defects in that
version are fixed here, both found while checking the manuscript against src/mine_cliffs.py:

  * Criterion 1 read "Shared Murcko scaffold". No Murcko scaffold is computed anywhere in
    src/; MurckoScaffold and GetScaffoldForMol appear nowhere in the tree. Mining is an ECFP4
    Tanimoto threshold with popcount bucketing plus a split on the InChIKey skeleton block
    (mine_cliffs.py). The figure therefore drew a gate the code never ran, and the Methods
    section was corrected to match the code on 2026-10-23, leaving the figure contradicting
    its own caption.
  * The restriction that actually defines the census was absent. mine_cliffs.py:104 keeps
    only structural pairs with d_clogp < 0.5 and d_mw < 30, and featurise() drops molecules
    under 10 heavy atoms. The old figure ran PAIR CLASSIFICATION straight into "census of
    3,365 pairs", so the step that halves the set was invisible.

The old figure also showed 3,450 as "raw pairs" entering classification. It is not raw: it is
the count after classification AND after the clogP/mass restriction. The flow below puts every
count where it belongs, so the figure now reconciles arrow by arrow.

Counts are module constants, each sourced in a comment, because they come from several
scripts and no single file holds them all.

    python src/fig_census_pipeline_v2.py

Writes Figures/manuscript_molvs/FigS1_census_pipeline_v2.{png,pdf,svg} at 600 dpi, and copies
them into figures/.
"""
import pathlib
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "Figures/manuscript_molvs"
FINAL = ROOT / "figures"

# ---------------------------------------------------------------- the numbers
N_CRUZAIN = "191,023"     # Methods, cleaned; _outputs_molvs_baselines/cruzain_cleaned.csv
PCT_CRUZAIN = "13.2%"
N_AMPC = "276,087"        # data/AmpC/ampc_cleaned_molvs.csv, after MolVS and dedup
PCT_AMPC = "10.2%"

# opposite-label near-neighbour pairs, before classification
# results/cruzain_full_all_pairs.csv = 2,975 rows; results/ampc_molvs_all_pairs.csv = 3,705
N_PAIRS = "6,680"
N_PAIRS_CRUZ, N_PAIRS_AMPC = "2,975", "3,705"

# classification, kind column of the same two files
N_STEREO = "403"          # cruzain 164 + AmpC 239, the opposite-label stereoisomer pairs
N_REAL = "6,277"          # cruzain 2,811 + AmpC 3,466
N_REAL_CRUZ, N_REAL_AMPC = "2,811", "3,466"

# restriction, mine_cliffs.py:104  (out.kind == "real") & (d_clogp < 0.5) & (d_mw < 30)
N_TIGHT = "3,450"         # cruzain_full_real_tight.csv 1,659 + ampc_molvs_real_tight.csv 1,791
N_TIGHT_CRUZ, N_TIGHT_AMPC = "1,659", "1,791"
N_DROPPED = "2,827"       # 6,277 - 3,450

N_DUP = "85"              # pairs whose members appear in both screens
N_CENSUS = "3,365"        # results/census_full_molvs.csv

ERR_AMPC = "21.5%"        # 239 of 1,110 stereoisomer pairs, logs/base_rate_ampc_molvs.log
ERR_CRUZAIN = "49.6%"     # 164 of 331; logs/base_rate_cruzain_fixed.log
PCT_FORMAL = "82.6%"      # 2,780 of 3,365; src/rebuild_mmp.py
PCT_METHYLENE = "59%"     # |dMW - 14.03| < 0.1 over the census

# ---------------------------------------------------------------- palette
BLUE, BLUE_F = "#2272B0", "#E8F1F8"
ORANGE, ORANGE_F = "#D97A2B", "#FBEBDC"
GREY, GREY_F = "#8E8E88", "#F1F0EC"
GOLD, GOLD_F = "#DDA02C", "#FCF4E0"
GREEN, GREEN_F = "#2E9E6B", "#E7F4EE"
INK = "#1A1A1A"
SOFT = "#555555"

fig = plt.figure(figsize=(8.5, 11.2))
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


def arrow(x, y0, y1, label=None, lx=None, size=8.4, bold=False, la="left"):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color=INK, zorder=4,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx if lx is not None else x + 1.6, (y0 + y1) / 2, label,
                fontsize=size, color=INK, ha=la, va="center", zorder=5,
                fontweight="bold" if bold else "normal")


# ---------------------------------------------------------------- outer frame
box(1.2, 1.0, 97.6, 98.2, "#CFCFC8", "white", lw=1.3, r=0.8, z=1)

# ---------------------------------------------------------------- 1. the screens
box(8, 87.4, 84, 10.4, BLUE, BLUE_F)
txt(50, 96.1, "TWO DETERGENT COUNTER-SCREENS", 11, "bold")
box(10.5, 88.3, 39, 6.6, BLUE, "white", lw=1.3)
txt(30, 93.3, "Cruzain set", 10, "bold")
txt(30, 91.2, f"{N_CRUZAIN} compounds", 9)
txt(30, 89.4, f"{PCT_CRUZAIN} aggregators", 9)
box(50.5, 88.3, 39, 6.6, ORANGE, ORANGE_F, lw=1.3)
txt(70, 93.3, "AmpC beta-lactamase set", 10, "bold")
txt(70, 91.2, f"{N_AMPC} compounds", 9)
txt(70, 89.4, f"{PCT_AMPC} aggregators", 9)

arrow(50, 87.4, 83.0,
      "merge & standardize (MolVS 0.1.1)\nstructures under 10 heavy atoms excluded", lx=52.0)

# ---------------------------------------------------------------- 2. mining criteria
box(10, 70.6, 80, 12.4, GREY, GREY_F)
txt(50, 81.0, "PAIR MINING CRITERIA", 11, "bold")
for i, (n, s) in enumerate((("1", "ECFP4 Tanimoto ≥ 0.85"),
                            ("2", "Shared InChIKey skeleton block"),
                            ("3", "Opposite aggregation labels"))):
    yy = 77.9 - i * 2.9
    ax.add_patch(Circle((17.5, yy), 0.95, facecolor=INK, edgecolor="none", zorder=5))
    txt(17.5, yy, n, 7.6, "bold", "white", z=6)
    txt(20.0, yy, s, 9.4, ha="left")

arrow(50, 70.6, 65.2, f"{N_PAIRS} near-neighbour pairs", lx=52.0, bold=True)
ax.text(52.0, 66.4, f"cruzain {N_PAIRS_CRUZ} · AmpC {N_PAIRS_AMPC}", fontsize=8.0,
        color=SOFT, ha="left", va="center", zorder=5)

# ---------------------------------------------------------------- 3. classification
dx, dy, dw, dh = 50, 58.6, 15.5, 6.4
ax.add_patch(Polygon([[dx, dy + dh], [dx + dw, dy], [dx, dy - dh], [dx - dw, dy]],
                     closed=True, facecolor=BLUE_F, edgecolor=BLUE, linewidth=1.6, zorder=2))
txt(50, 59.9, "PAIR", 10, "bold")
txt(50, 57.5, "CLASSIFICATION", 10, "bold")

for x_end, lab, lx, la in ((27.0, "stereoisomers", 29.2, "right"),
                           (73.0, "genuine structural pairs", 70.8, "left")):
    ax.add_patch(FancyArrowPatch((dx - dw + 1.0 if x_end < 50 else dx + dw - 1.0, dy - 2.2),
                                 (x_end, 48.4), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color=INK, zorder=4))
    ax.text(lx, 53.0, lab, fontsize=8.6, color=INK, ha=la, va="center", zorder=5)

box(9.5, 37.6, 35, 10.8, GOLD, GOLD_F)
txt(27, 45.4, "STEREOISOMER PAIRS", 9.6, "bold")
txt(27, 43.1, f"{N_STEREO} pairs with opposite labels", 8.8, color=SOFT)
txt(27, 41.1, "label error estimate", 8.6, color=SOFT)
txt(27, 39.2, f"AmpC {ERR_AMPC} · cruzain {ERR_CRUZAIN}", 8.6, color=SOFT)

box(55.5, 37.6, 35, 10.8, BLUE, BLUE_F)
txt(73, 45.4, "GENUINE STRUCTURAL PAIRS", 9.6, "bold")
txt(73, 43.1, f"{N_REAL} pairs", 8.8, color=SOFT)
txt(73, 40.6, f"cruzain {N_REAL_CRUZ} · AmpC {N_REAL_AMPC}", 8.6, color=SOFT)

# ---------------------------------------------------------------- 4. the restriction
arrow(73, 37.6, 30.6)
ax.text(71.4, 34.1, "ΔclogP < 0.5 and ΔMW < 30 Da\n"
                    f"{N_DROPPED} pairs set aside", fontsize=8.4, color=INK,
        ha="right", va="center", zorder=5)

box(10, 20.8, 80, 9.8, GREEN, GREEN_F)
txt(50, 28.3, "LIPOPHILICITY- AND MASS-MATCHED PAIRS", 10.6, "bold")
txt(50, 25.7, f"{N_TIGHT} pairs · cruzain {N_TIGHT_CRUZ} · AmpC {N_TIGHT_AMPC}", 9)
txt(50, 23.0, "the census is this restricted set, not the full structural population",
    8.4, color=SOFT)

arrow(73, 20.8, 17.2)
ax.text(71.4, 19.0, f"{N_DUP} pairs found in both screens removed", fontsize=8.4,
        color=INK, ha="right", va="center", zorder=5)

# ---------------------------------------------------------------- 5. the census
box(10, 3.2, 80, 14.0, BLUE, BLUE_F)
txt(50, 15.1, f"THE CENSUS · {N_CENSUS} PAIRS", 11, "bold")
box(13.5, 8.6, 73, 4.6, BLUE, "white", lw=1.2)
txt(50, 11.6, f"a single methylene separates {PCT_METHYLENE} of pairs", 8.8)
txt(50, 9.7, f"{PCT_FORMAL} also satisfy the formal transformation definition "
             "(mmpdb 3.1.4)", 8.8)
txt(50, 6.2, "released with every filter verdict and the structures nominated "
             "for measurement", 8.6, color=SOFT)

# ---------------------------------------------------------------- side bracket
bx = 93.6
ax.plot([bx, bx], [4.6, 15.8], color=INK, lw=1.3, zorder=4)
ax.plot([bx - 1.4, bx], [15.8, 15.8], color=INK, lw=1.3, zorder=4)
ax.plot([bx - 1.4, bx], [4.6, 4.6], color=INK, lw=1.3, zorder=4)
ax.text(bx + 1.0, 10.2, "molecular\nrepresentation\nscoring\n(12 fingerprint\ntypes)",
        fontsize=8.4, color=INK, ha="left", va="center", zorder=5)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"FigS1_census_pipeline_v2.{ext}", dpi=600, facecolor="white",
                bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/FigS1_census_pipeline_v2.png / .pdf / .svg")

if FINAL.is_dir():
    for ext in ("png", "pdf", "svg"):
        dst = FINAL / f"figure_S1_census_pipeline_2026-10-23.{ext}"
        shutil.copy2(OUT / f"FigS1_census_pipeline_v2.{ext}", dst)
        print(f"copied -> {dst}")
else:
    print(f"NOTE: {FINAL} not found, nothing copied")
