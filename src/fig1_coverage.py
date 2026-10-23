"""Figure 1. Measurement coverage of the aggregator and non-aggregator classes
in the two detergent counter-screens used to train published SCAM filters."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

BLUE, VERM = "#0072B2", "#D55E00"          # Okabe-Ito, colour-blind safe

rows = [  # label, n_total, n_measured, colour
    ("Aggregator",     25171,  25171, BLUE),
    ("Non-aggregator", 166328,     0, VERM),
    ("Aggregator",     28254,  28254, BLUE),
    ("Non-aggregator", 281977,     0, VERM),
]
ypos = [3.55, 2.75, 1.25, 0.45]

fig, ax = plt.subplots(figsize=(6.6, 2.5))
for (lab, tot, meas, col), y in zip(rows, ypos):
    pct = 100.0 * meas / tot
    ax.barh(y, 100, height=0.52, color="#EEEEEE", edgecolor="none", zorder=1)
    if pct > 0:
        ax.barh(y, pct, height=0.52, color=col, edgecolor="none", zorder=2)
    else:                                   # make the zero visible, not absent
        ax.plot([0.35], [y], marker="|", ms=11, mew=1.4, color=col, zorder=3)
    txt = f"{meas:,} of {tot:,}"
    ax.text(101.5, y, txt, va="center", ha="left", fontsize=8,
            color="#111111" if pct else VERM,
            fontweight="bold" if pct == 0 else "normal")
    ax.text(-1.5, y, lab, va="center", ha="right", fontsize=8)

ax.text(-1.5, 3.15, "Cruzain", ha="right", va="center", fontsize=8.5,
        fontweight="bold", color="#333333")
ax.text(-1.5, 0.85, "AmpC", ha="right", va="center", fontsize=8.5,
        fontweight="bold", color="#333333")
ax.text(101.5, 3.15, "AID 1476 / 1478", ha="left", va="center",
        fontsize=7, color="#666666", style="italic")
ax.text(101.5, 0.85, "AID 485341 / 485294", ha="left", va="center",
        fontsize=7, color="#666666", style="italic")

ax.axhline(2.0, color="#CCCCCC", lw=0.6)
ax.set_xlim(0, 100); ax.set_ylim(-0.15, 4.15)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.xaxis.set_major_formatter(PercentFormatter())
ax.set_xlabel("Compounds with a fitted dose–response curve")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.subplots_adjust(left=0.135, right=0.845, top=0.97, bottom=0.20)
for ext, dpi in (("png", 600), ("pdf", 600), ("svg", 600)):
    fig.savefig(f"Figures/Figure 1.{ext}", dpi=dpi)
print("wrote Figures/Figure 1.png / .pdf / .svg")
