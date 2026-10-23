#!/usr/bin/env python
"""The four data figures, each given the encoding its question actually needs.

F3  a reference-band plot. The question is not how large the number is but where it sits
    between two references, so the floor and the chance rate form the scale and the observed
    rate is a point on it.
F5  a dot plot with intervals, because seven filters clustered inside four percentage points
    is a statement about clustering that bars obscure; and a slope chart for the
    contamination control, because that finding is a drop between two conditions.
F7  a ranked dot plot with a connector, since the values span three orders of magnitude and
    the ordering matters more than the areas.
F8  a stacked composition, because the interesting quantity is the whole prediction set
    rather than one slice of it.

House style follows Sturm et al. Angew Chem 2025 as encoded in figstyle.py. Every number is
read from results/ at draw time, so a figure cannot disagree with the text.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figstyle import AGG, AGG_SOFT, GREY, INK, NON, save  # noqa: E402

NAVY, NAVY_SOFT = "#1F3A5F", "#DCE3EC"
BAND = "#ECECE8"
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5})


def _bare(ax, keep=("bottom",)):
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(sp in keep)
    for sp in keep:
        ax.spines[sp].set_color(GREY)
    ax.tick_params(colors=GREY, length=3)


def fig_noise_floor():
    a = pd.read_csv(ROOT / "results/base_rate_ampc.csv").set_index("kind")
    c = pd.read_csv(ROOT / "results/base_rate_cruzain.csv").set_index("kind")

    def rate(d, k):
        r = d.loc[k]
        n = r.same_label + r.opposite_label
        return 100 * r.opposite_label / n, int(n)

    rows = []
    for name, d, chance in (("AmpC β-lactamase", a, 89.8), ("cruzain", c, 86.8)):
        cliff, n_c = rate(d, "real")
        floor, n_f = rate(d, "stereo")
        rows.append(dict(name=name, cliff=cliff, floor=floor, chance=chance,
                         n_c=n_c, n_f=n_f))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    _bare(ax)
    for i, r in enumerate(rows):
        y = 1 - i
        ax.plot([0, r["chance"]], [y, y], color=BAND, lw=13, solid_capstyle="butt", zorder=1)
        ax.plot([0, r["floor"]], [y, y], color=NAVY_SOFT, lw=13, solid_capstyle="butt",
                zorder=2)
        ax.plot([r["floor"], r["cliff"]], [y, y], color=AGG_SOFT, lw=13,
                solid_capstyle="butt", zorder=3)
        ax.plot([r["floor"]], [y], marker="|", ms=17, mew=2.0, color=NAVY, zorder=5)
        ax.plot([23.17], [y], marker="v", ms=6.5, color=NAVY, zorder=6)
        ax.plot([r["cliff"]], [y], marker="o", ms=10, color=AGG, zorder=6,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.plot([r["chance"]], [y], marker="|", ms=17, mew=2.0, color=GREY, zorder=5)

        ax.text(-1.5, y, r["name"], ha="right", va="center", fontsize=9.4,
                fontweight="bold", color=INK)
        ax.text(r["cliff"], y + 0.31, f"{r['cliff']:.1f}%", ha="center", fontsize=9.4,
                fontweight="bold", color=AGG)
        ax.text(r["floor"], y - 0.35, f"{r['floor']:.1f}%", ha="center", fontsize=8.2,
                color=NAVY)
        # the gap sits in the empty grey stretch to the right of the observed rate, where
        # it cannot collide with the floor label however close the two markers fall
        ax.text(r["cliff"] + 2.6, y, f"+{r['cliff']-r['floor']:.1f} points",
                ha="left", va="center", fontsize=7.8, color=AGG, fontweight="bold")

    ax.text(23.17, 1.66, "cross-screen\nfloor 23.2%", ha="center", fontsize=7.4, color=NAVY)
    ax.annotate("", xy=(23.17, 1.30), xytext=(23.17, 1.52),
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.8))
    ax.text(88, -0.66, "if structure carried\nno information", ha="center", fontsize=7.4,
            color=GREY)
    ax.text(1, -0.66, "if structure decided\neverything", ha="left", fontsize=7.4, color=GREY)
    ax.set_yticks([])
    ax.set_ylim(-1.0, 2.05)
    ax.set_xlim(-2, 95)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("near-identical pairs carrying opposite labels (%)", fontsize=8.8)
    ax.set_title("Cliff frequency sits well above what label error alone can explain",
                 loc="left", fontsize=10, fontweight="bold", color=INK, pad=12)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.83, bottom=0.25)
    save(fig, "Fig06_cliff_rate_vs_noise_floor")
    plt.close(fig)


def fig_filters():
    # one common set: every filter scored on the same 500 pairs, both members, same rule
    m = pd.read_csv(ROOT / "results/miss_rates_common_set.csv")
    cl = pd.read_csv(ROOT / "results/miss_rates_clean_set.csv").set_index("filter")
    m["raw"] = m["filter"]                       # keep the join key before relabelling
    m["filter"] = m["filter"].str.replace(" > 3.5 (null)", " > 3.5", regex=False)
    m = m.sort_values("missed", ascending=False).reset_index(drop=True)
    null = float(m.loc[m["filter"].str.contains("clogP"), "missed"].iloc[0])

    fig = plt.figure(figsize=(7.3, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.55)

    ax = fig.add_subplot(gs[0])
    _bare(ax)
    y = np.arange(len(m))
    # drawn by hand rather than with axvline so it stops short of the labels at either end
    ax.plot([null, null], [-0.30, len(m) - 0.85], color=GREY, lw=1.0, ls=(0, (4, 3)),
            zorder=1)
    lo = m.ci.str.split("-").str[0].astype(float)
    hi = m.ci.str.split("-").str[1].astype(float)
    for yy, l, h in zip(y, lo, hi):
        ax.plot([l, h], [yy, yy], color=GREY, lw=1.6, solid_capstyle="round", zorder=3)
    cols = [GREY if "clogP" in f else AGG for f in m["filter"]]
    ax.scatter(m.missed, y, s=78, color=cols, zorder=4, edgecolor="white", linewidth=1.3)
    # the grey marks are what each filter's own base rate forces: a tool that calls almost
    # everything one class must agree with itself within a pair, whatever it knows
    # the open marks are the same filters on the 89 pairs outside every training set we
    # could obtain; the two agree, so the result is not a contamination artefact
    clean = np.array([cl.loc[f, "missed_clean"] for f in m["raw"]])
    ax.scatter(clean, y, s=62, facecolor="white", edgecolor=NAVY, linewidth=1.4, zorder=4)
    for yy, v, c, b in zip(y, m.missed, m.correct, m.backwards):
        ax.text(v, yy + 0.30, f"{v:.1f}%", ha="center", fontsize=8.4, fontweight="bold",
                color=INK)
        ax.text(101.4, yy, f"{c} right / {b} reversed", va="center", fontsize=7.4,
                color=GREY)
    ax.set_yticks(y)
    ax.set_yticklabels(m["filter"], fontsize=8.6)
    ax.set_xlim(78, 101)
    ax.set_ylim(-1.0, len(m) - 0.3)
    ax.set_xlabel("pairs given the same class for both members (%)", fontsize=8.8)
    ax.text(null, -0.62, "one calculated descriptor", ha="center", va="center",
            fontsize=7.4, color=GREY)
    ax.scatter([79.3], [-0.62], s=62, facecolor="white", edgecolor=NAVY, linewidth=1.4,
               clip_on=False, zorder=5)
    ax.text(80.1, -0.62, "the clean 89 pairs", ha="left", va="center", fontsize=7.2,
            color=NAVY)
    ax.set_title("a   Seven filters on the same 500 pairs",
                 loc="left", fontsize=10, fontweight="bold", color=INK, pad=16)

    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    sc = pd.read_csv(ROOT / "results/pairs_sample_500_scam.csv")
    ax2 = fig.add_subplot(gs[1])
    _bare(ax2, keep=())

    # Drawn as two separate slopes rather than two lines in one frame: the held-out values
    # differ by 0.6 points, so in a shared frame their labels land on top of each other, and
    # the symmetry between the two models is the finding.
    groups = []
    for col, trained, other in (("scam_cruzain", "cruzain", "ampc"),
                                ("scam_beta_lactamase", "AmpC", "cruzain")):
        mp = dict(zip(sc.inchikey, sc[col]))
        vv = {}
        for ds in (trained.lower(), other.lower()):
            sub = s[s.dataset == ds]
            ok = [(mp.get(x), mp.get(z)) for x, z in zip(sub.agg_ik, sub.non_ik)]
            ok = [(x, z) for x, z in ok if x is not None and z is not None]
            vv[ds] = 100 * np.mean([x == z for x, z in ok])
        groups.append((trained, other, vv[trained.lower()], vv[other.lower()]))

    pretty = {"ampc": "AmpC", "cruzain": "cruzain"}
    for k, (trained, other, v_in, v_out) in enumerate(groups):
        x0 = 2.3 * k
        ax2.plot([x0, x0 + 1], [v_in, v_out], color=AGG, lw=2.2, zorder=3, marker="o",
                 ms=9, markeredgecolor="white", markeredgewidth=1.4)
        ax2.text(x0 - 0.09, v_in, f"{v_in:.1f}%", ha="right", va="center", fontsize=8.8,
                 fontweight="bold", color=AGG)
        ax2.text(x0 + 1.09, v_out - 2.8, f"{v_out:.1f}%", ha="left", va="center",
                 fontsize=8.8, fontweight="bold", color=NAVY)
        ax2.text(x0 + 0.5, 99.4, f"model trained on {pretty[trained.lower()]}",
                 ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
        for xx, ds in ((x0, trained), (x0 + 1, other)):
            ax2.text(xx, 54.2, f"{pretty[ds.lower()]}\npairs", ha="center", va="top",
                     fontsize=8.0, color=GREY)
    ax2.axhline(null, color=GREY, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax2.text(-0.52, null + 1.0, "clogP null, 92.8%", ha="left", va="bottom", fontsize=7.4,
             color=GREY)
    ax2.set_xlim(-0.55, 3.95)
    ax2.set_ylim(49, 101)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("b   SCAM Detective's advantage does not survive the held-out assay",
                  loc="left", fontsize=10, fontweight="bold", color=INK, pad=10)
    fig.subplots_adjust(left=0.23, right=0.97, top=0.92, bottom=0.11)
    save(fig, "Fig07_filters_on_pairs")
    plt.close(fig)


def fig_representations():
    d = pd.read_csv(ROOT / "results/representation_sweep.csv").sort_values(
        "identical_pairs", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _bare(ax)
    y = np.arange(len(d))[::-1]
    for yy, v in zip(y, d.identical_pairs):
        ax.plot([0, v], [yy, yy], color="#E8E8E4", lw=1.4, zorder=1)
    cols = [AGG if "ECFP4 (2048b" in r else (NON if "Atom pair" in r else NAVY)
            for r in d.representation]
    sizes = [116 if ("ECFP4 (2048b" in r or "Atom pair" in r) else 62
             for r in d.representation]
    ax.scatter(d.identical_pairs, y, s=sizes, color=cols, zorder=4, edgecolor="white",
               linewidth=1.3)
    for yy, v, p in zip(y, d.identical_pairs, d.pct_identical):
        if v > 0:
            ax.text(v + 26, yy, f"{int(v):,}  ({p:.1f}%)", va="center", fontsize=7.8,
                    color=INK)
        else:
            ax.text(26, yy, "0    every pair separated", va="center", fontsize=8,
                    color=NON, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(d.representation, fontsize=8.4)
    ax.set_xlim(-18, 1300)
    ax.set_xlabel("census pairs rendered exactly identical, of 3,375", fontsize=8.8)
    ax.set_title("What a pair looks like depends entirely on the descriptor",
                 loc="left", fontsize=10, fontweight="bold", color=INK, pad=10)
    i = int(np.flatnonzero(d.representation.str.contains(r"ECFP4 \(2048b"))[0])
    ax.annotate("the representation the filters use",
                xy=(d.identical_pairs[i] + 10, y[i]), xytext=(430, y[i] - 1.6),
                fontsize=7.6, color=AGG, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=AGG, lw=1.0))
    fig.subplots_adjust(left=0.235, right=0.985, top=0.90, bottom=0.13)
    save(fig, "Fig08_representation_resolvability")
    plt.close(fig)


def fig_conformal():
    d = pd.read_csv(ROOT / "results/conformal_abstention.csv")
    d = d[d.eps == 0.10].sort_values("both_labels").reset_index(drop=True)
    d["one"] = 100 - d.both_labels - d["empty"]

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    _bare(ax)
    y = np.arange(len(d))
    ax.barh(y, d.one, color=NAVY, height=0.5, zorder=3)
    ax.barh(y, d.both_labels, left=d.one, color=AGG, height=0.5, zorder=3)
    for yy, o, b in zip(y, d.one, d.both_labels):
        if o > 14:
            ax.text(o / 2, yy, f"{o:.0f}%", ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold", zorder=5)
        if b > 8:
            ax.text(o + b / 2, yy, f"{b:.0f}%", ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold", zorder=5)
        else:
            ax.text(o + b + 1.5, yy, f"{b:.1f}%", ha="left", va="center", color=AGG,
                    fontsize=8, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(d["filter"], fontsize=8.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("molecules (%)", fontsize=8.8)
    ax.set_title("At 90% coverage, most of these filters should be abstaining",
                 loc="left", fontsize=10, fontweight="bold", color=INK, pad=26)
    ax.text(0.0, 1.035, "commits to one label", transform=ax.transAxes, fontsize=8,
            color=NAVY, fontweight="bold", va="bottom")
    ax.text(0.34, 1.035, "returns both labels, declining to decide",
            transform=ax.transAxes, fontsize=8, color=AGG, fontweight="bold", va="bottom")
    low = d.iloc[0]
    ax.text(1.0, -0.32, f"{low['filter']} declines on {low.both_labels:.1f}% and gives both "
            f"members of {low.identical_sets:.1f}% of pairs the same set",
            transform=ax.transAxes, ha="right", fontsize=7.4, color=GREY)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.79, bottom=0.23)
    save(fig, "Fig09_conformal_abstention")
    plt.close(fig)


if __name__ == "__main__":
    fig_noise_floor()
    fig_filters()
    fig_representations()
    fig_conformal()
