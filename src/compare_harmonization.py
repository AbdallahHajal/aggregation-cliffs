#!/usr/bin/env python
"""Compare every AmpC-dependent result before and after the sets were prepared alike.

The AmpC set was re-prepared with MolVS fragment_parent, the operation the cruzain set had
already received (src/prepare_sets.py). Every figure the manuscript quotes that draws on
AmpC is recomputed here from the new files, so no number is carried over from the earlier
run. Cruzain is untouched and its figures are reproduced unchanged as a control.

Covers, by manuscript location:
  Methods   total compounds, AmpC record count, label split
  Results   AmpC pairs, pooled census before and after deduplication, single-methylene
            share, cliff rate and CI, stereoisomer label-error rate and CI, the excess
            and its CI, cross-screen overlap and its reversal rate
  Abstract / Conclusions / Figure 3b all quote the cliff rate and the excess.

Reads only. Writes nothing but its report to stdout.

Usage: python src/compare_harmonization.py
"""
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OLD_CLEAN = ROOT / "data/AmpC/ampc_cleaned.csv"
NEW_CLEAN = ROOT / "data/AmpC/ampc_cleaned_molvs.csv"
OLD_PAIRS = ROOT / "results/ampc_full_all_pairs.csv"
NEW_PAIRS = ROOT / "results/ampc_molvs_all_pairs.csv"
OLD_TIGHT = ROOT / "results/ampc_full_real_tight.csv"
NEW_TIGHT = ROOT / "results/ampc_molvs_real_tight.csv"
CRZ_TIGHT = ROOT / "results/cruzain_full_real_tight.csv"
OLD_BASE = ROOT / "logs/base_rate_ampc_fixed.log"
NEW_BASE = ROOT / "logs/base_rate_ampc_molvs.log"
CRZ_BASE = ROOT / "logs/base_rate_cruzain_fixed.log"

CRZ_CLEAN = (__import__("os").environ.get("CLIFFS_SOURCE_DATA", "data/source")
             + "/Cruzain/_outputs_molvs_baselines/cruzain_cleaned.csv")

BAR = "=" * 86
DOT = r"\."


def head(t):
    print(f"\n{BAR}\n{t}\n{BAR}")


def row(label, old, new, fmt="{:,}", paper=None):
    o = fmt.format(old) if old is not None else "--"
    n = fmt.format(new) if new is not None else "--"
    tag = ""
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        d = new - old
        tag = "same" if d == 0 else f"{d:+,.2f}".rstrip("0").rstrip(".")
    pap = f"  [paper: {paper}]" if paper else ""
    print(f"  {label:<44} {o:>12} -> {n:>12}  {tag:>9}{pap}")


def wilson(k, n, z=1.96):
    """Wilson score interval, the one the manuscript states for these rates."""
    if not n:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p * 100, (c - h) * 100, (c + h) * 100


def parse_base(path):
    if not path.exists():
        return None
    txt = path.read_text()
    g = {}
    m = re.search(r"CLIFF RATE = ([\d.]+)%\s+\(95% CI ([\d.]+) to ([\d.]+)\)", txt)
    if m:
        g["rate"], g["lo"], g["hi"] = (float(m.group(i)) for i in (1, 2, 3))
    m = re.search(r"stereo pairs measured here\s+([\d.]+)%", txt)
    if m:
        g["floor"] = float(m.group(1))
    m = re.search(r"excess over the noise floor: ([\d.-]+) percentage", txt)
    if m:
        g["excess"] = float(m.group(1))
    for kind in ("stereo", "real"):
        m = re.search(rf"{kind}\s+same-label\s+([\d,]+)\s+opposite\s+([\d,]+)\s+total\s+([\d,]+)",
                      txt)
        if m:
            g[f"{kind}_same"] = int(m.group(1).replace(",", ""))
            g[f"{kind}_opp"] = int(m.group(2).replace(",", ""))
            g[f"{kind}_tot"] = int(m.group(3).replace(",", ""))
    m = re.search(r"at least one genuine near-twin: ([\d,]+)/([\d,]+)", txt)
    if m:
        g["twin"] = int(m.group(1).replace(",", ""))
        g["twin_of"] = int(m.group(2).replace(",", ""))
    return g


# ------------------------------------------------------------------ 1. the sets
head("1. THE PREPARED SETS   [Methods]")
oc = pd.read_csv(OLD_CLEAN, usecols=["SMILES", "label", "can_smiles"])
nc = pd.read_csv(NEW_CLEAN, usecols=["SMILES", "label", "can_smiles"])
crz_n = sum(1 for _ in open(CRZ_CLEAN, encoding="utf-8")) - 1
row("AmpC records", len(oc), len(nc), paper="276,232")
row("  aggregators", int((oc.label == 1).sum()), int((nc.label == 1).sum()), paper="28,254")
row("  non-aggregators", int((oc.label == 0).sum()), int((nc.label == 0).sum()),
    paper="247,978")
row("  multi-fragment structures",
    int(oc.can_smiles.str.contains(DOT, regex=True).sum()),
    int(nc.can_smiles.str.contains(DOT, regex=True).sum()))
row("aggregator prevalence %", 100 * (oc.label == 1).mean(), 100 * (nc.label == 1).mean(),
    fmt="{:.3f}", paper="10.2%")
row("total across both sets", crz_n + len(oc), crz_n + len(nc), paper="467,255")
print(f"\n  cruzain is untouched at {crz_n:,} records, so the whole change is AmpC's.")
print("  NOTE the attrition chain in Results: 373,117 - 62,886 - 33,999 = 276,232 exactly.")
print(f"       it must now absorb the {len(oc) - len(nc):,} records removed by deduplication.")

# ------------------------------------------------------------------ 2. pairs
missing = [p for p in (NEW_PAIRS, NEW_TIGHT) if not p.exists()]
if missing:
    print(f"\n  mining output not present yet: {[p.name for p in missing]}")
    print("  re-run this script once mine_cliffs.py has finished.")
    sys.exit(0)

head("2. MINED PAIRS   [Results]")
op, npr = pd.read_csv(OLD_PAIRS), pd.read_csv(NEW_PAIRS)
ot, nt = pd.read_csv(OLD_TIGHT), pd.read_csv(NEW_TIGHT)
ct = pd.read_csv(CRZ_TIGHT)
row("AmpC pairs at Tc >= 0.85", len(op), len(npr), paper="3,797")
for kind in ("duplicate", "stereo", "real"):
    row(f"  kind = {kind}", int((op.kind == kind).sum()), int((npr.kind == kind).sum()))
row("AmpC tight cliffs", len(ot), len(nt), paper="1,798")
row("  unique aggregators", ot.agg_ik.nunique(), nt.agg_ik.nunique())
row("  unique non-aggregators", ot.non_ik.nunique(), nt.non_ik.nunique())

head("3. POOLED CENSUS   [Abstract, Results x7, Figure S1]")


def pool(ampc):
    """Pooled census: cruzain + AmpC tight pairs, deduplicated across the two screens."""
    keys_c = {frozenset((r.agg_smiles, r.non_smiles)): r.agg_smiles
              for r in ct.itertuples()}
    keys_a = {frozenset((r.agg_smiles, r.non_smiles)): r.agg_smiles
              for r in ampc.itertuples()}
    shared = set(keys_c) & set(keys_a)
    reversed_ = {k for k in shared if keys_c[k] != keys_a[k]}
    raw = len(ct) + len(ampc)
    return raw, raw - len(shared), shared, reversed_


o_raw, o_ded, o_sh, o_rev = pool(ot)
n_raw, n_ded, n_sh, n_rev = pool(nt)
row("raw pairs (cruzain + AmpC)", o_raw, n_raw, paper="3,457")
row("census after cross-screen dedup", o_ded, n_ded, paper="3,375")
row("pairs present in BOTH screens", len(o_sh), len(n_sh), paper="82")
row("  of those, label reversed", len(o_rev), len(n_rev), paper="19")
if n_sh:
    p, lo, hi = wilson(len(n_rev), len(n_sh))
    print(f"  cross-screen disagreement: {p:.2f}% (Wilson 95% CI {lo:.1f} to {hi:.1f})"
          f"   [paper: 23.17% (15.4 to 33.4)]")


def census_frame(ampc):
    """The deduplicated pooled census itself, cruzain first then AmpC's additions."""
    seen = {frozenset((r.agg_smiles, r.non_smiles)) for r in ct.itertuples()}
    extra = [r for r in ampc.itertuples()
             if frozenset((r.agg_smiles, r.non_smiles)) not in seen]
    return pd.DataFrame(list(ct.itertuples()) + extra)


def methylene(ampc, by="mass"):
    """Share of the census separated by a single methylene.

    The published 59% is a mass criterion: a difference of one CH2, 14.027 Da, which
    gives 1,990 of 3,375 = 59.0% on the original files. The heavy-atom variant gives
    59.3%, so both round to 59, and both are reported here. A bare d_heavy == 1 gives
    76% and is NOT the published definition.
    """
    d = census_frame(ampc)
    if by == "mass":
        return 100 * ((d.d_mw - 14.03).abs() < 0.1).mean()
    return 100 * ((d.d_heavy == 1) & ((d.d_mw - 14).abs() < 1)).mean()


row("single-methylene share %, by mass", methylene(ot), methylene(nt),
    fmt="{:.1f}", paper="59%")
row("  same, by heavy atom + mass", methylene(ot, "heavy"), methylene(nt, "heavy"),
    fmt="{:.1f}")

# ------------------------------------------------------------------ 4. the headline
head("4. CLIFF RATE, LABEL ERROR, AND THE EXCESS   [Results, Discussion, Conclusions, Fig 3b]")
ob, nb, cb = parse_base(OLD_BASE), parse_base(NEW_BASE), parse_base(CRZ_BASE)
if not nb or "rate" not in nb:
    print("  base-rate run not finished yet; re-run this script when it is.")
else:
    row("genuine pairs queried", ob["real_tot"], nb["real_tot"], paper="6,685")
    row("  opposite label", ob["real_opp"], nb["real_opp"], paper="3,560")
    row("CLIFF RATE %", ob["rate"], nb["rate"], fmt="{:.2f}", paper="53.25%")
    print(f"  {'  95% CI':<44} {ob['lo']:>5.2f}-{ob['hi']:<6.2f} -> "
          f"{nb['lo']:>5.2f}-{nb['hi']:<6.2f}            [paper: 52.06 to 54.45]")
    row("stereo pairs", ob["stereo_tot"], nb["stereo_tot"], paper="1,108")
    row("  flipped", ob["stereo_opp"], nb["stereo_opp"], paper="237")
    row("LABEL ERROR (noise floor) %", ob["floor"], nb["floor"], fmt="{:.2f}", paper="21.4%")
    p, lo, hi = wilson(nb["stereo_opp"], nb["stereo_tot"])
    print(f"  {'  95% CI (Wilson)':<44} {'':>12}    {lo:>5.1f}-{hi:<6.1f}"
          f"            [paper: 19.0 to 23.8]")
    row("EXCESS over error (points)", ob["excess"], nb["excess"], fmt="{:.2f}", paper="31.9")
    row("aggregators with a genuine near-twin", ob["twin"], nb["twin"])

    # excess CI as the difference of two independent proportions, as the Methods states
    def diff_ci(k1, n1, k2, n2, z=1.96):
        p1, p2 = k1 / n1, k2 / n2
        se = (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5
        d = p1 - p2
        return 100 * d, 100 * (d - z * se), 100 * (d + z * se)
    d, lo, hi = diff_ci(nb["real_opp"], nb["real_tot"], nb["stereo_opp"], nb["stereo_tot"])
    print(f"\n  excess recomputed as a difference of two proportions: "
          f"{d:.2f} points (95% CI {lo:.1f} to {hi:.1f})")
    print(f"  [paper: 31.9 points (95% CI 29.2 to 34.6)]")
    print(f"  direction preserved and interval excludes zero: "
          f"{'YES' if lo > 0 else 'NO'}")
    if cb and "excess" in cb:
        print(f"\n  cruzain control, unchanged: cliff rate {cb['rate']:.2f}%, "
              f"error {cb['floor']:.2f}%, excess {cb['excess']:.2f} points")

# ------------------------------------------------------------------ 5. physchem
head("5. PAIRED PHYSCHEM PROFILE   [Results]")
try:
    from cliff_physchem import profile
    print("\n  --- AmpC tight pairs, original ---")
    profile(str(OLD_TIGHT), "AmpC original")
    print("\n  --- AmpC tight pairs, MolVS-prepared ---")
    profile(str(NEW_TIGHT), "AmpC MolVS")
except Exception as e:
    print(f"  skipped: {e}")

print(f"\n{BAR}\nOriginal files untouched. results/cliff_physchem_*.csv not rewritten.\n{BAR}")
