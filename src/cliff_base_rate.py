#!/usr/bin/env python
"""The denominator the census is missing.

mine_cliffs.py crosses aggregators against non-aggregators only, so it counts
opposite-label near-identical pairs without ever counting how many near-identical
pairs there are. Without that, 3,375 cannot be compared to the 5.5% stereoisomer
noise floor and the census cannot be defended.

This asks, for every aggregator, how many of its near-identical neighbours in the
FULL set carry the opposite label:

    cliff rate = neighbours labelled non-aggregator / all neighbours at Tc >= 0.85

Three reference points make the number readable:

    0%            structure fully determines aggregation
    floor (5.5%)  stereoisomer pairs, which cannot differ by construction
    2*p*(1-p)     what the rate would be if the label were independent of structure

Same fingerprint, threshold and popcount pruning as mine_cliffs.py so the counts
are comparable to the census.

Usage: python cliff_base_rate.py <cleaned_csv> <tag> [--threshold 0.85]
"""
import argparse
import collections
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import inchi, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ap = argparse.ArgumentParser()
ap.add_argument("cleaned_csv")
ap.add_argument("tag")
ap.add_argument("--threshold", type=float, default=0.85)
ap.add_argument("--smiles-col", default="can_smiles")
a = ap.parse_args()

T = a.threshold
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


df = pd.read_csv(a.cleaned_csv).dropna(subset=[a.smiles_col, "label"])
df["label"] = df["label"].astype(int)
log(f"loaded {len(df):,} rows; labels {df['label'].value_counts().to_dict()}")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

recs = []
for smi, lab in zip(df[a.smiles_col], df["label"]):
    m = Chem.MolFromSmiles(str(smi))
    if m is None or m.GetNumHeavyAtoms() < 10:
        continue
    try:
        ik = inchi.MolToInchiKey(m)
    except Exception:
        ik = ""
    fp = gen.GetFingerprint(m)
    recs.append((smi, int(lab), fp, int(fp.GetNumOnBits()), ik, ik.split("-")[0] if ik else ""))
log(f"featurised {len(recs):,}")

# sort by popcount so the admissible window is a contiguous slice
recs.sort(key=lambda r: r[3])
pops = np.array([r[3] for r in recs])
fps = [r[2] for r in recs]
labs = np.array([r[1] for r in recs])
smis = [r[0] for r in recs]
iks = [r[4] for r in recs]
skels = [r[5] for r in recs]

p = labs.mean()
log(f"aggregator prevalence p = {p:.4f}; chance opposite-label rate 2p(1-p) = {2*p*(1-p)*100:.2f}%")

agg_idx = np.flatnonzero(labs == 1)
log(f"querying {len(agg_idx):,} aggregators against the full set")

# counters, split by how the pair relates structurally
C = collections.Counter()
per_query_neighbours = []

for n, i in enumerate(agg_idx, 1):
    a_pop = pops[i]
    lo = np.searchsorted(pops, int(np.ceil(T * a_pop)), "left")
    hi = np.searchsorted(pops, int(np.floor(a_pop / T)), "right")
    if hi <= lo:
        per_query_neighbours.append(0)
        continue
    cand = list(range(lo, hi))
    cand = [j for j in cand if j != i]
    if not cand:
        per_query_neighbours.append(0)
        continue
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in cand])
    hits = [cand[k] for k, s in enumerate(sims) if s >= T]

    n_nb = 0
    for j in hits:
        # Queries run over aggregators only, so an aggregator-aggregator pair is seen
        # from both ends while an aggregator-non-aggregator pair is seen once. Counting
        # both ends inflates the same-label denominator and deflates every rate.
        # Fixed 2026-09-29 after an adversarial review caught it: the signature was that
        # all four same-label counts were even.
        if labs[j] == 1 and j < i:
            continue
        if smis[i] == smis[j]:
            kind = "duplicate"
        elif skels[i] and skels[i] == skels[j] and iks[i] != iks[j]:
            kind = "stereo"
        else:
            kind = "real"
        opp = "opposite" if labs[j] != labs[i] else "same"
        C[(kind, opp)] += 1
        if kind == "real":
            n_nb += 1
    per_query_neighbours.append(n_nb)

    if n % 2000 == 0:
        log(f"  {n:,}/{len(agg_idx):,} queries; real pairs so far "
            f"{C[('real','same')] + C[('real','opposite')]:,}")

log("done querying")

print("\n" + "=" * 70)
print(f"NEAR-IDENTICAL NEIGHBOURS OF AGGREGATORS  (Tc >= {T})   [{a.tag}]")
print("=" * 70)
rows = []
for kind in ("duplicate", "stereo", "real"):
    same, opp = C[(kind, "same")], C[(kind, "opposite")]
    tot = same + opp
    rate = 100 * opp / tot if tot else float("nan")
    rows.append((kind, same, opp, tot, rate))
    print(f"  {kind:10s} same-label {same:8,}   opposite {opp:8,}   "
          f"total {tot:8,}   opposite-label rate {rate:6.2f}%")

real_same, real_opp = C[("real", "same")], C[("real", "opposite")]
real_tot = real_same + real_opp

print("\n" + "-" * 70)
print("THE NUMBER THE CENSUS NEEDS")
print("-" * 70)
if real_tot:
    r = real_opp / real_tot
    se = np.sqrt(r * (1 - r) / real_tot)
    print(f"  Of {real_tot:,} near-identical genuine pairs involving an aggregator,")
    print(f"  {real_opp:,} carry the opposite label.")
    print(f"  CLIFF RATE = {r*100:.2f}%  (95% CI {100*(r-1.96*se):.2f} to {100*(r+1.96*se):.2f})")
    print(f"\n  compare against:")
    print(f"    structure fully determines label    0.00%  (+ noise)")
    print(f"    stereo pairs measured here         "
          f"{100*C[('stereo','opposite')]/max(1, C[('stereo','same')]+C[('stereo','opposite')]):.2f}%"
          f"   <- the noise floor, same threshold, same run")
    print(f"    label independent of structure    {100*(1-p):.2f}%"
          f"   (queries are all aggregators, so the conditional baseline is 1-p, NOT 2p(1-p))")
    floor = C[("stereo", "opposite")] / max(1, C[("stereo", "same")] + C[("stereo", "opposite")])
    print(f"\n  excess over the noise floor: {100*(r-floor):.2f} percentage points")
else:
    print("  no genuine near-identical pairs found")

nb = np.array(per_query_neighbours)
print(f"\n  aggregators with at least one genuine near-twin: "
      f"{(nb > 0).sum():,}/{len(nb):,} ({100*(nb>0).mean():.1f}%)")

pd.DataFrame(rows, columns=["kind", "same_label", "opposite_label", "total", "opposite_rate_pct"]) \
    .to_csv(f"results/base_rate_{a.tag}.csv", index=False)
print(f"\nwrote results/base_rate_{a.tag}.csv")
