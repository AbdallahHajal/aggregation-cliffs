#!/usr/bin/env python
"""Score the sampled census molecules with the BAD Molecule Filter.

Uses bad_client.score, which bisects on failure so one degenerate molecule cannot
invalidate the batch. Thresholds are the ones implemented in the distributed desktop
application: > 0.6 aggregator, < 0.3 non-aggregator, between the two ambiguous.
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bad_client  # noqa: E402

INP = ROOT / "results/census_full_molecules.csv"
OUT = ROOT / "results/census_full_bad.csv"
CHUNK = 100

df = pd.read_csv(INP)
smiles = df["smiles"].astype(str).tolist()
print(f"scoring {len(smiles):,} molecules in chunks of {CHUNK}", flush=True)

probs, failed = [], 0
for i in range(0, len(smiles), CHUNK):
    part = smiles[i:i + CHUNK]
    p, f = bad_client.score(part, tag=f"cen{i}")
    probs.extend(p)
    failed += f
    print(f"  {min(i+CHUNK, len(smiles)):4d}/{len(smiles)}  failures so far {failed}", flush=True)

df["bad_prob"] = probs
df["bad_call"] = pd.cut(df["bad_prob"], bins=[-0.01, 0.3, 0.6, 1.01],
                        labels=["Non-Aggregator", "Ambiguous", "Aggregator"])
df.to_csv(OUT, index=False)

print(f"\nwrote {OUT.name}")
print(f"molecules that failed scoring: {failed}")
print(df["bad_call"].value_counts(dropna=False).to_string())
