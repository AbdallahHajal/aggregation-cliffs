#!/usr/bin/env python
"""Is the BAD Molecule Filter's within-pair directional signal recall or prediction?

The filter ranks the true aggregator above the true non-aggregator in 66.2% of census
pairs. That is the same shape of result that turned out to be memorisation for SCAM
Detective, so it cannot be reported before the training set is checked.

The distributed release ships its training matrices as features only, with no SMILES or
InChIKey, so structures cannot be looked up directly. They can be reconstructed: the binary
matrix is exactly the 602 Morgan bits named in the app's own `descriptor_columns_MM`, taken
from a radius 2, 2048 bit fingerprint. Recomputing those 602 bits for every census compound
and looking for an exact row match gives training-set membership up to fingerprint identity.

Fingerprint identity is not structure identity, and this census contains pairs whose members
share an ECFP4 vector, so a match is an upper bound on overlap rather than proof of it. A
count of zero, on the other hand, is conclusive.
"""
import pathlib
import pickle
import re
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/bad_filter"


def binary_columns():
    """The Col_N names the app keeps, in the order it keeps them."""
    src = (VENDOR / "app.py").read_text()
    m = re.search(r"descriptor_columns_MM\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        raise SystemExit("could not find descriptor_columns_MM in app.py")
    names = re.findall(r"'([^']+)'", m.group(1))
    return [n for n in names if n.startswith("Col_")]


def main():
    cols = binary_columns()
    idx = [int(c.split("_")[1]) for c in cols]
    print(f"binary feature columns in the app: {len(cols)}")

    # pickle.load is safe here: this file is part of the BAD Molecule Filter's own
    # distributed release, vendored in this repo and already executed by its container.
    # It is not third-party input.
    train = pickle.load(open(VENDOR / "x_train1_MM_Binary_training_set.pkl", "rb"))
    train = np.asarray(train)
    print(f"training matrix: {train.shape}")
    if train.shape[1] != len(cols):
        raise SystemExit(f"column count mismatch: {train.shape[1]} vs {len(cols)}")

    seen = {row.astype(np.uint8).tobytes() for row in train}
    print(f"distinct training rows: {len(seen):,} of {len(train):,}")

    cen = pd.read_csv(ROOT / "results/chemfh_input_keys.csv")
    hits, bad_parse = [], 0
    for smi, ik in zip(cen.smiles, cen.inchikey):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad_parse += 1
            continue
        fp = np.zeros(2048, dtype=np.uint8)
        for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048).GetOnBits():
            fp[b] = 1
        if fp[idx].tobytes() in seen:
            hits.append(ik)

    n = len(cen) - bad_parse
    print(f"\ncensus compounds checked: {n:,}   unparsed: {bad_parse}")
    print(f"matching a training row exactly: {len(hits):,}  ({100*len(hits)/n:.2f}%)")

    # what the overlap means for the paired result
    pairs = pd.read_csv(ROOT / "results/census_full_3375.csv")
    h = set(hits)
    both = int(((pairs.agg_ik.isin(h)) & (pairs.non_ik.isin(h))).sum())
    one = int(((pairs.agg_ik.isin(h)) ^ (pairs.non_ik.isin(h))).sum())
    print(f"\ncensus pairs with BOTH members in training: {both:,} of {len(pairs):,} "
          f"({100*both/len(pairs):.1f}%)")
    print(f"census pairs with ONE member in training:   {one:,} "
          f"({100*one/len(pairs):.1f}%)")
    print(f"census pairs with NEITHER member:           {len(pairs)-both-one:,} "
          f"({100*(len(pairs)-both-one)/len(pairs):.1f}%)")

    pd.DataFrame({"inchikey": hits}).to_csv(
        ROOT / "results/bad_training_overlap_inchikeys.csv", index=False)

    # the directional test restricted to pairs the model cannot have memorised
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    b = pd.read_csv(ROOT / "results/pairs_sample_500_bad.csv")
    p = dict(zip(b.inchikey, b.bad_prob))
    pa = np.array([p.get(k, np.nan) for k in s.agg_ik], float)
    pn = np.array([p.get(k, np.nan) for k in s.non_ik], float)
    clean = ~s.agg_ik.isin(h).to_numpy() & ~s.non_ik.isin(h).to_numpy()
    for tag, mask in (("all sampled pairs", np.ones(len(s), bool)),
                      ("neither member in training", clean)):
        ok = mask & ~np.isnan(pa) & ~np.isnan(pn) & (pa != pn)
        if ok.sum() == 0:
            print(f"\n{tag}: no pairs")
            continue
        hi = int((pa[ok] > pn[ok]).sum())
        from scipy.stats import binomtest
        print(f"\n{tag}: aggregator ranked higher in {hi} of {ok.sum()} "
              f"({100*hi/ok.sum():.1f}%), p = {binomtest(hi, int(ok.sum()), 0.5).pvalue:.2e}")


if __name__ == "__main__":
    main()
