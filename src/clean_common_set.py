#!/usr/bin/env python
"""One evaluation set that is outside the training data of every filter at once.

Scoring each filter on its own training-disjoint subset answers the contamination
objection for each tool separately but compares them on different pairs, which is the
problem the common set was introduced to fix. This builds the intersection instead: pairs
in which neither member appears in the training data of ANY filter whose training data can
be obtained, so a single set serves all of them.

Training data obtained, and how membership is tested:
  ChemFH       ChemFH_dataset/Aggregators.csv, 51,952 structures      InChIKey
  CAGE-Fusion  data/cage_fusion/train.csv                             InChIKey
  DeepSCAMs    vendor/deepscams/sample_files/train_data.txt            InChIKey
  BAD          602 Morgan bits recovered from its binary train matrix  fingerprint row
  Advisor      a similarity of exactly 1.000 to its own database       reported by the tool

Not obtainable, and stated as a limitation rather than assumed clean:
  ChemAGG          training list is in supporting information we do not have
  SCAM Detective   trained on these two campaigns by construction, handled separately

Writes results/clean_common_set_pairs.csv and results/miss_rates_clean_set.csv.
"""
import pathlib
import pickle
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"


def keys_from_smiles(smiles):
    out = set()
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            out.add(Chem.MolToInchiKey(m))
    return out


def bad_training_keys(sample_keys, sample_smiles):
    """BAD ships features, not structures. Recompute its 602 bits and match rows."""
    src = (VENDOR / "bad_filter/app.py").read_text()
    names = re.findall(r"'([^']+)'",
                       re.search(r"descriptor_columns_MM\s*=\s*\[(.*?)\]", src, re.S).group(1))
    idx = [int(n.split("_")[1]) for n in names if n.startswith("Col_")]
    # the release's own training matrix, already executed by its own container
    train = np.asarray(pickle.load(
        open(VENDOR / "bad_filter/x_train1_MM_Binary_training_set.pkl", "rb")))
    seen = {r.astype(np.uint8).tobytes() for r in train}
    hit = set()
    for ik, smi in zip(sample_keys, sample_smiles):
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        fp = np.zeros(2048, dtype=np.uint8)
        for b in AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048).GetOnBits():
            fp[b] = 1
        if fp[idx].tobytes() in seen:
            hit.add(ik)
    return hit


def main():
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    structures = pd.concat([
        s[["agg_ik", "agg_smiles"]].rename(columns={"agg_ik": "ik", "agg_smiles": "smi"}),
        s[["non_ik", "non_smiles"]].rename(columns={"non_ik": "ik", "non_smiles": "smi"}),
    ]).drop_duplicates("ik")
    print(f"sample: {len(s)} pairs, {len(structures)} unique structures\n")

    contaminated = {}

    fh = pd.read_csv(VENDOR / "eos7ye0/model/framework/code/ChemFH/ChemFH_dataset/"
                              "Aggregators.csv")
    contaminated["ChemFH"] = keys_from_smiles(fh.SMILES)

    cg = pd.read_csv(ROOT / "data/cage_fusion/train.csv")
    col = next(c for c in cg.columns if "smi" in c.lower())
    contaminated["CAGE-Fusion"] = keys_from_smiles(cg[col])

    ds = pd.read_csv(VENDOR / "deepscams/sample_files/train_data.txt", sep="\t")
    col = next(c for c in ds.columns if "smi" in c.lower())
    contaminated["DeepSCAMs"] = keys_from_smiles(ds[col])

    contaminated["BAD Molecule Filter"] = bad_training_keys(structures.ik, structures.smi)

    adv = pd.read_csv(ROOT / "results/pairs_sample_500_advisor.csv")
    contaminated["Aggregator Advisor"] = set(adv.loc[adv.advisor_sim >= 0.999, "inchikey"])

    print(f"{'filter':<22}{'training set':>14}{'in sample':>12}{'% of sample':>13}")
    print("-" * 61)
    union = set()
    for k, v in contaminated.items():
        inside = len(set(structures.ik) & v)
        print(f"{k:<22}{len(v):>14,}{inside:>12,}{100*inside/len(structures):>12.1f}%")
        union |= v
    inside = len(set(structures.ik) & union)
    print("-" * 61)
    print(f"{'ANY of them':<22}{'':>14}{inside:>12,}{100*inside/len(structures):>12.1f}%")

    clean = ((~s.agg_ik.isin(union)) & (~s.non_ik.isin(union))).to_numpy()
    print(f"\npairs with NEITHER member in ANY known training set: {int(clean.sum())} "
          f"of {len(s)}")
    s.loc[clean].to_csv(ROOT / "results/clean_common_set_pairs.csv", index=False)
    pd.DataFrame({"inchikey": sorted(set(structures.ik) & union)}).to_csv(
        ROOT / "results/contaminated_structures.csv", index=False)

    # every filter, on those pairs, and each against the null by an exact paired test
    from common_set_miss_rates import verdicts, boot_ci
    from paired_filter_tests import mcnemar_exact
    _, v = verdicts()
    miss, rows = {}, []
    for name, m in v.items():
        a = np.array([m.get(k) for k in s.agg_ik[clean]])
        b = np.array([m.get(k) for k in s.non_ik[clean]])
        miss[name] = (a == b).astype(int)
        lo, hi = boot_ci(a == b)
        full_a = np.array([m.get(k) for k in s.agg_ik])
        full_b = np.array([m.get(k) for k in s.non_ik])
        rows.append(dict(filter=name, n_clean=int(clean.sum()),
                         missed_all_500=round(100 * (full_a == full_b).mean(), 1),
                         missed_clean=round(100 * (a == b).mean(), 1),
                         ci=f"{lo:.1f}-{hi:.1f}"))
    null = "clogP > 3.5 (null)"
    for r in rows:
        if r["filter"] == null:
            r["p_vs_null"] = np.nan
            continue
        x, y = miss[r["filter"]], miss[null]
        r["p_vs_null"] = round(mcnemar_exact(int(((x == 1) & (y == 0)).sum()),
                                             int(((x == 0) & (y == 1)).sum())), 3)
    out = pd.DataFrame(rows).sort_values("missed_clean").reset_index(drop=True)
    out.to_csv(ROOT / "results/miss_rates_clean_set.csv", index=False)
    print()
    print(out.to_string(index=False))
    return clean


if __name__ == "__main__":
    main()
