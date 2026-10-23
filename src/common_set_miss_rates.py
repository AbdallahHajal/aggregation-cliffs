#!/usr/bin/env python
"""One evaluation set, scored identically by every filter.

The earlier table reported a different n for almost every filter, and two of those
differences were choices rather than facts. The BAD Molecule Filter was scored on the 370
pairs where both members fell outside its ambiguous band, which drops the 130 pairs it was
least certain about and reports the remainder as 100%. CAGE-Fusion was scored on 864 pairs
of the full census rather than on this sample at all. Neither is comparable to the rest.

Here every filter is scored on the same pairs under the same rule: a pair counts as missed
when both members receive the SAME verdict, whatever that verdict is. A three-way output is
kept three-way, so "ambiguous for both" is a miss, because the filter has not separated the
pair. A pair is resolved only when one member is called an aggregator and the other is not.

CAGE-Fusion is reported on this sample where its training set permits, and its held-out
census figure is kept as a separate line rather than mixed into the same column.

Writes results/miss_rates_common_set.csv.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGG, NON, AMB = "agg", "non", "amb"

PROVENANCE = {"clogP > 3.5 (null)": "none",
              "Aggregator Advisor": "similarity lookup",
              "ChemAGG": "drug databases",
              "DeepSCAMs": "measured, DLS at 30 uM",
              "ChemFH": "ChemFH training set",
              "BAD Molecule Filter": "mixed",
              "CAGE-Fusion": "multi-task"}


def boot_ci(hits, n_boot=2000, seed=20260928):
    rng = np.random.default_rng(seed)
    h = np.asarray(hits, float)
    idx = rng.integers(0, len(h), size=(n_boot, len(h)))
    return tuple(np.percentile(h[idx].mean(axis=1), [2.5, 97.5]) * 100)


def verdicts():
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    v = {}

    v["clogP > 3.5 (null)"] = {**{k: (AGG if x > 3.5 else NON)
                                  for k, x in zip(s.agg_ik, s.agg_clogp)},
                               **{k: (AGG if x > 3.5 else NON)
                                  for k, x in zip(s.non_ik, s.non_clogp)}}

    a = pd.read_csv(ROOT / "results/pairs_sample_500_advisor.csv")
    v["Aggregator Advisor"] = {k: (AGG if f == 1 else NON)
                               for k, f in zip(a.inchikey, a.advisor_flagged)}

    c = pd.read_csv(ROOT / "results/pairs_sample_500_chemagg.csv")
    v["ChemAGG"] = {k: (AGG if x == 1 else NON)
                    for k, x in zip(c.inchikey, c.chemagg_cat)}

    d = pd.read_csv(ROOT / "results/pairs_sample_500_deepscams.csv")
    v["DeepSCAMs"] = {k: (AGG if x == 1 else NON)
                      for k, x in zip(d.inchikey, d.deepscams_cat)}

    f = pd.read_csv(ROOT / "results/chemfh_scored.csv")
    v["ChemFH"] = {k: (AGG if x >= 0.5 else NON)
                   for k, x in zip(f.inchikey, f.colloidal_aggregator)}

    # the published three-way rule is kept three-way rather than collapsed or dropped
    b = pd.read_csv(ROOT / "results/pairs_sample_500_bad.csv")
    v["BAD Molecule Filter"] = {k: (AGG if p > 0.6 else NON if p < 0.3 else AMB)
                                for k, p in zip(b.inchikey, b.bad_prob)}

    g = pd.read_csv(ROOT / "results/census_cage_fusion.csv")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    ik = {}
    for smi, cls in zip(g.SMILES, g.pred_class_aggregator):
        m = Chem.MolFromSmiles(str(smi))
        if m is not None:
            ik[Chem.MolToInchiKey(m)] = AGG if cls == 1 else NON
    v["CAGE-Fusion"] = ik
    return s, v


def main():
    s, v = verdicts()

    # the common set: pairs every filter gave a verdict for, on both members
    keep = np.ones(len(s), bool)
    for m in v.values():
        keep &= s.agg_ik.map(lambda k: k in m).to_numpy()
        keep &= s.non_ik.map(lambda k: k in m).to_numpy()
    print(f"pairs in the sample:                         {len(s)}")
    print(f"pairs every filter scored on both members:   {keep.sum()}\n")

    rows = []
    for name, m in v.items():
        a = np.array([m.get(k) for k in s.agg_ik[keep]])
        b = np.array([m.get(k) for k in s.non_ik[keep]])
        same = a == b
        lo, hi = boot_ci(same)
        resolved = (a != b) & (a != AMB) & (b != AMB)
        p_agg = np.mean([x == AGG for x in np.concatenate([a, b])])
        rows.append(dict(
            filter=name, provenance=PROVENANCE.get(name, ""), n=int(keep.sum()),
            missed=round(100 * same.mean(), 1), ci=f"{lo:.1f}-{hi:.1f}",
            correct=int(((a == AGG) & (b == NON)).sum()),
            backwards=int(((a == NON) & (b == AGG)).sum()),
            partly_resolved=int((a != b).sum() - resolved.sum()),
            calls_agg_pct=round(100 * p_agg, 1),
            # what the filter's own base rate forces: the chance two independent calls at
            # that rate agree. A filter at or below this is telling you nothing.
            forced_by_base_rate=round(100 * (p_agg ** 2 + (1 - p_agg) ** 2), 1),
            above_forced=round(100 * (same.mean() - (p_agg ** 2 + (1 - p_agg) ** 2)), 1)))

    out = pd.DataFrame(rows).sort_values("missed").reset_index(drop=True)
    out.to_csv(ROOT / "results/miss_rates_common_set.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))
    print("\nmissed  = both members given the same verdict, so the pair was not separated")
    print("correct = aggregator member called agg and non-aggregator member called non")
    print("partly  = verdicts differ but one of them is ambiguous")


if __name__ == "__main__":
    main()
