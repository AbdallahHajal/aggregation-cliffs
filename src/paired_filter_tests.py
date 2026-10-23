#!/usr/bin/env python
"""Paired comparison of filter miss rates, and the forced-floor audit.

Two objections from adversarial review are answered here.

Marginal bootstrap intervals drawn under one fixed seed share their resample indices, so
overlap between them carries no information about whether two filters differ. The filters
are scored on the SAME pairs, so the comparison is paired and belongs in a McNemar test.

Separately, some pairs cannot be resolved by some filters as a matter of arithmetic: a
pair whose members have identical ECFP4 fingerprints cannot be separated by a model whose
input is that fingerprint, and a pair whose members have identical clogP cannot be
separated by a clogP threshold. Those pairs contribute a forced floor to the miss rate, so
every rate is reported with and without them.
"""
import itertools
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLOGP_CUT = 3.5


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def verdicts():
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    out = {"clogP > 3.5 (null)": (
        np.where(s.agg_clogp > CLOGP_CUT, "agg", "non"),
        np.where(s.non_clogp > CLOGP_CUT, "agg", "non"))}

    adv = pd.read_csv(ROOT / "results/pairs_sample_500_advisor.csv")
    m = dict(zip(adv.inchikey, np.where(adv.advisor_flagged == 1, "agg", "non")))
    out["Aggregator Advisor"] = ([m.get(k) for k in s.agg_ik], [m.get(k) for k in s.non_ik])

    cag = pd.read_csv(ROOT / "results/pairs_sample_500_chemagg.csv")
    m = dict(zip(cag.inchikey, np.where(cag.chemagg_cat == 1, "agg", "non")))
    out["ChemAGG"] = ([m.get(k) for k in s.agg_ik], [m.get(k) for k in s.non_ik])

    bad = pd.read_csv(ROOT / "results/pairs_sample_500_bad.csv")
    def call(p):
        return "agg" if p > 0.6 else ("non" if p < 0.3 else "amb")
    m = {k: call(v) for k, v in zip(bad.inchikey, bad.bad_prob)}
    out["BAD Molecule Filter"] = ([m.get(k) for k in s.agg_ik], [m.get(k) for k in s.non_ik])

    # ChemFH runs locally, so it covers the whole census; it is scored here on the same 500
    # pairs as the rest so the McNemar comparison stays paired. Threshold 0.5 is the one the
    # model card gives.
    fh = pd.read_csv(ROOT / "results/chemfh_scored.csv")
    m = dict(zip(fh.inchikey, np.where(fh.colloidal_aggregator >= 0.5, "agg", "non")))
    out["ChemFH"] = ([m.get(k) for k in s.agg_ik], [m.get(k) for k in s.non_ik])
    return s, out


def main():
    s, v = verdicts()
    n = len(s)

    # pairs no fingerprint model can separate, and pairs no clogP rule can separate
    fp_identical = (s.tanimoto >= 0.999999).to_numpy()
    clogp_identical = (s.d_clogp.abs() < 1e-9).to_numpy()
    print(f"pairs: {n}")
    print(f"  ECFP4-identical (Tanimoto 1.000): {fp_identical.sum()}  "
          f"-> unresolvable by any model on that fingerprint")
    print(f"  clogP-identical:                  {clogp_identical.sum()}  "
          f"-> unresolvable by a clogP threshold")

    missed = {}
    print("\n" + "=" * 78)
    print("MISS RATE, WITH AND WITHOUT THE PAIRS THAT CANNOT BE RESOLVED IN PRINCIPLE")
    print("=" * 78)
    for name, (a, b) in v.items():
        ok = np.array([x is not None and y is not None for x, y in zip(a, b)])
        same = np.array([x == y for x, y in zip(a, b)])
        missed[name] = np.where(ok, same, np.nan)
        excl = clogp_identical if "clogP" in name else fp_identical
        keep = ok & ~excl
        print(f"  {name:22s} all {100*same[ok].mean():5.1f}% (n={ok.sum():3d})   "
              f"excluding forced {100*same[keep].mean():5.1f}% (n={keep.sum():3d})")

    print("\n" + "=" * 78)
    print("PAIRED COMPARISON (exact McNemar on the same 500 pairs)")
    print("=" * 78)
    rows = []
    for x, y in itertools.combinations(missed, 2):
        mx, my = missed[x], missed[y]
        ok = ~np.isnan(mx) & ~np.isnan(my)
        b = int(((mx == 1) & (my == 0) & ok).sum())
        c = int(((mx == 0) & (my == 1) & ok).sum())
        p = mcnemar_exact(b, c)
        rows.append({"a": x, "b": y, "n": int(ok.sum()), "a_only": b, "b_only": c,
                     "p_exact": round(p, 4)})
        flag = "" if p >= 0.05 else "  <- differ"
        print(f"  {x:22s} vs {y:22s} n={ok.sum():3d}  "
              f"{b:3d} / {c:3d} discordant   p = {p:.3f}{flag}")

    pd.DataFrame(rows).to_csv(ROOT / "results/filter_paired_tests.csv", index=False)
    print("\nwrote results/filter_paired_tests.csv")
    print("\nA non-significant p means the two filters miss the SAME pairs, not that")
    print("their miss rates are similar by coincidence.")


if __name__ == "__main__":
    main()
