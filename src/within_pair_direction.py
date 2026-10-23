#!/usr/bin/env python
"""Two questions per filter, kept apart because they have different answers.

The first is whether the filter's DECISION separates a pair. That is the miss rate already
reported, and it depends on where the tool's thresholds sit.

The second is whether the filter's SCORE ranks the two members in the direction the measured
labels imply, ignoring thresholds entirely. A tool can fail the first and pass the second,
which is exactly what the BAD Molecule Filter does: its probability moves by a median of
0.014 across a pair while its own decision band is 0.3 wide, so the ranking information
never reaches the output.

Filters returning only a category, Aggregator Advisor and ChemAGG, cannot enter the second
question and are reported as not applicable rather than as zero.

Writes results/within_pair_direction.csv.
"""
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    overlap = set(pd.read_csv(
        ROOT / "results/bad_training_overlap_inchikeys.csv").inchikey)

    scores = {}
    b = pd.read_csv(ROOT / "results/pairs_sample_500_bad.csv")
    scores["BAD Molecule Filter"] = dict(zip(b.inchikey, b.bad_prob))
    f = pd.read_csv(ROOT / "results/chemfh_scored.csv")
    scores["ChemFH"] = dict(zip(f.inchikey, f.colloidal_aggregator))
    d = ROOT / "results/pairs_sample_500_deepscams.csv"
    if d.exists():
        dd = pd.read_csv(d)
        pc = [c for c in dd.columns if "prob" in c.lower()]
        if pc:
            scores["DeepSCAMs"] = dict(zip(dd.inchikey, dd[pc[0]]))
    scores["clogP > 3.5 (null)"] = {**dict(zip(s.agg_ik, s.agg_clogp)),
                                    **dict(zip(s.non_ik, s.non_clogp))}

    rows = []
    for name, m in scores.items():
        pa = np.array([m.get(k, np.nan) for k in s.agg_ik], float)
        pn = np.array([m.get(k, np.nan) for k in s.non_ik], float)
        # ties carry no direction and are excluded rather than split
        ok = ~np.isnan(pa) & ~np.isnan(pn) & (pa != pn)
        hi, n = int((pa[ok] > pn[ok]).sum()), int(ok.sum())
        lo_ci, hi_ci = wilson(hi, n)
        rec = dict(filter=name, n_ranked=n, correct_direction=round(100 * hi / n, 1),
                   ci=f"{lo_ci:.1f}-{hi_ci:.1f}",
                   p_exact=binomtest(hi, n, 0.5).pvalue,
                   median_abs_delta=round(float(np.median(np.abs(pa - pn)[ok])), 4))
        if name == "BAD Molecule Filter":
            clean = ok & ~s.agg_ik.isin(overlap).to_numpy() \
                       & ~s.non_ik.isin(overlap).to_numpy()
            k2, n2 = int((pa[clean] > pn[clean]).sum()), int(clean.sum())
            rec["n_untrained"] = n2
            rec["correct_direction_untrained"] = round(100 * k2 / n2, 1)
            rec["p_exact_untrained"] = binomtest(k2, n2, 0.5).pvalue
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("correct_direction", ascending=False)
    out.to_csv(ROOT / "results/within_pair_direction.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))

    b_all = b.bad_prob
    print(f"\nBAD probability over {len(b_all)} sampled structures: "
          f"min {b_all.min():.3f}, median {b_all.median():.3f}, max {b_all.max():.3f}")
    print(f"  above its 0.6 aggregator threshold: {(b_all > 0.6).sum()}")
    print(f"  inside its 0.3-0.6 ambiguous band:  {b_all.between(0.3, 0.6).sum()}")


if __name__ == "__main__":
    main()
