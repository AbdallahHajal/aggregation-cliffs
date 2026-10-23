#!/usr/bin/env python
"""Result 2 report: per-filter miss rate over the sampled census pairs.

A filter is asked twice, once per member of the pair. Same verdict for both members
means it did not see the cliff. Joins whatever per-molecule verdict files exist and
reports each filter it finds, so it can be re-run as BAD and ChemAGG land.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20260928
CLOGP_CUT = 3.5


def boot_ci(hits, n_boot=2000):
    rng = np.random.default_rng(SEED)
    hits = np.asarray(hits, dtype=float)
    idx = rng.integers(0, len(hits), size=(n_boot, len(hits)))
    return tuple(np.percentile(hits[idx].mean(axis=1), [2.5, 97.5]) * 100)


def report(name, agg_call, non_call, n_total):
    """agg_call / non_call are 'agg' | 'non' | None per pair."""
    ok = [(a, b) for a, b in zip(agg_call, non_call) if a is not None and b is not None]
    if not ok:
        print(f"  {name:22s} no verdicts")
        return None
    missed = np.array([a == b for a, b in ok])
    correct = np.array([(a == "agg") and (b == "non") for a, b in ok])
    backwards = np.array([(a == "non") and (b == "agg") for a, b in ok])
    lo, hi = boot_ci(missed)
    print(f"  {name:22s} n={len(ok):4d}  MISSED {missed.mean()*100:5.1f}% "
          f"[{lo:4.1f}-{hi:4.1f}]   correct {correct.sum():3d}   backwards {backwards.sum():3d}")
    return dict(filter=name, n=len(ok), missed_pct=round(missed.mean() * 100, 2),
                ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                n_correct=int(correct.sum()), n_backwards=int(backwards.sum()),
                n_missed=int(missed.sum()))


def main():
    pairs = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    print(f"pairs: {len(pairs):,}\n")
    print("=" * 86)
    print("RESULT 2 - does the filter give different verdicts to the two members of a cliff?")
    print("=" * 86)

    rows = []

    # --- clogP null: inputs already in the census ---
    rows.append(report(
        "clogP > 3.5 (null)",
        np.where(pairs["agg_clogp"] > CLOGP_CUT, "agg", "non"),
        np.where(pairs["non_clogp"] > CLOGP_CUT, "agg", "non"),
        len(pairs)))

    # --- external filters, joined by InChIKey ---
    specs = [
        ("Aggregator Advisor", "results/pairs_sample_500_advisor.csv",
         lambda d: np.where(d["advisor_flagged"] == 1, "agg", "non")),
        ("BAD Molecule Filter", "results/pairs_sample_500_bad.csv",
         lambda d: np.where(d["bad_prob"] > 0.6, "agg",
                            np.where(d["bad_prob"] < 0.3, "non", None))),
        ("ChemAGG", "results/pairs_sample_500_chemagg.csv",
         lambda d: np.where(d["chemagg_cat"].astype(str).isin(["1", "Aggregator"]),
                            "agg", "non")),
    ]

    for name, rel, call in specs:
        path = ROOT / rel
        if not path.exists():
            print(f"  {name:22s} not run yet ({rel})")
            continue
        d = pd.read_csv(path)
        d["call"] = call(d)
        m = dict(zip(d["inchikey"], d["call"]))
        rows.append(report(name,
                           [m.get(k) for k in pairs["agg_ik"]],
                           [m.get(k) for k in pairs["non_ik"]],
                           len(pairs)))

    rows = [r for r in rows if r]
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/RESULT2_miss_rates.csv", index=False)
    print(f"\nwrote results/RESULT2_miss_rates.csv")

    # Advisor detail: how often is its verdict identical, not merely same class
    adv = ROOT / "results/pairs_sample_500_advisor.csv"
    if adv.exists():
        d = pd.read_csv(adv)
        v = dict(zip(d["inchikey"], d["advisor_verdict"]))
        same_exact = sum(1 for a, b in zip(pairs["agg_ik"], pairs["non_ik"])
                         if v.get(a) is not None and v.get(a) == v.get(b))
        print(f"\nAdvisor: identical verdict STRING for both members in "
              f"{same_exact}/{len(pairs)} pairs ({100*same_exact/len(pairs):.1f}%)")
        print("Advisor verdict distribution over the sampled molecules:")
        print(d["advisor_verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
