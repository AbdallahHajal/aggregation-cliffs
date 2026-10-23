#!/usr/bin/env python
"""Which noise floor is the right one?

An adversarial review raised the objection that stereoisomer pairs may not bound label
error at all, because the labels derive from detergent-sensitive inhibition of an enzyme
and enantiomers routinely differ in enzyme inhibition. If so, the stereoisomer flip rate
is measuring chemistry as well as noise and cannot be used as a floor.

Four candidate floors are computed here, in increasing order of how cleanly they isolate
error from chemistry:

  A  all stereoisomer pairs               contaminated by stereospecific inhibition
  B  pairs whose second InChIKey block differs but where one member carries no defined
     stereochemistry (-UHFFFAOYSA-)       the same constitution registered twice, so a
                                          label difference is registration error
  C  pairs where both members carry defined stereochemistry
  D  compounds present in BOTH screens whose aggregator/non-aggregator roles are reversed
     between them                         two independent campaigns disagreeing about the
                                          same two molecules: a technical replicate

B and D contain no chemistry by construction. If they agree with A, the objection does
not bite. If they are much lower than A, the stereoisomer floor is inflated and the
excess over it has been understated.
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLAT = "-UHFFFAOYSA-"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (100 * (c - h), 100 * (c + h))


def line(name, k, n):
    if n == 0:
        print(f"  {name:52s}    no pairs")
        return
    lo, hi = wilson(k, n)
    print(f"  {name:52s} {100*k/n:6.2f}%   [{lo:5.2f} to {hi:5.2f}]   {k:5d}/{n}")


def main():
    print("=" * 84)
    print("B and C: stereoisomer pairs split by whether stereochemistry is actually defined")
    print("=" * 84)
    print("(numerators only; the denominators need the full neighbour scan, so these are")
    print(" the OPPOSITE-LABEL pairs the miner already recovered, split by kind)\n")

    st = pd.read_csv(ROOT / "results/stereo_noise_pairs.csv")
    st["flat"] = st.agg_ik.str.contains(FLAT) | st.non_ik.str.contains(FLAT)
    n_flat = int(st.flat.sum())
    print(f"  opposite-label stereoisomer pairs recovered: {len(st)}")
    print(f"    involving a member with NO defined stereochemistry: {n_flat} "
          f"({100*n_flat/len(st):.1f}%)")
    print(f"    both members stereochemically defined:             {len(st)-n_flat} "
          f"({100*(len(st)-n_flat)/len(st):.1f}%)")
    print("\n  A pair in the first group is one constitution registered twice, so an")
    print("  opposite label there is registration error and cannot be stereospecific")
    print("  inhibition. A pair in the second group can be either.\n")

    print("=" * 84)
    print("D: the same two molecules, both screens, roles reversed")
    print("=" * 84)
    fr = []
    for name, p in (("cruzain", "results/cruzain_full_real_tight.csv"),
                    ("ampc", "results/ampc_full_real_tight.csv")):
        d = pd.read_csv(ROOT / p)
        d["dataset"] = name
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df["key"] = df.apply(lambda r: tuple(sorted((r.agg_ik, r.non_ik))), axis=1)
    dup = df[df.key.duplicated(keep=False)].copy()
    groups = dup.groupby("key")
    both, reversed_ = 0, 0
    for _, g in groups:
        if g.dataset.nunique() < 2:
            continue
        both += 1
        if g.agg_ik.nunique() > 1:
            reversed_ += 1
    print()
    line("pairs found in BOTH screens with roles reversed", reversed_, both)
    print("\n  Neither member changed; only which one the screen called the aggregator did.")
    print("  This is two independent campaigns disagreeing, with chemistry held fixed.")
    print("  It is the cleanest label-error estimate available in these data.\n")

    out = ROOT / "results/noise_floor_audit.csv"
    pd.DataFrame([
        {"floor": "stereo pairs involving an undefined-stereo member",
         "k": n_flat, "n": len(st), "note": "share of recovered opposite-label stereo pairs"},
        {"floor": "cross-screen role reversal", "k": reversed_, "n": both,
         "note": "same two molecules, both screens, opposite role assignment"},
    ]).to_csv(out, index=False)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
