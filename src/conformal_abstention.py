#!/usr/bin/env python
"""What would these filters say if they were allowed to abstain?

Every filter evaluated here returns a class. On a matched pair whose two members it cannot
distinguish, returning a class is the wrong output: the honest answer is that the evidence
does not separate them. Conformal prediction makes that answer available. At a chosen
significance level the predictor returns a SET, which may hold one label, both labels, or
neither, and the set is guaranteed to contain the true label at the stated rate.

Each filter's continuous score is calibrated with an inductive Mondrian conformal
classifier (crepes; Bostrom, COPA) on one half of the census molecules and evaluated on the
other. Two quantities are reported: how often the predictor returns both labels, which is
the filter admitting it cannot decide, and how often the two members of a pair receive
identical prediction sets, which is the conformal restatement of the miss rate.

This does not add evidence. It restates the audit as a recommendation: a calibrated version
of each of these tools would decline to separate the pairs rather than guessing.
"""
import pathlib

import numpy as np
import pandas as pd
from crepes import WrapClassifier  # noqa: F401  (import checks the dependency)
from crepes.extras import MondrianCategorizer  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20260928
EPS = (0.05, 0.10, 0.20)


def nonconformity(scores, labels):
    """Inductive conformal, class-conditional: alpha = 1 - p(true class)."""
    p1 = np.asarray(scores, float)
    y = np.asarray(labels, int)
    return np.where(y == 1, 1 - p1, p1)


def predict_sets(cal_alpha, cal_y, test_p1, eps):
    """Mondrian (class-conditional) prediction sets for a binary score."""
    out = []
    q = {}
    for c in (0, 1):
        a = np.sort(cal_alpha[cal_y == c])
        n = len(a)
        k = int(np.ceil((n + 1) * (1 - eps))) - 1
        q[c] = a[min(max(k, 0), n - 1)] if n else 1.0
    for p1 in test_p1:
        s = set()
        if p1 <= q[1]:
            s.add(1)
        if (1 - p1) <= q[0]:
            s.add(0)
        out.append(frozenset(s))
    return out


def load():
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    feeds = {}

    bad = pd.read_csv(ROOT / "results/pairs_sample_500_bad.csv")
    feeds["BAD Molecule Filter"] = dict(zip(bad.inchikey, bad.bad_prob))

    sc = pd.read_csv(ROOT / "results/pairs_sample_500_scam.csv")
    for tag, nm in (("beta_lactamase", "SCAM Detective (AmpC)"),
                    ("cruzain", "SCAM Detective (cruzain)")):
        col = "scam_%s_prob" % tag
        if col in sc:
            feeds[nm] = dict(zip(sc.inchikey, sc[col]))

    ds = pd.read_csv(ROOT / "results/pairs_sample_500_deepscams.csv")
    feeds["DeepSCAMs"] = dict(zip(ds.inchikey, ds.deepscams_prob))

    # the null model scored as a probability by rank of clogP
    clog = {**dict(zip(s.agg_ik, s.agg_clogp)), **dict(zip(s.non_ik, s.non_clogp))}
    lo, hi = min(clog.values()), max(clog.values())
    feeds["clogP (null)"] = {k: (v - lo) / (hi - lo) for k, v in clog.items()}
    return s, feeds


def main():
    s, feeds = load()
    truth = {**dict(zip(s.agg_ik, np.ones(len(s), int))),
             **dict(zip(s.non_ik, np.zeros(len(s), int)))}

    rng = np.random.default_rng(SEED)
    keys = np.array(sorted(truth))
    rng.shuffle(keys)
    cal_keys = set(keys[: len(keys) // 2])

    rows = []
    print("=" * 88)
    print("CONFORMAL RESTATEMENT: how often can each filter not decide?")
    print("=" * 88)
    for name, sc in feeds.items():
        ck = [k for k in keys if k in cal_keys and k in sc and not pd.isna(sc[k])]
        tk = [k for k in keys if k not in cal_keys and k in sc and not pd.isna(sc[k])]
        if len(ck) < 30 or not tk:
            print(f"  {name:26s} too few scored molecules")
            continue
        cal_y = np.array([truth[k] for k in ck])
        cal_a = nonconformity([sc[k] for k in ck], cal_y)
        for eps in EPS:
            sets = predict_sets(cal_a, cal_y, [sc[k] for k in tk], eps)
            both = np.mean([len(x) == 2 for x in sets]) * 100
            empty = np.mean([len(x) == 0 for x in sets]) * 100
            # pair-level: do both members get the same prediction set?
            m = dict(zip(tk, sets))
            pairs = [(m[a], m[b]) for a, b in zip(s.agg_ik, s.non_ik)
                     if a in m and b in m]
            same = np.mean([x == y for x, y in pairs]) * 100 if pairs else float("nan")
            rows.append(dict(filter=name, eps=eps, both_labels=round(both, 1),
                             empty=round(empty, 1), identical_sets=round(same, 1),
                             n_pairs=len(pairs)))
            print(f"  {name:26s} eps={eps:<5.2f} returns both labels {both:5.1f}%   "
                  f"empty {empty:4.1f}%   identical sets within a pair {same:5.1f}% "
                  f"(n={len(pairs)})")
    pd.DataFrame(rows).to_csv(ROOT / "results/conformal_abstention.csv", index=False)
    print("\nwrote results/conformal_abstention.csv")
    print("\nA filter returning both labels is declining to separate the pair, which is the")
    print("correct output when its own calibration says the evidence does not support one.")


if __name__ == "__main__":
    main()
