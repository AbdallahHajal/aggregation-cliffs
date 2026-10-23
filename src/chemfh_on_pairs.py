#!/usr/bin/env python
"""ChemFH over the census, read two ways.

ChemFH returns seven learned probabilities and ten substructure-rule counts. Its
colloidal_aggregator head makes it an eighth aggregation filter, so it is scored on the
pairs exactly as the others were. Its other six heads answer a different question the
manuscript so far only asserts: whether the pairs would look any different under the other
mechanisms grouped under assay interference.

Both members of a pair are scored, so a filter that cannot tell the two apart shows up as a
pair given the same class, and a rule set that cannot tell them apart shows up as a pair
given the same alert count.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROB = ["colloidal_aggregator", "fluc_inhibitor", "blue_fluorescence", "green_fluorescence",
        "reactive", "promiscuous", "other_assay_interference"]
RULE = ["alarm_nmr_rule", "bms_rule", "chelator_rule", "gst_fhs_rule", "his_fhs_rule",
        "luciferace_inhibitor_rule", "ntd_rule", "pains_rule",
        "potential_electrophilic_rule", "lilly_medchem_rule"]
PRETTY = {"colloidal_aggregator": "colloidal aggregation",
          "fluc_inhibitor": "firefly luciferase inhibition",
          "blue_fluorescence": "blue autofluorescence",
          "green_fluorescence": "green autofluorescence",
          "reactive": "chemical reactivity",
          "promiscuous": "promiscuity",
          "other_assay_interference": "other assay interference",
          "alarm_nmr_rule": "ALARM NMR (thiol reactivity)",
          "bms_rule": "BMS",
          "chelator_rule": "chelators",
          "gst_fhs_rule": "GST / GSH",
          "his_fhs_rule": "His-tag chelation",
          "luciferace_inhibitor_rule": "luciferase inhibitor rule",
          "ntd_rule": "NTD",
          "pains_rule": "PAINS",
          "potential_electrophilic_rule": "potential electrophile",
          "lilly_medchem_rule": "Lilly MedChem"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def main():
    keys = pd.read_csv(ROOT / "results/chemfh_input_keys.csv")
    raw = pd.read_csv(ROOT / "results/chemfh_raw.csv")
    if len(keys) != len(raw):
        raise SystemExit(f"row mismatch: {len(keys)} keys vs {len(raw)} predictions")
    d = pd.concat([keys.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    d.to_csv(ROOT / "results/chemfh_scored.csv", index=False)

    cen = pd.read_csv(ROOT / "results/census_full_3375.csv")
    a = d.set_index("inchikey")
    cen = cen[cen.agg_ik.isin(a.index) & cen.non_ik.isin(a.index)].copy()
    print(f"pairs with both members scored: {len(cen):,} of 3,375\n")

    rows = []

    # --- the learned heads, thresholded at the 0.5 the model card specifies ---
    for col in PROB:
        pa = a.loc[cen.agg_ik, col].to_numpy()
        pn = a.loc[cen.non_ik, col].to_numpy()
        same = (pa >= 0.5) == (pn >= 0.5)
        n = len(same)
        k = int(same.sum())
        lo, hi = wilson(k, n)
        # among the pairs it does separate, how often in the direction the labels imply
        diff = ~same
        right = int(((pa >= 0.5) & (pn < 0.5) & diff).sum())
        wrong = int(((pa < 0.5) & (pn >= 0.5) & diff).sum())
        p = binomtest(right, right + wrong, 0.5).pvalue if right + wrong else np.nan
        rows.append(dict(endpoint=PRETTY[col], kind="learned", n=n,
                         same_class_pct=round(100 * k / n, 2),
                         ci=f"{lo:.1f}-{hi:.1f}", separated=int(diff.sum()),
                         correct=right, reversed_=wrong,
                         p_direction=round(p, 4) if p == p else np.nan,
                         median_abs_delta=round(float(np.median(np.abs(pa - pn))), 4)))

    # --- the substructure rules, which either fire identically on both members or do not ---
    for col in RULE:
        ca = a.loc[cen.agg_ik, col].to_numpy()
        cn = a.loc[cen.non_ik, col].to_numpy()
        same = ca == cn
        n = len(same)
        k = int(same.sum())
        lo, hi = wilson(k, n)
        fires = int(((ca > 0) | (cn > 0)).sum())
        rows.append(dict(endpoint=PRETTY[col], kind="rule", n=n,
                         same_class_pct=round(100 * k / n, 2),
                         ci=f"{lo:.1f}-{hi:.1f}", separated=int((~same).sum()),
                         correct=np.nan, reversed_=np.nan, p_direction=np.nan,
                         median_abs_delta=np.nan, pairs_with_any_alert=fires))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/chemfh_on_pairs.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))

    # --- headline numbers for the text ---
    agg = out[out.endpoint == "colloidal aggregation"].iloc[0]
    print(f"\nChemFH colloidal aggregation head: same class for both members of "
          f"{agg.same_class_pct}% of {agg.n:,} pairs (95% CI {agg.ci})")
    learned = out[out.kind == "learned"]
    print(f"learned heads span {learned.same_class_pct.min()}% to "
          f"{learned.same_class_pct.max()}%")
    rule = out[out.kind == "rule"]
    print(f"substructure rules span {rule.same_class_pct.min()}% to "
          f"{rule.same_class_pct.max()}%")


if __name__ == "__main__":
    main()
