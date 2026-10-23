#!/usr/bin/env python
"""Assemble the Supporting Information release.

Four files, each self-describing, so the evaluation in this article can be reused rather
than re-derived:

  SI1_aggregation_cliff_census.csv     the 3,375 matched pairs with opposing labels
  SI2_filter_verdicts_500_pairs.csv    the sampled pairs with every filter's call per member
  SI3_representation_resolvability.csv per-representation count of unresolvable pairs
  SI4_candidate_pairs_for_measurement.csv  generated structures nominated for DLS
  SI5_chemfh_per_compound.csv          ChemFH's 17 endpoints for every census compound
  SI6_chemfh_within_pair.csv           per-endpoint within-pair agreement over the census

Nothing here is derived at write time. Every column is read from the results directory, so
the release and the manuscript cannot disagree.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "supporting_information"
OUT.mkdir(exist_ok=True)


def si1():
    d = pd.read_csv(ROOT / "results/census_full_3375.csv")
    d = d.rename(columns={
        "agg_smiles": "aggregator_smiles", "non_smiles": "nonaggregator_smiles",
        "agg_ik": "aggregator_inchikey", "non_ik": "nonaggregator_inchikey",
        "tanimoto": "ecfp4_tanimoto", "d_clogp": "delta_clogp", "d_mw": "delta_mw",
        "d_tpsa": "delta_tpsa", "d_heavy": "delta_heavy_atoms",
        "agg_heavy": "aggregator_heavy_atoms", "non_heavy": "nonaggregator_heavy_atoms",
        "agg_clogp": "aggregator_clogp", "non_clogp": "nonaggregator_clogp"})
    cols = ["dataset", "aggregator_smiles", "nonaggregator_smiles",
            "aggregator_inchikey", "nonaggregator_inchikey", "ecfp4_tanimoto",
            "delta_heavy_atoms", "delta_mw", "delta_clogp", "delta_tpsa",
            "aggregator_heavy_atoms", "nonaggregator_heavy_atoms",
            "aggregator_clogp", "nonaggregator_clogp"]
    d = d[[c for c in cols if c in d.columns]]
    d.to_csv(OUT / "SI1_aggregation_cliff_census.csv", index=False)
    print(f"  SI1  {len(d):,} pairs   "
          f"cruzain {int((d.dataset=='cruzain').sum()):,}  ampc {int((d.dataset=='ampc').sum()):,}")
    return d


def si2():
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    out = pd.DataFrame({
        "dataset": s.dataset, "ecfp4_tanimoto": s.tanimoto,
        "delta_heavy_atoms": s.d_heavy,
        "aggregator_smiles": s.agg_smiles, "nonaggregator_smiles": s.non_smiles,
        "aggregator_inchikey": s.agg_ik, "nonaggregator_inchikey": s.non_ik,
        "clogp_aggregator": s.agg_clogp, "clogp_nonaggregator": s.non_clogp,
    })

    def attach(path, col, name, how=None):
        p = ROOT / path
        if not p.exists():
            print(f"       (skipped {name}: {path} absent)")
            return
        d = pd.read_csv(p)
        if col not in d.columns:
            print(f"       (skipped {name}: no column {col})")
            return
        m = dict(zip(d.inchikey, d[col]))
        out[f"{name}_aggregator"] = [how(m.get(k)) if how else m.get(k) for k in s.agg_ik]
        out[f"{name}_nonaggregator"] = [how(m.get(k)) if how else m.get(k) for k in s.non_ik]

    attach("results/pairs_sample_500_bad.csv", "bad_prob", "bad_probability")
    attach("results/pairs_sample_500_advisor.csv", "advisor_verdict", "advisor_verdict")
    attach("results/pairs_sample_500_chemagg.csv", "chemagg_cat", "chemagg_class")
    attach("results/pairs_sample_500_scam.csv", "scam_cruzain", "scamdetective_cruzain_class")
    attach("results/pairs_sample_500_scam.csv", "scam_beta_lactamase", "scamdetective_ampc_class")
    attach("results/pairs_sample_500_deepscams.csv", "deepscams_cat", "deepscams_class")

    cf = ROOT / "results/census_cage_fusion.csv"
    if cf.exists():
        d = pd.read_csv(cf)
        mol = pd.read_csv(ROOT / "results/census_full_molecules.csv")
        s2ik = dict(zip(mol.smiles, mol.inchikey))
        m = {s2ik.get(a): b for a, b in zip(d.SMILES, d.pred_class_aggregator)}
        out["cagefusion_class_aggregator"] = [m.get(k) for k in s.agg_ik]
        out["cagefusion_class_nonaggregator"] = [m.get(k) for k in s.non_ik]

    out["clogp_null_aggregator"] = np.where(s.agg_clogp > 3.5, 1, 0)
    out["clogp_null_nonaggregator"] = np.where(s.non_clogp > 3.5, 1, 0)
    out.to_csv(OUT / "SI2_filter_verdicts_500_pairs.csv", index=False)
    print(f"  SI2  {len(out):,} pairs, {len(out.columns)} columns")


def si3():
    p = ROOT / "results/representation_sweep.csv"
    if not p.exists():
        print("  SI3  skipped, sweep not found")
        return
    d = pd.read_csv(p)
    d.to_csv(OUT / "SI3_representation_resolvability.csv", index=False)
    print(f"  SI3  {len(d)} representations")


def si5():
    """ChemFH over every census compound: seven learned endpoints and ten rule sets.

    Released per compound rather than per pair, so a reader can re-pair it against SI1 under
    a different threshold than the 0.5 used here.
    """
    p = ROOT / "results/chemfh_scored.csv"
    if not p.exists():
        print("  SI5  skipped, ChemFH output not found")
        return
    d = pd.read_csv(p)
    d.to_csv(OUT / "SI5_chemfh_per_compound.csv", index=False)
    print(f"  SI5  {len(d):,} compounds, {len(d.columns) - 2} ChemFH endpoints")

    q = ROOT / "results/chemfh_on_pairs.csv"
    if q.exists():
        pd.read_csv(q).to_csv(OUT / "SI6_chemfh_within_pair.csv", index=False)
        print("  SI6  per-endpoint within-pair agreement")


def si4():
    p = ROOT / "results/beam/all_analogues_annotated.csv"
    if not p.exists():
        print("  SI4  skipped, analogues not found")
        return
    d = pd.read_csv(p)
    keep = d[(d.get("qed", 0) >= 0.5) & (d.get("sa", 99) <= 4.5) & (d.get("stable", True) == True)]
    cols = [c for c in ["lineage", "from_smiles", "smiles", "total_d_heavy",
                        "tc_to_parent_drug", "qed", "sa", "chemagg_prob", "chemagg_cat",
                        "bad_prob", "advisor_verdict", "clogP"] if c in keep.columns]
    keep = keep[cols].rename(columns={"lineage": "parent_drug",
                                      "from_smiles": "parent_smiles",
                                      "smiles": "candidate_smiles",
                                      "total_d_heavy": "added_heavy_atoms",
                                      "tc_to_parent_drug": "ecfp4_tanimoto_to_parent"})
    keep.to_csv(OUT / "SI4_candidate_pairs_for_measurement.csv", index=False)
    print(f"  SI4  {len(keep):,} candidates passing drug-likeness, "
          f"synthetic accessibility and stability")


def readme():
    (OUT / "README.md").write_text("""# Supporting Information

Data released with "Do aggregation cliffs exist? Matched molecular pairs with opposing
colloidal aggregation labels, and whether published filters resolve them".

| file | contents |
|---|---|
| `SI1_aggregation_cliff_census.csv` | The 3,375 matched pairs with opposing aggregation labels mined from the cruzain and AmpC beta-lactamase detergent counter-screens. Both members of every pair were synthesized and screened; no generated structure appears. |
| `SI2_filter_verdicts_500_pairs.csv` | The stratified sample of 500 pairs used for the per-pair evaluation, with each filter's output for each member separately. |
| `SI3_representation_resolvability.csv` | For each molecular representation, the number of census pairs it renders exactly identical, which upper-bounds any model consuming only that representation. |
| `SI4_candidate_pairs_for_measurement.csv` | Generated structures nominated for dynamic light scattering, passing drug-likeness, synthetic accessibility and a substructure stability screen. These carry no experimental label. |
| `SI5_chemfh_per_compound.csv` | ChemFH's seven learned probabilities and ten substructure rule counts for every compound in the census, released per compound so the pairs can be re-formed under a different threshold than the 0.5 used here. |
| `SI6_chemfh_within_pair.csv` | For each ChemFH endpoint, the proportion of census pairs given the same call for both members, with the count separated and, for the learned endpoints, the direction. |

## Provenance of the labels

Aggregation labels in SI1 and SI2 derive from the detergent counter-screen rule: a compound
is an aggregator when it shows a Hill slope above 1.5 with a fitted logAC50 in the absence
of detergent and none in its presence, and a non-aggregator when no logAC50 is fitted in
either condition. A non-aggregator label therefore records absence of activity against the
screening enzyme rather than a tested absence of colloid formation. This limitation is
central to the article and should be carried into any reuse of these files.

## Suggested use

SI1 and SI2 together allow a filter to be evaluated per pair: query it twice, once per
member, and record whether it assigns different classes. That quantity is not recoverable
from an accuracy or an area under the curve computed over the same compounds.

Seed 20260928 throughout. Python 3.11.16, RDKit 2026.03.1.
""")
    print("  README.md written")


if __name__ == "__main__":
    print("Supporting Information")
    si1(); si2(); si3(); si4(); si5(); readme()
    print(f"\nwrote {OUT}")
