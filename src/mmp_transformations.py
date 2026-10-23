#!/usr/bin/env python
"""Which formal MMP transformations flip the aggregation label?

The census was mined by shared Murcko scaffold plus an ECFP4 similarity threshold, which
is a similarity criterion rather than a transformation one. A reviewer will ask why a
formal matched-pair definition was not used, so the pairs are re-expressed here as mmpdb
transformations (Dalke, Hert & Kramer, J Chem Inf Model 2018, 10.1021/acs.jcim.8b00173),
and the label-flip rate is reported per transformation.

Stereochemistry is stripped before indexing. mmpdb's environment-SMARTS builder raises
inside RDKit 2026.03.1 on certain double-bond stereo specifications, and matched-pair
analysis is constitutional in any case, so nothing the transformation captures is lost.

Kombo et al. (ACS Omega 2024, 10.1021/acsomega.4c02886) ran MMP analysis on aggregation
and concluded that the vast majority of transformations are neutral. Their pairs were
unconditioned; the pairs here are conditioned on carrying opposite labels, so the two
numbers answer different questions and the comparison is made explicitly.
"""
import pathlib
import sqlite3

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "results/mmp/flat.mmpdb"


def main():
    census = pd.read_csv(ROOT / "results/census_full_3375.csv")
    opp = set()
    for a, b in zip(census.agg_ik, census.non_ik):
        opp.add((a, b))
        opp.add((b, a))
    agg_side = dict()
    for a, b in zip(census.agg_ik, census.non_ik):
        agg_side[(a, b)] = a
        agg_side[(b, a)] = a
    print(f"census pairs with opposing labels: {len(census):,}")

    con = sqlite3.connect(DB)
    q = """
        SELECT c1.public_id AS id1, c2.public_id AS id2,
               rs1.smiles AS frm, rs2.smiles AS to_
        FROM pair
        JOIN compound c1 ON c1.id = pair.compound1_id
        JOIN compound c2 ON c2.id = pair.compound2_id
        JOIN rule_environment re ON re.id = pair.rule_environment_id
        JOIN rule r ON r.id = re.rule_id
        JOIN rule_smiles rs1 ON rs1.id = r.from_smiles_id
        JOIN rule_smiles rs2 ON rs2.id = r.to_smiles_id
    """
    mmp = pd.read_sql(q, con)
    print(f"formal MMP pairs recovered by mmpdb:  {len(mmp):,}")

    mmp["key"] = list(zip(mmp.id1, mmp.id2))
    mmp["in_census"] = [k in opp for k in mmp.key]
    hit = mmp[mmp.in_census].copy()
    uniq = {frozenset(k) for k in hit.key}
    print(f"census pairs mmpdb also calls a formal MMP: {len(uniq):,} "
          f"({100*len(uniq)/len(census):.1f}% of the census)")
    print()

    # orient every transformation from the aggregator to the non-aggregator
    hit["from_agg"] = [agg_side[k] == k[0] for k in hit.key]
    hit["transform"] = [
        (f"{f} >> {t}" if fa else f"{t} >> {f}")
        for f, t, fa in zip(hit.frm, hit.to_, hit.from_agg)
    ]
    hit = hit.drop_duplicates(subset=["transform"] + ["key"])

    top = hit["transform"].value_counts().head(25)
    print("=" * 74)
    print("TRANSFORMATIONS THAT ABOLISH AGGREGATION, oriented aggregator >> non-aggregator")
    print("=" * 74)
    for t, n in top.items():
        print(f"  {n:5d}   {t}")

    out = ROOT / "results/mmp_label_flipping_transforms.csv"
    hit["transform"].value_counts().rename_axis("transform").reset_index(
        name="n_pairs").to_csv(out, index=False)
    print(f"\ndistinct label-flipping transformations: "
          f"{hit['transform'].nunique():,}")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
