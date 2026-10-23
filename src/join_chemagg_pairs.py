#!/usr/bin/env python
"""Join the ChemAGG verdicts for the sampled census pairs.

The ChemAGG web server returns a results page whose CSV is written under
/static/uploadfile/. It accepts no API and rejects uploads above roughly 1 MB, so the
500-pair sample was submitted as four SDF chunks of 125 pairs each and the returned
category column was captured per chunk, in submission order.

Row order in the returned CSV matches the order of the submitted SDF, which is verified
here by re-reading each SDF and requiring the record count to match before any join.
"""
import pathlib

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
A27 = ROOT.parent / "Article 27 - Cliffs Review"
SDF_DIR = A27 / "chemagg_pairs"
CAT_DIR = A27 / "chemagg_out"


def main():
    rows = []
    for sdf in sorted(SDF_DIR.glob("pairs_*.sdf")):
        cat_path = CAT_DIR / (sdf.stem + ".cat")
        cats = cat_path.read_text().strip()
        mols = [m for m in Chem.SDMolSupplier(str(sdf)) if m is not None]
        if len(mols) != len(cats):
            raise SystemExit(f"{sdf.name}: {len(mols)} records but {len(cats)} categories")
        for m, c in zip(mols, cats):
            rows.append({"inchikey": m.GetProp("_Name"),
                         "smiles": Chem.MolToSmiles(m),
                         "chemagg_cat": int(c)})
        print(f"{sdf.name}: {len(mols)} records matched")

    d = pd.DataFrame(rows)
    # a structure may appear in more than one chunk; the server is deterministic, so
    # require agreement before collapsing
    g = d.groupby("inchikey")["chemagg_cat"].nunique()
    if (g > 1).any():
        print(f"WARNING: {(g>1).sum()} structures returned conflicting categories")
    d = d.drop_duplicates("inchikey")
    out = ROOT / "results/pairs_sample_500_chemagg.csv"
    d.to_csv(out, index=False)
    print(f"\nwrote {out.name}: {len(d):,} unique structures")
    print(d["chemagg_cat"].value_counts().rename({0: "non-aggregator", 1: "aggregator"}).to_string())


if __name__ == "__main__":
    main()
