#!/usr/bin/env python
"""Reproduce data/AmpC/ampc_cleaned.csv from the two PubChem AID tables.

Written 2026-09-29 to close a reproducibility gap: the file existed with a `std_smiles`
column but no script in src/ produced it, so the preprocessing could not be audited.

What the existing file actually contains, verified before this was written:

  * labels from the BAD Molecule Filter rule, as in build_ampc.py
        Nonaggregator = no logAC50 in EITHER condition
        Aggregator    = Hill slope > 1.5 AND logAC50 without detergent but not with it
  * can_smiles  = RDKit canonical SMILES
  * std_smiles  = IDENTICAL to can_smiles in 276,232 of 276,232 rows

So AmpC received canonicalization only. It did NOT receive the MolVS treatment that
Cruzain got via the Article 19 pipeline (`_outputs_molvs_baselines/`). Salt forms, charge
states and tautomers were never normalized here. That asymmetry between the two data sets
is now on the record; it is NOT silently corrected, because re-standardizing AmpC would
change the mined pairs and invalidate every downstream number in the manuscript.

--report compares this reconstruction against the existing file instead of overwriting.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data/AmpC"
KEEP = ["PUBCHEM_CID", "PUBCHEM_EXT_DATASOURCE_SMILES", "Fit_LogAC50", "Fit_HillSlope"]

ap = argparse.ArgumentParser()
ap.add_argument("--report", action="store_true",
                help="compare against the existing file rather than writing")
a = ap.parse_args()


def load(aid):
    d = pd.read_csv(D / f"AID_{aid}_datatable_all.csv", low_memory=False)
    d = d[pd.to_numeric(d["PUBCHEM_CID"], errors="coerce").notna()].copy()
    d["PUBCHEM_CID"] = pd.to_numeric(d["PUBCHEM_CID"])
    for c in ("Fit_LogAC50", "Fit_HillSlope"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[KEEP].drop_duplicates("PUBCHEM_CID")


nod = load(485341).rename(columns={"Fit_LogAC50": "LogAC50_no_detergent",
                                   "Fit_HillSlope": "HillSlope_no_detergent"})
det = load(485294).rename(columns={"Fit_LogAC50": "LogAC50_with_detergent",
                                   "Fit_HillSlope": "HillSlope_with_detergent"})
m = nod.merge(det.drop(columns=["PUBCHEM_EXT_DATASOURCE_SMILES"]),
              on="PUBCHEM_CID", how="outer")
print(f"merged {len(m):,} unique CIDs")

has_nod = m.LogAC50_no_detergent.notna()
has_det = m.LogAC50_with_detergent.notna()
steep = m.HillSlope_no_detergent > 1.5

m["Aggregation_Class"] = np.where(
    (~has_nod) & (~has_det), "Nonaggregator",
    np.where(has_nod & (~has_det) & steep, "Aggregator", "Other"))
print(m.Aggregation_Class.value_counts().to_string())

m = m[m.Aggregation_Class != "Other"].copy()
m["label"] = (m.Aggregation_Class == "Aggregator").astype(int)
m = m.rename(columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "SMILES"})
m = m.dropna(subset=["SMILES"])


def canon(s):
    mol = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(mol) if mol else None


m["can_smiles"] = [canon(s) for s in m["SMILES"]]
m = m.dropna(subset=["can_smiles"])
# reproduced faithfully: the original carries std_smiles as a copy of can_smiles
m["std_smiles"] = m["can_smiles"]

out = m[["SMILES", "label", "can_smiles", "std_smiles"]].reset_index(drop=True)
print(f"\nreconstructed {len(out):,} rows; labels {out.label.value_counts().to_dict()}")

existing = D / "ampc_cleaned.csv"
if a.report and existing.exists():
    old = pd.read_csv(existing)
    print(f"existing file:      {len(old):,} rows; labels {old.label.value_counts().to_dict()}")
    so, sn = set(old.can_smiles.dropna()), set(out.can_smiles.dropna())
    print(f"canonical SMILES in both:      {len(so & sn):,}")
    print(f"only in existing file:         {len(so - sn):,}")
    print(f"only in this reconstruction:   {len(sn - so):,}")
    print(f"\nstd_smiles == can_smiles in the existing file: "
          f"{(old.can_smiles == old.std_smiles).mean()*100:.2f}% of rows")
    print("-> AmpC had canonicalization only, no MolVS. Cruzain had MolVS. Declare this.")
else:
    out.to_csv(existing, index=False)
    print(f"wrote {existing}")
