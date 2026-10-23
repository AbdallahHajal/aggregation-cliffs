"""Build the AmpC beta-lactamase aggregator classification using the BAD Molecule
Filter rule, then audit measurement coverage per class exactly as done for Cruzain.

BAD filter rule (Abou Hajal 2024, JCIM 64:4991, Methods):
  Nonaggregator = no logAC50 in BOTH the presence and absence of detergent
  Aggregator    = Hill slope > 1.5 AND logAC50 without detergent but NOT with detergent
"""
import pandas as pd, numpy as np

D="data/AmpC/"
KEEP=["PUBCHEM_CID","PUBCHEM_EXT_DATASOURCE_SMILES","Fit_LogAC50","Fit_HillSlope"]

def load(aid):
    d=pd.read_csv(D+f"AID_{aid}_datatable_all.csv", low_memory=False)
    # drop PubChem metadata rows (RESULT_TYPE / RESULT_DESCR / RESULT_UNIT ...)
    d=d[pd.to_numeric(d["PUBCHEM_CID"], errors="coerce").notna()].copy()
    d["PUBCHEM_CID"]=pd.to_numeric(d["PUBCHEM_CID"])
    for c in ["Fit_LogAC50","Fit_HillSlope"]:
        d[c]=pd.to_numeric(d[c], errors="coerce")
    return d[KEEP]

nod=load(485341).rename(columns={"Fit_LogAC50":"LogAC50_no_detergent","Fit_HillSlope":"HillSlope_no_detergent"})
det=load(485294).rename(columns={"Fit_LogAC50":"LogAC50_with_detergent","Fit_HillSlope":"HillSlope_with_detergent"})
print(f"no-detergent AID 485341: {len(nod):,} rows, {nod.PUBCHEM_CID.nunique():,} CIDs")
print(f"with-detergent AID 485294: {len(det):,} rows, {det.PUBCHEM_CID.nunique():,} CIDs")

nod=nod.drop_duplicates("PUBCHEM_CID"); det=det.drop_duplicates("PUBCHEM_CID")
m=nod.merge(det.drop(columns=["PUBCHEM_EXT_DATASOURCE_SMILES"]), on="PUBCHEM_CID", how="outer")
print(f"merged: {len(m):,} unique compounds\n")

has_nod = m.LogAC50_no_detergent.notna()
has_det = m.LogAC50_with_detergent.notna()
steep   = m.HillSlope_no_detergent > 1.5

m["Aggregation_Class"]=np.where(
    (~has_nod) & (~has_det), "Nonaggregator",
    np.where(has_nod & (~has_det) & steep, "Aggregator", "Other"))

print("=== AmpC classification (BAD filter rule) ===")
print(m.Aggregation_Class.value_counts().to_string())

cols=["LogAC50_no_detergent","HillSlope_no_detergent","LogAC50_with_detergent","HillSlope_with_detergent"]
m["n_measured"]=m[cols].notna().sum(axis=1)
print("\n=== MEASUREMENT COVERAGE AUDIT (the replication test) ===")
t=m.groupby("Aggregation_Class").agg(
    n=("Aggregation_Class","size"),
    any_measurement=("n_measured", lambda s:int((s>0).sum())))
t["pct_with_any_measurement"]=(100*t.any_measurement/t.n).round(3)
print(t.to_string())

agg=m[m.Aggregation_Class=="Aggregator"]
print(f"\nAggregator Hill slope: median {agg.HillSlope_no_detergent.median():.2f}, min {agg.HillSlope_no_detergent.min():.2f}")
print(f"Aggregator logAC50: median {agg.LogAC50_no_detergent.median():.2f} "
      f"(= {10**agg.LogAC50_no_detergent.median()*1e6:.1f} uM)")

out=m[m.Aggregation_Class.isin(["Aggregator","Nonaggregator"])][
    ["PUBCHEM_CID","PUBCHEM_EXT_DATASOURCE_SMILES","Aggregation_Class"]+cols]
out.to_csv("data/AmpC/ampc_aggregation_classification.csv", index=False)
print(f"\nwrote data/AmpC/ampc_aggregation_classification.csv ({len(out):,} rows)")
