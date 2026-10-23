#!/usr/bin/env python
"""
Mine aggregation cliffs: pairs of molecules that are near-identical by 2D
fingerprint but carry opposite experimental aggregation labels.

Separates three populations, because they mean different things:
  duplicate  - identical canonical SMILES, opposite labels   -> pure label noise
  stereo     - same InChIKey skeleton block, different full key -> stereo/tautomer noise
  real       - different skeleton                            -> genuine chemistry cliff

Usage: python mine_cliffs.py <cleaned_csv> <out_prefix> [--threshold 0.85] [--sample N]
"""
import sys, time, argparse, collections
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen, inchi
RDLogger.DisableLog('rdApp.*')

ap = argparse.ArgumentParser()
ap.add_argument("cleaned_csv"); ap.add_argument("out_prefix")
ap.add_argument("--threshold", type=float, default=0.85)
ap.add_argument("--sample", type=int, default=0, help="0 = all aggregators")
ap.add_argument("--smiles-col", default="can_smiles")
a = ap.parse_args()

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

df = pd.read_csv(a.cleaned_csv)
log(f"loaded {len(df):,} rows from {a.cleaned_csv}")
df = df.dropna(subset=[a.smiles_col, "label"])
df["label"] = df["label"].astype(int)
log(f"  labels: {df['label'].value_counts().to_dict()}")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def featurise(sub, tag):
    recs = []
    for smi in sub[a.smiles_col].tolist():
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumHeavyAtoms() < 10:
            continue
        try:
            ik = inchi.MolToInchiKey(m)
        except Exception:
            ik = ""
        fp = gen.GetFingerprint(m)
        recs.append(dict(smiles=smi, fp=fp, pop=int(fp.GetNumOnBits()),
                         ik=ik, skel=ik.split("-")[0] if ik else "",
                         mw=Descriptors.MolWt(m), clogp=Crippen.MolLogP(m),
                         tpsa=Descriptors.TPSA(m), heavy=m.GetNumHeavyAtoms()))
    log(f"  featurised {tag}: {len(recs):,}")
    return recs

AGG = featurise(df[df.label == 1], "aggregators")
NON = featurise(df[df.label == 0], "non-aggregators")
if a.sample and a.sample < len(AGG):
    rng = np.random.default_rng(0)
    AGG = [AGG[i] for i in rng.choice(len(AGG), a.sample, replace=False)]
    log(f"  sampled {len(AGG):,} aggregators")

# popcount bucketing: Tanimoto(A,B) >= t requires min(|A|,|B|)/max(|A|,|B|) >= t
order = np.argsort([r["pop"] for r in NON])
NON = [NON[i] for i in order]
pops = np.array([r["pop"] for r in NON])
nfps = [r["fp"] for r in NON]
log(f"  non-agg popcount range {pops.min()}-{pops.max()}")

rows = []
for i, q in enumerate(AGG):
    lo = int(np.searchsorted(pops, q["pop"] * a.threshold, "left"))
    hi = int(np.searchsorted(pops, q["pop"] / a.threshold, "right"))
    if hi <= lo:
        continue
    sims = DataStructs.BulkTanimotoSimilarity(q["fp"], nfps[lo:hi])
    for j, s in enumerate(sims):
        if s < a.threshold:
            continue
        n = NON[lo + j]
        if q["skel"] and n["skel"] and q["ik"] == n["ik"]:
            kind = "duplicate"
        elif q["skel"] and n["skel"] and q["skel"] == n["skel"]:
            kind = "stereo"
        else:
            kind = "real"
        rows.append(dict(kind=kind, tanimoto=round(s, 4),
                         agg_smiles=q["smiles"], non_smiles=n["smiles"],
                         agg_ik=q["ik"], non_ik=n["ik"],
                         d_clogp=round(abs(q["clogp"] - n["clogp"]), 3),
                         d_mw=round(abs(q["mw"] - n["mw"]), 2),
                         d_tpsa=round(abs(q["tpsa"] - n["tpsa"]), 2),
                         d_heavy=abs(q["heavy"] - n["heavy"]),
                         agg_heavy=q["heavy"], non_heavy=n["heavy"],
                         agg_clogp=round(q["clogp"], 3), non_clogp=round(n["clogp"], 3)))
    if (i + 1) % 2000 == 0:
        log(f"  {i+1:,}/{len(AGG):,} queries, {len(rows):,} pairs")

out = pd.DataFrame(rows)
log(f"TOTAL PAIRS at Tanimoto>={a.threshold}: {len(out):,}")
if len(out):
    out.to_csv(f"{a.out_prefix}_all_pairs.csv", index=False)
    print("\n--- pair populations ---")
    print(out["kind"].value_counts().to_string())
    tight = out[(out.kind == "real") & (out.d_clogp < 0.5) & (out.d_mw < 30)]
    tight = tight.sort_values("tanimoto", ascending=False)
    tight.to_csv(f"{a.out_prefix}_real_tight.csv", index=False)
    print(f"\nREAL TIGHT CLIFFS (kind=real, dclogP<0.5, dMW<30): {len(tight):,}")
    print(f"  unique aggregators involved:     {tight.agg_ik.nunique():,}")
    print(f"  unique non-aggregators involved: {tight.non_ik.nunique():,}")
    print(f"  both partners <=25 heavy atoms:  {len(tight[(tight.agg_heavy<=25)&(tight.non_heavy<=25)]):,}  (Auto-Martini_M3 ceiling)")
    print(f"\nwrote {a.out_prefix}_all_pairs.csv and {a.out_prefix}_real_tight.csv")
