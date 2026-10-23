"""What distinguishes the aggregator from the non-aggregator inside a matched cliff pair?

The pairs are matched on calculated logP and molecular weight by construction, so any
descriptor that still separates them is a candidate driver of aggregation that
hydrophobicity alone does not capture. Kombo et al. 2024 reported Fsp3 and the
electrotopological state of hydroxyl groups as informative, which this tests directly
on paired compounds rather than across a whole library.
"""
import sys, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as rd
from scipy import stats
RDLogger.DisableLog('rdApp.*')

DESC = {
 "clogP":        Crippen.MolLogP,
 "MW":           Descriptors.MolWt,
 "TPSA":         Descriptors.TPSA,
 "Fsp3":         rd.CalcFractionCSP3,
 "aromatic rings": rd.CalcNumAromaticRings,
 "aliphatic rings": rd.CalcNumAliphaticRings,
 "HBD":          rd.CalcNumHBD,
 "HBA":          rd.CalcNumHBA,
 "rotatable":    rd.CalcNumRotatableBonds,
 "heavy atoms":  lambda m: m.GetNumHeavyAtoms(),
 "aromatic frac": lambda m: (sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
                             / max(m.GetNumHeavyAtoms(),1)),
 "MolMR":        Crippen.MolMR,
 "ring atoms":   lambda m: sum(1 for a in m.GetAtoms() if a.IsInRing()),
 "N atoms":      lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol()=="N"),
 "O atoms":      lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol()=="O"),
 "halogens":     lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() in ("F","Cl","Br","I")),
}

def profile(csv, label):
    d = pd.read_csv(csv)
    rows = []
    for _, r in d.iterrows():
        a = Chem.MolFromSmiles(r.agg_smiles); n = Chem.MolFromSmiles(r.non_smiles)
        if a is None or n is None: continue
        rec = {}
        ok = True
        for k, f in DESC.items():
            try: rec[k] = f(a) - f(n)
            except Exception: ok = False; break
        if ok: rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"\n=== {label}  ({len(df):,} cliff pairs) ===")
    print(f"  paired difference, aggregator minus non-aggregator")
    print(f"  {'descriptor':<16} {'mean':>8} {'median':>8} {'p (Wilcoxon)':>13}  {'effect':>7}")
    out=[]
    for k in DESC:
        v = df[k].values
        if np.allclose(v, 0): continue
        try: p = stats.wilcoxon(v, zero_method="zsplit").pvalue
        except Exception: p = np.nan
        # rank-biserial effect size
        nz = v[v != 0]
        eff = (np.sum(nz > 0) - np.sum(nz < 0)) / len(nz) if len(nz) else 0
        out.append((k, v.mean(), np.median(v), p, eff))
    out.sort(key=lambda x: -abs(x[4]))
    for k, m, md, p, eff in out:
        star = "***" if p < 1e-10 else ("**" if p < 1e-4 else ("*" if p < 0.05 else ""))
        print(f"  {k:<16} {m:>8.3f} {md:>8.2f} {p:>13.2e}  {eff:>+7.3f} {star}")
    return df

if __name__ == "__main__":
    c = profile("results/cruzain_full_real_tight.csv", "CRUZAIN")
    a = profile("results/ampc_full_real_tight.csv",   "AmpC")
    c.to_csv("results/cliff_physchem_cruzain.csv", index=False)
    a.to_csv("results/cliff_physchem_ampc.csv", index=False)
    print("\nEffect is the rank-biserial correlation over non-zero differences:")
    print("  +1 means the aggregator always has more, -1 means always fewer.")
