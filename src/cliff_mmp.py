"""Extract the structural transformation inside each cliff pair.

For every pair the maximum common substructure is found, the atoms it matches are
deleted from both molecules, and the residues are taken as the transformation that
converts the non-aggregator into the aggregator. Transformations are then ranked by
how often they appear, which names the moieties associated with crossing the
aggregation call rather than merely correlating with it across a library.
"""
import sys, collections, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')

def residue(mol, match):
    """SMILES of what is left after removing the MCS match atoms."""
    em = Chem.RWMol(mol)
    for i in sorted(match, reverse=True):
        em.RemoveAtom(i)
    frag = em.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        return None
    s = Chem.MolToSmiles(frag)
    return s if s else "[H]"

def transform(sa, sn, timeout=5):
    a = Chem.MolFromSmiles(sa); n = Chem.MolFromSmiles(sn)
    if a is None or n is None: return None
    try:
        r = rdFMCS.FindMCS([a, n], timeout=timeout, ringMatchesRingOnly=True,
                           completeRingsOnly=True, matchValences=False)
    except Exception:
        return None
    if r.canceled or r.numAtoms < 4: return None
    q = Chem.MolFromSmarts(r.smartsString)
    ma = a.GetSubstructMatch(q); mn = n.GetSubstructMatch(q)
    if not ma or not mn: return None
    ra = residue(a, ma); rn = residue(n, mn)
    if ra is None or rn is None or ra == rn: return None
    return (rn, ra)          # non-aggregator fragment -> aggregator fragment

def run(csv, label, limit=None):
    d = pd.read_csv(csv)
    if limit: d = d.head(limit)
    cnt = collections.Counter(); ok = 0
    for i, r in enumerate(d.itertuples()):
        t = transform(r.agg_smiles, r.non_smiles)
        if t: cnt[t] += 1; ok += 1
        if (i+1) % 400 == 0: print(f"   {i+1}/{len(d)} processed", flush=True)
    print(f"\n=== {label}: {ok}/{len(d)} pairs gave a clean transformation ===")
    print(f"  {'non-aggregator':>26}  ->  {'aggregator':<26} {'n':>5}")
    for (fn, fa), c in cnt.most_common(18):
        if c < 3: break
        print(f"  {fn[:26]:>26}  ->  {fa[:26]:<26} {c:>5}")
    return cnt

if __name__ == "__main__":
    for assay in ("cruzain", "ampc"):
        c = run(f"results/{assay}_full_real_tight.csv", assay.upper())
        pd.DataFrame([{"from": k[0], "to": k[1], "count": v} for k, v in c.most_common()]
                     ).to_csv(f"results/mmp_transforms_{assay}.csv", index=False)
        print(f"  wrote results/mmp_transforms_{assay}.csv ({len(c)} distinct transformations)")
