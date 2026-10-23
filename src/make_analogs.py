"""Generate single-heavy-atom analogs with CReM.

Parameters are fixed in PREANALYSIS.md Amendment 1, written before any analog existed:
mutate_mol with max_size=1, min_size=0, radius=3, against chembl33_sa2_f5.db. At most 10
analogs per parent, taken as the first 10 after sorting by canonical SMILES so the
selection is structural and independent of any filter's output.
"""
import pathlib
import sys

import pandas as pd
from crem.crem import mutate_mol
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "vendor" / "crem" / "chembl33_sa2_f5.db"
MAX_PER_PARENT = 10


def analogs_for(smiles: str, name: str) -> list[dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    parent_ik = Chem.MolToInchiKey(mol)
    try:
        raw = set(mutate_mol(mol, db_name=str(DB), max_size=1, min_size=0, radius=3))
    except Exception as e:
        print(f"  {name}: CReM failed ({type(e).__name__}: {e})", flush=True)
        return []

    keep, seen = [], set()
    for s in sorted(raw):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        ik = Chem.MolToInchiKey(m)
        if ik == parent_ik or ik in seen:
            continue
        seen.add(ik)
        keep.append({
            "parent": name,
            "parent_smiles": smiles,
            "smiles": Chem.MolToSmiles(m),
            "inchikey": ik,
        })
    n_total = len(keep)
    for i, r in enumerate(keep[:MAX_PER_PARENT]):
        r["analog_idx"] = i
        r["n_analogs_available"] = n_total
    return keep[:MAX_PER_PARENT]


def main() -> None:
    if not DB.exists():
        sys.exit(f"fragment database missing: {DB}")
    parents = pd.read_csv(ROOT / "data/compounds/external_58.csv")[["name", "smiles", "label", "class"]]
    anchor = pd.read_csv(ROOT / "data/compounds/anchor_pair.csv")[["name", "smiles", "label"]]
    anchor["class"] = anchor["label"].map({1: "aggregator", 0: "non-aggregator"})
    allp = pd.concat([parents, anchor]).drop_duplicates(subset="name").reset_index(drop=True)

    rows, counts = [], []
    for i, r in allp.iterrows():
        a = analogs_for(r["smiles"], r["name"])
        for x in a:
            x["parent_label"] = r["label"]
            x["parent_class"] = r["class"]
        rows.extend(a)
        counts.append({"parent": r["name"], "n_kept": len(a),
                       "n_available": a[0]["n_analogs_available"] if a else 0})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(allp)} parents, {len(rows)} analogs so far", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/compounds/analogs.csv", index=False)
    pd.DataFrame(counts).to_csv(ROOT / "results/filters/analog_counts.csv", index=False)
    print(f"\nwrote {len(out)} analogs from {out['parent'].nunique()} parents")
    print(f"  parents yielding 0 analogs: {sum(c['n_kept']==0 for c in counts)}")
    print(f"  median analogs available per parent: {pd.DataFrame(counts)['n_available'].median():.0f}")


if __name__ == "__main__":
    main()
