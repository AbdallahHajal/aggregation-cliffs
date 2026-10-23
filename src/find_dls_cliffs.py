"""Find matched pairs with OPPOSITE DLS labels among the experimentally measured set.

These are real aggregation cliffs: both members measured, same technique, mostly the
same laboratory and protocol. They are the experimental yardstick the filters are then
asked to reproduce. Nothing here is generated, so accuracy claims are legitimate.

Structures are standardised first (largest fragment, uncharged, canonical tautomer),
which the earlier scripts did not do. Multi-fragment metal complexes are excluded and
the exclusion is recorded rather than silent.
"""
import itertools
import pathlib

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdFingerprintGenerator, rdFMCS
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = pathlib.Path(__file__).resolve().parents[1]
MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_lfc, _unc, _taut = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger(), rdMolStandardize.TautomerEnumerator()


def standardise(smiles):
    """Return (mol, note) or (None, reason)."""
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None, "unparseable"
    nfrag = len(Chem.GetMolFrags(m))
    metals = [a.GetSymbol() for a in m.GetAtoms()
              if a.GetAtomicNum() not in (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53)]
    if metals:
        return None, f"metal/unsupported atoms {sorted(set(metals))}"
    note = []
    if nfrag > 1:
        m = _lfc.choose(m); note.append(f"largest of {nfrag} fragments")
    if Chem.GetFormalCharge(m) != 0:
        before = Chem.GetFormalCharge(m)
        m = _unc.uncharge(m)
        if Chem.GetFormalCharge(m) != before:
            note.append(f"uncharged {before:+d}->{Chem.GetFormalCharge(m):+d}")
    try:
        m = _taut.Canonicalize(m)
    except Exception:
        note.append("tautomer canonicalisation failed")
    return m, "; ".join(note) or "clean"


def mcs_delta(a, b, timeout=10):
    """Heavy atoms outside the maximum common substructure, per side."""
    try:
        r = rdFMCS.FindMCS([a, b], timeout=timeout, ringMatchesRingOnly=True,
                           completeRingsOnly=False)
    except Exception:
        return None, None, None
    if r.canceled or r.numAtoms == 0:
        return None, None, None
    n = r.numAtoms
    return a.GetNumHeavyAtoms() - n, b.GetNumHeavyAtoms() - n, r.smartsString


def main() -> None:
    ext = pd.read_csv(ROOT / "data/compounds/external_58.csv")[["name", "smiles", "label", "class", "source_ref"]]
    new = pd.read_csv(ROOT / "data/dls/new_aggregators_smiles.csv")[["name", "smiles"]]
    new["label"], new["class"], new["source_ref"] = 1, "aggregator", "ODonnell2021 Table1/3"
    allc = pd.concat([ext, new]).drop_duplicates(subset="name").reset_index(drop=True)

    mols, excl = [], []
    for _, r in allc.iterrows():
        m, note = standardise(r["smiles"])
        if m is None:
            excl.append({"name": r["name"], "reason": note}); continue
        mols.append({"name": r["name"].strip(), "mol": m, "label": int(r["label"]),
                     "smiles": Chem.MolToSmiles(m), "std_note": note,
                     "source": r["source_ref"], "fp": MFPGEN.GetFingerprint(m)})

    print(f"{len(allc)} compounds in, {len(mols)} standardised, {len(excl)} excluded")
    for e in excl:
        print(f"   EXCLUDED {e['name']}: {e['reason']}")
    pd.DataFrame(excl).to_csv(ROOT / "results/filters/dls_excluded.csv", index=False)
    nagg = sum(m["label"] == 1 for m in mols)
    print(f"   {nagg} aggregators / {len(mols)-nagg} non-aggregators\n")

    pairs = []
    for a, b in itertools.combinations(mols, 2):
        if a["label"] == b["label"]:
            continue
        tc = DataStructs.TanimotoSimilarity(a["fp"], b["fp"])
        if tc < 0.40:
            continue
        agg, non = (a, b) if a["label"] == 1 else (b, a)
        da, dn, sm = mcs_delta(agg["mol"], non["mol"])
        pairs.append({
            "aggregator": agg["name"], "non_aggregator": non["name"],
            "tanimoto": round(tc, 3),
            "agg_smiles": agg["smiles"], "non_smiles": non["smiles"],
            "mcs_delta_agg": da, "mcs_delta_non": dn,
            "transform_size": (da + dn) if (da is not None and dn is not None) else None,
            "d_heavy": agg["mol"].GetNumHeavyAtoms() - non["mol"].GetNumHeavyAtoms(),
            "d_MW": round(Descriptors.MolWt(agg["mol"]) - Descriptors.MolWt(non["mol"]), 2),
            "d_clogP": round(Crippen.MolLogP(agg["mol"]) - Crippen.MolLogP(non["mol"]), 2),
            "d_TPSA": round(Descriptors.TPSA(agg["mol"]) - Descriptors.TPSA(non["mol"]), 1),
            "d_HBD": Descriptors.NumHDonors(agg["mol"]) - Descriptors.NumHDonors(non["mol"]),
            "agg_source": agg["source"], "non_source": non["source"],
        })

    p = pd.DataFrame(pairs).sort_values("tanimoto", ascending=False)
    p.to_csv(ROOT / "results/filters/dls_cliffs.csv", index=False)
    print(f"{len(p)} opposite-label pairs at Tanimoto >= 0.40")
    print(f"   >= 0.60 : {(p['tanimoto']>=0.60).sum()}")
    print(f"   >= 0.70 : {(p['tanimoto']>=0.70).sum()}")
    print(f"   transform_size <= 4 heavy atoms : {(p['transform_size']<=4).sum()}\n")
    print("top 15 by similarity:")
    for _, r in p.head(15).iterrows():
        print(f"   Tc {r['tanimoto']:.3f}  {r['aggregator'][:20]:22s} vs {r['non_aggregator'][:20]:22s} "
              f"transform={r['transform_size']} dMW={r['d_MW']:+8.2f} dclogP={r['d_clogP']:+5.2f}")


if __name__ == "__main__":
    main()
