"""Build the study compound set from the MEGAN external validation data.

Source: DavisGroup/MEGAN-aggregation-data, external_validation_dataset.csv
Labels are DLS-backed, from two Shoichet laboratory papers:
  REF 1  O'Donnell et al. J Med Chem 2021, 64, 17530   doi:10.1021/acs.jmedchem.1c01547
  REF 2  Glenn et al.     J Med Chem 2024, 67, 10263   doi:10.1021/acs.jmedchem.4c00597
"""
import csv
import pathlib
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "compounds" / "megan_external_raw.csv"
OUT = ROOT / "data" / "compounds" / "external_58.csv"

REFS = {
    "1": "O'Donnell 2021 JMedChem 10.1021/acs.jmedchem.1c01547",
    "2": "Glenn 2024 JMedChem 10.1021/acs.jmedchem.4c00597",
}


def main() -> None:
    rows = list(csv.reader(RAW.open(encoding="utf-8-sig")))
    hdr = next(i for i, r in enumerate(rows) if r and r[0].strip() == "REF")

    out, seen = [], {}
    for r in rows[hdr + 1:]:
        if len(r) < 4 or not r[2].strip():
            continue
        name, smi, label, ref = r[1].strip(), r[2].strip(), r[3].strip(), r[0].strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  UNPARSEABLE  {name}: {smi}")
            continue
        can = Chem.MolToSmiles(mol)
        ik = Chem.MolToInchiKey(mol)
        if ik in seen:
            print(f"  DUPLICATE    {name} == {seen[ik]}")
            continue
        seen[ik] = name
        out.append({
            "name": name,
            "smiles": can,
            "smiles_original": smi,
            "inchikey": ik,
            "label": int(label),
            "class": "aggregator" if label == "1" else "non-aggregator",
            "source_ref": REFS.get(ref, ref),
            "set": "megan_external",
        })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    n_agg = sum(r["label"] == 1 for r in out)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(out)} compounds: {n_agg} aggregators, {len(out) - n_agg} non-aggregators")
    for ref in sorted({r['source_ref'] for r in out}):
        print(f"  {sum(r['source_ref'] == ref for r in out):>3} from {ref}")
    cq = [r for r in out if "clioquinol" in r["name"].lower()]
    for r in cq:
        print(f"  ANCHOR: {r['name']:12s} label={r['label']} {r['smiles']}")


if __name__ == "__main__":
    main()
