"""Do the flip distances survive when only drug-like analogues are counted?

Also quantifies how far each lineage drifted from its parent drug, since the escalation
is cumulative: a round-5 molecule carries five accumulated modifications, so later flips
are less recognisably related to the starting compound.
"""
import glob, json, os, pathlib, sys
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, QED, RDConfig, rdFingerprintGenerator
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

RDLogger.DisableLog("rdApp.*")
ROOT = pathlib.Path(__file__).resolve().parents[1]
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def main():
    st = json.loads((ROOT / "results/beam/state.json").read_text())
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/beam/rung*_scored.csv"))):
        r = int(f.split("rung")[1].split("_")[0])
        d = pd.read_csv(f); d["round"] = r
        rows.append(d)
    a = pd.concat(rows, ignore_index=True)

    recs = []
    for _, r in a.iterrows():
        m = Chem.MolFromSmiles(r["smiles"])
        root = Chem.MolFromSmiles(st["lineages"][r["lineage"]]["root"])
        recs.append({
            "qed": round(QED.qed(m), 3),
            "sa": round(sascorer.calculateScore(m), 2),
            "tc_to_parent_drug": round(DataStructs.TanimotoSimilarity(
                FP.GetFingerprint(root), FP.GetFingerprint(m)), 3),
            "total_d_heavy": m.GetNumHeavyAtoms() - root.GetNumHeavyAtoms(),
        })
    a = pd.concat([a, pd.DataFrame(recs)], axis=1)
    a.to_csv(ROOT / "results/beam/all_analogues_annotated.csv", index=False)

    print("=" * 70)
    print("DRIFT FROM THE PARENT DRUG, BY ROUND")
    print("=" * 70)
    g = a.groupby("round").agg(
        n=("smiles", "size"),
        med_tc=("tc_to_parent_drug", "median"),
        med_dheavy=("total_d_heavy", "median"),
        med_qed=("qed", "median"),
        med_sa=("sa", "median"))
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n  Tanimoto to the parent drug falls as rounds accumulate: the later a flip,")
    print("  the less the molecule resembles the compound it started from.")

    DRUGLIKE = (a["qed"] >= 0.5) & (a["sa"] <= 4.5)
    print("\n" + "=" * 70)
    print("SENSITIVITY: restricting to drug-like analogues (QED >= 0.5 and SA <= 4.5)")
    print("=" * 70)
    print(f"  {DRUGLIKE.sum()}/{len(a)} analogues are drug-like ({DRUGLIKE.mean():.1%})")

    print(f"\n{'lineage':20s} {'all analogues':>14s} {'drug-like only':>16s}")
    kept = 0
    for nm, L in sorted(st["lineages"].items()):
        fl = L.get("flip")
        allmin = fl["d_heavy_total"] if fl else None
        sub = a[(a["lineage"] == nm) & (a["chemagg_cat"] == 1) & DRUGLIKE]
        dmin = int(sub["total_d_heavy"].min()) if len(sub) else None
        if dmin is not None:
            kept += 1
        s1 = f"+{allmin}" if allmin is not None else "no flip"
        s2 = f"+{dmin}" if dmin is not None else "no flip"
        flag = "" if dmin == allmin else "   <- differs"
        print(f"{nm[:20]:20s} {s1:>14s} {s2:>16s}{flag}")
    print(f"\n  {kept}/12 lineages still flip when only drug-like analogues count.")
    both = a[(a["chemagg_cat"] == 1)]
    print(f"  median flip distance, all: +{both.groupby('lineage')['total_d_heavy'].min().median():.1f}")
    bd = a[(a["chemagg_cat"] == 1) & DRUGLIKE]
    if len(bd):
        print(f"  median flip distance, drug-like only: +{bd.groupby('lineage')['total_d_heavy'].min().median():.1f}")


if __name__ == "__main__":
    main()
