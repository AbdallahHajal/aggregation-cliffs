"""exmol/MMACE counterfactuals for miconazole against the BAD filter.

exmol's sample_space calls the scoring function ONCE with the whole candidate list,
so this is a single Docker batch regardless of num_samples. Default is 3000, which at
ChemAGG's ~20 s/query would be unusable; here BAD is local so a bounded run is possible
and lets us compare exmol's generator against CReM's on the same molecule.
"""
import pathlib, subprocess
import pandas as pd
import exmol
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
MICONAZOLE = "Clc1ccc(COC(Cn2ccnc2)c2ccc(Cl)cc2Cl)c(Cl)c1"
CALLS = {"n": 0, "mols": 0}


def bad_batch(smiles_list):
    inp, out = ROOT / "data/work/_ex_in.csv", ROOT / "data/work/_ex_out.csv"
    pd.DataFrame({"smiles": smiles_list}).to_csv(inp, index=False)
    subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64",
                    "-v", f"{ROOT}/vendor/bad_filter:/app", "-v", f"{ROOT}/data:/data",
                    "badfilter:1.1", "python", "/app/bad_predict.py",
                    "/data/work/_ex_in.csv", "/data/work/_ex_out.csv"],
                   capture_output=True, check=True)
    return pd.read_csv(out)["bad_prob"].tolist()


def f(smiles_list):
    """exmol scorer. Returns the DISCRETE verdict code, because cf_explain tests yhat inequality."""
    CALLS["n"] += 1; CALLS["mols"] += len(smiles_list)
    probs = bad_batch(list(smiles_list))
    # 0 non-aggregator, 1 ambiguous, 2 aggregator - the tool's own published bands
    return [2 if p > 0.6 else (0 if p < 0.3 else 1) for p in probs]


def main():
    print("running exmol sample_space (num_samples=200, preset='medium')...")
    space = exmol.sample_space(MICONAZOLE, f, batched=True, num_samples=200, preset="medium")
    print(f"   exmol generated {len(space)} molecules")
    print(f"   scoring function called {CALLS['n']} time(s) on {CALLS['mols']} molecules")

    base = space[0]
    print(f"   miconazole yhat = {base.yhat}  (0 non-agg, 1 ambiguous, 2 aggregator)")
    flipped = [e for e in space[1:] if e.yhat != base.yhat]
    print(f"   molecules with a DIFFERENT verdict: {len(flipped)} / {len(space)-1}")
    agg = [e for e in space[1:] if e.yhat == 2]
    print(f"   molecules exmol pushed to Aggregator: {len(agg)}")

    cfs = exmol.cf_explain(space, nmols=4)
    rows = []
    for e in cfs:
        m = Chem.MolFromSmiles(e.smiles)
        rows.append({"smiles": e.smiles, "yhat": e.yhat, "similarity": round(e.similarity, 3),
                     "is_origin": e.is_origin,
                     "heavy": m.GetNumHeavyAtoms() if m else None})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/filters/exmol_miconazole.csv", index=False)
    print("\ncf_explain counterfactuals:")
    print(d.to_string(index=False))

    ys = [e.yhat for e in space]
    print(f"\nverdict distribution over exmol's {len(space)} molecules: "
          f"non-agg {ys.count(0)}, ambiguous {ys.count(1)}, aggregator {ys.count(2)}")


if __name__ == "__main__":
    main()
