"""Greedy escalation: modify a molecule step by step until a filter's verdict flips.

At each step CReM proposes analogs of the CURRENT molecule (not the original), every
analog is scored by the filter in one batch, and the highest-scoring analog becomes the
next step. The path is recorded with the measured heavy-atom change at each step.

Generation costs no filter queries; only the scoring does.
"""
import json, pathlib, subprocess, sys
import pandas as pd
from crem.crem import grow_mol, mutate_mol
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen
RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = str(ROOT / "vendor/crem/chembl33_sa2_f5.db")
SEED = 20260928


def bad_score(smiles_list):
    """Score a batch with the BAD filter inside its container."""
    inp = ROOT / "data/work/_esc_in.csv"
    out = ROOT / "data/work/_esc_out.csv"
    pd.DataFrame({"smiles": smiles_list}).to_csv(inp, index=False)
    subprocess.run([
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "-v", f"{ROOT}/vendor/bad_filter:/app", "-v", f"{ROOT}/data:/data",
        "badfilter:1.1", "python", "/app/bad_predict.py",
        "/data/work/_esc_in.csv", "/data/work/_esc_out.csv",
    ], capture_output=True, check=True)
    d = pd.read_csv(out)
    return list(zip(d["bad_prob"], d["bad_call"]))


def propose(mol, rung):
    """CReM analogs of the current molecule at this rung."""
    out = set()
    try:
        out |= set(grow_mol(mol, db_name=DB, min_atoms=1, max_atoms=rung, radius=3,
                            max_replacements=60, seed=SEED))
    except Exception as e:
        print(f"    grow_mol failed: {e}")
    if rung >= 2:
        try:
            out |= set(mutate_mol(mol, db_name=DB, max_size=rung, min_size=1, radius=3,
                                  min_inc=0, max_inc=rung, max_replacements=60,
                                  seed=SEED))
        except Exception as e:
            print(f"    mutate_mol failed: {e}")
    keep = []
    seen = {Chem.MolToInchiKey(mol)}
    for s in sorted(out):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        ik = Chem.MolToInchiKey(m)
        if ik in seen:
            continue
        seen.add(ik)
        keep.append(Chem.MolToSmiles(m))
    return keep


def props(m):
    return {"heavy": m.GetNumHeavyAtoms(), "MW": round(Descriptors.MolWt(m), 1),
            "clogP": round(Crippen.MolLogP(m), 2), "TPSA": round(Descriptors.TPSA(m), 1),
            "HBD": Descriptors.NumHDonors(m)}


def main(name, smiles, target="Aggregator", max_steps=5):
    cur = Chem.MolFromSmiles(smiles)
    cur_smi = Chem.MolToSmiles(cur)
    p0, c0 = bad_score([cur_smi])[0]
    print(f"\nSTART  {name}: p={p0:.3f} -> {c0}")
    print(f"       {props(cur)}")
    path = [{"step": 0, "smiles": cur_smi, "bad_prob": p0, "bad_call": c0,
             "n_scored": 1, **props(cur)}]

    for step in range(1, max_steps + 1):
        cand = propose(cur, rung=step)
        if not cand:
            print(f"step {step}: no analogs proposed, stopping"); break
        scored = bad_score(cand)
        best_i = max(range(len(cand)), key=lambda i: scored[i][0])
        bp, bc = scored[best_i]
        bm = Chem.MolFromSmiles(cand[best_i])
        prev = path[-1]["bad_prob"]
        print(f"step {step}: scored {len(cand):3d} analogs -> best p={bp:.3f} ({bc})  "
              f"delta {bp-prev:+.3f}  dHeavy {bm.GetNumHeavyAtoms()-cur.GetNumHeavyAtoms():+d}")
        path.append({"step": step, "smiles": cand[best_i], "bad_prob": bp, "bad_call": bc,
                     "n_scored": len(cand), **props(bm)})
        cur, cur_smi = bm, cand[best_i]
        if bc == target:
            print(f"\n*** FLIPPED to {target} at step {step} ***"); break
    else:
        print(f"\nCENSORED: did not reach {target} within {max_steps} steps")

    pd.DataFrame(path).to_csv(ROOT / f"results/filters/escalate_{name}.csv", index=False)
    print(f"wrote results/filters/escalate_{name}.csv")
    return path


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
