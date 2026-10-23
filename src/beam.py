"""Rung-synchronous beam escalation against ChemAGG.

ChemAGG is driven through a browser, so scoring is the expensive step. This driver keeps
every lineage in lockstep: at rung k it generates analogues for all active beam members
of all parents, pools them into ONE SDF, and stops. Scores are ingested from the result
CSV, the beam advances, and the next rung is emitted. Five rungs is five uploads.

    python src/beam.py generate      # writes data/beam/rung<k>.sdf, then stops
    python src/beam.py ingest <csv>  # reads scores, advances the beam, reports

State lives in results/beam/state.json so the run is resumable across interruptions.

Modifications are REPLACEMENTS (mutate_mol) with grow_mol added, because a cliff is a
matched pair. Net heavy-atom change is measured per analogue and never inferred from the
rung number. A Crippen clogP threshold classifier is scored locally on the identical
analogues as the null, at zero query cost.
"""
import json, pathlib, random, sys
import pandas as pd
from crem.crem import grow_mol, mutate_mol
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = str(ROOT / "vendor/crem/chembl33_sa2_f5.db")
STATE = ROOT / "results/beam/state.json"
SEED, BEAM, MAX_RUNG, PER_MEMBER = 20260928, 3, 5, 10
GEN_POOL = 60            # generate this many, then draw PER_MEMBER deterministically
CLOGP_CUT = 3.5          # the null: clogP above this = "aggregator"
_lfc, _unc = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()


def std(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    if len(Chem.GetMolFrags(m)) > 1:
        m = _lfc.choose(m)
    if Chem.GetFormalCharge(m) != 0:
        m = _unc.uncharge(m)
    return m


def descriptors(m):
    return {"heavy": m.GetNumHeavyAtoms(), "MW": round(Descriptors.MolWt(m), 1),
            "clogP": round(Crippen.MolLogP(m), 2), "TPSA": round(Descriptors.TPSA(m), 1),
            "HBD": Descriptors.NumHDonors(m), "QED": round(QED.qed(m), 3)}


def init_state():
    t = pd.read_csv(ROOT / "results/filters/TRUE_compounds.csv")
    ext = pd.read_csv(ROOT / "data/compounds/external_58.csv").set_index("name")
    parents = t[(t["tier"] == "B: multi-source agree") & (t["class"] == "non-aggregator")]
    st = {"rung": 0, "beam_width": BEAM, "seed": SEED, "lineages": {}}
    for _, r in parents.iterrows():
        nm = r["name"].strip()
        if nm not in ext.index:
            print(f"   skip {nm}: no SMILES"); continue
        m = std(ext.loc[nm]["smiles"])
        if m is None:
            print(f"   skip {nm}: unparseable"); continue
        smi = Chem.MolToSmiles(m)
        st["lineages"][nm] = {
            "status": "active", "root": smi, "root_heavy": m.GetNumHeavyAtoms(),
            "beam": [{"smiles": smi, "rung": 0, "score": None, "parent_of": None}],
            "history": [],
        }
    print(f"initialised {len(st['lineages'])} lineages "
          f"(Tier B non-aggregators), beam width {BEAM}, ceiling rung {MAX_RUNG}")
    return st


def propose(smi, rung):
    m = Chem.MolFromSmiles(smi)
    out = set()
    try:
        out |= set(mutate_mol(m, db_name=DB, radius=3, min_size=1, max_size=rung,
                              min_inc=0, max_inc=rung, max_replacements=GEN_POOL,
                              seed=SEED))
    except Exception:
        pass
    try:
        out |= set(grow_mol(m, db_name=DB, radius=3, min_atoms=1, max_atoms=rung,
                            max_replacements=GEN_POOL, seed=SEED))
    except Exception:
        pass
    keep, seen = [], {Chem.MolToInchiKey(m)}
    for s in sorted(out):
        a = Chem.MolFromSmiles(s)
        if a is None:
            continue
        ik = Chem.MolToInchiKey(a)
        if ik in seen:
            continue
        seen.add(ik)
        keep.append(Chem.MolToSmiles(a))
    # PREANALYSIS Amendment 2: a fixed count per beam member per rung, drawn at random
    # under a seed rather than by alphabetical order, which was not outcome-neutral.
    if len(keep) > PER_MEMBER:
        random.Random(f"{SEED}|{smi}|{rung}").shuffle(keep)
        keep = sorted(keep[:PER_MEMBER])
    return keep


def cmd_generate():
    st = json.loads(STATE.read_text()) if STATE.exists() else init_state()
    rung = st["rung"] + 1
    if rung > MAX_RUNG:
        print(f"ceiling rung {MAX_RUNG} already reached"); return
    rows, n_active = [], 0
    for nm, L in st["lineages"].items():
        if L["status"] != "active":
            continue
        n_active += 1
        for bi, member in enumerate(L["beam"]):
            for s in propose(member["smiles"], rung):
                rows.append({"lineage": nm, "beam_idx": bi,
                             "from_smiles": member["smiles"], "smiles": s})
    if not rows:
        print("no analogues proposed; nothing to score"); return
    d = pd.DataFrame(rows).drop_duplicates(subset=["lineage", "smiles"]).reset_index(drop=True)
    d["mol_id"] = [f"r{rung}_{i}" for i in range(len(d))]

    sdf = ROOT / f"data/beam/rung{rung}.sdf"
    w = Chem.SDWriter(str(sdf)); n = 0
    for _, r in d.iterrows():
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        m.SetProp("_Name", r["mol_id"]); w.write(m); n += 1
    w.close()
    d.to_csv(ROOT / f"data/beam/rung{rung}_manifest.csv", index=False)
    st["pending_rung"] = rung
    STATE.write_text(json.dumps(st, indent=1))
    print(f"\nRUNG {rung}: {n_active} active lineages -> {n} unique analogues")
    print(f"  SDF      : data/beam/rung{rung}.sdf")
    print(f"  manifest : data/beam/rung{rung}_manifest.csv")
    print(f"\n  NEXT: upload that SDF to ChemAGG, then:")
    print(f"        python src/beam.py ingest <downloaded_result.csv>")


def cmd_ingest(csv_path):
    st = json.loads(STATE.read_text())
    rung = st.get("pending_rung")
    man = pd.read_csv(ROOT / f"data/beam/rung{rung}_manifest.csv")
    res = pd.read_csv(csv_path, usecols=lambda c: c in ("Identifier", "Category", "Probability"))
    res = res.rename(columns={"Identifier": "ca_smiles", "Category": "chemagg_cat",
                              "Probability": "chemagg_prob"})

    def canon(s):
        m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
    res["can"] = res["ca_smiles"].map(canon)
    man["can"] = man["smiles"].map(canon)
    lut = res.drop_duplicates("can").set_index("can")
    man["chemagg_prob"] = man["can"].map(lut["chemagg_prob"])
    man["chemagg_cat"] = man["can"].map(lut["chemagg_cat"])
    miss = man["chemagg_prob"].isna().sum()
    print(f"rung {rung}: matched {len(man)-miss}/{len(man)} analogues by structure"
          + (f"   ({miss} UNMATCHED)" if miss else ""))

    # measured deltas, drug-likeness, and the free clogP null on the same molecules
    recs = []
    for _, r in man.iterrows():
        m = Chem.MolFromSmiles(r["smiles"]); p = Chem.MolFromSmiles(r["from_smiles"])
        dd = descriptors(m)
        recs.append({**dd,
                     "d_heavy": m.GetNumHeavyAtoms() - p.GetNumHeavyAtoms(),
                     "null_clogp_call": int(dd["clogP"] > CLOGP_CUT)})
    man = pd.concat([man, pd.DataFrame(recs)], axis=1)
    man.to_csv(ROOT / f"results/beam/rung{rung}_scored.csv", index=False)

    print(f"  measured d_heavy: {dict(sorted(man['d_heavy'].value_counts().items()))}")
    print(f"  ChemAGG called aggregator (cat=1): {int((man['chemagg_cat']==1).sum())}/{len(man)}")
    print(f"  clogP NULL called aggregator     : {int(man['null_clogp_call'].sum())}/{len(man)}")

    flipped = []
    for nm, L in st["lineages"].items():
        if L["status"] != "active":
            continue
        sub = man[man["lineage"] == nm].dropna(subset=["chemagg_prob"])
        if sub.empty:
            L["status"] = "exhausted"; continue
        sub = sub.sort_values("chemagg_prob", ascending=False)
        L["history"].append({"rung": rung, "n_scored": int(len(sub)),
                             "best_prob": float(sub.iloc[0]["chemagg_prob"]),
                             "best_smiles": sub.iloc[0]["smiles"],
                             "best_d_heavy": int(sub.iloc[0]["d_heavy"])})
        if (sub["chemagg_cat"] == 1).any():
            hit = sub[sub["chemagg_cat"] == 1].iloc[0]
            L["status"] = "flipped"
            L["flip"] = {"rung": rung, "smiles": hit["smiles"],
                         "prob": float(hit["chemagg_prob"]),
                         "d_heavy_total": int(hit["heavy"] - L["root_heavy"])}
            flipped.append(nm); continue
        L["beam"] = [{"smiles": r["smiles"], "rung": rung,
                      "score": float(r["chemagg_prob"]), "parent_of": r["from_smiles"]}
                     for _, r in sub.head(BEAM).iterrows()]

    st["rung"] = rung; st.pop("pending_rung", None)
    if rung >= MAX_RUNG:
        for L in st["lineages"].values():
            if L["status"] == "active":
                L["status"] = "censored"
    STATE.write_text(json.dumps(st, indent=1))

    tally = {}
    for L in st["lineages"].values():
        tally[L["status"]] = tally.get(L["status"], 0) + 1
    print(f"\n  lineage status: {tally}")
    if flipped:
        print(f"  FLIPPED this rung: {', '.join(flipped)}")
    print("\n  best probability per lineage so far:")
    for nm, L in sorted(st["lineages"].items()):
        h = L["history"]
        if h:
            print(f"    {nm[:20]:22s} rung {h[-1]['rung']}  p={h[-1]['best_prob']:.3f}  "
                  f"total dHeavy {int(Chem.MolFromSmiles(h[-1]['best_smiles']).GetNumHeavyAtoms()-L['root_heavy']):+d}  [{L['status']}]")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("generate", "ingest"):
        sys.exit(__doc__)
    if sys.argv[1] == "generate":
        cmd_generate()
    else:
        cmd_ingest(sys.argv[2])
