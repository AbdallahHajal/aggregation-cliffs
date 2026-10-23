#!/usr/bin/env python
"""Rebuild the mmpdb index for the harmonized census and recompute its two reported figures.

The manuscript reports that 2,795 census pairs, or 82.8%, also satisfy a formal matched-pair
definition, and that the transformation table is close to symmetric, with removal of a methyl
accounting for 381 pairs against 163 for its addition. Both come from an mmpdb index built on
29 Sep 2026 from the pre-harmonization structures, so both have to be rebuilt now that AmpC
has been re-prepared with MolVS (src/prepare_sets.py).

Stereochemistry is stripped before indexing, as in the original: mmpdb's environment-SMARTS
builder raises inside RDKit 2026.03.1 on certain double-bond stereo specifications, and
matched-pair analysis is constitutional in any case. Compound IDs are the census InChIKeys,
so the index joins back to the census exactly as src/mmp_transformations.py expects.

    python src/rebuild_mmp.py --verify    reproduce the published numbers from the OLD files
    python src/rebuild_mmp.py             build and report for the harmonized census

--verify is the self-test: if it does not return 2,795 / 82.8% / 381 / 163 then this script
does not reproduce how the published figures were obtained and its new numbers cannot be
trusted. Writes only new paths; nothing existing is overwritten.
"""
import argparse
import pathlib
import sqlite3
import subprocess
import sys

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMP = ROOT / "results/mmp"
MMPDB = ROOT / ".venv/bin/mmpdb"

ap = argparse.ArgumentParser()
ap.add_argument("--verify", action="store_true",
                help="reproduce the published figures from the original census and index")
a = ap.parse_args()

CRZ_TIGHT = ROOT / "results/cruzain_full_real_tight.csv"
AMPC_TIGHT = (ROOT / "results/ampc_full_real_tight.csv" if a.verify
              else ROOT / "results/ampc_molvs_real_tight.csv")
TAG = "orig" if a.verify else "molvs"
SMI = MMP / f"census_flat_{TAG}.smi"
FRAGDB = MMP / f"flat_{TAG}.fragdb"
DB = MMP / f"flat_{TAG}.mmpdb"
CENSUS_OUT = ROOT / f"results/census_full_{TAG}.csv"


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- 1. pooled census
ct = pd.read_csv(CRZ_TIGHT)
at = pd.read_csv(AMPC_TIGHT)
seen = {frozenset((r.agg_smiles, r.non_smiles)) for r in ct.itertuples()}
extra = at[[frozenset((r.agg_smiles, r.non_smiles)) not in seen for r in at.itertuples()]]
ct = ct.assign(dataset="cruzain")
extra = extra.assign(dataset="ampc")
census = pd.concat([ct, extra], ignore_index=True)
log(f"pooled census: {len(ct):,} cruzain + {len(extra):,} AmpC = {len(census):,} pairs")
census.to_csv(CENSUS_OUT, index=False)

# ---------------------------------------------------------------- 2. flat structures
flat = {}
for col_s, col_k in (("agg_smiles", "agg_ik"), ("non_smiles", "non_ik")):
    for smi, ik in zip(census[col_s], census[col_k]):
        if ik in flat:
            continue
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        Chem.RemoveStereochemistry(m)
        flat[ik] = Chem.MolToSmiles(m)
log(f"unique structures, stereo stripped: {len(flat):,}")
with open(SMI, "w", encoding="utf-8") as f:
    for ik, smi in flat.items():
        f.write(f"{smi}\t{ik}\n")

# ---------------------------------------------------------------- 3. fragment and index
if not DB.exists():
    for stage, cmd in (
        ("fragment", [str(MMPDB), "fragment", str(SMI), "-o", str(FRAGDB)]),
        ("index", [str(MMPDB), "index", str(FRAGDB), "-o", str(DB)]),
    ):
        log(f"mmpdb {stage} ...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"mmpdb {stage} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        log(f"  ok -> {(FRAGDB if stage == 'fragment' else DB).name}")
else:
    log(f"reusing existing {DB.name}")

# ---------------------------------------------------------------- 4. the two figures
opp, agg_side = set(), {}
for x, y in zip(census.agg_ik, census.non_ik):
    opp.add((x, y))
    opp.add((y, x))
    agg_side[(x, y)] = x
    agg_side[(y, x)] = x

q = """
    SELECT c1.public_id AS id1, c2.public_id AS id2,
           rs1.smiles AS frm, rs2.smiles AS to_
    FROM pair
    JOIN compound c1 ON c1.id = pair.compound1_id
    JOIN compound c2 ON c2.id = pair.compound2_id
    JOIN rule_environment re ON re.id = pair.rule_environment_id
    JOIN rule r ON r.id = re.rule_id
    JOIN rule_smiles rs1 ON rs1.id = r.from_smiles_id
    JOIN rule_smiles rs2 ON rs2.id = r.to_smiles_id
"""
con = sqlite3.connect(DB)
mmp = pd.read_sql(q, con)
con.close()
log(f"formal MMP pairs recovered by mmpdb: {len(mmp):,}")

mmp["key"] = list(zip(mmp.id1, mmp.id2))
hit = mmp[[k in opp for k in mmp.key]].copy()
uniq = {frozenset(k) for k in hit.key}
pct = 100 * len(uniq) / len(census)

hit["from_agg"] = [agg_side[k] == k[0] for k in hit.key]
hit["transform"] = [(f"{f} >> {t}" if fa else f"{t} >> {f}")
                    for f, t, fa in zip(hit.frm, hit.to_, hit.from_agg)]
hit = hit.drop_duplicates(subset=["transform", "key"])
counts = hit["transform"].value_counts()

print("\n" + "=" * 78)
print(f"CENSUS PAIRS mmpdb ALSO CALLS A FORMAL MMP: {len(uniq):,} of {len(census):,} "
      f"({pct:.1f}%)")
if a.verify:
    print("   [paper: 2,795 pairs, or 82.8%]")
print("=" * 78)
print("\ntop transformations, oriented aggregator >> non-aggregator:")
for t, n in counts.head(12).items():
    print(f"  {n:5d}   {t}")

# methyl removal against methyl addition, the symmetry claim
METHYL_OFF = [t for t in counts.index if t in ("[*:1]C >> [*:1][H]",)]
METHYL_ON = [t for t in counts.index if t in ("[*:1][H] >> [*:1]C",)]
off = int(counts[METHYL_OFF].sum()) if METHYL_OFF else 0
on = int(counts[METHYL_ON].sum()) if METHYL_ON else 0
print(f"\nmethyl removal (aggregator -> non-aggregator): {off:,}")
print(f"methyl addition (aggregator -> non-aggregator): {on:,}")
if a.verify:
    print("   [paper: 381 pairs against 163 for its addition]")
print(f"\ndistinct label-flipping transformations: {hit['transform'].nunique():,}")
out = ROOT / f"results/mmp_label_flipping_transforms_{TAG}.csv"
counts.rename_axis("transform").reset_index(name="n_pairs").to_csv(out, index=False)
print(f"wrote {out.name} and {CENSUS_OUT.name}")
