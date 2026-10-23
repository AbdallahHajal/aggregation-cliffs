#!/usr/bin/env python
"""Validate, across every Cruzain record, the recipe used to harmonize AmpC.

src/prepare_sets.py assumes the Cruzain set received MolVS fragment_parent followed by
canonical SMILES, and applies that same operation to AmpC. The assumption was established
on a 6,000-record sample (99.967% match). Since every prepared AmpC number rests on it,
this re-derives Cruzain's deposited std_smiles from its raw SMILES column for all 191,023
records and reports the exact mismatch count and what the mismatches are.

Pass "molvs" (default) or "rdkit" to pick the library. MolVS reproduces the deposited set
in 190,876 of 191,023 records with no structural disagreement; RDKit reaches 190,867 but
emits 17 SMILES it cannot itself re-parse, which is why MolVS is the one used.

Reads only. Writes its report to stdout.
"""
import collections
import pathlib
import sys
import time

import pandas as pd
from molvs import Standardizer
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# Which library to test. MolVS fragment_parent is the recipe reported in the Methods;
# RDKit FragmentParent is kept as a cross-check of the same operation.
TOOL = sys.argv[1] if len(sys.argv) > 1 else "molvs"
_S = Standardizer()


def apply_recipe(m):
    if TOOL == "molvs":
        return _S.fragment_parent(m)
    return rdMolStandardize.FragmentParent(m)

RDLogger.DisableLog("rdApp.*")

CRZ = (__import__("os").environ.get("CLIFFS_SOURCE_DATA", "data/source")
       + "/Cruzain/_outputs_molvs_baselines/cruzain_cleaned.csv")

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


df = pd.read_csv(CRZ, usecols=["SMILES", "std_smiles"])
log(f"loaded {len(df):,} Cruzain records")

ok = 0
mismatches = []
unparsed = 0
for i, (raw, dep) in enumerate(zip(df.SMILES, df.std_smiles)):
    m = Chem.MolFromSmiles(str(raw))
    if m is None:
        unparsed += 1
        continue
    try:
        got = Chem.MolToSmiles(apply_recipe(m))
    except Exception:
        unparsed += 1
        continue
    if got == dep:
        ok += 1
    else:
        mismatches.append((raw, dep, got))
    if (i + 1) % 50000 == 0:
        log(f"  {i + 1:,}/{len(df):,} checked, {len(mismatches):,} mismatches so far")

tot = ok + len(mismatches)
log("done")
print(f"\n{'=' * 74}")
print(f"RECIPE:  {TOOL} fragment parent -> canonical SMILES")
print(f"{'=' * 74}")
print(f"  records compared     {tot:,}")
print(f"  reproduced exactly   {ok:,}  ({100 * ok / tot:.4f}%)")
print(f"  mismatched           {len(mismatches):,}  ({100 * len(mismatches) / tot:.4f}%)")
print(f"  unparseable          {unparsed:,}")


def classify(raw, dep, got):
    """Why did this one differ? Same-molecule rewrites are harmless."""
    if dep.replace("/", "").replace("\\", "") == got.replace("/", "").replace("\\", ""):
        return "double-bond stereo written the other way (same molecule)"
    dn = Chem.MolFromSmiles(dep)
    gn = Chem.MolFromSmiles(got)
    if dn is None or gn is None:
        return "deposited or derived SMILES does not parse"
    try:
        if Chem.MolToSmiles(dn) == Chem.MolToSmiles(gn):
            return "same molecule, different canonical form"
        if rdMolDescriptors.CalcMolFormula(dn) == rdMolDescriptors.CalcMolFormula(gn):
            return "same formula, different charge or bond assignment"
    except Exception:
        return "comparison failed"
    if dn.GetNumHeavyAtoms() <= 2 or gn.GetNumHeavyAtoms() <= 2:
        return "inorganic or single-atom species"
    return "genuinely different structure"


if mismatches:
    kinds = collections.Counter(classify(*m) for m in mismatches)
    print("\n  mismatches by cause:")
    for k, v in kinds.most_common():
        print(f"    {v:>5,}  {k}")
    print("\n  first few:")
    for raw, dep, got in mismatches[:8]:
        print(f"    raw: {raw[:68]}")
        print(f"    dep: {dep[:68]}")
        print(f"    got: {got[:68]}\n")

material = sum(v for k, v in collections.Counter(
    classify(*m) for m in mismatches).items() if k == "genuinely different structure")
print(f"{'=' * 74}")
print(f"  structurally material disagreements: {material:,} of {tot:,} "
      f"({100 * material / tot:.4f}%)")
print(f"{'=' * 74}")
