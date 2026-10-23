#!/usr/bin/env python
"""Prepare the AmpC set with the same tool the cruzain set received: MolVS.

Written 2026-10-21 to remove a preparation asymmetry. The cruzain set arrived from Article
19 already reduced to its largest covalent fragments with MolVS, while the AmpC set had
received RDKit canonicalization alone, so all 10,795 of its multi-component records kept
their counter-ions. Unequal preparation of two sets that are compared against each other is
a confound: any difference between them could be the preparation rather than the chemistry.

MolVS is the right tool rather than a substitute for it. Its last release, 0.1.1, is from
2019, so the library applied here is the library Article 19 applied, with no version drift
to argue about. Candidate recipes were scored against cruzain's deposited std_smiles on
6,000 records (3,000 multi-fragment, 3,000 single):

    MolVS standardize only                 49.967%
    RDKit FragmentParent                   99.950%
    MolVS fragment_parent + standardize    99.950%
    MolVS fragment_parent                  99.967%   <- used here

MolVS fragment_parent calls standardize internally and then keeps the largest organic
fragment. Charges are NOT neutralized and tautomers NOT canonicalized, which is what
cruzain received: its deposited file retains [O-] carboxylates, and sodium acetate reduces
to CC(=O)[O-] rather than to acetic acid.

Deduplication: collapse to unique canonical structure, then drop any structure left carrying
both labels, since a compound whose label contradicts itself cannot support a cliff either
way. This matches cruzain's end state, which holds 191,023 rows for 191,023 unique
structures with no structure in both classes.

Usage:
    python src/prepare_sets.py ampc
    python src/prepare_sets.py cruzain     # re-derives cruzain, for cross-checking only

Writes data/AmpC/ampc_cleaned_molvs.csv. Overwrites no original.
"""
import pathlib
import sys
import time

import pandas as pd
from molvs import Standardizer
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
A19 = pathlib.Path(__import__("os").environ.get("CLIFFS_SOURCE_DATA", "data/source"))
DOT = r"\."

SETS = {
    "ampc": {
        "src": ROOT / "data/AmpC/ampc_cleaned.csv",
        "out": ROOT / "data/AmpC/ampc_cleaned_molvs.csv",
    },
    "cruzain": {
        "src": A19 / "Cruzain/_outputs_molvs_baselines/cruzain_cleaned.csv",
        "out": ROOT / "data/Cruzain/cruzain_cleaned_molvs.csv",
    },
}

t0 = time.time()
_S = Standardizer()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def standardize(smi):
    """The single operation applied to every structure, in both sets."""
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(_S.fragment_parent(m))
    except Exception:
        return None


def prepare(name):
    cfg = SETS[name]
    log(f"=== {name} : MolVS fragment_parent ===")
    df = pd.read_csv(cfg["src"], usecols=lambda c: c in ("SMILES", "label"))
    df = df.dropna(subset=["SMILES", "label"])
    df["label"] = df["label"].astype(int)
    log(f"  loaded {len(df):,} records from {cfg['src'].name}")
    log(f"  multi-fragment on input: {int(df.SMILES.str.contains(DOT, regex=True).sum()):,}")

    std = []
    for i, smi in enumerate(df["SMILES"].tolist()):
        std.append(standardize(smi))
        if (i + 1) % 25000 == 0:
            log(f"    standardized {i + 1:,}/{len(df):,}")
    df["std_smiles"] = std

    n_fail = int(df.std_smiles.isna().sum())
    df = df.dropna(subset=["std_smiles"]).copy()
    log(f"  failed to standardize: {n_fail:,} (dropped)")
    log(f"  multi-fragment after: "
        f"{int(df.std_smiles.str.contains(DOT, regex=True).sum()):,}")

    grp = df.groupby("std_smiles")["label"].nunique()
    conflicted = set(grp[grp > 1].index)
    n_before = len(df)
    df = df[~df.std_smiles.isin(conflicted)]
    log(f"  structures carrying BOTH labels: {len(conflicted):,} "
        f"({n_before - len(df):,} records dropped)")

    n_before = len(df)
    df = df.drop_duplicates("std_smiles", keep="first")
    log(f"  redundant duplicates removed: {n_before - len(df):,}")

    df["can_smiles"] = df["std_smiles"]
    out = df[["SMILES", "label", "can_smiles", "std_smiles"]].reset_index(drop=True)

    assert out.std_smiles.nunique() == len(out), "dedup failed"
    assert not out.std_smiles.str.contains(DOT, regex=True).any(), "fragments survived"

    log(f"  FINAL {len(out):,} records | labels {out.label.value_counts().to_dict()} | "
        f"aggregator prevalence {100 * out.label.mean():.3f}%")

    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["out"], index=False)
    log(f"  wrote {cfg['out']}")

    # If the RDKit-prepared file exists, report how far the two libraries disagree.
    rdk = cfg["out"].with_name(cfg["out"].name.replace("_molvs", "_harmonized"))
    if rdk.exists():
        other = pd.read_csv(rdk, usecols=["SMILES", "std_smiles"])
        j = out[["SMILES", "std_smiles"]].merge(other, on="SMILES", how="inner",
                                                suffixes=("_molvs", "_rdkit"))
        agree = int((j.std_smiles_molvs == j.std_smiles_rdkit).sum())
        log(f"  vs RDKit preparation: {len(j):,} shared records, "
            f"{agree:,} identical ({100 * agree / len(j):.4f}%), "
            f"{len(j) - agree:,} differ")
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["ampc"]
    for name in which:
        if name not in SETS:
            sys.exit(f"unknown set {name!r}; choose from {list(SETS)}")
        prepare(name)
    log("done")
