"""Check the 58 DLS-labelled compounds against the BAD filter's training sources.

The BAD filter (JCIM 2024, 10.1021/acs.jcim.4c00363) was trained on three data sets
named in its Methods: Lactamase (AmpC), Cruzain, and Shoichet. Any compound of ours
that appears in those sets was seen during development, so a performance figure that
includes it is not an external estimate.

Writes results/filters/overlap_58.csv and prints the label-conflict table.
"""
import pathlib
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = pathlib.Path(__file__).resolve().parents[1]
A19 = pathlib.Path(__import__("os").environ.get("CLIFFS_SOURCE_DATA", "data/source"))
A25 = ROOT / "data"

SOURCES = {
    "cruzain":    (A19 / "Cruzain/_outputs_molvs_baselines/cruzain_cleaned.csv", "can_smiles", "label"),
    "lactamase":  (A25 / "AmpC/ampc_cleaned.csv", "can_smiles", "label"),
    "shoichet":   (A19 / "Cruzain/_outputs_molvs_baselines/aggadvisor_cleaned.csv", "can_smiles", None),
}


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def main() -> None:
    ext = pd.read_csv(ROOT / "data/work/bad_external58.csv")
    ext["can"] = ext["smiles"].map(canon)

    for nm, (path, col, labcol) in SOURCES.items():
        if not path.exists():
            print(f"  {nm}: FILE MISSING {path}")
            ext[nm] = pd.NA
            continue
        use = [col] + ([labcol] if labcol else [])
        df = pd.read_csv(path, usecols=use).dropna(subset=[col])
        if labcol:
            # a structure can appear more than once; keep max label (aggregator wins)
            m = df.groupby(col)[labcol].max()
            ext[nm] = ext["can"].map(m)
        else:
            present = set(df[col])
            ext[nm] = ext["can"].map(lambda c: 1 if c in present else pd.NA)
        n = ext[nm].notna().sum()
        print(f"  {nm:10s}: {n:2d}/58 present")

    src = ["cruzain", "lactamase", "shoichet"]
    ext["n_sources"] = ext[src].notna().sum(axis=1)
    ext["seen"] = ext["n_sources"] > 0

    out = ROOT / "results/filters/overlap_58.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ext.to_csv(out, index=False)
    print(f"\nseen in >=1 training source: {ext['seen'].sum()}/58")
    print(f"genuinely unseen           : {(~ext['seen']).sum()}/58")

    print("\n=== label conflicts (training sources disagreeing with DLS, or each other) ===")
    print(f"{'compound':22s} {'DLS':>4s} {'cruz':>5s} {'lact':>5s} {'shoi':>5s}  {'BADp':>6s}  verdict")
    rows = []
    for _, r in ext[ext["seen"]].iterrows():
        vals = {k: r[k] for k in src if pd.notna(r[k])}
        conflict = any(int(v) != int(r["label"]) for v in vals.values())
        if conflict:
            f = lambda k: ("-" if pd.isna(r[k]) else str(int(r[k])))
            print(f"{r['name'][:22]:22s} {int(r['label']):>4d} {f('cruzain'):>5s} {f('lactamase'):>5s} {f('shoichet'):>5s}  {r['bad_prob']:>6.3f}  {r['bad_call']}")
            rows.append(r["name"])
    print(f"\n{len(rows)} of {ext['seen'].sum()} seen compounds carry a label that contradicts the DLS measurement.")


if __name__ == "__main__":
    main()
