#!/usr/bin/env python
"""Table 3, recomputed on the harmonized (MolVS) census.

src/representation_sweep.py reads results/census_full_3375.csv, the census as it stood on
30 September, before the AmpC set was re-prepared with MolVS on 21 October to match the
cruzain preparation. That left Table 3 computed on one preparation while Table 1 and the
census itself were computed on another, and left the released census unable to reproduce
the table.

This re-runs the identical panel and the identical identity criterion on
results/census_full_molvs.csv, so Table 3, Table 1 and the released data all rest on the
same preparation of both screens. Nothing else changes: same twelve representations, same
bit sizes, same row-wise Tanimoto, same >= 0.999999 identity test.

    "<project>/.venv/bin/python" src/representation_sweep_molvs.py

Writes results/representation_sweep_molvs.csv and prints the old and new tables together.
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation_sweep import PANEL, tanimoto_rows  # noqa: E402

OLD = ROOT / "results/representation_sweep.csv"
NEW = ROOT / "results/representation_sweep_molvs.csv"


def main():
    df = pd.read_csv(ROOT / "results/census_full_molvs.csv")
    print(f"harmonized census pairs: {len(df):,}\n")
    agg, non = df["agg_smiles"].tolist(), df["non_smiles"].tolist()

    rows = []
    for name, fp in PANEL:
        try:
            A = np.asarray(fp.transform(agg))
            B = np.asarray(fp.transform(non))
        except Exception as e:
            print(f"  {name:26s} FAILED  {type(e).__name__}: {str(e)[:70]}")
            continue
        t = tanimoto_rows(A, B)
        ident = int((t >= 0.999999).sum())
        rows.append({"representation": name, "bits": A.shape[1],
                     "identical_pairs": ident,
                     "pct_identical": round(100 * ident / len(t), 2),
                     "median_tanimoto": round(float(np.median(t)), 4),
                     "pairs_above_0.95": int((t >= 0.95).sum())})
        print(f"  {name:26s} identical {ident:5d} ({100 * ident / len(t):5.2f}%)", flush=True)

    out = pd.DataFrame(rows).sort_values("identical_pairs", ascending=False)
    out.to_csv(NEW, index=False)

    print("\n" + "=" * 86)
    print("TABLE 3, OLD (3,375 pre-MolVS)  vs  NEW (%d harmonized)" % len(df))
    print("=" * 86)
    old = pd.read_csv(OLD).set_index("representation")
    print("%-26s %7s %8s   %7s %8s   %s" % ("representation", "n_old", "pct_old",
                                            "n_new", "pct_new", "delta"))
    for _, r in out.iterrows():
        rep = r["representation"]
        if rep in old.index:
            o = old.loc[rep]
            print("%-26s %7d %7.2f%%   %7d %7.2f%%   %+.2f pts"
                  % (rep, o["identical_pairs"], o["pct_identical"],
                     r["identical_pairs"], r["pct_identical"],
                     r["pct_identical"] - o["pct_identical"]))
        else:
            print("%-26s %7s %8s   %7d %7.2f%%   new" % (rep, "-", "-",
                                                         r["identical_pairs"], r["pct_identical"]))
    print(f"\nwrote {NEW.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
