#!/usr/bin/env python
"""How much of the census is invisible to each molecular representation?

The manuscript reports that 285 census pairs are identical under the ECFP4 fingerprint the
filters use, and that only 54 of those survive one extra step of radius. That is a single
observation about one parameter of one representation. This generalises it: every pair is
scored under a panel of fingerprints, and the fraction rendered indistinguishable is
reported per representation.

A pair that a representation cannot separate is one that no model consuming only that
representation can resolve, whatever it was trained on. The resulting curve is therefore an
upper bound on any such model's performance on this census, derived without training
anything.

Uses scikit-fingerprints (Adamczyk & Ludynia, SoftwareX 2024, 10.1016/j.softx.2024.101944).
"""
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]

from skfp.fingerprints import (  # noqa: E402
    AtomPairFingerprint, AvalonFingerprint, ECFPFingerprint, MACCSFingerprint,
    PhysiochemicalPropertiesFingerprint,
    PubChemFingerprint, RDKitFingerprint, TopologicalTorsionFingerprint,
)

# MAP4 and MHFP6 are optional. Neither contributes a row to the reported table, and importing
# them by name at module scope meant that a class renamed in a later scikit-fingerprints
# release stopped the sweep before a single representation was scored. Resolved here instead,
# so the twelve reported rows no longer depend on the two that do not appear in them.
OPTIONAL = []
for _attr, _label, _kwargs in (("MAPFingerprint", "MAP4", {"fp_size": 2048}),
                               ("MHFPFingerprint", "MHFP6", {"fp_size": 2048})):
    try:
        import skfp.fingerprints as _skfp  # noqa: E402

        OPTIONAL.append((_label, getattr(_skfp, _attr)(**_kwargs)))
    except (ImportError, AttributeError, TypeError) as _exc:
        print(f"  {_label:26s} SKIPPED at import  {type(_exc).__name__}: {str(_exc)[:60]}")

PANEL = [
    ("ECFP4 (2048b, r2)", ECFPFingerprint(fp_size=2048, radius=2)),
    ("ECFP6 (2048b, r3)", ECFPFingerprint(fp_size=2048, radius=3)),
    ("ECFP4 unfolded (32768b)", ECFPFingerprint(fp_size=32768, radius=2)),
    ("ECFP4 counts", ECFPFingerprint(fp_size=2048, radius=2, count=True)),
    ("FCFP4 (features, r2)", ECFPFingerprint(fp_size=2048, radius=2, use_pharmacophoric_invariants=True)),
    ("RDKit path", RDKitFingerprint(fp_size=2048)),
    ("Atom pair", AtomPairFingerprint(fp_size=2048)),
    ("Topological torsion", TopologicalTorsionFingerprint(fp_size=2048)),
    ("Avalon", AvalonFingerprint(fp_size=2048)),
    ("MACCS (166b)", MACCSFingerprint()),
    ("PubChem (881b)", PubChemFingerprint()),
    ("Physicochemical", PhysiochemicalPropertiesFingerprint(fp_size=2048)),
] + OPTIONAL


def tanimoto_rows(A, B):
    """Row-wise Tanimoto between two equal-length binary/count matrices."""
    A = A.astype(float)
    B = B.astype(float)
    num = (np.minimum(A, B)).sum(axis=1)
    den = (np.maximum(A, B)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(den > 0, num / den, 1.0)
    return t


def main():
    df = pd.read_csv(ROOT / "results/census_full_3375.csv")
    print(f"census pairs: {len(df):,}")
    agg = df["agg_smiles"].tolist()
    non = df["non_smiles"].tolist()

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
        print(f"  {name:26s} identical {ident:5d} ({100*ident/len(t):5.2f}%)   "
              f"median Tc {np.median(t):.3f}", flush=True)

    out = pd.DataFrame(rows).sort_values("identical_pairs", ascending=False)
    out.to_csv(ROOT / "results/representation_sweep.csv", index=False)
    print("\n" + "=" * 74)
    print("PAIRS NO MODEL ON THAT REPRESENTATION CAN RESOLVE")
    print("=" * 74)
    print(out.to_string(index=False))
    print("\nwrote results/representation_sweep.csv")


if __name__ == "__main__":
    main()
