"""Chemical-stability screen for generated analogues.

QED and synthetic accessibility do NOT catch hydrolytic or chemical instability. The
nortriptyline analogue that passed both contained a carbamic acid, which decomposes to
the amine and CO2 in water and cannot meaningfully be simulated in aqueous MD.

Two layers: Pat Walters' REOS rules via synspace, plus an explicit list of groups that
are unstable in water or otherwise unsuitable for an aggregation study.
"""
import pathlib
import pandas as pd
from rdkit import Chem, RDLogger
from synspace import reos as _reos

RDLogger.DisableLog("rdApp.*")
ROOT = pathlib.Path(__file__).resolve().parents[1]

UNSTABLE = {
    "carbamic acid":      "[NX3][CX3](=O)[OX2H1]",
    "carbonic acid/ester":"[OX2][CX3](=O)[OX2H1]",
    "gem-diol":           "[CX4]([OX2H1])[OX2H1]",
    "hemiacetal":         "[CX4]([OX2H1])[OX2][CX4]",
    "acyl halide":        "[CX3](=O)[F,Cl,Br,I]",
    "anhydride":          "[CX3](=O)[OX2][CX3]=O",
    "enol":               "[CX3]=[CX3][OX2H1]",
    "peroxide":           "[OX2][OX2]",
    "N-halo":             "[NX3][F,Cl,Br,I]",
    "thiol ester":        "[CX3](=O)[SX2]",
    "imine (unstab.)":    "[CX3H1]=[NX2H1]",
    "azide":              "[NX1]~[NX2]~[NX2]",
    "isocyanate":         "[NX2]=[CX2]=[OX1]",
    "aziridine/epoxide":  "[C;r3][O,N;r3]",
}
_PATTS = {k: Chem.MolFromSmarts(v) for k, v in UNSTABLE.items()}
_reos_obj = _reos.REOS()


def screen(smiles):
    """Return (ok: bool, reason: str)."""
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return False, "unparseable"
    for name, patt in _PATTS.items():
        if patt is not None and m.HasSubstructMatch(patt):
            return False, name
    try:
        rule, _ = _reos_obj.process_mol(m)
        if rule and rule.lower() not in ("ok", "none"):
            return False, f"REOS:{rule}"
    except Exception:
        pass
    return True, "ok"


if __name__ == "__main__":
    a = pd.read_csv(ROOT / "results/beam/all_analogues_annotated.csv")
    res = [screen(s) for s in a["smiles"]]
    a["stable"] = [r[0] for r in res]
    a["stability_reason"] = [r[1] for r in res]
    a.to_csv(ROOT / "results/beam/all_analogues_annotated.csv", index=False)

    print(f"{a['stable'].sum()}/{len(a)} analogues pass the stability screen "
          f"({a['stable'].mean():.1%})")
    print("\ntop rejection reasons:")
    print(a.loc[~a["stable"], "stability_reason"].value_counts().head(12).to_string())

    DL = (a["qed"] >= 0.5) & (a["sa"] <= 4.5) & a["stable"]
    print(f"\ndrug-like AND stable: {DL.sum()}/{len(a)} ({DL.mean():.1%})")
    print("\nflip distance under the full filter:")
    print(f"{'lineage':20s} {'all':>7s} {'QED+SA':>9s} {'+stable':>9s}")
    n_ok = 0
    for nm, g in a.groupby("lineage"):
        f_all = g[g["chemagg_cat"] == 1]["total_d_heavy"]
        f_dl = g[(g["chemagg_cat"] == 1) & (g["qed"] >= 0.5) & (g["sa"] <= 4.5)]["total_d_heavy"]
        f_st = g[(g["chemagg_cat"] == 1) & (g["qed"] >= 0.5) & (g["sa"] <= 4.5) & g["stable"]]["total_d_heavy"]
        if len(f_st): n_ok += 1
        fmt = lambda s: f"+{int(s.min())}" if len(s) else "none"
        print(f"{nm[:20]:20s} {fmt(f_all):>7s} {fmt(f_dl):>9s} {fmt(f_st):>9s}")
    print(f"\n{n_ok}/12 lineages flip using only stable, drug-like analogues")
