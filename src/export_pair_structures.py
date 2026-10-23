#!/usr/bin/env python
"""Export every structure drawn in the paper as an editable SVG, one file per molecule.

The figures bake their structures into a raster panel, which is no use for redrawing. Here
each molecule is written on its own, still aligned to the maximum common substructure it
shares with its partner, so a pair opened side by side in a vector editor lines up exactly
as it does in the figure. The differing atoms and the bonds carrying them keep their class
colour, and a plain uncoloured copy is written alongside for anyone who wants to restyle.

Also writes a manifest CSV so each file can be traced back to its pair, its labels and its
measured values.
"""
import pathlib
import sys

import pandas as pd
from rdkit import Chem, RDLogger

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figstyle import AGG, NON  # noqa: E402
from pairdraw import draw_pair  # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "Figures/structures_svg"
SEED = 20260928


def safe(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))[:40]


def emit(rows, folder, size=(600, 500), bond_px=46):
    folder.mkdir(parents=True, exist_ok=True)
    man = []
    for r in rows:
        stem = f"{r['tag']}"
        for coloured in (True, False):
            a, b = draw_pair(r["agg_smiles"], r["non_smiles"], size=size, bond_px=bond_px,
                             colour_a=AGG if coloured else "#1A1A1A",
                             colour_b=NON if coloured else "#1A1A1A",
                             stereo=bool(r.get("stereo")))
            suffix = "" if coloured else "_plain"
            (folder / f"{stem}_aggregator{suffix}.svg").write_text(a)
            (folder / f"{stem}_nonaggregator{suffix}.svg").write_text(b)
        man.append(r)
    return man


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []

    # --- the measured pair, Figure 5 -------------------------------------------------
    anchor = [dict(tag="anchor_clioquinol", figure="Figure 5",
                   agg_name="clioquinol", non_name="clioquinol O-methyl ether",
                   agg_smiles="Oc1c(I)cc(Cl)c2cccnc12",
                   non_smiles="COc1c(I)cc(Cl)c2cccnc12",
                   dataset="dynamic light scattering", tanimoto="",
                   note="CAC 15 uM, radius 1818 nm vs no scattering 0.5-100 uM")]
    manifest += emit(anchor, OUT / "anchor_pair", size=(620, 520), bond_px=54)

    # --- the six gallery pairs, Figure 4, under the study seed -----------------------
    s = pd.read_csv(ROOT / "results/pairs_sample_500.csv")
    s = s[(s.agg_heavy <= 32) & (s.non_heavy <= 32)]

    def pick(df, **n):
        return pd.concat([df[df.dataset == k].sample(v, random_state=SEED)
                          for k, v in n.items()])

    iso = pick(s[(s.d_heavy == 0) & (s.d_clogp.abs() < 1e-9)], cruzain=1, ampc=1)
    ch2 = pick(s[(s.d_heavy == 1) & (s.d_mw.abs().between(13.9, 14.1))], cruzain=1, ampc=2)
    oth = pick(s[(s.d_heavy == 1) & (~s.d_mw.abs().between(13.9, 14.1))], cruzain=1)
    ex = pd.concat([iso, ch2, oth]).reset_index(drop=True)

    rows = []
    for i, r in ex.iterrows():
        tag = ("same formula" if r.d_heavy == 0
               else ("one methylene" if abs(abs(r.d_mw) - 14.03) < 0.1
                     else f"plus 1 heavy, dMW {r.d_mw:+.1f}"))
        rows.append(dict(tag=f"gallery_{chr(97+i)}_{safe(r.dataset)}", figure="Figure 4",
                         agg_name="", non_name="",
                         agg_smiles=r.agg_smiles, non_smiles=r.non_smiles,
                         dataset=r.dataset, tanimoto=f"{r.tanimoto:.2f}", note=tag,
                         stereo=bool(r.d_heavy == 0)))
    manifest += emit(rows, OUT / "gallery_pairs")

    cols = ["tag", "figure", "dataset", "tanimoto", "agg_name", "non_name",
            "agg_smiles", "non_smiles", "note"]
    pd.DataFrame(manifest)[cols].to_csv(OUT / "manifest.csv", index=False)

    n = len(list(OUT.rglob("*.svg")))
    print(f"wrote {n} SVG files and manifest.csv to {OUT.relative_to(ROOT)}")
    for p in sorted(OUT.rglob("*.svg")):
        print("   ", p.relative_to(OUT))


if __name__ == "__main__":
    main()
