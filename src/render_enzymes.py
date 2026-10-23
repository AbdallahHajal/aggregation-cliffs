# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
# ]
# ///
"""Render AmpC beta-lactamase and cruzain for figure assembly.

These are the two enzymes whose detergent counter-screens the census is mined from, so the
figures use their real crystal structures rather than a generic protein. Each is rendered
twice: a cartoon, for showing the enzyme as an object, and a solvent-excluded surface, for
showing it adsorbed onto a colloid in the manner of Ghattas et al.

Backgrounds are transparent and the palette matches the paper's figures, so a rendering can
be dropped straight onto a panel without masking.
"""
import os
import pathlib

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pymol  # noqa: E402  pytype: disable=import-error
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()
from pymol import cmd  # noqa: E402  pytype: disable=import-error

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDB = ROOT / "data/pdb"
OUT = ROOT / "Figures/enzymes"
OUT.mkdir(parents=True, exist_ok=True)

NAVY, AGG, NON, GREY = "0x1F3A5F", "0xE08A2E", "0x2FA98C", "0x7C8285"

TARGETS = [
    # pdb id, label, chain to keep, cartoon colour, surface colour
    ("1KE4", "ampc_betalactamase", "A", NAVY, NAVY),
    ("2OZ2", "cruzain", "A", NON, NON),
]


def setup():
    cmd.set("ray_opaque_background", 0)      # transparent, for compositing
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("surface_quality", 1)
    cmd.set("transparency", 0.0)
    cmd.set("specular", 0.2)
    cmd.bg_color("white")


for pdb_id, label, chain, cart_col, surf_col in TARGETS:
    path = PDB / f"{pdb_id}.pdb"
    if not path.exists():
        print(f"MISSING {path}")
        continue

    cmd.reinitialize()
    setup()
    cmd.load(str(path), "s")
    n = cmd.count_atoms("all")
    if n == 0:
        print(f"ERROR: {pdb_id} loaded 0 atoms")
        cmd.quit()
        raise SystemExit(1)

    # one biological copy, no waters, no crystallisation additives
    cmd.remove("solvent")
    cmd.remove("not polymer and not resn HEM+FAD+NAD")
    cmd.create("prot", f"s and polymer and chain {chain}")
    cmd.delete("s")
    cmd.remove("hydrogens")
    kept = cmd.count_atoms("prot")

    # --- cartoon ---------------------------------------------------------------
    cmd.hide("everything")
    cmd.show("cartoon", "prot")
    cmd.color(cart_col, "prot")
    cmd.color(GREY, "prot and ss L+''")
    cmd.orient("prot")
    cmd.zoom("prot", 3)
    cmd.png(str(OUT / f"{label}_cartoon.png"), width=2000, height=1600, dpi=300, ray=1)

    # secondary-structure coloured variant, useful when the enzyme is the subject
    cmd.color("0xC44E52", "prot and ss H")
    cmd.color("0xE0B33A", "prot and ss S")
    cmd.color(GREY, "prot and ss L+''")
    cmd.png(str(OUT / f"{label}_cartoon_ss.png"), width=2000, height=1600, dpi=300, ray=1)

    # --- surface ---------------------------------------------------------------
    cmd.hide("everything")
    cmd.show("surface", "prot")
    cmd.color(surf_col, "prot")
    cmd.orient("prot")
    cmd.zoom("prot", 3)
    cmd.png(str(OUT / f"{label}_surface.png"), width=2000, height=1600, dpi=300, ray=1)

    cmd.save(str(OUT / f"{label}.pse"))
    print(f"{pdb_id}  {label}: {kept} atoms in chain {chain}, 3 renderings written")

cmd.quit()
