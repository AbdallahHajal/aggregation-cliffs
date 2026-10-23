"""Draw the two members of a matched pair so the difference is visible.

The problem with drawing each member independently is that RDKit lays each one out on
its own, so the shared core appears rotated or reflected between the two panels and the
reader cannot see what changed. Here the maximum common substructure is computed once,
given a single 2D depiction, and both members are then laid out against that template,
so the common core is drawn identically in both panels and only the differing atoms move.

Atoms and bonds outside the MCS are highlighted on both members, in the class colour.
Bond highlighting matters: an added methyl is one atom and one bond, and highlighting the
atom alone reads as a dot rather than as a substituent.

Output is SVG, which is what a journal wants, with PNG available for preview.
"""
import pathlib
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figstyle import AGG, NON, hex2rgb  # noqa: E402

rdDepictor.SetPreferCoordGen(True)


def mcs_template(a, b, timeout=10):
    """Return (mcs_mol_with_2D, smarts) for the two SMILES."""
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return None, None
    res = rdFMCS.FindMCS([ma, mb], timeout=timeout, ringMatchesRingOnly=True,
                         completeRingsOnly=False,
                         bondCompare=rdFMCS.BondCompare.CompareOrderExact)
    if not res.smartsString:
        return None, None
    patt = Chem.MolFromSmarts(res.smartsString)
    rdDepictor.Compute2DCoords(patt)
    return patt, res.smartsString


def outside_mcs(mol, patt):
    """Atom and bond indices of mol that lie outside the MCS match."""
    match = set(mol.GetSubstructMatch(patt))
    atoms = [i for i in range(mol.GetNumAtoms()) if i not in match]
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if b.GetBeginAtomIdx() not in match or b.GetEndAtomIdx() not in match]
    return atoms, bonds


def _span(mol):
    c = mol.GetConformer()
    xs = [c.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [c.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    return max(max(xs) - min(xs), 1e-6), max(max(ys) - min(ys), 1e-6)


def _bond_unit(mol):
    c = mol.GetConformer()
    lens = []
    for b in mol.GetBonds():
        p, q = c.GetAtomPosition(b.GetBeginAtomIdx()), c.GetAtomPosition(b.GetEndAtomIdx())
        lens.append(((p.x - q.x) ** 2 + (p.y - q.y) ** 2) ** 0.5)
    return sorted(lens)[len(lens) // 2] if lens else 1.5


def draw_pair(smi_a, smi_b, colour_a=AGG, colour_b=NON, size=(430, 300),
              legend_a="", legend_b="", stereo=False, bond_px=34, opaque=True):
    """Return (svg_a, svg_b) with both members aligned to their shared core."""
    patt, _ = mcs_template(smi_a, smi_b)

    prepared = []
    for smi, colour, legend in ((smi_a, colour_a, legend_a), (smi_b, colour_b, legend_b)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            prepared.append(None)
            continue
        if patt is not None and mol.HasSubstructMatch(patt):
            rdDepictor.GenerateDepictionMatching2DStructure(mol, patt, acceptFailure=True)
        else:
            rdDepictor.Compute2DCoords(mol)
        prepared.append((mol, colour, legend))

    # A fixed bond length keeps both panels on one scale, but it overrides RDKit's
    # fit-to-canvas, so a long molecule runs off the edge and over the panel caption.
    # Take the nominal length, or whatever length lets the larger member fit, whichever
    # is smaller, and give both panels that.
    margin = 0.86                      # leaves room for atom labels at the extremes
    for item in prepared:
        if item is None:
            continue
        w, h = _span(item[0])
        fit = min(size[0] * margin / w, size[1] * margin / h)
        bond_px = min(bond_px, fit * _bond_unit(item[0]))

    out = []
    for item in prepared:
        if item is None:
            out.append("")
            continue
        mol, colour, legend = item
        atoms, bonds = outside_mcs(mol, patt) if patt is not None else ([], [])
        rgb = hex2rgb(colour)
        d = rdMolDraw2D.MolDraw2DSVG(*size)
        o = d.drawOptions()
        o.bondLineWidth = 2.0
        o.padding = 0.04
        o.clearBackground = opaque       # False lets a tinted panel show through
        o.fixedBondLength = bond_px     # identical scale in both panels
        o.continuousHighlight = True    # fill the highlight rather than ring it
        o.fillHighlights = True
        o.highlightBondWidthMultiplier = 14
        o.setHighlightColour((rgb[0], rgb[1], rgb[2], 0.30))
        o.addStereoAnnotation = stereo  # on for stereoisomer pairs, off otherwise
        o.annotationFontScale = 0.65
        o.legendFontSize = 15
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, legend=legend,
            highlightAtoms=atoms, highlightBonds=bonds,
            highlightAtomColors={i: rgb for i in atoms},
            highlightBondColors={i: rgb for i in bonds})
        d.FinishDrawing()
        out.append(d.GetDrawingText())
    return out[0], out[1]


def write_pair(smi_a, smi_b, stem, outdir, **kw):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sa, sb = draw_pair(smi_a, smi_b, **kw)
    (outdir / f"{stem}_agg.svg").write_text(sa)
    (outdir / f"{stem}_non.svg").write_text(sb)
    return outdir / f"{stem}_agg.svg", outdir / f"{stem}_non.svg"
