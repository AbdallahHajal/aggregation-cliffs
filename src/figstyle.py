"""Shared figure style, following Sturm et al. Angew Chem 2025.

Design grammar taken from that paper: rounded panels with coloured borders, warm orange
for the aggregator class and teal for the non-aggregator class, structures drawn with
coloured atom halos, lettered badges, and light neutral backgrounds.
"""
import io
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

AGG      = "#E08A2E"   # aggregator, warm orange
AGG_SOFT = "#FBE6CC"
NON      = "#2FA98C"   # non-aggregator, teal
NON_SOFT = "#D6EFE8"
INK      = "#26292B"
GREY     = "#7C8285"
PANEL    = "#F7F7F5"
RULE     = "#D8D8D4"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.9,
    "savefig.facecolor": "white",
})


def hex2rgb(h, alpha=1.0):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def mol_png(smiles, highlight=None, colour=AGG, size=(430, 330), lw=2.1,
            sigma=0.40, alpha=0.62):
    """Draw a molecule with a soft radial glow over the highlighted atoms.

    Uses RDKit's SimilarityMaps, which paints a smooth Gaussian field from per-atom
    weights, rather than the hard highlight discs of PrepareAndDrawMolecule. This is the
    rendering used for explanation masks in Sturm et al. and reads far better at figure
    scale, because the glow sits behind the structure instead of occluding it.
    """
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    hl = set(highlight or [])
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.bondLineWidth = lw
    o.padding = 0.10
    cmap = LinearSegmentedColormap.from_list("glow", ["#FFFFFF", "#FFFFFF", colour])
    # Always route through SimilarityMaps so every panel shares the same grid padding
    # and therefore the same drawing scale. With no atoms to highlight the field is
    # drawn fully transparent rather than skipped, which would shrink the structure.
    weights = [1.0 if a.GetIdx() in hl else 0.0 for a in m.GetAtoms()] or [0.0]
    SimilarityMaps.GetSimilarityMapFromWeights(
        m, weights, draw2d=d, colorMap=cmap, sigma=sigma,
        contourLines=0, alpha=(alpha if hl else 0.0))
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def panel(ax, colour, face=None, lw=1.6, pad=0.012, radius=0.022):
    """Rounded bordered panel filling the axes, in the Sturm style."""
    ax.add_patch(FancyBboxPatch(
        (pad, pad), 1 - 2 * pad, 1 - 2 * pad, transform=ax.transAxes,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face or "white", edgecolor=colour, linewidth=lw, zorder=0))


def badge(ax, x, y, letter, colour, r=0.030, fs=9):
    """Lettered circular badge as used to key structures in Sturm's figures."""
    ax.add_patch(Circle((x, y), r, transform=ax.transAxes,
                        facecolor=colour, edgecolor="none", zorder=6))
    ax.text(x, y, letter, transform=ax.transAxes, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white", zorder=7)


def panel_title(ax, text, colour, x=0.5, y=0.955, fs=9.5):
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=colour, zorder=8)


def save(fig, name, root="Figures/manuscript"):
    """Figures/manuscript holds the eight the paper submits, Figures/parked the earlier
    rounds, and Figures/concept the mechanism artwork."""
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / root
    p.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(p / f"{name}.{ext}", dpi=600, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {root}/{name}.png / .pdf / .svg")
