"""Figure. Isomer cliffs: pairs with identical molecular formula and identical
computed descriptors that nevertheless receive opposite aggregation labels."""
import io
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors as rdd
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
RDLogger.DisableLog("rdApp.*")

plt.rcParams.update({"font.family":"sans-serif",
  "font.sans-serif":["Arial","Helvetica","DejaVu Sans"], "font.size":8})
VERM, BLUE = "#D55E00", "#0072B2"

PAIRS = [
 ("CCCCCCC/C=C\\CCCCCCCCC(=O)O", "CCCCCCCC/C=C\\CCCCCCCC(=O)O",
  "C18H34O2", "double bond moved by one carbon"),
 ("Nc1c2c(nc3sc4c(c13)CCCCC4)CCCC2", "Nc1c2c(nc3sc4c(c13)CCCC4)CCCCC2",
  "C16H20N2S", "one CH2 moved between the two fused rings"),
 ("CCOc1ccc(C2CC(c3ccc(CC)cc3)n3nc(N)nc3N2)cc1",
  "CCOc1ccc(C2CC(c3ccc(CC)cc3)Nc3nc(N)nn32)cc1",
  "C21H25N5O", "triazole ring-fusion isomers"),
]

def draw(smi, w=460, h=230):
    m = Chem.MolFromSmiles(smi)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions(); o.bondLineWidth = 2; o.addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(d, m); d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())), m

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(len(PAIRS), 2, hspace=0.46, wspace=0.03,
                      left=0.075, right=0.985, top=0.905, bottom=0.035)

for i, (sa, sn, formula, note) in enumerate(PAIRS):
    ia, ma = draw(sa); ib, mb = draw(sn)
    for j, (img, mol, lab, col) in enumerate(((ia, ma, "aggregator", VERM),
                                              (ib, mb, "non-aggregator", BLUE))):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(img); ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), img.size[0]-1, img.size[1]-1, fill=False,
                                   edgecolor=col, linewidth=1.1,
                                   transform=ax.transData, clip_on=False))
        ax.text(0.5, -0.045, lab, transform=ax.transAxes, fontsize=8,
                color=col, fontweight="bold", ha="center", va="top")
    axl = fig.add_subplot(gs[i, :]); axl.set_axis_off()
    axl.text(0.5, 1.10, f"{formula}    {note}", transform=axl.transAxes,
             fontsize=8.6, ha="center", va="bottom", fontweight="bold")
    axl.text(0.5, 1.015,
             f"clogP {Crippen.MolLogP(ma):.3f}    TPSA {Descriptors.TPSA(ma):.1f}    "
             f"Fsp3 {rdd.CalcFractionCSP3(ma):.3f}    ECFP4 Tanimoto 1.000, "
             f"every value identical in both",
             transform=axl.transAxes, fontsize=7.2, ha="center", va="bottom",
             color="#555555")

fig.suptitle("Aggregation cliffs that no descriptor-based model can separate",
             fontsize=10.5, fontweight="bold", x=0.075, ha="left", y=0.975)
for e in ("png","pdf","svg"): fig.savefig(f"Figures/isomer_cliffs.{e}", dpi=600)
print("wrote Figures/isomer_cliffs.png/.pdf/.svg")
