"""Figure. Aggregation cliffs are dominated by a single methylene, and a subset of
them is invisible to any descriptor-based model."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif",
  "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
  "font.size":8,"axes.labelsize":8,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
  "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6})
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#9A9A9A"

cru = pd.read_csv("results/cruzain_full_real_tight.csv")
amp = pd.read_csv("results/ampc_full_real_tight.csv")
pc  = {"cruzain": pd.read_csv("results/cliff_physchem_cruzain.csv"),
       "ampc":    pd.read_csv("results/cliff_physchem_ampc.csv")}

fig, axA = plt.subplots(1, 1, figsize=(5.2, 2.9))

# (a) delta MW distribution
bins = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
for d, lab, col, off in ((cru,"Cruzain",BLUE,-0.35), (amp,"AmpC",VERM,0.35)):
    h,_ = np.histogram(d.d_mw.values, bins=bins)
    pct = 100*h/len(d)
    axA.bar(np.arange(len(h))+off, pct, 0.7, color=col, label=f"{lab} (n={len(d):,})")
axA.set_xticks(np.arange(len(bins)-1))
axA.set_xticklabels([f"{bins[i]}" for i in range(len(bins)-1)], fontsize=6.6)
axA.set_xlabel("Difference in molecular weight within the pair (Da)")
axA.set_ylabel("Cliff pairs (%)")
i14 = bins.index(14)
axA.annotate("one CH$_2$\n55% / 63%", xy=(i14, 58), xytext=(i14-3.6, 46),
             fontsize=7.4, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.7, color="#444444"))
axA.legend(frameon=False, fontsize=7.2, loc="upper left")
for s in ("top","right"): axA.spines[s].set_visible(False)
axA.set_title("Most aggregation cliffs differ by a single methylene",
              fontsize=8.5, loc="left", fontweight="bold", pad=4)

fig.subplots_adjust(left=0.115, right=0.985, top=0.87, bottom=0.185)
for e in ("png","pdf","svg"): fig.savefig(f"Figures/cliff_dmw.{e}", dpi=600)
n_iso_c = (cru.d_mw.values==0.0).sum(); n_iso_a = (amp.d_mw.values==0.0).sum()
print("wrote Figures/cliff_dmw.png/.pdf/.svg")
print(f"  dMW=14: Cruzain {100*(cru.d_mw.round()==14).mean():.1f}%  AmpC {100*(amp.d_mw.round()==14).mean():.1f}%")
print(f"  isomer pairs: Cruzain {n_iso_c}, AmpC {n_iso_a}, total {n_iso_c+n_iso_a}")
