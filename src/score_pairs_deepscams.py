#!/usr/bin/env python
"""Score the mined pairs with DeepSCAMs.

Lee K, Yang A, Lin Y-C, Reker D, Bernardes GJL, Rodrigues T. "Combating small-molecule
aggregation with machine learning." Cell Rep Phys Sci 2021, 2(9), 100573.
DOI 10.1016/j.xcrp.2021.100573.  Code: github.com/tcorodrigues/DeepSCAMs (BSD-3).

Why this filter completes the audit. The negative class of the other filters is either a
drug database (ChemAGG) or an assay counter-screen (SCAM Detective). DeepSCAMs is the only
widely cited aggregation model whose non-aggregators were MEASURED: 653 compounds scored
by dynamic light scattering at 30 micromolar in one homogeneous screen. The three filters
therefore form a ladder in label provenance, from asserted to assayed to measured, against
which the per-pair miss rate can be read.

The published script ships no weights. It retrains at run time from `train_data.txt` under
a fixed seed, so the model is reproduced exactly rather than approximated. Featurisation,
hyperparameters and the AMBIG exclusion below are copied verbatim from DeepSCAMs.py; only
the input and output plumbing is ours.
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[1]
VEND = ROOT / "vendor/deepscams/sample_files"

# verbatim from DeepSCAMs.py: Morgan r3 2048 bits concatenated with the RDKit
# descriptor list, with descriptor index 2 skipped as in the original
DESCR = Descriptors._descList[0:2] + Descriptors._descList[3:]
CALC = [x[1] for x in DESCR]


def describe(mols):
    out = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        fp_expl = [float(x) for x in list(fp.ToBitString())]
        ds_n = []
        for d in CALC:
            try:
                v = d(mol)
            except Exception:
                v = np.finfo(np.float32).max
            ds_n.append(np.float32(min(v, np.finfo(np.float32).max)))
        out.append(fp_expl + list(ds_n))
    return out


def train():
    scams = pd.read_csv(VEND / "train_data.txt", sep="\t")
    smiles, y = scams.iloc[:, 1], scams.iloc[:, 2]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]
    fps = np.array(describe([mols[i] for i in keep]))
    lab = np.array(["1" if y.iloc[i] == "AGG" else ("0" if y.iloc[i] == "NONAGG" else "-1")
                    for i in keep])
    x, yy = fps[lab != "-1"], lab[lab != "-1"]
    print(f"training on {len(yy)} measured compounds "
          f"({(yy=='1').sum()} aggregators, {(yy=='0').sum()} non-aggregators)", flush=True)

    scaler = MinMaxScaler().fit(x)
    mlp = MLPClassifier(activation="tanh", alpha=0.0001, batch_size="auto", beta_1=0.9,
                        beta_2=0.999, early_stopping=False, epsilon=1e-08,
                        hidden_layer_sizes=(100, 1000, 1000), learning_rate="constant",
                        learning_rate_init=0.001, max_iter=200, momentum=0.9,
                        n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                        random_state=1234, shuffle=True, solver="sgd", tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
    mlp.fit(scaler.transform(x), yy)
    return mlp, scaler


def main():
    inp = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/pairs_sample_500_molecules.csv")
    outp = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/pairs_sample_500_deepscams.csv")

    mlp, scaler = train()
    df = pd.read_csv(inp)
    mols = [Chem.MolFromSmiles(str(s)) for s in df["smiles"]]
    ok = [i for i, m in enumerate(mols) if m is not None]
    print(f"scoring {len(ok)} molecules", flush=True)

    X = np.array(describe([mols[i] for i in ok]))
    Xs = pd.DataFrame(scaler.transform(X))
    good = ~Xs.isna().any(axis=1)
    print(f"  {int((~good).sum())} dropped for non-finite descriptors", flush=True)

    res = df.iloc[ok].reset_index(drop=True)
    res["deepscams_cat"] = np.nan
    res["deepscams_prob"] = np.nan
    idx = np.flatnonzero(good.to_numpy())
    res.loc[idx, "deepscams_cat"] = mlp.predict(Xs[good]).astype(int)
    res.loc[idx, "deepscams_prob"] = mlp.predict_proba(Xs[good])[:, 1]
    res.to_csv(outp, index=False)

    print(f"wrote {outp.name}")
    print(res["deepscams_cat"].value_counts(dropna=False)
          .rename({0.0: "non-aggregator", 1.0: "aggregator"}).to_string())


if __name__ == "__main__":
    main()
