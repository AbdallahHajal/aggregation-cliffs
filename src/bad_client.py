"""Robust BAD filter client.

The published app computes Mordred over the whole batch then subsets to a fixed column
list. When Mordred returns an error object for any molecule, that column becomes object
dtype, is dropped by the app's own select_dtypes filter, and the subset raises KeyError -
so a SINGLE unusual molecule invalidates the entire batch. This client bisects on failure
so the damage is isolated to the offending molecules, which are returned as NaN and
counted rather than silently lost.
"""
import pathlib, subprocess
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run(smiles, tag):
    inp, out = ROOT / f"data/work/_bc_{tag}_in.csv", ROOT / f"data/work/_bc_{tag}_out.csv"
    pd.DataFrame({"smiles": smiles}).to_csv(inp, index=False)
    r = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64",
                        "-v", f"{ROOT}/vendor/bad_filter:/app", "-v", f"{ROOT}/data:/data",
                        "badfilter:1.1", "python", "/app/bad_predict.py",
                        f"/data/work/_bc_{tag}_in.csv", f"/data/work/_bc_{tag}_out.csv"],
                       capture_output=True)
    if r.returncode != 0:
        return None
    return pd.read_csv(out)["bad_prob"].tolist()


def score(smiles, tag="x", _depth=0):
    """Return (probs, n_failed). Failures are None in the list."""
    if not smiles:
        return [], 0
    got = _run(smiles, tag)
    if got is not None:
        return got, 0
    if len(smiles) == 1:
        return [None], 1
    mid = len(smiles) // 2
    a, fa = score(smiles[:mid], tag + "a", _depth + 1)
    b, fb = score(smiles[mid:], tag + "b", _depth + 1)
    return a + b, fa + fb
