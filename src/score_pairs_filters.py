"""Result 2: do the filters respond to a cliff?

For each mined matched pair with opposing labels, every filter is asked twice, once per
member. A filter that returns the same verdict for both members did not see the cliff.
The miss rate is that count over the pairs scored.

Stage 1 (this file, no external service): build the stratified sample and score the
clogP null model, whose inputs are already stored in the pair files.

Stage 2 (score_pairs_advisor.py, score_pairs_bad.py, score_pairs_chemagg.py) scores the
unique molecules of that sample with the three published filters.

Sampling is stratified by data set and by heavy-atom delta, because 59% of the census is
a single methylene and an unstratified draw would report the filters' behaviour on that
one transformation.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20260928
N_SAMPLE = 500
CLOGP_CUT = 3.5  # Aggregator Advisor's own threshold, read from its client bundle

SOURCES = {
    "cruzain": ROOT / "results/cruzain_full_real_tight.csv",
    "ampc": ROOT / "results/ampc_full_real_tight.csv",
}


def load_census() -> pd.DataFrame:
    frames = []
    for name, path in SOURCES.items():
        d = pd.read_csv(path)
        d["dataset"] = name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    print(f"raw pairs: {len(df):,}")
    print("\nkind:")
    print(df["kind"].value_counts().to_string())

    df = df[df["kind"] == "real"].copy()
    print(f"\ngenuine structural pairs: {len(df):,}")

    # deduplicate on the unordered InChIKey pair
    key = df.apply(lambda r: tuple(sorted((r["agg_ik"], r["non_ik"]))), axis=1)
    df = df.loc[~key.duplicated()].copy()
    print(f"after InChIKey pair dedup: {len(df):,}")
    return df


def stratify(df: pd.DataFrame) -> pd.DataFrame:
    """Proportional allocation over dataset x heavy-atom-delta bucket."""
    df = df.copy()
    df["dheavy_bucket"] = pd.cut(
        df["d_heavy"].abs(), bins=[-0.1, 0.5, 1.5, np.inf], labels=["0", "1", "2+"]
    )
    df["stratum"] = df["dataset"].astype(str) + ":" + df["dheavy_bucket"].astype(str)

    counts = df["stratum"].value_counts()
    print("\nstrata in census:")
    print(counts.to_string())

    rng = np.random.default_rng(SEED)
    take = (counts / counts.sum() * N_SAMPLE).round().astype(int)
    parts = []
    for stratum, n in take.items():
        pool = df[df["stratum"] == stratum]
        n = min(n, len(pool))
        if n:
            parts.append(pool.iloc[rng.choice(len(pool), size=n, replace=False)])
    out = pd.concat(parts, ignore_index=True)
    print(f"\nsampled: {len(out):,} pairs")
    print(out["stratum"].value_counts().to_string())
    return out


def boot_ci(hits: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(hits), size=(n_boot, len(hits)))
    props = hits[idx].mean(axis=1)
    return tuple(np.percentile(props, [2.5, 97.5]) * 100)


def score_null(sample: pd.DataFrame) -> pd.DataFrame:
    """clogP > 3.5 applied to each member. No service, no model, no RDKit call:
    both clogP values are already stored in the census."""
    s = sample.copy()
    s["null_agg"] = (s["agg_clogp"] > CLOGP_CUT).map({True: "agg", False: "non"})
    s["null_non"] = (s["non_clogp"] > CLOGP_CUT).map({True: "agg", False: "non"})

    s["null_missed"] = s["null_agg"] == s["null_non"]
    # a pair is called correctly only if the aggregator member is called 'agg'
    # AND the non-aggregator member is called 'non'
    s["null_correct"] = (s["null_agg"] == "agg") & (s["null_non"] == "non")
    return s


def main() -> None:
    census = load_census()
    sample = stratify(census)
    sample = score_null(sample)

    out = ROOT / "results/pairs_sample_500.csv"
    sample.to_csv(out, index=False)

    miss = sample["null_missed"].to_numpy()
    corr = sample["null_correct"].to_numpy()
    lo, hi = boot_ci(miss)
    clo, chi = boot_ci(corr)

    print("\n" + "=" * 62)
    print("clogP > 3.5 null model on the sampled pairs")
    print("=" * 62)
    print(f"  same verdict for both members (MISSED the cliff): "
          f"{miss.mean()*100:.1f}%  [{lo:.1f} to {hi:.1f}]  {miss.sum()}/{len(miss)}")
    print(f"  both members called correctly:                    "
          f"{corr.mean()*100:.1f}%  [{clo:.1f} to {chi:.1f}]  {corr.sum()}/{len(corr)}")
    print("\n  breakdown of the two verdicts per pair:")
    print(sample.groupby(["null_agg", "null_non"]).size().to_string())

    mols = pd.concat([
        sample[["agg_smiles", "agg_ik"]].rename(
            columns={"agg_smiles": "smiles", "agg_ik": "inchikey"}).assign(member="agg"),
        sample[["non_smiles", "non_ik"]].rename(
            columns={"non_smiles": "smiles", "non_ik": "inchikey"}).assign(member="non"),
    ], ignore_index=True).drop_duplicates("inchikey")
    mols.to_csv(ROOT / "results/pairs_sample_500_molecules.csv", index=False)

    print(f"\nwrote {out.name} and pairs_sample_500_molecules.csv "
          f"({len(mols):,} unique molecules to score externally)")


if __name__ == "__main__":
    main()
