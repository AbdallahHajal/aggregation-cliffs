"""Select the compounds whose aggregation label is unambiguous.

Design v2 needs anchors whose ground truth is not in doubt, because the whole experiment
measures how far a molecule must be pushed before a filter's verdict flips. A compound
whose starting label is contested cannot anchor that measurement.

Evidence sources per compound:
  dls        - the DLS measurement from O'Donnell 2021 / Glenn 2024 (the reference truth)
  shoichet   - presence in the Aggregator Advisor reference database (positives only)
  cruzain    - label in the Cruzain assay set
  lactamase  - label in the AmpC assay set

A compound is GOLD if every source that mentions it agrees with the DLS label.
Anything contradicted by any source is CONTESTED and is excluded from the anchor set,
then discussed separately, because the contradictions are themselves a finding.
"""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ["cruzain", "lactamase", "shoichet"]


def main() -> None:
    d = pd.read_csv(ROOT / "results/filters/merged_58.csv")
    ov = pd.read_csv(ROOT / "results/filters/overlap_58.csv")
    d = d.merge(ov[["name"] + SRC], on="name", how="left")

    def classify(r):
        vals = {s: r[s] for s in SRC if pd.notna(r[s])}
        if not vals:
            return "dls_only"
        if all(int(v) == int(r["label"]) for v in vals.values()):
            return "gold"
        return "contested"

    d["evidence"] = d.apply(classify, axis=1)
    d["n_sources"] = d[SRC].notna().sum(axis=1)

    d.to_csv(ROOT / "results/filters/goldset.csv", index=False)

    print("=== evidence status of all 58 ===")
    print(pd.crosstab(d["evidence"], d["class"]).to_string())

    for cls in ["aggregator", "non-aggregator"]:
        g = d[(d["evidence"] == "gold") & (d["class"] == cls)].sort_values("n_sources", ascending=False)
        print(f"\n=== GOLD {cls}s ({len(g)}), every source agrees with DLS ===")
        for _, r in g.iterrows():
            srcs = ",".join(s for s in SRC if pd.notna(r[s])) or "none"
            print(f"   {r['name'][:26]:28s} corroborated by: {srcs}")

    c = d[d["evidence"] == "contested"]
    print(f"\n=== CONTESTED ({len(c)}), excluded from anchors ===")
    for _, r in c.iterrows():
        dis = ",".join(f"{s}={int(r[s])}" for s in SRC if pd.notna(r[s]) and int(r[s]) != int(r["label"]))
        print(f"   {r['name'][:26]:28s} DLS={int(r['label'])} but {dis}")

    o = d[d["evidence"] == "dls_only"]
    print(f"\n=== DLS ONLY ({len(o)}), single source, usable but weaker ===")
    print("   " + ", ".join(o["name"].str.strip().tolist()))


if __name__ == "__main__":
    main()
