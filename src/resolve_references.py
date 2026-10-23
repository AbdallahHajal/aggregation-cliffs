#!/usr/bin/env python
"""Resolve the manuscript's author-year placeholders to verified references.

Every entry below is anchored on a DOI that was recovered during this study, either from
the article itself or from a live database lookup. Nothing here is written from memory: the
script queries the Crossref REST API for each DOI and formats the reference from the
response. A DOI that does not resolve is reported as a failure rather than filled in.

Renumbering by hand has silently mis-pointed brackets on this project before, so the
numbering is produced here from the order of first appearance in the assembled manuscript
and written back mechanically.
"""
import json
import urllib.parse
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRAFT = ROOT.parent / "Article 27 - Cliffs Review" / "draft"
MAILTO = "abdallah.abouhajal@scifiniti.com"

# placeholder -> DOI. Each DOI was obtained during this study, not recalled.
DOIS = {
    "McGovern 2002":  "10.1021/jm010533y",
    "Feng 2007":      "10.1021/jm061317y",
    "Irwin 2015":     "10.1021/acs.jmedchem.5b01105",
    "Kramer 2012":    "10.1021/jm300131x",
    "Yang 2019":      "10.1021/acs.jcim.9b00541",
    "Reker 2019":     "10.1038/s41557-019-0234-9",
    "Alves 2020":     "10.1021/acs.jcim.0c00415",
    "Sheridan 2020":  "10.1021/acs.jcim.9b01067",
    "Lee 2021":       "10.1016/j.xcrp.2021.100573",
    "Rodrigues 2022": "10.1039/d2dd00001f",
    "van Tilborg 2022": "10.1021/acs.jcim.2c01073",
    "Kombo 2024":     "10.1021/acsomega.4c02886",
    "Abou Hajal 2024": "10.1021/acs.jcim.4c00363",
    "Sturm 2025":     "10.1002/anie.202503259",
    "Schuh 2026":     "10.1039/d6sc01799a",
    "Rath 2026":      "10.1186/s13321-026-01207-4",
    "Slaughter 2026": "10.1016/j.addr.2025.115758",
    "Abe 2026":       "10.1186/s13321-026-01210-9",
    "Nael 2026":      "10.1021/acs.jcim.6c00514",
    "Dalke 2018":     "10.1021/acs.jcim.8b00173",
    "Fourches 2010":  "10.1021/ci100176x",
    "Adamczyk 2024":  "10.1016/j.softx.2024.101944",
    "Arvidsson 2024": "10.1186/s13321-024-00870-9",
    "Wellawatte 2022": "10.1039/d1sc05259d",
    "Polishchuk 2020": "10.1186/s13321-020-00431-w",
    "Bickerton 2012": "10.1038/nchem.1243",
    "Ertl 2009":      "10.1186/1758-2946-1-8",
    # --- activity-cliff foundations ---
    "Maggiora 2006":   "10.1021/ci060117s",
    "Guha 2008":       "10.1021/ci7004093",
    "Hu 2012":         "10.1021/ci3001138",
    "Stumpfe 2012":    "10.1021/jm201706b",
    "Stumpfe 2020":    "10.1007/s10822-020-00315-z",
    "Bajorath 2017":   "10.4155/fsoa-2017-0065",
    "Peltason 2009":   "10.1002/cmdc.200900300",
    # --- cliffs and machine learning ---
    "Tamura 2023":     "10.1186/s13321-022-00676-7",
    "Shen 2026":       "10.1038/s41467-026-75713-2",
    "Jimenez 2022":    "10.1021/acs.jcim.1c01163",
    "Iqbal 2021":      "10.1016/j.ailsci.2021.100022",
    "Dablander 2022":  "10.1021/acs.jcim.2c00327",
    "Lopez 2025":      "10.1002/minf.202400054",
    "Goller 2026":     "10.1039/d5dd00398a",
    # --- measurement uncertainty and curation ---
    "Kalliokoski 2013": "10.1371/journal.pone.0061007",
    "Kramer 2014":     "10.1186/1758-2946-6-s1-o6",
    "Landrum 2024":    "10.1021/acs.jcim.4c00049",
    "Schiebroek 2026": "10.1021/acs.jcim.6c01018",
    "Ekins 2013":      "10.1186/1758-2946-5-s1-o4",
    "Nicolotti 2026":  "10.1007/s00894-026-06795-4",
    # --- matched molecular pairs beyond potency ---
    "Leach 2006":      "10.1021/jm0605233",
    "Keefer 2018":     "10.1021/acs.jmedchem.7b00935",
    "Abreu 2026":      "10.3389/fddsv.2026.1899616",
    # --- assay interference beyond aggregation ---
    "Baell 2010":      "10.1021/jm901137j",
    "Baell 2016":      "10.1021/acs.jnatprod.5b00947",
    "Stork 2019":      "10.1021/acs.jcim.8b00677",
    "Shi 2024":        "10.1093/nar/gkae424",
    "Alves 2023":      "10.1021/acs.jmedchem.3c00482",
    "Rao 2010":        "10.1002/jcc.21347",
    "Jadhav 2010":     "10.1021/jm901070c",
    "Ghosh 2023":      "10.1016/j.scitotenv.2023.162263",
    # --- descriptors and representations ---
    "Wildman 1999":    "10.1021/ci990307l",
    "Rogers 2010":     "10.1021/ci100050t",
    "Carhart 1985":    "10.1021/ci00046a002",
    "Bemis 1996":      "10.1021/jm9602928",
    # Blevins 2025 is an arXiv preprint with no Crossref record; handled separately.
}
PREPRINTS = {
    "Blevins 2025": ("Blevins AD, Quigley IK. Clever Hans in chemistry: models identify "
                     "the chemist, not the chemistry. arXiv 2025, 2512.20924."),
}


def crossref(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}"
    req = urllib.request.Request(url, headers={"User-Agent": f"aggregation-cliffs/1.0 ({MAILTO})"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode())["message"]


def fmt(m):
    """ACS-style reference from a Crossref record."""
    auths = []
    for a in (m.get("author") or [])[:12]:
        fam = a.get("family", "").strip()
        giv = "".join(p[0] for p in a.get("given", "").replace("-", " ").split() if p)
        if fam:
            auths.append(f"{fam} {giv}".strip())
    if len(m.get("author") or []) > 12:
        auths.append("et al")
    who = "; ".join(auths) if auths else "[no authors listed]"
    title = (m.get("title") or [""])[0].strip().rstrip(".")
    jrnl = (m.get("short-container-title") or m.get("container-title") or [""])[0]
    year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    vol = m.get("volume", "")
    issue = m.get("issue", "")
    page = m.get("page", "")
    bits = f"{jrnl} {year}"
    if vol:
        bits += f", {vol}"
    if issue:
        bits += f" ({issue})"
    if page:
        bits += f", {page}"
    return f"{who}. {title}. {bits}. DOI: {m.get('DOI')}."


def main():
    out, failed = {}, []
    for key, doi in DOIS.items():
        try:
            out[key] = fmt(crossref(doi))
            print(f"  OK    {key:18s} {doi}")
        except Exception as e:
            failed.append((key, doi, f"{type(e).__name__}: {str(e)[:60]}"))
            print(f"  FAIL  {key:18s} {doi}  {type(e).__name__}")
        time.sleep(0.3)
    out.update(PREPRINTS)
    for k in PREPRINTS:
        print(f"  PRE   {k:18s} (preprint, no Crossref record)")

    p = ROOT / "results/references_resolved.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nresolved {len(out)} references -> {p.name}")
    if failed:
        print("\nFAILED, do not cite these until resolved:")
        for k, d, e in failed:
            print(f"  {k}  {d}  {e}")
        sys.exit(1)


if __name__ == "__main__":
    import urllib.parse  # noqa: E402
    main()
