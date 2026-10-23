from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

TITLE = ("Aggregation cliffs and the meaning of a negative: the non-aggregator class "
         "in colloidal aggregation benchmarks is defined by absent measurement")

AUTHORS = ("Abdallah Abou Hajal¹*, [Molham Sakkal], [Richard A. Bryce²], "
           "[Mohammad A. Ghattas¹*]")
AFFIL = ("¹ College of Pharmacy and AAU Health and Biomedical Research Center, "
         "Al Ain University, Abu Dhabi 112612, United Arab Emirates\n"
         "² Division of Pharmacy and Optometry, University of Manchester, Manchester, UK\n"
         "* Corresponding authors")

HIGHLIGHTS = [
 "In the cruzain detergent counter-screen used to train most SCAM filters, 0 of 166,328 compounds labelled non-aggregator carry any dose-response measurement.",
 "The negative class therefore encodes assay inactivity, not a tested absence of colloid formation.",
 "Mining near-neighbour pairs with opposite labels yields 1,659 aggregation cliffs, 25 of which have identical molecular formula and identical calculated logP.",
 "164 of 498 stereoisomer pairs carry opposite labels, a data-internal label-flip rate of 32.9% in cruzain and 12.0% in AmpC.",
 "Molecular dynamics needs no labels and is therefore the only available arbiter for these pairs.",
]

ABSTRACT = [
 ("Background: Small colloidally aggregating molecules (SCAMs) are a leading source of "
  "false positives in high-throughput screening. Computational filters for SCAMs are trained "
  "almost exclusively on PubChem detergent counter-screens, and report accuracies between 82% "
  "and 95%. We asked what the negative class in those benchmarks actually represents."),
 ("Methods: We analysed the cruzain detergent counter-screen (PubChem AID 1476/1478; 191,499 "
  "labelled compounds) used in the ChemAGG, SCAM Detective and BAD Molecule Filter training sets. "
  "We recorded, per class, how many compounds carry a measured AC50 or Hill slope. We then mined "
  "aggregation cliffs, defined as pairs of compounds with ECFP4 Tanimoto at or above 0.85 and "
  "opposite experimental labels, and separated them into genuine structural cliffs and "
  "stereoisomer pairs using InChIKey skeleton blocks."),
 ("Results: All 25,171 compounds labelled aggregator carry a Hill slope (median 3.93) and an AC50 "
  "(median 31.6 µM). None of the 166,328 compounds labelled non-aggregator carries any "
  "measurement. The negative class is defined by absence of activity against cruzain rather than "
  "by a tested absence of colloid formation. Mining returned 2,975 near-neighbour pairs with "
  "opposite labels, of which 1,659 differ by less than 0.5 log units in calculated logP and less "
  "than 30 Da. In 104 of the 161 pairs small enough for coarse-grained simulation the two members "
  "differ by a single methylene group, and in 25 pairs they are isomers with identical formula and "
  "identical calculated logP. Of 498 stereoisomer pairs, which cannot differ in bulk "
  "hydrophobicity, 164 carry opposite labels, placing the label-flip rate at 32.9% in cruzain "
  "and 12.0% in AmpC. Among genuine structural pairs the opposite-label rate is 43.4% in cruzain "
  "and 36.3% in AmpC, exceeding each data set's own floor by 10.5 and 24.3 percentage points."),
 ("Conclusion: Reported SCAM classifier accuracies are measured against a negative class that was "
  "never tested for aggregation. Aggregation cliffs concentrate this problem and are unresolvable "
  "by any descriptor that depends on bulk physicochemistry. Because molecular dynamics requires no "
  "training labels, simulation offers an assay-independent arbiter, and we set out the protocol and "
  "the compound set for that test."),
]

TOC = [
 ("1. Introduction", [
   "1.1 Colloidal aggregation as the dominant HTS artifact",
   "1.2 How computational SCAM filters are trained, and on what",
   "1.3 What a negative label is assumed to mean",
   "1.4 Aim of this work"]),
 ("2. Methods", [
   "2.1 Datasets and provenance (PubChem AID 1476/1478; Aggregator Advisor)",
   "2.2 Standardisation, following the BAD Molecule Filter pipeline",
   "2.3 Auditing measurement coverage per class",
   "2.4 Definition and mining of aggregation cliffs",
   "2.5 Separating structural cliffs from stereoisomer pairs",
   "2.6 Benchmarking published filters on the cliff set",
   "2.7 Coarse-grained and atomistic simulation protocol"]),
 ("3. Results", [
   "3.1 Measurement coverage is total for positives and zero for negatives",
   "3.2 The aggregation cliff landscape",
   "3.3 The one-methylene signature",
   "3.4 Isomer cliffs: identical formula, identical logP, opposite label",
   "3.5 Stereoisomer pairs bound the label-flip rate",
   "3.6 Published filters on the cliff set versus matched controls",
   "3.7 Simulation as an assay-independent arbiter"]),
 ("4. Discussion", [
   "4.1 What reported accuracies are actually measuring",
   "4.2 Why logD outperformed trained filters in previous work",
   "4.3 Implications for benchmark construction and dataset contracts",
   "4.4 Limitations"]),
 ("5. Conclusion", []),
 ("Data and software availability", []),
 ("References", []),
]

def styled(doc):
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
    return doc

# ---------- Manuscript Draft ----------
d = styled(Document())
for s in d.sections:
    s.left_margin = s.right_margin = Inches(1)
p = d.add_paragraph(); r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(14)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = d.add_paragraph(); p.add_run(AUTHORS); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = d.add_paragraph(); r = p.add_run(AFFIL); r.font.size = Pt(10)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

d.add_heading("Article Highlights", level=1)
for h in HIGHLIGHTS:
    d.add_paragraph(h, style="List Bullet")

d.add_heading("Abstract", level=1)
for a in ABSTRACT:
    d.add_paragraph(a)
p = d.add_paragraph(); r = p.add_run(
    "KEYWORDS: colloidal aggregation; SCAMs; high-throughput screening; false positives; "
    "activity cliffs; benchmark quality; molecular dynamics; coarse-grained simulation; Martini 3")
r.bold = True

for head, subs in TOC:
    d.add_heading(head, level=1)
    for s in subs:
        d.add_heading(s, level=2)
        d.add_paragraph("[to be written]")
d.save("Manuscript Draft.docx")

# ---------- Table of Contents ----------
t = styled(Document())
p = t.add_paragraph(); r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(13)
for head, subs in TOC:
    t.add_paragraph(head, style="List Number" if False else "Normal").runs[0].bold = True
    for s in subs:
        t.add_paragraph("    " + s)
t.save("Table of Contents.docx")
print("wrote Manuscript Draft.docx and Table of Contents.docx")
