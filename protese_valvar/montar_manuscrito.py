#!/usr/bin/env python3
"""Builds the manuscript and the supplement (.docx) in EJCTS format.

Inputs: manuscrito/manuscrito_EJCTS_v1.md (text), manuscrito/Table*.csv, manuscrito/supp/*.csv
(suplemento_tabelas.py), manuscrito/supl_metodos.md, figures (PNG).
Outputs: manuscrito/Manuscript_EJCTS.docx and manuscrito/Supplementary_Material_EJCTS.docx
(Calibri 11, double spacing in the text, continuous line numbers, page numbers).
"""
import os
import re
import subprocess

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

os.chdir("protese_valvar/manuscrito")
tab = lambda df: df.to_markdown(index=False)  # noqa: E731
TITLE = ("Mechanical versus biological valve prostheses in Brazil's public health system, 2000–2025: "
         "a nationwide analysis of 305 566 admissions")

# ------------------------------------------------------------------ tables of the main text
t1 = pd.read_csv("Table1_baseline.csv").fillna("")
t2 = pd.read_csv("Table2_multilevel.csv")
main = t2.dropna(subset=["OR (95% CrI)"])
LAB = [("C(fx2, Treatment('60–64'))[T.", "Age "), ("C(sexo, Treatment('F'))[T.M]", "Male sex"),
       ("C(valva)[T.mitral]", "Mitral position"), ("C(etiologia, Treatment('não reumática'))[T.", "Aetiology: "),
       ("C(per2, Treatment('2018–2019'))[T.", "Period "), ("C(regiao, Treatment('Southeast'))[T.", "Region: "),
       ("C(vol2, Treatment('<25'))[T.", "Hospital volume, operations/year: "), ("urgencia", "Urgent admission"),
       ("reumática", "rheumatic"), ("congênita", "congenital"), ("endocardite", "endocarditis"),
       ("indeterminada", "not specified")]


def lab(t):
    for a, b in LAB:
        t = t.replace(a, b)
    t = t.rstrip("]")
    return re.sub(r"^Age (.+)$", r"Age \1 years", t)


m2 = pd.DataFrame({"Variable": main.term.map(lab), "OR (95% CrI)": main["OR (95% CrI)"]})
foot = dict(zip(t2[t2["OR (95% CrI)"].isna()].term, t2[t2["OR (95% CrI)"].isna()].OR))
note2 = (f"Mixed-effects logistic model with a random intercept for hospital; {int(foot['admissions']):,} admissions in "
         f"{int(foot['hospitals'])} hospitals with ≥30 admissions. Reference categories: age 60–64 years, female sex, "
         "aortic position, non-rheumatic aetiology, 2018–2019, Southeast region and <25 operations/year. "
         f"Hospital random-intercept SD {foot['hospital random-intercept SD']}; MOR {foot['MOR']}; ICC {foot['ICC']}. "
         "CrI, credible interval; ICC, intraclass correlation coefficient; MOR, median odds ratio; OR, odds ratio.")
t3 = pd.read_csv("Table3_2020_shift_robustness.csv", dtype=str).fillna("")
t3["P"] = t3.P.replace({"0.000": "<0.001"})
t3 = t3.drop(columns="Estimate").rename(columns={"Value (95% CI)": "OR (95% CI)"})
note3 = ("Patients aged 18–59 years, aortic and mitral position. Interrupted time series: segmented binomial model of "
         "monthly proportions with calendar-month terms and HAC standard errors; placebo breaks fitted on the "
         "pre-2020 series. Other rows: logistic models comparing 2022–2025 with 2018–2019, adjusted for valve "
         "position, aetiology, sex, age, admission type and region. CI, confidence interval; OR, odds ratio.")

md = open("manuscrito_EJCTS_v1.md").read()
md = md.replace("Table 1. Characteristics of the analysis cohort by prosthesis type, 2008–2025 (Table1_baseline.csv).",
                "**Table 1.** Characteristics of the analysis cohort by prosthesis type, 2008–2025.\n\n" + tab(t1)
                + "\n\nValues are n (%) unless stated otherwise. IQR, interquartile range.\n")
md = md.replace("Table 2. Determinants of mechanical-prosthesis use: mixed-effects logistic model (Table2_multilevel.csv).",
                "**Table 2.** Determinants of mechanical-prosthesis use in adults, 2008–2025.\n\n" + tab(m2) + "\n\n" + note2 + "\n")
md = re.sub(r"Table 3\. Change in mechanical-prosthesis use after 2020 in patients aged 18–59 years: main and sensitivity analyses\n\(Table3_2020_shift_robustness\.csv\)\.",
            lambda _: "**Table 3.** Change in mechanical-prosthesis use after 2020: main and sensitivity analyses.\n\n"
            + tab(t3) + "\n\n" + note3 + "\n", md)
figs = "\n\n# FIGURES\n\n" + "\n\n".join(
    f"**{c}**\n\n![]({f})" for f, c in [("Figure1_flowchart.png", "Figure 1"), ("Figure1_series_2002_2025.png", "Figure 2"),
                                        ("Figure2_monthly_18_59.png", "Figure 3"), ("Figure3_funnel.png", "Figure 4")])
# EJCTS order: legends, tables, references; figures appended for the co-authors' convenience
corpo, refs = md.split("# REFERENCES")
md = corpo + "# REFERENCES" + refs + figs
open("manuscrito_EJCTS_v1_completo.md", "w").write(md)
abst = md.split("# ABSTRACT")[1].split("# INTRODUCTION")[0]
texto = md.split("# INTRODUCTION")[1].split("**Funding:**")[0]
print("words: abstract", len(re.sub(r"\*\*[A-Z ]+:\*\*", "", abst).split()), "| main text", len(texto.split()),
      "| references", len(refs.split()))
subprocess.run(["pandoc", "manuscrito_EJCTS_v1_completo.md", "-o", "Manuscript_EJCTS.docx"], check=True)

# ------------------------------------------------------------------ supplement
tit = pd.read_csv("supp/titles.csv", index_col=0).title
DISC = open("S7_disclaimer.txt").read().strip()
NOTES = {"S1": "Percentage of mechanical prostheses among admissions with one prosthesis type and aortic or mitral "
               "position. Valve position is derived from ICD-10 codes and is available from 2002.",
         "S3": "OR for mechanical prosthesis in patients aged 18–59 years, 2022–2025 vs 2018–2019. Multilevel model as in "
               "Table 2 (regional ORs vs Southeast). ICC, intraclass correlation coefficient; MOR, median odds ratio.",
         "S4": "Position not coded: no aortic, mitral, tricuspid or pulmonary ICD-10 code in the diagnostic fields.",
         "S7": DISC,
         "S9": "Admissions with the corresponding SIGTAP procedure code, including procedures billed within "
               "multiple-procedure admissions."}
partes = [f"---\ntitle: \"Supplementary Material\"\nsubtitle: \"{TITLE}\"\n---\n", open("supl_metodos.md").read()]
partes.append("# Supplementary Figure\n\n**Figure S2.** Proportion of mechanical prostheses in patients aged 15–44 years "
              "by sex, aortic and mitral position, 2002–2025.\n\n![](Figure4_reproductive_age_by_sex.png)")
partes.append("# Supplementary Tables")
for k in ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]:
    df = pd.read_csv(f"supp/{k}.csv").fillna("")
    tb = df.to_markdown(index=False, floatfmt=".1f") if k in ("S1", "S2", "S4", "S6") else tab(df)
    partes.append(f"**{tit[k]}**\n\n" + tb + ("\n\n" + NOTES[k] if k in NOTES else ""))
open("suplemento_EJCTS_v1.md", "w").write("\n\n".join(partes))
subprocess.run(["pandoc", "suplemento_EJCTS_v1.md", "-o", "Supplementary_Material_EJCTS.docx"], check=True)


# ------------------------------------------------------------------ EJCTS formatting
def campo_pagina(par):
    r = par.add_run()
    for tipo, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tipo:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), tipo)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = txt
        r._r.append(e)


def formatar(path, duplo, numerar_linhas):
    d = Document(path)
    for st in d.styles:
        if st.type == 1 and hasattr(st, "font"):
            st.font.name = "Calibri"
            st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri") if st.element.rPr is not None and st.element.rPr.rFonts is not None else None
            st.font.color.rgb = None
    for nome, tam in (("Normal", 11), ("Body Text", 11), ("First Paragraph", 11), ("Compact", 9), ("Title", 14),
                      ("Subtitle", 12), ("Heading 1", 12), ("Heading 2", 11)):
        try:
            st = d.styles[nome]
        except KeyError:
            continue
        if True:
            st.font.size = Pt(tam)
            st.font.name = "Calibri"
            if nome.startswith("Heading") or nome in ("Title",):
                st.font.bold = True
            pf = st.paragraph_format
            if nome in ("Normal", "Body Text", "First Paragraph"):
                pf.line_spacing = 2.0 if duplo else 1.15
                pf.space_after = Pt(0 if duplo else 4)
            if nome == "Compact":
                pf.line_spacing = 1.0
    for t in d.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(0)
                    for r in p.runs:
                        r.font.size = Pt(9)
    for s in d.sections:
        if numerar_linhas:
            ln = OxmlElement("w:lnNumType")
            ln.set(qn("w:countBy"), "1")
            ln.set(qn("w:restart"), "continuous")
            s._sectPr.append(ln)
        fp = s.footer.paragraphs[0] if s.footer.paragraphs else s.footer.add_paragraph()
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        campo_pagina(fp)
    d.save(path)


formatar("Manuscript_EJCTS.docx", duplo=True, numerar_linhas=True)
formatar("Supplementary_Material_EJCTS.docx", duplo=False, numerar_linhas=False)
for f in ("manuscrito_EJCTS_v1.docx", "suplemento_EJCTS_v1.docx", "Tabelas_EJCTS.xlsx"):
    if os.path.exists(f):
        os.remove(f)
print("ok")
