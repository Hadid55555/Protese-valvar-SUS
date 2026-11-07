#!/usr/bin/env python3
"""Monta o manuscrito completo (tabelas + figuras embutidas) e o suplemento em .docx (pandoc)."""
import os, re, subprocess
import pandas as pd
os.chdir("protese_valvar/manuscrito")
md = open("manuscrito_EJCTS_v1.md").read()
tab = lambda df: df.to_markdown(index=False)
t1 = pd.read_csv("Table1_baseline.csv").fillna("")
t2 = pd.read_csv("Table2_multilevel.csv")
main = t2.dropna(subset=["OR (95% CrI)"])
def lab(t):
    for a, b in [("C(fx2, Treatment('60–64'))[T.", "Age "), ("C(sexo, Treatment('F'))[T.M]", "Male sex"),
                 ("C(valva)[T.mitral]", "Mitral position"), ("C(etiologia, Treatment('não reumática'))[T.", "Aetiology: "),
                 ("C(per2, Treatment('2018–2019'))[T.", "Period "), ("C(regiao, Treatment('Southeast'))[T.", "Region: "),
                 ("C(vol2, Treatment('<25'))[T.", "Hospital volume/year "), ("urgencia", "Urgent admission"),
                 ("reumática", "rheumatic"), ("congênita", "congenital"), ("endocardite", "endocarditis"),
                 ("indeterminada", "not specified")]:
        t = t.replace(a, b)
    return t.rstrip("]")
m2 = pd.DataFrame({"Variable": main.term.map(lab), "OR (95% CrI)": main["OR (95% CrI)"]})
foot = t2[t2["OR (95% CrI)"].isna()][["term", "OR"]]
note = ("References: age 60–64 years; female; aortic; non-rheumatic; 2018–2019; Southeast; <25 admissions/year. "
        + "; ".join(f"{r.term} {int(r.OR) if r.term in ('admissions', 'hospitals') else r.OR}" for r in foot.itertuples()))
t3 = pd.read_csv("Table3_2020_shift_robustness.csv", dtype=str).fillna("")
t3["P"] = t3.P.replace({"0.000": "<0.001"})
t3 = t3.drop(columns="Estimate").rename(columns={"Value (95% CI)": "OR (95% CI)"})
md = md.replace("Table 1. Characteristics of the analysis cohort by prosthesis type, 2008–2025 (Table1_baseline.csv).",
                "**Table 1.** Characteristics of the analysis cohort by prosthesis type, 2008–2025.\n\n" + tab(t1) + "\n")
md = md.replace("Table 2. Determinants of mechanical-prosthesis use: mixed-effects logistic model (Table2_multilevel.csv).",
                "**Table 2.** Determinants of mechanical-prosthesis use, adults 2008–2025 (mixed-effects logistic model, "
                "hospital random intercept).\n\n" + tab(m2) + "\n\n" + note + "\n")
md = re.sub(r"Table 3\. Change in mechanical-prosthesis use after 2020 in patients aged 18–59 years: main and sensitivity analyses\n\(Table3_2020_shift_robustness\.csv\)\.",
            lambda _: "**Table 3.** Change in mechanical-prosthesis use after 2020, patients aged 18–59 years.\n\n" + tab(t3) + "\n", md)
figs = "\n\n# FIGURES (embedded for review)\n\n" + "\n\n".join(
    f"![{c}]({f})" for f, c in [("Figure1_flowchart.png", "Figure 1"), ("Figure1_series_2002_2025.png", "Figure 2"),
                                ("Figure2_monthly_18_59.png", "Figure 3"), ("Figure3_funnel.png", "Figure 4")])
md = md.replace("# REFERENCES", figs + "\n\n# REFERENCES")
open("manuscrito_EJCTS_v1_completo.md", "w").write(md)
abst = md.split("# ABSTRACT")[1].split("# INTRODUCTION")[0]
corpo = md.split("# INTRODUCTION")[1].split("# FIGURE LEGENDS")[0]
refs = md.split("# REFERENCES")[1]
print("palavras: resumo", len(abst.split()), "| texto", len(corpo.split()), "| referências", len(refs.split()))
subprocess.run(["pandoc", "manuscrito_EJCTS_v1_completo.md", "-o", "manuscrito_EJCTS_v1.docx"], check=True)
# suplemento
DISC = open("S7_disclaimer.txt").read() if os.path.exists("S7_disclaimer.txt") else ""
partes = ["---\ntitle: Supplementary Material\n---\n"]
partes.append("# Figure S1. Flowchart counts\n\n" + tab(pd.read_csv("Flowchart_counts.csv")))
partes.append("# Figure S2. Mechanical prostheses in patients aged 15–44 years by sex, 2002–2025\n\n![Figure S2](Figure4_reproductive_age_by_sex.png)")
nomes = {"S1_pct_mech_by_sex_age_valve": "Table S1. Mechanical prostheses (%) by sex, valve position, age group and period, 2002–2025",
         "S2_annual_series_2000_2025": "Table S2. Annual series 2000–2025",
         "S3_sensitivity_main_findings": "Table S3. Sensitivity analyses of the main findings",
         "S4_valve_classification_audit": "Table S4. Valve-position classification by year (audit)",
         "S5_supplier_switch": "Table S5. Change in mechanical share (percentage points) by hospital supplier switch, 2016–19 vs 2021–25",
         "S6_volume_18_59": "Table S6. Annual implants by prosthesis type, patients aged 18–59 years",
         "S8_included_vs_excluded": "Table S8. Included vs excluded admissions, 2008–2025",
         "S7_SUS_captured_events_DISCLAIMER": "Table S7. EXPLORATORY SUS-captured reintervention and complications (not used for conclusions)"}
for k, t in nomes.items():
    df = pd.read_csv(f"{k}.csv")
    partes.append(f"# {t}\n\n" + (DISC + "\n\n" if k.startswith("S7") else "") + tab(df.fillna("")))
partes.append(open("supl_metodos.md").read() if os.path.exists("supl_metodos.md") else "")
open("suplemento_EJCTS_v1.md", "w").write("\n\n".join(partes))
subprocess.run(["pandoc", "suplemento_EJCTS_v1.md", "-o", "suplemento_EJCTS_v1.docx"], check=True)
