#!/usr/bin/env python3
"""Supplementary tables S1–S9 in English (final labels) from the analysis outputs, and Tables_EJCTS.xlsx.

Run after manuscrito_tabelas.py. Output: protese_valvar/manuscrito/supp/ (CSV) and protese_valvar/manuscrito/Tables_EJCTS.xlsx.
"""
import os

import pandas as pd

PV = "protese_valvar"
M = f"{PV}/manuscrito"
O = f"{M}/supp"
os.makedirs(O, exist_ok=True)
VAL = {"aórtica": "Aortic", "mitral": "Mitral"}
SC = {"principal": "Main analysis", "sem_cid_padrao": "Excluding hospital-years with a default ICD code (≥70%)",
      "sem_opm_baixa": "Excluding hospitals with <80% OPM recording", "sem_2008_09": "Excluding 2008–2009",
      "sem_endocardite": "Excluding endocarditis", "todas": "All four exclusions combined"}
S = {}

# S1
s = pd.read_parquet("saida_protese/serie_2000_2025.parquet")
s = s[(s.protese != "ambas") & s.valva.isin(["aórtica", "mitral"]) & (s.idade >= 18) & (s.ano >= 2002)].copy()
s["Period"] = pd.cut(s.ano, [2001, 2007, 2012, 2017, 2019, 2021, 2025],
                     labels=["2002–2007", "2008–2012", "2013–2017", "2018–2019", "2020–2021", "2022–2025"])
s["Age, years"] = pd.cut(s.idade, [18, 40, 50, 60, 65, 70, 200], right=False,
                         labels=["18–39", "40–49", "50–59", "60–64", "65–69", "≥70"])
s["Valve"] = s.valva.map(VAL)
f = lambda g: round(100 * (g.protese == "mec").mean(), 1)  # noqa: E731
partes = []
for sx, lab in ((None, "All"), ("F", "Women"), ("M", "Men")):
    g = s if sx is None else s[s.sexo == sx]
    t = g.groupby(["Valve", "Age, years", "Period"], observed=True).apply(f, include_groups=False).unstack()
    partes.append(t.reset_index().assign(Sex=lab))
S["S1"] = pd.concat(partes)[["Sex", "Valve", "Age, years"] + list(s.Period.cat.categories)]

# S2
t = pd.read_csv(f"{PV}/resultados_serie/T1_serie_ano.csv")
S["S2"] = pd.DataFrame({"Year": t.ano, "Admissions (all)": t.AIH_total_com_ambas,
                        "Admissions with one prosthesis type": t.AIH, "Mechanical, %": t.pct_mec,
                        "Valve position not coded, %": t.pct_valva_indet, "Mean age, years": t.idade_media})

# S3
o = pd.read_csv(f"{PV}/resultados_sensibilidade/S2_or_2022_25.csv")
m = pd.read_csv(f"{PV}/resultados_sensibilidade/S3_multinivel_regiao.csv")
x = o.merge(m, on="cenario")
S["S3"] = pd.DataFrame({"Scenario": x.cenario.map(SC), "Admissions aged 18–59 y (2008–2025)": x.n,
                        "OR 2022–2025 vs 2018–2019 (95% CI)": x.OR_2022_25_vs_2018_19.astype(str) + " (" + x.IC95 + ")",
                        "Hospitals in multilevel model": x.hospitais, "MOR": x.MOR, "ICC": x.ICC,
                        "OR South": x.OR_Sul_misto, "OR Northeast": x.OR_Nordeste_misto, "OR North": x.OR_Norte_misto,
                        "OR Centre-West": x["OR_Centro-Oeste_misto"]})

# S4
t = pd.read_csv(f"{PV}/resultados_auditoria/A1_valva_pct_por_ano.csv")
S["S4"] = t.rename(columns={"ano": "Year", "aórtica": "Aortic, %", "mitral": "Mitral, %",
                            "dupla/múltipla": "Two or more prostheses, %", "indeterminada": "Position not coded, %",
                            "múltipla": "Multivalve code (I08), %", "tric/pulm": "Tricuspid/pulmonary, %"})[
    ["Year", "Aortic, %", "Mitral, %", "Two or more prostheses, %", "Multivalve code (I08), %",
     "Tricuspid/pulmonary, %", "Position not coded, %"]]

# S5
t = pd.read_csv(f"{PV}/resultados_v2/F2_hospital_troca_fornecedor.csv")
g = t.groupby("trocou").delta_pp
S["S5"] = pd.DataFrame({"Main supplier of mechanical prostheses": ["Unchanged", "Changed"],
                        "Hospitals": [int(g.size()[False]), int(g.size()[True])],
                        "Change in mechanical share, percentage points, mean (SD)":
                            [f"{g.mean()[k]:+.1f} ({g.std()[k]:.1f})" for k in (False, True)],
                        "Median (IQR)": [f"{g.median()[k]:+.1f} ({g.quantile(.25)[k]:+.1f} to {g.quantile(.75)[k]:+.1f})"
                                         for k in (False, True)]})

# S6
t = pd.read_csv(f"{PV}/resultados_artigo1/E2_volume_ano_18_59.csv")
S["S6"] = pd.DataFrame({"Year": t.ano_i, "Biological": t.bio, "Mechanical": t.mec, "Total": t.total,
                        "Mechanical, %": t.pct_mec})

# S7
r = pd.read_csv(f"{PV}/resultados_sensibilidade/S4_reoperacao_hr.csv")
r = r[r.cenario == "principal"]
k = pd.read_csv(f"{PV}/resultados_complicacoes/K1_complicacoes_18_64.csv")
OUT = {"sangramento": "Major bleeding", "intracraniano": "Intracranial bleeding", "tromboembolismo": "Thromboembolism",
       "protese": "Prosthesis complication", "obito_hosp": "Death in a later admission"}
lin = [{"Outcome": f"Valve reintervention, {VAL[v.valva].lower()} position", "Mechanical: patients / events": "",
        "Mechanical: 5-y / 10-y, %": "", "Biological: patients / events": f"{int(v.n):,} / {int(v.reop):,} (both types)",
        "Biological: 5-y / 10-y, %": "", "HR biological vs mechanical (95% CI)": f"{v.HR_bio_vs_mec} ({v.IC95})"}
       for v in r.itertuples()]
for v in k.itertuples():
    lin.append({"Outcome": OUT[v.desfecho], "Mechanical: patients / events": f"{int(v.mec_n):,} / {int(v.mec_eventos):,}",
                "Mechanical: 5-y / 10-y, %": f"{v.mec_5a} / {v.mec_10a}",
                "Biological: patients / events": f"{int(v.bio_n):,} / {int(v.bio_eventos):,}",
                "Biological: 5-y / 10-y, %": f"{v.bio_5a} / {v.bio_10a}",
                "HR biological vs mechanical (95% CI)": f"{v.HR_bio_vs_mec} ({v.IC95})"})
S["S7"] = pd.DataFrame(lin)

# S8
t = pd.read_csv(f"{M}/S8_included_vs_excluded.csv")
S["S8"] = t.rename(columns={"grupo": "Group", "n": "Admissions", "age_median": "Median age, years",
                            "female_pct": "Women, %", "mechanical_pct": "Mechanical, %",
                            "in_hospital_death_pct": "In-hospital death, %", "south_pct": "South region, %",
                            "northeast_pct": "Northeast region, %"})

# S9
t = pd.read_csv(f"{PV}/resultados_v2/C2_aorta_ross_plastica_corrigido_ano.csv")
t = t[t.ano.between(2008, 2025)]
S["S9"] = pd.DataFrame({"Year": t.ano, "Valved conduit (aortic root)": t.tubo_valvado, "Ross procedure": t.ross,
                        "Valve repair": t.plastica})
tot = S["S9"].sum(numeric_only=True)
S["S9"] = pd.concat([S["S9"].astype({"Year": str}), pd.DataFrame([{"Year": "Total", **tot.drop("Year").to_dict()}])])

TIT = {"S1": "Table S1. Mechanical prostheses (%) by sex, valve position, age group and period, adults, 2002–2025",
       "S2": "Table S2. Annual series, 2000–2025",
       "S3": "Table S3. Sensitivity analyses of the main findings",
       "S4": "Table S4. Valve position derived from ICD-10 codes, by year, 2008–2025 (% of admissions)",
       "S5": "Table S5. Change in the mechanical share in patients aged 18–59 years between 2016–2019 and 2021–2025, "
             "by change of the hospital's main supplier of mechanical prostheses (127 hospitals with ≥30 admissions "
             "in each period)",
       "S6": "Table S6. Annual implants by prosthesis type, patients aged 18–59 years, aortic and mitral, 2008–2025",
       "S7": "Table S7. Exploratory: reintervention and complications captured in later SUS admissions, patients aged "
             "18–64 years",
       "S8": "Table S8. Included and excluded admissions, 2008–2025",
       "S9": "Table S9. Other valve procedures billed in the SUS, 2008–2025 (admissions)"}
for k_, df in S.items():
    df.to_csv(f"{O}/{k_}.csv", index=False)
pd.Series(TIT).to_csv(f"{O}/titles.csv", header=["title"])
with pd.ExcelWriter(f"{M}/Tables_EJCTS.xlsx", engine="xlsxwriter") as xw:
    pd.DataFrame({"Sheet": ["Table1", "Table2", "Table3"] + list(S),
                  "Content": ["Characteristics of the analysis cohort by prosthesis type, 2008–2025",
                              "Determinants of mechanical-prosthesis use (mixed-effects logistic model)",
                              "Change after 2020: main and sensitivity analyses"] + [TIT[k_] for k_ in S]}
                 ).to_excel(xw, sheet_name="Contents", index=False)
    pd.read_csv(f"{M}/Table1_baseline.csv").to_excel(xw, sheet_name="Table1", index=False)
    pd.read_csv(f"{M}/Table2_multilevel.csv").to_excel(xw, sheet_name="Table2", index=False)
    pd.read_csv(f"{M}/Table3_2020_shift_robustness.csv").to_excel(xw, sheet_name="Table3", index=False)
    for k_, df in S.items():
        df.to_excel(xw, sheet_name=k_, index=False)
for k_ in S:
    print(k_, S[k_].shape)
print(S["S5"].to_string(index=False))
print(S["S1"][(S["S1"].Sex == "All") & (S["S1"]["Age, years"] == "18–39")].to_string(index=False))
