#!/usr/bin/env python3
"""Tabelas, figuras e suplemento do manuscrito (EJCTS): escolha da prótese valvar no SUS, 2000–2025.

Coorte principal (modelos): AIH 2008–2025 com OPM de prótese valvar, um só tipo de prótese, aórtica ou mitral,
≥18 anos. Série descritiva: 2000–2025 (serie_2000_2025.py; posição valvar confiável a partir de 2002).
Saídas: protese_valvar/manuscrito/ (CSV, PNG e Tabelas_EJCTS.xlsx). Usa resultados já gerados por analises_v2,
artigo1_extra, sensibilidades_v1, auditoria_v1, reoperacao e complicacoes.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

M = "protese_valvar/manuscrito"
os.makedirs(M, exist_ok=True)
PV = "protese_valvar"
AZ, LA = "#1f6f8b", "#c9733a"

a = pd.read_csv("saida_protese/aih_protese_classificada.csv", dtype=str, keep_default_na=False)
a["idade"] = pd.to_numeric(a.idade, errors="coerce")
a["mec"] = (a.mec == "True").astype(int)
a["ano_i"] = a.ano.astype(int)
a["regiao"] = a.uf.str[0].map({"1": "North", "2": "Northeast", "3": "Southeast", "4": "South", "5": "Centre-West"})
a["urgencia"] = (a.rd_CAR_INT != "01").astype(int)
a["obito"] = (a.rd_MORTE == "1").astype(int)
a["dias"] = pd.to_numeric(a.rd_DIAS_PERM, errors="coerce")

# ------------------------------------------------------------------ fluxograma
f = [("AIH with a valve-prosthesis OPM code, 2008–2025", len(a))]
x = a[a.protese != "ambas"]
f.append(("Excluded: both biological and mechanical prostheses billed", len(a) - len(x)))
y = x[x.valva.isin(["aórtica", "mitral"])]
for v, n in x[~x.valva.isin(["aórtica", "mitral"])].valva.value_counts().items():
    f.append((f"Excluded: valve position {v}", n))
z = y[y.idade >= 18]
f.append(("Excluded: age <18 years or missing", len(y) - len(z)))
f.append(("Analysis cohort (adults, isolated aortic or mitral position, one prosthesis type)", len(z)))
pd.DataFrame(f, columns=["step", "n"]).to_csv(f"{M}/Flowchart_counts.csv", index=False)
c = z.copy()

# ------------------------------------------------------------------ Tabela 1
ETI = {"reumática": "Rheumatic (coded)", "não reumática": "Non-rheumatic", "congênita": "Congenital",
       "endocardite": "Endocarditis", "indeterminada": "Not specified"}
c["fx"] = pd.cut(c.idade, [18, 40, 50, 60, 65, 70, 200], right=False,
                 labels=["18–39", "40–49", "50–59", "60–64", "65–69", "≥70"])
c["per"] = pd.cut(c.ano_i, [2007, 2012, 2017, 2019, 2021, 2025],
                  labels=["2008–2012", "2013–2017", "2018–2019", "2020–2021", "2022–2025"])
vol = c.groupby(["cnes", "ano"]).size().rename("vol_ano")
c = c.join(vol, on=["cnes", "ano"])
c["vol"] = pd.cut(c.vol_ano, [0, 25, 50, 100, 10000], labels=["<25", "25–49", "50–99", "≥100"])


def linha_cat(var, lab, mapa=None):
    out = [{"Characteristic": lab, "All": "", "Biological": "", "Mechanical": ""}]
    for k in (mapa or {v: v for v in c[var].dropna().unique()}):
        r = {"Characteristic": "  " + (mapa[k] if mapa else str(k))}
        for col, g in (("All", c), ("Biological", c[c.mec == 0]), ("Mechanical", c[c.mec == 1])):
            n = (g[var] == k).sum()
            r[col] = f"{n:,} ({100 * n / len(g):.1f})"
        out.append(r)
    return out


t1 = [{"Characteristic": "Admissions, n", "All": f"{len(c):,}", "Biological": f"{(c.mec == 0).sum():,}",
       "Mechanical": f"{(c.mec == 1).sum():,}"}]
r = {"Characteristic": "Age, years, median (IQR)"}
for col, g in (("All", c), ("Biological", c[c.mec == 0]), ("Mechanical", c[c.mec == 1])):
    q = g.idade.quantile([.5, .25, .75])
    r[col] = f"{q.iloc[0]:.0f} ({q.iloc[1]:.0f}–{q.iloc[2]:.0f})"
t1.append(r)
t1 += linha_cat("fx", "Age group, n (%)", {k: k for k in ["18–39", "40–49", "50–59", "60–64", "65–69", "≥70"]})
t1 += linha_cat("sexo", "Sex, n (%)", {"F": "Female", "M": "Male"})
t1 += linha_cat("valva", "Valve position, n (%)", {"aórtica": "Aortic", "mitral": "Mitral"})
t1 += linha_cat("etiologia", "Aetiology (ICD-10), n (%)", ETI)
t1 += linha_cat("urgencia", "Admission type, n (%)", {0: "Elective", 1: "Urgent/emergency"})
t1 += linha_cat("regiao", "Region, n (%)", {k: k for k in ["North", "Northeast", "Southeast", "South", "Centre-West"]})
t1 += linha_cat("vol", "Hospital annual volume, n (%)", {k: k for k in ["<25", "25–49", "50–99", "≥100"]})
t1 += linha_cat("per", "Period, n (%)", {k: k for k in ["2008–2012", "2013–2017", "2018–2019", "2020–2021", "2022–2025"]})
r = {"Characteristic": "In-hospital death, n (%)"}
for col, g in (("All", c), ("Biological", c[c.mec == 0]), ("Mechanical", c[c.mec == 1])):
    r[col] = f"{g.obito.sum():,} ({100 * g.obito.mean():.1f})"
t1.append(r)
T1 = pd.DataFrame(t1)
T1.to_csv(f"{M}/Table1_baseline.csv", index=False)

# ------------------------------------------------------------------ Tabela 2: modelo multinível
d = c.dropna(subset=["sexo", "regiao"]).copy()
nh = d.cnes.value_counts()
d = d[d.cnes.isin(nh[nh >= 30].index)].copy()
d["fx2"] = d.fx.astype(str)
d["per2"] = d.per.astype(str)
d["vol2"] = d.vol.astype(str)
form = ("mec ~ C(fx2, Treatment('60–64')) + C(sexo, Treatment('F')) + C(valva) + "
        "C(etiologia, Treatment('não reumática')) + urgencia + C(per2, Treatment('2018–2019')) + "
        "C(regiao, Treatment('Southeast')) + C(vol2, Treatment('<25'))")
md = BinomialBayesMixedGLM.from_formula(form, {"hosp": "0 + C(cnes)"}, d).fit_vb()
fe = smf.logit(form, data=d).fit(disp=0)
sd = float(np.exp(md.vcp_mean[0]))
t2 = pd.DataFrame({"term": fe.params.index, "OR": np.exp(md.fe_mean),
                   "lo": np.exp(md.fe_mean - 1.96 * md.fe_sd), "hi": np.exp(md.fe_mean + 1.96 * md.fe_sd)})
t2 = t2[t2.term != "Intercept"]
t2["OR (95% CrI)"] = t2.apply(lambda r: f"{r.OR:.2f} ({r.lo:.2f}–{r.hi:.2f})", axis=1)
mor = np.exp(np.sqrt(2) * sd * 0.6745)
icc = sd ** 2 / (sd ** 2 + np.pi ** 2 / 3)
t2.to_csv(f"{M}/Table2_multilevel.csv", index=False)
with open(f"{M}/Table2_multilevel.csv", "a") as fh:
    fh.write(f"hospital random-intercept SD,{sd:.2f},,,\nMOR,{mor:.2f},,,\nICC,{icc:.2f},,,\n"
             f"admissions,{len(d)},,,\nhospitals,{d.cnes.nunique()},,,\n")

# ------------------------------------------------------------------ Tabela 3: mudança de 2020 (robustez)
e1 = pd.read_csv(f"{PV}/resultados_artigo1/E1_its.csv")
e3 = pd.read_csv(f"{PV}/resultados_artigo1/E3_E5_or_2022_25.csv")
s2 = pd.read_csv(f"{PV}/resultados_sensibilidade/S2_or_2022_25.csv")
lin = []
for _, r in e1.iterrows():
    lab = ("Interrupted time series, level change at " + r.quebra[:7]
           + (" (placebo, pre-2020 series)" if r.quebra < "2020" else ""))
    lin.append({"Analysis": lab, "Estimate": "OR", "Value (95% CI)": f"{r.OR_nivel:.2f} ({r.IC95_HAC.replace('–', '–')})",
                "P": f"{r.p_nivel:.3f}"})
NOMES = {"principal (região, sem hospital)": "Main model (2022–2025 vs 2018–2019)",
         "efeito fixo de hospital (hospitais ativos nos 2 períodos)": "Hospital fixed effects (hospitals active in both periods)",
         "hospitais com CID plausível (nunca padrão; indeterminada <20%)": "Hospitals with plausible ICD coding",
         "etiologia reumática": "Rheumatic aetiology only", "etiologia não reumática": "Non-rheumatic aetiology only"}
for _, r in e3.iterrows():
    lin.append({"Analysis": NOMES.get(r.analise, r.analise), "Estimate": "OR",
                "Value (95% CI)": f"{r.OR_2022_25_vs_2018_19:.2f} ({r.IC95})", "P": ""})
SN = {"sem_cid_padrao": "Excluding hospital-years with a default ICD code (≥70%)",
      "sem_opm_baixa": "Excluding hospitals with <80% OPM recording", "sem_2008_09": "Excluding 2008–2009",
      "sem_endocardite": "Excluding endocarditis", "todas": "All four exclusions combined"}
for _, r in s2[s2.cenario != "principal"].iterrows():
    lin.append({"Analysis": SN[r.cenario], "Estimate": "OR", "Value (95% CI)": f"{r.OR_2022_25_vs_2018_19:.2f} ({r.IC95})",
                "P": ""})
T3 = pd.DataFrame(lin)
T3.to_csv(f"{M}/Table3_2020_shift_robustness.csv", index=False)

# ------------------------------------------------------------------ Figuras
s = pd.read_parquet("saida_protese/serie_2000_2025.parquet")
u = s[(s.protese != "ambas") & s.valva.isin(["aórtica", "mitral"]) & (s.idade >= 18)]


def pct(g):
    return 100 * (g.protese == "mec").mean()


fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for i, (v, vn) in enumerate((("aórtica", "Aortic"), ("mitral", "Mitral"))):
    for (lo, hi, lab, cor, ls, mk) in ((18, 60, "18–59 years", AZ, "-", "o"), (60, 200, "≥60 years", LA, "--", "s")):
        g = u[(u.valva == v) & (u.idade >= lo) & (u.idade < hi) & (u.ano >= 2002)]
        sr = g.groupby("ano").apply(pct, include_groups=False)
        ax[i].plot(sr.index, sr.values, marker=mk, ms=3, color=cor, ls=ls, label=lab)
    ax[i].axvline(2007.5, color="grey", ls=":", lw=1)
    ax[i].axvline(2020, color="grey", ls="--", lw=1)
    ax[i].set_title(f"{vn} position")
    ax[i].set_ylim(0, 70)
    ax[i].legend(frameon=False, fontsize=8)
ax[0].set_ylabel("Mechanical prostheses, %")
fig.tight_layout()
fig.savefig(f"{M}/Figure1_series_2002_2025.png", dpi=300)
plt.close(fig)

q = pd.read_csv(f"{PV}/resultados_v2/Q1_mensal_18_59.csv", parse_dates=["mes"])
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(q.mes, 100 * q.pct, color="#bbbbbb", lw=0.8, label="Monthly")
ax.plot(q.mes, 100 * q.pct.rolling(12, center=True).mean(), color=AZ, lw=2, label="12-month moving average")
ax.axvline(pd.Timestamp("2020-05-01"), color="k", ls="--", lw=1)
ax.set_ylabel("Mechanical prostheses, %")
ax.set_title("Patients aged 18–59 years, aortic and mitral, 2008–2025")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{M}/Figure2_monthly_18_59.png", dpi=300)
plt.close(fig)

fu = pd.read_csv(f"{PV}/resultados_v2/M2_funnel_18_59_2020_2025.csv", dtype={"cnes": str})
p0 = (fu.obs * fu.n).sum() / fu.n.sum()
nn = np.linspace(fu.n.min(), fu.n.max(), 200)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(fu.n, 100 * fu.ajust, s=12, color=AZ, alpha=.7)
for z_, ls in ((1.96, ":"), (3.09, "--")):
    ax.plot(nn, 100 * (p0 + z_ * np.sqrt(p0 * (1 - p0) / nn)), "k", ls=ls, lw=.8)
    ax.plot(nn, 100 * (p0 - z_ * np.sqrt(p0 * (1 - p0) / nn)), "k", ls=ls, lw=.8)
ax.axhline(100 * p0, color="k", lw=.8)
ax.set_xlabel("Admissions per hospital, 2020–2025 (patients aged 18–59 years)")
ax.set_ylabel("Case-mix-adjusted mechanical prostheses, %")
fig.tight_layout()
fig.savefig(f"{M}/Figure3_funnel.png", dpi=300)
plt.close(fig)

# mulheres 15–44 (estratificação por sexo exigida pela EJCTS)
w = s[(s.protese != "ambas") & s.valva.isin(["aórtica", "mitral"]) & (s.ano >= 2002) & (s.idade >= 15) & (s.idade < 45)]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for (sx, lab, cor) in (("F", "Women 15–44", "#8b1f5a"), ("M", "Men 15–44", AZ)):
    g = w[w.sexo == sx].groupby("ano").apply(pct, include_groups=False)
    ax.plot(g.index, g.values, marker="o", ms=3, color=cor, label=lab)
ax.axvline(2020, color="grey", ls="--", lw=1)
ax.set_ylim(0, 70)
ax.set_ylabel("Mechanical prostheses, %")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{M}/Figure4_reproductive_age_by_sex.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ suplemento
sup = {}
per = pd.cut(s.ano, [1999, 2007, 2012, 2017, 2019, 2021, 2025],
             labels=["2000–2007", "2008–2012", "2013–2017", "2018–2019", "2020–2021", "2022–2025"])
s["per"] = per
ss = s[(s.protese != "ambas") & s.valva.isin(["aórtica", "mitral"]) & (s.idade >= 18) & (s.ano >= 2002)].copy()
ss["fx"] = pd.cut(ss.idade, [18, 40, 50, 60, 65, 70, 200], right=False,
                  labels=["18–39", "40–49", "50–59", "60–64", "65–69", "≥70"])
sup["S1_pct_mech_by_sex_age_valve"] = (ss.groupby(["sexo", "valva", "fx", "per"], observed=True)
                                       .apply(pct, include_groups=False).round(1).unstack())
sup["S2_annual_series_2000_2025"] = pd.read_csv(f"{PV}/resultados_serie/T1_serie_ano.csv")
sup["S3_sensitivity_main_findings"] = pd.concat(
    [pd.read_csv(f"{PV}/resultados_sensibilidade/S2_or_2022_25.csv").assign(estimate="OR 2022–25 vs 2018–19"),
     pd.read_csv(f"{PV}/resultados_sensibilidade/S3_multinivel_regiao.csv").assign(estimate="multilevel")])
sup["S4_valve_classification_audit"] = pd.read_csv(f"{PV}/resultados_auditoria/A1_valva_pct_por_ano.csv")
sup["S5_supplier_switch"] = pd.read_csv(f"{PV}/resultados_v2/F2_hospital_troca_fornecedor.csv").describe().round(2)
sup["S6_volume_18_59"] = pd.read_csv(f"{PV}/resultados_artigo1/E2_volume_ano_18_59.csv")
k1 = pd.read_csv(f"{PV}/resultados_complicacoes/K1_complicacoes_18_64.csv")
ro = pd.read_csv(f"{PV}/resultados_sensibilidade/S4_reoperacao_hr.csv")
ro = ro[ro.cenario == "principal"]
sup["S7_SUS_captured_events_DISCLAIMER"] = pd.concat([
    ro.rename(columns={"HR_bio_vs_mec": "HR_bio_vs_mech"}).assign(outcome="valve reintervention (SUS-captured), 18–64 y"),
    k1.rename(columns={"desfecho": "outcome", "HR_bio_vs_mec": "HR_bio_vs_mech"})], ignore_index=True)
DISC = ("EXPLORATORY. Events captured only in SUS hospital admissions; persons linked by a deterministic pseudonymous key "
        "(date of birth + sex + postal code; 40% generic postal codes) without external validation; deaths outside "
        "hospital and care in the private sector are not observed, so cumulative incidences do not represent "
        "population risks. Not used for any conclusion.")
for k, v in sup.items():
    v.to_csv(f"{M}/{k}.csv")

with pd.ExcelWriter(f"{M}/Tabelas_EJCTS.xlsx", engine="xlsxwriter") as xw:
    leia = pd.DataFrame({"Item": ["Fonte", "Coorte", "Série", "Tabelas principais", "S7", "Gerado por"],
                         "Descrição": ["DATASUS SIH-RD + SIH-SP (espelho healthbr-data); 2000–2007 pela TUNEP/ANS",
                                       f"{len(c):,} AIH 2008–2025 (fluxograma)",
                                       "2000–2025; posição valvar a partir de 2002",
                                       "Table1–3; Figure1–4 em PNG", DISC, "protese_valvar/manuscrito_tabelas.py"]})
    leia.to_excel(xw, sheet_name="Leia-me", index=False)
    pd.DataFrame(f, columns=["step", "n"]).to_excel(xw, sheet_name="Flowchart", index=False)
    T1.to_excel(xw, sheet_name="Table1", index=False)
    pd.read_csv(f"{M}/Table2_multilevel.csv").to_excel(xw, sheet_name="Table2", index=False)
    T3.to_excel(xw, sheet_name="Table3", index=False)
    for k, v in sup.items():
        v.to_excel(xw, sheet_name=k[:31])
    ws = xw.sheets["S7_SUS_captured_events_DISCLAIMER"[:31]]
    ws.write(0, 12, DISC)
print(pd.DataFrame(f, columns=["step", "n"]).to_string(index=False))
print(T1.to_string(index=False))
print(t2[["term", "OR (95% CrI)"]].to_string(index=False), f"\nMOR {mor:.2f} ICC {icc:.2f} hospitals {d.cnes.nunique()}")
print(T3.to_string(index=False))

# ------------------------------------------------------------------ revisão v1.2: incluídos × excluídos; fluxograma
g = a.copy()
g["grupo"] = np.select(
    [g.protese == "ambas", ~g.valva.isin(["aórtica", "mitral"]), g.idade.isna() | (g.idade < 18)],
    ["Excluded: both types", "Excluded: position not aortic/mitral", "Excluded: <18 y or missing age"],
    "Included")
g.loc[(g.grupo == "Excluded: position not aortic/mitral"), "grupo"] = "Excluded: " + g.valva.map(
    {"indeterminada": "position not coded", "dupla/múltipla": "two or more prostheses", "tric/pulm": "tricuspid/pulmonary",
     "múltipla": "multivalve code (I08)"}).fillna("other position")
s8 = g.groupby("grupo").agg(n=("mec", "size"), age_median=("idade", "median"),
                            female_pct=("sexo", lambda x: round(100 * (x == "F").mean(), 1)),
                            mechanical_pct=("protese", lambda x: round(100 * (x == "mec").mean(), 1)),
                            in_hospital_death_pct=("obito", lambda x: round(100 * x.mean(), 1)),
                            south_pct=("regiao", lambda x: round(100 * (x == "South").mean(), 1)),
                            northeast_pct=("regiao", lambda x: round(100 * (x == "Northeast").mean(), 1)))
s8.to_csv(f"{M}/S8_included_vs_excluded.csv")
print(s8.to_string())
# fluxograma: figura_fluxograma.py
