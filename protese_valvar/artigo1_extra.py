#!/usr/bin/env python3
"""Artigo 1, análises pedidas pelos revisores externos (AVALIACAO_REVISORES_v1.md).

Base: igual a analises_v2 (AIH com OPM, 1 tipo de prótese, aórtica/mitral, idade conhecida), 18–59 anos.
  E1 ITS mensal (binomial, % mecânica): tendência + nível e inclinação após a quebra + mês do ano; EP HAC (12 lags).
     Quebras: mai/2020 (principal), mar/2020 e placebos (mai/2014, mai/2016, mai/2018; série até a véspera de 2020).
  E2 volume absoluto por ano (biológica, mecânica) e variação 2018–19 → 2022–25.
  E3 OR 2022–25 × 2018–19 com efeito fixo de hospital (hospitais com ≥10 AIH em cada período).
  E4 OR em hospitais com CID plausível (nunca CID padrão ≥70% em hospital-ano com ≥10 AIH; valva indeterminada <20%).
  E5 OR por etiologia codificada (reumática × não reumática).
Saídas: protese_valvar/resultados_artigo1/ (agregados).
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

R = "protese_valvar/resultados_artigo1"
os.makedirs(R, exist_ok=True)
txt = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    txt.append(s)


a = pd.read_csv("saida_protese/aih_protese_classificada.csv", dtype=str, keep_default_na=False)
a["idade"] = pd.to_numeric(a.idade, errors="coerce")
a["mec"] = (a.mec == "True").astype(int)
a["ano_i"] = a.ano.astype(int)
a["mes"] = pd.to_datetime(a.dt_inter.str[:6], format="%Y%m", errors="coerce")
a["regiao"] = a.uf.str[0].map({"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"})
a["urgencia"] = (a.rd_CAR_INT != "01").astype(int)
cid = a.rd_DIAG_PRINC.where(a.rd_DIAG_PRINC.str.len() >= 3, a.cid_pri).str[:3]
g = pd.DataFrame({"cnes": a.cnes, "ano": a.ano, "cid": cid}).groupby(["cnes", "ano"]).cid.agg(
    n="size", top=lambda s: s.value_counts(normalize=True).iat[0])
hosp_padrao = set(g[(g.n >= 10) & (g.top >= 0.70)].index.get_level_values(0))
indet = a[(a.protese != "ambas") & (a.idade >= 18)].groupby("cnes").valva.apply(lambda s: (s == "indeterminada").mean())
b = a[(a.protese != "ambas") & a.valva.isin(["aórtica", "mitral"]) & a.idade.notna()].copy()
jov = b[(b.idade >= 18) & (b.idade < 60)].copy()
jov["per"] = pd.cut(jov.ano_i, [2007, 2012, 2017, 2019, 2021, 2025],
                    labels=["2008-12", "2013-17", "2018-19", "2020-21", "2022-25"])
K = "C(per, Treatment('2018-19'))[T.2022-25]"
BASE = "mec ~ C(per, Treatment('2018-19')) + C(valva) + C(etiologia) + C(sexo) + idade + urgencia"

# ------------------------------------------------------------------ E1 ITS
m = jov.groupby("mes").agg(n=("mec", "size"), mec=("mec", "sum"))
m = m[(m.index >= "2008-01-01") & (m.index <= "2025-12-01")].copy()
m["t"] = np.arange(len(m))
m["mes_ano"] = m.index.month


def its(quebra, fim=None):
    d = m if fim is None else m[m.index < fim].copy()
    k = int(np.where(d.index == pd.Timestamp(quebra))[0][0])
    d = d.assign(pos=(d.t >= k).astype(int), tpos=(d.t - k).clip(lower=0))
    X = pd.get_dummies(d[["t", "pos", "tpos", "mes_ano"]].astype({"mes_ano": str}), columns=["mes_ano"],
                       drop_first=True, dtype=float)
    X = sm.add_constant(X)
    y = np.c_[d.mec, d.n - d.mec]
    f = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    sc = f.pearson_chi2 / f.df_resid
    lvl, lo, hi = np.exp([f.params["pos"], *f.conf_int().loc["pos"]])
    sl = np.exp(12 * f.params["tpos"])
    return {"quebra": quebra, "meses": len(d), "OR_nivel": round(lvl, 2), "IC95_HAC": f"{lo:.2f}–{hi:.2f}",
            "p_nivel": round(float(f.pvalues["pos"]), 4), "OR_inclinacao_por_ano_apos": round(float(sl), 3),
            "p_inclinacao": round(float(f.pvalues["tpos"]), 4), "superdispersao": round(float(sc), 2)}


e1 = [its("2020-05-01"), its("2020-03-01")] + [its(q, fim="2020-03-01") for q in ("2014-05-01", "2016-05-01", "2018-05-01")]
e1 = pd.DataFrame(e1)
e1.to_csv(f"{R}/E1_its.csv", index=False)
log("E1. ITS mensal, 18–59 anos, % mecânica (binomial; mês do ano; EP HAC 12 lags). Placebos só com a série pré-2020:")
log(e1.to_string(index=False))

# ------------------------------------------------------------------ E2 volume absoluto
v = jov.groupby(["ano_i", "protese"]).size().unstack()
v["total"] = v.sum(axis=1)
v["pct_mec"] = (100 * v.mec / v.total).round(1)
v.to_csv(f"{R}/E2_volume_ano_18_59.csv")
p1, p2 = v.loc[2018:2019].mean(), v.loc[2022:2025].mean()
log("\nE2. Volume anual 18–59 anos (aórtica+mitral):")
log(v.loc[2014:].to_string())
log(f"  média/ano 2018–19 → 2022–25: biológica {p1.bio:.0f} → {p2.bio:.0f} ({100 * (p2.bio / p1.bio - 1):+.0f}%); "
    f"mecânica {p1.mec:.0f} → {p2.mec:.0f} ({100 * (p2.mec / p1.mec - 1):+.0f}%); total {p1.total:.0f} → {p2.total:.0f}")

# ------------------------------------------------------------------ E3–E5
lin = []


def ajusta(nome, d, form):
    f = smf.logit(form, data=d.dropna(subset=["sexo"])).fit(disp=0, method="newton", maxiter=100)
    if not f.mle_retvals.get("converged", True):
        log(f"  [aviso] {nome}: não convergiu")
    o = np.exp([f.params[K], *f.conf_int().loc[K]])
    lin.append({"analise": nome, "AIH": int(f.nobs), "hospitais": d.cnes.nunique(), "OR_2022_25_vs_2018_19": round(o[0], 2),
                "IC95": f"{o[1]:.2f}–{o[2]:.2f}"})
    log(f"  {nome}: OR {o[0]:.2f} ({o[1]:.2f}–{o[2]:.2f}); {int(f.nobs):,} AIH, {d.cnes.nunique()} hospitais")


log("\nE3–E5. OR de mecânica 2022–25 × 2018–19, 18–59 anos:")
ajusta("principal (região, sem hospital)", jov, BASE + " + C(regiao)")
nA = jov[jov.per == "2018-19"].cnes.value_counts()
nB = jov[jov.per == "2022-25"].cnes.value_counts()
est = set(nA[nA >= 10].index) & set(nB[nB >= 10].index)
d3 = jov[jov.cnes.isin(est) & jov.per.isin(["2018-19", "2022-25"])].copy()
d3["per"] = d3.per.cat.remove_unused_categories()
var = d3.groupby("cnes").mec.transform("mean")
d3 = d3[(var > 0) & (var < 1)]  # sem hospitais só-bio/só-mec (separação)
ajusta("efeito fixo de hospital (hospitais ativos nos 2 períodos)", d3, BASE + " + C(cnes)")
ok = [h for h in jov.cnes.unique() if h not in hosp_padrao and indet.get(h, 1) < 0.20]
ajusta("hospitais com CID plausível (nunca padrão; indeterminada <20%)", jov[jov.cnes.isin(ok)], BASE + " + C(regiao)")
for et in ("reumática", "não reumática"):
    ajusta(f"etiologia {et}", jov[jov.etiologia == et],
           BASE.replace(" + C(etiologia)", "") + " + C(regiao)")
pd.DataFrame(lin).to_csv(f"{R}/E3_E5_or_2022_25.csv", index=False)
open(f"{R}/resumo.txt", "w").write("\n".join(txt) + "\n")
