#!/usr/bin/env python3
"""Sensibilidades dos achados principais (pendência do HANDOFF; VALIDACAO_CENTROS.md, consequência 2).

Cenários (cada um exclui AIH índice da base principal: ≥18 anos, aórtica/mitral, 1 tipo de prótese):
  principal       nenhuma exclusão
  sem_cid_padrao  hospital-ano em que um único CID principal (3 caracteres; DIAG_PRINC do RD, senão CID do SP)
                  responde por ≥70% das AIH (hospital-anos com ≥10 AIH): a valva pelo CID não é informativa
  sem_opm_baixa   hospitais (≥100 AIH de troca valvar no RD) com <80% das AIH de troca com OPM de prótese
  sem_2008_09     índice a partir de 2010
  sem_endocardite I33 em qualquer CID (SP ou RD) ou etiologia = endocardite
  todas           as quatro exclusões juntas
Estimativas por cenário:
  (1) % de mecânica por valva e período, 18–59 anos;
  (2) OR de mecânica 2022–25 × 2018–19, 18–59 anos (mesmo modelo de analises_v2, Q3);
  (3) OR de região (ref. Sudeste) e MOR do hospital (mesmo modelo de analises_v2, M1);
  (4) HR de reoperação biológica × mecânica, 18–64 anos, por valva (mesmo modelo de reoperacao.py).
Saídas: protese_valvar/resultados_sensibilidade/ (agregados).
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

import reoperacao as ro

R = "protese_valvar/resultados_sensibilidade"
os.makedirs(R, exist_ok=True)
txt = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    txt.append(s)


a = pd.read_csv("saida_protese/aih_protese_classificada.csv", dtype=str, keep_default_na=False)
for c in ("n_bio", "n_mec", "n_prot"):
    a[c] = pd.to_numeric(a[c]).astype(int)
a["idade"] = pd.to_numeric(a.idade, errors="coerce")
a["mec"] = (a.mec == "True").astype(int)
a["ano_i"] = a.ano.astype(int)
a["regiao"] = a.uf.str[0].map({"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"})
a["urgencia"] = (a.rd_CAR_INT != "01").astype(int)

# ------------------------------------------------------------------ marcadores
cid = a.rd_DIAG_PRINC.where(a.rd_DIAG_PRINC.str.len() >= 3, a.cid_pri).str[:3]
hy = pd.DataFrame({"cnes": a.cnes, "ano": a.ano, "cid": cid})
g = hy.groupby(["cnes", "ano"]).cid.agg(n="size", top=lambda s: s.value_counts(normalize=True).iat[0],
                                         cid_top=lambda s: s.value_counts().index[0])
padrao = g[(g.n >= 10) & (g.top >= 0.70)]
a["f_cid_padrao"] = pd.MultiIndex.from_arrays([a.cnes, a.ano]).isin(padrao.index)
log(f"CID padrão (≥70%, hospital-anos com ≥10 AIH): {len(padrao)} hospital-anos, {a.f_cid_padrao.mean():.1%} das AIH. "
    f"CID mais comum nesses: {padrao.cid_top.value_counts().head(4).to_dict()}  "
    f"[VALIDACAO_CENTROS: 163 hospital-anos, 5,1%]")

rd = pd.read_csv("saida_protese/rd_valvas_consolidado.csv", dtype=str, keep_default_na=False,
                 usecols=["N_AIH", "PROC_REA", "CNES"]).drop_duplicates("N_AIH")
rd["opm"] = rd.N_AIH.isin(a.N_AIH)
p = rd.groupby("PROC_REA").opm.agg(["mean", "size"])
troca = p[(p["mean"] >= 0.5) & (p["size"] >= 100)].index
t = rd[rd.PROC_REA.isin(troca)].groupby("CNES").opm.agg(["mean", "size"])
baixa = t[(t["size"] >= 100) & (t["mean"] < 0.80)].index
a["f_opm_baixa"] = a.cnes.isin(baixa)
log(f"Procedimentos de troca (≥50% com OPM, ≥100 AIH): {list(troca)}")
log(f"Hospitais com ≥100 AIH de troca e <80% com OPM: {len(baixa)}; {a.f_opm_baixa.mean():.1%} das AIH com OPM  "
    f"[VALIDACAO_CENTROS: 15 hospitais, 3,6%]")

dx = ["cid_pri", "cid_sec", "rd_DIAG_PRINC", "rd_DIAG_SECUN"] + [f"rd_DIAGSEC{i}" for i in range(1, 10)]
a["f_endocardite"] = a[dx].apply(lambda s: s.str.upper().str.startswith("I33")).any(axis=1) | (a.etiologia == "endocardite")
a["f_2008_09"] = a.ano_i < 2010
log(f"Endocardite (I33 em qualquer CID): {a.f_endocardite.mean():.1%} das AIH; 2008–09: {a.f_2008_09.mean():.1%}")

FLAGS = {"sem_cid_padrao": ["f_cid_padrao"], "sem_opm_baixa": ["f_opm_baixa"], "sem_2008_09": ["f_2008_09"],
         "sem_endocardite": ["f_endocardite"]}
FLAGS["todas"] = sum(FLAGS.values(), [])
CENARIOS = {"principal": []} | FLAGS

b0 = a[(a.protese != "ambas") & a.valva.isin(["aórtica", "mitral"]) & (a.idade >= 18)].copy()
d_reop = ro.analisa(["NASC", "SEXO", "CEP"], "sens", idade_min=18, so_dados=True)
d_reop = d_reop.merge(a[["N_AIH"] + sum(FLAGS.values(), [])[:4]], on="N_AIH", how="left")

per_lab = ["2008-12", "2013-17", "2018-19", "2020-21", "2022-25"]
tab_pct, tab_or, tab_reg, tab_hr = [], [], [], []
for nome, fl in CENARIOS.items():
    excl = b0[fl].any(axis=1) if fl else pd.Series(False, index=b0.index)
    b = b0[~excl].copy()
    log(f"\n=== {nome}: {len(b):,} AIH (excluídas {excl.sum():,}; {excl.mean():.1%})")
    jov = b[b.idade < 60].copy()
    jov["per"] = pd.cut(jov.ano_i, [2007, 2012, 2017, 2019, 2021, 2025], labels=per_lab)
    # (1)
    pc = (jov.groupby(["valva", "per"], observed=True).mec.mean() * 100).round(1).unstack()
    for v, row in pc.iterrows():
        tab_pct.append({"cenario": nome, "valva": v, **row.to_dict()})
    log("% mecânica 18–59:\n" + pc.to_string())
    # (2)
    f = smf.logit("mec ~ C(per, Treatment('2018-19')) + C(valva) + C(etiologia) + C(sexo) + idade + urgencia + "
                  "C(regiao)", data=jov.dropna(subset=["sexo"])).fit(disp=0)
    k = "C(per, Treatment('2018-19'))[T.2022-25]"
    o = np.exp([f.params[k], *f.conf_int().loc[k]])
    tab_or.append({"cenario": nome, "n": int(f.nobs), "OR_2022_25_vs_2018_19": round(o[0], 2),
                   "IC95": f"{o[1]:.2f}–{o[2]:.2f}"})
    log(f"OR 2022–25 × 2018–19: {o[0]:.2f} ({o[1]:.2f}–{o[2]:.2f})")
    # (3)
    d = b.dropna(subset=["sexo", "regiao"]).copy()
    d = d.join(d.groupby(["cnes", "ano"]).size().rename("vol_ano"), on=["cnes", "ano"])
    d["vol"] = pd.cut(d.vol_ano, [0, 25, 50, 100, 10000], labels=["<25", "25-49", "50-99", "100+"])
    d["fx"] = pd.cut(d.idade, [18, 30, 40, 50, 60, 65, 70, 200], right=False,
                     labels=["18-29", "30-39", "40-49", "50-59", "60-64", "65-69", "70+"])
    d["per"] = pd.cut(d.ano_i, [2007, 2012, 2017, 2021, 2025], labels=["2008-12", "2013-17", "2018-21", "2022-25"])
    nh = d.cnes.value_counts()
    d = d[d.cnes.isin(nh[nh >= 30].index)]
    form = ("mec ~ C(fx, Treatment('60-64')) + C(sexo) + C(valva) + C(etiologia) + urgencia + C(per) + "
            "C(regiao, Treatment('Sudeste')) + C(vol)")
    fe = smf.logit(form, data=d).fit(disp=0)
    r = BinomialBayesMixedGLM.from_formula(form, {"hosp": "0 + C(cnes)"}, d).fit_vb()
    sd = np.exp(r.vcp_mean[0])
    orm = pd.Series(np.exp(r.fe_mean), index=fe.params.index)
    lin = {"cenario": nome, "AIH": len(d), "hospitais": d.cnes.nunique(),
           "MOR": round(float(np.exp(np.sqrt(2) * sd * 0.6745)), 2),
           "ICC": round(float(sd ** 2 / (sd ** 2 + np.pi ** 2 / 3)), 2)}
    for reg in ("Sul", "Nordeste", "Norte", "Centro-Oeste"):
        lin[f"OR_{reg}_misto"] = round(float(orm[f"C(regiao, Treatment('Sudeste'))[T.{reg}]"]), 2)
    tab_reg.append(lin)
    log(f"Multinível: {lin}")
    # (4)
    fr = [x for x in fl if x in d_reop.columns]
    c = d_reop[~d_reop[fr].fillna(False).any(axis=1)] if fr else d_reop
    c = c[c.idade < 65].dropna(subset=["sexo"]).copy()
    c["bio"] = (c.protese == "bio").astype(int)
    c["masc"] = (c.sexo == "M").astype(int)
    c["reum"] = (c.etiologia == "reumática").astype(int)
    c["ano_idx"] = c.ano.astype(int) - 2008
    for v in ("aórtica", "mitral"):
        cc = c[c.valva == v]
        m = PHReg(cc.anos.values, cc[["bio", "idade", "masc", "reum", "ano_idx"]].values, status=cc.evento.values).fit()
        h = np.exp([m.params[0], *m.conf_int()[0]])
        tab_hr.append({"cenario": nome, "valva": v, "n": len(cc), "reop": int(cc.evento.sum()),
                       "HR_bio_vs_mec": round(h[0], 2), "IC95": f"{h[1]:.2f}–{h[2]:.2f}"})
        log(f"Reoperação {v}: HR {h[0]:.2f} ({h[1]:.2f}–{h[2]:.2f}), n={len(cc):,}")

pd.DataFrame(tab_pct).to_csv(f"{R}/S1_pct_mecanica_18_59.csv", index=False)
pd.DataFrame(tab_or).to_csv(f"{R}/S2_or_2022_25.csv", index=False)
pd.DataFrame(tab_reg).to_csv(f"{R}/S3_multinivel_regiao.csv", index=False)
pd.DataFrame(tab_hr).to_csv(f"{R}/S4_reoperacao_hr.csv", index=False)
open(f"{R}/resumo.txt", "w").write("\n".join(txt) + "\n")
