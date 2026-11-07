#!/usr/bin/env python3
"""Complicações e gestação após troca valvar, biológica × mecânica (exploratório), por ligação NASC+SEXO+CEP.

Coorte índice: a mesma de reoperacao.py (primeira troca aórtica ou mitral com 1 prótese, alta vivo, 2008–2025), aqui
a partir de 10 anos. Seguimento: todas as internações SUS da mesma chave (extrair_internacoes_coorte.py) após a alta do
índice, até 31/12/2025. Censura no óbito hospitalar de qualquer internação posterior. O óbito fora do hospital não é
observado (ver ligação com o SIM).

Desfechos (primeira internação com o evento; CID principal salvo indicação):
  AVC/sangramento intracraniano: I60–I62, S06.4–S06.6
  sangramento maior: intracraniano + digestivo (K92.0–K92.2; K25–K28 .0/.2/.4/.6; K62.5) + D62, R58, R04, R31,
     N92, N93 como principal; ou D68.3/T45.5 em qualquer posição (sangramento por anticoagulante)
  tromboembolismo: I63, I64, G45, I74
  complicação de prótese: T82.0, T82.6, T82.8, I33 (principal ou secundário, só após 30 dias da alta)
  óbito hospitalar em qualquer internação posterior
Gestação (mulheres de 15–44 anos no índice): internação com parto (procedimento 0310* ou 0411*, ou CID O80–O84) e
aborto (O00–O08).
"""
import glob
import os

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

import reoperacao as ro

R = "protese_valvar/resultados_complicacoes"
os.makedirs(R, exist_ok=True)
txt = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    txt.append(s)


FIM = pd.Timestamp("2025-12-31")
coh = ro.analisa(["NASC", "SEXO", "CEP"], "complicacoes_coorte", idade_min=10, so_dados=True)
coh = coh[["N_AIH", "ano", "idade", "sexo", "valva", "etiologia", "protese", "cnes", "ds"]].copy()
coh.index.name = "k"
log(f"Coorte índice: {len(coh):,}")

cols = ["N_AIH", "PROC_REA", "DT_INTER", "NASC", "SEXO", "CEP", "MORTE", "DIAG_PRINC", "DIAG_SECUN", "CID_MORTE"] + \
       [f"DIAGSEC{i}" for i in range(1, 10)]
fs = [f for f in glob.glob("saida_protese/coorte_rd/parciais/*.csv") if os.path.getsize(f) > 0]
rd = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=lambda c: c in cols) for f in fs],
               ignore_index=True).drop_duplicates("N_AIH")
rd["k"] = rd.NASC.str.strip() + "|" + rd.SEXO.str.strip() + "|" + rd.CEP.str.strip()
rd = rd[rd.k.isin(coh.index)]
rd["di"] = pd.to_datetime(rd.DT_INTER, format="%Y%m%d", errors="coerce")
rd = rd.join(coh[["ds", "N_AIH"]].rename(columns={"N_AIH": "aih_idx"}), on="k")
rd = rd[(rd.di > rd.ds) & (rd.di <= FIM) & (rd.N_AIH != rd.aih_idx)]
log(f"Internações posteriores ligadas: {len(rd):,} em {rd.k.nunique():,} pacientes")

p3 = rd.DIAG_PRINC.str[:3]
p4 = rd.DIAG_PRINC.str[:4]
sec = rd[["DIAG_SECUN"] + [f"DIAGSEC{i}" for i in range(1, 10)]].apply(lambda s: s.str[:4])
qualquer4 = lambda cods: sec.isin(cods).any(axis=1) | p4.isin(cods)  # noqa: E731
ulc = {f"K{n}{d}" for n in (25, 26, 27, 28) for d in (0, 2, 4, 6)}
ic = p3.isin(["I60", "I61", "I62"]) | p4.isin(["S064", "S065", "S066"])
gi = p4.isin(["K920", "K921", "K922", "K625"] + sorted(ulc))
outros = p3.isin(["D62", "R58", "R04", "R31", "N92", "N93"]) | qualquer4(["D683", "T455"])
rd["ev_intracraniano"] = ic
rd["ev_sangramento"] = ic | gi | outros
rd["ev_tromboembolismo"] = p3.isin(["I63", "I64", "G45", "I74"])
rd["ev_protese"] = (qualquer4(["T820", "T826", "T828"]) | p3.eq("I33") | sec.apply(lambda s: s.str[:3]).eq("I33").any(axis=1)) & \
                   ((rd.di - rd.ds).dt.days > 30)
rd["ev_obito_hosp"] = rd.MORTE == "1"
rd["ev_parto"] = rd.PROC_REA.str.startswith(("0310", "0411")) | p3.isin(["O80", "O81", "O82", "O83", "O84"])
rd["ev_aborto"] = p3.isin([f"O0{i}" for i in range(9)])
EVS = ["ev_sangramento", "ev_intracraniano", "ev_tromboembolismo", "ev_protese", "ev_obito_hosp", "ev_parto", "ev_aborto"]

# censura: óbito hospitalar posterior
obito = rd[rd.ev_obito_hosp].groupby("k").di.min()
coh["fim_obs"] = obito.reindex(coh.index).fillna(FIM)
for e in EVS:
    first = rd[rd[e]].groupby("k").di.min()
    coh[e + "_t"] = first.reindex(coh.index)
coh["fx"] = pd.cut(coh.idade, [10, 18, 40, 50, 60, 65, 200], right=False,
                   labels=["10-17", "18-39", "40-49", "50-59", "60-64", "65+"])


def km(t, e, pontos):
    d = pd.DataFrame({"t": t, "e": e}).groupby("t").e.agg(["sum", "size"]).sort_index()
    risco = d["size"][::-1].cumsum()[::-1]
    s = (1 - d["sum"] / risco).cumprod()
    return [100 * (1 - (s[s.index <= p].iloc[-1] if (s.index <= p).any() else 1.0)) for p in pontos]


def prepara(e, base):
    x = base.copy()
    t_ev = x[e + "_t"]
    fim = x.fim_obs if e != "ev_obito_hosp" else pd.Series(FIM, index=x.index)
    x["evento"] = (t_ev.notna() & (t_ev <= fim)).astype(int)
    x["anos"] = ((t_ev.where(x.evento == 1, fim)) - x.ds).dt.days / 365.25
    return x[x.anos > 0]


def cox(x):
    X = pd.DataFrame({"bio": (x.protese == "bio").astype(int), "idade": x.idade, "masc": (x.sexo == "M").astype(int),
                      "reum": (x.etiologia == "reumática").astype(int), "ano": x.ano.astype(int) - 2008})
    m = PHReg(x.anos.values, X.values, status=x.evento.values, strata=x.valva.values).fit()
    return np.exp(m.params[0]), *np.exp(m.conf_int()[0])


linhas = []
adultos = coh[(coh.idade >= 18) & (coh.idade < 65)].dropna(subset=["sexo"])
log("\nAdultos 18–64 anos: incidência acumulada (%) em 5 e 10 anos e HR biológica × mecânica (Cox estratificado por "
    "valva; ajustado por idade, sexo, reumática, ano)")
for e in EVS[:5]:
    x = prepara(e, adultos)
    r = {"desfecho": e[3:]}
    for p in ("mec", "bio"):
        y = x[x.protese == p]
        c5, c10 = km(y.anos, y.evento, [5, 10])
        r.update({f"{p}_n": len(y), f"{p}_eventos": int(y.evento.sum()), f"{p}_5a": round(c5, 1), f"{p}_10a": round(c10, 1),
                  f"{p}_taxa_100pa": round(100 * y.evento.sum() / y.anos.sum(), 2)})
    hr, lo, hi = cox(x)
    r.update({"HR_bio_vs_mec": round(hr, 2), "IC95": f"{lo:.2f}–{hi:.2f}"})
    linhas.append(r)
t = pd.DataFrame(linhas)
t.to_csv(f"{R}/K1_complicacoes_18_64.csv", index=False)
log(t[["desfecho", "mec_eventos", "mec_taxa_100pa", "mec_10a", "bio_eventos", "bio_taxa_100pa", "bio_10a",
       "HR_bio_vs_mec", "IC95"]].to_string(index=False))

# por faixa: sangramento maior e tromboembolismo
log("\nPor faixa etária, incidência em 10 anos (%) mec / bio e HR bio × mec:")
lin = []
for e in ("ev_sangramento", "ev_tromboembolismo", "ev_protese"):
    x = prepara(e, coh.dropna(subset=["sexo"]))
    for fx, g in x.groupby("fx", observed=True):
        if g.evento.sum() < 10:
            continue
        cm = km(g[g.protese == "mec"].anos, g[g.protese == "mec"].evento, [10])[0]
        cb = km(g[g.protese == "bio"].anos, g[g.protese == "bio"].evento, [10])[0]
        hr, lo, hi = cox(g)
        lin.append({"desfecho": e[3:], "faixa": fx, "n": len(g), "mec_10a": round(cm, 1), "bio_10a": round(cb, 1),
                    "HR": round(hr, 2), "IC95": f"{lo:.2f}–{hi:.2f}"})
t2 = pd.DataFrame(lin)
t2.to_csv(f"{R}/K2_por_faixa.csv", index=False)
log(t2.to_string(index=False))

# gestação: mulheres 15–44 no índice
w = coh[(coh.sexo == "F") & (coh.idade >= 15) & (coh.idade <= 44)]
log(f"\nGestação — mulheres de 15–44 anos no índice: {len(w):,} (mec {int((w.protese == 'mec').sum()):,}, "
    f"bio {int((w.protese == 'bio').sum()):,})")
lin = []
for e in ("ev_parto", "ev_aborto"):
    x = prepara(e, w)
    for fx, g in x.groupby(pd.cut(x.idade, [15, 25, 35, 45], right=False, labels=["15-24", "25-34", "35-44"]),
                           observed=True):
        for p in ("mec", "bio"):
            y = g[g.protese == p]
            c5, c10 = km(y.anos, y.evento, [5, 10])
            lin.append({"desfecho": e[3:], "faixa": fx, "protese": p, "n": len(y), "eventos": int(y.evento.sum()),
                        "pct_5a": round(c5, 1), "pct_10a": round(c10, 1)})
    X = pd.DataFrame({"bio": (x.protese == "bio").astype(int), "idade": x.idade,
                      "reum": (x.etiologia == "reumática").astype(int), "ano": x.ano.astype(int) - 2008})
    m = PHReg(x.anos.values, X.values, status=x.evento.values, strata=x.valva.values).fit()
    log(f"  {e[3:]}: HR biológica × mecânica {np.exp(m.params[0]):.2f} "
        f"({np.exp(m.conf_int()[0][0]):.2f}–{np.exp(m.conf_int()[0][1]):.2f}), ajustado por idade, reumática, ano")
t3 = pd.DataFrame(lin)
t3.to_csv(f"{R}/K3_gestacao.csv", index=False)
log(t3.to_string(index=False))
# desfechos maternos nas internações de parto
pw = rd[rd.ev_parto & rd.k.isin(w.index)].join(w[["protese"]], on="k")
log("\nInternações de parto/aborto após a troca (mulheres 15–44): óbito materno hospitalar por prótese:")
log(pw.groupby("protese").agg(partos=("ev_parto", "size"), obitos=("ev_obito_hosp", "sum")).to_string())
open(f"{R}/resumo.txt", "w").write("\n".join(txt) + "\n")
