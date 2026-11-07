#!/usr/bin/env python3
"""Série 2000–2025 da prótese valvar (biológica × mecânica): emenda 2000–2007 (extrair_pre2008.py, tabela SIH antiga,
OPM pela TUNEP) com 2008–2025 (aih_protese_classificada.csv). Mesma regra nos dois períodos: AIH com OPM de prótese
valvar; tipo pela OPM; valva e etiologia pelo CID (classificar_valva.py; antes de 2008 só DIAG_PRINC e DIAG_SECUN, pois
o RD antigo não tem DIAGSEC1–9 e o SP antigo não tem CID). Ano = ano da internação.
Saídas: protese_valvar/resultados_serie/ (agregados) e figura.
"""
import glob
import os

import numpy as np
import pandas as pd

R = "protese_valvar/resultados_serie"
os.makedirs(R, exist_ok=True)
txt = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    txt.append(s)


OPM = {"93323131": "bio", "93329156": "bio", "93323158": "mec", "93323166": "mec"}
fs = [f for f in glob.glob("saida_protese/pre2008/sp/parciais/*.csv") if os.path.getsize(f) > 0]
sp = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in fs], ignore_index=True)
sp["N_AIH"] = sp.SP_NAIH.str.strip()
sp["t"] = sp.SP_ATOPROF.str.strip().map(OPM)
sp["q"] = pd.to_numeric(sp.SP_QTD_ATO, errors="coerce").fillna(1).clip(lower=1)
sp = sp.drop_duplicates(["N_AIH", "SP_ATOPROF", "SP_QTD_ATO", "SP_VALATO"])
g = sp.groupby("N_AIH").apply(lambda d: pd.Series({"n_bio": d.q[d.t == "bio"].sum(), "n_mec": d.q[d.t == "mec"].sum(),
                                                    "sp_dtinter": d.SP_DTINTER.iat[0], "sp_uf": d.SP_UF.iat[0],
                                                    "sp_proc": d.SP_PROCREA.iat[0]}), include_groups=False)
fr = [f for f in glob.glob("saida_protese/pre2008/rd/parciais/*.csv") if os.path.getsize(f) > 0]
rd = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in fr], ignore_index=True)
rd["N_AIH"] = rd.N_AIH.str.strip()
rd = rd.sort_values(["ANO_CMPT", "MES_CMPT"]).drop_duplicates("N_AIH", keep="last").set_index("N_AIH")
p = g.join(rd, how="left")
log(f"2000–2007: AIH com OPM valvar no SP {len(g):,}; com RD encontrado {p.DT_INTER.notna().sum():,} "
    f"({p.DT_INTER.notna().mean():.1%})")
p["dt"] = p.DT_INTER.where(p.DT_INTER.fillna("").str.len() == 8, p.sp_dtinter)
p["ano"] = p.dt.str[:4]
p["protese"] = np.select([(p.n_bio > 0) & (p.n_mec > 0), p.n_mec > 0], ["ambas", "mec"], "bio")
p["n_prot"] = p.n_bio + p.n_mec


def valva_cid(c):
    c = str(c).strip().upper()
    if c[:3] in ("I35", "I06") or c[:4] in ("Q230", "Q231"):
        return "aórtica", "reumática" if c[:3] == "I06" else ("congênita" if c[0] == "Q" else "não reumática")
    if c[:3] in ("I34", "I05") or c[:4] in ("Q232", "Q233"):
        return "mitral", "reumática" if c[:3] == "I05" else ("congênita" if c[0] == "Q" else "não reumática")
    if c[:3] == "I08":
        return "múltipla", "reumática"
    if c[:3] in ("I36", "I37", "I07", "Q22"):
        return "tric/pulm", "reumática" if c[:3] == "I07" else ("congênita" if c[0] == "Q" else "não reumática")
    return None


p["idade"] = pd.to_numeric(p.IDADE, errors="coerce").where(p.COD_IDADE.fillna("").isin(["4", ""]))
p["sexo"] = p.SEXO.map({"1": "M", "3": "F", "2": "F"})


def classif(r):
    adulto = r.idade >= 18
    fontes = [r.DIAG_PRINC, r.DIAG_SECUN]
    endo = any(str(f).startswith("I33") for f in fontes)
    for f in fontes:
        if adulto and str(f).strip().upper()[:4].startswith(("Q225", "I38")):
            continue
        v = valva_cid(f)
        if v:
            valva, etio = v
            break
    else:
        valva, etio = "indeterminada", "indeterminada"
    if r.n_prot >= 2 or r.protese == "ambas":
        valva = "dupla/múltipla"
    return valva, ("endocardite" if endo else etio)


p[["valva", "etiologia"]] = p.apply(lambda r: pd.Series(classif(r)), axis=1)
pre = p[p.ano.between("2000", "2007")].reset_index()[["N_AIH", "ano", "protese", "n_prot", "valva", "etiologia", "idade",
                                                        "sexo"]].assign(periodo="2000–2007")
pos = pd.read_csv("saida_protese/aih_protese_classificada.csv", dtype=str, keep_default_na=False,
                  usecols=["N_AIH", "ano", "protese", "n_prot", "valva", "etiologia", "idade", "sexo"])
pos["idade"] = pd.to_numeric(pos.idade, errors="coerce")
pos["n_prot"] = pd.to_numeric(pos.n_prot)
pos = pos[pos.ano.between("2008", "2025")].assign(periodo="2008–2025")
s = pd.concat([pre, pos], ignore_index=True)
s["ano"] = s.ano.astype(int)
s.to_parquet("saida_protese/serie_2000_2025.parquet", index=False)  # fora do git
log(f"Série emendada: {len(s):,} AIH ({len(pre):,} em 2000–2007; {len(pos):,} em 2008–2025)")

# T1 por ano
u = s[s.protese != "ambas"]
t1 = u.groupby("ano").agg(AIH=("protese", "size"), pct_mec=("protese", lambda x: round(100 * (x == "mec").mean(), 1)),
                          pct_valva_indet=("valva", lambda x: round(100 * (x == "indeterminada").mean(), 1)),
                          idade_media=("idade", lambda x: round(x.mean(), 1)))
t1["AIH_total_com_ambas"] = s.groupby("ano").size()
t1.to_csv(f"{R}/T1_serie_ano.csv")
log("\nT1. Por ano (só AIH com um tipo de prótese):\n" + t1.to_string())


def pct(d):
    return d.groupby(["ano", "valva"]).protese.apply(lambda x: round(100 * (x == "mec").mean(), 1)).unstack()


ad = u[(u.idade >= 18) & u.valva.isin(["aórtica", "mitral"])]
t2 = pd.concat({"18+": pct(ad), "18-59": pct(ad[ad.idade < 60]), "60+": pct(ad[ad.idade >= 60]),
                "mulher_15_44": pct(u[(u.idade >= 15) & (u.idade < 45) & (u.sexo == "F") & u.valva.isin(["aórtica", "mitral"])])},
               axis=1)
t2.to_csv(f"{R}/T2_pct_mec_valva_idade.csv")
log("\nT2. % mecânica por valva e faixa:\n" + t2.to_string())
# continuidade 2006–07 × 2008–09
c = ad.assign(b=np.where(ad.ano <= 2007, "2006–07", "2008–09"))
c = c[c.ano.between(2006, 2009)]
log("\nContinuidade (≥18, aórtica/mitral): % mecânica e n")
log(c.groupby(["b", "valva"]).protese.agg(n="size", pct_mec=lambda x: round(100 * (x == "mec").mean(), 1)).to_string())
log("\nValva por período (%):\n" + (pd.crosstab(s.periodo, s.valva, normalize="index") * 100).round(1).to_string())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for i, v in enumerate(["aórtica", "mitral"]):
        for k, cor in (("18-59", "#1f6f8b"), ("60+", "#c9733a")):
            ax[i].plot(t2.index, t2[(k, v)], marker="o", ms=3, color=cor, label=k)
        ax[i].axvline(2007.5, color="grey", ls=":", lw=1)
        ax[i].axvline(2020, color="grey", ls="--", lw=1)
        ax[i].set_title(f"{v.capitalize()}: % mecânica (SUS)")
        ax[i].set_ylim(0, 80)
        ax[i].legend(frameon=False)
    ax[0].set_ylabel("% de mecânica")
    fig.text(0.5, 0.01, "Linha pontilhada: troca da tabela SIH (2008). Tracejada: 2020.", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_serie_2000_2025.png", dpi=150)
except Exception as e:  # noqa: BLE001
    log(f"figura não gerada: {e}")
open(f"{R}/resumo.txt", "w").write("\n".join(txt) + "\n")
