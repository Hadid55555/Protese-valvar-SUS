#!/usr/bin/env python3
"""Auditorias v1 (após a segunda opinião): classificação da valva por ano, AIH com 2+ próteses, teste de artefato da
subida da mecânica em mulheres de 15–44 anos, comparação com Alkhouli 2021 (NIS, EUA, 2008–2017).

Entradas: saida_protese/aih_protese_classificada.csv (classificar_valva.py) e saida_protese/sp_valvas_consolidado.csv.
Saídas: protese_valvar/resultados_auditoria/*.csv e resumo.txt.
"""
import os

import numpy as np
import pandas as pd

R = "protese_valvar/resultados_auditoria"
os.makedirs(R, exist_ok=True)
txt = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    txt.append(s)


def salva(df, nome):
    df.to_csv(f"{R}/{nome}.csv")
    return df


a = pd.read_csv("saida_protese/aih_protese_classificada.csv", dtype=str, keep_default_na=False)
for c in ("n_bio", "n_mec", "n_prot"):
    a[c] = pd.to_numeric(a[c]).astype(int)
a["idade"] = pd.to_numeric(a.idade, errors="coerce")
a["mec"] = a.mec == "True"
a["ano_i"] = a.ano.astype(int)
MULT = {"0415010012", "0415020034"}
a["multipla_0415"] = a.proc_aih.isin(MULT)

# ---------- detalhe das OPM por AIH (códigos e CNPJ), na competência com mais OPM
sp = pd.read_csv("saida_protese/sp_valvas_consolidado.csv", dtype=str, keep_default_na=False,
                 usecols=["SP_NAIH", "SP_ATOPROF", "SP_QTD_ATO", "SP_PJ_DOC", "arquivo_origem"])
OPM = {"0702040541": "bio_c_suporte", "0702040550": "bio_s_suporte", "0702040568": "mec_disco",
       "0702040576": "mec_duplo_folheto"}
o = sp[sp.SP_ATOPROF.isin(OPM)].copy()
o["q"] = pd.to_numeric(o.SP_QTD_ATO, errors="coerce").fillna(1).clip(lower=1)
tot = o.groupby(["SP_NAIH", "arquivo_origem"]).q.sum().reset_index()
melhor = tot.sort_values("q").drop_duplicates("SP_NAIH", keep="last")[["SP_NAIH", "arquivo_origem"]]
o = o.merge(melhor, on=["SP_NAIH", "arquivo_origem"])
det = o.groupby("SP_NAIH").agg(
    codigos=("SP_ATOPROF", lambda s: "+".join(sorted(OPM[x] for x in s))),
    n_codigos_distintos=("SP_ATOPROF", "nunique"),
    n_linhas_opm=("SP_ATOPROF", "size"),
    cnpj=("SP_PJ_DOC", lambda s: s.mode().iat[0] if len(s) else ""),
    n_cnpj=("SP_PJ_DOC", "nunique"))
a = a.merge(det, left_on="N_AIH", right_index=True, how="left")
a["tipo_mec"] = np.select([a.codigos.str.contains("mec_duplo", na=False), a.codigos.str.contains("mec_disco", na=False)],
                          ["duplo folheto", "disco"], "")

# ---------- 1. classificação da valva por ano
v = salva(pd.crosstab(a.ano, a.valva, normalize="index").mul(100).round(1), "A1_valva_pct_por_ano")
log("A1. % por valva e ano (seleção):")
log(v.iloc[::3].to_string())
log(f"Indeterminada: {100 * (a.valva == 'indeterminada').mean():.1f}% no total")
ind = a[a.valva == "indeterminada"]
salva(pd.crosstab(ind.proc_aih, ind.ano), "A1b_indeterminada_proc_ano")
salva(ind.cid_pri.str[:3].value_counts().head(25).to_frame("n"), "A1c_indeterminada_cid")

# ---------- 2. AIH com 2+ próteses
d2 = a[a.n_prot >= 2].copy()
d2["um_cid_valva"] = ~d2.cid_pri.str[:3].isin(["I08"])
log(f"\nA2. AIH com 2+ próteses: {len(d2):,} ({100 * len(d2) / len(a):.1f}%)")
t = d2.groupby("proc_aih").agg(n=("N_AIH", "size"), pct_codigos_diferentes=("n_codigos_distintos", lambda s: 100 * (s > 1).mean()),
                               pct_cid_I08=("cid_pri", lambda s: 100 * s.str.startswith("I08").mean()))
log(salva(t.sort_values("n", ascending=False).round(1), "A2_duas_proteses_por_proc").head(8).to_string())
log("Combinações de OPM mais comuns:")
log(salva(d2.codigos.value_counts().head(10).to_frame("n"), "A2b_combinacoes").to_string())
salva(pd.crosstab(d2.ano, d2.proc_aih), "A2c_duas_proteses_ano_proc")

# ---------- 3. mulheres de 15–44 anos: teste de artefato
b = a[a.protese != "ambas"].copy()
b["periodo"] = pd.cut(b.ano_i, [2007, 2012, 2017, 2021, 2025], labels=["2008-12", "2013-17", "2018-21", "2022-25"])
jov = b[(b.idade >= 15) & (b.idade <= 44) & b.valva.isin(["aórtica", "mitral"])]
cid_valvar = jov.cid_pri.str[:3].isin(["I35", "I06", "I34", "I05"]) | jov.cid_pri.str[:4].isin(["Q230", "Q231", "Q232", "Q233"])
cenarios = {
    "todas": jov.index,
    "sem_multiplas_0415": jov.index[~jov.multipla_0415],
    "so_0406010692": jov.index[jov.proc_aih == "0406010692"],
    "1_protese_e_CID_principal_valvar": jov.index[(jov.n_prot == 1) & cid_valvar],
    "sem_endocardite": jov.index[jov.etiologia != "endocardite"],
    "so_reumatica": jov.index[jov.etiologia == "reumática"],
    "so_nao_reumatica": jov.index[jov.etiologia == "não reumática"],
    "so_eletiva": jov.index[jov.rd_CAR_INT == "01"],
}
linhas = []
for nome, idx in cenarios.items():
    x = jov.loc[idx]
    g3 = x.groupby(["periodo", "valva", "sexo"], observed=True).mec.agg(["mean", "size"]).reset_index()
    for r in g3.itertuples():
        linhas.append({"cenario": nome, "periodo": r.periodo, "valva": r.valva, "sexo": r.sexo,
                       "pct_mec": round(100 * r.mean, 1), "n": int(r.size)})
c3 = pd.DataFrame(linhas)
c3.to_csv(f"{R}/A3_mulher_15_44_cenarios.csv", index=False)
log("\nA3. % mecânica, 15–44 anos, mulheres (F) e homens (M), por cenário:")
log(c3.pivot_table(index=["cenario", "periodo"], columns=["valva", "sexo"], values="pct_mec", observed=True).to_string())

ano = jov.pivot_table(index="ano", columns=["valva", "sexo"], values="mec", aggfunc="mean") * 100
salva(ano.round(1), "A3b_15_44_mec_por_ano")
log("\nA3b. % mecânica por ano, 15–44:")
log(ano.round(0).iloc[8:].to_string())
log("\n% de AIH 0415 entre as 15–44 por período:")
log((jov.groupby("periodo", observed=True).multipla_0415.mean() * 100).round(1).to_string())
# comparação: mesma tendência em 45–59 e 60+? (mudança geral × específica da mulher jovem)
outras = b[b.valva.isin(["aórtica", "mitral"])].copy()
outras["fx"] = pd.cut(outras.idade, [0, 15, 45, 60, 200], right=False, labels=["<15", "15-44", "45-59", "60+"])
g = outras.pivot_table(index="periodo", columns=["valva", "fx"], values="mec", aggfunc="mean", observed=True) * 100
log("\nA3c. % mecânica por faixa (ambos os sexos):")
log(salva(g.round(1), "A3c_mec_por_faixa_periodo").to_string())
# tipo de mecânica e fornecedor nas mulheres jovens com mecânica
mj = jov[jov.mec & (jov.sexo == "F")]
log("\nA3d. Mulheres 15–44 com mecânica: tipo (disco × duplo folheto) por período, %:")
log(salva((pd.crosstab(mj.periodo, mj.tipo_mec, normalize="index") * 100).round(1), "A3d_tipo_mec_mulher").to_string())
topc = mj.cnpj.value_counts().head(6).index
log(salva((pd.crosstab(mj.periodo, mj.cnpj.where(mj.cnpj.isin(topc), "outros"), normalize="index") * 100).round(1),
          "A3e_cnpj_mec_mulher").to_string())
# concentração: a subida vem de poucos hospitais?
h = jov[(jov.sexo == "F")].groupby(["cnes", "periodo"], observed=True).agg(n=("mec", "size"), mec=("mec", "sum")).reset_index()
p1 = h[h.periodo == "2018-21"].set_index("cnes")[["n", "mec"]]
p2 = h[h.periodo == "2022-25"].set_index("cnes")[["n", "mec"]]
j = p1.join(p2, lsuffix="_18", rsuffix="_22", how="outer").fillna(0)
j["delta_mec"] = j.mec_22 - j.mec_18
j["delta_n"] = j.n_22 - j.n_18
j = j.sort_values("delta_mec", ascending=False)
salva(j, "A3f_hospitais_mulher_delta")
log(f"\nA3f. Mulheres 15–44: mecânicas 2018–21 = {int(j.mec_18.sum())}, 2022–25 = {int(j.mec_22.sum())}; "
    f"os 10 hospitais com maior aumento respondem por {int(j.delta_mec.head(10).sum())} do aumento líquido de "
    f"{int(j.delta_mec.sum())}. Total de AIH (todas próteses): {int(j.n_18.sum())} → {int(j.n_22.sum())}")

# ---------- 4. Alkhouli 2021: % mecânica por ano, aórtica e mitral, todas as idades (adultos ≥18)
ad = b[(b.idade >= 18) & b.valva.isin(["aórtica", "mitral"])]
al = ad.pivot_table(index="ano", columns="valva", values="mec", aggfunc="mean") * 100
al["ref_EUA_aortica"] = np.nan
al["ref_EUA_mitral"] = np.nan
al.loc["2008", ["ref_EUA_aortica", "ref_EUA_mitral"]] = [45.3, 59.5]
al.loc["2017", ["ref_EUA_aortica", "ref_EUA_mitral"]] = [17.0, 29.2]
log("\nA4. % mecânica em adultos, SUS × EUA (Alkhouli 2021, NIS):")
log(salva(al.round(1), "A4_comparacao_alkhouli").loc[["2008", "2011", "2014", "2017", "2020", "2025"]].to_string())

open(f"{R}/resumo.txt", "w").write("\n".join(txt) + "\n")
