#!/usr/bin/env python3
"""Análises v2: ponto de quebra da mecânica <60 anos (mensal) e COVID; fornecedor × hospital; modelo multinível
(MOR) e funnel; descritivas cirúrgicas (plástica mitral, Ross, tubo valvado).

Entradas: saida_protese/aih_protese_classificada.csv, saida_protese/sp_valvas_consolidado.csv,
saida_protese/rd_valvas_consolidado.csv. Saídas: protese_valvar/resultados_v2/.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

R = "protese_valvar/resultados_v2"
os.makedirs(R, exist_ok=True)
txt = []
ETAPA = sys.argv[1:] or ["quebra", "fornecedor", "multinivel", "cirurgia"]


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
a["mes"] = pd.to_datetime(a.dt_inter.str[:6], format="%Y%m", errors="coerce")
a["regiao"] = a.uf.str[0].map({"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"})
a["urgencia"] = (a.rd_CAR_INT != "01").astype(int)
b = a[(a.protese != "ambas") & a.valva.isin(["aórtica", "mitral"]) & a.idade.notna()].copy()
jov = b[(b.idade >= 18) & (b.idade < 60)].copy()

# ------------------------------------------------------------------ 1. ponto de quebra
if "quebra" in ETAPA:
    m = jov.groupby("mes").agg(n=("mec", "size"), mec=("mec", "sum"), idade=("idade", "mean"),
                               urg=("urgencia", "mean"), reum=("etiologia", lambda s: (s == "reumática").mean()))
    m = m[(m.index >= "2008-01-01") & (m.index <= "2025-12-01")]
    m["pct"] = m.mec / m.n
    m["t"] = np.arange(len(m))
    m.to_csv(f"{R}/Q1_mensal_18_59.csv")

    def ajusta(k):
        d = m.copy()
        d["pos"] = (d.t >= k).astype(int)
        d["tpos"] = (d.t - k).clip(lower=0)
        X = sm.add_constant(d[["t", "pos", "tpos"]])
        f = sm.GLM(d[["mec"]].assign(nao=d.n - d.mec), X, family=sm.families.Binomial()).fit()
        return f, d

    cand = range(36, len(m) - 24)
    dev = [(k, ajusta(k)[0].deviance) for k in cand]
    kbest = min(dev, key=lambda x: x[1])[0]
    fb, _ = ajusta(kbest)
    log(f"Q1. Mecânica 18–59 anos (aórtica+mitral), série mensal {len(m)} meses. Melhor ponto de quebra (deviance "
        f"mínima, modelo segmentado binomial): {m.index[kbest]:%Y-%m}")
    log(fb.summary2().tables[1][["Coef.", "P>|z|"]].round(4).to_string())
    k20 = int(np.where(m.index == pd.Timestamp("2020-03-01"))[0][0])
    f20, _ = ajusta(k20)
    log(f"  deviance: melhor {fb.deviance:.1f}; quebra fixa em mar/2020 {f20.deviance:.1f}")
    pd.DataFrame(dev, columns=["k", "deviance"]).assign(mes=lambda d: m.index[d.k]).to_csv(f"{R}/Q1b_perfil_quebra.csv")
    # por semestre, com case-mix
    jov["sem"] = jov.ano + np.where(pd.to_numeric(jov.dt_inter.str[4:6]) <= 6, "-1", "-2")
    s = jov.groupby("sem").agg(n=("mec", "size"), pct_mec=("mec", "mean"), idade=("idade", "mean"),
                               urgencia=("urgencia", "mean"), reumatica=("etiologia", lambda s: (s == "reumática").mean()),
                               mitral=("valva", lambda s: (s == "mitral").mean()), mulher=("sexo", lambda s: (s == "F").mean()))
    s.round(3).to_csv(f"{R}/Q2_semestral_casemix.csv")
    log("\nQ2. 18–59 anos por semestre (2017–2025): n, % mecânica e case-mix")
    log(s.loc["2017-1":"2025-2"].round(2).to_string())
    # ajuste de case-mix: logística com período × controles
    jov["per"] = pd.cut(jov.ano_i, [2007, 2012, 2017, 2019, 2021, 2025],
                        labels=["2008-12", "2013-17", "2018-19", "2020-21", "2022-25"])
    f = smf.logit("mec ~ C(per, Treatment('2018-19')) + C(valva) + C(etiologia) + C(sexo) + idade + urgencia + "
                  "C(regiao)", data=jov.dropna(subset=["sexo"])).fit(disp=0)
    or_ = np.exp(pd.concat([f.params, f.conf_int()], axis=1)).round(2)
    or_.columns = ["OR", "IC95_inf", "IC95_sup"]
    or_.to_csv(f"{R}/Q3_or_periodo_ajustado.csv")
    log("\nQ3. OR de mecânica por período (ref. 2018–19), ajustado por valva, etiologia, sexo, idade, urgência, região:")
    log(or_[or_.index.str.contains("per")].to_string())

# ------------------------------------------------------------------ 2. fornecedor × hospital
if "fornecedor" in ETAPA:
    sp = pd.read_csv("saida_protese/sp_valvas_consolidado.csv", dtype=str, keep_default_na=False,
                     usecols=["SP_NAIH", "SP_ATOPROF", "SP_PJ_DOC"])
    BIO, MEC = {"0702040541", "0702040550"}, {"0702040568", "0702040576"}
    o = sp[sp.SP_ATOPROF.isin(BIO | MEC)].drop_duplicates(["SP_NAIH", "SP_ATOPROF", "SP_PJ_DOC"])
    o["tipo"] = np.where(o.SP_ATOPROF.isin(MEC), "mec", "bio")
    o = o.merge(a[["N_AIH", "ano", "cnes", "idade"]], left_on="SP_NAIH", right_on="N_AIH")
    import json
    nomes = json.load(open("tavi_savr/dados_ref/cnpj_fornecedores_protese.json"))
    extra = {"21996505000128": "JOTEC DO BRASIL", "00944324000188": "MAQUET CARDIOPULMONARY DO BRASIL",
             "05409930000117": "X MED", "01328114000128": "EXTRACOR"}
    nm = lambda c: extra.get(c) or (nomes.get(c) or {}).get("razao", c)  # noqa: E731
    for tp in ("mec", "bio"):
        x = o[o.tipo == tp]
        top = x.SP_PJ_DOC.value_counts().head(8).index
        t = pd.crosstab(x.ano, x.SP_PJ_DOC.where(x.SP_PJ_DOC.isin(top), "outros"), normalize="index") * 100
        t.columns = [nm(c)[:28] if c != "outros" else c for c in t.columns]
        t.round(1).to_csv(f"{R}/F1_fornecedor_{tp}_ano.csv")
        log(f"\nF1. Fornecedores de {tp} (% das AIH no ano), seleção:")
        log(t.round(0).loc[["2008", "2012", "2016", "2018", "2020", "2022", "2025"]].to_string())
    # hospitais: trocaram o fornecedor principal de mecânica entre 2016–19 e 2021–25?
    om = o[o.tipo == "mec"].copy()
    om["fase"] = np.select([om.ano.between("2016", "2019"), om.ano.between("2021", "2025")], ["A", "B"], "")
    princ = om[om.fase != ""].groupby(["cnes", "fase"]).SP_PJ_DOC.agg(lambda s: s.mode().iat[0]).unstack()
    princ["trocou"] = princ.A != princ.B
    j = jov.copy()
    j["fase"] = np.select([j.ano.between("2016", "2019"), j.ano.between("2021", "2025")], ["A", "B"], "")
    h = j[j.fase != ""].groupby(["cnes", "fase"]).mec.agg(["mean", "size"]).unstack()
    h.columns = [f"{x}_{y}" for x, y in h.columns]
    h = h.join(princ[["trocou", "A", "B"]], how="inner").dropna(subset=["mean_A", "mean_B"])
    h = h[(h.size_A >= 30) & (h.size_B >= 30)]
    h["delta_pp"] = 100 * (h.mean_B - h.mean_A)
    h.to_csv(f"{R}/F2_hospital_troca_fornecedor.csv")
    log(f"\nF2. Hospitais com ≥30 AIH de 18–59 anos em 2016–19 e 2021–25: {len(h)}. Variação do % de mecânica (p.p.):")
    log(h.groupby("trocou").delta_pp.describe().round(1).to_string())
    log("  Para quem trocou, novo fornecedor principal (nº de hospitais):")
    log(h[h.trocou].B.map(nm).value_counts().head(8).to_string())
    log(f"  Hospitais com aumento ≥10 p.p.: {(h.delta_pp >= 10).sum()}; com queda ≥10 p.p.: {(h.delta_pp <= -10).sum()}")

# ------------------------------------------------------------------ 3. multinível + funnel
if "multinivel" in ETAPA:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    d = b[(b.idade >= 18)].dropna(subset=["sexo", "regiao"]).copy()
    vol = d.groupby(["cnes", "ano"]).size().rename("vol_ano")
    d = d.join(vol, on=["cnes", "ano"])
    d["vol"] = pd.cut(d.vol_ano, [0, 25, 50, 100, 10000], labels=["<25", "25-49", "50-99", "100+"])
    d["fx"] = pd.cut(d.idade, [18, 30, 40, 50, 60, 65, 70, 200], right=False,
                     labels=["18-29", "30-39", "40-49", "50-59", "60-64", "65-69", "70+"])
    d["per"] = pd.cut(d.ano_i, [2007, 2012, 2017, 2021, 2025], labels=["2008-12", "2013-17", "2018-21", "2022-25"])
    nh = d.cnes.value_counts()
    d = d[d.cnes.isin(nh[nh >= 30].index)]
    log(f"\nM1. Multinível: {len(d):,} AIH (≥18 anos, aórtica/mitral) em {d.cnes.nunique()} hospitais com ≥30 AIH")
    form = ("mec ~ C(fx, Treatment('60-64')) + C(sexo) + C(valva) + C(etiologia) + urgencia + C(per) + "
            "C(regiao, Treatment('Sudeste')) + C(vol)")
    fe = smf.logit(form, data=d).fit(disp=0)
    md = BinomialBayesMixedGLM.from_formula(form, {"hosp": "0 + C(cnes)"}, d)
    r = md.fit_vb()
    sd = np.exp(r.vcp_mean[0])
    mor = np.exp(np.sqrt(2) * sd * 0.6745)
    icc = sd ** 2 / (sd ** 2 + np.pi ** 2 / 3)
    tab = pd.DataFrame({"OR_misto": np.exp(r.fe_mean), "OR_fixo": np.exp(fe.params.values)},
                       index=fe.params.index).round(2)
    tab.to_csv(f"{R}/M1_or_multinivel.csv")
    log(tab.to_string())
    log(f"  DP do intercepto do hospital = {sd:.2f}; MOR = {mor:.2f}; ICC = {icc:.2f}")
    log(f"  (MOR: mediana do OR entre dois hospitais quaisquer para o mesmo paciente)")
    # funnel: 18–59 anos, 2020–2025, observado × esperado (modelo sem hospital)
    d["p_esp"] = fe.predict(d)
    f = d[(d.idade < 60) & (d.ano_i >= 2020)].groupby("cnes").agg(n=("mec", "size"), obs=("mec", "mean"),
                                                                   esp=("p_esp", "mean"))
    f = f[f.n >= 30]
    p0 = (f.obs * f.n).sum() / f.n.sum()
    f["ajust"] = (f.obs / f.esp * p0).clip(0, 1)
    f["lim95_sup"] = p0 + 1.96 * np.sqrt(p0 * (1 - p0) / f.n)
    f["lim95_inf"] = p0 - 1.96 * np.sqrt(p0 * (1 - p0) / f.n)
    f["lim998_sup"] = p0 + 3.09 * np.sqrt(p0 * (1 - p0) / f.n)
    f["lim998_inf"] = p0 - 3.09 * np.sqrt(p0 * (1 - p0) / f.n)
    f.to_csv(f"{R}/M2_funnel_18_59_2020_2025.csv")
    log(f"\nM2. Funnel 18–59 anos, 2020–25: {len(f)} hospitais (≥30 AIH); média {p0:.1%}. Acima do limite 99,8%: "
        f"{(f.ajust > f.lim998_sup).sum()}; abaixo: {(f.ajust < f.lim998_inf).sum()}. "
        f"Ajustado P10–P90: {f.ajust.quantile(.1):.0%}–{f.ajust.quantile(.9):.0%}")

# ------------------------------------------------------------------ 4. descritivas cirúrgicas
if "cirurgia" in ETAPA:
    rd = pd.read_csv("saida_protese/rd_valvas_consolidado.csv", dtype=str, keep_default_na=False,
                     usecols=["N_AIH", "PROC_REA", "DIAG_PRINC", "DT_INTER", "IDADE", "COD_IDADE"])
    rd = rd.drop_duplicates("N_AIH")
    rd["ano"] = rd.DT_INTER.str[:4]
    rd = rd[rd.ano.between("2008", "2025")]
    rd["mitral"] = rd.DIAG_PRINC.str[:3].isin(["I34", "I05"])
    pl = rd[rd.PROC_REA.isin(["0406010803", "0406010811"]) & rd.mitral].groupby("ano").size().rename("plastica_mitral")
    tr = a[a.valva == "mitral"].groupby("ano").size().rename("troca_mitral_com_opm")
    t = pd.concat([pl, tr], axis=1).fillna(0).astype(int)
    t["pct_plastica"] = (100 * t.plastica_mitral / (t.plastica_mitral + t.troca_mitral_com_opm)).round(1)
    ross = rd[rd.PROC_REA.isin(["0406010552", "0406011478"])].groupby("ano").size().rename("ross")
    tubo = rd[rd.PROC_REA == "0406010846"].groupby("ano").size().rename("tubo_valvado")
    t = t.join(ross).join(tubo).fillna(0)
    t.to_csv(f"{R}/C1_plastica_ross_tubo_ano.csv")
    log("\nC1. Plástica mitral (0406010803/0406010811 com CID mitral principal; não inclui plásticas dentro de "
        "múltiplas) × troca mitral com OPM; Ross; tubo valvado:")
    log(t.loc[["2008", "2011", "2014", "2017", "2020", "2023", "2025"]].to_string())

open(f"{R}/resumo_{'_'.join(ETAPA)}.txt", "w").write("\n".join(txt) + "\n")
