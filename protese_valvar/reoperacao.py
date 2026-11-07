#!/usr/bin/env python3
"""Reoperação valvar após biológica × mecânica (exploratório), por ligação determinística de AIH da mesma pessoa.

Chave principal: NASC + SEXO + CEP (8 dígitos). Sensibilidades: (a) excluir CEP genérico (termina em 000);
(b) chave NASC + SEXO + MUNIC_RES; (c) índice a partir de 2011 (3 anos de "washout" para reduzir a inclusão de quem já
tinha cirurgia antes de 2008).
Índice: primeira AIH de troca valvar com OPM (só bio ou só mec, 1 prótese), aórtica ou mitral, ≥18 anos, alta vivo,
internação 2008–2025, sem AIH valvar anterior da mesma chave desde 2008.
Evento: AIH posterior (internação após a alta do índice) com OPM de prótese valvar ou TAVI (valve-in-valve).
Censura: 31/12/2025. Óbito fora do hospital não é observado (sem ligação com o SIM): a incidência acumulada é
estimada por Kaplan-Meier tratando óbito não observado como "em risco", o que **subestima** a reoperação, mais na
biológica (pacientes mais velhos). Por isso a comparação principal é restrita a 18–64 anos e estratificada por idade.
Reoperação só no SUS (quem reoperou pela saúde suplementar fica invisível).
"""
import os

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

R = "protese_valvar/resultados_reoperacao"
os.makedirs(R, exist_ok=True)
txt = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    txt.append(s)


# chave de ligação (fora do git)
import glob
ch = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in glob.glob("saida_protese/chave/parciais/*.csv")
                if os.path.getsize(f) > 0], ignore_index=True)
ch = ch.sort_values(["ANO_CMPT", "MES_CMPT"]).drop_duplicates("N_AIH", keep="last")
log(f"Chave: {len(ch):,} AIH; NASC válido {ch.NASC.str.match(r'^(19|20)\d{6}$').mean():.1%}; "
    f"CEP 8 dígitos {ch.CEP.str.match(r'^\d{8}$').mean():.1%}; CEP genérico (…000) {ch.CEP.str.endswith('000').mean():.1%}")

a = pd.read_csv("saida_protese/aih_protese_classificada.csv", dtype=str, keep_default_na=False)
a = a.merge(ch[["N_AIH", "NASC", "CEP", "MUNIC_RES", "SEXO"]], on="N_AIH", how="left")
for c in ("n_bio", "n_mec", "n_prot"):
    a[c] = pd.to_numeric(a[c]).astype(int)
a["idade"] = pd.to_numeric(a.idade, errors="coerce")
a["di"] = pd.to_datetime(a.dt_inter, format="%Y%m%d", errors="coerce")
a["ds"] = pd.to_datetime(a.dt_saida, format="%Y%m%d", errors="coerce")
a["obito"] = a.rd_MORTE == "1"
# TAVI (valve-in-valve possível) entra como evento: AIH do RD com código TAVI
rd = pd.read_csv("saida_protese/rd_valvas_consolidado.csv", dtype=str, keep_default_na=False,
                 usecols=["N_AIH", "PROC_REA", "DT_INTER"]).drop_duplicates("N_AIH")
tavi = rd[rd.PROC_REA.isin(["0406030162", "0406011524"])].merge(ch[["N_AIH", "NASC", "CEP", "MUNIC_RES", "SEXO"]],
                                                                   on="N_AIH")
tavi["di"] = pd.to_datetime(tavi.DT_INTER, format="%Y%m%d", errors="coerce")
FIM = pd.Timestamp("2025-12-31")


def km(t, e, pontos):
    """Kaplan-Meier: incidência acumulada (1 - S) nos pontos (anos)."""
    d = pd.DataFrame({"t": t, "e": e}).sort_values("t")
    g = d.groupby("t").e.agg(["sum", "size"])
    risco = len(d) - g["size"].cumsum().shift(fill_value=0)
    s = (1 - g["sum"] / risco).cumprod()
    return [1 - (s[s.index <= p].iloc[-1] if (s.index <= p).any() else 1.0) for p in pontos]


def analisa(chave, nome, filtro_cep=False, ini="2008", idade_min=18, so_dados=False):
    x = a.copy()
    x["k"] = x[chave].agg("|".join, axis=1)
    ok = x.NASC.str.match(r"^(19|20)\d{6}$") & x.SEXO.isin(["1", "3"])
    if "CEP" in chave:
        ok &= x.CEP.str.match(r"^\d{8}$")
        if filtro_cep:
            ok &= ~x.CEP.str.endswith("000")
    x = x[ok & x.di.notna()].sort_values("di")
    first = x.groupby("k").head(1)
    idx = first[(first.ano >= ini) & (first.ano <= "2025") & (first.idade >= idade_min) & first.valva.isin(["aórtica", "mitral"]) &
                (first.n_prot == 1) & first.protese.isin(["bio", "mec"]) & ~first.obito & first.ds.notna()]
    # eventos: próxima AIH com OPM ou TAVI após a alta
    ev = pd.concat([x[["k", "di", "valva", "protese"]].assign(tipo="opm"),
                    tavi.assign(k=tavi[chave].agg("|".join, axis=1))[["k", "di"]].assign(tipo="tavi")])
    ev = ev.merge(idx[["k", "ds"]], on="k")
    ev = ev[ev.di > ev.ds].sort_values("di").groupby("k").head(1).set_index("k")
    d = idx.set_index("k")[["N_AIH", "ano", "idade", "sexo", "valva", "etiologia", "protese", "cnes", "ds"]].join(
        ev[["di", "tipo", "valva", "protese"]].rename(columns={"valva": "valva_reop", "protese": "protese_reop"}))
    d["evento"] = d.di.notna().astype(int)
    d["fim"] = d.di.where(d.evento == 1, FIM)
    d["anos"] = (d.fim - d.ds).dt.days / 365.25
    d = d[d.anos > 0]
    if so_dados:
        return d
    d["fx"] = pd.cut(d.idade, [18, 40, 50, 60, 65, 70, 200], right=False,
                     labels=["18-39", "40-49", "50-59", "60-64", "65-69", "70+"])
    log(f"\n=== {nome}: {len(d):,} pacientes índice; reoperações {d.evento.sum():,}; seguimento mediano "
        f"{d.anos.median():.1f} anos (máx {d.anos.max():.1f})")
    linhas = []
    for (fx, v, p), g in d.groupby(["fx", "valva", "protese"], observed=True):
        ci = km(g.anos, g.evento, [1, 5, 10, 15])
        linhas.append({"faixa": fx, "valva": v, "protese": p, "n": len(g), "reop": int(g.evento.sum()),
                       "CI_1a": 100 * ci[0], "CI_5a": 100 * ci[1], "CI_10a": 100 * ci[2], "CI_15a": 100 * ci[3],
                       "anos_ate_reop_mediana": g.loc[g.evento == 1, "anos"].median()})
    t = pd.DataFrame(linhas).round(1)
    t.to_csv(f"{R}/{nome}_incidencia_acumulada.csv", index=False)
    log(t[t.faixa.isin(["18-39", "40-49", "50-59", "60-64"])].to_string(index=False))
    # Cox 18–64, estratificado por valva, ajustado por idade, sexo, etiologia, ano do índice
    c = d[(d.idade < 65)].dropna(subset=["sexo"]).copy()
    c["bio"] = (c.protese == "bio").astype(int)
    c["masc"] = (c.sexo == "M").astype(int)
    c["reum"] = (c.etiologia == "reumática").astype(int)
    c["ano_idx"] = c.ano.astype(int) - 2008
    res = {}
    for v in ("aórtica", "mitral"):
        cc = c[c.valva == v]
        m = PHReg(cc.anos.values, cc[["bio", "idade", "masc", "reum", "ano_idx"]].values, status=cc.evento.values).fit()
        hr = np.exp(m.params[0])
        lo, hi = np.exp(m.conf_int()[0])
        res[v] = (hr, lo, hi)
        # HR por tempo: <2 anos × ≥2 anos (não proporcionalidade esperada: mecânica reopera cedo, biológica tarde)
        for lab, (t0, t1) in {"<2a": (0, 2), "≥2a": (2, 99)}.items():
            s = cc[cc.anos > t0].copy()
            s["ev2"] = ((s.evento == 1) & (s.anos <= t1)).astype(int)
            s["t2"] = s.anos.clip(upper=t1) - t0
            m2 = PHReg(s.t2.values, s[["bio", "idade", "masc", "reum", "ano_idx"]].values, status=s.ev2.values).fit()
            res[f"{v} {lab}"] = (np.exp(m2.params[0]), *np.exp(m2.conf_int()[0]))
    log("HR (biológica × mecânica) de reoperação, 18–64 anos, ajustado por idade, sexo, reumática e ano:")
    for k, (h, lo, hi) in res.items():
        log(f"  {k}: HR {h:.2f} (IC95% {lo:.2f}–{hi:.2f})")
    log(f"Tipo do evento: {d[d.evento == 1].tipo.value_counts().to_dict()}; mesma valva na reoperação: "
        f"{(d.loc[d.evento == 1, 'valva_reop'] == d.loc[d.evento == 1, 'valva']).mean():.0%} (entre as com OPM)")
    log("Prótese na reoperação, por prótese do índice (%):")
    log((pd.crosstab(d.loc[d.evento == 1, "protese"], d.loc[d.evento == 1, "protese_reop"], normalize="index") * 100)
        .round(0).to_string())
    return d


if __name__ == "__main__":
    d = analisa(["NASC", "SEXO", "CEP"], "principal_NASC_SEXO_CEP")
    analisa(["NASC", "SEXO", "CEP"], "sens_sem_CEP_generico", filtro_cep=True)
    analisa(["NASC", "SEXO", "MUNIC_RES"], "sens_NASC_SEXO_MUNIC")
    analisa(["NASC", "SEXO", "CEP"], "sens_indice_2011", ini="2011")
    open(f"{R}/resumo.txt", "w").write("\n".join(txt) + "\n")
