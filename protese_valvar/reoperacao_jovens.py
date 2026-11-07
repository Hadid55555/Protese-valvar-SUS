#!/usr/bin/env python3
"""Reoperação nos mais jovens: quanto mais cedo reopera quem recebeu biológica × mecânica na mesma faixa etária.

Usa a ligação de reoperacao.py (NASC+SEXO+CEP), agora a partir de 10 anos. Por faixa (10–17, 18–29, 30–39, 40–49,
50–59) × valva × prótese: incidência acumulada (Kaplan-Meier), tempo até atingir 5% e 10% de reoperação, tempo médio
livre de reoperação em 10 anos (RMST, com IC por bootstrap) e HR biológica × mecânica (Cox, ajustado por sexo,
reumática, valva e ano). Auditoria dos eventos precoces (≤30 e 31–90 dias). Figura das curvas.
Os mesmos limites de reoperacao.py valem aqui (óbito fora do hospital não observado; só SUS): os números são piso.
"""
import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

import reoperacao as ro

R = ro.R
log = ro.log
ro.txt.clear()
d = ro.analisa(["NASC", "SEXO", "CEP"], "jovens", idade_min=10, so_dados=True)
d["fx"] = pd.cut(d.idade, [10, 18, 30, 40, 50, 60], right=False, labels=["10-17", "18-29", "30-39", "40-49", "50-59"])
d = d[d.fx.notna()].copy()
d["dias"] = (d.fim - d.ds).dt.days


def curva(t, e):
    g = pd.DataFrame({"t": t, "e": e}).groupby("t").e.agg(["sum", "size"]).sort_index()
    risco = g["size"][::-1].cumsum()[::-1]
    s = (1 - g["sum"] / risco).cumprod()
    return s  # sobrevida livre de reoperação nos tempos de evento/censura


def ci_em(s, p):
    s = s[s.index <= p]
    return 100 * (1 - (s.iloc[-1] if len(s) else 1.0))


def tempo_ate(s, x):
    ci = 1 - s
    acima = ci[ci >= x / 100]
    return acima.index[0] if len(acima) else np.nan


def rmst(t, e, tau=10):
    s = curva(t, e)
    s = s[s.index <= tau]
    tt = np.r_[0, s.index.values, tau]
    ss = np.r_[1, s.values, s.values[-1] if len(s) else 1]
    return float(np.sum(np.diff(tt) * ss[:-1]))


rng = np.random.default_rng(2026)
linhas = []
for (fx, v), g in d.groupby(["fx", "valva"], observed=True):
    res = {}
    for p in ("bio", "mec"):
        x = g[g.protese == p]
        s = curva(x.anos.values, x.evento.values)
        res[p] = dict(n=len(x), reop=int(x.evento.sum()), CI_2a=ci_em(s, 2), CI_5a=ci_em(s, 5), CI_10a=ci_em(s, 10),
                      CI_15a=ci_em(s, 15), anos_ate_5pct=tempo_ate(s, 5), anos_ate_10pct=tempo_ate(s, 10),
                      rmst10=rmst(x.anos.values, x.evento.values),
                      mediana_anos_reoperados=x.loc[x.evento == 1, "anos"].median())
    boot = []
    xb, xm = g[g.protese == "bio"], g[g.protese == "mec"]
    for _ in range(200):
        bb = xb.sample(len(xb), replace=True, random_state=int(rng.integers(1e9)))
        mm = xm.sample(len(xm), replace=True, random_state=int(rng.integers(1e9)))
        boot.append(rmst(mm.anos.values, mm.evento.values) - rmst(bb.anos.values, bb.evento.values))
    dif = res["mec"]["rmst10"] - res["bio"]["rmst10"]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    for p in ("bio", "mec"):
        linhas.append({"faixa": fx, "valva": v, "protese": p, **res[p]})
    linhas.append({"faixa": fx, "valva": v, "protese": "dif_mec_menos_bio_meses_livres_em_10a",
                   "rmst10": 12 * dif, "CI_2a": 12 * lo, "CI_5a": 12 * hi})
t = pd.DataFrame(linhas)
t.round(2).to_csv(f"{R}/J1_jovens_por_faixa.csv", index=False)
log("\nJ1. Reoperação por faixa, valva e prótese (incidência acumulada %, anos até 5%/10%, RMST 10 anos):")
cols = ["faixa", "valva", "protese", "n", "reop", "CI_5a", "CI_10a", "CI_15a", "anos_ate_5pct", "anos_ate_10pct",
        "mediana_anos_reoperados"]
log(t[t.protese.isin(["bio", "mec"])][cols].round(1).to_string(index=False))
dd = t[t.protese.str.startswith("dif")][["faixa", "valva", "rmst10", "CI_2a", "CI_5a"]]
dd.columns = ["faixa", "valva", "meses_a_mais_sem_reop_mec_10a", "IC95_inf", "IC95_sup"]
log("\nDiferença no tempo médio livre de reoperação em 10 anos (mecânica − biológica, meses):")
log(dd.round(1).to_string(index=False))

# Cox por faixa (aórtica + mitral, estratificado por valva)
log("\nJ2. HR biológica × mecânica por faixa (Cox estratificado por valva; ajustado por sexo, reumática, ano):")
hrs = []
for fx, g in d.dropna(subset=["sexo"]).groupby("fx", observed=True):
    X = pd.DataFrame({"bio": (g.protese == "bio").astype(int), "masc": (g.sexo == "M").astype(int),
                      "reum": (g.etiologia == "reumática").astype(int), "ano": g.ano.astype(int) - 2008})
    m = PHReg(g.anos.values, X.values, status=g.evento.values, strata=g.valva.values).fit()
    hr, (lo, hi) = np.exp(m.params[0]), np.exp(m.conf_int()[0])
    hrs.append({"faixa": fx, "n": len(g), "HR": hr, "IC95_inf": lo, "IC95_sup": hi})
    log(f"  {fx}: n={len(g):,}  HR {hr:.2f} ({lo:.2f}–{hi:.2f})")
pd.DataFrame(hrs).round(2).to_csv(f"{R}/J2_hr_por_faixa.csv", index=False)

# auditoria dos eventos precoces
ev = d[d.evento == 1].copy()
ev["janela"] = pd.cut(ev.dias, [-1, 30, 90, 365, 1e6], labels=["≤30d", "31-90d", "91d-1a", ">1a"])
ev["mesma_valva"] = ev.valva_reop == ev.valva
au = ev.groupby(["janela", "protese"], observed=True).agg(n=("evento", "size"), mesma_valva=("mesma_valva", "mean"))
au.round(2).to_csv(f"{R}/J3_eventos_precoces.csv")
log("\nJ3. Eventos por janela após a alta do índice (n; proporção com mesma valva):")
log(au.round(2).to_string())

# figura: incidência acumulada, 18–29 e 30–39, aórtica e mitral
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COR = {"bio": "#eb6834", "mec": "#2a78d6"}
NOME = {"bio": "Biológica", "mec": "Mecânica"}
fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
for i, fx in enumerate(["18-29", "30-39"]):
    for j, v in enumerate(["aórtica", "mitral"]):
        ax = axs[i, j]
        for p in ("bio", "mec"):
            x = d[(d.fx == fx) & (d.valva == v) & (d.protese == p)]
            s = curva(x.anos.values, x.evento.values)
            s = s[s.index <= 15]
            ax.step(np.r_[0, s.index], 100 * (1 - np.r_[1, s.values]), where="post", color=COR[p], lw=2,
                    label=f"{NOME[p]} (n={len(x):,})".replace(",", "."))
        ax.set_title(f"{v.capitalize()}, {fx.replace('-', '–')} anos", fontsize=11, loc="left")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
for ax in axs[1]:
    ax.set_xlabel("Anos após a alta")
for ax in axs[:, 0]:
    ax.set_ylabel("Reoperação valvar no SUS (%)")
fig.suptitle("Incidência acumulada de reoperação por tipo de prótese (piso; Kaplan-Meier)", fontsize=12, x=0.01,
             ha="left")
fig.tight_layout()
fig.savefig(f"{R}/fig_reoperacao_jovens.png", dpi=200)
open(f"{R}/resumo_jovens.txt", "w").write("\n".join(ro.txt) + "\n")
