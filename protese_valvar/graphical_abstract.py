#!/usr/bin/env python3
"""Graphical abstract (EJCTS): key question, two panels of results and take-home message."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZ, LA, CZ = "#1f6f8b", "#c9733a", "#555555"
s = pd.read_parquet("saida_protese/serie_2000_2025.parquet")
s = s[(s.protese != "ambas") & s.valva.isin(["aórtica", "mitral"]) & (s.ano >= 2002)]
fu = pd.read_csv("protese_valvar/resultados_v2/M2_funnel_18_59_2020_2025.csv")

fig = plt.figure(figsize=(13.33, 7.5))
fig.patch.set_facecolor("white")
fig.text(0.03, 0.93, "Mechanical versus biological valve prostheses in Brazil's public health system, 2000–2025",
         fontsize=17, weight="bold", color="#222222")
fig.text(0.03, 0.885, "Nationwide hospital-billing data (SIH/SUS) · 305 566 admissions · prosthesis type from the billed "
         "device code", fontsize=11.5, color=CZ)
fig.text(0.03, 0.82, "Key question", fontsize=12, weight="bold", color=AZ)
fig.text(0.03, 0.785, "How has the choice between mechanical and biological prostheses changed in a middle-income "
         "country with a high burden of rheumatic disease?", fontsize=11.5, color="#222222")

ax1 = fig.add_axes([0.06, 0.25, 0.4, 0.44])
y = s[(s.idade >= 18) & (s.idade < 60)]
for v, lab, cor in (("aórtica", "Aortic", AZ), ("mitral", "Mitral", LA)):
    g = y[y.valva == v].groupby("ano").protese.apply(lambda x: 100 * (x == "mec").mean())
    ax1.plot(g.index, g.values, marker="o", ms=3.5, lw=2, color=cor, label=lab)
ax1.axvline(2020, color="grey", ls="--", lw=1)
ax1.text(2020.3, 31, "2020", color="grey", fontsize=10)
ax1.set_ylim(30, 62)
ax1.set_ylabel("Mechanical prostheses, %", fontsize=11)
ax1.set_title("Patients aged 18–59 years", fontsize=12, loc="left", weight="bold")
ax1.legend(frameon=False, fontsize=10, loc="upper left")
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)
ax1.text(0.0, -0.22, "Level change after 2020: OR 1.50 (95% CI 1.28–1.77); within hospitals OR 1.64",
         transform=ax1.transAxes, fontsize=10.5, color="#222222")

ax2 = fig.add_axes([0.56, 0.25, 0.4, 0.44])
p0 = (fu.obs * fu.n).sum() / fu.n.sum()
import numpy as np
nn = np.linspace(fu.n.min(), fu.n.max(), 200)
fora = (fu.ajust > fu.lim998_sup) | (fu.ajust < fu.lim998_inf)
ax2.scatter(fu.n[~fora], 100 * fu.ajust[~fora], s=14, color="#9bb7c4")
ax2.scatter(fu.n[fora], 100 * fu.ajust[fora], s=14, color=AZ)
for z in (3.09,):
    ax2.plot(nn, 100 * (p0 + z * np.sqrt(p0 * (1 - p0) / nn)), "k--", lw=0.8)
    ax2.plot(nn, 100 * (p0 - z * np.sqrt(p0 * (1 - p0) / nn)), "k--", lw=0.8)
ax2.axhline(100 * p0, color="k", lw=0.8)
ax2.set_xlabel("Admissions per hospital, 2020–2025", fontsize=11)
ax2.set_ylabel("Adjusted mechanical share, %", fontsize=11)
ax2.set_title("Variation between 171 hospitals", fontsize=12, loc="left", weight="bold")
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
ax2.text(0.0, -0.24, "Median odds ratio 3.15; 98 hospitals outside 99.8% limits", transform=ax2.transAxes,
         fontsize=10.5, color="#222222")

fig.text(0.03, 0.075, "Take-home message", fontsize=12, weight="bold", color=AZ)
fig.text(0.03, 0.035, "Mechanical prosthesis use did not decline as in high-income countries, rose in younger "
         "patients after 2020, and depended strongly on the hospital and region.", fontsize=11.5, color="#222222")
fig.savefig("protese_valvar/manuscrito/Graphical_Abstract.png", dpi=300)
