#!/usr/bin/env python3
"""Figure 1 (flowchart) from manuscrito/Flowchart_counts.csv (written by manuscrito_tabelas.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

M = "protese_valvar/manuscrito"
fl = pd.read_csv(f"{M}/Flowchart_counts.csv")
fig, ax = plt.subplots(figsize=(7, 5.2))
ax.axis("off")
EN = {"indeterminada": "position not coded", "dupla/múltipla": "two or more prostheses", "tric/pulm": "tricuspid/pulmonary",
      "múltipla": "multivalve code (I08)"}
top = f"Admissions billing a valve prosthesis (OPM), 2008–2025\nn = {fl.n.iloc[0]:,}"
tot = fl.n.iloc[0]
exc = "\n".join(f"{(r.step.replace('Excluded: ', '') if 'valve position' not in r.step else EN.get(r.step.split('valve position ')[1], r.step))}: {r.n:,} ({100 * r.n / tot:.1f}%)"
                for r in fl.iloc[1:-1].itertuples())
bot = f"Analysis cohort: adults, aortic or mitral position,\none prosthesis type\nn = {fl.n.iloc[-1]:,}"
kw = dict(ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="black"))
ax.text(0.35, 0.9, top, **kw)
ax.text(0.35, 0.12, bot, **kw)
ax.text(0.8, 0.5, "Excluded\n" + exc, ha="center", va="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.6", fc="#f2f2f2", ec="black"))
ax.annotate("", xy=(0.35, 0.2), xytext=(0.35, 0.82), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(0.6, 0.5), xytext=(0.35, 0.5), arrowprops=dict(arrowstyle="->"))
fig.savefig(f"{M}/Figure1_flowchart.png", dpi=300, bbox_inches="tight")
plt.close(fig)
