#!/usr/bin/env python3
"""Re-extrai do SIH-RD as colunas de ligação (NASC, SEXO, CEP, MUNIC_RES) das AIH já presentes na base valvar.

Serve para ligar internações da mesma pessoa (reoperação valvar) com chave determinística NASC+SEXO+CEP
(sensibilidade: NASC+SEXO+MUNIC_RES). Os dados de ligação ficam fora do git (pseudoidentificadores).

Uso: python protese_valvar/extrair_chave_rd.py --saida saida_protese/chave --workers 3
"""
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tavi_savr"))
import extrair_valvas_sih as rd  # noqa: E402

base = rd.base
base.COLUNAS = ["ANO_CMPT", "MES_CMPT", "N_AIH", "PROC_REA", "DT_INTER", "DT_SAIDA", "NASC", "SEXO", "CEP",
                "MUNIC_RES", "IDADE", "CNES", "MORTE", "RACA_COR"]
ALVO = pa.array(pd.read_csv("saida_protese/rd_valvas_consolidado.csv", dtype=str, usecols=["N_AIH"]).N_AIH.unique())


def filtro(t):
    return t.filter(pc.fill_null(pc.is_in(pc.utf8_trim_whitespace(t["N_AIH"]), value_set=ALVO), False))


rd.filtro = filtro

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ini", type=int, default=2008)
    ap.add_argument("--fim", type=int, default=2025)
    ap.add_argument("--cmpt-fim", type=int, default=202608)
    ap.add_argument("--saida", default="saida_protese/chave")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    a.fonte = "espelho"
    t = base.extrair(a)
    print(base.cobertura(a, t)["status"].value_counts().to_string())
