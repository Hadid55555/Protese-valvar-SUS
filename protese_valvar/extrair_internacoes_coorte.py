#!/usr/bin/env python3
"""SIH-RD: TODAS as internações (qualquer procedimento/CID) das pessoas da coorte valvar, 2008–ago/2026.

Ligação determinística pela chave NASC|SEXO|CEP das AIH valvares (saida_protese/chave). Serve para desfechos após a
alta: AVC hemorrágico/isquêmico, embolia, sangramento, trombose/disfunção/endocardite de prótese, gestação/parto e
"prova de vida" (qualquer internação posterior). Pseudoidentificadores ficam fora do git.

Uso: python protese_valvar/extrair_internacoes_coorte.py --saida saida_protese/coorte_rd --workers 3
"""
import glob
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
base.COLUNAS = (["ANO_CMPT", "MES_CMPT", "N_AIH", "IDENT", "PROC_REA", "DT_INTER", "DT_SAIDA", "NASC", "SEXO", "CEP",
                 "MUNIC_RES", "IDADE", "COD_IDADE", "CNES", "MORTE", "CAR_INT", "DIAG_PRINC", "DIAG_SECUN", "CID_MORTE",
                 "DIAS_PERM", "RACA_COR"] + [f"DIAGSEC{i}" for i in range(1, 10)])
ch = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["NASC", "SEXO", "CEP"])
                for f in glob.glob("saida_protese/chave/parciais/*.csv") if os.path.getsize(f) > 0])
CHAVES = pa.array((ch.NASC.str.strip() + "|" + ch.SEXO.str.strip() + "|" + ch.CEP.str.strip()).unique())


def filtro(t):
    k = pc.binary_join_element_wise(pc.utf8_trim_whitespace(t["NASC"]), pc.utf8_trim_whitespace(t["SEXO"]),
                                    pc.utf8_trim_whitespace(t["CEP"]), "|")
    return t.filter(pc.fill_null(pc.is_in(k, value_set=CHAVES), False))


rd.filtro = filtro

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ini", type=int, default=2008)
    ap.add_argument("--fim", type=int, default=2025)
    ap.add_argument("--cmpt-fim", type=int, default=202608)
    ap.add_argument("--saida", default="saida_protese/coorte_rd")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    a.fonte = "espelho"
    print(f"{len(CHAVES):,} chaves", flush=True)
    t = base.extrair(a)
    print(base.cobertura(a, t)["status"].value_counts().to_string())
