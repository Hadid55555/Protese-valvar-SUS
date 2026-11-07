#!/usr/bin/env python3
"""Prótese valvar 2000–2007: SIH-SP (OPM) + SIH-RD das mesmas AIH, tabela SIH antiga.

Dicionário (TUNEP/ANS, cópia pública rmxvgit/susprocessing/dados/TUNEP.csv; conferido no piloto SP-SP 2000–2007):
  OPM (iguais em 2000–2007): 93323131 prótese valvular biológica; 93329156 biológica sem suporte ou anel;
       93323158 mecânica de duplo folheto; 93323166 mecânica de baixo perfil (disco).
       São os equivalentes dos SIGTAP 0702040541/0702040550 (bio) e 0702040568/0702040576 (mec).
  Procedimentos (tabela 48, jul/2004–2007): 48010138 implante de prótese valvar; 48010146 plástica e/ou troca múltipla;
       48010154 troca valvar com RM; 48010120 plástica; 48010162 Ross; 48010243 tubo valvado.
       Antes de jul/2004 (tabela 32) o código não está na TUNEP; a AIH entra pela OPM (32020015 é o mais frequente).
Etapas: sp  → saida_protese/pre2008/sp/parciais (linhas SP com OPM valvar);
        rd  → saida_protese/pre2008/rd/parciais (RD das AIH achadas no SP).
Uso: python protese_valvar/extrair_pre2008.py sp --workers 6 ; depois ... rd --workers 6
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
OPM = ["93323131", "93329156", "93323158", "93323166"]
etapa = sys.argv[1]
resto = sys.argv[2:]

if etapa == "sp":
    import extrair_valvas_sp as sp  # noqa: E402

    def filtro(t):
        ato = pc.utf8_trim_whitespace(pc.cast(t["SP_ATOPROF"], pa.string()))
        return t.filter(pc.fill_null(pc.is_in(ato, value_set=pa.array(OPM)), False))

    sp.filtro = filtro
    sys.argv = [sys.argv[0], "--ini", "2000", "--cmpt-fim", "200712", "--saida", "saida_protese/pre2008/sp"] + resto
    sp.main()
elif etapa == "rd":
    import extrair_valvas_sih as rd  # noqa: E402
    base = rd.base
    base.COLUNAS = ["ANO_CMPT", "MES_CMPT", "N_AIH", "PROC_REA", "DT_INTER", "DT_SAIDA", "NASC", "SEXO", "IDADE",
                    "COD_IDADE", "DIAG_PRINC", "DIAG_SECUN", "MORTE", "CNES", "CGC_HOSP", "MUNIC_RES", "MUNIC_MOV",
                    "CAR_INT", "RACA_COR", "CEP", "UF_ZI"]
    fs = [f for f in glob.glob("saida_protese/pre2008/sp/parciais/*.csv") if os.path.getsize(f) > 0]
    alvo = pd.concat([pd.read_csv(f, dtype=str, usecols=["SP_NAIH"]) for f in fs]).SP_NAIH.str.strip().unique()
    ALVO = pa.array(alvo)
    print(f"AIH-alvo (SP com OPM valvar): {len(alvo):,}", flush=True)

    def filtro(t):
        n = pc.utf8_trim_whitespace(pc.cast(t["N_AIH"], pa.string()))
        return t.filter(pc.fill_null(pc.is_in(n, value_set=ALVO), False))

    rd.filtro = filtro
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(resto)
    args = argparse.Namespace(ini=2000, fim=2008, cmpt_fim=200812, saida="saida_protese/pre2008/rd", fonte="espelho",
                              workers=a.workers)
    t = base.extrair(args)
    print(base.cobertura(args, t)["status"].value_counts().to_string())
