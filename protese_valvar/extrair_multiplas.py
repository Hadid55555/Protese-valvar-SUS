#!/usr/bin/env python3
"""Re-extração das AIH de cirurgias múltiplas/sequenciais (0415010012/0415020034) com CID cardiovascular.

Motivo: na extração do projeto TAVI, as linhas de OPM (prótese) dessas AIH só entraram quando o CID principal era
aórtico; as múltiplas com CID mitral (I05/I34), I08 etc. ficaram sem a prótese no SP e ausentes do RD.

Filtro por linha (aplicado a cada row group; o recorte fino por AIH é feito na consolidação):
  - SP: SP_PROCREA em MULTIPLAS e SP_CIDPRI começando com "I" ou "Q2" (doenças do aparelho circulatório e
    malformações congênitas do coração). Guarda TODAS as linhas dessas AIH (atos e OPM).
  - RD: PROC_REA em MULTIPLAS e DIAG_PRINC começando com "I" ou "Q2".
Na consolidação fica a AIH que tem ato valvar (PROCS de tavi_savr/extrair_valvas_sp.py) ou OPM de prótese valvar.

Reaproveita os extratores do projeto TAVI (espelho healthbr-data, SHA-256, parciais retomáveis, _cobertura.jsonl).

Uso:
  python protese_valvar/extrair_multiplas.py sp --saida saida_protese/sp --workers 3
  python protese_valvar/extrair_multiplas.py rd --saida saida_protese/rd --workers 3
"""
import os
import sys

import pyarrow as pa
import pyarrow.compute as pc

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tavi_savr"))

MULTIPLAS = pa.array(["0415010012", "0415020034"])


def cid_cardio(cid):
    return pc.or_(pc.starts_with(cid, "I"), pc.starts_with(cid, "Q2"))


def filtro_sp(t):
    proc = pc.utf8_trim_whitespace(t["SP_PROCREA"])
    cid = pc.utf8_trim_whitespace(t["SP_CIDPRI"])
    m = pc.and_(pc.is_in(proc, value_set=MULTIPLAS), cid_cardio(cid))
    return latin1(t.filter(pc.fill_null(m, False)))


def latin1(t):
    """Alguns arquivos SP têm bytes inválidos em UTF-8 (ex.: SP_PJ_DOC) que quebram o to_pandas; decodifica latin-1."""
    cols = []
    for nome in t.column_names:
        c = t[nome]
        if pa.types.is_string(c.type) or pa.types.is_large_string(c.type):
            b = c.cast(pa.binary()).to_pylist()
            c = pa.array([None if v is None else v.decode("utf-8", errors="replace") for v in b], pa.string())
        cols.append(c)
    return pa.table(cols, names=t.column_names)


OPM_VALVA = ["0702040541", "0702040550", "0702040568", "0702040576"]


def por_aih(t):
    """Mantém só as AIH (SP_NAIH) com ato valvar (PROCS do SP do projeto TAVI) ou OPM de prótese valvar."""
    import extrair_valvas_sp as sp
    ato = pc.utf8_trim_whitespace(t["SP_ATOPROF"])
    alvo = pa.array(sp.PROCS + OPM_VALVA)
    naih = pc.unique(t.filter(pc.fill_null(pc.is_in(ato, value_set=alvo), False))["SP_NAIH"])
    return t.filter(pc.fill_null(pc.is_in(t["SP_NAIH"], value_set=naih), False))


def filtro_rd(t):
    proc = pc.utf8_trim_whitespace(t["PROC_REA"])
    diag = pc.utf8_trim_whitespace(t["DIAG_PRINC"])
    m = pc.and_(pc.is_in(proc, value_set=MULTIPLAS), cid_cardio(diag))
    return t.filter(pc.fill_null(m, False))


if __name__ == "__main__":
    modo = sys.argv.pop(1)
    if modo == "sp":
        import extrair_valvas_sp as sp
        sp.filtro = filtro_sp

        class PaPorAih:
            """pyarrow com concat_tables que aplica o recorte por AIH no arquivo inteiro (após juntar os row groups)."""
            def __getattr__(self, nome):
                return getattr(pa, nome)

            @staticmethod
            def concat_tables(partes, **kw):
                return por_aih(pa.concat_tables(partes, **kw))
        sp.pa = PaPorAih()
        sp.main()
    elif modo == "rd":
        import argparse
        import extrair_valvas_sih as rd
        rd.filtro = filtro_rd
        base = rd.base
        ap = argparse.ArgumentParser()
        ap.add_argument("--ini", type=int, default=2008)
        ap.add_argument("--fim", type=int, default=2025)
        ap.add_argument("--cmpt-fim", type=int, default=202608)
        ap.add_argument("--saida", default="saida_protese/rd")
        ap.add_argument("--workers", type=int, default=3)
        a = ap.parse_args()
        a.fonte = "espelho"
        tarefas = base.extrair(a)
        print(base.cobertura(a, tarefas)["status"].value_counts().to_string())
    else:
        sys.exit("modo: sp ou rd")
