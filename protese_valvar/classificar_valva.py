#!/usr/bin/env python3
"""Classificação preliminar da valva, etiologia e faixas (v0) sobre saida_protese/aih_protese.csv.

Valva: nº de próteses ≥2 → dupla/múltipla; senão pelo 1º CID valvar encontrado em: CID principal do SP, DIAG_PRINC do
RD, CID secundário do SP e DIAGSEC1–9 do RD. Aórtica I35/I06/Q23.0/Q23.1; mitral I34/I05/Q23.2/Q23.3; I08 múltipla;
tricúspide/pulmonar I36/I37/I07/Q22. Sem CID valvar: tubo valvado (0406010846) → aórtica; restante → indeterminada.
Em adultos, Q22.5 e I38 são ignorados (CID coringa; auditoria da sessão valvar reumática).
Etiologia (do mesmo CID): reumática I05–I09; não reumática I34–I37; congênita Q22–Q23; endocardite I33 (se houver).
"""
import pandas as pd

a = pd.read_csv("saida_protese/aih_protese.csv", dtype=str, keep_default_na=False)
a = a[a.ano.between("2008", "2025")].copy()
for c in ("n_bio", "n_mec"):
    a[c] = pd.to_numeric(a[c]).astype(int)
a["n_prot"] = a.n_bio + a.n_mec


def valva_cid(c):
    c = c.strip().upper()
    if c[:3] in ("I35", "I06") or c[:4] in ("Q230", "Q231"):
        return "aórtica", "reumática" if c[:3] == "I06" else ("congênita" if c[0] == "Q" else "não reumática")
    if c[:3] in ("I34", "I05") or c[:4] in ("Q232", "Q233"):
        return "mitral", "reumática" if c[:3] == "I05" else ("congênita" if c[0] == "Q" else "não reumática")
    if c[:3] == "I08":
        return "múltipla", "reumática"
    if c[:3] in ("I36", "I37", "I07", "Q22"):
        return "tric/pulm", "reumática" if c[:3] == "I07" else ("congênita" if c[0] == "Q" else "não reumática")
    return None


fontes = ["cid_pri", "rd_DIAG_PRINC", "cid_sec"] + [f"rd_DIAGSEC{i}" for i in range(1, 10)]
fontes = [f for f in fontes if f in a.columns]


CORINGA_ADULTO = ("Q225", "I38")  # Q22.5 (Ebstein) e I38 usados como coringa em adultos (auditoria da sessão reumática)


def classif(r):
    adulto = pd.to_numeric(r.rd_IDADE, errors="coerce") >= 18 and r.rd_COD_IDADE in ("4", "")
    endo = any(r[f].startswith("I33") for f in fontes)
    for f in fontes:
        if adulto and r[f].strip().upper()[:4].startswith(CORINGA_ADULTO):
            continue
        v = valva_cid(r[f])
        if v:
            valva, etio = v
            break
    else:
        valva, etio = ("aórtica", "indeterminada") if r.proc_aih == "0406010846" else ("indeterminada", "indeterminada")
    if r.n_prot >= 2 or r.protese == "ambas":
        valva = "dupla/múltipla"
    if endo:
        etio = "endocardite"
    return valva, etio


a[["valva", "etiologia"]] = a.apply(lambda r: pd.Series(classif(r)), axis=1)
a["idade"] = pd.to_numeric(a.rd_IDADE, errors="coerce").where(a.rd_COD_IDADE.isin(["4", ""]))
a["sexo"] = a.rd_SEXO.map({"1": "M", "3": "F", "2": "F"})
a["mec"] = (a.n_mec > 0) & (a.n_bio == 0)
a.to_csv("saida_protese/aih_protese_classificada.csv", index=False)
