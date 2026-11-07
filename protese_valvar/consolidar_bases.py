#!/usr/bin/env python3
"""Consolida a base analítica de prótese valvar (SIH-SP + SIH-RD), 2008–2025 por ano da internação.

Entradas
  SP antigo  : protese_valvar/sih_sp_valvas_atos_opm_2008_2026.zip (projeto TAVI)
  SP múltiplas: saida_protese/sp/parciais/*.csv  (extrair_multiplas.py sp)
  RD antigo  : saida_tavi/sih_valvas_2008_2026_extrato.zip
  RD múltiplas: saida_protese/rd/parciais/*.csv  (extrair_multiplas.py rd)

Regra de junção
  SP: do antigo saem as linhas de AIH 0415* (substituídas pelo novo, que tem todas as linhas dessas AIH, inclusive
      a OPM); o restante do antigo fica igual. Duplicatas exatas removidas.
  RD: antigo + novo, duplicatas exatas removidas.

Saídas (saida_protese/)
  sp_valvas_consolidado.csv(.zip), rd_valvas_consolidado.csv(.zip), aih_protese.csv: uma linha por AIH com OPM de
  prótese valvar (nº de OPM bio/mec, CID, procedimento, CNES, datas) + campos do RD (idade, sexo, óbito...).
  conferencia.txt: contagens independentes (parciais crus × consolidado).
"""
import glob
import os
import zipfile

import pandas as pd

S = "saida_protese"
MULT = {"0415010012", "0415020034"}
BIO = {"0702040541", "0702040550"}
MEC = {"0702040568", "0702040576"}
out = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


def ler_zip(path, nome=None):
    z = zipfile.ZipFile(path)
    nome = nome or [n for n in z.namelist() if n.endswith(".csv") and "parciais" not in n][0]
    return pd.read_csv(z.open(nome), dtype=str, keep_default_na=False)


def ler_parciais(pasta):
    fs = sorted(glob.glob(os.path.join(pasta, "*.csv")))
    dfs = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in fs if os.path.getsize(f) > 0]
    dfs = [d for d in dfs if len(d)]
    return pd.concat(dfs, ignore_index=True), len(fs)


# ---------------- SP
sp_old = ler_zip("protese_valvar/sih_sp_valvas_atos_opm_2008_2026.zip")
sp_new, nf = ler_parciais(f"{S}/sp/parciais")
log(f"SP antigo: {len(sp_old):,} linhas; SP múltiplas novo: {len(sp_new):,} linhas de {nf} arquivos")
for d in (sp_old, sp_new):
    for c in d.columns:
        d[c] = d[c].str.strip()
old_mult = sp_old[sp_old.SP_PROCREA.isin(MULT)]
log(f"  linhas 0415 no antigo (substituídas): {len(old_mult):,}")
sp = pd.concat([sp_old[~sp_old.SP_PROCREA.isin(MULT)], sp_new], ignore_index=True)
# Sem deduplicação: no SP, linhas idênticas são legítimas (CNS do profissional removido; 2 próteses = 2 linhas).
log(f"SP consolidado: {len(sp):,} linhas")

# AIH 0415 com OPM no antigo que não reaparecem no novo (perda pela regra CID I/Q2?)
opm_old_mult = set(old_mult.loc[old_mult.SP_ATOPROF.isin(BIO | MEC), "SP_NAIH"])
opm_new = set(sp_new.loc[sp_new.SP_ATOPROF.isin(BIO | MEC), "SP_NAIH"])
log(f"  AIH 0415 com OPM: antigo {len(opm_old_mult):,}; novo {len(opm_new):,}; do antigo ausentes no novo: "
    f"{len(opm_old_mult - opm_new):,}")

# ---------------- RD
rd_old = ler_zip("saida_tavi/sih_valvas_2008_2026_extrato.zip", "sih_valvas_2008_2026_extrato.csv")
rd_new, nf = ler_parciais(f"{S}/rd/parciais")
log(f"RD antigo: {len(rd_old):,}; RD múltiplas novo: {len(rd_new):,} de {nf} arquivos")
rd = pd.concat([rd_old, rd_new], ignore_index=True)
rd = rd.drop_duplicates(subset=[c for c in rd.columns if c != "arquivo_origem"] + ["arquivo_origem"])
log(f"RD consolidado: {len(rd):,} registros; N_AIH distintos {rd.N_AIH.nunique():,}")

# ---------------- AIH com OPM
sp["ano_inter"] = sp.SP_DTINTER.str[:4]
g = sp.groupby("SP_NAIH")
opm = sp[sp.SP_ATOPROF.isin(BIO | MEC)].copy()
opm["q"] = pd.to_numeric(opm.SP_QTD_ATO, errors="coerce").fillna(1).clip(lower=1)
opm["bio"] = opm.SP_ATOPROF.isin(BIO) * opm.q
opm["mec"] = opm.SP_ATOPROF.isin(MEC) * opm.q
# AIH reapresentada aparece em mais de uma competência (arquivo): soma dentro do arquivo, máximo entre arquivos
a = (opm.groupby(["SP_NAIH", "arquivo_origem"])[["bio", "mec"]].sum().groupby(level=0).max()
     .rename(columns={"bio": "n_bio", "mec": "n_mec"}))
info = g.agg(uf=("SP_UF", "first"), cnes=("SP_CNES", "first"), proc_aih=("SP_PROCREA", "first"),
             dt_inter=("SP_DTINTER", "min"), dt_saida=("SP_DTSAIDA", "max"), cid_pri=("SP_CIDPRI", "first"),
             cid_sec=("SP_CIDSEC", "first"), mun_pac=("SP_M_PAC", "first"),
             atos=("SP_ATOPROF", lambda s: " ".join(sorted(set(x for x in s if x.startswith("0406"))))))
a = a.join(info, how="left")
a["ano"] = a.dt_inter.str[:4]
a["protese"] = (a.n_bio > 0).map({True: "bio", False: ""}) + (a.n_mec > 0).map({True: "mec", False: ""})
a["protese"] = a.protese.replace({"biomec": "ambas"})

rdk = rd.sort_values(["N_AIH", "ANO_CMPT", "MES_CMPT"]).drop_duplicates("N_AIH", keep="last").set_index("N_AIH")
keep = [c for c in ["IDADE", "COD_IDADE", "NASC", "SEXO", "RACA_COR", "MORTE", "DIAS_PERM", "UTI_MES_TO", "CAR_INT",
                    "DIAG_PRINC", "DIAG_SECUN"] + [f"DIAGSEC{i}" for i in range(1, 10)] +
        ["MUNIC_RES", "MUNIC_MOV", "CNES", "PROC_REA", "VAL_TOT", "NAT_JUR", "GESTAO", "COMPLEX", "FINANC"]
        if c in rdk.columns]
a = a.join(rdk[keep].add_prefix("rd_"), how="left")
a["no_rd"] = a.rd_PROC_REA.notna()
a.index.name = "N_AIH"
a.to_csv(f"{S}/aih_protese.csv")

b = a[a.ano.between("2008", "2025")]
log(f"\nAIH com OPM de prótese valvar, internação 2008–2025: {len(b):,}")
log(b.protese.value_counts().to_string())
log(f"Ligadas ao RD: {b.no_rd.mean():.1%}")
log("Ligação ao RD por procedimento da AIH:")
log(b.groupby(b.proc_aih.where(b.proc_aih.isin(MULT | {"0406010692", "0406010820", "0406011206", "0406010846"}),
                               "outros")).no_rd.agg(["size", "mean"]).to_string())

# ---------------- conferência independente: recontagem direta dos parciais crus do SP novo
cru = 0
for f in glob.glob(f"{S}/sp/parciais/*.csv"):
    if os.path.getsize(f) == 0:
        continue
    d = pd.read_csv(f, dtype=str, usecols=["SP_NAIH", "SP_ATOPROF"], keep_default_na=False)
    cru += d.loc[d.SP_ATOPROF.str.strip().isin(BIO | MEC), "SP_NAIH"].nunique()
log(f"\nConferência: AIH com OPM nos parciais crus (soma por arquivo) = {cru:,}; no consolidado (0415) = "
    f"{a.proc_aih.isin(MULT).sum():,} (diferença = AIH presentes em >1 competência)")

for nome, df in (("sp_valvas_consolidado", sp.drop(columns="ano_inter")), ("rd_valvas_consolidado", rd)):
    p = f"{S}/{nome}.csv"
    df.to_csv(p, index=False)
    with zipfile.ZipFile(p.replace(".csv", ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p, os.path.basename(p))
with zipfile.ZipFile(f"{S}/aih_protese.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{S}/aih_protese.csv", "aih_protese.csv")
open(f"{S}/conferencia_consolidacao.txt", "w").write("\n".join(out) + "\n")
