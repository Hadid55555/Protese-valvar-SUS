# Mechanical versus biological valve prostheses in Brazil's public health system, 2000–2025

Code and aggregated results for the manuscript *"Mechanical versus biological valve prostheses in Brazil's public
health system, 2000–2025: a nationwide analysis of 305 566 admissions"* (Hadid L, Salek L, Passos FS, Caldonazo T;
submitted to the European Journal of Cardio-Thoracic Surgery).

## Data
- **Source:** public microdata of the Brazilian Hospital Information System (SIH/SUS, DATASUS).
  - RD files: reduced admission records.
  - SP files: professional services and billed items, including orthoses, prostheses and special materials (OPM).
- **Period:** January 2000 to December 2025.
- **Mirror:** the official FTP server was not reachable, so files were read from the healthbr-data public mirror
  (https://data.sidneybissoli.com, parquet conversion of the original `.dbc` files).
- **Verification:** every file was checked by SHA-256. `inputs_sha256.csv` lists, for each state-month partition, the
  original DATASUS file name, the mirror path and the hash.
- **Not in this repository:** no microdata are distributed here. Everything can be rebuilt from the public files with
  the code below. Results are aggregated, and hospitals appear only under pseudonymous IDs.

**Prosthesis codes**

| Type | 2008–2025 (SIGTAP) | 2000–2007 (former SIH table; TUNEP/ANS) |
|---|---|---|
| Biological, stented / stentless | 0702040541 / 0702040550 | 93323131 / 93329156 |
| Mechanical, bileaflet / tilting disc | 0702040568 / 0702040576 | 93323158 / 93323166 |

## Reproducing the analysis
Run all scripts from the repository root (Python ≥3.11; `pip install -r requirements.txt`; pandoc for `.docx`).
Scripts that import `reoperacao` need `PYTHONPATH=protese_valvar`.

**1. 2008–2025 extraction** (downloads of several GB; resumable)
```
python tavi_savr/extrair_valvas_sih.py --ini 2008 --fim 2025 --cmpt-fim 202608 --saida saida_tavi --workers 6
python tavi_savr/extrair_valvas_sp.py  --ini 2008 --cmpt-fim 202608 --saida saida_tavi/sp --workers 6
python protese_valvar/extrair_multiplas.py sp --saida saida_protese/sp --workers 3
python protese_valvar/extrair_multiplas.py rd --saida saida_protese/rd --workers 3
```
- `consolidar_bases.py` reads the RD and SP extracts as two zipped CSVs:
  - `saida_tavi/sih_valvas_2008_2026_extrato.zip` (RD);
  - `protese_valvar/sih_sp_valvas_atos_opm_2008_2026.zip` (SP; group 0406 acts and valve OPM rows, professional ID
    removed).
- Build these two zips from the partial outputs of the two extractors above.

**2. Consolidation and classification**
```
python protese_valvar/consolidar_bases.py      # -> saida_protese/aih_protese.csv
python protese_valvar/classificar_valva.py     # -> saida_protese/aih_protese_classificada.csv
```

**3. 2000–2007 extension and the 2000–2025 series**
```
python protese_valvar/extrair_pre2008.py sp --workers 6
python protese_valvar/extrair_pre2008.py rd --workers 6
python protese_valvar/serie_2000_2025.py
```

**4. Analyses, tables and figures**

| Script | Content | Output folder |
|---|---|---|
| `analises_v2.py` | break point, supplier, multilevel, funnel | `resultados_v2/` |
| `auditoria_v1.py` | valve-coding audits | `resultados_auditoria/` |
| `artigo1_extra.py` | interrupted time series, placebos, volume, hospital fixed effects | `resultados_artigo1/` |
| `sensibilidades_v1.py` | sensitivity analyses | `resultados_sensibilidade/` |
| `manuscrito_tabelas.py` | Tables 1–3, Figures 1–4 | `manuscrito/` |
| `suplemento_tabelas.py` | Tables S1–S9 (English labels) | `manuscrito/supp/` |
| `montar_manuscrito.py` | manuscript and supplement (.docx); the manuscript text will be added after publication | `manuscrito/` |

**5. Exploratory Table S7** (SUS-captured reintervention and complications; not used for conclusions)
- `extrair_chave_rd.py` and `extrair_internacoes_coorte.py` link admissions by a deterministic pseudonymous key.
  The key and the linked data are never published.
- `reoperacao.py`, `reoperacao_jovens.py` and `complicacoes.py` produce Table S7.

## Main code files
- `sih_crvm.py`: download, checksum and resumable extraction from the mirror.
- `tavi_savr/extrair_valvas_*.py`: RD and SP extractors.
- `protese_valvar/classificar_valva.py`: valve position and aetiology from ICD-10. Q22.5 and I38 are ignored in adults.

## Citation
See `CITATION.cff`. A versioned archive with a DOI is available through Zenodo (see the badge or release page once
created).

## License
Code: MIT License (see `LICENSE`). Aggregated results: CC BY 4.0.
