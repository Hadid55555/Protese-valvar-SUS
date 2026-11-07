# Supplementary Methods

## Data sources and coverage
- SIH-RD (reduced admission file) and SIH-SP (professional services/billed items), all states, January 2000 to
  December 2025, by admission date. Files were obtained from the healthbr-data public mirror of DATASUS (parquet
  conversion of the original .dbc files; SHA-256 checked for each file).
- Missing files (not available in the mirror): Roraima, January to May 2000 (RD and SP); Amapá, October 2007 (RD
  and SP); Acre and Roraima, August 2026 (outside the study period).
- RD and SP were linked by the admission number (N_AIH). For 2000–2007, an RD record was found for 96.7% of the
  admissions identified in the SP file.

## Prosthesis codes
| Type | 2008–2025 (SIGTAP) | 2000–2007 (former SIH table; TUNEP) |
|---|---|---|
| Biological, stented | 0702040541 | 93323131 |
| Biological, stentless | 0702040550 | 93329156 |
| Mechanical, bileaflet | 0702040568 | 93323158 |
| Mechanical, tilting disc | 0702040576 | 93323166 |

- Continuity: the mechanical share was 33.5% in 2007 and 33.9% in 2008.
- In adults with aortic or mitral replacement, 2006–2007 vs 2008–2009: aortic 33.6% vs 34.8%; mitral 33.3% vs 36.0%.

## Validation of prosthesis and valve coding
- OPM recording: a prosthesis was billed in 93–98% of valve-replacement admissions. Prosthesis type agreed with a
  published single-centre series (Hospital Ana Nery, Salvador).
- Default diagnostic codes: hospital-years in which one code accounted for ≥70% of admissions (5.1% of admissions;
  mostly I05) were excluded in a sensitivity analysis.
- In one reference centre, the valve position could not be derived from the codes.
- Coded rheumatic aetiology underestimated the proportion reported by a reference centre (32% vs 58.5%).

## Statistical details
- **Mixed model:** BinomialBayesMixedGLM (statsmodels 0.14), variational Bayes. The median odds ratio is
  MOR = exp(√(2σ²) × 0.6745).
- **Interrupted time series:** binomial GLM of monthly counts with time, level and slope terms after the break, and
  calendar-month indicators. Standard errors are HAC with 12 lags; Pearson dispersion was reported.
- **Funnel plot:** observed/expected proportions from the fixed-effects model, multiplied by the national mean.
  Limits are binomial at 95% and 99.8%.

## RECORD checklist (abbreviated; items mapped to the manuscript)
| Item | Where addressed |
|---|---|
| 1.1–1.3 Type of data, databases and linkage in title/abstract | Title; Abstract (Methods) |
| 6.1 Methods of population selection (codes, algorithms) | Methods (study population; codes table above) |
| 6.2 Validation studies of codes | Methods (validation); Supplementary Methods |
| 6.3 Linkage | Methods (RD–SP by admission number); no person-level linkage in main analyses |
| 7.1 Codes and algorithms for exposures, outcomes, confounders | Methods; Supplementary Methods |
| 12.1 Extent of access to the database | Methods (public files) |
| 12.2 Data cleaning methods | Methods; Supplementary Methods |
| 12.3 Linkage methods and quality | Supplementary Methods (RD found for 96.7% of 2000–2007 admissions) |
| 13.1 Selection of the population (flowchart) | Figure S1 |
| 19.1 Limitations specific to routinely collected data | Discussion (Limitations) |
| 22.1 Access to protocol, raw data and code | Data availability statement |
