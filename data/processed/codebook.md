# Analytic dataset codebook

Rows: 41,151  |  one row per eligible respondent (ages 30-79, ELIGSTAT==1).


| variable | source | units | % missing | derivation |
|---|---|---|---|---|
| SEQN | DEMO / LMF | id | 0.0% | Respondent sequence number (merge key) |
| cycle | derived | - | 0.0% | NHANES 2-year cycle label |
| cycle_startyr | derived | year | 0.0% | Cycle start year (1999..2017) |
| age | DEMO RIDAGEYR | years | 0.0% | Age at exam; cohort restricted to 30-79 |
| sex | DEMO RIAGENDR | code | 0.0% | 1=male, 2=female |
| race | DEMO RIDRETH1 | code | 0.0% | 1=MexAm 2=OthHisp 3=NHWhite 4=NHBlack 5=Other |
| race_eth3 | DEMO RIDRETH3 | code | 58.0% | Extended race/eth (2011+ only; NaN earlier) |
| sbp | BPX BPXSY1-4 (+BPXO BPXOSY1-3 in 2017-18) | mmHg | 8.1% | Mean systolic: drop 1st reading if >=2 available else use single; manual (BPX) preferred, oscillometric (BPXO) fills gaps in 2017-18 |
| antihtn | BPQ BPQ050A | 0/1 | 0.0% | 1 if currently taking antihypertensive (BPQ050A==1); BPQ skip-outs (never told high BP) coded 0 per NHANES convention |
| totchol | TCHOL/L13_x/LAB13 LBXTC | mg/dL | 9.6% | Total cholesterol |
| hdl | HDL/TCHOL LBDHDD|LBXHDD|LBDHDL | mg/dL | 9.6% | HDL cholesterol; variable name varies by cycle |
| statin | RXQ_RX RXDDRUG|RXD240B | 0/1 | 0.0% | 1 if any prescription drug name contains a statin ingredient (atorva/simva/rosuva/prava/lova/fluva/pitava/cerivastatin); 0 if in RXQ_RX file without statin; NaN if not in file |
| diabetes | DIQ DIQ010 + GHB LBXGH | 0/1 | 0.0% | 1 if DIQ010==1 OR HbA1c>=6.5; NaN only if both unknown |
| hba1c | GHB/L10_x/LAB10 LBXGH | % | 8.5% | Glycohemoglobin (continuous) |
| smoke_current | SMQ SMQ040 (+SMQ020) | 0/1 | 0.0% | 1 if SMQ040 in {1,2} (every/some days); never-smokers (SMQ020==2) set 0 |
| bmi | BMX BMXBMI | kg/m^2 | 5.8% | Body mass index |
| scr | BIOPRO/L40_x/LAB18 LBXSCR|LBDSCR | mg/dL | 9.9% | Serum creatinine |
| egfr | derived from scr,age,sex | mL/min/1.73m^2 | 9.9% | CKD-EPI 2021 creatinine (race-free), Inker NEJM 2021 |
| uacr | ALB_CR URDACT or URXUMA/URXUCR | mg/g | 6.2% | Urine albumin-creatinine ratio; URDACT used when present (2009+), else URXUMA(mg/L)/URXUCR(mg/dL)*100 |
| followup_years_exm | LMF PERMTH_EXM/12 | years | 4.1% | PRIMARY follow-up (exam to death/censor) |
| followup_years_int | LMF PERMTH_INT/12 | years | 0.0% | Sensitivity follow-up (interview to death/censor) |
| event | LMF MORTSTAT,UCOD_LEADING | 0/1/2 | 0.0% | 0=censored, 1=CVD death (UCOD_LEADING in {1,5}), 2=competing death |
| MORTSTAT | LMF | code | 0.0% | 0=alive, 1=deceased |
| UCOD_LEADING | LMF | code | 85.3% | Recoded leading cause (1=heart..5=cerebro..10=other) |
| ELIGSTAT | LMF | code | 0.0% | 1=eligible for linkage (cohort==1) |
| SDMVPSU | DEMO | id | 0.0% | Masked survey PSU (variance estimation) |
| SDMVSTRA | DEMO | id | 0.0% | Masked survey stratum (variance estimation) |
| WTMEC2YR | DEMO | weight | 0.0% | 2-year MEC exam weight |
| WTINT2YR | DEMO | weight | 0.0% | 2-year interview weight |

## Survey-weight guidance
- Use **WTMEC2YR** for analyses using MEC/lab-based predictors (the
  primary predictor set here). Use WTINT2YR only for interview-only vars.
- When pooling K cycles, divide the 2-year weight by K (pooled weight = WTMEC2YR / n_cycles).
- Always Taylor-linearize with SDMVSTRA (strata) and SDMVPSU (PSU).
- Weights are provided but NOT applied in this file; downstream code
  decides weighted vs unweighted ML training.

## Notes
- Missingness is preserved (no imputation here).
- ~1,680 rows have PERMTH_EXM missing (interviewed, not MEC-examined; MEC weight ~0). followup_years_int is available for them.
