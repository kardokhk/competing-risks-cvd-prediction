"""Configuration for NHANES + NCHS Linked Mortality File download & build.

Cycles: NHANES continuous 1999-2018 (10 two-year cycles).
File-name suffixes per cycle and the URL path segment used by wwwn.cdc.gov.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_NHANES = ROOT / "data" / "raw" / "nhanes"
RAW_MORT = ROOT / "data" / "raw" / "mortality"
PROCESSED = ROOT / "data" / "processed"
LOGS = ROOT / "logs"
MANIFEST = ROOT / "data" / "raw" / "manifest.csv"

for p in (RAW_NHANES, RAW_MORT, PROCESSED, LOGS):
    p.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0"

# cycle -> (url_path_segment, file_suffix, start_year)
CYCLES = {
    "1999-2000": ("1999-2000", "", 1999),
    "2001-2002": ("2001-2002", "_B", 2001),
    "2003-2004": ("2003-2004", "_C", 2003),
    "2005-2006": ("2005-2006", "_D", 2005),
    "2007-2008": ("2007-2008", "_E", 2007),
    "2009-2010": ("2009-2010", "_F", 2009),
    "2011-2012": ("2011-2012", "_G", 2011),
    "2013-2014": ("2013-2014", "_H", 2013),
    "2015-2016": ("2015-2016", "_I", 2015),
    "2017-2018": ("2017-2018", "_J", 2017),
}

# Logical component -> ordered list of candidate BASE filenames (without suffix / .XPT).
# The suffix is appended for continuous-cycle files; early relabeled files (LAB##, L##_x)
# are given with their full name and the token {SUF} replaced by the cycle suffix.
# We try each candidate in order until one returns HTTP 200.
#
# NHANES early-cycle lab renames:
#   1999-2000 uses LAB## names WITHOUT suffix (LAB13, LAB10, LAB18, LAB16, ...)
#   2001-2004 uses L##_B / L##_C names
#   2005-2018 uses modern component names (TCHOL_D, HDL_D, ...)
#
# Confirmed empirically (2026): total & HDL chol are in the combined lipid file
#   LAB13 (1999-2000), L13_B (2001-02), L13_C (2003-04); standalone TCHOL_x/HDL_x
#   files only exist from 2005-06 (_D) onward. Verified TCHOL_C/HDL_C do NOT exist.
# {SUF}  = full suffix incl. underscore ("", "_B", ..., "_J")
# {SUFX} = suffix letter only ("B","C",...); undefined for 1999-2000 (handled by
#          the bare LAB## fallback which needs no letter).
COMPONENT_CANDIDATES = {
    "DEMO":    ["DEMO{SUF}"],
    # blood pressure: manual BPX all cycles; oscillometric BPXO only 2017-2018.
    "BPX":     ["BPX{SUF}"],
    "BPXO":    ["BPXO{SUF}"],
    "BMX":     ["BMX{SUF}"],
    # total cholesterol: standalone from _D; combined lipid file earlier
    "TCHOL":   ["TCHOL{SUF}", "L13_{SUFX}", "LAB13"],
    # HDL cholesterol: standalone from _D; combined lipid file earlier
    "HDL":     ["HDL{SUF}", "L13_{SUFX}", "LAB13"],
    # glycohemoglobin / HbA1c
    "GHB":     ["GHB{SUF}", "L10_{SUFX}", "LAB10"],
    "DIQ":     ["DIQ{SUF}"],
    "SMQ":     ["SMQ{SUF}"],
    # serum creatinine (standard biochemistry)
    "BIOPRO":  ["BIOPRO{SUF}", "L40_{SUFX}", "LAB18"],
    # urine albumin/creatinine
    "ALB_CR":  ["ALB_CR{SUF}", "L16_{SUFX}", "LAB16"],
    # antihypertensive medication use
    "BPQ":     ["BPQ{SUF}"],
    # prescription medications (statin identification)
    "RXQ_RX":  ["RXQ_RX{SUF}"],
}

# suffix without the leading underscore, for L13_B style names
def suffix_letter(suffix: str) -> str:
    return suffix.lstrip("_")

# Verified working NHANES XPT endpoint (2026). Uses the cycle START YEAR, not the
# range: e.g. .../Public/1999/DataFiles/DEMO.XPT  (the older /Nchs/Nhanes/<range>/
# path now soft-404s to an HTML "Page Not Found").
NHANES_XPT_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{startyr}/DataFiles/{fname}.XPT"

MORT_BASE = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/"
             "datalinkage/linked_mortality/")
MORT_FILE = "NHANES_{s}_{e}_MORT_2019_PUBLIC.dat"

# ---- Linked Mortality File fixed-width layout (2019 public-use release) ----
# Verified against the actual .dat records (48-char fixed width, 0-indexed):
#   SEQN            idx 0-5    (cols 1-6)
#   ELIGSTAT        idx 14     (col 15)   1=eligible,2=under age,3=ineligible
#   MORTSTAT        idx 15     (col 16)   0=assumed alive,1=assumed deceased,blank=ineligible
#   UCOD_LEADING    idx 16-18  (cols 17-19) recoded leading cause of death
#   DIABETES        idx 19     (col 20)
#   HYPERTEN        idx 20     (col 21)
#   PERMTH_INT      idx 42-44  (cols 43-45) person-months interview->death/censor
#   PERMTH_EXM      idx 45-47  (cols 46-48) person-months exam->death/censor
LMF_COLSPECS = {
    "SEQN": (0, 6),
    "ELIGSTAT": (14, 15),
    "MORTSTAT": (15, 16),
    "UCOD_LEADING": (16, 19),
    "DIABETES": (19, 20),
    "HYPERTEN": (20, 21),
    "PERMTH_INT": (42, 45),
    "PERMTH_EXM": (45, 48),
}

# UCOD_LEADING recode categories (NCHS public-use):
#   1 = Diseases of heart
#   2 = Malignant neoplasms
#   3 = Chronic lower respiratory diseases
#   4 = Accidents (unintentional injuries)
#   5 = Cerebrovascular diseases
#   6 = Alzheimer's disease
#   7 = Diabetes mellitus
#   8 = Influenza and pneumonia
#   9 = Nephritis, nephrotic syndrome and nephrosis
#  10 = All other causes (residual)
CVD_CODES = {1, 5}  # CVD death = heart + cerebrovascular
