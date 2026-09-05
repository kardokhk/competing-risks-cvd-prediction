#!/usr/bin/env Rscript
# =============================================================================
# P2-A step 01. Multiple imputation of the 11 primary predictors (and the three
# sensitivity extras) on the FULL mortality-eligible cohort, N = 41,151.
#
# Design decisions, all argued in data/processed/imputed/README.md:
#   * Outcome-aware imputation model (White & Royston 2009): the event
#     indicators and the CAUSE-SPECIFIC Nelson-Aalen cumulative hazards enter
#     as predictors; the follow-up time itself does not.
#   * m = 20 completed datasets, seed 20260903, maxit = 10, method pmm.
#   * Cycle, sex, race and the survey design variables enter as auxiliaries.
#   * Fold assignment and the temporal split are functions of SEQN and event
#     only, so they are IDENTICAL across all 20 imputations.
#
# Run:
#   Rscript src/v2/01_impute.R [m] [maxit] [ncores]
# defaults: m=20 maxit=10 ncores=min(20, detectCores())
# Smoke test:  Rscript src/v2/01_impute.R 2 2 2
# =============================================================================

suppressPackageStartupMessages({
  library(arrow); library(mice); library(survival); library(parallel)
})

# ---- repo root, derived relatively; never hard-code an absolute path --------
args_all <- commandArgs(trailingOnly = FALSE)
fa <- grep("^--file=", args_all, value = TRUE)
this_file <- if (length(fa)) normalizePath(sub("^--file=", "", fa[1])) else
  normalizePath(file.path(getwd(), "src/v2/01_impute.R"), mustWork = FALSE)
ROOT <- normalizePath(file.path(dirname(this_file), "..", ".."))

args <- commandArgs(trailingOnly = TRUE)
M      <- if (length(args) >= 1) as.integer(args[1]) else 30L
MAXIT  <- if (length(args) >= 2) as.integer(args[2]) else 10L
# Optional 4th and 5th arguments give a chain range to write, e.g. "21 30".
# Each chain is seeded independently as SEED + 1000 * k, so chain k is the same
# dataset whether it is produced in a run of m = 20 or a run of m = 30. That
# makes the imputation extensible: the von Hippel check in 02_fmi_check.R can
# demand more imputations and the existing ones do not have to be regenerated,
# and the model fits already made against them stay valid.
K_FROM <- if (length(args) >= 4) as.integer(args[4]) else 1L
K_TO   <- if (length(args) >= 5) as.integer(args[5]) else M
NCORES <- if (length(args) >= 3) as.integer(args[3]) else
  min(M, max(1L, as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", detectCores()))))

SEED <- 20260903L
OUTDIR <- file.path(ROOT, "data", "processed", "imputed")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

PRIMARY <- c("age","sex","sbp","antihtn","totchol","hdl","statin",
             "diabetes","smoke_current","bmi","egfr")
EXTRAS  <- c("race","hba1c","uacr")
# uacr is imputed on the log scale: it is extremely right skewed (observed SD 611
# on a mean of 54) and predictive mean matching driven by a linear model on the
# raw scale matched the missing rows to the extreme upper donors, tripling the
# imputed mean. Back-transformed after imputation. uacr is a sensitivity extra,
# not one of the 11 primary predictors.
INCOMPLETE <- c("sbp","totchol","hdl","bmi","egfr","hba1c","luacr")

TRAIN_YR <- c(1999,2001,2003,2005); TEST_YR <- c(2007,2009)
CV_FOLDS <- 5L; CV_REPEATS <- 5L

cat(sprintf("P2-A step 01  m=%d maxit=%d ncores=%d seed=%d\nROOT=%s\n",
            M, MAXIT, NCORES, SEED, ROOT))

# ---------------------------------------------------------------- load ------
d <- as.data.frame(read_parquet(file.path(ROOT, "data/processed/analytic.parquet")))
stopifnot(nrow(d) == 41151)

# time: exam-based where observed, interview-based for the 1,680 respondents who
# were interviewed but never MEC-examined (max discrepancy 0.25 y). See
# src/v2/PREDICTION_FORMAT.md section 2.
d$time <- ifelse(is.na(d$followup_years_exm), d$followup_years_int, d$followup_years_exm)
d$time_source <- ifelse(is.na(d$followup_years_exm), "int", "exm")
# Eight respondents have follow-up recorded as exactly 0 (death in the same
# calendar month as the examination; PERMTH is integer months). They carry 3 CVD
# deaths and 5 competing deaths. A zero time is not admissible for any survival
# estimator, so they are moved to half the measurement resolution, 1/24 y
# (about 15 days). They are kept rather than dropped because dropping deaths is
# exactly the outcome-dependent selection this analysis exists to remove.
n_zero <- sum(d$time <= 0)
d$time <- pmax(d$time, 1 / 24)
cat(sprintf("follow-up: %d rows with time <= 0 moved to 1/24 y\n", n_zero))
stopifnot(all(is.finite(d$time)), all(d$time > 0))
d$event <- as.integer(d$event)
stopifnot(all(d$event %in% 0:2))

# sex recoded 1 = male, 0 = female (NHANES RIAGENDR 1/2)
d$sex <- as.integer(d$sex == 1)
for (b in c("antihtn","statin","diabetes","smoke_current")) d[[b]] <- as.integer(d[[b]])
stopifnot(!any(is.na(d[, c("age","sex","antihtn","statin","diabetes","smoke_current","race")])))

d$R_miss_any <- as.integer(apply(is.na(d[, PRIMARY]), 1, any))

# ------------------------------------------------- outcome-aware auxiliaries -
# White & Royston (Stat Med 2009;28:1982-98): include the event indicator and the
# Nelson-Aalen cumulative hazard of the observed time, not the time itself.
# Competing-risks extension: one cause-specific cumulative hazard per cause plus
# the corresponding indicator, so the imputation model is compatible with both
# cause-specific substantive models rather than only the cause of interest.
# Bonneville et al. (Stat Methods Med Res 2022;31:1860-80; Stat Med 2025,
# doi:10.1002/sim.70166) call this predictor set CS-Approx / CH12: the event
# indicator as a THREE-LEVEL factor plus the marginal Nelson-Aalen cause-specific
# cumulative hazards for BOTH causes, each evaluated at the subject's own
# event or censoring time, computed once on the full data and held fixed.
d$event_f <- factor(d$event, levels = c(0, 1, 2), labels = c("cens", "cvd", "comp"))
d$d_cvd  <- as.integer(d$event == 1L)
d$d_comp <- as.integer(d$event == 2L)
d$na_cvd  <- mice::nelsonaalen(d, "time", "d_cvd")   # H1(T)
d$na_comp <- mice::nelsonaalen(d, "time", "d_comp")  # H2(T)
# d_any / H_any are omitted: they are exact linear combinations of the above and
# would make the imputation design matrix rank deficient.

# Design auxiliaries. NHANES strata are arbitrary identifiers, so they enter as a
# factor, not as a number; strata with fewer than 10 respondents are collapsed
# into a single "small" level for stability. Kim et al. 2006 argue the imputer
# needs the design information; NCHS practice conditions on the determinants of
# the selection probability (age, race/ethnicity, cycle, survey location), which
# are all present here. The design is applied at the ANALYSIS stage, not here.
d$logwt   <- log1p(pmax(0, d$WTMEC2YR))
d$psu_f   <- factor(d$SDMVPSU)
# SDMVSTRA is NOT in the imputation model. Entering it as a factor (about 130
# levels) raised one mice iteration from 39 s to over 8 minutes on 41,151 rows,
# for information that cycle, PSU, age and race/ethnicity already carry. NCHS
# practice for NHANES imputation conditions on the determinants of the selection
# probability rather than on the stratum identifier, and all of those
# determinants are in the model. The survey design is applied at the ANALYSIS
# stage, not here. This is an approximation and is recorded as one.
d$cycle_n <- as.numeric(d$cycle_startyr)
d$race_f  <- factor(d$race)

# ---------------------------------------------- temporal split + CV folds ----
split_of <- function(y) ifelse(y %in% TRAIN_YR, "train",
                        ifelse(y %in% TEST_YR, "test", "holdout_later"))
d$split <- split_of(d$cycle_startyr)

# Folds are a function of (SEQN, event) only, hence identical across imputations.
# Stratified on the 3-level event indicator so every fold carries CVD deaths.
ord <- order(d$SEQN)                      # canonical, imputation-independent order
set.seed(SEED)
fold_mat <- matrix(NA_integer_, nrow = nrow(d), ncol = CV_REPEATS)
for (r in seq_len(CV_REPEATS)) {
  f <- integer(nrow(d))
  for (lev in 0:2) {
    idx <- ord[d$event[ord] == lev]
    idx <- sample(idx)                     # depends only on the RNG stream + SEQN order
    f[idx] <- (seq_along(idx) - 1L) %% CV_FOLDS
  }
  fold_mat[, r] <- f
}
for (r in seq_len(CV_REPEATS)) d[[paste0("cv_rep", r - 1L)]] <- fold_mat[, r]
# every fold must contain CVD deaths in every repeat
for (r in seq_len(CV_REPEATS)) {
  tb <- table(d[[paste0("cv_rep", r - 1L)]], d$event)
  stopifnot(all(tb[, "1"] > 0))
}
cat("CV folds built: 5 folds x 5 repeats, stratified on event, all folds carry CVD deaths.\n")

# ------------------------------------------------------- imputation frame ----
d$luacr <- log(pmax(d$uacr, 0.1))
IMP_VARS <- c(PRIMARY, setdiff(EXTRAS, "uacr"), "luacr",
              "event_f","na_cvd","na_comp",
              "logwt","psu_f","cycle_n","race_f")
IMP_VARS <- unique(IMP_VARS)
dm <- d[, IMP_VARS]
dm$race <- NULL                # `race` is complete; keep only its factor version
dm$sex <- as.integer(dm$sex)

# method: pmm for the 7 incomplete continuous variables, "" for everything else
meth <- make.method(dm)
meth[] <- ""
meth[INCOMPLETE] <- "pmm"

# predictor matrix: every incomplete variable is predicted by everything else
pm <- make.predictorMatrix(dm)
pm[] <- 0L
pm[INCOMPLETE, ] <- 1L
for (v in INCOMPLETE) pm[v, v] <- 0L
# nothing predicts a complete variable (no imputation is drawn for it)
stopifnot(all(rowSums(pm[setdiff(rownames(pm), INCOMPLETE), , drop = FALSE]) == 0))

cat("Imputation model:\n")
cat("  imputed  :", paste(INCOMPLETE, collapse = ", "), "\n")
cat("  predictors:", paste(colnames(pm), collapse = ", "), "\n")
saveRDS(list(method = meth, predictorMatrix = pm, vars = IMP_VARS),
        file.path(OUTDIR, "imputation_model.rds"))
write.csv(as.data.frame(pm), file.path(OUTDIR, "predictor_matrix.csv"))

# ---------------------------------------------------------------- run --------
# m independent single-imputation chains in parallel (the mice::parlmice scheme).
# Each chain gets its own seed derived from the project seed.
one_chain <- function(k) {
  set.seed(SEED + 1000L * k)
  fit <- mice::mice(dm, m = 1L, maxit = MAXIT, method = meth,
                    predictorMatrix = pm, seed = SEED + 1000L * k,
                    printFlag = FALSE, donors = 5L)
  cmp <- mice::complete(fit, 1L)
  list(k = k, data = cmp,
       chainMean = fit$chainMean, chainVar = fit$chainVar,
       loggedEvents = fit$loggedEvents)
}

CHAINS <- K_FROM:K_TO
cat(sprintf("writing chains %d..%d of m = %d\n", K_FROM, K_TO, M))
t0 <- Sys.time()
if (NCORES > 1L) {
  res <- parallel::mclapply(CHAINS, one_chain, mc.cores = NCORES,
                            mc.preschedule = FALSE)
} else {
  res <- lapply(CHAINS, one_chain)
}
bad <- vapply(res, function(x) inherits(x, "try-error") || is.null(x$data), logical(1))
if (any(bad)) {
  for (i in which(bad)) cat("CHAIN", i, "FAILED:", as.character(res[[i]]), "\n")
  stop("imputation chains failed")
}
cat(sprintf("mice: %d chains, maxit %d, %.1f min on %d cores\n",
            length(CHAINS), MAXIT, as.numeric(Sys.time() - t0, units = "mins"), NCORES))

# ---------------------------------------------------------------- write -----
carry <- data.frame(
  SEQN = as.integer(d$SEQN), cycle = as.character(d$cycle),
  cycle_startyr = as.integer(d$cycle_startyr),
  time = as.numeric(d$time), event = as.integer(d$event),
  event_naive = as.integer(d$event == 1L),
  split = as.character(d$split),
  stringsAsFactors = FALSE)
for (r in seq_len(CV_REPEATS)) carry[[paste0("cv_rep", r - 1L)]] <- as.integer(fold_mat[, r])
carry$SDMVPSU  <- as.numeric(d$SDMVPSU)
carry$SDMVSTRA <- as.numeric(d$SDMVSTRA)
carry$WTMEC2YR <- as.numeric(d$WTMEC2YR)
carry$R_miss_any <- as.integer(d$R_miss_any)

COLS <- c("SEQN","cycle","cycle_startyr", PRIMARY,
          "time","event","event_naive","split",
          paste0("cv_rep", 0:(CV_REPEATS - 1L)),
          "race","hba1c","uacr","SDMVPSU","SDMVSTRA","WTMEC2YR","imp","R_miss_any")

conv <- list()
for (x in res) {
  k <- x$k; cmp <- x$data
  out <- carry
  for (v in PRIMARY) out[[v]] <- as.numeric(cmp[[v]])
  out$race  <- as.numeric(as.character(cmp$race_f))
  out$hba1c <- as.numeric(cmp$hba1c)
  out$uacr  <- exp(as.numeric(cmp$luacr))
  out$imp   <- as.integer(k)
  out <- out[, COLS]
  stopifnot(!any(is.na(out[, PRIMARY])), nrow(out) == 41151)
  stopifnot(all(out$sex %in% c(0,1)))
  for (b in c("antihtn","statin","diabetes","smoke_current"))
    stopifnot(all(out[[b]] %in% c(0,1)))
  write_parquet(out, file.path(OUTDIR, sprintf("imp_%d.parquet", k)))
  conv[[as.character(k)]] <- list(chainMean = x$chainMean, chainVar = x$chainVar)
  cat(sprintf("  wrote imp_%d.parquet  mean sbp %.2f  mean egfr %.2f\n",
              k, mean(out$sbp), mean(out$egfr)))
}
saveRDS(conv, file.path(OUTDIR, "convergence_chains.rds"))

# ------------------------------------------------- observed vs imputed check -
chk <- do.call(rbind, lapply(c(INCOMPLETE, "uacr"), function(v) {
  if (v == "uacr") { d$uacr_chk <- d$uacr }
  obs <- d[[v]][!is.na(d[[v]])]
  imp <- unlist(lapply(res, function(x) {
    z <- x$data[[v]]; if (v == "uacr") z <- exp(x$data[["luacr"]]); z[is.na(d[[v]])] }))
  data.frame(variable = v, n_obs = length(obs), n_imputed_cells = length(imp),
             obs_mean = mean(obs), imp_mean = mean(imp),
             obs_sd = sd(obs), imp_sd = sd(imp),
             obs_p5 = quantile(obs, .05), obs_p95 = quantile(obs, .95),
             imp_p5 = quantile(imp, .05), imp_p95 = quantile(imp, .95),
             row.names = NULL)
}))
write.csv(chk, file.path(OUTDIR, "observed_vs_imputed.csv"), row.names = FALSE)
print(chk, digits = 4)

si <- capture.output(sessionInfo())
writeLines(si, file.path(OUTDIR, "sessionInfo.txt"))
cat("DONE. m =", M, "files in", OUTDIR, "\n")
