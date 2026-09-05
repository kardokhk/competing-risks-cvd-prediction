#!/usr/bin/env Rscript
# P2-B / closed form, step 1.
# Fit the cause-specific Cox model on the training cycles (1999-2006) with the
# same 11 predictors and the same design as src/models/c3_competing_classical.R,
# then emit for the temporal test set (2007-2010), per subject and per horizon:
#   risk1_h   cause-1 (CVD) cumulative incidence, Aalen-Johansen form   = F_1i
#   risk2_h   cause-2 (non-CVD) cumulative incidence                    = F_2i
#   cumhaz1_h cause-1 cumulative cause-specific hazard  Lambda_1i(h)
#   cumhaz2_h cause-2 cumulative cause-specific hazard  Lambda_2i(h)
#   naive1_h  1 - exp(-Lambda_1i(h)), the naive (1-KM) estimand implied by the
#             SAME fit, so that naive1_h - risk1_h is the exact change-of-estimand
#             gap with the model held fixed.
# Also writes the baseline cumulative hazards so the hazard-timing overlap
# coefficient kappa(t) can be computed downstream.
#
# Seed 20260903.  Nothing outside results/closed_form/ is written.

suppressPackageStartupMessages({
  library(arrow); library(survival); library(riskRegression); library(prodlim)
})
set.seed(20260903)

args <- commandArgs(trailingOnly = FALSE)
sf <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- normalizePath(file.path(dirname(sf), "..", ".."))
MAT  <- file.path(ROOT, "data/processed/model_matrices")
OUT  <- file.path(ROOT, "results/closed_form")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

FEATURES <- c("age","sex","sbp","antihtn","totchol","hdl","statin",
              "diabetes","smoke_current","bmi","egfr")
STD_FEATURES <- c("age","sbp","totchol","hdl","bmi","egfr")
HORIZONS <- c(5, 10, 15)

train_raw <- read_parquet(file.path(MAT, "train.parquet"))
test_raw  <- read_parquet(file.path(MAT, "test.parquet"))

ctr <- sapply(STD_FEATURES, function(f) mean(train_raw[[f]]))
scl <- sapply(STD_FEATURES, function(f) sd(train_raw[[f]]))
scl[scl == 0 | is.na(scl)] <- 1
apply_std <- function(df) {
  for (f in STD_FEATURES) df[[f]] <- (df[[f]] - ctr[[f]]) / scl[[f]]
  d <- as.data.frame(df[, c("time", "event", FEATURES)])
  d$event <- as.integer(d$event); d
}
train_s <- apply_std(train_raw); test_s <- apply_std(test_raw)

frm <- as.formula(paste("Hist(time, event) ~", paste(FEATURES, collapse = " + ")))
fit <- CSC(frm, data = train_s, cause = 1)

r1 <- as.matrix(predictRisk(fit, newdata = test_s, times = HORIZONS, cause = 1))
r2 <- as.matrix(predictRisk(fit, newdata = test_s, times = HORIZONS, cause = 2))
L1 <- as.matrix(predictCox(fit$models[["Cause 1"]], newdata = test_s,
                           times = HORIZONS, type = "cumhazard")$cumhazard)
L2 <- as.matrix(predictCox(fit$models[["Cause 2"]], newdata = test_s,
                           times = HORIZONS, type = "cumhazard")$cumhazard)

out <- data.frame(
  SEQN  = as.integer(test_raw$SEQN),
  cycle = as.character(test_raw$cycle),
  time  = as.numeric(test_raw$time),
  event = as.integer(test_raw$event)
)
for (j in seq_along(HORIZONS)) {
  h <- HORIZONS[j]
  out[[paste0("risk1_",   h)]] <- as.numeric(r1[, j])
  out[[paste0("risk2_",   h)]] <- as.numeric(r2[, j])
  out[[paste0("cumhaz1_", h)]] <- as.numeric(L1[, j])
  out[[paste0("cumhaz2_", h)]] <- as.numeric(L2[, j])
  out[[paste0("naive1_",  h)]] <- 1 - exp(-as.numeric(L1[, j]))
}
write_parquet(out, file.path(OUT, "cause2_predictions.parquet"))

# Baseline cumulative hazards on a fine grid, for kappa(t).
tg <- sort(unique(c(fit$eventTimes[fit$eventTimes <= 15], HORIZONS)))
b1 <- predictCox(fit$models[["Cause 1"]], times = tg, type = "cumhazard",
                 centered = TRUE, newdata = test_s[1, , drop = FALSE])$cumhazard
b2 <- predictCox(fit$models[["Cause 2"]], times = tg, type = "cumhazard",
                 centered = TRUE, newdata = test_s[1, , drop = FALSE])$cumhazard
write_parquet(data.frame(t = tg, L1_ref = as.numeric(b1), L2_ref = as.numeric(b2)),
              file.path(OUT, "baseline_cumhaz.parquet"))

co <- data.frame(term = names(coef(fit$models[["Cause 1"]])),
                 beta_cause1 = as.numeric(coef(fit$models[["Cause 1"]])),
                 beta_cause2 = as.numeric(coef(fit$models[["Cause 2"]])))
write.csv(co, file.path(OUT, "cause_specific_coefficients.csv"), row.names = FALSE)

cat("n test =", nrow(out), "\n")
for (h in HORIZONS) cat(sprintf("h=%2d  mean F1=%.5f  mean F2=%.5f  mean naive=%.5f  mean gap=%.5f\n",
    h, mean(out[[paste0("risk1_",h)]]), mean(out[[paste0("risk2_",h)]]),
    mean(out[[paste0("naive1_",h)]]),
    mean(out[[paste0("naive1_",h)]] - out[[paste0("risk1_",h)]])))
cat("R version:", R.version.string, " riskRegression:",
    as.character(packageVersion("riskRegression")), "\n")
cat("DONE\n")
