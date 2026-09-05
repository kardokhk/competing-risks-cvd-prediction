#!/usr/bin/env Rscript
# Stability check for cf2_decomposition.py: a bootstrap that RESAMPLES THE
# TRAINING SET and refits the cause-specific Cox each replicate, so the
# decomposition shares carry model-fitting uncertainty as well as test-set
# sampling uncertainty.  The test set is resampled in the same replicate.
# Terms are the R-internal ones (D_model, T_het, T_pool); the implementation
# term is deterministic and is measured once in cf2.
# Seed 20260903.  ~300 replicates.

suppressPackageStartupMessages({
  library(arrow); library(survival); library(riskRegression); library(prodlim)
})
set.seed(20260903)
NB <- 300

args <- commandArgs(trailingOnly = FALSE)
sf <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- normalizePath(file.path(dirname(sf), "..", ".."))
MAT  <- file.path(ROOT, "data/processed/model_matrices")
OUT  <- file.path(ROOT, "results/closed_form")

FEATURES <- c("age","sex","sbp","antihtn","totchol","hdl","statin",
              "diabetes","smoke_current","bmi","egfr")
STD_FEATURES <- c("age","sbp","totchol","hdl","bmi","egfr")
HORIZONS <- c(5, 10, 15)

train_raw <- as.data.frame(read_parquet(file.path(MAT, "train.parquet")))
test_raw  <- as.data.frame(read_parquet(file.path(MAT, "test.parquet")))

prep <- function(tr, te) {
  ctr <- sapply(STD_FEATURES, function(f) mean(tr[[f]]))
  scl <- sapply(STD_FEATURES, function(f) sd(tr[[f]])); scl[scl == 0 | is.na(scl)] <- 1
  ap <- function(df) { for (f in STD_FEATURES) df[[f]] <- (df[[f]] - ctr[[f]]) / scl[[f]]
                       d <- df[, c("time","event",FEATURES)]; d$event <- as.integer(d$event); d }
  list(ap(tr), ap(te))
}
frm <- as.formula(paste("Hist(time, event) ~", paste(FEATURES, collapse = " + ")))

gap_exact <- function(F1, F2) {
  p <- F1 + F2; pi_ <- ifelse(p > 0, F1 / pmax(p, 1e-300), 0)
  ifelse(p > 0, 1 - pmax(1 - p, 1e-300)^pi_, 0) - F1
}

one <- function(itr, ite) {
  s <- prep(train_raw[itr, , drop = FALSE], test_raw[ite, , drop = FALSE])
  fit <- CSC(frm, data = s[[1]], cause = 1)
  F1 <- as.matrix(predictRisk(fit, newdata = s[[2]], times = HORIZONS, cause = 1))
  F2 <- as.matrix(predictRisk(fit, newdata = s[[2]], times = HORIZONS, cause = 2))
  L1 <- as.matrix(predictCox(fit$models[["Cause 1"]], newdata = s[[2]],
                             times = HORIZONS, type = "cumhazard")$cumhazard)
  V <- 1 - exp(-L1)
  out <- c()
  for (j in seq_along(HORIZONS)) {
    f1 <- F1[, j]; f2 <- F2[, j]; v <- V[, j]
    Dm <- mean(v) - mean(f1)
    Th <- mean(gap_exact(f1, f2))
    Tp <- gap_exact(mean(f1), mean(f2))
    out <- c(out, setNames(c(Dm, Th, Tp, Tp/Dm, (Th-Tp)/Dm, (Dm-Th)/Dm),
      paste0(c("D_model","T_het","T_pool","sh_estimand","sh_heterogeneity","sh_timing"),
             "_h", HORIZONS[j])))
  }
  out
}

pt <- one(seq_len(nrow(train_raw)), seq_len(nrow(test_raw)))
draws <- matrix(NA_real_, NB, length(pt), dimnames = list(NULL, names(pt)))
for (b in seq_len(NB)) {
  itr <- sample.int(nrow(train_raw), nrow(train_raw), replace = TRUE)
  ite <- sample.int(nrow(test_raw),  nrow(test_raw),  replace = TRUE)
  r <- tryCatch(one(itr, ite), error = function(e) rep(NA_real_, length(pt)))
  draws[b, ] <- r
  if (b %% 50 == 0) cat("rep", b, "\n")
}
res <- data.frame(
  quantity = names(pt), value = as.numeric(pt),
  lo = apply(draws, 2, quantile, 0.025, na.rm = TRUE),
  hi = apply(draws, 2, quantile, 0.975, na.rm = TRUE),
  n_ok = colSums(!is.na(draws)))
write.csv(res, file.path(OUT, "decomposition_refit_boot.csv"), row.names = FALSE)
print(res, digits = 4)
cat("DONE\n")
