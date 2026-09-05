#!/usr/bin/env Rscript
# =============================================================================
# P2-A step 07a. Independent reference values for every estimator in
# src/v2/eval_lib_v2.py, computed with the established R packages rather than
# with the numpy reimplementation.
#
# The numpy versions exist because a 2,000-replicate shared-index bootstrap over
# 30 imputations, 10 models and 3 horizons is not affordable through
# riskRegression and pec. They are only usable if they agree with those packages
# on the point estimates, so this script produces the reference and 07b_validate_py.py
# does the comparison. Any estimator that disagrees beyond the stated tolerance is
# a defect and is reported as one.
#
# Reference implementations, matching ValidationCompRisks (van Geloven et al.,
# BMJ 2022;377:e069249, code accessed 2026-09-03):
#   observed CIF   prodlim::prodlim, Aalen-Johansen
#   pseudo-values  prodlim::jackknife
#   intercept/slope geepack::geese, family gaussian, mean.link cloglog,
#                   corstr independence, cll(predicted) as offset
#   ICI/E50/E90    riskRegression::FGR with splines::ns(cll_pred, df = 6)
#   AUC            riskRegression::Score, then timeROC::timeROC AUC_2
#   C-index        pec::cindex
#   net benefit    the stdca.R formula, written out directly
#
# Run on the temporal split of one imputation:
#   Rscript src/v2/07a_validate_R.R [imp] [model]
# =============================================================================
suppressPackageStartupMessages({
  library(arrow); library(survival); library(prodlim); library(riskRegression)
  library(pec); library(timeROC); library(splines)
})
has_geepack <- requireNamespace("geepack", quietly = TRUE)

af <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
this_file <- if (length(af)) normalizePath(sub("^--file=", "", af[1])) else
  normalizePath("src/v2/07a_validate_R.R")
ROOT <- normalizePath(file.path(dirname(this_file), "..", ".."))
args <- commandArgs(trailingOnly = TRUE)
IMP   <- if (length(args) >= 1) as.integer(args[1]) else 1L
MODEL <- if (length(args) >= 2) args[2] else "csc_cox"
HORIZON <- 10
THRESHOLDS <- c(0.01, 0.02, 0.03, 0.05, 0.075, 0.20)

OUTDIR <- file.path(ROOT, "results/metrics_v2/validation")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

f <- file.path(ROOT, sprintf("results/metrics_v2/predictions/%s__temporal__imp%d.parquet",
                             MODEL, IMP))
stopifnot(file.exists(f))
d <- as.data.frame(read_parquet(f))
d <- d[is.finite(d$risk_10), ]
d$status <- as.integer(d$event)
d$pred <- pmin(pmax(d$risk_10, 1e-8), 1 - 1e-8)
d$cll_pred <- log(-log(1 - d$pred))
n <- nrow(d)
cat(sprintf("validation reference: model %s, imputation %d, n = %d, horizon %g y\n",
            MODEL, IMP, n, HORIZON))

res <- list()
add <- function(k, v) res[[k]] <<- as.numeric(v)

# ---- observed CIF, Aalen-Johansen -----------------------------------------
fit_aj <- prodlim::prodlim(Hist(time, status) ~ 1, data = d)
obs <- as.numeric(predict(fit_aj, cause = 1, times = HORIZON, type = "cuminc"))
add("obs_cif", obs)
add("mean_pred", mean(d$pred))
add("eo", mean(d$pred) / obs)

# ---- pseudo-observations ---------------------------------------------------
pseudo <- prodlim::jackknife(fit_aj, times = HORIZON, cause = 1)
pseudo <- as.numeric(pseudo)
add("pseudo_mean", mean(pseudo))
add("pseudo_sd", sd(pseudo))

# ---- weak calibration ------------------------------------------------------
if (has_geepack) {
  pf <- data.frame(pv = pseudo, cll = d$cll_pred, id = seq_len(n))
  fi <- geepack::geese(pv ~ offset(cll), data = pf, id = id, scale.fix = TRUE,
                       family = gaussian, mean.link = "cloglog",
                       corstr = "independence", jack = TRUE)
  fs <- geepack::geese(pv ~ offset(cll) + cll, data = pf, id = id, scale.fix = TRUE,
                       family = gaussian, mean.link = "cloglog",
                       corstr = "independence", jack = TRUE)
  add("calib_intercept", summary(fi)$mean["(Intercept)", "estimate"])
  add("calib_slope", 1 + summary(fs)$mean["cll", "estimate"])
  add("calib_intercept_se", summary(fi)$mean["(Intercept)", "san.se"])
  add("calib_slope_se", summary(fs)$mean["cll", "san.se"])
} else {
  cat("geepack not available; weak calibration reference skipped\n")
}

# ---- moderate calibration, the FGR spline route ---------------------------
fg <- tryCatch(riskRegression::FGR(Hist(time, status) ~ ns(cll_pred, df = 6),
                                   data = d, cause = 1),
               error = function(e) { cat("FGR ns(df=6) failed:", conditionMessage(e), "\n"); NULL })
if (!is.null(fg)) {
  obs_smooth <- as.numeric(riskRegression::predictRisk(fg, newdata = d,
                                                       times = HORIZON, cause = 1))
  dd <- abs(obs_smooth - d$pred)
  add("ici_fgr_ns6", mean(dd)); add("e50_fgr_ns6", median(dd))
  add("e90_fgr_ns6", quantile(dd, 0.90))
}
# pseudo-value spline route, 3 knots, which is what eval_lib_v2 uses
if (has_geepack) {
  kn <- quantile(d$cll_pred, c(0.10, 0.50, 0.90))   # Austin et al. 2022
  rcs <- function(x, k) {
    K <- length(k); den <- (k[K] - k[1])^2           # rms::rcspline.eval norm
    cols <- list(x)
    for (j in seq_len(K - 2)) {
      tp <- function(a, b) pmax(a - b, 0)^3
      cols[[j + 1]] <- (tp(x, k[j])
        - tp(x, k[K - 1]) * (k[K] - k[j]) / (k[K] - k[K - 1])
        + tp(x, k[K]) * (k[K - 1] - k[j]) / (k[K] - k[K - 1])) / den
    }
    do.call(cbind, cols)
  }
  B <- rcs(d$cll_pred, kn)
  pf2 <- data.frame(pv = pseudo, B1 = B[, 1], B2 = B[, 2], id = seq_len(n))
  fm <- geepack::geese(pv ~ B1 + B2, data = pf2, id = id, scale.fix = TRUE,
                       family = gaussian, mean.link = "cloglog",
                       corstr = "independence")
  eta <- cbind(1, B) %*% as.numeric(fm$beta)
  os <- 1 - exp(-exp(pmin(pmax(eta, -30), 10)))
  dd2 <- abs(as.numeric(os) - d$pred)
  add("ici_pseudo_rcs3", mean(dd2)); add("e50_pseudo_rcs3", median(dd2))
  add("e90_pseudo_rcs3", quantile(dd2, 0.90))
}

# ---- discrimination --------------------------------------------------------
tr <- tryCatch(timeROC::timeROC(T = d$time, delta = d$status, marker = d$pred,
                                cause = 1, weighting = "marginal",
                                times = c(HORIZON / 2, HORIZON), iid = FALSE),
               error = function(e) NULL)
if (!is.null(tr)) {
  a <- if (!is.null(tr$AUC_2)) tr$AUC_2 else tr$AUC
  add("auc_timeROC_marginal", a[length(a)])
}
sc <- tryCatch(riskRegression::Score(list(m = d$pred), formula = Hist(time, status) ~ 1,
                                     data = d, times = HORIZON, cause = 1,
                                     metrics = "auc", cens.model = "km",
                                     conf.int = FALSE, null.model = FALSE),
               error = function(e) NULL)
if (!is.null(sc)) add("auc_Score_km", sc$AUC$score$AUC[1])

ci <- tryCatch(pec::cindex(list(m = matrix(d$pred, ncol = 1)),
                           formula = Hist(time, status) ~ 1, data = d,
                           cause = 1, eval.times = HORIZON),
               error = function(e) NULL)
if (!is.null(ci)) add("cindex_pec_marginal", ci$AppCindex$m[1])

# ---- net benefit, the stdca.R formula --------------------------------------
nb <- function(pt) {
  fl <- d$pred >= pt
  pex <- mean(fl)
  if (pex == 0) return(0)
  fs <- prodlim::prodlim(Hist(time, status) ~ 1, data = d[fl, ])
  fx <- as.numeric(predict(fs, cause = 1, times = HORIZON, type = "cuminc"))
  if (is.na(fx)) fx <- 0
  fx * pex - (1 - fx) * pex * (pt / (1 - pt))
}
for (pt in THRESHOLDS) {
  add(sprintf("nb_at_%g", pt), nb(pt))
  add(sprintf("frac_above_%g", pt), mean(d$pred >= pt))
}
add("nb_all", obs - (1 - obs) * (0.075 / (1 - 0.075)))   # at pt = 0.075, for the sign check

out <- data.frame(metric = names(res), value_R = unlist(res), row.names = NULL)
out$model <- MODEL; out$imp <- IMP; out$horizon_y <- HORIZON
write.csv(out, file.path(OUTDIR, sprintf("reference_R__%s__imp%d.csv", MODEL, IMP)),
          row.names = FALSE)
print(out, digits = 6)
cat("wrote", file.path(OUTDIR, sprintf("reference_R__%s__imp%d.csv", MODEL, IMP)), "\n")
