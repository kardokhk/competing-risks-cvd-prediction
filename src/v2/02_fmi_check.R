#!/usr/bin/env Rscript
# =============================================================================
# P2-A step 02. Is m = 20 enough?
#
# von Hippel (Sociol Methods Res 2020;49:699-718, doi:10.1177/0049124117747303)
# gives the required number of imputations as
#     m = ceiling(1 + 0.5 * (fmi_upper / cv)^2),  cv = 0.05,
#     fmi_upper = plogis(qlogis(fmi) + 1.96 * sqrt(2 / m_pilot)),
# with fmi the largest fraction of missing information across the estimates that
# matter. The 14% incomplete-case figure is NOT the FMI and must not be used in
# its place. This script estimates the FMI from the m = 20 pilot on the
# coefficients of both cause-specific Cox models, which are the substantive
# models the predictions come from, and reports the m that would be required.
#
# Output: results/metrics_v2/fmi_check.csv and a line in the log.
# =============================================================================
suppressPackageStartupMessages({
  library(arrow); library(mice); library(survival)
})
af <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
this_file <- if (length(af)) normalizePath(sub("^--file=", "", af[1])) else
  normalizePath("src/v2/02_fmi_check.R")
ROOT <- normalizePath(file.path(dirname(this_file), "..", ".."))
IMPDIR <- file.path(ROOT, "data/processed/imputed")
OUT <- file.path(ROOT, "results/metrics_v2")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

FEATURES <- c("age","sex","sbp","antihtn","totchol","hdl","statin",
              "diabetes","smoke_current","bmi","egfr")
files <- sort(Sys.glob(file.path(IMPDIR, "imp_*.parquet")))
M <- length(files)
stopifnot(M >= 2)
cat(sprintf("FMI check on m = %d imputations\n", M))

imps <- lapply(files, function(f) as.data.frame(read_parquet(f)))

pool_one <- function(cause_code, label) {
  fits <- lapply(imps, function(d) {
    d$y <- as.integer(d$event == cause_code)
    survival::coxph(as.formula(paste("Surv(time, y) ~", paste(FEATURES, collapse = "+"))),
                    data = d)
  })
  est <- t(vapply(fits, coef, numeric(length(FEATURES))))
  vr  <- t(vapply(fits, function(f) diag(vcov(f)), numeric(length(FEATURES))))
  qbar <- colMeans(est); ubar <- colMeans(vr)
  bvar <- apply(est, 2, var)
  tvar <- ubar + (1 + 1 / M) * bvar
  lambda <- (1 + 1 / M) * bvar / tvar               # relative increase in variance
  nu_old <- (M - 1) / lambda^2
  # Barnard-Rubin adjusted degrees of freedom
  nu_com <- nrow(imps[[1]]) - length(FEATURES) - 1
  nu_obs <- ((nu_com + 1) / (nu_com + 3)) * nu_com * (1 - lambda)
  nu_br <- 1 / (1 / nu_old + 1 / nu_obs)
  fmi <- (lambda + 2 / (nu_br + 3))                  # Rubin's FMI
  data.frame(model = label, term = FEATURES, est = qbar,
             se = sqrt(tvar), lambda = lambda, fmi = fmi, df_br = nu_br,
             row.names = NULL)
}

res <- rbind(pool_one(1L, "cause_specific_cvd"), pool_one(2L, "cause_specific_competing"))
res$ci_lo <- res$est - qt(0.975, res$df_br) * res$se
res$ci_hi <- res$est + qt(0.975, res$df_br) * res$se

fmi_max <- max(res$fmi, na.rm = TRUE)
fmi_u <- plogis(qlogis(min(max(fmi_max, 1e-4), 0.999)) + 1.96 * sqrt(2 / M))
m_req <- ceiling(1 + 0.5 * (fmi_u / 0.05)^2)
res$fmi_max <- fmi_max; res$fmi_upper <- fmi_u; res$m_required <- m_req; res$m_used <- M

write.csv(res, file.path(OUT, "fmi_check.csv"), row.names = FALSE)
cat(sprintf("largest FMI across the 22 cause-specific coefficients: %.3f\n", fmi_max))
cat(sprintf("  which term: %s (%s)\n", res$term[which.max(res$fmi)],
            res$model[which.max(res$fmi)]))
cat(sprintf("upper bound on FMI at m = %d: %.3f\n", M, fmi_u))
cat(sprintf("von Hippel required m for a 5%% coefficient of variation: %d\n", m_req))
if (m_req > M) {
  cat(sprintf("*** m = %d IS NOT SUFFICIENT. Rerun 01_impute.R with m = %d. ***\n", M, m_req))
} else {
  cat(sprintf("m = %d is sufficient (required %d).\n", M, m_req))
}
cat("wrote", file.path(OUT, "fmi_check.csv"), "\n")
