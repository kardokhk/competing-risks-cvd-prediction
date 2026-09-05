#!/usr/bin/env Rscript
# =============================================================================
# P2-A step 11b. ICI, E50 and E90 computed with a smoother CONCORDANT with the
# type of model whose calibration is being assessed.
#
# Austin PC, Putter H. Stat Med 2026;45:e70468 (doi:10.1002/sim.70468):
#   "When computing the ICI, E50, and E90, we recommend that researchers use a
#    model for computing smoothed event probabilities that is concordant with the
#    type of model whose calibration is being assessed."
# Their two types are the Fine-Gray subdistribution model and the pair of
# cause-specific hazard models. In their Weibull cause-specific-hazard scenarios
# a discordant smoother raised the mean ICI from 0.018 to 0.024, always away from
# zero, so a discordant smoother manufactures apparent miscalibration.
#
# The operational specification comes from their earlier paper, Austin, Putter,
# Lee and van Buuren, Diagn Progn Res 2022;6:2 (doi:10.1186/s41512-021-00114-6):
# regress the outcome on a restricted cubic spline of cloglog(predicted risk),
# three knots at the 10th, 50th and 90th percentiles, and read the fitted values
# as the smoothed observed probabilities.
#
# Three smoothers are fitted for every model, so that both the concordant value
# and the size of the discordance are visible:
#
#   smoothfg    FGR(Hist(time, status) ~ rcs(cll_pred, 3), cause = 1)
#               concordant with a Fine-Gray model
#   smoothcsh   CSC(Hist(time, status) ~ rcs(cll_pred, 3), cause = 1)
#               concordant with cause-specific hazard models
#   smoothnet   coxph(Surv(time, status == 1) ~ rcs(cll_pred, 3)), risk = 1 - S(t)
#               competing death censored. THIS IS AN EXTENSION, NOT IN THE PAPER.
#               Austin and Putter consider only models that predict a cumulative
#               incidence. A naive model predicts 1 - S(t) with the competing
#               event censored, so the smoother concordant with how the
#               prediction was made is the one that estimates the same quantity.
#               It is labelled as an extension wherever it is reported.
#
# The concordant choice per model is recorded in the `concordant` column:
#   fine_gray                       -> smoothfg
#   csc_cox                         -> smoothcsh
#   cox_naive, rsf_naive,
#     gbs_naive, logistic           -> smoothnet   (the extension)
#   rsf_cr, logistic_cr, deephit, nfg -> undetermined; both smoothfg and
#     smoothcsh are reported and neither is called concordant, because these are
#     not Fine-Gray or cause-specific hazard models and the paper's argument does
#     not reach them.
#
# The pseudo-observation smoother used in src/v2/eval_lib_v2.py is nonparametric
# and is concordant with neither type. It is kept because it is what
# van Geloven et al. BMJ 2022 use and it is what makes version 2 comparable with
# version 1, and it is described that way and not as concordant.
#
# Bootstrap replicates use the SAME subject indices as src/v2/06_evaluate.py,
# read from the file written by 11a, so these contrasts are paired with the rest.
#
# Run:
#   Rscript src/v2/11b_concordant_ici.R --imps 1-30 --split temporal -B 300 --ncores 128
#   Rscript src/v2/11b_concordant_ici.R --imps 1 --split temporal -B 2 --ncores 1  # smoke
# =============================================================================
suppressPackageStartupMessages({
  library(arrow); library(survival); library(riskRegression); library(prodlim)
  library(parallel)
})
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")

af <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
this_file <- if (length(af)) normalizePath(sub("^--file=", "", af[1])) else
  normalizePath("src/v2/11b_concordant_ici.R")
ROOT <- normalizePath(file.path(dirname(this_file), "..", ".."))
arg <- function(f, d) { a <- commandArgs(trailingOnly = TRUE); i <- match(f, a)
  if (is.na(i) || i == length(a)) d else a[i + 1L] }
parse_range <- function(s) unlist(lapply(strsplit(s, ",")[[1]], function(p)
  if (grepl("-", p)) { z <- as.integer(strsplit(p, "-")[[1]]); z[1]:z[2] } else as.integer(p)))

IMPS   <- parse_range(arg("--imps", "1-30"))
SPLIT  <- arg("--split", "temporal")
B      <- as.integer(arg("-B", "300"))
NCORES <- as.integer(arg("--ncores", Sys.getenv("SLURM_CPUS_PER_TASK", "4")))
HORIZ  <- as.numeric(arg("--horizon", "10"))
CHUNK  <- as.integer(arg("--chunk", "25"))

PRED <- file.path(ROOT, "results/metrics_v2/predictions")
OUTD <- file.path(ROOT, "results/metrics_v2/concordant")
dir.create(OUTD, recursive = TRUE, showWarnings = FALSE)

CONCORDANT <- c(fine_gray = "smoothfg", csc_cox = "smoothcsh",
                cox_naive = "smoothnet", rsf_naive = "smoothnet",
                gbs_naive = "smoothnet", logistic = "smoothnet",
                rsf_cr = "undetermined", logistic_cr = "undetermined",
                deephit = "undetermined", nfg = "undetermined")

# Any exported index file for this split with at least B replicates will do:
# replicate b is a pure function of b, so the first B columns of a longer file are
# the same B replicates.
cands <- Sys.glob(file.path(ROOT, "results/metrics_v2/boot",
                            sprintf("boot_index__%s__B*.parquet", SPLIT)))
if (!length(cands)) stop("no boot_index file for split ", SPLIT,
                         "; run src/v2/11a_export_boot_index.py first")
sizes <- as.integer(sub(".*__B(\\d+)\\.parquet$", "\\1", cands))
ok <- which(sizes >= B)
if (!length(ok)) stop("largest exported boot index has ", max(sizes),
                      " replicates but B = ", B,
                      "; rerun src/v2/11a_export_boot_index.py with -B ", B)
bi_f <- cands[ok[which.min(sizes[ok])]]
cat("bootstrap indices from", basename(bi_f), "\n")
BIDX <- as.matrix(as.data.frame(read_parquet(bi_f)))
stopifnot(ncol(BIDX) >= B)

# Harrell restricted cubic spline, 3 knots at the 10th, 50th and 90th percentiles
rcs3 <- function(x) {
  k <- as.numeric(quantile(x, c(0.10, 0.50, 0.90)))
  if (length(unique(k)) < 3) k <- as.numeric(quantile(x, c(0.02, 0.5, 0.98)))
  if (length(unique(k)) < 3) return(NULL)
  den <- (k[3] - k[1])^2          # rms::rcspline.eval normalisation
  tp <- function(a, b) pmax(a - b, 0)^3
  cbind(B1 = x,
        B2 = (tp(x, k[1]) - tp(x, k[2]) * (k[3] - k[1]) / (k[3] - k[2])
              + tp(x, k[3]) * (k[2] - k[1]) / (k[3] - k[2])) / den)
}

metrics_from <- function(obs, p) {
  d <- abs(as.numeric(obs) - p)
  c(ici = mean(d), e50 = as.numeric(median(d)), e90 = as.numeric(quantile(d, 0.90)))
}

one_model <- function(dd, p) {
  p <- pmin(pmax(p, 1e-8), 1 - 1e-8)
  cll <- log(-log(1 - p))
  Bm <- rcs3(cll)
  out <- rep(NA_real_, 9)
  names(out) <- c(outer(c("ici", "e50", "e90"),
                        c("smoothfg", "smoothcsh", "smoothnet"),
                        function(a, b) paste0(a, "_", b)))
  if (is.null(Bm)) return(out)
  df <- data.frame(time = dd$time, status = dd$status, B1 = Bm[, 1], B2 = Bm[, 2])
  o <- tryCatch({
    fg <- riskRegression::FGR(Hist(time, status) ~ B1 + B2, data = df, cause = 1)
    metrics_from(riskRegression::predictRisk(fg, newdata = df, times = HORIZ, cause = 1), p)
  }, error = function(e) rep(NA_real_, 3))
  out[c("ici_smoothfg", "e50_smoothfg", "e90_smoothfg")] <- o
  o <- tryCatch({
    cs <- riskRegression::CSC(Hist(time, status) ~ B1 + B2, data = df, cause = 1)
    metrics_from(riskRegression::predictRisk(cs, newdata = df, times = HORIZ, cause = 1), p)
  }, error = function(e) rep(NA_real_, 3))
  out[c("ici_smoothcsh", "e50_smoothcsh", "e90_smoothcsh")] <- o
  o <- tryCatch({
    df$ev1 <- as.integer(df$status == 1L)
    cx <- coxph(Surv(time, ev1) ~ B1 + B2, data = df, x = TRUE, y = TRUE)
    metrics_from(riskRegression::predictRisk(cx, newdata = df, times = HORIZ), p)
  }, error = function(e) rep(NA_real_, 3))
  out[c("ici_smoothnet", "e50_smoothnet", "e90_smoothnet")] <- o
  out
}

run_chunk <- function(job) {
  imp <- job$imp; b0 <- job$b0; b1 <- job$b1
  dest <- file.path(OUTD, sprintf("%s__h%g__imp%d__chunk%d.csv", SPLIT, HORIZ, imp, b0))
  if (file.exists(dest)) return(dest)
  fs <- Sys.glob(file.path(PRED, sprintf("*__%s__imp%d.parquet", SPLIT, imp)))
  if (!length(fs)) return(NA_character_)
  risks <- list(); base <- NULL
  for (f in fs) {
    d <- as.data.frame(read_parquet(f))
    d <- d[order(d$SEQN), ]
    if (is.null(base)) base <- data.frame(time = d$time, status = as.integer(d$event))
    risks[[as.character(d$model_name[1])]] <- d[[sprintf("risk_%g", HORIZ)]]
  }
  rows <- list()
  for (bb in c(-1L, b0:(b1 - 1L))) {         # -1 is the unbootstrapped estimate
    sel <- if (bb < 0L) seq_len(nrow(base)) else BIDX[, bb + 1L]
    dd <- base[sel, ]
    for (m in names(risks)) {
      pm <- risks[[m]][sel]
      if (all(!is.finite(pm))) next
      v <- one_model(dd[is.finite(pm), ], pm[is.finite(pm)])
      rows[[length(rows) + 1L]] <- data.frame(
        split = SPLIT, horizon_y = HORIZ, imp = imp, b = bb, model = m,
        concordant = unname(ifelse(m %in% names(CONCORDANT), CONCORDANT[m], "undetermined")),
        t(v), stringsAsFactors = FALSE)
    }
  }
  if (!length(rows)) return(NA_character_)
  write.csv(do.call(rbind, rows), dest, row.names = FALSE)
  dest
}

jobs <- list()
for (k in IMPS) for (b0 in seq(0L, B - 1L, by = CHUNK))
  jobs[[length(jobs) + 1L]] <- list(imp = k, b0 = b0, b1 = min(b0 + CHUNK, B))
cat(sprintf("P2-A step 11b: %d chunk jobs, split %s, horizon %g, B = %d, %d cores\n",
            length(jobs), SPLIT, HORIZ, B, NCORES))

t0 <- Sys.time()
bs <- max(1L, 4L * NCORES)
res <- character(0)
for (bi in seq_len(ceiling(length(jobs) / bs))) {
  lo <- (bi - 1L) * bs + 1L; hi <- min(bi * bs, length(jobs))
  rb <- if (NCORES > 1L)
    parallel::mclapply(jobs[lo:hi], function(j) tryCatch(run_chunk(j),
        error = function(e) paste("ERR", conditionMessage(e))),
        mc.cores = NCORES, mc.preschedule = TRUE)
    else lapply(jobs[lo:hi], run_chunk)
  res <- c(res, unlist(rb))
  gc(verbose = FALSE)
  cat(sprintf("  %d/%d chunks, %.1f min\n", hi, length(jobs),
              as.numeric(Sys.time() - t0, units = "mins")))
  flush.console()
}
err <- grep("^ERR", res, value = TRUE)
if (length(err)) { cat("FAILURES:\n"); print(head(unique(err), 10)) }
cat(sprintf("DONE step 11b: %d chunk files in %s\n",
            sum(!is.na(res) & !grepl("^ERR", res)), OUTD))
