#!/usr/bin/env Rscript
# =============================================================================
# P2-A step 03. Fit the five R-side model families on every imputed dataset and
# write predictions to results/metrics_v2/predictions/ per src/v2/PREDICTION_FORMAT.md.
#
#   naive      cox_naive   coxph on (time, event_naive),   risk = 1 - S(t)
#              rsf_naive   rfsrc  on (time, event_naive),  risk = 1 - S(t)
#   competing  csc_cox     riskRegression::CSC,            risk = CIF_1(t)
#              fine_gray   riskRegression::FGR,            risk = CIF_1(t)
#              rsf_cr      rfsrc splitrule logrankCR,      risk = CIF_1(t)
#
# cox_naive and csc_cox share the coxph learner exactly; that is the primary
# learner-matched pair. rsf_naive and rsf_cr share the rfsrc learner.
#
# Standardization is fitted on the TRAINING rows of the fold being fitted and
# applied to the held-out rows. Never on pooled data. (Cox and forest predictions
# are invariant to it, but it is kept so every family sees the identical matrix
# and so the audited no-leakage property is preserved verbatim.)
#
# Run:
#   Rscript src/v2/03_fit_r_models.R --imps 1-20 --splits temporal,cv --ncores 128
#   Rscript src/v2/03_fit_r_models.R --imps 1-2 --splits temporal --ncores 2   # smoke
# =============================================================================
suppressPackageStartupMessages({
  library(arrow); library(survival); library(riskRegression)
  library(randomForestSRC); library(prodlim); library(parallel)
})

# ---- repo root, derived relatively -----------------------------------------
af <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
this_file <- if (length(af)) normalizePath(sub("^--file=", "", af[1])) else
  normalizePath("src/v2/03_fit_r_models.R")
ROOT <- normalizePath(file.path(dirname(this_file), "..", ".."))

arg <- function(flag, default) {
  a <- commandArgs(trailingOnly = TRUE)
  i <- match(flag, a)
  if (is.na(i) || i == length(a)) default else a[i + 1L]
}
parse_range <- function(s) {
  unlist(lapply(strsplit(s, ",")[[1]], function(p) {
    if (grepl("-", p)) { z <- as.integer(strsplit(p, "-")[[1]]); z[1]:z[2] } else as.integer(p)
  }))
}
IMPS   <- parse_range(arg("--imps", "1-20"))
SPLITS <- strsplit(arg("--splits", "temporal,cv"), ",")[[1]]
NCORES <- as.integer(arg("--ncores", Sys.getenv("SLURM_CPUS_PER_TASK", "4")))
NTREE  <- as.integer(arg("--ntree", "300"))
NTIME  <- as.integer(arg("--ntime", "100"))
BATCH  <- as.integer(arg("--batch", "4"))   # batches of BATCH * NCORES tasks
WANT_MODELS <- strsplit(arg("--models", "cox_naive,rsf_naive,csc_cox,fine_gray,rsf_cr"), ",")[[1]]
REPEATS <- 0:2      # analysis repeats, see PREDICTION_FORMAT.md section 4
FOLDS   <- 0:4
SEED <- 20260903L

# one thread per worker; the node is billed whole and oversubscription wastes it
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
options(rf.cores = 1L, mc.cores = 1L)

IMPDIR  <- file.path(ROOT, "data/processed/imputed")
PREDDIR <- file.path(ROOT, "results/metrics_v2/predictions")
dir.create(PREDDIR, recursive = TRUE, showWarnings = FALSE)

FEATURES <- c("age","sex","sbp","antihtn","totchol","hdl","statin",
              "diabetes","smoke_current","bmi","egfr")
STD <- c("age","sbp","totchol","hdl","bmi","egfr")
HORIZ <- c(5, 10, 15)

MODELS <- list(
  cox_naive = list(family = "naive",     learner = "CoxPH"),
  rsf_naive = list(family = "naive",     learner = "RandomSurvivalForest"),
  csc_cox   = list(family = "competing", learner = "CauseSpecificCox"),
  fine_gray = list(family = "competing", learner = "FineGray"),
  rsf_cr    = list(family = "competing", learner = "RandomSurvivalForestCR")
)

# ---------------------------------------------------------------- helpers ----
std_fit_apply <- function(tr, te) {
  ctr <- vapply(STD, function(f) mean(tr[[f]]), 0)
  scl <- vapply(STD, function(f) stats::sd(tr[[f]]), 0)
  scl[!is.finite(scl) | scl == 0] <- 1
  for (f in STD) { tr[[f]] <- (tr[[f]] - ctr[[f]]) / scl[[f]]
                   te[[f]] <- (te[[f]] - ctr[[f]]) / scl[[f]] }
  list(tr = tr, te = te)
}

# 1 - S(t) from a naive Cox fitted with competing death treated as censoring.
fit_cox_naive <- function(tr, te, times, seed) {
  frm <- as.formula(paste("Surv(time, event_naive) ~", paste(FEATURES, collapse = "+")))
  f <- survival::coxph(frm, data = tr, x = TRUE, y = TRUE)
  riskRegression::predictRisk(f, newdata = te, times = times)
}
fit_csc_cox <- function(tr, te, times, seed) {
  frm <- as.formula(paste("Hist(time, event) ~", paste(FEATURES, collapse = "+")))
  f <- riskRegression::CSC(frm, data = tr, cause = 1)
  riskRegression::predictRisk(f, newdata = te, times = times, cause = 1)
}
fit_fine_gray <- function(tr, te, times, seed) {
  frm <- as.formula(paste("Hist(time, event) ~", paste(FEATURES, collapse = "+")))
  f <- riskRegression::FGR(frm, data = tr, cause = 1)
  riskRegression::predictRisk(f, newdata = te, times = times, cause = 1)
}
fit_rsf_naive <- function(tr, te, times, seed) {
  d <- tr[, c("time", "event_naive", FEATURES)]; names(d)[2] <- "status"
  # save.memory keeps the terminal-node statistics out of the forest object and
  # recomputes them at prediction time. Without it, 300 trees x ~1,100 terminal
  # nodes x ntime x 2 causes is on the order of a gigabyte per fitted forest, and
  # 128 concurrent forests exhaust the node: that is what killed job 5177708.
  f <- randomForestSRC::rfsrc(Surv(time, status) ~ ., data = d, ntree = NTREE,
                              nodesize = 30, splitrule = "logrank",
                              samptype = "swor", seed = -abs(seed), ntime = NTIME,
                              save.memory = TRUE, importance = "none",
                              perf.type = "none", statistics = FALSE)
  p <- predict(f, newdata = te[, FEATURES], importance = "none")
  rm(f); gc(verbose = FALSE)
  # survival matrix over p$time.interest -> risk = 1 - S(t)
  idx <- vapply(times, function(t) {
    w <- which(p$time.interest <= t); if (length(w)) max(w) else 0L }, 0L)
  out <- matrix(NA_real_, nrow = nrow(te), ncol = length(times))
  for (j in seq_along(times)) out[, j] <- if (idx[j] == 0L) 0 else 1 - p$survival[, idx[j]]
  out
}
fit_rsf_cr <- function(tr, te, times, seed) {
  d <- tr[, c("time", "event", FEATURES)]; names(d)[2] <- "status"
  d$status <- as.integer(d$status)
  f <- randomForestSRC::rfsrc(Surv(time, status) ~ ., data = d, ntree = NTREE,
                              nodesize = 30, splitrule = "logrankCR", cause = c(1, 0),
                              samptype = "swor", seed = -abs(seed), ntime = NTIME,
                              save.memory = TRUE, importance = "none",
                              perf.type = "none", statistics = FALSE)
  p <- predict(f, newdata = te[, FEATURES], importance = "none")
  rm(f); gc(verbose = FALSE)
  # p$cif is n x ntime x ncause
  idx <- vapply(times, function(t) {
    w <- which(p$time.interest <= t); if (length(w)) max(w) else 0L }, 0L)
  out <- matrix(NA_real_, nrow = nrow(te), ncol = length(times))
  for (j in seq_along(times)) out[, j] <- if (idx[j] == 0L) 0 else p$cif[, idx[j], 1]
  out
}
FITTERS <- list(cox_naive = fit_cox_naive, rsf_naive = fit_rsf_naive,
                csc_cox = fit_csc_cox, fine_gray = fit_fine_gray, rsf_cr = fit_rsf_cr)

assemble <- function(te, risk, model, imp, cv_rep = NA, cv_fold = NA) {
  risk[!is.finite(risk)] <- NA_real_
  risk[risk < 0] <- 0; risk[risk > 1] <- 1
  o <- data.frame(
    SEQN = as.integer(te$SEQN), cycle = as.character(te$cycle),
    time = as.numeric(te$time), event = as.integer(te$event),
    risk_5 = risk[, 1], risk_10 = risk[, 2], risk_15 = risk[, 3],
    model_name = model, family = MODELS[[model]]$family,
    learner_class = MODELS[[model]]$learner, imp = as.integer(imp),
    stringsAsFactors = FALSE)
  if (!is.na(cv_rep)) { o$cv_rep <- as.integer(cv_rep); o$cv_fold <- as.integer(cv_fold) }
  o
}

# ---------------------------------------------------------------- task list --
tasks <- list()
for (k in IMPS) {
  use <- intersect(names(MODELS), WANT_MODELS)
  if ("temporal" %in% SPLITS)
    for (m in use) tasks[[length(tasks) + 1L]] <-
      list(imp = k, model = m, split = "temporal", rep = NA, fold = NA)
  if ("cv" %in% SPLITS)
    for (m in use) for (r in REPEATS) for (f in FOLDS)
      tasks[[length(tasks) + 1L]] <-
        list(imp = k, model = m, split = "cv", rep = r, fold = f)
}
cat(sprintf("P2-A step 03: %d tasks, %d cores, ntree=%d\nROOT=%s\n",
            length(tasks), NCORES, NTREE, ROOT))

# cache the imputed frames once per worker
.cache <- new.env(parent = emptyenv())
get_imp <- function(k) {
  # cache exactly one imputation: holding 20 frames in every one of 128 forked
  # workers is a needless several gigabytes per worker.
  key <- as.character(k)
  if (!identical(.cache$key, key)) {
    .cache$key <- key
    .cache$data <- as.data.frame(
      read_parquet(file.path(IMPDIR, sprintf("imp_%d.parquet", k))))
  }
  .cache$data
}

run_task <- function(tk) {
  t0 <- Sys.time()
  d <- get_imp(tk$imp)
  if (tk$split == "temporal") {
    tr <- d[d$split == "train", ]; te <- d[d$split == "test", ]
    times <- c(5, 10, 15); seed <- SEED + 1000L * tk$imp
  } else {
    col <- paste0("cv_rep", tk$rep)
    tr <- d[d[[col]] != tk$fold, ]; te <- d[d[[col]] == tk$fold, ]
    times <- c(5, 10, 15); seed <- SEED + 1000L * tk$imp + 100L * tk$rep + tk$fold
  }
  s <- std_fit_apply(tr, te)
  set.seed(seed)
  risk <- FITTERS[[tk$model]](s$tr, s$te, times, seed)
  risk <- as.matrix(risk)
  # 15 y is not estimable on the temporal test set (max follow-up 13.2 y)
  if (tk$split == "temporal") risk[, 3] <- NA_real_
  o <- assemble(te, risk, tk$model, tk$imp, tk$rep, tk$fold)
  rm(risk, s, tr, te); gc(verbose = FALSE)
  tag <- if (tk$split == "temporal") sprintf("%s__temporal__imp%d", tk$model, tk$imp)
         else sprintf("%s__cvr%df%d__imp%d", tk$model, tk$rep, tk$fold, tk$imp)
  write_parquet(o, file.path(PREDDIR, "parts", paste0(tag, ".parquet")))
  data.frame(task = tag, n = nrow(o), mean_risk10 = mean(o$risk_10, na.rm = TRUE),
             secs = as.numeric(Sys.time() - t0, units = "secs"), stringsAsFactors = FALSE)
}

dir.create(file.path(PREDDIR, "parts"), recursive = TRUE, showWarnings = FALSE)
# skip work already on disk so an interrupted run resumes
done <- basename(Sys.glob(file.path(PREDDIR, "parts", "*.parquet")))
tag_of <- function(tk) if (tk$split == "temporal")
  sprintf("%s__temporal__imp%d.parquet", tk$model, tk$imp) else
  sprintf("%s__cvr%df%d__imp%d.parquet", tk$model, tk$rep, tk$fold, tk$imp)
keep <- !vapply(tasks, function(tk) tag_of(tk) %in% done, TRUE)
cat(sprintf("  %d already on disk, %d to run\n", sum(!keep), sum(keep)))
tasks <- tasks[keep]

t0 <- Sys.time()
safe_run <- function(tk) {
  tryCatch(run_task(tk), error = function(e)
    data.frame(task = tag_of(tk), n = NA_integer_, mean_risk10 = NA_real_,
               secs = NA_real_, err = conditionMessage(e),
               stringsAsFactors = FALSE))
}
# Batched, prescheduled parallelism. mc.preschedule = FALSE forks once per task,
# which for 1,600 tasks means 1,600 forks of a parent that has grown, and that is
# what failed in job 5177708. Prescheduling forks NCORES workers per batch and
# splits the batch among them; the batch boundary is also where memory is
# reclaimed and where progress becomes visible in the log.
t0 <- Sys.time()
res <- list()
if (NCORES > 1L && length(tasks) > 1L) {
  bs <- max(1L, BATCH * NCORES)
  nb <- ceiling(length(tasks) / bs)
  for (bi in seq_len(nb)) {
    lo <- (bi - 1L) * bs + 1L; hi <- min(bi * bs, length(tasks))
    tb <- tasks[lo:hi]
    rb <- parallel::mclapply(tb, safe_run, mc.cores = min(NCORES, length(tb)),
                             mc.preschedule = TRUE)
    res <- c(res, rb)
    gc(verbose = FALSE)
    cat(sprintf("  batch %d/%d done (%d/%d tasks), %.1f min elapsed\n",
                bi, nb, hi, length(tasks),
                as.numeric(Sys.time() - t0, units = "mins")))
    flush.console()
  }
} else {
  res <- lapply(tasks, safe_run)
}
cat(sprintf("fitting done in %.1f min\n", as.numeric(Sys.time() - t0, units = "mins")))

errs <- Filter(function(x) "err" %in% names(x), res)
if (length(errs)) { for (e in errs) cat("FAILED:", e$task, "|", e$err, "\n")
                    stop(sprintf("%d tasks failed", length(errs))) }
if (length(res)) {
  log <- do.call(rbind, res)
  write.csv(log, file.path(ROOT, "results/metrics_v2", "fit_log_r.csv"),
            row.names = FALSE)
  cat(sprintf("median fit time %.1f s, total %.2f core-hours\n",
              median(log$secs), sum(log$secs) / 3600))
}
cat("DONE step 03\n")
