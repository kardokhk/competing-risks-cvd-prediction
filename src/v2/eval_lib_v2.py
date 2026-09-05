"""P2-A evaluation library, version 2.

Implements the estimators of van Geloven et al., BMJ 2022;377:e069249 and their
companion code at github.com/survival-lumc/ValidationCompRisks (accessed
2026-09-03), reimplemented in numpy so that a 2,000-replicate shared-index
bootstrap over 20 imputations is affordable. `src/v2/07_validate_metrics.R`
checks every estimator here against the R reference implementations
(riskRegression, pec, geepack, prodlim) on one imputation.

Estimator set, in the Van Calster hierarchy:

  mean      E/O       mean(predicted) / Aalen-Johansen observed CIF
  weak      slope     pseudo-observation GEE, complementary log-log MEAN link,
            intercept cloglog(predicted) as offset. Independence working
                      correlation with one row per subject reduces the GEE to a
                      quasi-likelihood GLM with constant variance, which is what
                      is fitted here.
  moderate  curve     the same pseudo-value model with a restricted cubic spline
            ICI/E50/E90  in cloglog(predicted); ICI = mean|observed - predicted|,
                      E50 and E90 the 50th and 90th percentiles of that absolute
                      difference.
  discrim   AUC       Blanche IPCW cause-specific time-dependent AUC, competing
                      events counted as controls (timeROC's AUC_2).
            C         Wolbers truncated IPCW cause-specific concordance.
  utility   NB        competing-risk net benefit with an Aalen-Johansen event
                      rate among those above threshold.

Follow-up time is recorded in whole months, so the sample has at most about 250
distinct times. Every estimator below exploits that: the Aalen-Johansen
estimator, its exact leave-one-out jackknife pseudo-values and the censoring
distribution all reduce to operations on a K x K table with K <= 260, which is
why exact pseudo-values are recomputed inside every bootstrap replicate rather
than being held fixed.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "risk_table", "aj_cif_at", "aj_cif_curve", "pseudo_cif", "pseudo_net_risk",
    "km_net_risk_at",
    "censor_surv", "censor_surv_stratified",
    "calib_intercept_slope", "flexible_calibration",
    "ipcw_auc", "cr_cindex", "net_benefit", "net_benefit_all",
    "cloglog", "rcs_basis", "at_risk_fraction",
]

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Risk-set table
# --------------------------------------------------------------------------- #
def risk_table(time, event, weights=None):
    """Collapse (time, event) to the K distinct times with at-risk and event counts.

    Returns u (K,), n_risk (K,), d_any (K,), d1 (K,), d2 (K,), d_cens (K,).
    `weights` allows a weighted (e.g. multiplicity-coded bootstrap) table.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, np.int64)
    w = np.ones(len(time)) if weights is None else np.asarray(weights, float)
    u, inv = np.unique(time, return_inverse=True)
    K = len(u)
    n_at = np.bincount(inv, weights=w, minlength=K)
    d_any = np.bincount(inv, weights=w * (event > 0), minlength=K)
    d1 = np.bincount(inv, weights=w * (event == 1), minlength=K)
    d2 = np.bincount(inv, weights=w * (event == 2), minlength=K)
    d_c = np.bincount(inv, weights=w * (event == 0), minlength=K)
    n_risk = w.sum() - np.concatenate(([0.0], np.cumsum(n_at)[:-1]))
    return u, n_risk, d_any, d1, d2, d_c


def _aj_from_table(u, n_risk, d_any, d_cause):
    """Aalen-Johansen CIF as a step function over u."""
    haz = np.divide(d_any, n_risk, out=np.zeros_like(d_any), where=n_risk > 0)
    s_prev = np.concatenate(([1.0], np.cumprod(1.0 - haz)[:-1]))
    inc = s_prev * np.divide(d_cause, n_risk, out=np.zeros_like(d_cause), where=n_risk > 0)
    return np.cumsum(inc)


def aj_cif_at(time, event, t, cause=1, weights=None):
    """Aalen-Johansen CIF for `cause` at time t. Exact, ties handled exactly."""
    u, n_risk, d_any, d1, d2, _ = risk_table(time, event, weights)
    cif = _aj_from_table(u, n_risk, d_any, d1 if cause == 1 else d2)
    k = np.searchsorted(u, t, side="right") - 1
    return float(cif[k]) if k >= 0 else 0.0


def aj_cif_curve(time, event, cause=1, weights=None):
    """Return (u, CIF(u)) for plotting."""
    u, n_risk, d_any, d1, d2, _ = risk_table(time, event, weights)
    return u, _aj_from_table(u, n_risk, d_any, d1 if cause == 1 else d2)


def at_risk_fraction(time, event, t):
    """Fraction of the sample that is informative about the horizon t.

    A subject is informative if they were followed to t (time >= t) or had an
    event of either cause before t. A subject censored before t is not.
    Reported alongside every stratified result: a stratum can be nominally large
    and still carry almost no information at 15 y.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, np.int64)
    return float(np.mean((time >= t) | ((event > 0) & (time < t))))


# --------------------------------------------------------------------------- #
# Exact leave-one-out jackknife pseudo-observations for the AJ CIF
# --------------------------------------------------------------------------- #
def pseudo_cif(time, event, t, cause=1):
    """Exact jackknife pseudo-observations for the AJ CIF at t.

        theta_i = n * CIF(t) - (n - 1) * CIF^{(-i)}(t)

    Two subjects with the same (time, event) have the same pseudo-value, and
    follow-up is recorded in whole months, so only 3K leave-one-out estimators
    are distinct with K the number of unique times (about 250 here). All 3K are
    computed at once as a (3K, K) array, which makes this exact and cheap enough
    to recompute inside every bootstrap replicate.

    Returns an array of length n.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, np.int64)
    n = len(time)
    u, inv = np.unique(time, return_inverse=True)
    K = len(u)
    n_at = np.bincount(inv, minlength=K)
    d_any = np.bincount(inv, weights=(event > 0).astype(float), minlength=K)
    d_c = np.bincount(inv, weights=(event == cause).astype(float), minlength=K)
    n_risk = float(n) - np.concatenate(([0.0], np.cumsum(n_at)[:-1]))

    kt = np.searchsorted(u, t, side="right") - 1
    if kt < 0:
        return np.zeros(n)
    full = _aj_from_table(u, n_risk, d_any, d_c)[kt]

    # Leave one out: a subject with time u_m and event type c leaves the risk set
    # at every k <= m, and removes one event at k == m if they had one.
    ks = np.arange(K)
    below = (ks[None, :] <= ks[:, None]).astype(float)      # (m, k) 1 if k <= m
    n_lo = n_risk[None, :] - below                           # (K, K)
    eye = np.eye(K)
    # three event types: 0 censored, 1 cause of interest, 2 competing
    d_any_lo = np.stack([d_any[None, :] - 0.0 * eye,
                         d_any[None, :] - eye,
                         d_any[None, :] - eye])              # (3, K, K)
    if cause == 1:
        d_c_lo = np.stack([d_c[None, :] - 0.0 * eye,
                           d_c[None, :] - eye,
                           d_c[None, :] - 0.0 * eye])
    else:
        d_c_lo = np.stack([d_c[None, :] - 0.0 * eye,
                           d_c[None, :] - 0.0 * eye,
                           d_c[None, :] - eye])
    nn = np.broadcast_to(n_lo, d_any_lo.shape)
    ok = nn > 0
    haz = np.divide(d_any_lo, nn, out=np.zeros_like(d_any_lo), where=ok)
    haz = np.clip(haz, 0.0, 1.0)
    s = np.cumprod(1.0 - haz, axis=2)
    s_prev = np.concatenate([np.ones(s.shape[:2] + (1,)), s[:, :, :-1]], axis=2)
    inc = s_prev * np.divide(d_c_lo, nn, out=np.zeros_like(d_c_lo), where=ok)
    cif_lo = np.cumsum(inc, axis=2)[:, :, kt]                # (3, K)

    theta_grid = n * full - (n - 1) * cif_lo                 # (3, K)
    return theta_grid[np.clip(event, 0, 2), inv]


def km_net_risk_at(time, event, t, cause=1):
    """1 - S(t) from a Kaplan-Meier that treats the competing event as censoring.

    This is the quantity a NAIVE model estimates. It is not a cumulative
    incidence and it is not directly observable, but it is the target the naive
    model is actually aiming at, and assessing a naive model against the
    Aalen-Johansen cumulative incidence is assessing it against a different
    estimand.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, np.int64)
    u, inv = np.unique(time, return_inverse=True)
    K = len(u)
    n_at = np.bincount(inv, minlength=K)
    d_c = np.bincount(inv, weights=(event == cause).astype(float), minlength=K)
    n_risk = float(len(time)) - np.concatenate(([0.0], np.cumsum(n_at)[:-1]))
    haz = np.divide(d_c, n_risk, out=np.zeros_like(d_c), where=n_risk > 0)
    S = np.cumprod(1.0 - haz)
    k = np.searchsorted(u, t, side="right") - 1
    return float(1.0 - S[k]) if k >= 0 else 0.0


def pseudo_net_risk(time, event, t, cause=1):
    """Exact jackknife pseudo-observations for 1 - S(t), competing event censored.

    The Kaplan-Meier analogue of `pseudo_cif`, and the concordant target for a
    naive model. Same construction: only 3K leave-one-out estimators are
    distinct, so all are computed at once.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, np.int64)
    n = len(time)
    u, inv = np.unique(time, return_inverse=True)
    K = len(u)
    n_at = np.bincount(inv, minlength=K)
    d_c = np.bincount(inv, weights=(event == cause).astype(float), minlength=K)
    n_risk = float(n) - np.concatenate(([0.0], np.cumsum(n_at)[:-1]))
    kt = np.searchsorted(u, t, side="right") - 1
    if kt < 0:
        return np.zeros(n)
    haz = np.divide(d_c, n_risk, out=np.zeros_like(d_c), where=n_risk > 0)
    full = float(1.0 - np.cumprod(1.0 - haz)[kt])

    ks = np.arange(K)
    below = (ks[None, :] <= ks[:, None]).astype(float)
    n_lo = n_risk[None, :] - below
    eye = np.eye(K)
    # event types 0, 1, 2: only an event of `cause` removes an event count
    d_lo = np.stack([d_c[None, :] - (eye if c == cause else 0.0 * eye)
                     for c in (0, 1, 2)])
    nn = np.broadcast_to(n_lo, d_lo.shape)
    ok = nn > 0
    h = np.clip(np.divide(d_lo, nn, out=np.zeros_like(d_lo), where=ok), 0.0, 1.0)
    risk_lo = 1.0 - np.cumprod(1.0 - h, axis=2)[:, :, kt]
    theta_grid = n * full - (n - 1) * risk_lo
    return theta_grid[np.clip(event, 0, 2), inv]


# --------------------------------------------------------------------------- #
# Censoring distribution (for IPCW), optionally stratified
# --------------------------------------------------------------------------- #
def censor_surv(time, event, eval_time, naive=False):
    """Kaplan-Meier of the censoring distribution G, evaluated at eval_time.

    `naive=True` treats a competing death as a censoring event, which is the
    censoring distribution that a naive analysis implicitly assumes.
    Evaluation is left-continuous, G(s-), as the IPCW estimators require.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, np.int64)
    cens = (event == 0) | (naive & (event == 2))
    u, inv = np.unique(time, return_inverse=True)
    K = len(u)
    n_at = np.bincount(inv, minlength=K)
    d_c = np.bincount(inv, weights=cens.astype(float), minlength=K)
    n_risk = float(len(time)) - np.concatenate(([0.0], np.cumsum(n_at)[:-1]))
    haz = np.divide(d_c, n_risk, out=np.zeros_like(d_c), where=n_risk > 0)
    G = np.cumprod(1.0 - haz)
    G_prev = np.concatenate(([1.0], G[:-1]))          # G(u_k -)
    del G_prev
    et = np.asarray(eval_time, float)
    # number of unique event times strictly before et; G(et-) is the product over those
    idx = np.searchsorted(u, et, side="left")
    out = np.where(idx == 0, 1.0, G[np.clip(idx - 1, 0, K - 1)])
    return np.maximum(out, EPS)


def censor_surv_stratified(time, event, strata, eval_time, naive=False):
    """G(s-) estimated WITHIN each stratum.

    Censoring here is administrative: it is driven entirely by the calendar gap
    between the examination and the end of mortality linkage, so it differs by
    NHANES cycle from 0% to essentially 100% before 10 y. A single marginal
    Kaplan-Meier averages censoring hazards that do not overlap, so all IPCW in
    this re-analysis is cycle-stratified. `06_evaluate.py` reports the size of
    the change against the marginal version.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, np.int64)
    strata = np.asarray(strata)
    eval_time = np.broadcast_to(np.asarray(eval_time, float), time.shape).copy()
    out = np.ones(len(time))
    for s in np.unique(strata):
        m = strata == s
        out[m] = censor_surv(time[m], event[m], eval_time[m], naive=naive)
    return np.maximum(out, EPS)


# --------------------------------------------------------------------------- #
# Weak and moderate calibration: quasi-likelihood GLM, cloglog mean link
# --------------------------------------------------------------------------- #
def cloglog(p, lo=1e-8, hi=1 - 1e-8):
    p = np.clip(np.asarray(p, float), lo, hi)
    return np.log(-np.log(1.0 - p))


def _irls_cloglog(X, y, offset, max_iter=60, tol=1e-9):
    """Gaussian-family quasi-likelihood with a complementary log-log MEAN link.

    Equivalent to geepack::geese(family = gaussian, mean.link = "cloglog",
    corstr = "independence", scale.fix = TRUE) when each id has one row, which
    is the case at a single horizon. Returns (beta, sandwich_se, converged).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    off = np.asarray(offset, float)
    n, p = X.shape
    beta = np.zeros(p)
    # start from the offset-only fit
    eta = off.copy()
    for _ in range(max_iter):
        eta = off + X @ beta
        eta = np.clip(eta, -30, 10)
        expe = np.exp(eta)
        mu = 1.0 - np.exp(-expe)
        dmu = np.exp(eta - expe)                    # dmu/deta
        dmu = np.maximum(dmu, 1e-10)
        w = dmu ** 2                                # gaussian variance == 1
        z = (eta - off) + (y - mu) / dmu            # working response net of offset
        XtW = X.T * w
        A = XtW @ X
        b = XtW @ z
        try:
            new = np.linalg.solve(A + 1e-10 * np.eye(p), b)
        except np.linalg.LinAlgError:
            return beta, np.full(p, np.nan), False
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    # sandwich (robust) standard errors
    eta = np.clip(off + X @ beta, -30, 10)
    expe = np.exp(eta)
    mu = 1.0 - np.exp(-expe)
    dmu = np.maximum(np.exp(eta - expe), 1e-10)
    r = y - mu
    D = X * dmu[:, None]
    bread = D.T @ D
    try:
        binv = np.linalg.inv(bread + 1e-10 * np.eye(p))
    except np.linalg.LinAlgError:
        return beta, np.full(p, np.nan), False
    meat = (D * r[:, None]).T @ (D * r[:, None])
    V = binv @ meat @ binv
    return beta, np.sqrt(np.maximum(np.diag(V), 0.0)), True


def calib_intercept_slope(pseudo, pred, with_se=False):
    """Weak calibration: intercept and slope on the complementary log-log scale.

    Follows ValidationCompRisks/Prediction_CSC.Rmd exactly:
      intercept model  pseudo ~ offset(cll_pred)                -> report (Intercept)
      slope model      pseudo ~ offset(cll_pred) + cll_pred     -> report 1 + b_cll
    Perfect calibration is intercept 0 and slope 1.
    """
    cll = cloglog(pred)
    n = len(cll)
    b_i, se_i, ok_i = _irls_cloglog(np.ones((n, 1)), pseudo, cll)
    Xs = np.column_stack([np.ones(n), cll])
    b_s, se_s, ok_s = _irls_cloglog(Xs, pseudo, cll)
    intercept = float(b_i[0]) if ok_i else np.nan
    slope = float(1.0 + b_s[1]) if ok_s else np.nan
    if with_se:
        return intercept, slope, (float(se_i[0]) if ok_i else np.nan,
                                  float(se_s[1]) if ok_s else np.nan)
    return intercept, slope


def rcs_basis(x, knots):
    """Harrell restricted cubic spline basis, k knots -> k-1 columns (x plus k-2).

    Austin, Putter, Lee & van Buuren (Diagn Progn Res 2022, doi:10.1186/s41512-021-00114-6)
    concluded that three knots is preferable to four or five for a calibration
    curve, so three is the default here and five is run as a sensitivity.
    """
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    K = len(k)
    # Harrell's normalisation, (t_K - t_1)^2, as in rms::rcspline.eval. It only
    # rescales the basis columns, so the fitted values are invariant to it; it is
    # matched to rms here so the basis is literally the same one.
    denom = (k[-1] - k[0]) ** 2.0
    cols = [x]
    for j in range(K - 2):
        t = lambda a, b: np.maximum(a - b, 0.0) ** 3
        term = (t(x, k[j])
                - t(x, k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                + t(x, k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2]))
        cols.append(term / denom)
    return np.column_stack(cols)


def flexible_calibration(pseudo, pred, nknots=3, grid=None, return_curve=False):
    """Moderate calibration: ICI, E50, E90 and the smoothed calibration curve.

    Pseudo-values regressed on a restricted cubic spline of cloglog(predicted),
    same quasi-likelihood cloglog-mean model as the weak calibration. The fitted
    values are the smoothed observed risks; ICI is their mean absolute distance
    from the predicted risk (Austin & Steyerberg).
    """
    cll = cloglog(pred)
    # Knots at the 10th, 50th and 90th percentiles for three knots, following
    # Austin, Putter, Lee and van Buuren (Diagn Progn Res 2022;6:2,
    # doi:10.1186/s41512-021-00114-6), whose conclusion is that three knots is
    # preferable to four or five for a calibration curve. Five knots are run as a
    # sensitivity.
    qs = (np.array([0.10, 0.50, 0.90]) if nknots == 3
          else np.linspace(0.05, 0.95, nknots))
    knots = np.quantile(cll, qs)
    if len(np.unique(knots)) < nknots:
        knots = np.quantile(cll, np.linspace(0.02, 0.98, nknots))
    if len(np.unique(knots)) < nknots:
        return dict(ici=np.nan, e50=np.nan, e90=np.nan, emax=np.nan, curve=None)
    B = rcs_basis(cll, knots)
    X = np.column_stack([np.ones(len(cll)), B])
    beta, _, ok = _irls_cloglog(X, pseudo, np.zeros(len(cll)))
    if not ok:
        return dict(ici=np.nan, e50=np.nan, e90=np.nan, emax=np.nan, curve=None)
    eta = np.clip(X @ beta, -30, 10)
    obs = 1.0 - np.exp(-np.exp(eta))
    d = np.abs(obs - np.asarray(pred, float))
    out = dict(ici=float(np.mean(d)), e50=float(np.median(d)),
               e90=float(np.quantile(d, 0.90)), emax=float(np.max(d)))
    if return_curve:
        if grid is None:
            grid = np.quantile(pred, np.linspace(0.005, 0.995, 200))
        gl = cloglog(grid)
        Xg = np.column_stack([np.ones(len(gl)), rcs_basis(gl, knots)])
        eg = np.clip(Xg @ beta, -30, 10)
        out["curve"] = (grid, 1.0 - np.exp(-np.exp(eg)))
    return out


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #
def ipcw_auc(time, event, risk, t, G_at_T=None, G_at_t=None, cause=1):
    """Blanche IPCW cause-specific time-dependent AUC at horizon t.

    Cases  : event == cause and time <= t, weight 1 / G(T_i-).
    Controls: time > t, weight 1 / G(t); or a competing event with time <= t,
              weight 1 / G(T_j-). Competing events are controls, which is
              timeROC's AUC_2 and the definition riskRegression::Score uses.

    Because case and control weights do not depend on the pair, the double sum
    factorises into one sort and one cumulative sum, so this is exact and O(n log n).
    `G_at_T` and `G_at_t` are supplied by the caller so the censoring model can be
    cycle-stratified.
    """
    time = np.asarray(time, float); event = np.asarray(event, np.int64)
    risk = np.asarray(risk, float)
    ok = np.isfinite(risk)
    time, event, risk = time[ok], event[ok], risk[ok]
    G_at_T = np.asarray(G_at_T, float)[ok]
    G_at_t = np.asarray(G_at_t, float)[ok]

    case = (event == cause) & (time <= t)
    ctrl_late = time > t
    ctrl_comp = (event != cause) & (event > 0) & (time <= t)
    if case.sum() == 0 or (ctrl_late.sum() + ctrl_comp.sum()) == 0:
        return np.nan
    wi = np.zeros(len(time)); wj = np.zeros(len(time))
    wi[case] = 1.0 / G_at_T[case]
    wj[ctrl_late] = 1.0 / G_at_t[ctrl_late]
    wj[ctrl_comp] = 1.0 / G_at_T[ctrl_comp]

    order = np.argsort(risk, kind="mergesort")
    r = risk[order]; a = wi[order]; b = wj[order]
    # cumulative control weight strictly below, and at, each position
    cum_b = np.concatenate(([0.0], np.cumsum(b)))
    # group ties
    starts = np.concatenate(([True], r[1:] != r[:-1]))
    grp = np.cumsum(starts) - 1
    gb = np.bincount(grp, weights=b)
    below = np.concatenate(([0.0], np.cumsum(gb)[:-1]))[grp]
    same = gb[grp]
    num = np.sum(a * (below + 0.5 * same))
    den = wi.sum() * wj.sum()
    return float(num / den) if den > 0 else np.nan


def cr_cindex(time, event, risk, t, G_at_T=None, cause=1, nbins=1024):
    """Wolbers truncated IPCW cause-specific concordance at horizon t.

    Comparable pairs, following Wolbers et al. and pec::cindex: a case i with
    event == cause and time <= t against either a subject still at risk after
    T_i, weight 1/G(T_i-)^2, or a subject with a competing event at any time,
    weight 1/(G(T_i-) G(T_j-)). Concordant if risk_i > risk_j.

    The weights do not factorise across the pair, so the double sum is evaluated
    by sweeping cases in increasing time and holding a histogram of the risk
    ranks of the currently comparable controls. The risk axis is discretised into
    `nbins` quantile bins; with 1,024 bins the discretisation error on the point
    estimate is below 1e-4, which `07_validate_metrics.R` checks against
    pec::cindex.
    """
    time = np.asarray(time, float); event = np.asarray(event, np.int64)
    risk = np.asarray(risk, float)
    ok = np.isfinite(risk)
    time, event, risk = time[ok], event[ok], risk[ok]
    G = np.asarray(G_at_T, float)[ok]
    n = len(time)
    case_mask = (event == cause) & (time <= t)
    if case_mask.sum() == 0:
        return np.nan

    edges = np.quantile(risk, np.linspace(0, 1, nbins + 1)[1:-1])
    bins = np.searchsorted(edges, risk, side="left")        # 0..nbins-1

    order_t = np.argsort(time, kind="mergesort")
    ts = time[order_t]; bs = bins[order_t]; es = event[order_t]; Gs = G[order_t]

    # Controls of type A: still at risk after T_i. Everyone starts in the pool and
    # leaves as the sweep advances. Controls of type B: competing events already
    # observed, carrying weight 1 / G(T_j-).
    histA = np.bincount(bs, minlength=nbins).astype(float)
    histB = np.zeros(nbins)

    is_case = (es == cause) & (ts <= t)
    case_times = np.unique(ts[is_case])
    num = 0.0
    den = 0.0
    i = 0
    # The histograms change only at distinct times, so the sweep runs over the
    # roughly 250 distinct case times, not over every case.
    for tm in case_times:
        while i < n and ts[i] <= tm:
            histA[bs[i]] -= 1.0
            if es[i] == 2:
                histB[bs[i]] += 1.0 / max(Gs[i], EPS)
            i += 1
        cumA = np.cumsum(histA); cumB = np.cumsum(histB)
        totA = cumA[-1]; totB = cumB[-1]
        sel = is_case & (ts == tm)
        b = bs[sel]
        gi = np.maximum(Gs[sel], EPS)
        wA = 1.0 / (gi * gi); wB = 1.0 / gi
        A_lo = np.where(b > 0, cumA[np.maximum(b - 1, 0)], 0.0)
        B_lo = np.where(b > 0, cumB[np.maximum(b - 1, 0)], 0.0)
        A_eq = histA[b]; B_eq = histB[b]
        num += np.sum(wA * (A_lo + 0.5 * A_eq) + wB * (B_lo + 0.5 * B_eq))
        den += np.sum(wA * totA + wB * totB)
    return float(num / den) if den > 0 else np.nan


# --------------------------------------------------------------------------- #
# Net benefit
# --------------------------------------------------------------------------- #
def net_benefit(time, event, risk, pt, t, cause=1):
    """Competing-risk net benefit of treating everyone with risk >= pt.

    ValidationCompRisks/R/stdca.R:
        p_exceed = mean(risk >= pt)
        f        = Aalen-Johansen CIF at t among those above threshold
        TP = f * p_exceed ;  FP = (1 - f) * p_exceed
        NB = TP - FP * pt/(1 - pt)
    """
    risk = np.asarray(risk, float)
    flag = risk >= pt
    p_ex = float(np.mean(flag))
    if p_ex == 0:
        return 0.0
    f = aj_cif_at(np.asarray(time)[flag], np.asarray(event)[flag], t, cause=cause)
    w = pt / (1.0 - pt)
    return float(f * p_ex - (1.0 - f) * p_ex * w)


def net_benefit_all(time, event, pt, t, cause=1):
    """Net benefit of treating everyone. NB_none is 0 by definition."""
    f = aj_cif_at(time, event, t, cause=cause)
    w = pt / (1.0 - pt)
    return float(f - (1.0 - f) * w)
