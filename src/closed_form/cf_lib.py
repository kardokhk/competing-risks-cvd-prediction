"""Closed-form relation between the naive (1-KM) and competing-risk (CIF) estimands.

Cause 1 = CVD death, cause 2 = competing (non-CVD) death.

Notation used throughout
-----------------------
lambda_j(u)   cause-specific hazard for cause j
Lambda_j(t)   = int_0^t lambda_j
S_j(t)        = exp(-Lambda_j(t))            cause-j NET (latent) survival
S(t)          = exp(-Lambda_1 - Lambda_2) = S_1(t) S_2(t)   all-cause survival
F_j(t)        = int_0^t S(u) lambda_j(u) du  cause-j cumulative incidence (Aalen-Johansen)
A(t)          = 1 - S_1(t)                   the NAIVE estimand (1 - Kaplan-Meier)
G(t)          = A(t) - F_1(t)                the over-prediction gap

Author: agent P2-B.  Seed 20260903.
"""
from __future__ import annotations

import numpy as np

SEED = 20260903

__all__ = [
    "SEED", "gap_exact_ph", "naive_exact_ph", "gap_product", "gap_kappa",
    "eo_inflation_exact_ph", "aalen_johansen", "km_naive", "cif_and_naive",
    "kappa_nelson_aalen", "bootstrap_ci",
]


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------
def naive_exact_ph(F1, F2):
    """Naive estimand A = 1 - S_1(t), EXACT under proportional cause-specific
    hazards (lambda_2(u) = theta * lambda_1(u) for all u, any theta >= 0, any
    time shape).  Depends on F_1 and F_2 only.

        p   = F_1 + F_2 = 1 - S(t)
        pi  = F_1 / p
        A   = 1 - (1 - p)^pi
    """
    F1 = np.asarray(F1, dtype=float)
    F2 = np.asarray(F2, dtype=float)
    p = F1 + F2
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = np.where(p > 0, F1 / np.where(p > 0, p, 1.0), 0.0)
    A = 1.0 - np.power(np.clip(1.0 - p, 1e-300, 1.0), pi)
    return np.where(p > 0, A, 0.0)


def gap_exact_ph(F1, F2):
    """G = A - F_1, exact under proportional cause-specific hazards."""
    return naive_exact_ph(F1, F2) - np.asarray(F1, dtype=float)


def gap_product(F1, F2, k=1.0):
    """The 'product' approximation G ~ k * F_1 * F_2.

    k = 0.5 is the correct SECOND-ORDER coefficient under proportional
    cause-specific hazards; k = 1.0 is the naive product used as a comparator.
    """
    return k * np.asarray(F1, dtype=float) * np.asarray(F2, dtype=float)


def gap_kappa(F1, F2, kappa):
    """General second-order form G ~ kappa * F_1 * F_2, with the hazard-timing
    overlap coefficient kappa = int_0^t [Lambda_2(u)/Lambda_2(t)] dLambda_1(u)/Lambda_1(t).
    kappa = 1/2 exactly when the two cause-specific hazards are proportional.
    """
    return np.asarray(kappa, dtype=float) * np.asarray(F1, dtype=float) * np.asarray(F2, dtype=float)


def eo_inflation_exact_ph(F1, F2):
    """Relative over-prediction A / F_1 (the multiplicative inflation of a naive
    model's expected/observed ratio), exact under proportional cause-specific hazards.
    Second-order limit: 1 + F_2 / 2."""
    F1 = np.asarray(F1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(F1 > 0, naive_exact_ph(F1, F2) / np.where(F1 > 0, F1, 1.0), np.nan)


# --------------------------------------------------------------------------
# Non-parametric estimators (numpy, no external survival dependency)
# --------------------------------------------------------------------------
def _risk_table(time, event):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    order = np.argsort(time, kind="mergesort")
    time, event = time[order], event[order]
    ut, first = np.unique(time, return_index=True)
    n = time.size
    at_risk = n - first                       # number at risk just before each unique time
    d1 = np.bincount(np.searchsorted(ut, time[event == 1]), minlength=ut.size).astype(float)
    d2 = np.bincount(np.searchsorted(ut, time[event == 2]), minlength=ut.size).astype(float)
    return ut, at_risk.astype(float), d1, d2, n


def cif_and_naive(time, event, horizon):
    """Aalen-Johansen CIFs for both causes, the naive 1-KM for cause 1, and
    diagnostics, all evaluated at `horizon`.

    Returns dict with
        F1, F2   Aalen-Johansen cause-1 / cause-2 CIF
        A_naive  1 - Kaplan-Meier for cause 1 with competing deaths censored
        G_obs    A_naive - F1  (the observed gap)
        S_all    overall Kaplan-Meier survival
        at_risk_frac  proportion of the stratum still at risk at `horizon`
        n_ev1, n_ev2  events of each cause on or before `horizon`
    """
    ut, nrisk, d1, d2, n = _risk_table(time, event)
    keep = ut <= horizon
    ut, nrisk, d1, d2 = ut[keep], nrisk[keep], d1[keep], d2[keep]
    if ut.size == 0:
        return dict(F1=0.0, F2=0.0, A_naive=0.0, G_obs=0.0, S_all=1.0,
                    at_risk_frac=1.0, n_ev1=0, n_ev2=0, n=n)
    d = d1 + d2
    # overall KM, left-continuous (value just before each event time)
    surv = np.cumprod(1.0 - d / nrisk)
    S_lag = np.concatenate(([1.0], surv[:-1]))
    F1 = float(np.sum(S_lag * d1 / nrisk))
    F2 = float(np.sum(S_lag * d2 / nrisk))
    # naive KM for cause 1: competing deaths treated as censored
    km1 = float(np.prod(1.0 - d1 / nrisk))
    at_risk_frac = float((np.asarray(time) > horizon).sum() / n)
    return dict(F1=F1, F2=F2, A_naive=1.0 - km1, G_obs=(1.0 - km1) - F1,
                S_all=float(surv[-1]), at_risk_frac=at_risk_frac,
                n_ev1=int(d1.sum()), n_ev2=int(d2.sum()), n=int(n))


def aalen_johansen(time, event, horizon):
    r = cif_and_naive(time, event, horizon)
    return r["F1"], r["F2"]


def km_naive(time, event, horizon):
    return cif_and_naive(time, event, horizon)["A_naive"]


def kappa_nelson_aalen(time, event, horizon):
    """Non-parametric estimate of the hazard-timing overlap coefficient

        kappa(t) = int_0^t [Lambda_2(u-)/Lambda_2(t)] dLambda_1(u) / Lambda_1(t)

    using Nelson-Aalen increments d_j / n_at_risk.  kappa = 1/2 exactly if the
    two cause-specific hazards are proportional over time; kappa > 1/2 when the
    competing hazard accrues EARLIER than the cause-1 hazard.
    """
    ut, nrisk, d1, d2, n = _risk_table(time, event)
    keep = ut <= horizon
    nrisk, d1, d2 = nrisk[keep], d1[keep], d2[keep]
    if d1.sum() == 0 or d2.sum() == 0:
        return np.nan
    dL1 = d1 / nrisk
    dL2 = d2 / nrisk
    L1t, L2t = dL1.sum(), dL2.sum()
    L2_lag = np.concatenate(([0.0], np.cumsum(dL2)[:-1]))   # Lambda_2(u-)
    return float(np.sum((L2_lag / L2t) * dL1) / L1t)


# --------------------------------------------------------------------------
def bootstrap_ci(fn, n, n_boot=2000, seed=SEED, alpha=0.05):
    """Percentile bootstrap over row indices 0..n-1.  `fn(idx)` returns a scalar
    or a 1-d array.  Returns (point, lo, hi) arrays."""
    rng = np.random.default_rng(seed)
    point = np.atleast_1d(np.asarray(fn(np.arange(n)), dtype=float))
    draws = np.empty((n_boot, point.size))
    for b in range(n_boot):
        draws[b] = np.atleast_1d(np.asarray(fn(rng.integers(0, n, n)), dtype=float))
    lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    if point.size == 1:
        return float(point[0]), float(lo[0]), float(hi[0])
    return point, lo, hi
