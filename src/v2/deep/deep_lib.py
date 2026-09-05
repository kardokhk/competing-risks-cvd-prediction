"""P2-C. Shared code for the two deep competing-risk families, DeepHit and Neural Fine-Gray.

Everything here is deterministic given a seed. Nothing here reads the temporal test
rows except `load_split("temporal")`, which is used only by the production prediction
step and by the explicitly labelled confirmatory sweep; tuning and selection use
`inner_folds`, which partitions the *training* cycles only.

Interpreter: ${ENV_PREFIX}/crcvd-dl/bin/python-cr (the -cr wrapper puts the
environment's libstdc++ ahead of the system one, which is required whenever torch is
imported alongside a conda-forge C++ extension such as scikit-survival).
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

# Keep every worker single-threaded; the outer parallelism is one process per fit.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[3]
SRC_V2 = ROOT / "src" / "v2"
IMPDIR = ROOT / "data" / "processed" / "imputed"
VENDOR = ROOT / "src" / "models" / "vendor" / "NeuralFineGray-main"

sys.path.insert(0, str(SRC_V2))
import eval_lib_v2 as ev  # noqa: E402

PROJECT_SEED = 20260903
HORIZONS = (5.0, 10.0, 15.0)
PRIMARY_H = 10.0
N_CAUSES = 2

with open(SRC_V2 / "feature_spec.json") as fh:
    FEATURE_SPEC = json.load(fh)
FEATS = list(FEATURE_SPEC["primary_features"])
CONT = set(FEATURE_SPEC["standardize_features"])
CONT_IDX = np.array([i for i, f in enumerate(FEATS) if f in CONT], dtype=int)

META = ["SEQN", "cycle", "time", "event"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_IMP_CACHE: dict[int, pd.DataFrame] = {}


def load_imp(k: int) -> pd.DataFrame:
    """One completed imputed dataset, cached in-process."""
    if k not in _IMP_CACHE:
        _IMP_CACHE[k] = pd.read_parquet(IMPDIR / f"imp_{k}.parquet")
    return _IMP_CACHE[k]


def standardize(train_X: np.ndarray, other_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z-score the continuous columns on the *training rows of this fold* only.

    This is the non-negotiable rule in src/v2/PREDICTION_FORMAT.md section 1. Binary
    columns are left alone. Returns copies, float32-castable.
    """
    a = np.array(train_X, dtype=np.float64, copy=True)
    b = np.array(other_X, dtype=np.float64, copy=True)
    mu = a[:, CONT_IDX].mean(axis=0)
    sd = a[:, CONT_IDX].std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    a[:, CONT_IDX] = (a[:, CONT_IDX] - mu) / sd
    b[:, CONT_IDX] = (b[:, CONT_IDX] - mu) / sd
    return a, b


def Xof(df: pd.DataFrame) -> np.ndarray:
    return df[FEATS].to_numpy(dtype=np.float64)


def inner_folds(df_train: pd.DataFrame, n_folds: int = 5,
                seed: int = PROJECT_SEED) -> np.ndarray:
    """Deterministic stratified fold ids for the *training* cycles only.

    A function of the sorted SEQN order and `seed`, so identical for every model,
    every configuration and every imputation. Not derived from cv_rep0..4, which are
    defined on the full cohort and would put temporal-test rows into a tuning fold.
    """
    from sklearn.model_selection import StratifiedKFold
    order = np.argsort(df_train["SEQN"].to_numpy())
    y = df_train["event"].to_numpy()[order]
    fold = np.empty(len(df_train), dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[order[te]] = f
    return fold


def set_seed(seed: int) -> None:
    np.random.seed(seed % (2**31 - 1))
    torch.manual_seed(seed % (2**31 - 1))


def temporal_seed(imp: int) -> int:
    return PROJECT_SEED + 1000 * imp


def cv_seed(imp: int, rep: int, fold: int) -> int:
    return PROJECT_SEED + 1000 * imp + 100 * rep + fold


# --------------------------------------------------------------------------- #
# DeepHit
# --------------------------------------------------------------------------- #
DEEPHIT_DEFAULT = dict(
    # the configuration the prior work used ("tuned lightly"); kept as the reference
    # point for the "does tuning close the gap" question
    num_durations=40, trunk=[64, 64], head=[32], dropout=0.1, lr=0.01,
    batch_size=256, alpha=0.2, sigma=0.1, epochs=200, patience=10,
)

NFG_DEFAULT = dict(
    layers=[32, 32], layers_surv=[32], dropout=0.1, lr=1e-3, batch_size=256,
    weight_decay=1e-3, epochs=200, patience=5,
)


class _MLP(torch.nn.Module):
    def __init__(self, sizes, dropout, out_dim):
        super().__init__()
        mods = []
        prev = sizes[0]
        for h in sizes[1:]:
            mods += [torch.nn.Linear(prev, h), torch.nn.ReLU(),
                     torch.nn.BatchNorm1d(h), torch.nn.Dropout(dropout)]
            prev = h
        mods += [torch.nn.Linear(prev, out_dim)] if out_dim else []
        self.net = torch.nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class CauseSpecificNet(torch.nn.Module):
    """Shared trunk, one head per cause. Output (batch, n_causes, n_bins)."""

    def __init__(self, in_features, trunk, head, dropout, n_causes, n_bins):
        super().__init__()
        self.trunk = _MLP([in_features] + list(trunk), dropout, 0)
        self.heads = torch.nn.ModuleList(
            [_MLP([trunk[-1]] + list(head), dropout, n_bins) for _ in range(n_causes)]
        )

    def forward(self, x):
        z = self.trunk(x)
        return torch.stack([h(z) for h in self.heads], dim=1)


def _safe_batch(n: int, bs: int) -> int:
    """Avoid a trailing batch of size 1, which BatchNorm1d rejects in train mode."""
    while n % bs == 1 and bs > 8:
        bs -= 1
    return bs


def fit_deephit(Xtr, dur, ev_, cfg, seed, val_frac=0.2):
    """Fit DeepHit (pycox competing-risk loss) with an internal early-stopping split.

    Returns (model, cuts, info). `cfg` keys are those of DEEPHIT_DEFAULT.
    """
    import torchtuples as tt
    from pycox.models import DeepHit
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime
    from sklearn.model_selection import train_test_split

    set_seed(seed)
    c = dict(DEEPHIT_DEFAULT); c.update(cfg or {})

    labtrans = LabTransDiscreteTime(int(c["num_durations"]), scheme="quantiles")
    # cuts are placed on any-cause event times; the multi-cause labels are kept
    # separately because pycox's discretiser would collapse them to bool
    labtrans.fit(dur.astype("float32"), (ev_ > 0).astype("float32"))
    cuts = np.asarray(labtrans.cuts, dtype=np.float64)
    idx, _ = labtrans.transform(dur.astype("float32"), (ev_ > 0).astype("float32"))
    idx = idx.astype("int64")
    evi = ev_.astype("int64")

    tri, vai = train_test_split(np.arange(len(Xtr)), test_size=val_frac,
                                random_state=int(seed % (2**31 - 1)), stratify=evi)
    Xf = np.asarray(Xtr, dtype="float32")
    bs = _safe_batch(len(tri), int(c["batch_size"]))

    net = CauseSpecificNet(Xf.shape[1], c["trunk"], c["head"], float(c["dropout"]),
                           N_CAUSES, len(cuts))
    model = DeepHit(net, tt.optim.Adam(float(c["lr"])), alpha=float(c["alpha"]),
                    sigma=float(c["sigma"]), duration_index=cuts)
    cb = [tt.callbacks.EarlyStopping(patience=int(c["patience"]))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = model.fit(Xf[tri], (idx[tri], evi[tri]), batch_size=bs,
                        epochs=int(c["epochs"]), callbacks=cb,
                        val_data=(Xf[vai], (idx[vai], evi[vai])), verbose=False)
    vl = log.to_pandas()["val_loss"].to_numpy()
    info = dict(best_epoch=int(np.argmin(vl)), n_epochs=len(vl),
                val_loss=float(np.min(vl)))
    return model, cuts, info


def predict_deephit(model, cuts, X, horizons=HORIZONS) -> np.ndarray:
    """CVD (cause index 0 == event value 1) CIF, interpolated onto the horizons.

    predict_cif returns (n_causes, n_bins, n_samples). The grid starts at 0 with a
    CIF of 0, so linear interpolation between grid points is well defined; horizons
    beyond the last cut clamp to the last value.
    """
    cif = np.asarray(model.predict_cif(np.asarray(X, dtype="float32"))[0])  # (bins, n)
    g = np.asarray(cuts, dtype=np.float64)
    out = np.empty((cif.shape[1], len(horizons)), dtype=np.float64)
    # linear interpolation of the discrete CIF grid, one bracketing lookup per horizon
    for j, h in enumerate(horizons):
        if h <= g[0]:
            out[:, j] = cif[0]
            continue
        if h >= g[-1]:
            out[:, j] = cif[-1]
            continue
        r = int(np.searchsorted(g, h, side="right"))
        lo, hi = r - 1, r
        w = (h - g[lo]) / (g[hi] - g[lo])
        out[:, j] = cif[lo] * (1 - w) + cif[hi] * w
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Neural Fine-Gray (authors' vendored implementation)
# --------------------------------------------------------------------------- #
_NFG_READY = False


def _ensure_nfg():
    global _NFG_READY
    if _NFG_READY:
        return
    for p in (str(VENDOR), str(VENDOR / "DeepSurvivalMachines")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import nfg.utilities as _u

    class _Quiet:  # the vendored trainer wants a tqdm-like object it can label
        def __init__(self, it, *a, **k):
            self._it = it

        def __iter__(self):
            return iter(self._it)

        def set_description(self, *a, **k):
            pass

    _u.tqdm = _Quiet
    _NFG_READY = True


def fit_nfg(Xtr, dur, ev_, cfg, seed, val_frac=0.2):
    """Fit Neural Fine-Gray using the authors' own code under src/models/vendor/."""
    _ensure_nfg()
    from nfg import NeuralFineGray

    set_seed(seed)
    c = dict(NFG_DEFAULT); c.update(cfg or {})
    model = NeuralFineGray(layers=list(c["layers"]), layers_surv=list(c["layers_surv"]),
                           dropout=float(c["dropout"]), cuda=False, normalise="minmax")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(np.asarray(Xtr, dtype=np.float64),
                  np.asarray(dur, dtype=np.float64),
                  np.asarray(ev_, dtype=np.float64),
                  vsize=val_frac, n_iter=int(c["epochs"]), lr=float(c["lr"]),
                  bs=int(c["batch_size"]), weight_decay=float(c["weight_decay"]),
                  patience_max=int(c["patience"]), random_state=int(seed % (2**31 - 1)))
    return model, None, dict(n_epochs=int(getattr(model, "speed", -1)))


def predict_nfg(model, _cuts, X, horizons=HORIZONS) -> np.ndarray:
    """CVD (cause 1) CIF at the horizons. NFG models continuous time directly."""
    surv = model.predict_survival(np.asarray(X, dtype=np.float64), list(horizons), risk=1)
    return np.clip(1.0 - np.asarray(surv, dtype=np.float64), 0.0, 1.0)


FIT = {"deephit": fit_deephit, "nfg": fit_nfg}
PREDICT = {"deephit": predict_deephit, "nfg": predict_nfg}
DEFAULTS = {"deephit": DEEPHIT_DEFAULT, "nfg": NFG_DEFAULT}
LEARNER = {"deephit": "DeepHit", "nfg": "NeuralFineGray"}


# --------------------------------------------------------------------------- #
# Metrics, identical estimators to src/v2/06_evaluate.py
# --------------------------------------------------------------------------- #
def metrics(time, event, cycle, pred, t=PRIMARY_H) -> dict:
    """E/O, weak and moderate calibration, AUC and C at horizon `t`.

    Uses src/v2/eval_lib_v2.py so that every number here is on the same scale as the
    numbers the main evaluation reports: Aalen-Johansen observed CIF, exact jackknife
    pseudo-observations, cloglog quasi-likelihood calibration, cycle-stratified IPCW
    for the Blanche AUC and the Wolbers concordance.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, np.int64)
    p = np.asarray(pred, float)
    ok = np.isfinite(p)
    if ok.sum() == 0 or not np.any((event == 1) & (time <= t)):
        return {k: np.nan for k in ("mean_pred", "obs_cif", "eo", "calib_intercept",
                                    "calib_slope", "ici", "e50", "e90", "auc",
                                    "cindex", "at_risk_fraction")}
    obs = ev.aj_cif_at(time, event, t, 1)
    ps = ev.pseudo_cif(time, event, t, 1)
    G_T = ev.censor_surv_stratified(time, event, np.asarray(cycle), time)
    G_t = ev.censor_surv_stratified(time, event, np.asarray(cycle),
                                    np.full(len(time), t))
    pp = np.clip(p[ok], 1e-8, 1 - 1e-8)
    tt, ee = time[ok], event[ok]
    icp, slp = ev.calib_intercept_slope(ps[ok], pp)
    fc = ev.flexible_calibration(ps[ok], pp, nknots=3)
    return dict(
        mean_pred=float(np.mean(pp)), obs_cif=float(obs),
        eo=float(np.mean(pp)) / obs if obs > 0 else np.nan,
        calib_intercept=icp, calib_slope=slp,
        ici=fc["ici"], e50=fc["e50"], e90=fc["e90"],
        auc=float(ev.ipcw_auc(tt, ee, pp, t, G_T[ok], G_t[ok])),
        cindex=float(ev.cr_cindex(tt, ee, pp, t, G_T[ok])),
        at_risk_fraction=float(ev.at_risk_fraction(time, event, t)),
    )


# --------------------------------------------------------------------------- #
# Fit-and-predict wrappers used by every driver
# --------------------------------------------------------------------------- #
def fit_predict(model_key, tr_df, te_df, cfg, seed, horizons=HORIZONS):
    """Standardise on tr_df, fit on tr_df, return the CIF matrix for te_df."""
    Xtr_raw, Xte_raw = Xof(tr_df), Xof(te_df)
    Xtr, Xte = standardize(Xtr_raw, Xte_raw)
    dur = tr_df["time"].to_numpy(dtype=np.float64)
    evn = tr_df["event"].to_numpy(dtype=np.int64)
    model, cuts, info = FIT[model_key](Xtr, dur, evn, cfg, seed)
    risk = PREDICT[model_key](model, cuts, Xte, horizons)
    return risk, info


def oof_train(model_key, df_train, folds, cfg, seed, n_folds=5, horizons=HORIZONS):
    """Out-of-fold CIF over the training cycles. Never touches the temporal test set.

    Returns a DataFrame with SEQN, cycle, time, event and one column per horizon,
    covering every training row exactly once.
    """
    parts = []
    for f in range(n_folds):
        tr = df_train.loc[folds != f]
        te = df_train.loc[folds == f]
        r, _ = fit_predict(model_key, tr, te, cfg, seed + 7 * f, horizons)
        d = te[META].copy()
        for j, h in enumerate(horizons):
            d[f"risk_{int(h)}"] = r[:, j]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def append_csv(path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV so that a killed job leaves every completed row on disk.

    Rows do not all carry the same keys: a fit that raised adds an `error` column that
    a successful fit does not have. Appending such a row blind would shift every field
    one column to the right, so the existing header is read first, and the file is
    rewritten whenever a genuinely new column appears.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not path.exists():
        df.to_csv(path, index=False)
        return
    header = list(pd.read_csv(path, nrows=0).columns)
    if set(df.columns) <= set(header):
        df.reindex(columns=header).to_csv(path, mode="a", header=False, index=False)
    else:
        old = pd.read_csv(path)
        pd.concat([old, df], ignore_index=True).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Model selection score
# --------------------------------------------------------------------------- #
def ipcw_brier(time, event, pred, t=PRIMARY_H, cause=1, G_at_T=None, G_at_t=None):
    """IPCW Brier score for the cause-specific cumulative incidence at t.

    Graf-type weights extended to competing risks (Schoop, Beyersmann, Schumacher):
      T_i <= t with event == cause      -> outcome 1, weight 1 / G(T_i-)
      T_i <= t with a competing event   -> outcome 0, weight 1 / G(T_i-)
      T_i >  t                          -> outcome 0, weight 1 / G(t)
      censored at T_i <= t              -> weight 0
    Lower is better. It is a proper scoring rule for absolute risk, so it rewards
    discrimination and calibration together, which is why it and not the C-index or
    the ICI is the primary tuning criterion: selecting on either of the two reported
    quantities would build the conclusion into the selection.
    """
    time = np.asarray(time, float); event = np.asarray(event, np.int64)
    p = np.clip(np.asarray(pred, float), 0.0, 1.0)
    if G_at_T is None:
        G_at_T = ev.censor_surv(time, event, time)
    if G_at_t is None:
        G_at_t = ev.censor_surv(time, event, np.full(len(time), t))
    G_at_T = np.maximum(np.asarray(G_at_T, float), 1e-8)
    G_at_t = np.maximum(np.asarray(G_at_t, float), 1e-8)

    w = np.zeros(len(time)); y = np.zeros(len(time))
    ev_before = time <= t
    case = ev_before & (event == cause)
    comp = ev_before & (event != cause) & (event != 0)
    after = time > t
    w[case] = 1.0 / G_at_T[case]; y[case] = 1.0
    w[comp] = 1.0 / G_at_T[comp]
    w[after] = 1.0 / G_at_t[after]
    ok = np.isfinite(p) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(w[ok] * (y[ok] - p[ok]) ** 2) / np.sum(w[ok]))


# --------------------------------------------------------------------------- #
# Parallel task runner with resume
# --------------------------------------------------------------------------- #
def run_tasks(fn, tasks, out_csv, key_cols, n_proc=None, chunk_flush=1):
    """Map `fn` over `tasks`, appending each returned row to `out_csv` as it lands.

    Rows already present in `out_csv` under the same `key_cols` are skipped, so a
    resubmitted job resumes rather than restarts. `fn` must return a dict or None.
    """
    import multiprocessing as mp

    out_csv = Path(out_csv)
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        if all(c in prev.columns for c in key_cols):
            done = set(map(tuple, prev[list(key_cols)].astype(str).to_numpy()))
    todo = [t for t in tasks
            if tuple(str(t[c]) for c in key_cols) not in done]
    print(f"[run_tasks] {out_csv.name}: {len(tasks)} tasks, {len(tasks)-len(todo)} "
          f"already on disk, {len(todo)} to run", flush=True)
    if not todo:
        return
    n_proc = n_proc or int(os.environ.get("P2C_NPROC", "128"))
    n_proc = max(1, min(n_proc, len(todo)))
    buf = []
    with mp.get_context("spawn").Pool(n_proc, maxtasksperchild=8) as pool:
        for i, row in enumerate(pool.imap_unordered(fn, todo), 1):
            if row is None:
                continue
            buf.append(row)
            if len(buf) >= chunk_flush:
                append_csv(out_csv, buf); buf = []
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} done", flush=True)
    append_csv(out_csv, buf)
    print(f"[run_tasks] {out_csv.name}: finished", flush=True)
