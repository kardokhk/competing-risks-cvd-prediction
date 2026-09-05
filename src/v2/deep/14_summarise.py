#!/usr/bin/env python
"""P2-C stage S. Turn the raw CSVs into the tables and the two markdown deliverables.

Reads results/deep/{alpha_sweep,tuning_raw,seed_stability}.csv and
tuning_selected.json; writes alpha_sweep_summary.csv, seed_stability_summary.csv,
tuning.md and SUMMARY.md. Every number in the markdown is computed here, so it is
traceable to this script and to the CSV row it came from.

Across replicates the point estimate is the median and the interval the 10th to 90th
percentile, because the seed-to-seed distributions are skewed. Differences between two
alpha settings carry a percentile bootstrap interval over replicates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deep_lib as D  # noqa: E402

OUT = D.ROOT / "results" / "deep"
METRICS = ["eo", "ici", "calib_slope", "cindex", "auc", "brier"]
BOOT_SEED = 20260903
N_BOOT = 4000


def med_ip(v) -> tuple[float, float, float]:
    v = np.asarray(pd.Series(v).dropna(), float)
    if not len(v):
        return (np.nan, np.nan, np.nan)
    return (float(np.median(v)), float(np.quantile(v, 0.10)),
            float(np.quantile(v, 0.90)))


def fmt(v, d=3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def cell(v, d=3) -> str:
    m, lo, hi = v
    return f"{fmt(m, d)} ({fmt(lo, d)} to {fmt(hi, d)})"


def boot_diff(a, b, rng) -> tuple[float, float, float]:
    """Median of a minus median of b, with a percentile bootstrap interval."""
    a = np.asarray(pd.Series(a).dropna(), float)
    b = np.asarray(pd.Series(b).dropna(), float)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan, np.nan)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        d[i] = (np.median(rng.choice(a, len(a), replace=True))
                - np.median(rng.choice(b, len(b), replace=True)))
    return (float(np.median(a) - np.median(b)),
            float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))


def drop_errors(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.error.isna()] if "error" in df.columns else df


# --------------------------------------------------------------------------- #
def summarise_sweep() -> tuple[pd.DataFrame, dict]:
    f = OUT / "alpha_sweep.csv"
    if not f.exists():
        return pd.DataFrame(), {}
    df = drop_errors(pd.read_csv(f))
    rows = []
    for (model, protocol, a, s), q in df.groupby(
            ["model", "protocol", "alpha_s", "sigma_s"], dropna=False):
        r = dict(model=model, protocol=protocol, alpha=a, sigma=s, n_rep=len(q))
        for m in METRICS:
            if m in q:
                r[f"{m}_median"], r[f"{m}_p10"], r[f"{m}_p90"] = med_ip(q[m])
        rows.append(r)
    summ = pd.DataFrame(rows).sort_values(["model", "protocol", "sigma", "alpha"])
    summ.to_csv(OUT / "alpha_sweep_summary.csv", index=False)

    rng = np.random.default_rng(BOOT_SEED)
    facts = {}
    for protocol in ("inner_oof", "temporal_test"):
        q = df[(df.model == "deephit") & (df.protocol == protocol)
               & (df.sigma_s.astype(str) == "0.1")]
        if not len(q):
            continue
        a_num = pd.to_numeric(q.alpha_s, errors="coerce")
        sub = q.assign(a=a_num).dropna(subset=["a"])
        d = {}
        for m in ("ici", "eo", "calib_slope", "cindex", "auc", "brier"):
            if m not in sub:
                continue
            v = sub[[m, "a"]].dropna()
            if len(v) > 3:
                # Spearman rank correlation of the metric against alpha
                ra = pd.Series(v["a"]).rank().to_numpy()
                rm = pd.Series(v[m]).rank().to_numpy()
                d[f"spearman_{m}_vs_alpha"] = float(np.corrcoef(ra, rm)[0, 1])
            lo_v = sub.loc[np.isclose(sub.a, 0.2), m]
            hi_v = sub.loc[np.isclose(sub.a, 1.0), m]
            d[f"diff_a1_minus_a02_{m}"] = boot_diff(hi_v, lo_v, rng)
        nfg = df[(df.model == "nfg") & (df.protocol == protocol)]
        for m in METRICS:
            if m in nfg:
                d[f"nfg_{m}"] = med_ip(nfg[m])
        facts[protocol] = d
    return summ, facts


def summarise_seeds() -> tuple[pd.DataFrame, dict]:
    f = OUT / "seed_stability.csv"
    if not f.exists():
        return pd.DataFrame(), {}
    df = drop_errors(pd.read_csv(f))
    rows = []
    for k, q in df.groupby(["model", "config", "protocol"]):
        r = dict(zip(["model", "config", "protocol"], k), n_seeds=len(q))
        for m in METRICS:
            if m in q:
                r[f"{m}_median"], r[f"{m}_p10"], r[f"{m}_p90"] = med_ip(q[m])
                v = pd.Series(q[m]).dropna()
                r[f"{m}_min"] = float(v.min()) if len(v) else np.nan
                r[f"{m}_max"] = float(v.max()) if len(v) else np.nan
        rows.append(r)
    summ = pd.DataFrame(rows).sort_values(["protocol", "model", "config"])
    summ.to_csv(OUT / "seed_stability_summary.csv", index=False)

    rng = np.random.default_rng(BOOT_SEED + 1)
    facts = {}
    for protocol in df.protocol.unique():
        p = df[df.protocol == protocol]
        g = lambda mo, co, me: p[(p.model == mo) & (p.config == co)][me]  # noqa: E731
        d = {}
        for me in ("ici", "eo", "calib_slope", "cindex"):
            if me not in p:
                continue
            d[f"gap_default_{me}"] = boot_diff(g("deephit", "default", me),
                                               g("nfg", "tuned_brier", me), rng)
            d[f"gap_tuned_{me}"] = boot_diff(g("deephit", "tuned_brier", me),
                                             g("nfg", "tuned_brier", me), rng)
            d[f"deephit_tuning_effect_{me}"] = boot_diff(g("deephit", "tuned_brier", me),
                                                         g("deephit", "default", me), rng)
        facts[protocol] = d
    return summ, facts


def table(summ: pd.DataFrame, sel: dict, protocol: str) -> str:
    q = summ[(summ.protocol == protocol) & (summ.sigma.astype(str) == "0.1")
             & (summ.model == "deephit")].copy()
    q["a"] = pd.to_numeric(q.alpha, errors="coerce")
    q = q.dropna(subset=["a"]).sort_values("a")
    out = ["| alpha | E/O | ICI | slope | C-index | AUC | IPCW Brier | n |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in q.iterrows():
        out.append("| {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.a,
            cell((r.eo_median, r.eo_p10, r.eo_p90), 2),
            cell((r.ici_median, r.ici_p10, r.ici_p90), 4),
            cell((r.calib_slope_median, r.calib_slope_p10, r.calib_slope_p90), 2),
            cell((r.cindex_median, r.cindex_p10, r.cindex_p90), 3),
            cell((r.auc_median, r.auc_p10, r.auc_p90), 3),
            cell((r.brier_median, r.brier_p10, r.brier_p90), 4),
            int(r.n_rep)))
    n = summ[(summ.protocol == protocol) & (summ.model == "nfg")]
    if len(n):
        r = n.iloc[0]
        out.append("| Neural Fine-Gray | {} | {} | {} | {} | {} | {} | {} |".format(
            cell((r.eo_median, r.eo_p10, r.eo_p90), 2),
            cell((r.ici_median, r.ici_p10, r.ici_p90), 4),
            cell((r.calib_slope_median, r.calib_slope_p10, r.calib_slope_p90), 2),
            cell((r.cindex_median, r.cindex_p10, r.cindex_p90), 3),
            cell((r.auc_median, r.auc_p10, r.auc_p90), 3),
            cell((r.brier_median, r.brier_p10, r.brier_p90), 4),
            int(r.n_rep)))
    return "\n".join(out)


def seed_table(summ: pd.DataFrame, protocol: str) -> str:
    q = summ[summ.protocol == protocol]
    out = ["| model | configuration | E/O | ICI | slope | C-index | seeds |",
           "|---|---|---|---|---|---|---|"]
    for _, r in q.iterrows():
        out.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r.model, r.config,
            cell((r.eo_median, r.eo_p10, r.eo_p90), 2),
            cell((r.ici_median, r.ici_p10, r.ici_p90), 4),
            cell((r.calib_slope_median, r.calib_slope_p10, r.calib_slope_p90), 2),
            cell((r.cindex_median, r.cindex_p10, r.cindex_p90), 3),
            int(r.n_seeds)))
    return "\n".join(out)


def diff_line(d, key, label, dec=4) -> str:
    v = d.get(key)
    if v is None or not np.isfinite(v[0]):
        return f"- {label}: not estimable"
    return f"- {label}: {fmt(v[0], dec)} (95% CI {fmt(v[1], dec)} to {fmt(v[2], dec)})"


def main() -> None:
    sweep, sweep_facts = summarise_sweep()
    seeds, seed_facts = summarise_seeds()
    sel = json.loads((OUT / "tuning_selected.json").read_text()) \
        if (OUT / "tuning_selected.json").exists() else {}
    space = json.loads((OUT / "tuning_space.json").read_text()) \
        if (OUT / "tuning_space.json").exists() else {}
    raw = (drop_errors(pd.read_csv(OUT / "tuning_raw.csv"))
           if (OUT / "tuning_raw.csv").exists() else pd.DataFrame())

    # ---------------- tuning.md ----------------
    t = ["# Hyperparameter search for DeepHit and Neural Fine-Gray",
         "",
         "Written by `src/v2/deep/14_summarise.py` from `results/deep/tuning_raw.csv` "
         "and `results/deep/tuning_selected.json`. Search executed by "
         "`src/v2/deep/11_tune.py`.",
         "",
         "## How no test data touched selection",
         "",
         "Three separate guarantees, each checkable in the code.",
         "",
         "1. Every fit scored in the search is trained on a subset of the rows with "
         "`split == \"train\"`, that is NHANES cycles 1999-2000 to 2005-2006, "
         "N = 14,551. `11_tune.py` never subsets on `split == \"test\"`; the only "
         "dataframe it builds is `d[d.split == \"train\"]`.",
         "2. The five scoring folds come from `deep_lib.inner_folds`, a stratified "
         "partition of those training rows keyed on sorted SEQN and the project seed "
         "20260903. It is deliberately not `cv_rep0..4`, which are defined on the full "
         "cohort of 41,151 and would place temporal-test rows inside a scoring fold.",
         "3. Standardisation centres and scales are recomputed inside every inner fold "
         "from that fold's training rows only (`deep_lib.standardize`), so no summary "
         "statistic crosses a fold boundary either.",
         "",
         "The temporal test rows are read in exactly two places: `13_predict.py`, which "
         "produces the final predictions after selection is closed, and the "
         "confirmatory arm of `10_alpha_sweep.py` and `12_seeds.py`, which are labelled "
         "`temporal_test` and are read-outs, not selections.",
         "",
         "## Selection criterion",
         "",
         "The IPCW Brier score for the cause-1 cumulative incidence at 10 y, pooled "
         "over the five inner folds and averaged over three initialisation seeds. It is "
         "a proper scoring rule, so it rewards discrimination and calibration together. "
         "Selecting on the C-index would pick a ranking-dominated DeepHit by "
         "construction and selecting on ICI or E/O would write the paper's conclusion "
         "into the selection, so both are recorded as secondary selections instead and "
         "the C-index-selected DeepHit is refitted in `12_seeds.py` as a labelled "
         "sensitivity analysis.",
         ""]
    if space:
        t += ["## Search space", "",
              "Random search, `numpy.random.default_rng(20260903)`, the prior work's "
              "configuration inserted as candidate 000 so the default is always scored.",
              "", "```json", json.dumps(space.get("space", {}), indent=1), "```", ""]
    if len(raw):
        t += [f"Configurations scored: "
              + ", ".join(f"{m} {raw[raw.model == m].cfg_id.nunique()}"
                          for m in sorted(raw.model.unique()))
              + f"; seeds per configuration {sorted(raw.seed.unique())}; "
              f"total fits {len(raw) * 5} (each row is five inner folds).", ""]
    if sel:
        t += ["## Selected configurations", "", "```json",
              json.dumps({m: {k: v["cfg"] for k, v in s["selected"].items()}
                          for m, s in sel.items()}, indent=1), "```", "",
              "Inner out-of-fold scores of each selection and of the default:", "",
              "| model | selection | cfg_id | Brier | ICI | E/O | slope | C-index |",
              "|---|---|---|---|---|---|---|---|"]
        for m, s in sel.items():
            items = list(s["selected"].items())
            if s.get("default_cfg", {}).get("inner_oof"):
                items.append(("default", s["default_cfg"]))
            for k, v in items:
                o = v.get("inner_oof") or {}
                t.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    m, k, v["cfg_id"], fmt(o.get("brier"), 4), fmt(o.get("ici"), 4),
                    fmt(o.get("eo"), 2), fmt(o.get("slope"), 2), fmt(o.get("cindex"), 3)))
        t.append("")
    t += ["## What the search does not cover", "",
          "- The number of training epochs is capped at 200 for both models with early "
          "stopping on an internal validation split, and is not itself searched. "
          "DeepHit reaches its early-stopping criterion well inside the cap; Neural "
          "Fine-Gray usually runs to it, so Neural Fine-Gray receives the larger "
          "gradient budget. That asymmetry favours Neural Fine-Gray and is stated "
          "rather than corrected, because raising DeepHit's cap does not change a "
          "model that has already early-stopped.",
          "- The search is run on imputation 1 only. The selected configuration is then "
          "refitted on all 30 imputations. Between-imputation variation in which "
          "configuration was chosen is therefore not quantified.",
          "- Three seeds per configuration is enough to rank configurations but not to "
          "characterise a configuration's seed distribution; `12_seeds.py` uses 12 "
          "seeds for that.", ""]
    (OUT / "tuning.md").write_text("\n".join(t) + "\n")

    # ---------------- SUMMARY.md ----------------
    s = ["# P2-C: deep competing-risk models, DeepHit and Neural Fine-Gray",
         "",
         "Produced by `src/v2/deep/`, summarised by `src/v2/deep/14_summarise.py`. "
         "Environment `${ENV_PREFIX}/crcvd-dl` invoked through the "
         "`python-cr` wrapper. Project seed 20260903.",
         "",
         "Point estimates are medians over replicates with the 10th to 90th percentile "
         "in brackets; differences carry a 2.5th-to-97.5th percentile bootstrap "
         "interval over replicates (4,000 resamples, seed 20260903). Horizon 10 y "
         "throughout. Estimators are those of `src/v2/eval_lib_v2.py`, so every number "
         "here is on the same scale as the main benchmark.",
         ""]
    # The verdict is written by hand once the numbers exist and is spliced in here so
    # that rerunning this script never silently drops it.
    vf = OUT / "verdict.md"
    if vf.exists():
        s += [vf.read_text().strip(), ""]
    else:
        s += ["## 0. Verdict", "",
              "_pending: results/deep/verdict.md has not been written yet_", ""]
    s += ["## 1. The alpha sweep: does the training objective drive the miscalibration?",
         "",
         "pycox writes the DeepHit objective as "
         "`loss = alpha * nll_pmf_cr + (1 - alpha) * rank_loss_deephit_cr(sigma)`, so "
         "alpha = 0 is a pure ranking loss and alpha = 1 a pure likelihood. The prior "
         "work used alpha = 0.2, putting 80% of the objective on the ranking term.",
         "",
         "### Primary: inner out-of-fold, training cycles only, sigma = 0.1",
         ""]
    if len(sweep):
        s += [table(sweep, sel, "inner_oof"), ""]
        f = sweep_facts.get("inner_oof", {})
        s += ["Trend across alpha (Spearman rank correlation over all replicates):", "",
              f"- ICI against alpha: {fmt(f.get('spearman_ici_vs_alpha'), 3)}",
              f"- E/O against alpha: {fmt(f.get('spearman_eo_vs_alpha'), 3)}",
              f"- C-index against alpha: {fmt(f.get('spearman_cindex_vs_alpha'), 3)}",
              f"- IPCW Brier against alpha: {fmt(f.get('spearman_brier_vs_alpha'), 3)}",
              "",
              "Pure likelihood (alpha = 1) minus the prior work's setting (alpha = 0.2):",
              "",
              diff_line(f, "diff_a1_minus_a02_ici", "ICI", 4),
              diff_line(f, "diff_a1_minus_a02_eo", "E/O", 3),
              diff_line(f, "diff_a1_minus_a02_calib_slope", "calibration slope", 3),
              diff_line(f, "diff_a1_minus_a02_cindex", "C-index", 4),
              diff_line(f, "diff_a1_minus_a02_auc", "AUC", 4),
              ""]
        s += ["### Confirmatory: temporal test set, no configuration selected on it", "",
              table(sweep, sel, "temporal_test"), ""]
        f2 = sweep_facts.get("temporal_test", {})
        s += [diff_line(f2, "diff_a1_minus_a02_ici", "ICI, alpha 1 minus alpha 0.2", 4),
              diff_line(f2, "diff_a1_minus_a02_eo", "E/O, alpha 1 minus alpha 0.2", 3),
              diff_line(f2, "diff_a1_minus_a02_cindex",
                        "C-index, alpha 1 minus alpha 0.2", 4), ""]
    else:
        s += ["_alpha_sweep.csv not present_", ""]

    s += ["## 2. Does a tuned DeepHit close the gap to Neural Fine-Gray?", ""]
    if len(seeds):
        for protocol, title in (("inner_oof", "Inner out-of-fold, training cycles only"),
                                ("temporal_test", "Temporal test set, confirmatory")):
            if protocol in set(seeds.protocol):
                s += [f"### {title}", "", seed_table(seeds, protocol), ""]
                f = seed_facts.get(protocol, {})
                s += ["DeepHit minus Neural Fine-Gray, both at their selected "
                      "configurations:", "",
                      diff_line(f, "gap_tuned_ici", "ICI", 4),
                      diff_line(f, "gap_tuned_eo", "E/O", 3),
                      diff_line(f, "gap_tuned_cindex", "C-index", 4), "",
                      "Default DeepHit minus Neural Fine-Gray, the prior work's "
                      "comparison:", "",
                      diff_line(f, "gap_default_ici", "ICI", 4),
                      diff_line(f, "gap_default_eo", "E/O", 3), "",
                      "Effect of tuning DeepHit (tuned minus default):", "",
                      diff_line(f, "deephit_tuning_effect_ici", "ICI", 4),
                      diff_line(f, "deephit_tuning_effect_eo", "E/O", 3),
                      diff_line(f, "deephit_tuning_effect_cindex", "C-index", 4), ""]
    else:
        s += ["_seed_stability.csv not present_", ""]

    s += ["## 3. Seed stability", "",
          "Twelve initialisation seeds per model per configuration. The tables in "
          "section 2 carry the 10th to 90th percentile across those seeds; the full "
          "per-seed rows are in `results/deep/seed_stability.csv` and the min and max "
          "in `results/deep/seed_stability_summary.csv`.", ""]

    s += ["## 4. Files", "",
          "| file | what it holds |",
          "|---|---|",
          "| `alpha_sweep.csv` | one row per (alpha, sigma, seed, imputation, protocol) |",
          "| `alpha_sweep_summary.csv` | medians and 10th-90th percentiles of the above |",
          "| `tuning_raw.csv` | one row per (configuration, seed), inner out-of-fold |",
          "| `tuning_space.json` | the search space and every sampled configuration |",
          "| `tuning_selected.json` | the selected configuration under each criterion |",
          "| `tuning.md` | search protocol and the no-test-data guarantee |",
          "| `seed_stability.csv` | one row per (model, configuration, protocol, seed) |",
          "| `predict_log.csv` | one row per production fit: seed, seconds, epochs |",
          "| `validate_report.txt` | `src/v2/check_predictions.py` on the 120 files |",
          "",
          "Scripts, in dependency order: `11_tune.py`, `10_alpha_sweep.py`, "
          "`12_seeds.py`, `13_predict.py`, `14_summarise.py`, driven by `run.sh` and "
          "`slurm_p2c.sh`.", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(s) + "\n")
    print("wrote tuning.md and SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
