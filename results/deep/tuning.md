# Hyperparameter search for DeepHit and Neural Fine-Gray

Written by `src/v2/deep/14_summarise.py` from `results/deep/tuning_raw.csv` and `results/deep/tuning_selected.json`. Search executed by `src/v2/deep/11_tune.py`.

## How no test data touched selection

Three separate guarantees, each checkable in the code.

1. Every fit scored in the search is trained on a subset of the rows with `split == "train"`, that is NHANES cycles 1999-2000 to 2005-2006, N = 14,551. `11_tune.py` never subsets on `split == "test"`; the only dataframe it builds is `d[d.split == "train"]`.
2. The five scoring folds come from `deep_lib.inner_folds`, a stratified partition of those training rows keyed on sorted SEQN and the project seed 20260903. It is deliberately not `cv_rep0..4`, which are defined on the full cohort of 41,151 and would place temporal-test rows inside a scoring fold.
3. Standardisation centres and scales are recomputed inside every inner fold from that fold's training rows only (`deep_lib.standardize`), so no summary statistic crosses a fold boundary either.

The temporal test rows are read in exactly two places: `13_predict.py`, which produces the final predictions after selection is closed, and the confirmatory arm of `10_alpha_sweep.py` and `12_seeds.py`, which are labelled `temporal_test` and are read-outs, not selections.

## Selection criterion

The IPCW Brier score for the cause-1 cumulative incidence at 10 y, pooled over the five inner folds and averaged over three initialisation seeds. It is a proper scoring rule, so it rewards discrimination and calibration together. Selecting on the C-index would pick a ranking-dominated DeepHit by construction and selecting on ICI or E/O would write the paper's conclusion into the selection, so both are recorded as secondary selections instead and the C-index-selected DeepHit is refitted in `12_seeds.py` as a labelled sensitivity analysis.

## Search space

Random search, `numpy.random.default_rng(20260903)`, the prior work's configuration inserted as candidate 000 so the default is always scored.

```json
{
 "deephit": {
  "trunk": [
   [
    32
   ],
   [
    64
   ],
   [
    64,
    64
   ],
   [
    128,
    128
   ],
   [
    64,
    64,
    64
   ],
   [
    128,
    64
   ]
  ],
  "head": [
   [
    32
   ],
   [
    64
   ],
   [
    32,
    32
   ]
  ],
  "dropout": [
   0.0,
   0.1,
   0.2,
   0.4
  ],
  "lr": [
   0.001,
   0.003,
   0.01
  ],
  "batch_size": [
   128,
   256,
   512
  ],
  "alpha": [
   0.0,
   0.1,
   0.2,
   0.3,
   0.5,
   0.7,
   0.9,
   1.0
  ],
  "sigma": [
   0.05,
   0.1,
   0.25,
   0.5
  ],
  "num_durations": [
   20,
   40,
   80
  ]
 },
 "nfg": {
  "layers": [
   [
    32
   ],
   [
    32,
    32
   ],
   [
    64,
    64
   ],
   [
    100,
    100
   ],
   [
    50,
    50,
    50
   ]
  ],
  "layers_surv": [
   [
    32
   ],
   [
    64
   ],
   [
    100
   ],
   [
    50,
    50
   ]
  ],
  "dropout": [
   0.0,
   0.1,
   0.25
  ],
  "lr": [
   0.0001,
   0.0005,
   0.001,
   0.005
  ],
  "batch_size": [
   128,
   256,
   512
  ],
  "weight_decay": [
   0.0001,
   0.001,
   0.01
  ]
 }
}
```

Configurations scored: deephit 80, nfg 60; seeds per configuration [np.int64(0), np.int64(1), np.int64(2)]; total fits 2100 (each row is five inner folds).

## Selected configurations

```json
{
 "deephit": {
  "brier": {
   "alpha": 0.9,
   "batch_size": 512,
   "dropout": 0.2,
   "head": [
    32,
    32
   ],
   "lr": 0.01,
   "num_durations": 20,
   "sigma": 0.1,
   "trunk": [
    128,
    128
   ]
  },
  "cindex": {
   "alpha": 0.2,
   "batch_size": 512,
   "dropout": 0.4,
   "head": [
    64
   ],
   "lr": 0.003,
   "num_durations": 40,
   "sigma": 0.25,
   "trunk": [
    128,
    128
   ]
  },
  "ici": {
   "alpha": 0.9,
   "batch_size": 512,
   "dropout": 0.2,
   "head": [
    32,
    32
   ],
   "lr": 0.01,
   "num_durations": 20,
   "sigma": 0.5,
   "trunk": [
    64,
    64
   ]
  }
 },
 "nfg": {
  "brier": {
   "batch_size": 128,
   "dropout": 0.0,
   "layers": [
    100,
    100
   ],
   "layers_surv": [
    50,
    50
   ],
   "lr": 0.005,
   "weight_decay": 0.001
  },
  "cindex": {
   "batch_size": 256,
   "dropout": 0.1,
   "layers": [
    32,
    32
   ],
   "layers_surv": [
    64
   ],
   "lr": 0.005,
   "weight_decay": 0.001
  },
  "ici": {
   "batch_size": 128,
   "dropout": 0.25,
   "layers": [
    64,
    64
   ],
   "layers_surv": [
    32
   ],
   "lr": 0.0001,
   "weight_decay": 0.0001
  }
 }
}
```

Inner out-of-fold scores of each selection and of the default:

| model | selection | cfg_id | Brier | ICI | E/O | slope | C-index |
|---|---|---|---|---|---|---|---|
| deephit | brier | deephit_070 | 0.0364 | 0.0026 | 1.06 | 1.04 | 0.834 |
| deephit | cindex | deephit_003 | 0.0376 | 0.0227 | 1.51 | 1.14 | 0.837 |
| deephit | ici | deephit_047 | 0.0364 | 0.0016 | 1.02 | 1.04 | 0.834 |
| deephit | default | deephit_000 | 0.0372 | 0.0214 | 1.47 | 1.70 | 0.832 |
| nfg | brier | nfg_048 | 0.0363 | 0.0024 | 0.96 | 1.03 | 0.834 |
| nfg | cindex | nfg_006 | 0.0364 | 0.0031 | 1.01 | 0.99 | 0.835 |
| nfg | ici | nfg_052 | 0.0363 | 0.0008 | 1.00 | 0.98 | 0.832 |
| nfg | default | nfg_000 | 0.0363 | 0.0030 | 0.95 | 0.92 | 0.833 |

## What the search does not cover

- The number of training epochs is capped at 200 for both models with early stopping on an internal validation split, and is not itself searched. DeepHit reaches its early-stopping criterion well inside the cap; Neural Fine-Gray usually runs to it, so Neural Fine-Gray receives the larger gradient budget. That asymmetry favours Neural Fine-Gray and is stated rather than corrected, because raising DeepHit's cap does not change a model that has already early-stopped.
- The search is run on imputation 1 only. The selected configuration is then refitted on all 30 imputations. Between-imputation variation in which configuration was chosen is therefore not quantified.
- Three seeds per configuration is enough to rank configurations but not to characterise a configuration's seed distribution; `12_seeds.py` uses 12 seeds for that.

