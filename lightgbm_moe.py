#!/usr/bin/env python3

"""
LightGBM mixture-of-experts with orthogonal partition tiers:
    global         — one expert on all rows
    macro          — one expert per MACRO_REGION
    region         — one expert per REGION
    district_type  — one expert per DISTRICT_TYPE
    county         — one expert per COUNTY
    assessment     — one expert per ASSESSMENT_NAME
    subgroup       — one expert per SUBGROUP_NAME

Stratification: each tier picks a stratification key that is NOT its own
partition key, since within a partition the partition key is constant. Rare
classes (< n_folds rows) are pooled into '__rare__' to keep StratifiedKFold
from blowing up on small partitions.

Caching: per-fold val/test predictions are cached to disk keyed by
(tier, name, fold, params_hash). Re-running stack with unchanged params is
near-instant; only experts whose params changed get refit.

Cache schema v2 (self-validating):
    val_preds, pred_preds          — per-fold predictions in original units
    val_idx, pred_idx              — row indices into train/pred (no kfold replay needed)
    params_hash, params_json       — for verification at load time
    val_rmse, best_iter            — diagnostics
    cache_version=2                — schema marker

Modes:
    --mode tune --tier {global,macro,region,district_type,county,assessment,subgroup} [--name NAME]
    --mode stack [--force]
    --mode submit-global [--force]
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
import polars as pl
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 8675309
np.random.seed(SEED)
random.seed(SEED)
pl.enable_string_cache()

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
PARAMS_DIR = Path("best_params_lightgbm")
PARAMS_DIR.mkdir(exist_ok=True)
EXPERTS_DIR = Path("fitted_experts_lightgbm")
EXPERTS_DIR.mkdir(exist_ok=True)
ARTIFACTS_PATH = Path("lightgbm_moe_artifacts.npz")
GLOBAL_SUBMISSION_PATH = Path("submission_lightgbm_global.csv")

CACHE_VERSION = 2


# Warm-start configurations
# Values here are LightGBM's actual library defaults. bagging_freq is kept at
# 1 (rather than the true default of 0) because the search space lower-bounds
# it at 1; with bagging_fraction=1.0 this is functionally identical to off.

LGB_LIBRARY_DEFAULTS = {
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_gain_to_split": 0.0,
    "bagging_freq": 1,
    "bagging_fraction": 1.0,
    "feature_fraction": 1.0,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "cat_smooth": 10.0,
    "cat_l2": 10.0,
    "min_child_samples": 20,
    "path_smooth": 0.0,
    "feature_fraction_bynode": 1.0,
    "max_bin": 255,
    "min_data_per_group": 100,
}

LGB_GLOBAL_ONLY_WARMSTARTS = [
    {
        "learning_rate": 0.07915115373276987,
        "num_leaves": 701,
        "min_gain_to_split": 0.01043681028522292,
        "bagging_freq": 4,
        "bagging_fraction": 0.9131597662775737,
        "feature_fraction": 0.9185460241704044,
        "lambda_l1": 0.2940068846435481,
        "lambda_l2": 0.07808769825130206,
        "cat_smooth": 10.658444460199851,
        "cat_l2": 0.023128315777729166,
        "min_child_samples": 10,
        "path_smooth": 95.79829622244954,
        "feature_fraction_bynode": 0.9760300688941463,
        "max_bin": 113,
        "min_data_per_group": 100,
    },
]


def _adapt_universal_warmstart(tier, n_rows):
    """Library defaults adapted so min_child_samples is in-range for the tier."""
    cfg = dict(LGB_LIBRARY_DEFAULTS)
    cap = max(5, min(25, n_rows // 100))
    cfg["min_child_samples"] = min(cfg["min_child_samples"], cap)
    return cfg


N_OUTER_FOLDS = 10
NUM_BOOST_ROUND = 100_000
EARLY_STOPPING_ROUNDS = 1000
TUNING_NUM_BOOST_ROUND = 2500
TUNING_EARLY_STOPPING = 100


# Data loading

X_train = (
    pl.read_parquet("X_train.parquet")
    .drop("ASSESSMENT_ID")
    .drop("all_students_y")
    .drop("estimated_se")
)
y_train = pl.read_parquet("y_train.parquet")
X_pred = (
    pl.read_parquet("X_pred.parquet")
    .drop("ASSESSMENT_ID")
    .drop("all_students_y")
    .drop("estimated_se")
)
X_pred_id = pl.read_parquet("X_pred_id.parquet")

y_np = y_train.to_numpy().ravel().astype(np.float32)


categorical_cols = [
    "SCHOOL",
    "SUBGROUP_NAME",
    "ASSESSMENT_NAME",
    "DISTRICT",
    "COUNTY",
    "DISTRICT_TYPE",
    "REGION",
    "MACRO_REGION",
    "school_type",
    "COUNTY_x_DTYPE",
    "COUNTY_x_ASSESSMENT_NAME",
    "DISTRICT_TYPE_x_ASSESSMENT_NAME",
    "DISTRICT_TYPE_x_SUBGROUP_NAME",
    "ASSESSMENT_NAME_x_SUBGROUP_NAME",
    "DISTRICT_TYPE_x_REGION",
]
numeric_cols_orig = [c for c in X_train.columns if c not in categorical_cols]
cat_cols_ordered = [c for c in X_train.columns if c in set(categorical_cols)]

X_train_pd = X_train.with_columns(
    [pl.col(c).cast(pl.Float32, strict=False) for c in numeric_cols_orig]
).to_pandas()
X_pred_pd = X_pred.with_columns(
    [pl.col(c).cast(pl.Float32, strict=False) for c in numeric_cols_orig]
).to_pandas()
for col in cat_cols_ordered:
    X_train_pd[col] = X_train_pd[col].astype("category")
    X_pred_pd[col] = X_pred_pd[col].astype("category")

n_train = len(X_train)
n_pred = len(X_pred)

region_train = X_train["REGION"].to_numpy()
region_pred = X_pred["REGION"].to_numpy()
macro_train = X_train["MACRO_REGION"].to_numpy()
macro_pred = X_pred["MACRO_REGION"].to_numpy()
dtype_train = X_train["DISTRICT_TYPE"].to_numpy()
dtype_pred = X_pred["DISTRICT_TYPE"].to_numpy()
subgroup_train = X_train["SUBGROUP_NAME"].to_numpy()
subgroup_pred = X_pred["SUBGROUP_NAME"].to_numpy()
assessment_train = X_train["ASSESSMENT_NAME"].to_numpy()
assessment_pred = X_pred["ASSESSMENT_NAME"].to_numpy()
county_train = X_train["COUNTY"].to_numpy()
county_pred = X_pred["COUNTY"].to_numpy()

UNIQUE_REGIONS = sorted(set(region_train) | set(region_pred))
UNIQUE_MACROS = sorted(set(macro_train) | set(macro_pred))
UNIQUE_DTYPES = sorted(set(dtype_train) | set(dtype_pred))
UNIQUE_GROUPS = sorted(set(subgroup_train) | set(subgroup_pred))
UNIQUE_ASSESSMENTS = sorted(set(assessment_train) | set(assessment_pred))
UNIQUE_COUNTIES = sorted(set(county_train) | set(county_pred))

print(f"Regions ({len(UNIQUE_REGIONS)}): {UNIQUE_REGIONS}")
print(f"Macros ({len(UNIQUE_MACROS)}): {UNIQUE_MACROS}")
print(f"District types ({len(UNIQUE_DTYPES)}): {UNIQUE_DTYPES}")
print(f"Group Types ({len(UNIQUE_GROUPS)}): {UNIQUE_GROUPS}")
print(f"Assessment types ({len(UNIQUE_ASSESSMENTS)}): {UNIQUE_ASSESSMENTS}")
print(f"Counties ({len(UNIQUE_COUNTIES)}): {len(UNIQUE_COUNTIES)} unique")


# Stratification helpers


def _pool_rare(key_arr, n_folds):
    """Pool classes with fewer than n_folds members into '__rare__'."""
    counts = Counter(key_arr)
    rare = {v for v, c in counts.items() if c < n_folds}
    if not rare:
        return key_arr
    return np.array([v if v not in rare else "__rare__" for v in key_arr])


def stratification_key_for_tier(tier, idx_subset, n_folds):
    """Return a stratification key for the given tier's partition, using the
    next-most-informative categorical that ISN'T the partition key.
    """
    if tier == "assessment":
        key = subgroup_train[idx_subset]
    else:
        key = assessment_train[idx_subset]
    return _pool_rare(key, n_folds)


# Tier dispatch


def get_tier_indices(tier, name):
    if tier == "global":
        return np.arange(n_train), np.arange(n_pred)
    if tier == "macro":
        return np.where(macro_train == name)[0], np.where(macro_pred == name)[0]
    if tier == "region":
        return np.where(region_train == name)[0], np.where(region_pred == name)[0]
    if tier == "district_type":
        return np.where(dtype_train == name)[0], np.where(dtype_pred == name)[0]
    if tier == "subgroup":
        return np.where(subgroup_train == name)[0], np.where(subgroup_pred == name)[0]
    if tier == "assessment":
        return (
            np.where(assessment_train == name)[0],
            np.where(assessment_pred == name)[0],
        )
    if tier == "county":
        return np.where(county_train == name)[0], np.where(county_pred == name)[0]
    raise ValueError(f"Unknown tier: {tier}")


def search_space_for_tier(tier, trial, n_rows):
    """Tier-aware LightGBM hyperparameter search space."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.103, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 1024),
        "min_gain_to_split": trial.suggest_float(
            "min_gain_to_split",
            0.0,
            200,
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.3, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.01, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 1000),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 1000),
        "cat_smooth": trial.suggest_float("cat_smooth", 0.0, 100),
        "cat_l2": trial.suggest_float("cat_l2", 0.0, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 200),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 100.0),
        "feature_fraction_bynode": trial.suggest_float(
            "feature_fraction_bynode", 0.3, 1.0
        ),
        "max_bin": trial.suggest_int("max_bin", 63, 511),
        "min_data_per_group": trial.suggest_int("min_data_per_group", 5, 200),
    }


def n_trials_for_tier(tier, n_rows):
    if tier == "global":
        return 5
    return 25


def n_inner_folds_for_tier(tier):
    return 10


def _safe(name):
    return (
        name.replace(" ", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _params_path(tier, name):
    return PARAMS_DIR / f"{tier}_{_safe(name)}.json"


def _meta_path(tier, name):
    return PARAMS_DIR / f"{tier}_{_safe(name)}.meta.json"


def _submitted_path(tier, name):
    return PARAMS_DIR / f"{tier}_{_safe(name)}.submitted.json"


def _params_hash(params):
    """Stable hash of a params dict. Sort keys so order doesn't matter."""
    canonical = json.dumps(params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _expert_cache_path(tier, name, fold_idx, params_hash):
    return EXPERTS_DIR / f"{tier}_{_safe(name)}_fold{fold_idx}_{params_hash}.npz"


def maybe_save_best(tier, name, study, tolerance=1e-6):
    """Save params iff study found a strictly better CV score than what's on disk."""
    new_value = study.best_value
    new_params = study.best_params
    out_path = _params_path(tier, name)
    meta_path = _meta_path(tier, name)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        prev_value = meta["best_value"]
        if new_value >= prev_value - tolerance:
            print(
                f"[{tier}/{name}] no improvement: new={new_value:.6f} "
                f"vs saved={prev_value:.6f} — keeping existing params"
            )
            return False
        print(
            f"[{tier}/{name}] IMPROVED: new={new_value:.6f} "
            f"vs saved={prev_value:.6f} (delta={prev_value - new_value:.6f})"
        )
    else:
        print(f"[{tier}/{name}] first save: best_value={new_value:.6f}")

    out_path.write_text(json.dumps(new_params, indent=2))
    meta_path.write_text(
        json.dumps(
            {
                "best_value": new_value,
                "n_trials": len(study.trials),
                "study_name": study.study_name,
            },
            indent=2,
        )
    )
    print(f"[{tier}/{name}] saved params to {out_path}")
    return True


# Tuning


def tune_one(tier, name, n_trials=None):
    train_idx_all, _ = get_tier_indices(tier, name)
    n_rows = len(train_idx_all)
    print(f"\n[{tier}/{name}] tuning on {n_rows} rows")

    if n_trials is None:
        n_trials = n_trials_for_tier(tier, n_rows)
    n_inner = n_inner_folds_for_tier(tier)

    X_subset_pd = X_train_pd.iloc[train_idx_all].reset_index(drop=True)
    y_subset_pct = y_np[train_idx_all]
    strat_key = stratification_key_for_tier(tier, train_idx_all, n_inner)

    def objective(trial):
        params = search_space_for_tier(tier, trial, n_rows)
        kf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED)
        fold_rmses_pct = []

        for fold_i, (tr_i, va_i) in enumerate(kf.split(X_subset_pd, strat_key)):
            X_tr_pd = X_subset_pd.iloc[tr_i]
            X_va_pd = X_subset_pd.iloc[va_i]
            y_tr_pct = y_subset_pct[tr_i]
            y_va_pct = y_subset_pct[va_i]

            dtrain = lgb.Dataset(
                X_tr_pd, label=y_tr_pct, categorical_feature=cat_cols_ordered
            )
            dval = lgb.Dataset(
                X_va_pd,
                label=y_va_pct,
                categorical_feature=cat_cols_ordered,
                reference=dtrain,
            )

            model = lgb.train(
                {
                    **params,
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                    "seed": SEED,
                },
                dtrain,
                num_boost_round=TUNING_NUM_BOOST_ROUND,
                valid_sets=[dval],
                valid_names=["valid"],
                callbacks=[
                    lgb.callback.early_stopping(
                        stopping_rounds=TUNING_EARLY_STOPPING,
                        verbose=False,
                        min_delta=0.005,
                    ),
                ],
            )
            preds_pct = model.predict(X_va_pd, num_iteration=model.best_iteration)

            rmse_pct = float(np.sqrt(np.mean((y_va_pct - preds_pct) ** 2)))
            fold_rmses_pct.append(rmse_pct)

            del dtrain, dval, model

            if fold_i == 0 and rmse_pct > y_subset_pct.std() * 1.5:
                return rmse_pct

        return float(np.mean(fold_rmses_pct))

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            str(LOGS_DIR / f"lightgbm_journal_{tier}_{_safe(name)}.log")
        )
    )
    study = optuna.create_study(
        study_name=f"lightgbm_{tier}_{name}",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(seed=SEED, n_startup_trials=5),
    )

    # Warm-start gating: source of truth is the study itself, not a sidecar
    # marker file. If the study is fresh (zero trials), enqueue defaults so
    # trial 0 is always the library default — even after journal wipes.
    if len(study.trials) == 0:
        warmstarts = [_adapt_universal_warmstart(tier, n_rows)]
        if tier == "global":
            warmstarts.extend(LGB_GLOBAL_ONLY_WARMSTARTS)
        for params in warmstarts:
            study.enqueue_trial(params, skip_if_exists=True)
        print(
            f"[{tier}/{name}] enqueued {len(warmstarts)} warm-start trials (fresh study)"
        )
    else:
        print(
            f"[{tier}/{name}] study has {len(study.trials)} existing trials, "
            f"skipping warm-start"
        )

    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True
    )

    saved = maybe_save_best(tier, name, study)
    print(f"[{tier}/{name}] best CV RMSE (pct): {study.best_value:.4f}")
    return study.best_params, study.best_value, saved


def load_params(tier, name):
    path = _params_path(tier, name)
    if path.exists():
        return json.loads(path.read_text())
    print(f"  [{tier}/{name}] no tuned params; using LightGBM library defaults")
    return dict(LGB_LIBRARY_DEFAULTS)


# Per-fold expert fit (with prediction caching)
# IMPORTANT: cache is keyed by (tier, name, fold, params_hash). It will
# NOT detect changes to: SEED, N_OUTER_FOLDS, the kfold stratification key,
# or the underlying X_train.parquet. After such changes, delete EXPERTS_DIR
# to force re-fitting:  rm -rf fitted_experts_lightgbm/
#
# Cache schema v2: stores val_idx + pred_idx + params_json + val_rmse + best_iter
# alongside predictions. Self-validating — no kfold replay required for
# downstream consumers (e.g., moe_submission).


def fit_one_expert(tier, name, train_idx, val_idx, pred_idx, fold_idx=None):
    """Fit (or load from cache) a tier-specific LightGBM on its subset of one
    outer fold. Returns (val_preds, test_preds) in original units.

    fold_idx controls caching:
      - Pass fold_idx (int) during stack() to enable cache read/write.
      - Pass fold_idx=None in submit_global() to skip caching.
    """
    params = load_params(tier, name)

    if fold_idx is not None:
        cache_key = _params_hash(params)
        cache_path = _expert_cache_path(tier, name, fold_idx, cache_key)
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            if (
                "cache_version" in cached.files
                and int(cached["cache_version"].item()) == CACHE_VERSION
            ):
                print(
                    f"  [{tier}/{name}] fold {fold_idx} loaded from cache (hash {cache_key})"
                )
                return cached["val_preds"], cached["pred_preds"]
            print(f"  [{tier}/{name}] fold {fold_idx} cache outdated schema, refitting")

    X_tr_pd = X_train_pd.iloc[train_idx]
    X_va_pd = X_train_pd.iloc[val_idx]
    X_te_pd = X_pred_pd.iloc[pred_idx]
    y_tr = y_np[train_idx]
    y_va = y_np[val_idx]

    dtrain = lgb.Dataset(X_tr_pd, label=y_tr, categorical_feature=cat_cols_ordered)
    dval = lgb.Dataset(
        X_va_pd, label=y_va, categorical_feature=cat_cols_ordered, reference=dtrain
    )

    model = lgb.train(
        {
            **params,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": SEED,
        },
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[
            lgb.callback.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True, min_delta=0.005
            ),
            lgb.callback.log_evaluation(period=200),
        ],
    )
    best_iter = model.best_iteration
    print(f"    [{tier}/{name}] best_iteration: {best_iter}")

    val_preds = model.predict(X_va_pd, num_iteration=best_iter)
    test_preds = model.predict(X_te_pd, num_iteration=best_iter)

    val_rmse = float(np.sqrt(np.mean((y_va - val_preds) ** 2)))

    del model, dtrain, dval

    if fold_idx is not None:
        np.savez_compressed(
            cache_path,
            val_preds=val_preds,
            pred_preds=test_preds,
            val_idx=np.asarray(val_idx, dtype=np.int64),
            pred_idx=np.asarray(pred_idx, dtype=np.int64),
            params_hash=cache_key,
            params_json=json.dumps(params),
            val_rmse=np.float64(val_rmse),
            best_iter=np.int64(best_iter),
            cache_version=np.int64(CACHE_VERSION),
        )
        print(
            f"  [{tier}/{name}] fold {fold_idx} cached (hash {cache_key}, val_rmse={val_rmse:.4f})"
        )

    return val_preds, test_preds


# Stacking


def run_partition_tier(
    tier,
    names,
    membership_train,
    membership_pred,
    train_idx,
    val_idx,
    oof_array,
    test_array,
    filled_train,
    filled_test,
    fallback_oof,
    fallback_test,
    fold_idx,
):
    for name in names:
        tr_idx_p = train_idx[membership_train[train_idx] == name]
        va_idx_p = val_idx[membership_train[val_idx] == name]
        te_idx_p = np.where(membership_pred == name)[0]

        if len(tr_idx_p) == 0:
            print(f"  [{tier}/{name}] FALLBACK: 0 train rows, copying global")
            oof_array[va_idx_p] = fallback_oof[va_idx_p]
            test_array[te_idx_p] += fallback_test[te_idx_p] / N_OUTER_FOLDS
            filled_train[va_idx_p] = True
            filled_test[te_idx_p] = True
            continue

        print(
            f"  [{tier}/{name}] train={len(tr_idx_p)} "
            f"val={len(va_idx_p)} test={len(te_idx_p)}"
        )
        val_preds, test_preds = fit_one_expert(
            tier, name, tr_idx_p, va_idx_p, te_idx_p, fold_idx=fold_idx
        )
        oof_array[va_idx_p] = val_preds
        test_array[te_idx_p] += test_preds / N_OUTER_FOLDS
        filled_train[va_idx_p] = True
        filled_test[te_idx_p] = True


def _params_changed_since_artifacts():
    if not ARTIFACTS_PATH.exists():
        return True
    artifacts_mtime = ARTIFACTS_PATH.stat().st_mtime
    param_files = [
        p
        for p in PARAMS_DIR.glob("*.json")
        if not p.name.endswith(".meta.json") and not p.name.endswith(".submitted.json")
    ]
    if not param_files:
        return True
    newest = max(p.stat().st_mtime for p in param_files)
    return newest > artifacts_mtime


def stack(force=False):
    if not force and not _params_changed_since_artifacts():
        print(
            f"\nAll params files are older than {ARTIFACTS_PATH.name} — "
            f"existing artifacts are still current. Skipping stack."
        )
        print(f"  To force regeneration, pass --force or delete {ARTIFACTS_PATH}.")
        print(
            "  Note: this check does NOT detect changes to MACRO_REGION, kfold seed, "
            "search space, or stratification logic — use --force after such changes."
        )
        return

    outer_strat_key = _pool_rare(assessment_train, N_OUTER_FOLDS)
    kfold = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)

    oof_global = np.zeros(n_train)
    oof_macro = np.zeros(n_train)
    oof_region = np.zeros(n_train)
    oof_dtype = np.zeros(n_train)
    oof_county = np.zeros(n_train)
    oof_group = np.zeros(n_train)
    oof_assessment = np.zeros(n_train)

    test_global = np.zeros(n_pred)
    test_macro = np.zeros(n_pred)
    test_region = np.zeros(n_pred)
    test_dtype = np.zeros(n_pred)
    test_county = np.zeros(n_pred)
    test_group = np.zeros(n_pred)
    test_assessment = np.zeros(n_pred)

    filled_macro_tr = np.zeros(n_train, dtype=bool)
    filled_region_tr = np.zeros(n_train, dtype=bool)
    filled_dtype_tr = np.zeros(n_train, dtype=bool)
    filled_county_tr = np.zeros(n_train, dtype=bool)
    filled_group_tr = np.zeros(n_train, dtype=bool)
    filled_assessment_tr = np.zeros(n_train, dtype=bool)

    filled_macro_te = np.zeros(n_pred, dtype=bool)
    filled_region_te = np.zeros(n_pred, dtype=bool)
    filled_dtype_te = np.zeros(n_pred, dtype=bool)
    filled_county_te = np.zeros(n_pred, dtype=bool)
    filled_group_te = np.zeros(n_pred, dtype=bool)
    filled_assessment_te = np.zeros(n_pred, dtype=bool)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kfold.split(X_train_pd, outer_strat_key)
    ):
        print(
            f"\nFOLD {fold_idx + 1}/{N_OUTER_FOLDS}  "
            f"train={len(train_idx)}  val={len(val_idx)}"
        )

        print("  [global]")
        val_preds, test_preds = fit_one_expert(
            "global", "all", train_idx, val_idx, np.arange(n_pred), fold_idx=fold_idx
        )
        oof_global[val_idx] = val_preds
        test_global += test_preds / N_OUTER_FOLDS

        run_partition_tier(
            "macro",
            UNIQUE_MACROS,
            macro_train,
            macro_pred,
            train_idx,
            val_idx,
            oof_macro,
            test_macro,
            filled_macro_tr,
            filled_macro_te,
            oof_global,
            test_global,
            fold_idx,
        )
        run_partition_tier(
            "region",
            UNIQUE_REGIONS,
            region_train,
            region_pred,
            train_idx,
            val_idx,
            oof_region,
            test_region,
            filled_region_tr,
            filled_region_te,
            oof_global,
            test_global,
            fold_idx,
        )
        run_partition_tier(
            "district_type",
            UNIQUE_DTYPES,
            dtype_train,
            dtype_pred,
            train_idx,
            val_idx,
            oof_dtype,
            test_dtype,
            filled_dtype_tr,
            filled_dtype_te,
            oof_global,
            test_global,
            fold_idx,
        )
        run_partition_tier(
            "county",
            UNIQUE_COUNTIES,
            county_train,
            county_pred,
            train_idx,
            val_idx,
            oof_county,
            test_county,
            filled_county_tr,
            filled_county_te,
            oof_global,
            test_global,
            fold_idx,
        )
        run_partition_tier(
            "assessment",
            UNIQUE_ASSESSMENTS,
            assessment_train,
            assessment_pred,
            train_idx,
            val_idx,
            oof_assessment,
            test_assessment,
            filled_assessment_tr,
            filled_assessment_te,
            oof_global,
            test_global,
            fold_idx,
        )
        run_partition_tier(
            "subgroup",
            UNIQUE_GROUPS,
            subgroup_train,
            subgroup_pred,
            train_idx,
            val_idx,
            oof_group,
            test_group,
            filled_group_tr,
            filled_group_te,
            oof_global,
            test_global,
            fold_idx,
        )

    assert filled_macro_tr.all(), "Some train rows missed macro OOF"
    assert filled_region_tr.all(), "Some train rows missed region OOF"
    assert filled_dtype_tr.all(), "Some train rows missed district_type OOF"
    assert filled_county_tr.all(), "Some train rows missed county OOF"
    assert filled_group_tr.all(), "Some train rows missed group OOF"
    assert filled_assessment_tr.all(), "Some train rows missed assessment OOF"

    assert filled_macro_te.all(), "Some test rows missed macro pred"
    assert filled_region_te.all(), "Some test rows missed region pred"
    assert filled_dtype_te.all(), "Some test rows missed district_type pred"
    assert filled_county_te.all(), "Some test rows missed county pred"
    assert filled_group_te.all(), "Some test rows missed group pred"
    assert filled_assessment_te.all(), "Some test rows missed assessment pred"

    # Single canonical tiers list — used for RMSE printing, residual
    # correlations, diagnostic JSON, ridge coefficients, and stack npz.
    # Column order here defines the column order in oof_stack/test_stack.
    tiers = [
        ("global", oof_global),
        ("macro", oof_macro),
        ("region", oof_region),
        ("district_type", oof_dtype),
        ("county", oof_county),
        ("subgroup", oof_group),
        ("assessment", oof_assessment),
    ]

    print("\nOOF RMSE by tier (percentage scale):")
    for name, oof in tiers:
        rmse = float(np.sqrt(np.mean((y_np - oof) ** 2)))
        print(f"  {name:14s}: {rmse:.4f}")

    resid = {name: y_np - oof for name, oof in tiers}
    keys = list(resid.keys())
    print("\nResidual correlations:")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            corr = np.corrcoef(resid[a], resid[b])[0, 1]
            print(f"  {a:14s} <-> {b:14s}: {corr:.4f}")

    np.savez_compressed(
        ARTIFACTS_PATH,
        oof_global=oof_global,
        oof_macro=oof_macro,
        oof_region=oof_region,
        oof_district_type=oof_dtype,
        oof_county=oof_county,
        oof_group=oof_group,
        oof_assessment=oof_assessment,
        test_global=test_global,
        test_macro=test_macro,
        test_region=test_region,
        test_district_type=test_dtype,
        test_county=test_county,
        test_group=test_group,
        test_assessment=test_assessment,
        y=y_np,
    )
    print(f"\nSaved {ARTIFACTS_PATH}")

    # Diagnostics

    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostic_path = Path(f"lightgbm_baseline_diagnostic_{version_tag}.json")
    diagnostic = {
        "run_timestamp": datetime.now().isoformat(),
        "n_outer_folds": N_OUTER_FOLDS,
        "n_train": int(n_train),
        "n_pred": int(n_pred),
        "y_std": float(y_np.std()),
        "y_mean": float(y_np.mean()),
        "per_tier_oof_rmse": {
            name: float(np.sqrt(np.mean((y_np - oof) ** 2))) for name, oof in tiers
        },
        "per_tier_oof_mae": {
            name: float(np.mean(np.abs(y_np - oof))) for name, oof in tiers
        },
        "residual_correlations": {
            f"{a}__{b}": float(np.corrcoef(resid[a], resid[b])[0, 1])
            for i, a in enumerate(keys)
            for b in keys[i + 1 :]
        },
    }

    partition_breakdowns = {}
    for tier_name, oof_arr, membership in [
        ("macro", oof_macro, macro_train),
        ("region", oof_region, region_train),
        ("district_type", oof_dtype, dtype_train),
        ("assessment", oof_assessment, assessment_train),
        ("subgroup", oof_group, subgroup_train),
        ("county", oof_county, county_train),
    ]:
        partition_breakdowns[tier_name] = {}
        for partition_name in np.unique(membership):
            mask = membership == partition_name
            if mask.sum() == 0:
                continue
            global_rmse_here = float(
                np.sqrt(np.mean((y_np[mask] - oof_global[mask]) ** 2))
            )
            tier_rmse_here = float(np.sqrt(np.mean((y_np[mask] - oof_arr[mask]) ** 2)))
            partition_breakdowns[tier_name][str(partition_name)] = {
                "n": int(mask.sum()),
                "global_rmse": global_rmse_here,
                "tier_rmse": tier_rmse_here,
                "improvement": global_rmse_here - tier_rmse_here,
            }
    diagnostic["partition_breakdowns"] = partition_breakdowns

    # Stack matrix — column order matches tiers list above
    oof_stack = np.column_stack([oof for _, oof in tiers])
    test_stack = np.column_stack(
        [
            test_global,
            test_macro,
            test_region,
            test_dtype,
            test_county,
            test_group,
            test_assessment,
        ]
    )

    equal_weight_oof = oof_stack.mean(axis=1)
    diagnostic["equal_weight_ensemble_rmse"] = float(
        np.sqrt(np.mean((y_np - equal_weight_oof) ** 2))
    )

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(oof_stack, y_np)
    ridge_oof_pred = ridge.predict(oof_stack)
    diagnostic["ridge_meta_rmse_in_sample"] = float(
        np.sqrt(np.mean((y_np - ridge_oof_pred) ** 2))
    )
    # zip against tiers so coefficient labels always match column order
    diagnostic["ridge_coefficients"] = {
        name: float(c) for (name, _), c in zip(tiers, ridge.coef_)
    }
    diagnostic["ridge_intercept"] = float(ridge.intercept_)
    diagnostic["ridge_alpha"] = float(ridge.alpha_)

    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stack_npz_path = f"lightgbm_baseline_stack_{version_tag}.npz"
    np.savez_compressed(
        stack_npz_path,
        oof_stack=oof_stack,
        test_stack=test_stack,
        y=y_np,
        tier_names=np.array([name for name, _ in tiers]),
    )
    diagnostic["stack_npz_path"] = stack_npz_path
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    print(f"Diagnostic dump written to {diagnostic_path}")
    print(f"Stack npz written to {stack_npz_path}")


# Global submission


def submit_global(force=False):
    tier, name = "global", "all"
    meta_path = _meta_path(tier, name)
    submitted_path = _submitted_path(tier, name)

    if not meta_path.exists():
        raise SystemExit(
            f"No tuned global params found at {meta_path}. Run --mode tune first."
        )

    meta = json.loads(meta_path.read_text())
    current_best = meta["best_value"]

    if not force and submitted_path.exists():
        submitted = json.loads(submitted_path.read_text())
        prev = submitted["best_value"]
        tolerance = 1e-6
        if current_best >= prev - tolerance:
            print(
                f"[submit-global] no improvement since last submission: "
                f"current={current_best:.6f} vs submitted={prev:.6f}"
            )
            print("  Skipping submission. Pass --force to regenerate anyway.")
            return

    print(f"[submit-global] training global LightGBM on full training data...")
    print(f"  current best CV RMSE: {current_best:.4f}")

    holdout_strat_key = _pool_rare(assessment_train, 20)
    tr_idx, va_idx = train_test_split(
        np.arange(n_train),
        test_size=0.05,
        stratify=holdout_strat_key,
        random_state=SEED,
    )
    pred_idx = np.arange(n_pred)

    val_preds, test_preds = fit_one_expert(
        tier, name, tr_idx, va_idx, pred_idx, fold_idx=None
    )
    val_rmse = float(np.sqrt(np.mean((y_np[va_idx] - val_preds) ** 2)))
    print(f"  holdout RMSE: {val_rmse:.4f}")
    test_preds = np.clip(test_preds, 0, 100)
    submission = X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", test_preds))
    submission.write_csv(GLOBAL_SUBMISSION_PATH)
    print(f"[submit-global] wrote {GLOBAL_SUBMISSION_PATH} ({len(test_preds)} rows)")

    submitted_path.write_text(
        json.dumps(
            {
                "best_value": current_best,
                "holdout_rmse": val_rmse,
                "n_train_used": int(len(tr_idx)),
            },
            indent=2,
        )
    )
    print(f"[submit-global] recorded submission state at {submitted_path}")


# CLI


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["tune", "stack", "submit-global"], required=True)
    p.add_argument(
        "--tier",
        choices=[
            "global",
            "macro",
            "region",
            "district_type",
            "county",
            "subgroup",
            "assessment",
        ],
    )
    p.add_argument("--name", help="Partition value; omit to loop all in tier")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if no improvement (stack: artifacts; submit-global: submission)",
    )
    args = p.parse_args()

    if args.mode == "tune":
        if args.tier is None:
            raise SystemExit("--tier required in tune mode")
        if args.tier == "global":
            tune_one("global", "all", args.n_trials)
        elif args.name is not None:
            tune_one(args.tier, args.name, args.n_trials)
        else:
            names_by_tier = {
                "macro": UNIQUE_MACROS,
                "region": UNIQUE_REGIONS,
                "district_type": UNIQUE_DTYPES,
                "county": UNIQUE_COUNTIES,
                "subgroup": UNIQUE_GROUPS,
                "assessment": UNIQUE_ASSESSMENTS,
            }
            for nm in names_by_tier[args.tier]:
                try:
                    tune_one(args.tier, nm, args.n_trials)
                except KeyboardInterrupt:
                    print(f"\nInterrupted during {args.tier}/{nm}; stopping.")
                    raise
                except Exception as e:
                    print(f"\n[{args.tier}/{nm}] FAILED: {type(e).__name__}: {e}")
                    print("Continuing with next partition.\n")
                    continue
    elif args.mode == "stack":
        stack(force=args.force)
    else:
        submit_global(force=args.force)


if __name__ == "__main__":
    main()
