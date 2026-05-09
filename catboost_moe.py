#!/usr/bin/env python3

#!/usr/bin/env python3

"""
CatBoost mixture-of-experts.

Default behavior: train once with library defaults across all outer folds
and partition tiers, write artifacts, exit. No tuning unless the user
explicitly opts in.

The argument for default-first: CatBoost's defaults are deliberately strong
out-of-the-box, especially for high-cardinality categoricals where its
internal ordered target statistics handle SCHOOL/DISTRICT/COUNTY without
manual encoding. Tuning typically buys 0.05-0.15 RMSE on top of defaults,
which isn't worth the time when CatBoost is the *third* base learner in a
stack and its main value is architectural diversity rather than peak
single-model performance.

Modes:
    --mode stack [--force]              # default; runs with library defaults
    --mode tune --enable-tune --tier T  # explicit opt-in for tuning
    --mode submit-global [--force]

Safety features matching ResNet/LightGBM/FT-Transformer MoE:
  - Per-fold prediction caching keyed by (tier, name, fold, params_hash)
  - StratifiedKFold on assessment for outer folds
  - Rare-class pooling
  - Stratified train_test_split in submit_global
  - Full diagnostic dump after stack()
"""

import argparse
import datetime
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 8675309
np.random.seed(SEED)
random.seed(SEED)
pl.enable_string_cache()

PARAMS_DIR = Path("best_params_catboost")
PARAMS_DIR.mkdir(exist_ok=True)
WARMSTART_DIR = Path("warmstart_markers_catboost")
WARMSTART_DIR.mkdir(exist_ok=True)
EXPERTS_DIR = Path("fitted_experts_catboost")
EXPERTS_DIR.mkdir(exist_ok=True)
ARTIFACTS_PATH = Path("catboost_moe_artifacts.npz")
GLOBAL_SUBMISSION_PATH = Path("submission_catboost_global.csv")


# ── Default params ────────────────────────────────────────────────────────────

# These are the library defaults plus a few opinionated tweaks (longer
# training with early stopping, GPU when available). The whole point of
# this file is that these are the "tuned" params for our purposes.
CATBOOST_DEFAULTS = {
    "iterations": 100000,  # i'm only running this once so get as much out of it as possible
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "border_count": 254,
    "min_data_in_leaf": 1,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 200,
    "random_seed": SEED,
    "verbose": 500,
}


def _gpu_available():
    """Best-effort detection of a usable CatBoost GPU. Falls back to CPU on
    any error (CatBoost's GPU mode is sensitive to library versions)."""
    try:
        from catboost.utils import get_gpu_device_count

        return get_gpu_device_count() > 0
    except Exception:
        return False


CATBOOST_TASK_TYPE = "GPU" if _gpu_available() else "CPU"
print(f"CatBoost task_type: {CATBOOST_TASK_TYPE}")


N_OUTER_FOLDS = 10
TUNING_ITERATIONS = 1500
TUNING_EARLY_STOPPING = 50


# ── Data loading ──────────────────────────────────────────────────────────────

X_train = pl.read_parquet("X_train.parquet").drop("ASSESSMENT_ID")
y_train = pl.read_parquet("y_train.parquet")
X_pred = pl.read_parquet("X_pred.parquet").drop("ASSESSMENT_ID")
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
]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
cat_cols_ordered = [c for c in X_train.columns if c in set(categorical_cols)]

# CatBoost wants categoricals as strings or pandas Categorical, NOT
# ordinally-encoded ints. Numerics cast to float32 for memory.
X_train_pd = X_train.with_columns(
    [pl.col(c).cast(pl.Float32, strict=False) for c in numeric_cols]
).to_pandas()
X_pred_pd = X_pred.with_columns(
    [pl.col(c).cast(pl.Float32, strict=False) for c in numeric_cols]
).to_pandas()
for col in cat_cols_ordered:
    X_train_pd[col] = X_train_pd[col].astype(str)
    X_pred_pd[col] = X_pred_pd[col].astype(str)

cat_features_idx = [X_train_pd.columns.get_loc(c) for c in cat_cols_ordered]

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

print(f"Regions ({len(UNIQUE_REGIONS)}): {len(UNIQUE_REGIONS)} unique")
print(f"Macros ({len(UNIQUE_MACROS)}): {len(UNIQUE_MACROS)} unique")
print(f"District types ({len(UNIQUE_DTYPES)}): {len(UNIQUE_DTYPES)} unique")
print(f"Subgroups ({len(UNIQUE_GROUPS)}): {len(UNIQUE_GROUPS)} unique")
print(f"Assessments ({len(UNIQUE_ASSESSMENTS)}): {len(UNIQUE_ASSESSMENTS)} unique")
print(f"Counties ({len(UNIQUE_COUNTIES)}): {len(UNIQUE_COUNTIES)} unique")


# ── Stratification ────────────────────────────────────────────────────────────


def _pool_rare(key_arr, n_folds):
    counts = Counter(key_arr)
    rare = {v for v, c in counts.items() if c < n_folds}
    if not rare:
        return key_arr
    return np.array([v if v not in rare else "__rare__" for v in key_arr])


def stratification_key_for_tier(tier, idx_subset, n_folds):
    if tier == "assessment":
        key = subgroup_train[idx_subset]
    else:
        key = assessment_train[idx_subset]
    return _pool_rare(key, n_folds)


# ── Tier dispatch ─────────────────────────────────────────────────────────────


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
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _expert_cache_path(tier, name, fold_idx, params_hash):
    return EXPERTS_DIR / f"{tier}_{_safe(name)}_fold{fold_idx}_{params_hash}.npz"


def load_params(tier, name):
    """Returns tuned params if they exist, else CatBoost defaults.
    The defaults are explicitly the intended behavior for this MoE."""
    path = _params_path(tier, name)
    if path.exists():
        params = json.loads(path.read_text())
        print(f"  [{tier}/{name}] using tuned params from {path}")
        return params
    return dict(CATBOOST_DEFAULTS)


# ── Tuning (gated behind --enable-tune flag) ──────────────────────────────────


def search_space_for_tier(tier, trial, n_rows):
    """Conservative search space — CatBoost defaults are strong, so
    perturb modestly rather than searching widely."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
    }


def n_trials_for_tier(tier, n_rows):
    return 20 if tier == "global" else 8


def maybe_save_best(tier, name, study, tolerance=1e-6):
    new_value = study.best_value
    new_params = study.best_params
    out_path = _params_path(tier, name)
    meta_path = _meta_path(tier, name)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        prev_value = meta["best_value"]
        if new_value >= prev_value - tolerance:
            print(
                f"[{tier}/{name}] no improvement: {new_value:.6f} vs {prev_value:.6f}"
            )
            return False
        print(f"[{tier}/{name}] IMPROVED: {new_value:.6f} vs {prev_value:.6f}")
    else:
        print(f"[{tier}/{name}] first save: {new_value:.6f}")

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
    return True


def tune_one(tier, name, n_trials=None):
    train_idx_all, _ = get_tier_indices(tier, name)
    n_rows = len(train_idx_all)
    print(f"\n[{tier}/{name}] tuning on {n_rows} rows")

    if n_rows < 50:
        print(f"[{tier}/{name}] too few rows; skipping")
        return None, float("inf"), False

    if n_trials is None:
        n_trials = n_trials_for_tier(tier, n_rows)
    n_inner = 5  # 5-fold inner CV for tuning speed

    X_subset = X_train_pd.iloc[train_idx_all].reset_index(drop=True)
    y_subset = y_np[train_idx_all]
    strat_key = stratification_key_for_tier(tier, train_idx_all, n_inner)

    def objective(trial):
        params = search_space_for_tier(tier, trial, n_rows)
        kf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED)
        fold_rmses = []

        for fold_i, (tr_i, va_i) in enumerate(kf.split(X_subset, strat_key)):
            train_pool = Pool(
                X_subset.iloc[tr_i],
                label=y_subset[tr_i],
                cat_features=cat_features_idx,
            )
            val_pool = Pool(
                X_subset.iloc[va_i],
                label=y_subset[va_i],
                cat_features=cat_features_idx,
            )

            model = CatBoostRegressor(
                **{**CATBOOST_DEFAULTS, **params},
                iterations=TUNING_ITERATIONS,
                early_stopping_rounds=TUNING_EARLY_STOPPING,
                task_type=CATBOOST_TASK_TYPE,
                verbose=0,
            )
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            preds = model.predict(val_pool)
            fold_rmse = float(np.sqrt(np.mean((y_subset[va_i] - preds) ** 2)))
            fold_rmses.append(fold_rmse)

            trial.report(float(np.mean(fold_rmses)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

            del model, train_pool, val_pool

        return float(np.mean(fold_rmses))

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            f"./catboost_journal_{tier}_{_safe(name)}.log"
        )
    )
    study = optuna.create_study(
        study_name=f"catboost_{tier}_{name}",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=2),
    )

    marker = WARMSTART_DIR / f"{tier}_{_safe(name)}.done"
    if not marker.exists():
        # Single warm-start: the library defaults
        study.enqueue_trial(
            {
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 3.0,
                "min_data_in_leaf": 1,
                "random_strength": 1.0,
                "bagging_temperature": 1.0,
            }
        )
        marker.write_text("enqueued defaults warm-start\n")
        print(f"[{tier}/{name}] enqueued library defaults as first trial")

    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True
    )
    saved = maybe_save_best(tier, name, study)
    print(f"[{tier}/{name}] best CV RMSE: {study.best_value:.4f}")
    return study.best_params, study.best_value, saved


# ── Per-fold expert fit (with caching) ───────────────────────────────────────


def fit_one_expert(tier, name, train_idx, val_idx, pred_idx, fold_idx=None):
    params = load_params(tier, name)

    if fold_idx is not None:
        cache_key = _params_hash(params)
        cache_path = _expert_cache_path(tier, name, fold_idx, cache_key)
        if cache_path.exists():
            cached = np.load(cache_path)
            print(f"  [{tier}/{name}] fold {fold_idx} cached (hash {cache_key})")
            return cached["val_preds"], cached["pred_preds"]

    X_tr = X_train_pd.iloc[train_idx]
    X_va = X_train_pd.iloc[val_idx]
    X_te = X_pred_pd.iloc[pred_idx]
    y_tr = y_np[train_idx]
    y_va = y_np[val_idx]

    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_va, label=y_va, cat_features=cat_features_idx)
    test_pool = Pool(X_te, cat_features=cat_features_idx)

    # Merge tuned params (if any) over defaults
    full_params = {**CATBOOST_DEFAULTS, **params, "task_type": CATBOOST_TASK_TYPE}

    model = CatBoostRegressor(**full_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"    [{tier}/{name}] best_iteration: {model.get_best_iteration()}")

    val_preds = model.predict(val_pool)
    pred_preds = model.predict(test_pool)

    del model, train_pool, val_pool, test_pool

    if fold_idx is not None:
        np.savez_compressed(
            cache_path,
            val_preds=val_preds,
            pred_preds=pred_preds,
            params_hash=cache_key,
        )
        print(f"  [{tier}/{name}] fold {fold_idx} cached (hash {cache_key})")

    return val_preds, pred_preds


# ── Stacking ──────────────────────────────────────────────────────────────────


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
            f"  [{tier}/{name}] train={len(tr_idx_p)} val={len(va_idx_p)} test={len(te_idx_p)}"
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
        # No tuned params: artifacts depend only on defaults, which don't
        # change across runs. Re-run only if explicitly forced.
        return False
    newest = max(p.stat().st_mtime for p in param_files)
    return newest > artifacts_mtime


def stack(force=False):
    if not force and not _params_changed_since_artifacts():
        print(f"\nArtifacts current. Pass --force or delete {ARTIFACTS_PATH} to rerun.")
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
            f"\nFOLD {fold_idx+1}/{N_OUTER_FOLDS}  "
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

    assert filled_macro_tr.all(), "macro train OOF incomplete"
    assert filled_region_tr.all(), "region train OOF incomplete"
    assert filled_dtype_tr.all(), "district_type train OOF incomplete"
    assert filled_county_tr.all(), "county train OOF incomplete"
    assert filled_group_tr.all(), "subgroup train OOF incomplete"
    assert filled_assessment_tr.all(), "assessment train OOF incomplete"

    tiers = [
        ("global", oof_global),
        ("macro", oof_macro),
        ("region", oof_region),
        ("district_type", oof_dtype),
        ("county", oof_county),
        ("subgroup", oof_group),
        ("assessment", oof_assessment),
    ]
    resid = {name: y_np - oof for name, oof in tiers}
    keys = list(resid.keys())

    print("\nOOF RMSE by tier:")
    for name, oof in tiers:
        rmse = float(np.sqrt(np.mean((y_np - oof) ** 2)))
        print(f"  {name:14s}: {rmse:.4f}")

    print("\nResidual correlations:")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            print(f"  {a:14s} <-> {b:14s}: {np.corrcoef(resid[a], resid[b])[0,1]:.4f}")

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

    version_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_diagnostics(
        tiers=tiers,
        oof_global=oof_global,
        resid=resid,
        keys=keys,
        test_stack_arrays=[
            test_global,
            test_macro,
            test_region,
            test_dtype,
            test_county,
            test_group,
            test_assessment,
        ],
        version_tag=version_tag,
    )


def _save_diagnostics(tiers, oof_global, resid, keys, test_stack_arrays, version_tag):
    diagnostic_path = Path("catboost_diagnostic.json")
    diagnostic = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "n_outer_folds": N_OUTER_FOLDS,
        "n_train": int(n_train),
        "n_pred": int(n_pred),
        "y_std": float(y_np.std()),
        "y_mean": float(y_np.mean()),
        "task_type": CATBOOST_TASK_TYPE,
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
        ("macro", dict(tiers)["macro"], macro_train),
        ("region", dict(tiers)["region"], region_train),
        ("district_type", dict(tiers)["district_type"], dtype_train),
        ("county", dict(tiers)["county"], county_train),
    ]:
        partition_breakdowns[tier_name] = {}
        for partition_name in np.unique(membership):
            mask = membership == partition_name
            if mask.sum() == 0:
                continue
            g_rmse = float(np.sqrt(np.mean((y_np[mask] - oof_global[mask]) ** 2)))
            t_rmse = float(np.sqrt(np.mean((y_np[mask] - oof_arr[mask]) ** 2)))
            partition_breakdowns[tier_name][str(partition_name)] = {
                "n": int(mask.sum()),
                "global_rmse": g_rmse,
                "tier_rmse": t_rmse,
                "improvement": g_rmse - t_rmse,
            }
    diagnostic["partition_breakdowns"] = partition_breakdowns

    oof_stack = np.column_stack([oof for _, oof in tiers])
    test_stack = np.column_stack(test_stack_arrays)
    diagnostic["equal_weight_ensemble_rmse"] = float(
        np.sqrt(np.mean((y_np - oof_stack.mean(axis=1)) ** 2))
    )

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(oof_stack, y_np)
    diagnostic["ridge_meta_rmse_in_sample"] = float(
        np.sqrt(np.mean((y_np - ridge.predict(oof_stack)) ** 2))
    )
    diagnostic["ridge_coefficients"] = {
        name: float(c) for (name, _), c in zip(tiers, ridge.coef_)
    }
    diagnostic["ridge_intercept"] = float(ridge.intercept_)
    diagnostic["ridge_alpha"] = float(ridge.alpha_)

    stack_npz_path = f"catboost_stack_{version_tag}.npz"
    np.savez_compressed(
        stack_npz_path,
        oof_stack=oof_stack,
        test_stack=test_stack,
        y=y_np,
        tier_names=np.array([name for name, _ in tiers]),
    )
    diagnostic["stack_npz_path"] = stack_npz_path

    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    print(f"  Diagnostic JSON: {diagnostic_path}")
    print(f"  Stack npz: {stack_npz_path}")


# ── Submit global ─────────────────────────────────────────────────────────────


def submit_global(force=False):
    """One-shot global CatBoost on (almost) all training data."""
    tier, name = "global", "all"
    submitted_path = _submitted_path(tier, name)

    params_path = _params_path(tier, name)
    if params_path.exists():
        params = json.loads(params_path.read_text())
        identity = _params_hash(params)
    else:
        params = dict(CATBOOST_DEFAULTS)
        identity = "defaults"

    if not force and submitted_path.exists():
        submitted = json.loads(submitted_path.read_text())
        if submitted.get("identity") == identity:
            print(
                f"[submit-global] already submitted with identity={identity}. "
                f"Pass --force to regenerate."
            )
            return

    print(f"[submit-global] training global CatBoost (identity={identity})")

    holdout_strat_key = _pool_rare(assessment_train, 20)
    tr_idx, va_idx = train_test_split(
        np.arange(n_train),
        test_size=0.05,
        stratify=holdout_strat_key,
        random_state=SEED,
    )

    val_preds, test_preds = fit_one_expert(
        tier, name, tr_idx, va_idx, np.arange(n_pred), fold_idx=None
    )
    val_rmse = float(np.sqrt(np.mean((y_np[va_idx] - val_preds) ** 2)))
    print(f"  holdout RMSE: {val_rmse:.4f}")
    test_preds = np.clip(test_preds, 0, 100)

    X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", test_preds)).write_csv(
        GLOBAL_SUBMISSION_PATH
    )
    print(f"[submit-global] wrote {GLOBAL_SUBMISSION_PATH}")

    submitted_path.write_text(
        json.dumps(
            {
                "identity": identity,
                "holdout_rmse": val_rmse,
                "n_train_used": int(len(tr_idx)),
            },
            indent=2,
        )
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["tune", "stack", "submit-global"], required=True)
    p.add_argument(
        "--enable-tune",
        action="store_true",
        help="Required to actually run tuning. Without this flag, --mode tune "
        "exits without doing anything. CatBoost defaults are strong enough "
        "that tuning is opt-in.",
    )
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
    p.add_argument("--name")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.mode == "tune":
        if not args.enable_tune:
            print("Tuning requires --enable-tune flag. CatBoost defaults are the")
            print("intended behavior; only tune if you specifically want to.")
            print(
                "Use:  uv run catboost_moe.py --mode tune --enable-tune --tier global"
            )
            return
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
                    print(f"\nInterrupted at {args.tier}/{nm}")
                    raise
                except Exception as e:
                    print(f"\n[{args.tier}/{nm}] FAILED: {type(e).__name__}: {e}")
                    continue
    elif args.mode == "stack":
        stack(force=args.force)
    else:
        submit_global(force=args.force)


if __name__ == "__main__":
    main()
