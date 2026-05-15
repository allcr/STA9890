#!/usr/bin/env python3

"""
TabResNet mixture-of-experts with orthogonal partition tiers:
    global         — one expert on all rows
    macro          — one expert per MACRO_REGION
    region         — one expert per REGION
    district_type  — one expert per DISTRICT_TYPE
    assessment     — one expert per ASSESSMENT_NAME
    subgroup       — one expert per SUBGROUP_NAME

Each tier+name combination gets warm-started exactly once via a marker file,
with the Gorishniy et al. 2021 ResNet paper defaults as the universal first
trial.

Stratification: each tier picks a stratification key that is NOT its own
partition key, since within a partition the partition key is constant and
provides no stratification signal. Rare classes (< n_folds rows) are pooled
into '__rare__' to keep StratifiedKFold from blowing up on small partitions.

Modes:
    --mode tune --tier {global,macro,region,district_type,assessment,subgroup} [--name NAME]
    --mode stack [--force]
    --mode submit-global [--force]

Cache schema v2 (per-fold expert npz files):
    val_preds, pred_preds          — predictions in original units
    val_idx, pred_idx              — row indices into train/pred arrays
    params_hash, params_json       — for verification at load time
    val_rmse, best_epoch           — diagnostics
    cache_version=2                — schema marker

Logs and Optuna journal storage are written to ./logs/ (created if absent).
"""

from sklearn.model_selection import StratifiedKFold
import argparse
import gc
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV
import datetime
import hashlib

SEED = 8675309
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pl.enable_string_cache()

PARAMS_DIR = Path("best_params_resnet")
PARAMS_DIR.mkdir(exist_ok=True)

WARMSTART_DIR = Path("warmstart_markers_resnet")
WARMSTART_DIR.mkdir(exist_ok=True)

ARTIFACTS_PATH = Path("resnet_moe_artifacts.npz")
GLOBAL_SUBMISSION_PATH = Path("submission_resnet_global.csv")


EXPERTS_DIR = Path("fitted_experts_resnet")
EXPERTS_DIR.mkdir(exist_ok=True)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

CACHE_VERSION = 2


def _load_global_best_as_warmstart():
    """Return the best global params if they exist, else None.
    Used to warm-start partition-tier tuning with a known-good config."""
    path = PARAMS_DIR / "global_all.json"
    if path.exists():
        params = json.loads(path.read_text())
        print(f"[warmstart] loaded global best params from {path}")
        return params
    return None


GLOBAL_BEST_PARAMS = _load_global_best_as_warmstart()


def _params_hash(params):
    """Stable hash of a params dict. Sorts keys so order doesn't matter."""
    canonical = json.dumps(params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _expert_cache_path(tier, name, fold_idx, params_hash):
    """One file per (tier, partition, fold, params hash). Cache key is the
    hash, so changing any param invalidates that file but not others.
    """
    return EXPERTS_DIR / f"{tier}_{_safe(name)}_fold{fold_idx}_{params_hash}.npz"


# Warm-start configurations

# Paper defaults (Gorishniy et al. 2021) — universal warm-start for ALL tiers.
RESNET_PAPER_DEFAULTS = {
    "d_token": 8,
    "d": 256,
    "d_hidden_mult": 2.0,
    "n_blocks": 4,
    "dropout_first": 0.25,
    "dropout_second": 0.0,
    "lr": 3e-4,
    "wd": 1e-5,
}


RESNET_GLOBAL_ONLY_WARMSTARTS = [
    {
        "d_token": 32,
        "d": 768,
        "d_hidden_mult": 3.0,
        "n_blocks": 2,
        "dropout_first": 0.4,
        "dropout_second": 0.2,
        "lr": 0.0001,
        "wd": 0.0001,
        "batch_size": 2048,
    },
    {
        "d_token": 16,
        "d": 448,
        "d_hidden_mult": 2.855843486491791,
        "n_blocks": 4,
        "dropout_first": 0.050824363137484764,
        "dropout_second": 0.04924498624137331,
        "lr": 0.00010002217344257816,
        "wd": 9.105318441022081e-05,
        "batch_size": 256,
    },
]
RESNET_DEFAULTS_FOR_BASELINE = {
    **RESNET_PAPER_DEFAULTS,
    "batch_size": 512,
}


def load_params(tier, name):
    path = _params_path(tier, name)
    if path.exists():
        return json.loads(path.read_text())
    print(f"  [{tier}/{name}] no tuned params; using paper defaults (baseline run)")
    return dict(RESNET_DEFAULTS_FOR_BASELINE)


def _adapt_universal_warmstart(tier):
    """Paper defaults with batch_size set to a value in-range for the tier."""
    cfg = dict(RESNET_PAPER_DEFAULTS)
    if tier == "global":
        cfg["batch_size"] = 2048
    elif tier == "macro":
        cfg["batch_size"] = 1024
    else:  # region, district_type, assessment, subgroup
        cfg["batch_size"] = 512
    return cfg


N_OUTER_FOLDS = 10
EPOCHS = 3500
PATIENCE = 600
EPOCHS_TUNING = 300
PATIENCE_TUNING = 25


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
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
cat_cols_ordered = [c for c in X_train.columns if c in set(categorical_cols)]

num_train_raw = X_train.select(numeric_cols).to_numpy()
num_pred_raw = X_pred.select(numeric_cols).to_numpy()
cat_train_raw = X_train.select(cat_cols_ordered).to_numpy()
cat_pred_raw = X_pred.select(cat_cols_ordered).to_numpy()

n_train = len(X_train)
n_pred = len(X_pred)

region_train = X_train["REGION"].to_numpy()
region_pred = X_pred["REGION"].to_numpy()
macro_train = X_train["MACRO_REGION"].to_numpy()
macro_pred = X_pred["MACRO_REGION"].to_numpy()
dtype_train = X_train["DISTRICT_TYPE"].to_numpy()
dtype_pred = X_pred["DISTRICT_TYPE"].to_numpy()
assessment_train = X_train["ASSESSMENT_NAME"].to_numpy()
assessment_pred = X_pred["ASSESSMENT_NAME"].to_numpy()
subgroup_train = X_train["SUBGROUP_NAME"].to_numpy()
subgroup_pred = X_pred["SUBGROUP_NAME"].to_numpy()

UNIQUE_REGIONS = sorted(set(region_train) | set(region_pred))
UNIQUE_MACROS = sorted(set(macro_train) | set(macro_pred))
UNIQUE_DTYPES = sorted(set(dtype_train) | set(dtype_pred))
UNIQUE_ASSESSMENTS = sorted(set(assessment_train) | set(assessment_pred))
UNIQUE_SUBGROUP = sorted(set(subgroup_train) | set(subgroup_pred))

print(f"Regions ({len(UNIQUE_REGIONS)}): {UNIQUE_REGIONS}")
print(f"Macros ({len(UNIQUE_MACROS)}): {UNIQUE_MACROS}")
print(f"District types ({len(UNIQUE_DTYPES)}): {UNIQUE_DTYPES}")
print(f"Assessment types ({len(UNIQUE_ASSESSMENTS)}): {UNIQUE_ASSESSMENTS}")
print(f"SUBGROUP ({len(UNIQUE_SUBGROUP)}): {UNIQUE_SUBGROUP}")

oe_global = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    dtype=np.int64,
)
oe_global.fit(np.vstack([cat_train_raw, cat_pred_raw]))
cat_train_oe = oe_global.transform(cat_train_raw) + 1
cat_pred_oe = oe_global.transform(cat_pred_raw) + 1
cat_dims = [len(cats) + 1 for cats in oe_global.categories_]
print(f"Global cat dims: {cat_dims}")


def preprocess_numeric_per_fold(num_train_arr, num_val_arr, num_pred_arr=None):
    """Imputer + power transformer fit on training rows only.
    NaN/inf-clean output guarded against PowerTransformer producing NaN on
    near-constant columns within a partition."""
    imp = SimpleImputer(strategy="median", add_indicator=True)
    pt = PowerTransformer(method="yeo-johnson", standardize=True)

    X_tr = pt.fit_transform(imp.fit_transform(num_train_arr)).astype(np.float32)
    X_va = pt.transform(imp.transform(num_val_arr)).astype(np.float32)

    np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    if num_pred_arr is not None:
        X_te = pt.transform(imp.transform(num_pred_arr)).astype(np.float32)
        np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return X_tr, X_va, X_te
    return X_tr, X_va


# Stratification helpers


def _pool_rare(key_arr, n_folds):
    """Pool classes with fewer than n_folds members into '__rare__' so
    StratifiedKFold doesn't raise on small classes."""
    counts = Counter(key_arr)
    rare = {v for v, c in counts.items() if c < n_folds}
    if not rare:
        return key_arr
    return np.array([v if v not in rare else "__rare__" for v in key_arr])


def stratification_key_for_tier(tier, idx_subset, n_folds):
    """Return a stratification key for the given tier's partition, using the
    next-most-informative categorical that ISN'T the partition key.

    Within an assessment partition, all rows share an assessment name, so
    stratifying on assessment is meaningless — we use SUBGROUP_NAME instead.
    Symmetrically, within a subgroup partition we use ASSESSMENT_NAME.
    """
    if tier == "assessment":
        key = subgroup_train[idx_subset]
    else:
        # global, macro, region, district_type, subgroup all benefit from
        # being balanced across assessments
        key = assessment_train[idx_subset]
    return _pool_rare(key, n_folds)


# Model


class CatEmbeddings(nn.Module):
    def __init__(self, cardinalities, d_token):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d_token) for c in cardinalities])
        for e in self.embs:
            nn.init.kaiming_uniform_(e.weight, a=math.sqrt(5))

    def forward(self, x_cat):
        return torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embs)], dim=1)


class ResNetBlock(nn.Module):
    def __init__(self, d, d_hidden, dropout_first, dropout_second):
        super().__init__()
        self.norm = nn.BatchNorm1d(d)
        self.lin1 = nn.Linear(d, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d)
        self.drop1 = nn.Dropout(dropout_first)
        self.drop2 = nn.Dropout(dropout_second)

    def forward(self, x):
        z = self.norm(x)
        z = F.relu(self.lin1(z))
        z = self.drop1(z)
        z = self.lin2(z)
        z = self.drop2(z)
        return x + z


class TabResNet(nn.Module):
    def __init__(
        self,
        n_num,
        cat_cardinalities,
        d_token=8,
        d=256,
        d_hidden=512,
        n_blocks=3,
        dropout_first=0.25,
        dropout_second=0.0,
    ):
        super().__init__()
        self.cat_emb = (
            CatEmbeddings(cat_cardinalities, d_token) if cat_cardinalities else None
        )
        in_dim = n_num + (len(cat_cardinalities) * d_token if cat_cardinalities else 0)
        self.input_proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList(
            [
                ResNetBlock(d, d_hidden, dropout_first, dropout_second)
                for _ in range(n_blocks)
            ]
        )
        self.head_norm = nn.BatchNorm1d(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x_num, x_cat=None):
        parts = [x_num]
        if self.cat_emb is not None and x_cat is not None:
            parts.append(self.cat_emb(x_cat))
        x = torch.cat(parts, dim=1)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.relu(self.head_norm(x))
        return self.head(x).squeeze(-1)


def make_loader(x_num, x_cat, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(x_num).float(),
        torch.from_numpy(x_cat).long(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )


def train_one_fold(
    model,
    train_loader,
    val_loader,
    lr,
    wd,
    epochs,
    patience,
    device,
    y_std_fold,
    fold_label="",
    verbose=True,
):
    """Returns (best_rmse, best_epoch). best_rmse is in original target units."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(x_num, x_cat), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                preds.append(model(x_num.to(device), x_cat.to(device)).cpu())
                targets.append(y)
        val_rmse = (
            float(torch.sqrt(F.mse_loss(torch.cat(preds), torch.cat(targets))))
            * y_std_fold
        )

        if verbose and epoch % 25 == 0:
            print(
                f"    {fold_label} ep {epoch:3d}  val_rmse={val_rmse:.4f}  "
                f"best={best_rmse:.4f}  stale={bad_epochs}",
                flush=True,
            )

        if val_rmse < best_rmse - 1e-6:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(
                        f"    {fold_label} early stop ep {epoch}  best={best_rmse:.4f}",
                        flush=True,
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_rmse, best_epoch


@torch.no_grad()
def predict(model, x_num, x_cat, device, y_mean, y_std, batch_size=2048):
    """Returns predictions in original units."""
    model.eval()
    loader = make_loader(
        x_num, x_cat, np.zeros(len(x_num), dtype=np.float32), batch_size, shuffle=False
    )
    preds = []
    for xn, xc, _ in loader:
        preds.append(model(xn.to(device), xc.to(device)).cpu())
    return torch.cat(preds).numpy() * y_std + y_mean


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
    if tier == "assessment":
        return (
            np.where(assessment_train == name)[0],
            np.where(assessment_pred == name)[0],
        )
    if tier == "subgroup":
        return (
            np.where(subgroup_train == name)[0],
            np.where(subgroup_pred == name)[0],
        )
    raise ValueError(f"Unknown tier: {tier}")


def search_space_for_tier(tier, trial):
    """Tier-aware search space. Smaller buckets get tighter caps."""

    d_token = trial.suggest_categorical("d_token", [2, 4, 8, 16, 32, 64])
    d = trial.suggest_int("d", 32, 1024, step=32)
    d_hidden_mult = trial.suggest_float("d_hidden_mult", 0.5, 5.0)
    n_blocks = trial.suggest_int("n_blocks", 1, 16)
    batch_size = trial.suggest_categorical(
        "batch_size", [64, 128, 256, 512, 1024, 2048, 4096]
    )

    dropout_first = trial.suggest_float("dropout_first", 0.0, 0.5)
    dropout_second = trial.suggest_float("dropout_second", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-7, 1e-3, log=True)
    return {
        "d_token": d_token,
        "d": d,
        "d_hidden_mult": d_hidden_mult,
        "n_blocks": n_blocks,
        "dropout_first": dropout_first,
        "dropout_second": dropout_second,
        "lr": lr,
        "wd": wd,
        "batch_size": batch_size,
    }


def n_trials_for_tier(tier, n_rows):
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


def maybe_save_best(tier, name, study, tolerance=1e-6):
    """Save params only if study found a strictly better CV score than what's on disk.
    Score is in percentage-space RMSE.
    """
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
def constraints_func(trial):
    d1 = trial.params.get("dropout_first", 0)
    d2 = trial.params.get("dropout_second", 0)
    violation = max(0, d2 - d1)
    return [violation]


def tune_one(tier, name, n_trials=None):
    train_idx_all, _ = get_tier_indices(tier, name)
    n_rows = len(train_idx_all)
    print(f"\n[{tier}/{name}] tuning on {n_rows} rows")

    if n_trials is None:
        n_trials = n_trials_for_tier(tier, n_rows)
    n_inner = n_inner_folds_for_tier(tier)

    num_subset = num_train_raw[train_idx_all]
    cat_subset = cat_train_oe[train_idx_all]

    y_subset_pct = y_np[train_idx_all]
    strat_key = stratification_key_for_tier(tier, train_idx_all, n_inner)

    def objective(trial):
        params = search_space_for_tier(tier, trial)
        kf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED)
        fold_rmses_pct = []

        for fold_i, (tr_i, va_i) in enumerate(kf.split(num_subset, strat_key)):
            X_tr_num, X_va_num = preprocess_numeric_per_fold(
                num_subset[tr_i], num_subset[va_i]
            )
            X_tr_cat = cat_subset[tr_i]
            X_va_cat = cat_subset[va_i]
            y_tr_np = y_subset_pct[tr_i]
            y_va_np = y_subset_pct[va_i]
            y_va_pct_fold = y_subset_pct[va_i]
            y_mean = y_tr_np.mean()
            y_std = y_tr_np.std() + 1e-8
            y_tr_n = ((y_tr_np - y_mean) / y_std).astype(np.float32)
            y_va_n = ((y_va_np - y_mean) / y_std).astype(np.float32)

            model = TabResNet(
                n_num=X_tr_num.shape[1],
                cat_cardinalities=cat_dims,
                d_token=params["d_token"],
                d=params["d"],
                d_hidden=int(params["d"] * params["d_hidden_mult"]),
                n_blocks=params["n_blocks"],
                dropout_first=params["dropout_first"],
                dropout_second=params["dropout_second"],
            ).to(DEVICE)

            tr_loader = make_loader(
                X_tr_num, X_tr_cat, y_tr_n, params["batch_size"], shuffle=True
            )
            va_loader = make_loader(
                X_va_num, X_va_cat, y_va_n, params["batch_size"], shuffle=False
            )

            _, _ = train_one_fold(
                model,
                tr_loader,
                va_loader,
                lr=params["lr"],
                wd=params["wd"],
                epochs=EPOCHS_TUNING,
                patience=PATIENCE_TUNING,
                device=DEVICE,
                y_std_fold=y_std,
                fold_label=f"T{trial.number}/F{fold_i+1}",
                verbose=True,
            )

            val_preds_pct = predict(model, X_va_num, X_va_cat, DEVICE, y_mean, y_std)

            if not np.isfinite(val_preds_pct).all():
                print(f"    T{trial.number}/F{fold_i+1} non-finite preds; failing")
                del model, tr_loader, va_loader
                torch.cuda.empty_cache()
                gc.collect()
                return float("inf")

            fold_rmse_pct = float(
                np.sqrt(np.mean((y_va_pct_fold - val_preds_pct) ** 2))
            )
            fold_rmses_pct.append(fold_rmse_pct)

            del model, tr_loader, va_loader
            torch.cuda.empty_cache()
            gc.collect()

            if fold_i == 0 and fold_rmse_pct > y_subset_pct.std() * 1.5:
                return fold_rmse_pct

        return float(np.mean(fold_rmses_pct))

    if tier == "global":
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                str(LOGS_DIR / "tabresnet_journal_storage.log")
            )
        )
    else:
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                str(LOGS_DIR / f"resnet_journal_{tier}_{_safe(name)}.log")
            )
        )
    study = optuna.create_study(
        study_name=f"resnet_{tier}_{name}",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(seed=SEED, n_startup_trials=5),
    )

    marker = WARMSTART_DIR / f"{tier}_{_safe(name)}.done"
    if not marker.exists():
        warmstarts = [_adapt_universal_warmstart(tier)]
        if tier == "global":
            warmstarts.extend(RESNET_GLOBAL_ONLY_WARMSTARTS)
        elif GLOBAL_BEST_PARAMS is not None:
            warmstarts.append(dict(GLOBAL_BEST_PARAMS))
        for params in warmstarts:
            study.enqueue_trial(params)
        marker.write_text(f"enqueued {len(warmstarts)} warm-start trials\n")
        print(
            f"[{tier}/{name}] enqueued {len(warmstarts)} warm-start trials (first run)"
        )
    else:
        print(f"[{tier}/{name}] warm-start already done, skipping")

    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True
    )

    saved = maybe_save_best(tier, name, study)
    print(f"[{tier}/{name}] best CV RMSE (pct): {study.best_value:.4f}")
    return study.best_params, study.best_value, saved


# Per-fold expert fit
def fit_one_expert(tier, name, train_idx, val_idx, pred_idx, fold_idx=None):
    """Fit (or load from cache) a tier-specific ResNet on its subset of
    one outer fold. Returns predictions in original units.

    If fold_idx is provided, predictions are cached to disk keyed by
    (tier, name, fold, params_hash). Subsequent runs with identical params
    skip retraining. Pass fold_idx=None to disable caching (e.g., during
    submit_global which has its own seeded ensembling).
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

    X_tr_num_raw = num_train_raw[train_idx]
    X_va_num_raw = num_train_raw[val_idx]
    X_te_num_raw = num_pred_raw[pred_idx]

    X_tr_cat = cat_train_oe[train_idx]
    X_va_cat = cat_train_oe[val_idx]
    X_te_cat = cat_pred_oe[pred_idx]

    y_tr = y_np[train_idx]
    y_va = y_np[val_idx]

    X_tr_num, X_va_num, X_te_num = preprocess_numeric_per_fold(
        X_tr_num_raw, X_va_num_raw, X_te_num_raw
    )

    y_mean = y_tr.mean()
    y_std = y_tr.std() + 1e-8
    y_tr_n = ((y_tr - y_mean) / y_std).astype(np.float32)
    y_va_n = ((y_va - y_mean) / y_std).astype(np.float32)

    d = params["d"]
    model = TabResNet(
        n_num=X_tr_num.shape[1],
        cat_cardinalities=cat_dims,
        d_token=params["d_token"],
        d=d,
        d_hidden=int(d * params["d_hidden_mult"]),
        n_blocks=params["n_blocks"],
        dropout_first=params["dropout_first"],
        dropout_second=params["dropout_second"],
    ).to(DEVICE)

    tr_loader = make_loader(
        X_tr_num, X_tr_cat, y_tr_n, params["batch_size"], shuffle=True
    )
    va_loader = make_loader(
        X_va_num, X_va_cat, y_va_n, params["batch_size"], shuffle=False
    )

    _, best_epoch = train_one_fold(
        model,
        tr_loader,
        va_loader,
        lr=params["lr"],
        wd=params["wd"],
        epochs=EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        y_std_fold=y_std,
        fold_label=f"{tier}/{name}",
        verbose=True,
    )

    val_preds = predict(model, X_va_num, X_va_cat, DEVICE, y_mean, y_std)
    # >>> ROCM BUG CHECK <
    import torch.nn.functional as F

    y_va_n_check = ((y_va - y_mean) / y_std).astype(np.float32)
    va_loader_check = make_loader(
        X_va_num, X_va_cat, y_va_n_check, params["batch_size"], shuffle=False
    )
    model.eval()
    pn, tn = [], []
    with torch.no_grad():
        for xn, xc, yy in va_loader_check:
            pn.append(model(xn.to(DEVICE), xc.to(DEVICE)).cpu())
            tn.append(yy)
    val_rmse_intrain_path = (
        float(torch.sqrt(F.mse_loss(torch.cat(pn).view(-1), torch.cat(tn).view(-1))))
        * y_std
    )
    val_rmse_predict_path = float(np.sqrt(np.mean((y_va - val_preds) ** 2)))
    corr = float(np.corrcoef(y_va, val_preds)[0, 1])
    sorted_rmse = float(np.sqrt(np.mean((np.sort(y_va) - np.sort(val_preds)) ** 2)))
    print(
        f"  ROCM CHECK [{tier}/{name}] fold {fold_idx}: "
        f"intrain_path={val_rmse_intrain_path:.4f}  "
        f"predict_path={val_rmse_predict_path:.4f}  "
        f"sorted={sorted_rmse:.4f}  "
        f"corr={corr:.4f}",
        flush=True,
    )
    # >>> END <
    pred_preds = predict(model, X_te_num, X_te_cat, DEVICE, y_mean, y_std)

    val_rmse = float(np.sqrt(np.mean((y_va - val_preds) ** 2)))

    del model, tr_loader, va_loader
    torch.cuda.empty_cache()
    gc.collect()

    if fold_idx is not None:
        np.savez_compressed(
            cache_path,
            val_preds=val_preds,
            pred_preds=pred_preds,
            val_idx=np.asarray(val_idx, dtype=np.int64),
            pred_idx=np.asarray(pred_idx, dtype=np.int64),
            params_hash=cache_key,
            params_json=json.dumps(params),
            val_rmse=np.float64(val_rmse),
            best_epoch=np.int64(best_epoch),
            cache_version=np.int64(CACHE_VERSION),
        )
        print(
            f"  [{tier}/{name}] fold {fold_idx} cached (hash {cache_key}, val_rmse={val_rmse:.4f}, best_epoch={best_epoch})"
        )

    return val_preds, pred_preds


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
    fold_idx,  # new
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


def _save_diagnostics(
    tiers,
    oof_global,
    resid,
    keys,
    test_stack_arrays,
    version_tag,
):
    """Write all diagnostic artefacts after stacking completes.

    Combines everything from the standalone tuning script:
      - per-tier OOF RMSE / MAE
      - residual correlation matrix
      - partition-level improvement breakdown
      - per-tier-partition sizes
      - per-row OOF parquet (for offline slice analysis)
      - band-by-band RMSE table
      - worst-100 predictions CSV
      - worst-100 categorical concentration analysis
      - per-school mean/max abs residual CSV
      - ridge meta-learner sanity check
      - oof_stack + test_stack saved as npz
      - JSON summary of everything above
    """

    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostic_path = Path(f"baseline_diagnostic_resnet_{version_tag}.json")
    diagnostic = {
        "run_timestamp": datetime.datetime.now().isoformat(),
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
        "per_tier_partition_sizes": {
            "macro": {str(m): int((macro_train == m).sum()) for m in UNIQUE_MACROS},
            "region": {str(r): int((region_train == r).sum()) for r in UNIQUE_REGIONS},
            "district_type": {
                str(d): int((dtype_train == d).sum()) for d in UNIQUE_DTYPES
            },
            "assessment": {
                str(a): int((assessment_train == a).sum()) for a in UNIQUE_ASSESSMENTS
            },
            "subgroup": {
                str(s): int((subgroup_train == s).sum()) for s in UNIQUE_SUBGROUP
            },
        },
    }

    # Per-partition improvement: global RMSE vs tier expert RMSE on same rows
    partition_breakdowns = {}
    for tier_name, oof_arr, membership in [
        ("macro", dict(tiers)["macro"], macro_train),
        ("region", dict(tiers)["region"], region_train),
        ("district_type", dict(tiers)["district_type"], dtype_train),
        ("assessment", dict(tiers)["assessment"], assessment_train),
        ("subgroup", dict(tiers)["subgroup"], subgroup_train),
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

    # Per-row OOF frame for offline slice analysis
    oof_dict = dict(tiers)
    diagnostic_df = pl.DataFrame(
        {
            "orig_idx": np.arange(n_train),
            "y_true": y_np,
            "pred_global": oof_global,
            "pred_macro": oof_dict["macro"],
            "pred_region": oof_dict["region"],
            "pred_district_type": oof_dict["district_type"],
            "pred_assessment": oof_dict["assessment"],
            "pred_subgroup": oof_dict["subgroup"],
            "resid_global": y_np - oof_global,
            "resid_macro": y_np - oof_dict["macro"],
            "resid_region": y_np - oof_dict["region"],
            "resid_district_type": y_np - oof_dict["district_type"],
            "resid_assessment": y_np - oof_dict["assessment"],
            "resid_subgroup": y_np - oof_dict["subgroup"],
            "ASSESSMENT_NAME": assessment_train,
            "SUBGROUP_NAME": subgroup_train,
            "REGION": region_train,
            "DISTRICT_TYPE": dtype_train,
            "MACRO_REGION": macro_train,
        }
    )
    oof_parquet_path = f"baseline_oof_{version_tag}.parquet"
    diagnostic_df.write_parquet(oof_parquet_path)
    diagnostic["oof_parquet_path"] = oof_parquet_path
    print(f"  OOF parquet: {oof_parquet_path}")

    # Band-by-band RMSE on global OOF predictions
    residuals_global = y_np - oof_global
    print("\nErrors by target band (global OOF):")
    band_stats = {}
    for lo, hi in [(0, 5), (5, 25), (25, 50), (50, 75), (75, 95), (95, 100)]:
        mask = (y_np >= lo) & (y_np <= hi)
        if mask.sum() > 0:
            band_rmse = float(np.sqrt(np.mean(residuals_global[mask] ** 2)))
            mean_resid = float(residuals_global[mask].mean())
            print(
                f"  y in [{lo:3d}, {hi:3d}]: n={mask.sum():6d}  "
                f"RMSE={band_rmse:.3f}  mean_resid={mean_resid:+.3f}"
            )
            band_stats[f"{lo}_{hi}"] = {
                "n": int(mask.sum()),
                "rmse": band_rmse,
                "mean_residual": mean_resid,
            }
    diagnostic["band_rmse"] = band_stats

    # Worst-100 predictions (by global OOF abs residual)
    abs_residuals = np.abs(residuals_global)
    worst_idx = np.argsort(abs_residuals)[-100:][::-1]

    residual_df = X_train.with_columns(
        [
            pl.Series("y_true", y_np),
            pl.Series("y_pred", oof_global),
            pl.Series("residual", residuals_global),
            pl.Series("abs_residual", abs_residuals),
        ]
    ).with_row_index("orig_idx")

    worst_df = residual_df[worst_idx.tolist()]

    print("\nResidual distribution (global OOF):")
    print(f"  mean abs:   {abs_residuals.mean():.3f}")
    print(f"  median abs: {float(np.median(abs_residuals)):.3f}")
    print(f"  p90:        {float(np.percentile(abs_residuals, 90)):.3f}")
    print(f"  p99:        {float(np.percentile(abs_residuals, 99)):.3f}")
    print(f"  max:        {abs_residuals.max():.3f}")
    diagnostic["global_oof_residual_stats"] = {
        "mean_abs": float(abs_residuals.mean()),
        "median_abs": float(np.median(abs_residuals)),
        "p90": float(np.percentile(abs_residuals, 90)),
        "p99": float(np.percentile(abs_residuals, 99)),
        "max": float(abs_residuals.max()),
    }

    print("\nTop 10 worst global OOF predictions:")
    print(
        worst_df.select(
            [
                "orig_idx",
                "y_true",
                "y_pred",
                "residual",
                "SCHOOL",
                "DISTRICT",
                "COUNTY",
                "DISTRICT_TYPE",
                "ASSESSMENT_NAME",
                "SUBGROUP_NAME",
            ]
        ).head(10)
    )
    residuals_full_path = f"residuals_full_{version_tag}.csv"
    residual_df.write_csv(residuals_full_path)
    worst_csv_path = f"worst_residuals_{version_tag}.csv"
    worst_df.write_csv(worst_csv_path)
    diagnostic["worst_residuals_path"] = worst_csv_path
    diagnostic["residuals_full_path"] = residuals_full_path
    print(f"  Full residuals CSV: {residuals_full_path}")
    print(f"  Worst-100 CSV: {worst_csv_path}")

    # Categorical concentration in worst-100 vs full train
    print("\nWorst-100 vs all-train, categorical concentration:")
    concentration = {}
    for col in [
        "DISTRICT_TYPE",
        "ASSESSMENT_NAME",
        "SUBGROUP_NAME",
        "school_type",
        "REGION",
        "MACRO_REGION",
    ]:
        full_dist = (
            X_train[col]
            .value_counts(sort=True)
            .with_columns((pl.col("count") / pl.col("count").sum()).alias("full_pct"))
        )
        worst_dist = (
            worst_df[col]
            .value_counts(sort=True)
            .with_columns((pl.col("count") / pl.col("count").sum()).alias("worst_pct"))
        )
        joined = (
            worst_dist.join(full_dist.select([col, "full_pct"]), on=col, how="left")
            .with_columns((pl.col("worst_pct") - pl.col("full_pct")).alias("over_rep"))
            .sort("over_rep", descending=True)
        )
        print(f"\n  {col} (top 5 over-represented in worst 100):")
        print(joined.head(5))
        concentration[col] = joined.head(5).to_dicts()
    diagnostic["worst_100_concentration"] = concentration

    # Per-school error aggregation
    school_errors = (
        residual_df.group_by("SCHOOL")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("abs_residual").mean().alias("mean_abs_resid"),
                pl.col("abs_residual").max().alias("max_abs_resid"),
            ]
        )
        .filter(pl.col("n") >= 5)
        .sort("mean_abs_resid", descending=True)
    )
    school_errors_path = f"school_errors_{version_tag}.csv"
    school_errors.write_csv(school_errors_path)
    diagnostic["school_errors_path"] = school_errors_path
    print(f"\nWorst 20 schools by mean abs residual (n>=5 rows):")
    print(school_errors.head(20))

    # Ridge meta-learner sanity check + stack npz
    oof_stack = np.column_stack([oof for _, oof in tiers])
    test_stack = np.column_stack(test_stack_arrays)

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
    diagnostic["ridge_coefficients"] = {
        name: float(c) for (name, _), c in zip(tiers, ridge.coef_)
    }
    diagnostic["ridge_intercept"] = float(ridge.intercept_)
    diagnostic["ridge_alpha"] = float(ridge.alpha_)

    stack_npz_path = f"baseline_stack_{version_tag}.npz"
    np.savez_compressed(
        stack_npz_path,
        oof_stack=oof_stack,
        test_stack=test_stack,
        y=y_np,
        tier_names=np.array([name for name, _ in tiers]),
    )
    diagnostic["stack_npz_path"] = stack_npz_path
    print(f"  Stack npz: {stack_npz_path}")

    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    print(f"  Diagnostic JSON: {diagnostic_path}")


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
            "or search space — use --force after such changes."
        )
        return

    # Outer split is stratified on assessment. The assessment-tier experts get
    # a slightly favorable val distribution out of this (their training rows
    # are guaranteed to be proportionally represented), but a single outer
    # split is required for OOF compatibility across tiers, and stratifying
    # on assessment is the right tradeoff for everything else.
    outer_strat_key = _pool_rare(assessment_train, N_OUTER_FOLDS)

    kfold = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)

    oof_global = np.zeros(n_train)
    oof_macro = np.zeros(n_train)
    oof_region = np.zeros(n_train)
    oof_dtype = np.zeros(n_train)
    oof_group = np.zeros(n_train)
    oof_assessment = np.zeros(n_train)

    test_global = np.zeros(n_pred)
    test_macro = np.zeros(n_pred)
    test_region = np.zeros(n_pred)
    test_dtype = np.zeros(n_pred)
    test_group = np.zeros(n_pred)
    test_assessment = np.zeros(n_pred)

    filled_macro_tr = np.zeros(n_train, dtype=bool)
    filled_region_tr = np.zeros(n_train, dtype=bool)
    filled_dtype_tr = np.zeros(n_train, dtype=bool)
    filled_group_tr = np.zeros(n_train, dtype=bool)
    filled_assessment_tr = np.zeros(n_train, dtype=bool)

    filled_macro_te = np.zeros(n_pred, dtype=bool)
    filled_region_te = np.zeros(n_pred, dtype=bool)
    filled_dtype_te = np.zeros(n_pred, dtype=bool)
    filled_group_te = np.zeros(n_pred, dtype=bool)
    filled_assessment_te = np.zeros(n_pred, dtype=bool)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kfold.split(num_train_raw, outer_strat_key)
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
            UNIQUE_SUBGROUP,
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
    assert filled_group_tr.all(), "Some train rows missed group OOF"
    assert filled_assessment_tr.all(), "Some train rows missed assessment OOF"

    assert filled_macro_te.all(), "Some test rows missed macro pred"
    assert filled_region_te.all(), "Some test rows missed region pred"
    assert filled_dtype_te.all(), "Some test rows missed district_type pred"
    assert filled_group_te.all(), "Some test rows missed group OOF"
    assert filled_assessment_te.all(), "Some test rows missed assessment OOF"
    assert filled_assessment_te.all(), "Some test rows missed assessment OOF"

    # Build these before calling _save_diagnostics
    tiers = [
        ("global", oof_global),
        ("macro", oof_macro),
        ("region", oof_region),
        ("district_type", oof_dtype),
        ("subgroup", oof_group),
        ("assessment", oof_assessment),
    ]
    resid = {name: y_np - oof for name, oof in tiers}
    keys = list(resid.keys())

    # Print the per-tier RMSE summary before delegating to diagnostics
    print("\nOOF RMSE by tier (percentage scale):")
    for name, oof in tiers:
        rmse = float(np.sqrt(np.mean((y_np - oof) ** 2)))
        print(f"  {name:14s}: {rmse:.4f}")

    print("\nResidual correlations:")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            corr = np.corrcoef(resid[a], resid[b])[0, 1]
            print(f"  {a:14s} <-> {b:14s}: {corr:.4f}")

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
            test_group,
            test_assessment,
        ],
        version_tag=version_tag,
    )

    np.savez_compressed(
        ARTIFACTS_PATH,
        oof_global=oof_global,
        oof_macro=oof_macro,
        oof_region=oof_region,
        oof_district_type=oof_dtype,
        oof_group=oof_group,
        oof_assessment=oof_assessment,
        test_global=test_global,
        test_macro=test_macro,
        test_region=test_region,
        test_district_type=test_dtype,
        test_group=test_group,
        test_assessment=test_assessment,
        y=y_np,
    )

    print(f"\nSaved {ARTIFACTS_PATH}")


# Global submission


def submit_global(force=False, n_seeds=5):
    """Train n_seeds global ResNets on (almost) all training data and write
    a submission CSV averaging their predictions. Skips if current best CV
    score hasn't improved since the last submission.
    """
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
        prev_n_seeds = submitted.get("n_seeds", 1)
        tolerance = 1e-6
        if current_best >= prev - tolerance and n_seeds <= prev_n_seeds:
            print(
                f"[submit-global] no improvement since last submission: "
                f"current={current_best:.6f} vs submitted={prev:.6f}, "
                f"n_seeds={n_seeds} vs prev={prev_n_seeds}"
            )
            print("  Skipping submission. Pass --force to regenerate anyway.")
            return

    print(f"[submit-global] training {n_seeds} global ResNet seeds...")
    print(f"  current best CV RMSE: {current_best:.4f}")

    # Stratified 95/5 split on assessment, with rare-pooling.
    holdout_strat_key = _pool_rare(assessment_train, 20)
    tr_idx, va_idx = train_test_split(
        np.arange(n_train),
        test_size=0.05,
        stratify=holdout_strat_key,
        random_state=SEED,
    )
    pred_idx = np.arange(n_pred)

    val_preds_accum = np.zeros(len(va_idx), dtype=np.float64)
    test_preds_accum = np.zeros(n_pred, dtype=np.float64)
    per_seed_val_rmse = []

    for seed_i in range(n_seeds):
        seed = SEED + seed_i
        print(f"\n  --- seed {seed_i + 1}/{n_seeds} (value={seed}) ---")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        val_preds, test_preds = fit_one_expert(tier, name, tr_idx, va_idx, pred_idx)
        seed_rmse = float(np.sqrt(np.mean((y_np[va_idx] - val_preds) ** 2)))
        per_seed_val_rmse.append(seed_rmse)
        print(f"  seed {seed_i + 1} holdout RMSE: {seed_rmse:.4f}")

        val_preds_accum += val_preds
        test_preds_accum += test_preds

    val_preds_avg = val_preds_accum / n_seeds
    test_preds_avg = test_preds_accum / n_seeds
    ensemble_rmse = float(np.sqrt(np.mean((y_np[va_idx] - val_preds_avg) ** 2)))
    test_preds_avg = np.clip(test_preds_avg, 0, 100)
    print(
        f"\n[submit-global] per-seed holdout RMSEs: "
        f"{[f'{r:.4f}' for r in per_seed_val_rmse]}"
    )
    print(
        f"[submit-global] mean of per-seed RMSEs: " f"{np.mean(per_seed_val_rmse):.4f}"
    )
    print(f"[submit-global] ensemble holdout RMSE:  {ensemble_rmse:.4f}")
    print(
        f"[submit-global] ensemble gain over single seed: "
        f"{np.mean(per_seed_val_rmse) - ensemble_rmse:.4f}"
    )

    submission = X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", test_preds_avg))
    submission.write_csv(GLOBAL_SUBMISSION_PATH)
    print(
        f"[submit-global] wrote {GLOBAL_SUBMISSION_PATH} "
        f"({len(test_preds_avg)} rows)"
    )

    submitted_path.write_text(
        json.dumps(
            {
                "best_value": current_best,
                "ensemble_holdout_rmse": ensemble_rmse,
                "per_seed_rmse": per_seed_val_rmse,
                "n_seeds": n_seeds,
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
            "assessment",
            "subgroup",
        ],
    )
    p.add_argument("--name", help="Partition value; omit to loop all in tier")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds for submit-global ensembling (default: 5)",
    )
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
                "assessment": UNIQUE_ASSESSMENTS,
                "subgroup": UNIQUE_SUBGROUP,
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
    else:  # submit-global
        submit_global(force=args.force)


if __name__ == "__main__":
    main()
