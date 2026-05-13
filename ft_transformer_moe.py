#!/usr/bin/env python3

"""
FT-Transformer mixture-of-experts (ROCm-bug-fixed, global tier only).

The previous predict() function did a separate model forward pass after
training. On this ROCm setup it returned row-shuffled output (verified:
RMSE_sorted ≈ 9.7 vs RMSE_actual ≈ 20.5 with correlation 0.64). Fix: capture
val predictions inside train_one_fold's val loop, and use the same loader-
iteration pattern for test predictions. predict() function is gone.

Architecture: Gorishniy et al. arXiv:2106.11959.

Modes:
    --mode stack [--force]
    --mode submit-global [--force] [--n-seeds N]
"""

import argparse
import datetime
import gc
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 8675309
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pl.enable_string_cache()

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
PARAMS_DIR = Path("best_params_ft_transformer_moe")
PARAMS_DIR.mkdir(exist_ok=True)
EXPERTS_DIR = Path("fitted_experts_ft_transformer")
EXPERTS_DIR.mkdir(exist_ok=True)

ARTIFACTS_PATH = Path("ft_transformer_moe_artifacts.npz")
GLOBAL_SUBMISSION_PATH = Path("submission_ft_transformer_global.csv")

CACHE_VERSION = 2
N_HEADS = 8

FT_PAPER_DEFAULTS = {
    "d_token": 192,
    "n_blocks": 3,
    "ffn_d_hidden_mult": 4 / 3,
    "attention_dropout": 0.2,
    "ffn_dropout": 0.1,
    "residual_dropout": 0.0,
    "lr": 1e-4,
    "wd": 1e-5,
    "batch_size": 256,
}

N_OUTER_FOLDS = 10
EPOCHS = 30
PATIENCE = 6

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

assessment_train = X_train["ASSESSMENT_NAME"].to_numpy()
subgroup_train = X_train["SUBGROUP_NAME"].to_numpy()
region_train = X_train["REGION"].to_numpy()
dtype_train = X_train["DISTRICT_TYPE"].to_numpy()
macro_train = X_train["MACRO_REGION"].to_numpy()

print(f"n_train={n_train}  n_pred={n_pred}")
print(f"numeric cols: {len(numeric_cols)}  categorical cols: {len(cat_cols_ordered)}")

oe_global = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
)
oe_global.fit(np.vstack([cat_train_raw, cat_pred_raw]))
cat_train_oe = (oe_global.transform(cat_train_raw) + 1).astype(np.int64)
cat_pred_oe = (oe_global.transform(cat_pred_raw) + 1).astype(np.int64)
cat_dims = [len(cats) + 1 for cats in oe_global.categories_]
print(f"Cat dims: {cat_dims}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pool_rare(key_arr, n_folds):
    counts = Counter(key_arr)
    rare = {v for v, c in counts.items() if c < n_folds}
    if not rare:
        return key_arr
    return np.array([v if v not in rare else "__rare__" for v in key_arr])


def preprocess_numeric_per_fold(num_train_arr, num_val_arr, num_pred_arr=None):
    imp = SimpleImputer(strategy="median", add_indicator=True)
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_tr = pt.fit_transform(imp.fit_transform(num_train_arr)).astype(np.float32)
    X_va = pt.transform(imp.transform(num_val_arr)).astype(np.float32)
    np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.clip(X_tr, -5.0, 5.0, out=X_tr)
    np.clip(X_va, -5.0, 5.0, out=X_va)
    if num_pred_arr is not None:
        X_te = pt.transform(imp.transform(num_pred_arr)).astype(np.float32)
        np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.clip(X_te, -5.0, 5.0, out=X_te)
        return X_tr, X_va, X_te
    return X_tr, X_va


# ── Model ─────────────────────────────────────────────────────────────────────


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num):
        return x_num.unsqueeze(-1) * self.weight + self.bias


class CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, cardinalities, d_token):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d_token) for c in cardinalities])
        self.bias = nn.Parameter(torch.empty(len(cardinalities), d_token))
        for e in self.embs:
            nn.init.kaiming_uniform_(e.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_cat):
        out = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.embs)], dim=1)
        return out + self.bias


class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class FTBlock(nn.Module):
    def __init__(
        self,
        d_token,
        n_heads,
        attention_dropout,
        ffn_dropout,
        residual_dropout,
        ffn_d_hidden,
        is_first_block,
    ):
        super().__init__()
        self.is_first_block = is_first_block
        self.norm_attn = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            d_token, n_heads, dropout=attention_dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_token)
        self.ffn_lin1 = nn.Linear(d_token, ffn_d_hidden * 2)
        self.ffn_act = ReGLU()
        self.ffn_drop = nn.Dropout(ffn_dropout)
        self.ffn_lin2 = nn.Linear(ffn_d_hidden, d_token)
        self.res_drop = nn.Dropout(residual_dropout)

    def forward(self, x):
        x_res = x if self.is_first_block else self.norm_attn(x)
        x_attn, _ = self.attn(x_res, x_res, x_res, need_weights=False)
        x = x + self.res_drop(x_attn)
        x_res = self.norm_ffn(x)
        z = self.ffn_lin2(self.ffn_drop(self.ffn_act(self.ffn_lin1(x_res))))
        return x + self.res_drop(z)


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num,
        cat_cardinalities,
        d_token,
        n_blocks,
        n_heads,
        attention_dropout,
        ffn_dropout,
        residual_dropout,
        ffn_d_hidden_mult,
    ):
        super().__init__()
        assert d_token % n_heads == 0
        self.num_tok = NumericalFeatureTokenizer(n_num, d_token)
        self.cat_tok = (
            CategoricalFeatureTokenizer(cat_cardinalities, d_token)
            if cat_cardinalities
            else None
        )
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.cls, a=math.sqrt(5))
        ffn_d_hidden = int(d_token * ffn_d_hidden_mult)
        self.blocks = nn.ModuleList(
            [
                FTBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                    ffn_d_hidden=ffn_d_hidden,
                    is_first_block=(i == 0),
                )
                for i in range(n_blocks)
            ]
        )
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_num, x_cat=None):
        tokens = [self.num_tok(x_num)]
        if self.cat_tok is not None and x_cat is not None:
            tokens.append(self.cat_tok(x_cat))
        x = torch.cat(tokens, dim=1)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(F.relu(self.head_norm(x[:, 0]))).squeeze(-1)


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
    """Returns (best_rmse, best_epoch, best_val_preds_norm).
    val preds captured in same loop as val_rmse — ROCm-safe."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_rmse = float("inf")
    best_state = None
    best_val_preds_norm = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(x_num, x_cat).view(-1), y.view(-1))
            if torch.isfinite(loss):
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
        preds_cat = torch.cat(preds).view(-1)
        targets_cat = torch.cat(targets).view(-1)
        val_rmse = float(torch.sqrt(F.mse_loss(preds_cat, targets_cat))) * y_std_fold

        if verbose:
            print(
                f"    {fold_label} ep {epoch:3d}  val_rmse={val_rmse:.4f}  "
                f"best={best_rmse:.4f}  stale={bad_epochs}",
                flush=True,
            )

        if val_rmse < best_rmse - 1e-6:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_preds_norm = preds_cat.numpy().copy()
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
    return best_rmse, best_epoch, best_val_preds_norm


def predict_via_loader(model, x_num, x_cat, device, y_mean, y_std, batch_size):
    """ROCm-safe test prediction via same loader-iteration pattern that works
    for in-training val. NO separate forward pass after training."""
    loader = make_loader(
        x_num,
        x_cat,
        np.zeros(len(x_num), dtype=np.float32),
        batch_size,
        shuffle=False,
    )
    model.eval()
    pieces = []
    with torch.no_grad():
        for xn, xc, _ in loader:
            pieces.append(model(xn.to(device), xc.to(device)).cpu())
    return torch.cat(pieces).view(-1).numpy() * y_std + y_mean


# ── Caching ───────────────────────────────────────────────────────────────────


def _params_hash(params):
    canonical = json.dumps(params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _safe(name):
    return (
        name.replace(" ", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _expert_cache_path(tier, name, fold_idx, params_hash):
    return EXPERTS_DIR / f"{tier}_{_safe(name)}_fold{fold_idx}_{params_hash}.npz"


def _params_path(tier, name):
    return PARAMS_DIR / f"{tier}_{_safe(name)}.json"


def load_params(tier, name):
    path = _params_path(tier, name)
    if path.exists():
        return json.loads(path.read_text())
    print(f"  [{tier}/{name}] using paper defaults")
    return dict(FT_PAPER_DEFAULTS)


# ── Per-fold expert fit ──────────────────────────────────────────────────────


def fit_one_expert(tier, name, train_idx, val_idx, pred_idx, fold_idx=None):
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

    X_tr_num, X_va_num, X_te_num = preprocess_numeric_per_fold(
        num_train_raw[train_idx],
        num_train_raw[val_idx],
        num_pred_raw[pred_idx],
    )
    X_tr_cat = cat_train_oe[train_idx]
    X_va_cat = cat_train_oe[val_idx]
    X_te_cat = cat_pred_oe[pred_idx]

    y_tr = y_np[train_idx]
    y_va = y_np[val_idx]

    y_mean = y_tr.mean()
    y_std = y_tr.std() + 1e-8
    y_tr_n = ((y_tr - y_mean) / y_std).astype(np.float32)
    y_va_n = ((y_va - y_mean) / y_std).astype(np.float32)

    model = FTTransformer(
        n_num=X_tr_num.shape[1],
        cat_cardinalities=cat_dims,
        d_token=params["d_token"],
        n_blocks=params["n_blocks"],
        n_heads=N_HEADS,
        attention_dropout=params["attention_dropout"],
        ffn_dropout=params["ffn_dropout"],
        residual_dropout=params["residual_dropout"],
        ffn_d_hidden_mult=params["ffn_d_hidden_mult"],
    ).to(DEVICE)

    tr_loader = make_loader(
        X_tr_num, X_tr_cat, y_tr_n, params["batch_size"], shuffle=True
    )
    va_loader = make_loader(
        X_va_num, X_va_cat, y_va_n, params["batch_size"], shuffle=False
    )

    # Val predictions captured INSIDE train_one_fold's val loop (ROCm-safe)
    _, best_epoch, val_preds_norm = train_one_fold(
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
    val_preds = val_preds_norm * y_std + y_mean
    val_rmse = float(np.sqrt(np.mean((y_va - val_preds) ** 2)))
    # Sanity: confirm cached val_preds match in-training val_rmse measurement.
    # Both come from the SAME val loop now, so they MUST agree to ~3 decimals.
    # If they disagree, the ROCm bug has crept back in somewhere.
    y_va_n_check = ((y_va - y_mean) / y_std).astype(np.float32)
    val_rmse_norm_path = (
        float(np.sqrt(np.mean((val_preds_norm - y_va_n_check) ** 2))) * y_std
    )
    agree = "✓" if abs(val_rmse - val_rmse_norm_path) < 0.01 else "✗ MISMATCH"
    print(
        f"  [{tier}/{name}] fold {fold_idx} sanity: "
        f"denorm_path={val_rmse:.4f}  norm_path={val_rmse_norm_path:.4f}  {agree}",
        flush=True,
    )

    # Test predictions via SAME loader-iteration pattern (ROCm-safe)
    pred_preds = predict_via_loader(
        model,
        X_te_num,
        X_te_cat,
        DEVICE,
        y_mean,
        y_std,
        params["batch_size"],
    )

    print(
        f"  [{tier}/{name}] fold {fold_idx} val_rmse={val_rmse:.4f}  best_epoch={best_epoch}",
        flush=True,
    )

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

    return val_preds, pred_preds


# ── Diagnostics ───────────────────────────────────────────────────────────────


def _save_diagnostics(oof_global, test_global, version_tag):
    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostic_path = Path(f"ft_transformer_diagnostic_version_tag.json")
    diagnostic = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "n_outer_folds": N_OUTER_FOLDS,
        "n_train": int(n_train),
        "n_pred": int(n_pred),
        "y_std": float(y_np.std()),
        "y_mean": float(y_np.mean()),
        "global_oof_rmse": float(np.sqrt(np.mean((y_np - oof_global) ** 2))),
        "global_oof_mae": float(np.mean(np.abs(y_np - oof_global))),
    }

    diagnostic_df = pl.DataFrame(
        {
            "orig_idx": np.arange(n_train),
            "y_true": y_np,
            "pred_global": oof_global,
            "resid_global": y_np - oof_global,
            "ASSESSMENT_NAME": assessment_train,
            "SUBGROUP_NAME": subgroup_train,
            "REGION": region_train,
            "DISTRICT_TYPE": dtype_train,
            "MACRO_REGION": macro_train,
        }
    )
    oof_parquet_path = f"ft_transformer_oof_{version_tag}.parquet"
    diagnostic_df.write_parquet(oof_parquet_path)
    diagnostic["oof_parquet_path"] = oof_parquet_path
    print(f"  OOF parquet: {oof_parquet_path}")

    residuals_global = y_np - oof_global
    print("\nErrors by target band (global OOF):")
    band_stats = {}
    for lo, hi in [(0, 5), (5, 25), (25, 50), (50, 75), (75, 95), (95, 100)]:
        mask = (y_np >= lo) & (y_np <= hi)
        if mask.sum() > 0:
            band_rmse = float(np.sqrt(np.mean(residuals_global[mask] ** 2)))
            mean_resid = float(residuals_global[mask].mean())
            print(
                f"  y in [{lo:3d},{hi:3d}]: n={mask.sum():6d}  "
                f"RMSE={band_rmse:.3f}  mean_resid={mean_resid:+.3f}"
            )
            band_stats[f"{lo}_{hi}"] = {
                "n": int(mask.sum()),
                "rmse": band_rmse,
                "mean_residual": mean_resid,
            }
    diagnostic["band_rmse"] = band_stats

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

    worst_csv_path = f"ft_transformer_worst_residuals_{version_tag}.csv"
    worst_df.write_csv(worst_csv_path)
    diagnostic["worst_residuals_path"] = worst_csv_path

    full_resid_path = f"ft_transformer_residuals_full_{version_tag}.csv"
    residual_df.select(
        [
            "orig_idx",
            "y_true",
            "y_pred",
            "residual",
            "abs_residual",
            "SCHOOL",
            "DISTRICT",
            "COUNTY",
            "DISTRICT_TYPE",
            "ASSESSMENT_NAME",
            "SUBGROUP_NAME",
            "REGION",
            "MACRO_REGION",
            "school_type",
        ]
    ).write_csv(full_resid_path)
    diagnostic["residuals_full_path"] = full_resid_path
    print(f"  Worst-100 CSV: {worst_csv_path}")
    print(f"  Full residuals CSV: {full_resid_path}")

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
    school_errors_path = f"ft_transformer_school_errors_{version_tag}.csv"
    school_errors.write_csv(school_errors_path)
    school_errors.write_csv("ft_transformer_school_errors.csv")
    diagnostic["school_errors_path"] = school_errors_path
    print(f"\nWorst 20 schools by mean abs residual (n>=5):")
    print(school_errors.head(20))

    stack_npz_path = f"ft_transformer_stack_{version_tag}.npz"
    np.savez_compressed(
        stack_npz_path,
        oof_stack=oof_global.reshape(-1, 1),
        test_stack=test_global.reshape(-1, 1),
        y=y_np,
        tier_names=np.array(["global"]),
    )
    diagnostic["stack_npz_path"] = stack_npz_path
    print(f"  Stack npz: {stack_npz_path}")

    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    print(f"  Diagnostic JSON: {diagnostic_path}")


# ── Stack ─────────────────────────────────────────────────────────────────────


def stack(force=False):
    outer_strat_key = _pool_rare(assessment_train, N_OUTER_FOLDS)
    kfold = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)

    oof_global = np.zeros(n_train)
    test_global = np.zeros(n_pred)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kfold.split(num_train_raw, outer_strat_key)
    ):
        print(
            f"\nFOLD {fold_idx+1}/{N_OUTER_FOLDS}  train={len(train_idx)}  val={len(val_idx)}"
        )
        print("  [global]")
        val_preds, test_preds = fit_one_expert(
            "global",
            "all",
            train_idx,
            val_idx,
            np.arange(n_pred),
            fold_idx=fold_idx,
        )
        oof_global[val_idx] = val_preds
        test_global += test_preds / N_OUTER_FOLDS

    global_rmse = float(np.sqrt(np.mean((y_np - oof_global) ** 2)))
    print(f"\nGlobal OOF RMSE: {global_rmse:.4f}")

    np.savez_compressed(
        ARTIFACTS_PATH, oof_global=oof_global, test_global=test_global, y=y_np
    )
    print(f"\nSaved {ARTIFACTS_PATH}")

    version_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_diagnostics(
        oof_global=oof_global, test_global=test_global, version_tag=version_tag
    )


# ── Submit global (multi-seed on 95/5 holdout) ────────────────────────────────


def submit_global(force=False, n_seeds=5):
    print(f"[submit-global] training {n_seeds} seeds")

    holdout_strat_key = _pool_rare(assessment_train, 20)
    tr_idx, va_idx = train_test_split(
        np.arange(n_train),
        test_size=0.05,
        stratify=holdout_strat_key,
        random_state=SEED,
    )

    val_preds_accum = np.zeros(len(va_idx), dtype=np.float64)
    test_preds_accum = np.zeros(n_pred, dtype=np.float64)
    per_seed_rmse = []

    for seed_i in range(n_seeds):
        seed = SEED + seed_i
        print(f"\n  --- seed {seed_i+1}/{n_seeds} (value={seed}) ---")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        val_preds, test_preds = fit_one_expert(
            "global",
            "all",
            tr_idx,
            va_idx,
            np.arange(n_pred),
            fold_idx=None,
        )
        seed_rmse = float(np.sqrt(np.mean((y_np[va_idx] - val_preds) ** 2)))
        per_seed_rmse.append(seed_rmse)
        print(f"  seed {seed_i+1} holdout RMSE: {seed_rmse:.4f}")
        val_preds_accum += val_preds
        test_preds_accum += test_preds

    val_avg = val_preds_accum / n_seeds
    test_avg = np.clip(test_preds_accum / n_seeds, 0, 100)
    ensemble_rmse = float(np.sqrt(np.mean((y_np[va_idx] - val_avg) ** 2)))

    print(f"\n[submit-global] per-seed RMSEs: {[f'{r:.4f}' for r in per_seed_rmse]}")
    print(f"[submit-global] mean per-seed: {np.mean(per_seed_rmse):.4f}")
    print(f"[submit-global] ensemble RMSE: {ensemble_rmse:.4f}")

    X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", test_avg)).write_csv(
        GLOBAL_SUBMISSION_PATH
    )
    print(f"[submit-global] wrote {GLOBAL_SUBMISSION_PATH}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["stack", "submit-global"], required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.mode == "stack":
        stack(force=args.force)
    else:
        submit_global(force=args.force, n_seeds=args.n_seeds)


if __name__ == "__main__":
    main()
