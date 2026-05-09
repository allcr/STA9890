"""
Compare meta-learners on the combined ResNet + LightGBM MoE stack.

Each meta-learner is evaluated by:
  1. In-sample OOF RMSE       (optimistic — fit and scored on same data)
  2. 10-fold CV RMSE on OOF   (honest generalisation estimate)
  3. 20% holdout RMSE         (least-biased; Varma & Simon 2006 correction)

Rank by holdout_rmse or cv_rmse. Never by in-sample.

Outputs
-------
Per meta-learner:
  submission_meta_<name>.csv
  predictions_<name>_<ts>.csv     per-row OOF preds + residuals + labels

Aggregate:
  meta_learner_summary.csv         one row per meta-learner, all three RMSEs
                                   plus all coefficients
  meta_learner_summary_<ts>.json   same in JSON for programmatic access
  predictions_all_metalearners_<ts>.csv
                                   wide: one row per train obs, one col per
                                   meta-learner OOF pred + base-learner OOF
                                   preds + per-row meta_pred_std
  residuals_full_<ts>.csv          same but residuals only (smaller)
  worst_residuals_meta_<ts>.csv    worst-100 by best meta-learner CV residual
  submission_meta_top3_ensemble.csv
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb

SEED = 8675309
N_META_FOLDS = 10
HOLDOUT_FRAC = 0.20
VERSION_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Load artifacts ────────────────────────────────────────────────────────────
catboost_art = np.load("catboost_moe_artifacts.npz)")
resnet = np.load("resnet_moe_artifacts.npz")
lgb_art = np.load("lightgbm_moe_artifacts.npz")
ft = np.load("ft_transformer_moe_artifacts.npz")
assert np.array_equal(
    resnet["y"], lgb_art["y"], ft["y"], catboost["y"]
), "y arrays differ between artifacts — outer fold seeds must match"
y = resnet["y"]

X_pred_id = pl.read_parquet("X_pred_id.parquet")
X_train = pl.read_parquet("X_train.parquet").drop("ASSESSMENT_ID")

oof_stack = np.column_stack(
    [
        resnet["oof_global"],
        resnet["oof_macro"],
        resnet["oof_region"],
        resnet["oof_district_type"],
        resnet["oof_group"],
        resnet["oof_assessment"],
        lgb_art["oof_global"],
        lgb_art["oof_macro"],
        lgb_art["oof_region"],
        lgb_art["oof_district_type"],
        lgb_art["oof_county"],
        lgb_art["oof_group"],
        lgb_art["oof_assessment"],
        ft["oof_global"],
        catboost_art["oof_global"],
        catboost_art["oof_macro"],
        catboost_art["oof_region"],
        catboost_art["oof_district_type"],
        catboost_art["oof_county"],
        catboost_art["oof_group"],
        catboost_art["oof_assessment"],
    ]
)
test_stack = np.column_stack(
    [
        resnet["test_global"],
        resnet["test_macro"],
        resnet["test_region"],
        resnet["test_district_type"],
        resnet["test_group"],
        resnet["test_assessment"],
        lgb_art["test_global"],
        lgb_art["test_macro"],
        lgb_art["test_region"],
        lgb_art["test_district_type"],
        lgb_art["test_county"],
        lgb_art["test_group"],
        lgb_art["test_assessment"],
        ft["test_global"],
        catboost_art["test_global"],
        catboost_art["test_macro"],
        catboost_art["test_region"],
        catboost_art["test_district_type"],
        catboost_art["test_county"],
        catboost_art["test_group"],
        catboost_art["test_assessment"],
    ]
)
column_names = [
    "resnet_global",
    "resnet_macro",
    "resnet_region",
    "resnet_district_type",
    "resnet_group",
    "resnet_assessment",
    "lgb_global",
    "lgb_macro",
    "lgb_region",
    "lgb_district_type",
    "lgb_county",
    "lgb_group",
    "lgb_assessment",
    "ft_global",
    "cb_global",
    "cb_macro",
    "cb_region",
    "cb_district_type",
    "cb_county",
    "cb_group",
    "cb_assessment",
]
assert oof_stack.shape[1] == len(column_names), (
    f"column_names length {len(column_names)} != "
    f"oof_stack columns {oof_stack.shape[1]}"
)


# ── Holdout split ─────────────────────────────────────────────────────────────
# Stratified 80/20 split. Meta-learners fit on 80%, scored on 20%.
# This is the Varma & Simon (2006) correction for nested CV optimism bias.

holdout_strat = X_train["ASSESSMENT_NAME"].to_numpy()
meta_tr_idx, meta_va_idx = train_test_split(
    np.arange(len(y)),
    test_size=HOLDOUT_FRAC,
    stratify=holdout_strat,
    random_state=SEED,
)
print(
    f"Meta-learner split: {len(meta_tr_idx)} train / "
    f"{len(meta_va_idx)} holdout ({int(HOLDOUT_FRAC*100)}%)"
)


# ── Raw stack diagnostics ─────────────────────────────────────────────────────

print("\nPer-column OOF RMSE / MAE:")
print(f"  {'column':<24s}  {'RMSE':>8s}  {'MAE':>8s}")
for name, col in zip(column_names, oof_stack.T):
    rmse = float(np.sqrt(np.mean((y - col) ** 2)))
    mae = float(np.mean(np.abs(y - col)))
    print(f"  {name:<24s}  {rmse:>8.4f}  {mae:>8.4f}")

print("\nResidual correlation matrix:")
resid_mat = y[:, None] - oof_stack
corr = np.corrcoef(resid_mat.T)
print(f"  {'':24s}" + "".join(f"{n[:8]:>10s}" for n in column_names))
for i, name in enumerate(column_names):
    row = (
        "  "
        + f"{name:24s}"
        + "".join(f"{corr[i, j]:>10.3f}" for j in range(len(column_names)))
    )
    print(row)

print("\nBand-by-band RMSE (lgb_global as reference):")
lgb_global_col = oof_stack[:, column_names.index("lgb_global")]
for lo, hi in [(0, 5), (5, 25), (25, 50), (50, 75), (75, 95), (95, 100)]:
    mask = (y >= lo) & (y <= hi)
    if mask.sum() > 0:
        band_rmse = float(np.sqrt(np.mean((y[mask] - lgb_global_col[mask]) ** 2)))
        print(
            f"  y in [{lo:3d},{hi:3d}]: n={mask.sum():6d}  "
            f"lgb_global RMSE={band_rmse:.3f}"
        )


# ── Meta-learner definitions ──────────────────────────────────────────────────


class NNLSRegressor:
    """Non-negative least squares with centred intercept."""

    def fit(self, X, y):
        self.y_mean_ = float(y.mean())
        self.X_mean_ = X.mean(axis=0)
        coef, _ = nnls(X - self.X_mean_, y - self.y_mean_)
        self.coef_ = coef
        self.intercept_ = self.y_mean_ - float(self.X_mean_ @ coef)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class ConvexRegressor:
    """Weights constrained to the probability simplex (sum=1, all>=0)."""

    def __init__(self, lr=0.01, n_iter=10000, tol=1e-9):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        n = X.shape[1]
        w = np.ones(n) / n
        prev_loss = np.inf
        for _ in range(self.n_iter):
            pred = X @ w
            grad = 2 * X.T @ (pred - y) / len(y)
            w = self._project_simplex(w - self.lr * grad)
            loss = float(np.mean((y - X @ w) ** 2))
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        self.coef_ = w
        self.intercept_ = 0.0
        return self

    @staticmethod
    def _project_simplex(v):
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(1, n + 1)
        cond = u - cssv / ind > 0
        theta = cssv[cond][-1] / ind[cond][-1]
        return np.maximum(v - theta, 0)

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class LGBMetaLearner:
    """Heavily regularised LightGBM meta-learner."""

    def fit(self, X, y):
        self.model_ = lgb.train(
            {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.01,
                "num_leaves": 8,
                "min_child_samples": 50,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "seed": SEED,
            },
            lgb.Dataset(X, label=y),
            num_boost_round=2000,
        )
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def feature_importances(self, names):
        imp = self.model_.feature_importance(importance_type="gain")
        return dict(zip(names, imp.tolist()))


meta_learners = {
    "RidgeCV": lambda: RidgeCV(alphas=np.logspace(-6, 6, 200)),
    "LassoCV": lambda: LassoCV(
        alphas=np.logspace(-4, 2, 100), cv=5, random_state=SEED, max_iter=20000
    ),
    "ElasticNetCV": lambda: ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=np.logspace(-4, 2, 50),
        cv=5,
        random_state=SEED,
        max_iter=20000,
    ),
    "OLS": lambda: LinearRegression(),
    "NNLS": lambda: NNLSRegressor(),
    "Convex": lambda: ConvexRegressor(),
    "GBR_shallow": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, random_state=SEED
    ),
    "LightGBM_meta": lambda: LGBMetaLearner(),
}


# ── Evaluation helpers ────────────────────────────────────────────────────────


def run_cv(meta_factory, oof_X, oof_y):
    """10-fold CV on full OOF. Returns (rmse, per-row OOF predictions)."""
    kf = KFold(n_splits=N_META_FOLDS, shuffle=True, random_state=SEED)
    preds = np.zeros_like(oof_y, dtype=np.float64)
    for tr_i, va_i in kf.split(oof_X):
        m = meta_factory()
        m.fit(oof_X[tr_i], oof_y[tr_i])
        preds[va_i] = m.predict(oof_X[va_i])
    return float(np.sqrt(np.mean((oof_y - preds) ** 2))), preds


def run_holdout(meta_factory, tr_X, tr_y, va_X, va_y):
    """Fit on tr, score on va. Returns (rmse, fitted model)."""
    m = meta_factory()
    m.fit(tr_X, tr_y)
    preds = m.predict(va_X)
    return float(np.sqrt(np.mean((va_y - preds) ** 2))), m


# ── Baselines ─────────────────────────────────────────────────────────────────

print("\n\nBASELINES:")
col_rmses = [
    float(np.sqrt(np.mean((y - oof_stack[:, i]) ** 2)))
    for i in range(oof_stack.shape[1])
]
best_col_idx = int(np.argmin(col_rmses))
print(
    f"  Best single column : {column_names[best_col_idx]} @ RMSE {col_rmses[best_col_idx]:.4f}"
)
equal_rmse = float(np.sqrt(np.mean((y - oof_stack.mean(axis=1)) ** 2)))
print(f"  Equal-weight avg   : {equal_rmse:.4f}")


# ── Main sweep ────────────────────────────────────────────────────────────────

print(
    f"\n\nMETA-LEARNER COMPARISON "
    f"({N_META_FOLDS}-fold CV + {int(HOLDOUT_FRAC*100)}% holdout):"
)
print(
    f"  {'meta-learner':<20s}  {'CV RMSE':>9s}  {'Holdout':>9s}  "
    f"{'In-sample':>10s}  {'gap(CV-IS)':>10s}"
)
print(f"  {'-'*20}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*10}")

results = {}

for name, factory in meta_learners.items():
    try:
        cv_score, cv_oof_preds = run_cv(factory, oof_stack, y)

        ho_score, _ = run_holdout(
            factory,
            oof_stack[meta_tr_idx],
            y[meta_tr_idx],
            oof_stack[meta_va_idx],
            y[meta_va_idx],
        )

        prod_model = factory()
        prod_model.fit(oof_stack, y)
        is_preds = prod_model.predict(oof_stack)
        is_score = float(np.sqrt(np.mean((y - is_preds) ** 2)))
        gap = cv_score - is_score

        print(
            f"  {name:<20s}  {cv_score:>9.4f}  {ho_score:>9.4f}  "
            f"{is_score:>10.4f}  {gap:>+10.4f}"
        )

        coef_dict = {}
        intercept = 0.0
        if hasattr(prod_model, "coef_"):
            coef_dict = dict(zip(column_names, prod_model.coef_.tolist()))
            intercept = float(prod_model.intercept_)
        elif isinstance(prod_model, LGBMetaLearner):
            coef_dict = prod_model.feature_importances(column_names)

        results[name] = {
            "cv_rmse": cv_score,
            "holdout_rmse": ho_score,
            "in_sample_rmse": is_score,
            "gap_cv_minus_insample": gap,
            "model": prod_model,
            "cv_oof_preds": cv_oof_preds,
            "is_preds": is_preds,
            "coef": coef_dict,
            "intercept": intercept,
        }

    except Exception as e:
        print(f"  {name:<20s}  FAILED: {type(e).__name__}: {e}")

ranked = sorted(results.items(), key=lambda kv: kv[1]["cv_rmse"])

print(f"\nRanked by CV RMSE:")
print(f"  {'meta-learner':<20s}  {'CV RMSE':>9s}  {'Holdout':>9s}  {'In-sample':>10s}")
for name, info in ranked:
    print(
        f"  {name:<20s}  {info['cv_rmse']:>9.4f}  "
        f"{info['holdout_rmse']:>9.4f}  {info['in_sample_rmse']:>10.4f}"
    )


# ── Coefficients ──────────────────────────────────────────────────────────────

print("\n\nLinear meta-learner coefficients (production model, 100% OOF):")
for name in ["RidgeCV", "LassoCV", "ElasticNetCV", "OLS", "NNLS", "Convex"]:
    if name not in results:
        continue
    m = results[name]["model"]
    if not hasattr(m, "coef_"):
        continue
    print(f"\n  {name} (intercept={results[name]['intercept']:+.4f}):")
    for cn, coef in zip(column_names, m.coef_):
        bar = "█" * min(40, int(abs(coef) * 30))
        sign = "+" if coef >= 0 else "-"
        print(f"    {cn:<24s} {sign}{abs(coef):.4f}  {bar}")

if "LightGBM_meta" in results:
    print("\n  LightGBM_meta feature importances (gain):")
    imps = results["LightGBM_meta"]["coef"]
    max_imp = max(imps.values()) if imps else 1
    for cn, imp in sorted(imps.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, int(imp / max_imp * 30))
        print(f"    {cn:<24s} {imp:>10.1f}  {bar}")


# ── Per-row prediction CSVs ───────────────────────────────────────────────────

label_cols = [
    "SCHOOL",
    "DISTRICT",
    "COUNTY",
    "DISTRICT_TYPE",
    "REGION",
    "MACRO_REGION",
    "ASSESSMENT_NAME",
    "SUBGROUP_NAME",
    "school_type",
]
label_df = X_train.select([c for c in label_cols if c in X_train.columns])

print("\n\nWriting per-meta-learner prediction CSVs...")
for name, info in results.items():
    is_pred = info["is_preds"]
    cv_pred = info["cv_oof_preds"]
    df = label_df.with_columns(
        [
            pl.Series("orig_idx", np.arange(len(y))),
            pl.Series("y_true", y),
            pl.Series("pred_insample", is_pred.astype(np.float32)),
            pl.Series("resid_insample", (y - is_pred).astype(np.float32)),
            pl.Series("abs_resid_insample", np.abs(y - is_pred).astype(np.float32)),
            pl.Series("pred_cv", cv_pred.astype(np.float32)),
            pl.Series("resid_cv", (y - cv_pred).astype(np.float32)),
            pl.Series("abs_resid_cv", np.abs(y - cv_pred).astype(np.float32)),
        ]
    )
    out = f"predictions_{name.lower().replace('_', '')}_{VERSION_TAG}.csv"
    df.write_csv(out)
    print(f"  {name:<20s} -> {out}")


# ── Wide comparison CSV ───────────────────────────────────────────────────────

print("\nBuilding wide prediction + residual CSVs...")
meta_preds_arr = np.column_stack([info["is_preds"] for _, info in results.items()])
meta_names_safe = [n.lower().replace("_", "") for n, _ in results.items()]

wide_df = label_df.with_columns(
    [
        pl.Series("orig_idx", np.arange(len(y))),
        pl.Series("y_true", y),
        pl.Series("meta_pred_std", meta_preds_arr.std(axis=1).astype(np.float32)),
        pl.Series("meta_pred_mean", meta_preds_arr.mean(axis=1).astype(np.float32)),
    ]
)
for safe_name, (name, info) in zip(meta_names_safe, results.items()):
    wide_df = wide_df.with_columns(
        [
            pl.Series(f"pred_{safe_name}", info["is_preds"].astype(np.float32)),
            pl.Series(f"resid_{safe_name}", (y - info["is_preds"]).astype(np.float32)),
        ]
    )
for col_name, col_data in zip(column_names, oof_stack.T):
    wide_df = wide_df.with_columns(
        pl.Series(f"base_{col_name}", col_data.astype(np.float32))
    )

wide_path = f"predictions_all_metalearners_{VERSION_TAG}.csv"
wide_df.write_csv(wide_path)
print(f"  Wide predictions: {wide_path}")

resid_df = label_df.with_columns(
    [
        pl.Series("orig_idx", np.arange(len(y))),
        pl.Series("y_true", y),
        pl.Series("meta_pred_std", meta_preds_arr.std(axis=1).astype(np.float32)),
    ]
)
for safe_name, (name, info) in zip(meta_names_safe, results.items()):
    resid_df = resid_df.with_columns(
        pl.Series(f"resid_{safe_name}", (y - info["is_preds"]).astype(np.float32))
    )
resid_path = f"residuals_full_{VERSION_TAG}.csv"
resid_df.write_csv(resid_path)
print(f"  Full residuals:   {resid_path}")


# ── Summary CSV + JSON ────────────────────────────────────────────────────────

summary_rows = []
for name, info in ranked:
    row = {
        "meta_learner": name,
        "cv_rmse": round(info["cv_rmse"], 6),
        "holdout_rmse": round(info["holdout_rmse"], 6),
        "in_sample_rmse": round(info["in_sample_rmse"], 6),
        "gap_cv_minus_insample": round(info["gap_cv_minus_insample"], 6),
        "intercept": round(info["intercept"], 6),
    }
    for col in column_names:
        row[f"coef_{col}"] = round(info["coef"].get(col, float("nan")), 6)
    summary_rows.append(row)

summary_df = pl.DataFrame(summary_rows)
summary_df.write_csv("meta_learner_summary.csv")
print(f"\nSummary CSV: meta_learner_summary.csv")

summary_json = {
    name: {
        k: v for k, v in info.items() if k not in ("model", "cv_oof_preds", "is_preds")
    }
    for name, info in results.items()
}
json_path = f"meta_learner_summary_{VERSION_TAG}.json"
Path(json_path).write_text(json.dumps(summary_json, indent=2))
print(f"Summary JSON: {json_path}")


# ── Submission CSVs ───────────────────────────────────────────────────────────

print("\n\nWriting submission CSVs:")
for name, info in results.items():
    test_pred = np.clip(info["model"].predict(test_stack), 0, 100)
    out = f"submission_meta_{name.lower().replace('_', '')}.csv"
    X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", test_pred)).write_csv(out)
    print(f"  {name:<20s} -> {out}")


# ── Top-3 ensemble submission ─────────────────────────────────────────────────

top_3_names = [name for name, _ in ranked[:3]]
print(f"\nTop-3 ensemble (by CV RMSE): {top_3_names}")
top_3_test = np.column_stack(
    [np.clip(results[n]["model"].predict(test_stack), 0, 100) for n in top_3_names]
).mean(axis=1)
top_3_oof = np.column_stack([results[n]["is_preds"] for n in top_3_names]).mean(axis=1)
print(f"  In-sample ensemble RMSE: {float(np.sqrt(np.mean((y - top_3_oof)**2))):.4f}")
X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", top_3_test)).write_csv(
    "submission_meta_top3_ensemble.csv"
)
print("  -> submission_meta_top3_ensemble.csv")


# ── Worst-row analysis ────────────────────────────────────────────────────────

best_name = ranked[0][0]
best_cv_preds = results[best_name]["cv_oof_preds"]
best_resid = y - best_cv_preds
best_abs_resid = np.abs(best_resid)
worst_idx = np.argsort(best_abs_resid)[-100:][::-1]

print(f"\nWorst-100 rows by {best_name} CV residual:")
worst_df = X_train.with_columns(
    [
        pl.Series("y_true", y),
        pl.Series("y_pred_cv", best_cv_preds.astype(np.float32)),
        pl.Series("residual", best_resid.astype(np.float32)),
        pl.Series("abs_residual", best_abs_resid.astype(np.float32)),
        pl.Series("meta_pred_std", meta_preds_arr.std(axis=1).astype(np.float32)),
    ]
).with_row_index("orig_idx")[worst_idx.tolist()]

print(
    worst_df.select(
        [
            "orig_idx",
            "y_true",
            "y_pred_cv",
            "residual",
            "meta_pred_std",
            "SCHOOL",
            "DISTRICT_TYPE",
            "ASSESSMENT_NAME",
            "SUBGROUP_NAME",
            "REGION",
        ]
    ).head(20)
)
worst_path = f"worst_residuals_meta_{VERSION_TAG}.csv"
worst_df.write_csv(worst_path)
print(f"  Worst-100 CSV: {worst_path}")

print(f"\nErrors by target band ({best_name} CV preds):")
for lo, hi in [(0, 5), (5, 25), (25, 50), (50, 75), (75, 95), (95, 100)]:
    mask = (y >= lo) & (y <= hi)
    if mask.sum() > 0:
        band_rmse = float(np.sqrt(np.mean(best_resid[mask] ** 2)))
        mean_resid = float(best_resid[mask].mean())
        print(
            f"  y in [{lo:3d},{hi:3d}]: n={mask.sum():6d}  "
            f"RMSE={band_rmse:.3f}  mean_resid={mean_resid:+.3f}"
        )


print("\n\nAll outputs written. Key files:")
print(f"  meta_learner_summary.csv                      — compare all meta-learners")
print(f"  {wide_path}")
print(f"  {resid_path}")
print(f"  submission_meta_top3_ensemble.csv             — safe submission")
print(f"\nDecision rule:")
print("  Rank by holdout_rmse (least biased) or cv_rmse.")
print("  If top-3 within 0.01 RMSE of each other, the ensemble is safer than one pick.")
print("  Never pick by in_sample_rmse.")
