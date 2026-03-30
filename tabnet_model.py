import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from pytorch_tabnet import TabNetRegressor, TabNetPretrainer
import polars as pl
import json

sqldb = "sqlite:///optuna.db"
study_name = "xgboost"

SEED = 8675309
np.random.seed(SEED)

X_train = pl.read_parquet("X_train.parquet")
y_train = pl.read_parquet("y_train.parquet")
X_pred = pl.read_parquet("X_pred.parquet")
X_pred_id = pl.read_parquet("X_pred_id.parquet")


CAT_COLS = [
    "ASSESSMENT_ID",
    "SCHOOL",
    "SUBGROUP_NAME",
    "ASSESSMENT_NAME",
    "DISTRICT",
    "COUNTY",
    "DISTRICT_TYPE",
    "REGION",
]
NUM_COLS = [c for c in X_train.columns if c not in CAT_COLS]


oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_all_cat = pd.concat([X_train[CAT_COLS], X_pred[CAT_COLS]], axis=0).astype(str)
oe.fit(X_all_cat)
X_train[CAT_COLS] = oe.transform(X_train[CAT_COLS].astype(str)).astype(int)
X_pred[CAT_COLS] = oe.transform(X_pred[CAT_COLS].astype(str)).astype(int)

cat_idxs = [X_train.columns.get_loc(c) for c in CAT_COLS]
cat_dims = [len(cats) for cats in oe.categories_]

imputer = SimpleImputer(strategy="median")
pt = PowerTransformer(method="yeo-johnson")
ss = StandardScaler()


X_train[NUM_COLS] = imputer.fit_transform(X_train[NUM_COLS])
X_pred[NUM_COLS] = imputer.transform(X_pred[NUM_COLS])

X_train[NUM_COLS] = ss.fit_transform(pt.fit_transform(X_train[NUM_COLS]))
X_pred[NUM_COLS] = ss.transform(pt.transform(X_pred[NUM_COLS]))

X_train_np = X_train.values.astype(np.float32)
X_pred_np = X_pred.values.astype(np.float32)
y_train_f = y_train.astype(np.float32)


def objective(trial):
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    bs = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096])
    params = dict(
        n_steps=trial.suggest_int("n_steps", 3, 10),
        n_a=trial.suggest_int("n_a", 8, 512),
        n_d=trial.suggest_int("n_d", 8, 512),
        gamma=trial.suggest_float("gamma", 1.0, 2.0),
        momentum=trial.suggest_float("momentum", 0.01, 0.4),
        lambda_sparse=trial.suggest_float("lambda_sparse", 1e-7, 1e-1, log=True),
        mask_type="sparsemax",
        optimizer_params=dict(lr=lr),
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        seed=SEED,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_np)):
        model = TabNetRegressor(**params)
        model.fit(
            X_train_np[tr_idx],
            y_train_f[tr_idx],
            eval_set=[(X_train_np[va_idx], y_train_f[va_idx])],
            eval_metric=["rmse"],
            max_epochs=400,
            patience=20,
            batch_size=bs,
            drop_last=False,
        )
        scores.append(model.best_cost)
        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(scores)


sampler = optuna.samplers.TPESampler(seed=8675309)
study = optuna.create_study(
    direction="minimize",
    study_name="tabnet",
    storage=sqldb,
    load_if_exists=True,
    sampler=sampler,
)
study.optimize(objective, n_trials=500, show_progress_bar=True)


best = study.best_params
with open("best_params.json", "w") as f:
    json.dump(best, f, indent=4)

pretrain = TabNetPretrainer(
    n_steps=best["n_steps"],
    n_a=best["n_a"],
    n_d=best["n_d"],
    gamma=best["gamma"],
    momentum=best["momentum"],
    lambda_sparse=best["lambda_sparse"],
    optimizer_params=dict(lr=best["lr"]),
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=1,
    seed=SEED,
    verbose=10,
)
pretrain.fit(
    X_train_np,
    max_epochs=1500,
    patience=50,
    batch_size=best["batch_size"],
    pretraining_ratio=0.5,
)


final = TabNetRegressor(
    n_steps=best["n_steps"],
    n_a=best["n_a"],
    n_d=best["n_d"],
    gamma=best["gamma"],
    momentum=best["momentum"],
    lambda_sparse=best["lambda_sparse"],
    optimizer_params=dict(lr=best["lr"]),
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=1,
    verbose=10,
)


n = len(X_train_np)
idx = np.random.permutation(n)
split = int(0.9 * n)
tr_idx, va_idx = idx[:split], idx[split:]

final.fit(
    X_train_np[tr_idx],
    y_train_f[tr_idx],
    eval_set=[(X_train_np[va_idx], y_train_f[va_idx])],
    eval_metric=["rmse"],
    max_epochs=1500,
    patience=50,
    batch_size=best["batch_size"],
    from_unsupervised=pretrain,
    drop_last=False,
)


train_preds = final.predict(X_train_np)
train_rmse = np.sqrt(np.mean((y_train.ravel() - train_preds.ravel()) ** 2))

y_pred = final.predict(X_pred_np)
out = X_pred_id.copy()
out["PERCENT_PROFICIENT"] = y_pred.ravel()
out.to_csv("submission_tabnet.csv", index=False)
print("wrote submission_tabnet.csv")
