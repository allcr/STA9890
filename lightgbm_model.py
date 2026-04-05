import polars as pl
from polars import selectors as cs
import lightgbm
import random
import optuna
from optuna.samplers import TPESampler
import json

pl.Config(set_tbl_cols=10000, set_fmt_str_lengths=1000, set_tbl_width_chars=10000)
pl.enable_string_cache()
random.seed(8675309)

study_name = "lightgbm"

storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend("./lightgbm_journal_storage.log")
)


def objective(trial):

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 20),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.95),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-10, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-10, 10, log=True),
    }

    es = lightgbm.callback.early_stopping(
        stopping_rounds=20, min_delta=0.01, verbose=False
    )

    pruning_callback = optuna.integration.LightGBMPruningCallback(
        trial, "rmse", valid_name="valid"
    )
    cv_results = lightgbm.cv(
        params | {"seed": 5558675309} | {"monotone_constraints": ordered_constraints},
        dtrain,
        nfold=5,
        num_boost_round=5000,
        metrics="rmse",
        callbacks=[es, pruning_callback],
    )

    actual_num_rounds = len(cv_results["valid rmse-mean"])
    trial.set_user_attr("actual_num_rounds", actual_num_rounds)

    return min(cv_results["valid rmse-mean"])


X_train = pl.read_parquet("X_train.parquet")
y_train = pl.read_parquet("y_train.parquet")
X_pred = pl.read_parquet("X_pred.parquet")
X_pred_id = pl.read_parquet("X_pred_id.parquet")


numeric_cols = (
    X_train.select(pl.exclude(cs.Categorical()))
    .select(pl.exclude("PERCENT_PROFICIENT"))
    .columns
)
for i in numeric_cols:
    X_train = X_train.with_columns(
        pl.col(i).cast(pl.Float32(), strict=False).name.keep()
    )

cat_cols = [
    col
    for col, dtype in zip(X_train.columns, X_train.dtypes)
    if dtype == pl.Categorical
]

X_train_pd = X_train.to_pandas()
X_pred_pd = X_pred.to_pandas()

y_train_np = y_train.to_numpy().ravel()

dtrain = lightgbm.Dataset(X_train_pd, label=y_train_np, free_raw_data=False)
monotone_constraints_dict = {"ATTENDANCE_RATE": 1, "TEACHER_TURNOVER_RATE": -1}
feature_names = X_train.columns
ordered_constraints = [monotone_constraints_dict.get(name, 0) for name in feature_names]

sampler = TPESampler(seed=8675309)
pruner = optuna.pruners.PatientPruner(
    optuna.pruners.MedianPruner(
        n_warmup_steps=150,
        n_startup_trials=50,
        n_min_trials=50,
        interval_steps=10,
    ),
    patience=5,
)

study = optuna.create_study(
    study_name=study_name,
    direction="minimize",
    storage=storage,
    load_if_exists=True,
    sampler=sampler,
    pruner=pruner,
)


study.optimize(objective, n_trials=10000, show_progress_bar=True, gc_after_trial=True)
best_params = study.best_params
best_rounds = study.best_trial.user_attrs["actual_num_rounds"]

with open("best_params_lightgbm.json", "w") as f:
    json.dump(best_params, f, indent=4)

model = lightgbm.LGBMRegressor(
    **best_params,
    metric="rmse",
    importance_type="gain",
    n_estimators=best_rounds,
    enable_categorical=True,
    monotone_constraints=ordered_constraints,
)


model.fit(X_train_pd, y_train_np)

y_pred = model.predict(X_pred_pd)

submission = X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", y_pred))

submission.write_csv("submission_lightgbm.csv")
