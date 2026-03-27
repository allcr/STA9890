import polars as pl
from polars import selectors as cs
import xgboost
import random
import optuna
from sklearn.model_selection import train_test_split
from optuna.samplers import TPESampler
import json
from data_processing import get_data

pl.Config(set_tbl_cols=10000, set_fmt_str_lengths=1000, set_tbl_width_chars=10000)
pl.enable_string_cache()
random.seed(8675309)

study_name = "xgboost"
sqldb = "sqlite:///optuna.db"


def objective(trial):
    params = {
        "eta": trial.suggest_float("eta", 0.01, 0.4),
        "max_depth": trial.suggest_int("max_depth", 3, 11),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20),
        "gamma": trial.suggest_float("gamma", 1, 20),
        "max_leaves": 0,
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["lossguide", "depthwise"]
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-10, 10, log=True),
        "lambda": trial.suggest_float("lambda", 1e-10, 10, log=True),
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "monotone_constraints": {"ATTENDANCE_RATE": 1},
    }
    early_stop = xgboost.callback.EarlyStopping(rounds=10, min_delta=0.01)

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")
    cv_results = xgboost.cv(
        params | {"seed": 5558675309},
        dtrain,
        nfold=10,
        num_boost_round=1000,
        verbose_eval=False,
        metrics="rmse",
        callbacks=[pruning_callback, early_stop],
    )

    trial.set_user_attr("actual_num_rounds", cv_results.shape[0] + 1)
    trial.set_user_attr("train-rmse-mean", cv_results["train-rmse-mean"].min())
    return cv_results["test-rmse-mean"].min()


X_train, y_train, X_pred, X_pred_id = get_data()


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

dtrain = xgboost.DMatrix(X_train_pd, label=y_train_np, enable_categorical=True)


sampler = TPESampler(seed=8675309)
pruner = optuna.pruners.HyperbandPruner()
study = optuna.create_study(
    study_name=study_name,
    direction="minimize",
    storage=sqldb,
    load_if_exists=True,
    sampler=sampler,
    pruner=pruner,
)


study.optimize(objective, n_trials=3000, show_progress_bar=True, gc_after_trial=True)

best_params = study.best_params


with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

model = xgboost.XGBRegressor(
    **best_params,
    eval_metric="rmse",
    n_estimators=1000,
    early_stopping_rounds=10,
    enable_categorical=True,
    monotone_constraints={"ATTENDANCE_RATE": 1},
)


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_pd, y_train_np, test_size=0.15, random_state=5256000
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

y_pred = model.predict(X_pred_pd)

submission = X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", y_pred))

submission.write_csv("submission_xgboost.csv")
