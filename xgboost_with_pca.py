#!/usr/bin/env python3

import polars as pl
from polars import selectors as cs
import xgboost
import random
import numpy as np
import optuna
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# TODO  see what i can do with onehot/other methods
# TODO CNN + Ensemble learning after speaking with BOB
# TODO BART this could be fun
# TODO Vanilla Ridge Regression + Lasso + Elastic Net with categorical features?
# TODO Brms multilevel hierarchical modelling of school performance
pl.Config(set_tbl_cols=10000, set_fmt_str_lengths=1000, set_tbl_width_chars=10000)
pl.enable_string_cache()
random.seed(8675309)

study_name = "xgboost_with_pca"
sqldb = "sqlite:///optuna.db"


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 2.5, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 0, 64),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "subsample": trial.suggest_float("subsample", 0.9, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.9, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.9, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.9, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 100),
        "lambda": trial.suggest_float("lambda", 0, 100),
        # i tried gpu with hist and mse was either 1e33 or 650 something. this is what i get for
        # trying to compile an unofficial ROCM xgboost
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }

    #  xgb_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")
    cv_results = xgboost.cv(
        params,
        dtrain,
        nfold=10,
        num_boost_round=10000,
        early_stopping_rounds=100,
        metrics="rmse",
        seed=8675309,
        callbacks=[
            xgboost.callback.EvaluationMonitor(show_stdv=True, period=1000),
            #      xgb_callback,
        ],
        as_pandas=True,
    )

    result = cv_results["test-rmse-mean"].min()
    trial.set_user_attr("best_num_boost_round", len(cv_results))
    return result


train = pl.read_csv(
    "https://michael-weylandt.com/STA9890/competition_data/scores_training.csv",
    schema_overrides={
        "ASSESSMENT_ID": pl.Categorical,
        "SCHOOL": pl.Categorical,
        "SUBGROUP_NAME": pl.Categorical,
        "ASSESSMENT_NAME": pl.Categorical,
    },
)
school = pl.read_csv(
    "https://michael-weylandt.com/STA9890/competition_data/school_covariates.csv",
    schema_overrides={
        "DISTRICT": pl.Categorical,
        "SCHOOL": pl.Categorical,
        "COUNTY": pl.Categorical,
        "DISTRICT_TYPE": pl.Categorical,
        "REGION": pl.Categorical,
    },
)
district = pl.read_csv(
    "https://michael-weylandt.com/STA9890/competition_data/district_covariates.csv",
    schema_overrides={"DISTRICT": pl.Categorical},
)
test = pl.read_csv(
    "https://michael-weylandt.com/STA9890/competition_data/scores_test.csv",
    schema_overrides={
        "ASSESSMENT_ID": pl.Categorical,
        "SCHOOL": pl.Categorical,
        "SUBGROUP_NAME": pl.Categorical,
        "ASSESSMENT_NAME": pl.Categorical,
    },
)
X_train_id = train.select(pl.col("ASSESSMENT_ID"))
y_train = train.select(pl.col("PERCENT_PROFICIENT"))
X_train = (
    train.select(pl.exclude("ASSESSMENT_ID"))
    .select(pl.exclude("PERCENT_PROFICIENT"))
    .join(school, on="SCHOOL", how="left")
    .join(district, on="DISTRICT", how="left")
)
X_pred_id = test.select(pl.col("ASSESSMENT_ID"))
X_pred = (
    test.select(pl.exclude("ASSESSMENT_ID"))
    .join(school, on="SCHOOL", how="left")
    .join(district, on="DISTRICT", how="left")
)
numeric_cols = X_train.select(pl.exclude(cs.Categorical())).columns
for i in numeric_cols:
    X_train = X_train.with_columns(
        pl.col(i).cast(pl.Float32(), strict=False).name.keep()
    )
    X_pred = X_pred.with_columns(pl.col(i).cast(pl.Float32(), strict=False).name.keep())

cat_cols = [
    col
    for col, dtype in zip(X_train.columns, X_train.dtypes)
    if dtype == pl.Categorical
]

X_train_pd = X_train.fill_null(-9).to_pandas()
X_pred_pd = X_pred.fill_null(-9).to_pandas()
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_train_pd[cat_cols] = enc.fit_transform(X_train_pd[cat_cols])
X_pred_pd[cat_cols] = enc.transform(X_pred_pd[cat_cols])
y_train_np = y_train.to_numpy().ravel()


study = optuna.create_study(
    study_name=study_name,
    direction="minimize",
    storage=sqldb,
    load_if_exists=True,
    #  pruner=pruner,
)


study.optimize(objective, n_trials=1000, show_progress_bar=True, gc_after_trial=True)
best_params = study.best_params
print("Best params:", study.best_params)
print("Best CV RMSE:", study.best_value)
best_num_boost_round = study.best_trial.user_attrs["best_num_boost_round"]
best_params.pop("num_boost_round")


scaler = StandardScaler()
pca = PCA(n_components=best_params.pop("pca__n_components"))

X_train_transformed = pca.fit_transform(scaler.fit_transform(X_train_pd))
X_pred_transformed = pca.transform(scaler.transform(X_pred_pd))

best_model = xgboost.XGBRegressor(
    **best_params,
    n_estimators=best_num_boost_round,
    n_jobs=-1,
)
best_model.fit(X_train_transformed, y_train_np)
yhat = best_model.predict(X_pred_transformed)


X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", yhat)).write_csv(
    "submission_xgboost_w_pca.csv"
)
