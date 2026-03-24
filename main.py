import polars as pl
from polars import selectors as cs
import xgboost
import random
import optuna
from optuna.samplers import TPESampler
import json

# TODO SCHOOL, DISTRICT,COUNTY, REGION Funding SPLIT BY LOCAL AND FEDERAL
# TODO Multiple high needs schools per district?
# TODO APPROXIMATE POOR GDP IN district/COUNTY/region WITH High needs and other vars
# maybe use min max local funding?
# TODO I WILL HAVE TO NORMALIZE MY DATA A BIT, A FEW DIFFERENT SCALES
# TODO TABNET
# TODO  see what i can do with onehot/other methods
# TODO CNN + Ensemble learning after speaking with BOB
# TODO BART this could be fun
# TODO Vanilla Ridge Regression + Lasso + Elastic Net with categorical features?
# TODO Brms multilevel hierarchical modelling of school performance
pl.Config(set_tbl_cols=10000, set_fmt_str_lengths=1000, set_tbl_width_chars=10000)
pl.enable_string_cache()
random.seed(8675309)

study_name = "xgboost"
sqldb = "sqlite:///optuna.db"


def objective(trial):
    params = {
        "eta": 0.1,
        "max_depth": 5,
        "min_child_weight": trial.suggest_float("min_child_weight", 0, 20),
        "gamma": trial.suggest_float("gamma", 5e-5, 10),
        "max_leaves": 0,
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["lossguide", "depthwise"]
        ),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.8, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.8, 1.0),
        "alpha": trial.suggest_float("alpha", 0, 100),
        "lambda": trial.suggest_float("lambda", 0, 100),
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "monotone_constraints": {"ATTENDANCE_RATE": 1},
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")
    cv_results = xgboost.cv(
        params,
        dtrain,
        nfold=5,
        num_boost_round=100,
        early_stopping_rounds=10,
        verbose_eval=True,
        metrics="rmse",
        callbacks=[pruning_callback],
    )

    trial.set_user_attr("actual_num_rounds", cv_results.shape[0])
    trial.set_user_attr("train-rmse-mean", cv_results["train-rmse-mean"].min())
    return cv_results["test-rmse-mean"].min()


def objective_round2(trial):
    params = {
        "min_child_weight": best_params_round_1["min_child_weight"],
        "gamma": best_params_round_1["gamma"],
        "subsample": best_params_round_1["subsample"],
        "colsample_bytree": best_params_round_1["colsample_bytree"],
        "colsample_bylevel": best_params_round_1["colsample_bylevel"],
        "colsample_bynode": best_params_round_1["colsample_bynode"],
        "alpha": best_params_round_1["alpha"],
        "lambda": best_params_round_1["lambda"],
        "max_leaves": 0,
        "grow_policy": best_params_round_1["grow_policy"],
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": trial.suggest_float("eta", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "monotone_constraints": {"ATTENDANCE_RATE": 1},
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")
    cv_results = xgboost.cv(
        params,
        dtrain,
        nfold=5,
        num_boost_round=5000,
        early_stopping_rounds=20,
        metrics="rmse",
        callbacks=[pruning_callback],
    )
    trial.set_user_attr("actual_num_rounds", cv_results.shape[0])
    trial.set_user_attr("train-rmse-mean", cv_results["train-rmse-mean"].min())
    return cv_results["test-rmse-mean"].min()


train = pl.read_csv(
    "https://michael-weylandt.com/STA9890/competition_data/scores_training.csv",
    schema_overrides={
        "ASSESSMENT_ID": pl.Categorical,
        "SCHOOL": pl.Categorical,
        "SUBGROUP_NAME": pl.Categorical,
        "ASSESSMENT_NAME": pl.Categorical,
    },
).with_columns(
    pl.exclude(["ASSESSMENT_ID", "SCHOOL", "SUBGROUP_NAME", "ASSESSMENT_NAME"])
    .cast(pl.Float32, strict=False)
    .name.keep()
)
school = (
    pl.read_csv(
        "https://michael-weylandt.com/STA9890/competition_data/school_covariates.csv",
        schema_overrides={
            "DISTRICT": pl.Categorical,
            "SCHOOL": pl.Categorical,
            "COUNTY": pl.Categorical,
            "DISTRICT_TYPE": pl.Categorical,
            "REGION": pl.Categorical,
        },
    )
    .with_columns(
        pl.exclude(["DISTRICT", "SCHOOL", "COUNTY", "DISTRICT_TYPE", "REGION"])
        .cast(pl.Float32, strict=False)
        .name.keep()
    )
    .with_columns(
        rough_total_funding_per_pupil=pl.col("FEDERAL_FUNDING_PER_PUPIL").add(
            pl.col("LOCAL_FUNDING_PER_PUPIL")
        ),
        student_teacher_ratio=pl.col("N_PUPILS").truediv(pl.col("NUMBER_OF_TEACHERS")),
    )
)

total_schools_per_district = school.group_by(pl.col("DISTRICT")).agg(
    pl.col("SCHOOL").n_unique().alias("total_schools_per_district")
)

total_schools_per_county = school.group_by(pl.col("COUNTY")).agg(
    pl.col("SCHOOL").n_unique().alias("total_schools_per_county")
)
total_districts_per_county = school.group_by(pl.col("COUNTY")).agg(
    pl.col("DISTRICT").n_unique().alias("total_districts_per_county")
)

total_district_types_per_county = school.group_by(pl.col("COUNTY")).agg(
    pl.col("DISTRICT_TYPE").n_unique().alias("total_district_types_per_county")
)

total_districts_per_region = school.group_by(pl.col("REGION")).agg(
    pl.col("DISTRICT").n_unique().alias("total_districts_per_region")
)

total_schools_per_region = school.group_by(pl.col("REGION")).agg(
    pl.col("SCHOOL").n_unique().alias("total_schools_per_region")
)

total_pupils_per_district = school.group_by(pl.col("DISTRICT")).agg(
    pl.col("N_PUPILS").sum().alias("total_district_pupils")
)

total_pupils_per_county = school.group_by(pl.col("COUNTY")).agg(
    pl.col("N_PUPILS").sum().alias("total_county_pupils")
)

total_pupils_per_region = school.group_by(pl.col("REGION")).agg(
    pl.col("N_PUPILS").sum().alias("total_county_pupils")
)

total_funding_per_school = school.group_by(pl.col("SCHOOL")).agg(
    pl.col("rough_total_funding_per_pupil")
    .mul(pl.col("N_PUPILS"))
    .sum()
    .alias("total_funding_per_school")
)

total_funding_per_district = (
    school.select(pl.col("DISTRICT"), pl.col("SCHOOL"))
    .unique()
    .join(total_funding_per_school, on="SCHOOL", how="left")
    .group_by("DISTRICT")
    .agg(pl.col("total_funding_per_school").sum().alias("total_funding_per_district"))
)


total_funding_per_county = (
    school.select(pl.col("COUNTY"), pl.col("DISTRICT"))
    .unique()
    .join(total_funding_per_district, on="DISTRICT", how="left")
    .group_by("COUNTY")
    .agg(pl.col("total_funding_per_district").sum().alias("total_funding_per_county"))
)


total_funding_per_region = (
    school.select(pl.col("REGION"), pl.col("COUNTY"))
    .unique()
    .join(total_funding_per_county, on="COUNTY", how="left")
    .group_by("REGION")
    .agg(pl.col("total_funding_per_county").sum().alias("total_funding_per_region"))
)

district = pl.read_csv(
    "https://michael-weylandt.com/STA9890/competition_data/district_covariates.csv",
    schema_overrides={"DISTRICT": pl.Categorical},
).with_columns(pl.exclude("DISTRICT").cast(pl.Float32, strict=False).name.keep())

test = pl.read_csv(
    "https://michael-weylandt.com/STA9890/competition_data/scores_test.csv",
    schema_overrides={
        "ASSESSMENT_ID": pl.Categorical,
        "SCHOOL": pl.Categorical,
        "SUBGROUP_NAME": pl.Categorical,
        "ASSESSMENT_NAME": pl.Categorical,
    },
).with_columns(
    pl.exclude(["ASSESSMENT_ID", "SCHOOL", "SUBGROUP_NAME", "ASSESSMENT_NAME"])
    .cast(pl.Float32, strict=False)
    .name.keep()
)
X_train_id = train.select(pl.col("ASSESSMENT_ID"))
y_train = train.select(pl.col("PERCENT_PROFICIENT"))
X_train = (
    train.select(pl.exclude("ASSESSMENT_ID"))
    .select(pl.exclude("PERCENT_PROFICIENT"))
    .join(school, on="SCHOOL", how="left")
    .join(district, on="DISTRICT", how="left")
    .join(total_pupils_per_county, on="COUNTY", how="left")
    .join(total_pupils_per_district, on="DISTRICT", how="left")
    .join(total_pupils_per_region, on="REGION", how="left")
    .join(total_schools_per_district, on="DISTRICT", how="left")
    .join(total_schools_per_county, on="COUNTY", how="left")
    .join(total_districts_per_county, on="COUNTY", how="left")
    .join(total_district_types_per_county, on="COUNTY", how="left")
    .join(total_districts_per_region, on="REGION", how="left")
    .join(total_schools_per_region, on="REGION", how="left")
    .join(total_funding_per_school, on="SCHOOL", how="left")
    .join(total_funding_per_district, on="DISTRICT", how="left")
    .join(total_funding_per_county, on="COUNTY", how="left")
    .join(total_funding_per_region, on="REGION", how="left")
    .with_columns(
        ratio_students_taking_assesment_to_total_pupils=pl.col("N_STUDENTS").truediv(
            pl.col("N_PUPILS").mul(100)
        )
    )
)
x_train_min_funding = X_train.select(
    pl.col("rough_total_funding_per_pupil").min().mean()
).item()
x_train_mean_funding = X_train.select(
    pl.col("rough_total_funding_per_pupil").mean()
).item()

x_train_median_funding = X_train.select(
    pl.col("rough_total_funding_per_pupil").median()
).item()

X_train = X_train.with_columns(
    ratio_of_funding_to_min_funding=pl.col("rough_total_funding_per_pupil").truediv(
        x_train_min_funding
    ),
    ratio_of_funding_to_mean_funding=pl.col("rough_total_funding_per_pupil").truediv(
        x_train_mean_funding
    ),
    ratio_of_funding_to_median_funding=pl.col("rough_total_funding_per_pupil").truediv(
        x_train_median_funding
    ),
)


X_pred_id = test.select(pl.col("ASSESSMENT_ID"))
X_pred = (
    test.select(pl.exclude("ASSESSMENT_ID"))
    .join(school, on="SCHOOL", how="left")
    .join(district, on="DISTRICT", how="left")
    .join(total_pupils_per_county, on="COUNTY", how="left")
    .join(total_pupils_per_district, on="DISTRICT", how="left")
    .join(total_pupils_per_region, on="REGION", how="left")
    .join(total_schools_per_district, on="DISTRICT", how="left")
    .join(total_schools_per_county, on="COUNTY", how="left")
    .join(total_districts_per_county, on="COUNTY", how="left")
    .join(total_district_types_per_county, on="COUNTY", how="left")
    .join(total_districts_per_region, on="REGION", how="left")
    .join(total_schools_per_region, on="REGION", how="left")
    .join(total_funding_per_school, on="SCHOOL", how="left")
    .join(total_funding_per_district, on="DISTRICT", how="left")
    .join(total_funding_per_county, on="COUNTY", how="left")
    .join(total_funding_per_region, on="REGION", how="left")
    .with_columns(
        ratio_students_taking_assesment_to_total_pupils=pl.col("N_STUDENTS").truediv(
            pl.col("N_PUPILS").mul(100)
        )
    )
)


X_pred = X_pred.with_columns(
    ratio_of_funding_to_min_funding=pl.col("rough_total_funding_per_pupil").truediv(
        x_train_min_funding
    ),
    ratio_of_funding_to_mean_funding=pl.col("rough_total_funding_per_pupil").truediv(
        x_train_mean_funding
    ),
    ratio_of_funding_to_median_funding=pl.col("rough_total_funding_per_pupil").truediv(
        x_train_median_funding
    ),
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
best_params_round_1 = study.best_params

study_2 = optuna.create_study(
    study_name=study_name + "_round_2",
    direction="minimize",
    storage=sqldb,
    load_if_exists=True,
    sampler=sampler,
    pruner=pruner,
)
study_2.optimize(
    objective_round2, n_trials=200, show_progress_bar=True, gc_after_trial=True
)
best_params = {**best_params_round_1, **study_2.best_params}

with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

model = xgboost.XGBRegressor(
    **best_params,
    eval_metric="rmse",
    n_estimators=15000,
    early_stopping_rounds=100,
    enable_categorical=True,
    monotone_constraints={"ATTENDANCE_RATE": 1},
)
model.fit(X_train_pd, y_train_np, eval_set=[(X_train_pd, y_train_np)])
y_pred = model.predict(X_pred_pd)

submission = X_pred_id.with_columns(pl.Series("PERCENT_PROFICIENT", y_pred))

submission.write_csv("submission_xgboost.csv")
