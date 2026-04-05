#!/usr/bin/env python3

import polars as pl
from sklearn.impute import KNNImputer

import warnings

warnings.filterwarnings("ignore", message="codecs.open")


categories = [
    "ASSESSMENT_ID",
    "SCHOOL",
    "SUBGROUP_NAME",
    "ASSESSMENT_NAME",
    "DISTRICT",
    "COUNTY",
    "DISTRICT_TYPE",
    "REGION",
    "school_type",
]
cat_set = set(categories)


def grouped_agg_features(df, cat_cols, num_cols):
    for cat in cat_cols:
        agg = df.group_by(cat).agg(
            [pl.col(n).mean().alias(f"{n}_mean_by_{cat}") for n in num_cols]
            + [pl.col(n).median().alias(f"{n}_median_by_{cat}") for n in num_cols]
            + [pl.col(n).sum().alias(f"{n}_sum_by_{cat}") for n in num_cols]
        )
        df = df.join(agg, on=cat, how="left")
    return df


def df_district_type_level_gb(df, col_to_use, new_col_name):
    df_new = (
        df.filter(pl.col(col_to_use).is_not_null())
        .group_by(["REGION", "COUNTY", "DISTRICT_TYPE"])
        .agg(pl.col(col_to_use).median().alias(new_col_name))
    )
    return df_new


def df_county_level_gb(df, col_to_use, new_col_name):
    df_new = (
        df.filter(pl.col(col_to_use).is_not_null())
        .group_by(["REGION", "COUNTY"])
        .agg(pl.col(col_to_use).median().alias(new_col_name))
    )
    return df_new


def df_region_level_gb(df, col_to_use, new_col_name):
    df_new = (
        df.filter(pl.col(col_to_use).is_not_null())
        .group_by(["REGION"])
        .agg(pl.col(col_to_use).median().alias(new_col_name))
    )
    return df_new


def df_total_unique_x_per_y(df, group_col, count_col, agg_name):
    return df.group_by(pl.col(group_col)).agg(
        pl.col(count_col).n_unique().alias(agg_name)
    )


def join_tables(df: pl.DataFrame, school: pl.DataFrame, district: pl.DataFrame):
    """
    One Function for feature engineering
    """

    district_type_level_columns_to_aggregate = {
        "GRADE_1_AVERAGE_CLASS_SIZE": "district_type_level_median_grade_1_size",
        "GRADE_2_AVERAGE_CLASS_SIZE": "district_type_level_median_grade_2_size",
        "KINDERGARTEN_AVERAGE_CLASS_SIZE": "district_type_level_median_kindergarten_class_size",
        "PRE_K": "district_type_level_median_pre_K_students",
        "K": "district_type_level_median_kindergarten_students",
        **{
            f"GRADE_{i:02d}": f"district_type_level_median_grade_{i}_students"
            for i in range(1, 13)
        },
        "ATTENDANCE_RATE": "district_type_level_median_attendance_rate",
        "TEACHER_TURNOVER_RATE": "district_type_level_median_teacher_turnover_rate",
        "LANGUAGE_ARTS_AVERAGE_CLASS_SIZE": "district_type_level_median_language_class_size",
        "MATHEMATICS_AVERAGE_CLASS_SIZE": "district_type_level_median_math_class_size",
        "SCIENCE_AVERAGE_CLASS_SIZE": "district_type_level_median_science_class_size",
        "HISTORY_GOVERNMENT_AND_GEOGRAPHY_AVERAGE_CLASS_SIZE": "district_type_level_median_class_size",
        "PERCENT_FREE_LUNCH": "district_type_level_median_percent_free_lunch",
        "PERCENT_REDUCED_LUNCH": "district_type_level_median_reduced_lunch",
        "NUMBER_OF_TEACHERS": "district_type_level_median_number_of_teachers",
        "NUMBER_OF_COUNSELORS": "district_type_level_median_number_of_counselors",
        "NUMBER_OF_SOCIAL_WORKERS": "district_type_level_median_number_of_social_workers",
        "N_PUPILS": "district_type_level_median_number_of_pupils",
    }

    county_level_columns_to_aggregate = {
        "GRADE_1_AVERAGE_CLASS_SIZE": "county_median_grade_1_size",
        "GRADE_2_AVERAGE_CLASS_SIZE": "county_median_grade_2_size",
        "KINDERGARTEN_AVERAGE_CLASS_SIZE": "county_median_kindergarten_class_size",
        "PRE_K": "county_median_pre_K_students",
        "K": "county_median_kindergarten_students",
        **{f"GRADE_{i:02d}": f"county_median_grade_{i}_students" for i in range(1, 13)},
        "ATTENDANCE_RATE": "county_median_attendance_rate",
        "TEACHER_TURNOVER_RATE": "county_median_teacher_turnover_rate",
        "LANGUAGE_ARTS_AVERAGE_CLASS_SIZE": "county_median_language_class_size",
        "MATHEMATICS_AVERAGE_CLASS_SIZE": "county_median_math_class_size",
        "SCIENCE_AVERAGE_CLASS_SIZE": "county_median_science_class_size",
        "HISTORY_GOVERNMENT_AND_GEOGRAPHY_AVERAGE_CLASS_SIZE": "county_median_class_size",
        "PERCENT_FREE_LUNCH": "county_median_percent_free_lunch",
        "PERCENT_REDUCED_LUNCH": "county_median_reduced_lunch",
        "NUMBER_OF_TEACHERS": "county_median_number_of_teachers",
        "NUMBER_OF_COUNSELORS": "county_median_number_of_counselors",
        "NUMBER_OF_SOCIAL_WORKERS": "county_median_number_of_social_workers",
        "N_PUPILS": "county_median_number_of_pupils",
    }

    region_level_columns_to_aggregate = {
        "GRADE_1_AVERAGE_CLASS_SIZE": "region_level_median_grade_1_size",
        "GRADE_2_AVERAGE_CLASS_SIZE": "region_level_median_grade_2_size",
        "KINDERGARTEN_AVERAGE_CLASS_SIZE": "region_level_median_kindergarten_class_size",
        "PRE_K": "region_level_median_pre_K_students",
        "K": "region_level_median_kindergarten_students",
        **{
            f"GRADE_{i:02d}": f"region_level_median_grade_{i}_students"
            for i in range(1, 13)
        },
        "ATTENDANCE_RATE": "region_level_median_attendance_rate",
        "TEACHER_TURNOVER_RATE": "region_level_median_teacher_turnover_rate",
        "LANGUAGE_ARTS_AVERAGE_CLASS_SIZE": "region_level_median_language_class_size",
        "MATHEMATICS_AVERAGE_CLASS_SIZE": "region_level_median_math_class_size",
        "SCIENCE_AVERAGE_CLASS_SIZE": "region_level_median_science_class_size",
        "HISTORY_GOVERNMENT_AND_GEOGRAPHY_AVERAGE_CLASS_SIZE": "region_level_median_class_size",
        "PERCENT_FREE_LUNCH": "region_level_median_percent_free_lunch",
        "PERCENT_REDUCED_LUNCH": "region_level_median_reduced_lunch",
        "NUMBER_OF_TEACHERS": "region_level_median_number_of_teachers",
        "NUMBER_OF_COUNSELORS": "region_level_median_number_of_counselors",
        "NUMBER_OF_SOCIAL_WORKERS": "region_level_median_number_of_social_workers",
        "N_PUPILS": "region_level_median_number_of_pupils",
    }

    x_per_y_unique_aggs = [
        ("DISTRICT", "SCHOOL", "total_schools_per_district"),
        ("COUNTY", "SCHOOL", "total_schools_per_county"),
        ("COUNTY", "DISTRICT", "total_districts_per_county"),
        ("COUNTY", "DISTRICT_TYPE", "total_district_types_per_county"),
        ("REGION", "DISTRICT", "total_districts_per_region"),
        ("REGION", "SCHOOL", "total_schools_per_region"),
    ]

    funding_types = [
        ("FEDERAL_FUNDING_PER_PUPIL", "fed_funding"),
        ("LOCAL_FUNDING_PER_PUPIL", "local_funding"),
        ("rough_total_funding_per_pupil", "total_funding"),
    ]

    hierarchy = [
        ("SCHOOL", "DISTRICT"),
        ("DISTRICT", "COUNTY"),
        ("COUNTY", "REGION"),
    ]

    funding_tables = {}

    for source_col, label in funding_types:
        key = f"{label}_per_school"
        funding_tables[key] = school.group_by("SCHOOL").agg(
            pl.col(source_col).mul(pl.col("N_PUPILS")).sum().alias(key)
        )

    for child_col, parent_col in hierarchy:
        for _, label in funding_types:
            child_key = f"{label}_per_{child_col.lower()}"
            parent_key = f"{label}_per_{parent_col.lower()}"
            funding_tables[parent_key] = (
                school.select(pl.col(parent_col), pl.col(child_col))
                .unique()
                .join(funding_tables[child_key], on=child_col, how="left")
                .group_by(parent_col)
                .agg(pl.col(child_key).sum().alias(parent_key))
            )

    imputer = KNNImputer(
        n_neighbors=5,
        weights="distance",
        add_indicator=True,
    )

    to_impute = [
        "LANGUAGE_ARTS_AVERAGE_CLASS_SIZE",
        "MATHEMATICS_AVERAGE_CLASS_SIZE",
        "SCIENCE_AVERAGE_CLASS_SIZE",
        "HISTORY_GOVERNMENT_AND_GEOGRAPHY_AVERAGE_CLASS_SIZE",
        "LOCAL_FUNDING_PER_PUPIL",
        "FEDERAL_FUNDING_PER_PUPIL",
        "ATTENDANCE_RATE",
        "N_PUPILS",
        "TEACHER_TURNOVER_RATE",
        "PERCENT_FREE_LUNCH",
        "PERCENT_REDUCED_LUNCH",
        "PERCENT_OF_STUDENTS_SUSPENDED",
        "PERCENT_MALE",
        "PERCENT_FEMALE",
        "PERCENT_ENGLISH_LANGUAGE_LEANERS",
        "PERCENT_AMERICAN_INDIAN",
        "PERCENT_BLACK",
        "PERCENT_ASIAN",
        "PERCENT_HISPANIC",
        "PERCENT_WHITE",
        "PERCENT_MULTIRACIAL",
        "PERCENT_WITH_DISABILITIES",
        "PERCENT_ECONOMICALLY_DISADVANTAGED",
        "PERCENT_MIGRANT",
        "PERCENT_HOMELESS",
        "PERCENT_IN_FOSTER_CARE",
        "PERCENT_PARENT_ARMED_FORCES",
    ]
    school_imputed = imputer.fit_transform(school.select(to_impute).to_numpy())
    col_names = to_impute + [f"missingindicator_{c}" for c in to_impute]

    school_imputed = school.with_columns(
        [
            pl.Series(name=col, values=school_imputed[:, i])
            for i, col in enumerate(col_names)
        ]
    )

    res = (
        df.join(school_imputed, on="SCHOOL", how="left")
        .join(district, on="DISTRICT", how="left")
        .with_columns(
            num_students_on_free_lunch=pl.col("N_PUPILS")
            .mul(pl.col("PERCENT_FREE_LUNCH"))
            .truediv(100),
            num_students_on_reduced_lunch=pl.col("N_PUPILS")
            .mul(pl.col("PERCENT_REDUCED_LUNCH"))
            .truediv(100),
        )
    )

    for key, tbl in funding_tables.items():
        level = key.split("_per_")[1]
        join_col = level.upper()
        res = res.join(tbl, on=join_col, how="left")

    for group_col, count_col, alias in x_per_y_unique_aggs:
        agg_df = df_total_unique_x_per_y(school, group_col, count_col, alias)
        res = res.join(agg_df, on=group_col, how="left")

    for col, alias in district_type_level_columns_to_aggregate.items():
        agg_df = df_district_type_level_gb(school, col, alias)
        res = res.join(agg_df, on=["REGION", "COUNTY", "DISTRICT_TYPE"], how="left")

    for col, alias in county_level_columns_to_aggregate.items():
        agg_df = df_county_level_gb(school, col, alias)
        res = res.join(agg_df, on=["REGION", "COUNTY"], how="left")

    for col, alias in region_level_columns_to_aggregate.items():
        agg_df = df_region_level_gb(school, col, alias)
        res = res.join(agg_df, on=["REGION"], how="left")

    return res


def get_data(cached=True):
    train = pl.read_csv(
        "https://michael-weylandt.com/STA9890/competition_data/scores_training.csv",
        schema_overrides={
            "ASSESSMENT_ID": pl.Categorical,
            "SCHOOL": pl.Categorical,
            "SUBGROUP_NAME": pl.Categorical,
            "ASSESSMENT_NAME": pl.Categorical,
        },
        null_values=[" ", "NA", ""],
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
            null_values=[" ", "NA", ""],
        )
        .with_columns(
            INFORMATION_NOT_REPORTED_OR_MISSING=pl.when(
                pl.sum_horizontal(pl.all().is_null()).gt(0)
            )
            .then(pl.lit(1))
            .otherwise(0)
        )
        .with_columns(
            INFORMATION_NOT_REPORTED_FLAG=pl.when(
                pl.sum_horizontal(pl.all().is_null()).gt(
                    5
                )  # quick look at the data in excel, the missing data threshold cuts off at 5 cols with NAs
            )  # the not reported data threshold is anything greater than 5
            .then(pl.lit(1))
            .otherwise(0)
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
            student_teacher_ratio=pl.col("N_PUPILS").truediv(
                pl.col("NUMBER_OF_TEACHERS")
            ),
        )
    ).with_columns(
        KINDERGARTEN_AVERAGE_CLASS_SIZE=pl.when(pl.col("K").eq(pl.lit(0)))
        .then(pl.lit(0))
        .when(
            pl.col("K").ne(pl.lit(0))
            & pl.col("K").is_not_null()
            & pl.col("KINDERGARTEN_AVERAGE_CLASS_SIZE").is_not_null()
        )
        .then(pl.col("KINDERGARTEN_AVERAGE_CLASS_SIZE"))
        .when(pl.col("KINDERGARTEN_AVERAGE_CLASS_SIZE").is_not_null())
        .then(pl.col("KINDERGARTEN_AVERAGE_CLASS_SIZE")),
        GRADE_1_AVERAGE_CLASS_SIZE=pl.when(pl.col("GRADE_01").eq(pl.lit(0)))
        .then(pl.lit(0))
        .when(
            pl.col("GRADE_01").ne(pl.lit(0))
            & pl.col("GRADE_01").is_not_null()
            & pl.col("GRADE_1_AVERAGE_CLASS_SIZE").is_not_null()
        )
        .then(pl.col("GRADE_1_AVERAGE_CLASS_SIZE"))
        .when(pl.col("GRADE_1_AVERAGE_CLASS_SIZE").is_not_null())
        .then(pl.col("GRADE_1_AVERAGE_CLASS_SIZE")),
        GRADE_2_AVERAGE_CLASS_SIZE=pl.when(pl.col("GRADE_02").eq(pl.lit(0)))
        .then(pl.lit(0))
        .when(
            pl.col("GRADE_02").ne(pl.lit(0))
            & pl.col("GRADE_02").is_not_null()
            & pl.col("GRADE_2_AVERAGE_CLASS_SIZE").is_not_null()
        )
        .then(pl.col("GRADE_2_AVERAGE_CLASS_SIZE"))
        .when(pl.col("GRADE_2_AVERAGE_CLASS_SIZE").is_not_null())
        .then(pl.col("GRADE_2_AVERAGE_CLASS_SIZE")),
    )

    grade_cols = {
        "PRE_K": -1,
        "K": 0,
        "GRADE_01": 1,
        "GRADE_02": 2,
        "GRADE_03": 3,
        "GRADE_04": 4,
        "GRADE_05": 5,
        "GRADE_06": 6,
        "GRADE_07": 7,
        "GRADE_08": 8,
        "GRADE_09": 9,
        "GRADE_10": 10,
        "GRADE_11": 11,
        "GRADE_12": 12,
    }

    has_pupils = [
        pl.col(col).is_not_null().and_(pl.col(col) > 0).alias(f"has_{col}")
        for col in grade_cols
    ]

    school = school.with_columns(has_pupils)

    min_grade = pl.min_horizontal(
        *[
            pl.when(pl.col(f"has_{col}")).then(pl.lit(level)).otherwise(pl.lit(None))
            for col, level in grade_cols.items()
        ]
    ).alias("min_grade")

    max_grade = pl.max_horizontal(
        *[
            pl.when(pl.col(f"has_{col}")).then(pl.lit(level)).otherwise(pl.lit(None))
            for col, level in grade_cols.items()
        ]
    ).alias("max_grade")

    school = school.with_columns(min_grade, max_grade)

    school = school.with_columns(
        pl.when((pl.col("min_grade") == -1) & (pl.col("max_grade") == -1))
        .then(pl.lit("nursery"))
        .when((pl.col("min_grade") == -1) & (pl.col("max_grade") <= 5))
        .then(pl.lit("pre_k_elementary"))
        .when((pl.col("min_grade") == -1) & (pl.col("max_grade") <= 8))
        .then(pl.lit("pre_k_8"))
        .when((pl.col("min_grade") == -1) & (pl.col("max_grade") <= 12))
        .then(pl.lit("pre_k_12"))
        .when((pl.col("min_grade") == 0) & (pl.col("max_grade") <= 5))
        .then(pl.lit("elementary"))
        .when((pl.col("min_grade") == 0) & (pl.col("max_grade") <= 8))
        .then(pl.lit("k_8"))
        .when((pl.col("min_grade") == 0) & (pl.col("max_grade") <= 12))
        .then(pl.lit("k_12"))
        .when((pl.col("min_grade") >= 6) & (pl.col("max_grade") <= 8))
        .then(pl.lit("middle_school"))
        .when((pl.col("min_grade") >= 9) & (pl.col("max_grade") <= 12))
        .then(pl.lit("high_school"))
        .when((pl.col("min_grade") >= 6) & (pl.col("max_grade") <= 12))
        .then(pl.lit("secondary"))
        .otherwise(pl.lit("other"))
        .alias("school_type")
    )

    district = pl.read_csv(
        "https://michael-weylandt.com/STA9890/competition_data/district_covariates.csv",
        schema_overrides={"DISTRICT": pl.Categorical},
        null_values=[" ", "NA", ""],
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

    X_train = join_tables(train, school, district)

    x_train_min_funding = X_train.select(
        pl.col("rough_total_funding_per_pupil").min().mean()
    ).item()
    x_train_mean_funding = X_train.select(
        pl.col("rough_total_funding_per_pupil").mean()
    ).item()

    x_train_median_funding = X_train.select(
        pl.col("rough_total_funding_per_pupil").median()
    ).item()

    x_train_local_funding_mean = X_train.select(
        pl.col("local_funding_per_school").mean()
    ).item()

    x_train_fed_funding_mean = X_train.select(
        pl.col("fed_funding_per_school").mean()
    ).item()

    x_train_local_funding_median = X_train.select(
        pl.col("local_funding_per_school").median()
    ).item()

    x_train_fed_funding_median = X_train.select(
        pl.col("fed_funding_per_school").median()
    ).item()

    def add_funding_ratio_cols(df):
        df = df.with_columns(
            funding_for_number_of_students_in_group_taking_assessment=pl.col(
                "rough_total_funding_per_pupil"
            ).mul("N_STUDENTS"),
            ratio_of_funding_to_min_funding=pl.col(
                "rough_total_funding_per_pupil"
            ).truediv(x_train_min_funding),
            ratio_of_funding_to_mean_funding=pl.col(
                "rough_total_funding_per_pupil"
            ).truediv(x_train_mean_funding),
            ratio_of_funding_to_median_funding=pl.col(
                "rough_total_funding_per_pupil"
            ).truediv(x_train_median_funding),
            ratio_of_total_funding_to_median_local_funding=pl.col(
                "total_funding_per_school"
            ).truediv(x_train_local_funding_median),
            ratio_of_total_funding_to_median_fed_funding=pl.col(
                "total_funding_per_school"
            ).truediv(x_train_fed_funding_median),
            ratio_of_total_funding_to_mean_local_funding=pl.col(
                "total_funding_per_school"
            ).truediv(x_train_local_funding_mean),
            ratio_of_total_funding_to_mean_fed_funding=pl.col(
                "total_funding_per_school"
            ).truediv(x_train_fed_funding_mean),
        )
        return df

    X_train = add_funding_ratio_cols(X_train)
    y_train = train.select(pl.col("PERCENT_PROFICIENT"))
    X_train = X_train.select(pl.exclude("PERCENT_PROFICIENT"))

    X_pred = join_tables(test, school, district)
    X_pred = add_funding_ratio_cols(X_pred)
    X_pred_id = test.select(pl.col("ASSESSMENT_ID"))
    num_cols = [
        c
        for c in X_train.columns
        if c not in cat_set
        and not c.startswith("has_")
        and X_train[c].dtype not in (pl.Boolean, pl.Utf8, pl.Categorical)
    ]

    group_cats = [c for c in categories if c in X_train.columns]

    X_train = grouped_agg_features(X_train, group_cats, num_cols)
    X_pred = grouped_agg_features(X_pred, group_cats, num_cols)

    if cached:
        X_pred.write_parquet("X_pred.parquet")
        X_train.write_parquet("X_train.parquet")
        y_train.write_parquet("y_train.parquet")
        X_pred_id.write_parquet("X_pred_id.parquet")

        return None
    else:
        return X_pred, X_train, y_train, X_pred_id
