#!/usr/bin/env python3

import polars as pl


def join_tables(df: pl.DataFrame, school: pl.DataFrame, district: pl.DataFrame):
    """
    Just make one function to make my dataframe with the funding local pivots and joins
    """

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
        pl.col("N_PUPILS").sum().alias("total_region_pupils")
    )

    total_federal_funding_per_school = school.group_by("SCHOOL").agg(
        pl.col("FEDERAL_FUNDING_PER_PUPIL")
        .mul(pl.col("N_PUPILS"))
        .sum()
        .alias("fed_funding_per_school")
    )

    total_local_funding_per_school = school.group_by("SCHOOL").agg(
        pl.col("LOCAL_FUNDING_PER_PUPIL")
        .mul(pl.col("N_PUPILS"))
        .sum()
        .alias("local_funding_per_school")
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
        .agg(
            pl.col("total_funding_per_school").sum().alias("total_funding_per_district")
        )
    )
    total_local_funding_per_district = (
        school.select(pl.col("DISTRICT"), pl.col("SCHOOL"))
        .unique()
        .join(total_local_funding_per_school, on="SCHOOL", how="left")
        .group_by("DISTRICT")
        .agg(
            pl.col("local_funding_per_school")
            .sum()
            .alias("total_local_funding_per_district")
        )
    )
    total_fed_funding_per_district = (
        school.select(pl.col("DISTRICT"), pl.col("SCHOOL"))
        .unique()
        .join(total_federal_funding_per_school, on="SCHOOL", how="left")
        .group_by("DISTRICT")
        .agg(
            pl.col("fed_funding_per_school")
            .sum()
            .alias("total_fed_funding_per_district")
        )
    )
    total_funding_per_district = (
        school.select(pl.col("DISTRICT"), pl.col("SCHOOL"))
        .unique()
        .join(total_funding_per_school, on="SCHOOL", how="left")
        .group_by("DISTRICT")
        .agg(
            pl.col("total_funding_per_school").sum().alias("total_funding_per_district")
        )
    )

    total_funding_per_county = (
        school.select(pl.col("COUNTY"), pl.col("DISTRICT"))
        .unique()
        .join(total_funding_per_district, on="DISTRICT", how="left")
        .group_by("COUNTY")
        .agg(
            pl.col("total_funding_per_district").sum().alias("total_funding_per_county")
        )
    )
    total_fed_funding_per_county = (
        school.select(pl.col("COUNTY"), pl.col("DISTRICT"))
        .unique()
        .join(total_fed_funding_per_district, on="DISTRICT", how="left")
        .group_by("COUNTY")
        .agg(
            pl.col("total_fed_funding_per_district")
            .sum()
            .alias("total_fed_funding_per_county")
        )
    )
    total_local_funding_per_county = (
        school.select(pl.col("COUNTY"), pl.col("DISTRICT"))
        .unique()
        .join(total_local_funding_per_district, on="DISTRICT", how="left")
        .group_by("COUNTY")
        .agg(
            pl.col("total_local_funding_per_district")
            .sum()
            .alias("total_local_funding_per_county")
        )
    )

    total_funding_per_region = (
        school.select(pl.col("REGION"), pl.col("COUNTY"))
        .unique()
        .join(total_funding_per_county, on="COUNTY", how="left")
        .group_by("REGION")
        .agg(pl.col("total_funding_per_county").sum().alias("total_funding_per_region"))
    )

    total_local_funding_per_region = (
        school.select(pl.col("REGION"), pl.col("COUNTY"))
        .unique()
        .join(total_fed_funding_per_county, on="COUNTY", how="left")
        .group_by("REGION")
        .agg(
            pl.col("total_fed_funding_per_county")
            .sum()
            .alias("total_fed_funding_per_region")
        )
    )
    total_fed_funding_per_region = (
        school.select(pl.col("REGION"), pl.col("COUNTY"))
        .unique()
        .join(total_fed_funding_per_county, on="COUNTY", how="left")
        .group_by("REGION")
        .agg(
            pl.col("total_fed_funding_per_county")
            .sum()
            .alias("total_fed_funding_per_region")
        )
    )

    res = (
        df.join(school, on="SCHOOL", how="left")
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
        .join(total_federal_funding_per_school, on="SCHOOL", how="left")
        .join(total_fed_funding_per_district, on="DISTRICT", how="left")
        .join(total_fed_funding_per_county, on="COUNTY", how="left")
        .join(total_fed_funding_per_region, on="REGION", how="left")
        .join(total_local_funding_per_school, on="SCHOOL", how="left")
        .join(total_local_funding_per_district, on="DISTRICT", how="left")
        .join(total_local_funding_per_county, on="COUNTY", how="left")
        .join(total_local_funding_per_region, on="REGION", how="left")
        .with_columns(
            num_students_on_free_lunch=pl.col("N_STUDENTS")
            .mul(pl.col("PERCENT_FREE_LUNCH"))
            .truediv(100),
            num_students_on_reduced_lunch=pl.col("N_STUDENTS")
            .mul(pl.col("PERCENT_REDUCED_LUNCH"))
            .truediv(100),
        )
    )

    return res


def get_data():
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
            student_teacher_ratio=pl.col("N_PUPILS").truediv(
                pl.col("NUMBER_OF_TEACHERS")
            ),
        )
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

    X_train = X_train.with_columns(
        ratio_of_funding_to_min_funding=pl.col("rough_total_funding_per_pupil").truediv(
            x_train_min_funding
        ),
        ratio_of_funding_to_mean_funding=pl.col(
            "rough_total_funding_per_pupil"
        ).truediv(x_train_mean_funding),
        ratio_of_funding_to_median_funding=pl.col(
            "rough_total_funding_per_pupil"
        ).truediv(x_train_median_funding),
    )

    y_train = train.select(pl.col("PERCENT_PROFICIENT"))
    X_train = X_train.select(pl.exclude("PERCENT_PROFICIENT"))

    X_pred = join_tables(test, school, district)

    X_pred = X_pred.with_columns(
        ratio_of_funding_to_min_funding=pl.col("rough_total_funding_per_pupil").truediv(
            x_train_min_funding
        ),
        ratio_of_funding_to_mean_funding=pl.col(
            "rough_total_funding_per_pupil"
        ).truediv(x_train_mean_funding),
        ratio_of_funding_to_median_funding=pl.col(
            "rough_total_funding_per_pupil"
        ).truediv(x_train_median_funding),
    )

    X_pred_id = test.select(pl.col("ASSESSMENT_ID"))

    X_pred.write_csv("X_pred.csv")
    X_train.write_csv("X_train.csv")
    y_train.write_csv("y_train.csv")
    X_pred_id.write_csv("X_pred_id.csv")

    return (X_train, y_train, X_pred, X_pred_id)


X_train, y_train, X_pred, X_pred_id = get_data()
