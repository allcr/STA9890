#!/usr/bin/env python3

import polars as pl
import numpy as np


import warnings

import networkx as nx

warnings.filterwarnings("ignore", message="codecs.open")


categories = [
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
    "DISTRICT_TYPE_x_REGION",
    "ASSESSMENT_NAME_x_SUBGROUP_NAME",
]

cat_set = set(categories)

school_demographic_cols = [
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
    "PERCENT_OF_STUDENTS_SUSPENDED",
    "PERCENT_ENGLISH_LANGUAGE_LEANERS",
]
district_demographic_cols = [
    "PERCENT_NON_DIPLOMA",
    "PERCENT_DIPLOMA",
    "PERCENT_DROPOUT",
    "PERCENT_GED",
    "PERCENT_STILL_ENROLLED",
]
demographic_cols = school_demographic_cols + district_demographic_cols

# def grouped_agg_features(df, cat_cols, num_cols):
#     for cat in cat_cols:
#         agg = df.group_by(cat).agg(
#             [pl.col(n).mean().alias(f"{n}_mean_by_{cat}") for n in num_cols]
#             + [pl.col(n).median().alias(f"{n}_median_by_{cat}") for n in num_cols]
#             + [pl.col(n).sum().alias(f"{n}_sum_by_{cat}") for n in num_cols]
#         )
#         df = df.join(agg, on=cat, how="left")
#     return df
zero_meaningful_cols = set(school_demographic_cols) | set(district_demographic_cols)


# ── NY Region adjacency graph ─────────────────────────────────────────────────
# Hand-encoded from NY State map. Symmetric (if A is adj to B, B is adj to A).
# Region names match your REGION column values exactly.

REGION_ADJ = {
    "New York City": ["Long Island", "Hudson Valley"],
    "Long Island": ["New York City"],
    "Hudson Valley": ["New York City", "Capital District, New York", "Southern Tier"],
    "Capital District, New York": [
        "Hudson Valley",
        "Mohawk Valley",
        "North Country (New York)",
        "Southern Tier",
    ],
    "Mohawk Valley": [
        "Capital District, New York",
        "North Country (New York)",
        "Central New York",
        "Southern Tier",
    ],
    "North Country (New York)": [
        "Capital District, New York",
        "Mohawk Valley",
        "Central New York",
    ],
    "Central New York": [
        "Mohawk Valley",
        "Finger Lakes",
        "Southern Tier",
        "North Country (New York)",
    ],
    "Finger Lakes": [
        "Central New York",
        "Western New York",
        "Southern Tier",
    ],
    "Western New York": ["Finger Lakes", "Southern Tier"],
    "Southern Tier": [
        "Hudson Valley",
        "Capital District, New York",
        "Mohawk Valley",
        "Central New York",
        "Finger Lakes",
        "Western New York",
    ],
}


def _validate_region_adj_symmetric(adj):
    for r, neighbors in adj.items():
        for n in neighbors:
            assert n in adj, f"neighbor '{n}' of '{r}' not in REGION_ADJ"
            assert r in adj[n], f"asymmetric: '{r}' lists '{n}' but not vice versa"


def add_geo_graph_features(X_train, X_pred, y_train, target_col="PERCENT_PROFICIENT"):
    """
    Level 1: leak-safe neighbor-aggregated target means (using train rows only).
    Level 2: spectral embeddings of the region adjacency graph (no leakage, structural only).

    Adds columns:
        neighbor_mean_y, neighbor_std_y, neighbor_n_regions
        region_emb_0, region_emb_1, region_emb_2, region_emb_3
    """
    _validate_region_adj_symmetric(REGION_ADJ)

    # extract y
    if isinstance(y_train, pl.DataFrame):
        y_series = (
            y_train[target_col]
            if target_col in y_train.columns
            else y_train.to_series()
        )
    else:
        y_series = y_train
    y_np = y_series.to_numpy().astype(np.float64)

    # ── Level 1: leak-safe neighbor target mean ──────────────────────────────
    # Compute per-region mean y from train only.
    train_regions = X_train["REGION"].cast(pl.String).to_numpy()
    region_to_ys = {}
    for r, y in zip(train_regions, y_np):
        region_to_ys.setdefault(r, []).append(y)
    region_mean = {r: float(np.mean(ys)) for r, ys in region_to_ys.items()}
    global_mean = float(np.mean(y_np))

    def neighbor_stats(region):
        """Return (mean, std, n) over neighbors of region. Falls back to global mean."""
        neighbors = REGION_ADJ.get(region, [])
        vals = [region_mean[n] for n in neighbors if n in region_mean]
        if not vals:
            return global_mean, 0.0, 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    # Apply to both
    def build_neighbor_cols(regions):
        n = len(regions)
        means = np.empty(n, dtype=np.float64)
        stds = np.empty(n, dtype=np.float64)
        ns = np.empty(n, dtype=np.int32)
        for i, r in enumerate(regions):
            means[i], stds[i], ns[i] = neighbor_stats(r)
        return means, stds, ns

    tr_mean, tr_std, tr_n = build_neighbor_cols(train_regions)
    pred_regions = X_pred["REGION"].cast(pl.String).to_numpy()
    pr_mean, pr_std, pr_n = build_neighbor_cols(pred_regions)

    # ── Level 2: Laplacian spectral embeddings ───────────────────────────────
    G = nx.Graph()
    for r in REGION_ADJ:
        G.add_node(r)
    for r, neighbors in REGION_ADJ.items():
        for n in neighbors:
            G.add_edge(r, n)

    # normalized Laplacian, skip first trivial eigenvector (eigenvalue 0)
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    K = 4  # number of embedding dims
    node_list = list(G.nodes())
    embeddings = eigvecs[:, 1 : K + 1]  # n_regions × K
    emb_map = {node: embeddings[i] for i, node in enumerate(node_list)}

    def build_emb_cols(regions):
        n = len(regions)
        out = np.zeros((n, K), dtype=np.float64)
        for i, r in enumerate(regions):
            if r in emb_map:
                out[i] = emb_map[r]
        return out

    tr_emb = build_emb_cols(train_regions)
    pr_emb = build_emb_cols(pred_regions)

    # ── Attach columns ───────────────────────────────────────────────────────
    new_train_cols = [
        pl.Series("neighbor_mean_y", tr_mean.astype(np.float32)),
        pl.Series("neighbor_std_y", tr_std.astype(np.float32)),
        pl.Series("neighbor_n_regions", tr_n),
    ]
    new_pred_cols = [
        pl.Series("neighbor_mean_y", pr_mean.astype(np.float32)),
        pl.Series("neighbor_std_y", pr_std.astype(np.float32)),
        pl.Series("neighbor_n_regions", pr_n),
    ]
    for k in range(K):
        new_train_cols.append(
            pl.Series(f"region_emb_{k}", tr_emb[:, k].astype(np.float32))
        )
        new_pred_cols.append(
            pl.Series(f"region_emb_{k}", pr_emb[:, k].astype(np.float32))
        )

    X_train = X_train.with_columns(new_train_cols)
    X_pred = X_pred.with_columns(new_pred_cols)
    return X_train, X_pred


def add_target_cross_features(
    X_train, X_pred, y_train, target_col="PERCENT_PROFICIENT"
):
    """
    Leak-safe cross-row target aggregations.
    y_train: polars DF or Series with target. Aligned by row order with X_train.
    """
    # attach y to a working copy of X_train
    if isinstance(y_train, pl.DataFrame):
        y_series = (
            y_train[target_col]
            if target_col in y_train.columns
            else y_train.to_series()
        )
    else:
        y_series = y_train

    Xt = X_train.with_columns(y_series.alias("__y"))
    global_mean = Xt.select(pl.col("__y").mean()).item()

    # ---- TRAIN: LOO features ----
    Xt = (
        Xt.with_columns(
            _sa_sum=pl.col("__y").sum().over(["SCHOOL", "ASSESSMENT_NAME"]),
            _sa_n=pl.len().over(["SCHOOL", "ASSESSMENT_NAME"]),
            _s_sum=pl.col("__y").sum().over("SCHOOL"),
            _s_n=pl.len().over("SCHOOL"),
        )
        .with_columns(
            siblings_mean_y=pl.when(pl.col("_sa_n") > 1)
            .then((pl.col("_sa_sum") - pl.col("__y")) / (pl.col("_sa_n") - 1))
            .otherwise(None),
            siblings_n=(pl.col("_sa_n") - 1).cast(pl.Int32),
            school_loo_mean_y=pl.when(pl.col("_s_n") > 1)
            .then((pl.col("_s_sum") - pl.col("__y")) / (pl.col("_s_n") - 1))
            .otherwise(None),
            school_loo_n=(pl.col("_s_n") - 1).cast(pl.Int32),
            school_std_y=pl.col("__y").std().over("SCHOOL").fill_null(0.0),
            siblings_std_y=pl.col("__y")
            .std()
            .over(["SCHOOL", "ASSESSMENT_NAME"])
            .fill_null(0.0),
        )
        .drop(["_sa_sum", "_sa_n", "_s_sum", "_s_n"])
    )

    # All Students y as feature (masked when self IS All Students)
    all_students_lookup = (
        Xt.filter(pl.col("SUBGROUP_NAME") == "All Students")
        .group_by(["SCHOOL", "ASSESSMENT_NAME"])
        .agg(pl.col("__y").first().alias("all_students_y"))
    )
    Xt = Xt.join(all_students_lookup, on=["SCHOOL", "ASSESSMENT_NAME"], how="left")
    Xt = Xt.with_columns(
        all_students_y=pl.when(pl.col("SUBGROUP_NAME") == "All Students")
        .then(None)
        .otherwise(pl.col("all_students_y"))
    )

    # Fallback hierarchy
    Xt = Xt.with_columns(
        siblings_mean_y=pl.col("siblings_mean_y")
        .fill_null(pl.col("school_loo_mean_y"))
        .fill_null(global_mean),
        school_loo_mean_y=pl.col("school_loo_mean_y").fill_null(global_mean),
    ).with_columns(
        all_students_y=pl.col("all_students_y").fill_null(pl.col("siblings_mean_y")),
    )

    # drop helper, return clean train
    X_train_out = Xt.drop("__y")

    # ---- TEST: full train aggs ----
    sa_agg = Xt.group_by(["SCHOOL", "ASSESSMENT_NAME"]).agg(
        pl.col("__y").mean().alias("siblings_mean_y"),
        pl.col("__y").std().fill_null(0.0).alias("siblings_std_y"),
        pl.len().alias("siblings_n"),
    )
    school_agg = Xt.group_by("SCHOOL").agg(
        pl.col("__y").mean().alias("school_loo_mean_y"),
        pl.col("__y").std().fill_null(0.0).alias("school_std_y"),
        pl.len().alias("school_loo_n"),
    )

    X_pred_out = (
        X_pred.join(sa_agg, on=["SCHOOL", "ASSESSMENT_NAME"], how="left")
        .join(school_agg, on="SCHOOL", how="left")
        .join(all_students_lookup, on=["SCHOOL", "ASSESSMENT_NAME"], how="left")
    )

    X_pred_out = X_pred_out.with_columns(
        siblings_mean_y=pl.col("siblings_mean_y")
        .fill_null(pl.col("school_loo_mean_y"))
        .fill_null(global_mean),
        school_loo_mean_y=pl.col("school_loo_mean_y").fill_null(global_mean),
    ).with_columns(
        all_students_y=pl.col("all_students_y").fill_null(pl.col("siblings_mean_y")),
    )

    return X_train_out, X_pred_out


def cross_county_dtype(df):
    return df.with_columns(
        (
            pl.col("COUNTY").cast(pl.Utf8)
            + "_x_"
            + pl.col("DISTRICT_TYPE").cast(pl.Utf8)
        ).alias("COUNTY_x_DTYPE")
    )


def cross_county_assessment(df):
    return df.with_columns(
        (
            pl.col("COUNTY").cast(pl.Utf8)
            + "_x_"
            + pl.col("ASSESSMENT_NAME").cast(pl.Utf8)
        ).alias("COUNTY_x_ASSESSMENT_NAME")
    )


def cross_dtype_assessment(df):
    return df.with_columns(
        (
            pl.col("DISTRICT_TYPE").cast(pl.Utf8)
            + "_x_"
            + pl.col("ASSESSMENT_NAME").cast(pl.Utf8)
        ).alias("DISTRICT_TYPE_x_ASSESSMENT_NAME")
    )


def cross_dtype_subgroup(df):
    return df.with_columns(
        (
            pl.col("DISTRICT_TYPE").cast(pl.Utf8)
            + "_x_"
            + pl.col("SUBGROUP_NAME").cast(pl.Utf8)
        ).alias("DISTRICT_TYPE_x_SUBGROUP_NAME")
    )


def cross_dtype_region(df):
    return df.with_columns(
        (
            pl.col("DISTRICT_TYPE").cast(pl.Utf8)
            + "_x_"
            + pl.col("REGION").cast(pl.Utf8)
        ).alias("DISTRICT_TYPE_x_REGION")
    )


def cross_assessment_subgroup(df):
    return df.with_columns(
        (
            pl.col("ASSESSMENT_NAME").cast(pl.Utf8)
            + "_x_"
            + pl.col("SUBGROUP_NAME").cast(pl.Utf8)
        ).alias("ASSESSMENT_NAME_x_SUBGROUP_NAME")
    )


def df_district_type_level_gb(df, col_to_use, new_col_name):
    df_new = df.filter(pl.col(col_to_use).is_not_null())
    if col_to_use not in zero_meaningful_cols:
        df_new = df_new.filter(pl.col(col_to_use).gt(0))
    return df_new.group_by(["DISTRICT_TYPE"]).agg(
        pl.col(col_to_use).median().alias(new_col_name)
    )


def df_county_level_gb(df, col_to_use, new_col_name):
    df_new = df.filter(pl.col(col_to_use).is_not_null())
    if col_to_use not in zero_meaningful_cols:
        df_new = df_new.filter(pl.col(col_to_use).gt(0))
    return df_new.group_by(["COUNTY"]).agg(
        pl.col(col_to_use).median().alias(new_col_name)
    )


def df_region_level_gb(df, col_to_use, new_col_name):
    df_new = df.filter(pl.col(col_to_use).is_not_null())
    if col_to_use not in zero_meaningful_cols:
        df_new = df_new.filter(pl.col(col_to_use).gt(0))
    return df_new.group_by(["REGION"]).agg(
        pl.col(col_to_use).median().alias(new_col_name)
    )


def df_district_level_gb(df, col_to_use, new_col_name):
    df_new = df.filter(pl.col(col_to_use).is_not_null())
    if col_to_use not in zero_meaningful_cols:
        df_new = df_new.filter(pl.col(col_to_use).gt(0))
    return df_new.group_by(["DISTRICT"]).agg(
        pl.col(col_to_use).median().alias(new_col_name)
    )


def df_macro_region_level_gb(df, col_to_use, new_col_name):
    df_new = df.filter(pl.col(col_to_use).is_not_null())
    if col_to_use not in zero_meaningful_cols:
        df_new = df_new.filter(pl.col(col_to_use).gt(0))
    return df_new.group_by(["MACRO_REGION"]).agg(
        pl.col(col_to_use).median().alias(new_col_name)
    )


def df_state_level_gb(df, col_to_use, new_col_name):
    df_new = df.filter(pl.col(col_to_use).is_not_null())
    if col_to_use not in zero_meaningful_cols:
        df_new = df_new.filter(pl.col(col_to_use).gt(0))
    return df_new.select(pl.col(col_to_use).median().alias(new_col_name)).with_columns(
        pl.lit(1).alias("_state_key")
    )


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
        **{c: f"median_district_{c.lower()}" for c in demographic_cols},
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
        **{c: f"median_county_{c.lower()}" for c in demographic_cols},
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
        **{c: f"median_region_{c.lower()}" for c in demographic_cols},
    }

    state_level_columns_to_aggregate = {
        "GRADE_1_AVERAGE_CLASS_SIZE": "state_median_grade_1_size",
        "GRADE_2_AVERAGE_CLASS_SIZE": "state_median_grade_2_size",
        "KINDERGARTEN_AVERAGE_CLASS_SIZE": "state_median_kindergarten_class_size",
        "PRE_K": "state_median_pre_K_students",
        "K": "state_median_kindergarten_students",
        **{f"GRADE_{i:02d}": f"state_median_grade_{i}_students" for i in range(1, 13)},
        "ATTENDANCE_RATE": "state_median_attendance_rate",
        "TEACHER_TURNOVER_RATE": "state_median_teacher_turnover_rate",
        "LANGUAGE_ARTS_AVERAGE_CLASS_SIZE": "state_median_language_class_size",
        "MATHEMATICS_AVERAGE_CLASS_SIZE": "state_median_math_class_size",
        "SCIENCE_AVERAGE_CLASS_SIZE": "state_median_science_class_size",
        "HISTORY_GOVERNMENT_AND_GEOGRAPHY_AVERAGE_CLASS_SIZE": "state_median_class_size",
        "PERCENT_FREE_LUNCH": "state_median_percent_free_lunch",
        "PERCENT_REDUCED_LUNCH": "state_median_reduced_lunch",
        "NUMBER_OF_TEACHERS": "state_median_number_of_teachers",
        "NUMBER_OF_COUNSELORS": "state_median_number_of_counselors",
        "NUMBER_OF_SOCIAL_WORKERS": "state_median_number_of_social_workers",
        "N_PUPILS": "state_median_number_of_pupils",
        **{c: f"median_state_{c.lower()}" for c in demographic_cols},
    }

    district_level_columns_to_aggregate = {
        "GRADE_1_AVERAGE_CLASS_SIZE": "district_level_median_grade_1_size",
        "GRADE_2_AVERAGE_CLASS_SIZE": "district_level_median_grade_2_size",
        "KINDERGARTEN_AVERAGE_CLASS_SIZE": "district_level_median_kindergarten_class_size",
        "PRE_K": "district_level_median_pre_K_students",
        "K": "district_level_median_kindergarten_students",
        **{
            f"GRADE_{i:02d}": f"district_level_median_grade_{i}_students"
            for i in range(1, 13)
        },
        "ATTENDANCE_RATE": "district_level_median_attendance_rate",
        "TEACHER_TURNOVER_RATE": "district_level_median_teacher_turnover_rate",
        "LANGUAGE_ARTS_AVERAGE_CLASS_SIZE": "district_level_median_language_class_size",
        "MATHEMATICS_AVERAGE_CLASS_SIZE": "district_level_median_math_class_size",
        "SCIENCE_AVERAGE_CLASS_SIZE": "district_level_median_science_class_size",
        "HISTORY_GOVERNMENT_AND_GEOGRAPHY_AVERAGE_CLASS_SIZE": "district_level_median_class_size",
        "PERCENT_FREE_LUNCH": "district_level_median_percent_free_lunch",
        "PERCENT_REDUCED_LUNCH": "district_level_median_reduced_lunch",
        "NUMBER_OF_TEACHERS": "district_level_median_number_of_teachers",
        "NUMBER_OF_COUNSELORS": "district_level_median_number_of_counselors",
        "NUMBER_OF_SOCIAL_WORKERS": "district_level_median_number_of_social_workers",
        "N_PUPILS": "district_level_median_number_of_pupils",
    }

    macro_region_level_columns_to_aggregate = {
        "GRADE_1_AVERAGE_CLASS_SIZE": "macro_region_level_median_grade_1_size",
        "GRADE_2_AVERAGE_CLASS_SIZE": "macro_region_level_median_grade_2_size",
        "KINDERGARTEN_AVERAGE_CLASS_SIZE": "macro_region_level_median_kindergarten_class_size",
        "PRE_K": "macro_region_level_median_pre_K_students",
        "K": "macro_region_level_median_kindergarten_students",
        **{
            f"GRADE_{i:02d}": f"macro_region_level_median_grade_{i}_students"
            for i in range(1, 13)
        },
        "ATTENDANCE_RATE": "macro_region_level_median_attendance_rate",
        "TEACHER_TURNOVER_RATE": "macro_region_level_median_teacher_turnover_rate",
        "LANGUAGE_ARTS_AVERAGE_CLASS_SIZE": "macro_region_level_median_language_class_size",
        "MATHEMATICS_AVERAGE_CLASS_SIZE": "macro_region_level_median_math_class_size",
        "SCIENCE_AVERAGE_CLASS_SIZE": "macro_region_level_median_science_class_size",
        "HISTORY_GOVERNMENT_AND_GEOGRAPHY_AVERAGE_CLASS_SIZE": "macro_region_level_median_class_size",
        "PERCENT_FREE_LUNCH": "macro_region_level_median_percent_free_lunch",
        "PERCENT_REDUCED_LUNCH": "macro_region_level_median_reduced_lunch",
        "NUMBER_OF_TEACHERS": "macro_region_level_median_number_of_teachers",
        "NUMBER_OF_COUNSELORS": "macro_region_level_median_number_of_counselors",
        "NUMBER_OF_SOCIAL_WORKERS": "macro_region_level_median_number_of_social_workers",
        "N_PUPILS": "macro_region_level_median_number_of_pupils",
        **{c: f"median_macro_region_{c.lower()}" for c in demographic_cols},
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

    school_imputed = school  # oops

    school_imputed = school_imputed.join(
        district.select(["DISTRICT"] + district_demographic_cols),
        on="DISTRICT",
        how="left",
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

    res = res.with_columns(pl.lit(1).alias("_state_key"))

    for col, alias in state_level_columns_to_aggregate.items():
        agg_df = df_state_level_gb(school_imputed, col, alias)
        res = res.join(agg_df, on="_state_key", how="left")
    res = res.drop("_state_key")

    for key, tbl in funding_tables.items():
        level = key.split("_per_")[1]
        join_col = level.upper()
        res = res.join(tbl, on=join_col, how="left")

    for group_col, count_col, alias in x_per_y_unique_aggs:
        agg_df = df_total_unique_x_per_y(school_imputed, group_col, count_col, alias)
        res = res.join(agg_df, on=group_col, how="left")

    for col, alias in district_type_level_columns_to_aggregate.items():
        agg_df = df_district_type_level_gb(school_imputed, col, alias)
        res = res.join(agg_df, on=["DISTRICT_TYPE"], how="left")

    for col, alias in county_level_columns_to_aggregate.items():
        agg_df = df_county_level_gb(school_imputed, col, alias)
        res = res.join(agg_df, on=["COUNTY"], how="left")

    for col, alias in region_level_columns_to_aggregate.items():
        agg_df = df_region_level_gb(school_imputed, col, alias)
        res = res.join(agg_df, on=["REGION"], how="left")

    for col, alias in district_level_columns_to_aggregate.items():
        agg_df = df_district_level_gb(school_imputed, col, alias)
        res = res.join(agg_df, on=["DISTRICT"], how="left")

    for col, alias in macro_region_level_columns_to_aggregate.items():
        agg_df = df_macro_region_level_gb(school_imputed, col, alias)
        res = res.join(agg_df, on=["MACRO_REGION"], how="left")

    res = res.with_columns(
        district_level_median_teacher_turnover_rate=pl.col(
            "district_level_median_teacher_turnover_rate"
        )
        .fill_null(pl.col("county_median_teacher_turnover_rate"))
        .fill_null(pl.col("region_level_median_teacher_turnover_rate"))
        .fill_null(pl.col("macro_region_level_median_teacher_turnover_rate"))
    )
    res = res.with_columns(
        district_type_level_median_teacher_turnover_rate=pl.col(
            "district_type_level_median_teacher_turnover_rate"
        )
        .fill_null(pl.col("county_median_teacher_turnover_rate"))
        .fill_null(pl.col("region_level_median_teacher_turnover_rate"))
        .fill_null(pl.col("macro_region_level_median_teacher_turnover_rate"))
    )

    res = res.with_columns(
        region_level_median_teacher_turnover_rate=pl.col(
            "region_level_median_teacher_turnover_rate"
        ).fill_null(pl.col("macro_region_level_median_teacher_turnover_rate"))
    )

    school_assessment_n_rows = (
        res.group_by(["SCHOOL", "ASSESSMENT_NAME"])
        .len()
        .rename({"len": "num_tests_for_school"})
    )

    res = res.join(
        school_assessment_n_rows, on=["SCHOOL", "ASSESSMENT_NAME"], how="left"
    )

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
            INFORMATION_MISSING=pl.when(
                (pl.sum_horizontal(pl.all().is_null()).gt(0))
                & (pl.sum_horizontal(pl.all().is_null()).le(5))
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
            INFORMATION_LIKELY_WITHHELD_FLAG=pl.when(
                pl.sum_horizontal(pl.all().is_null()).gt(
                    8
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
    school = school.with_columns(
        pl.col("DISTRICT_TYPE")
        .cast(pl.String)
        .eq("Charter School")
        .cast(pl.Int8)
        .alias("is_charter")
    )

    school = school.with_columns(
        is_high_need=pl.when(
            pl.col("DISTRICT_TYPE")
            .cast(pl.String)
            .is_in(["High-Need Rural", "High-Need Urban/Suburban"])
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    ).with_columns(
        is_rural=pl.when(pl.col("DISTRICT_TYPE").eq("High-Need Rural"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )
    school = school.with_columns(
        is_low_need=pl.when(pl.col("DISTRICT_TYPE").cast(pl.String).is_in(["Low Need"]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        is_aveage_need=pl.when(
            pl.col("DISTRICT_TYPE").cast(pl.String).is_in(["Average Need"])
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
    )

    school = school.with_columns(
        is_large_city=pl.when(
            pl.col("DISTRICT_TYPE").cast(pl.String).is_in(["NYC", "Other Large City"])
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        is_charter_school=pl.when(
            pl.col("DISTRICT_TYPE").cast(pl.String).is_in(["Charter School"])
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
    )

    MACRO_REGION = {
        "New York City": "downstate_urban",
        "Long Island": "downstate_suburb",
        "Hudson Valley": "downstate_suburb",
        "Western New York": "upstate_corridor",
        "Finger Lakes": "upstate_corridor",
        "Central New York": "upstate_corridor",
        "Mohawk Valley": "upstate_corridor",
        "Capital District, New York": "upstate_corridor",
        "Southern Tier": "rural_fringe",
        "North Country (New York)": "rural_fringe",
    }

    school = (
        school.with_columns(
            pl.col("REGION")
            .cast(pl.String)
            .replace_strict(MACRO_REGION)
            .cast(pl.Categorical)
            .alias("MACRO_REGION")
        )
        .with_columns(
            is_rural=pl.when(pl.col("MACRO_REGION").eq("rural_fringe"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_upstate=pl.when(
                (pl.col("MACRO_REGION").cast(pl.String).str.contains("upstate"))
                | (
                    pl.col("REGION")
                    .cast(pl.String)
                    .str.contains("North Country (New York)")
                )
                | (pl.col("REGION").cast(pl.String).str.contains("Southern Tier"))
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_down=pl.when(
                (pl.col("MACRO_REGION").cast(pl.String).str.contains("downstate"))
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
    ).with_columns(
        is_nyc=pl.when(pl.col("REGION").eq("New York City"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )

    macro_region_pupil_share = school.group_by("MACRO_REGION").agg(
        (pl.col("N_PUPILS").sum() / 2_412_693).alias(
            "macro_region_share_of_state_pupils"
        )  # 2412693 calculated by hand
    )
    district_pupil_share = school.group_by("DISTRICT").agg(
        (pl.col("N_PUPILS").sum() / 2_412_693).alias("district_share_of_state_pupils")
    )
    district_type_pupil_share = school.group_by("DISTRICT_TYPE").agg(
        (pl.col("N_PUPILS").sum() / 2_412_693).alias(
            "district_type_share_of_state_pupils"
        )
    )

    county_pupil_share = school.group_by("COUNTY").agg(
        (pl.col("N_PUPILS").sum() / 2_412_693).alias("county_share_of_state_pupils")
    )

    region_pupil_share = school.group_by("REGION").agg(
        (pl.col("N_PUPILS").sum() / 2_412_693).alias("region_share_of_state_pupils")
    )
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
    ).with_columns((pl.col("N_PUPILS") / 2_412_693).alias("share_of_state_pupils"))

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

    X_train = (
        (
            (
                add_funding_ratio_cols(X_train)
                .with_columns(
                    is_regent=pl.when(
                        pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Regent")
                    )
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                )
                .with_columns(
                    is_agg_level_stat=pl.when(
                        pl.col("SUBGROUP_NAME").eq("All Students")
                    )
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                )
            )
            .with_columns(
                is_ela=pl.when(
                    (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("ELA"))
                    | (
                        pl.col("ASSESSMENT_NAME")
                        .cast(pl.String)
                        .str.contains("English")
                    )
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
            )
            .with_columns(
                is_math=pl.when(
                    (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Math"))
                    | (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("MATH"))
                    | (
                        pl.col("ASSESSMENT_NAME")
                        .cast(pl.String)
                        .str.contains("Geometry")
                    )
                    | (
                        pl.col("ASSESSMENT_NAME")
                        .cast(pl.String)
                        .str.contains("Algebra")
                    )
                )
                .then(pl.lit(1))
                .otherwise(0)
            )
        )
        .with_columns(
            is_common_core=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Common Core")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_science=pl.when(
                (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Science"))
                | (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Phy Set"))
                | (
                    pl.col("ASSESSMENT_NAME")
                    .cast(pl.String)
                    .str.contains("Living Environment")
                )
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_eight=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("8")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_seven=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("7")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_six=pl.when(pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("6"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_five=pl.when(pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("5"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_four=pl.when(pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("4"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_three=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("3")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_combined=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Combined")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_history=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("History")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_algebra=pl.when(
                (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Algebra"))
            )
            .then(pl.lit(1))
            .otherwise(0)
        )
        .with_columns(
            is_algebra_2=pl.when(
                (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("II"))
            )
            .then(pl.lit(1))
            .otherwise(0)
        )
        .with_columns(
            is_physics=pl.when(
                (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Physics"))
            )
            .then(pl.lit(1))
            .otherwise(0)
        )
        .with_columns(
            is_chem=pl.when(
                (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Chemistry"))
            )
            .then(pl.lit(1))
            .otherwise(0)
        )
    )

    y_train = train.select(pl.col("PERCENT_PROFICIENT"))

    X_pred = join_tables(test, school, district)
    X_pred = (
        (
            add_funding_ratio_cols(X_pred)
            .with_columns(
                is_regent=pl.when(
                    pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Regent")
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
            )
            .with_columns(
                is_agg_level_stat=pl.when(pl.col("SUBGROUP_NAME").eq("All Students"))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
            )
            .with_columns(
                is_ela=pl.when(
                    (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("ELA"))
                    | (
                        pl.col("ASSESSMENT_NAME")
                        .cast(pl.String)
                        .str.contains("English")
                    )
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
            )
            .with_columns(
                is_math=pl.when(
                    (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Math"))
                    | (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("MATH"))
                    | (
                        pl.col("ASSESSMENT_NAME")
                        .cast(pl.String)
                        .str.contains("Geometry")
                    )
                    | (
                        pl.col("ASSESSMENT_NAME")
                        .cast(pl.String)
                        .str.contains("Algebra")
                    )
                )
                .then(pl.lit(1))
                .otherwise(0)
            )
            .with_columns(
                is_algebra=pl.when(
                    (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Algebra"))
                )
                .then(pl.lit(1))
                .otherwise(0)
            )
            .with_columns(
                is_common_core=pl.when(
                    pl.col("ASSESSMENT_NAME")
                    .cast(pl.String)
                    .str.contains("Common Core")
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
            )
            .with_columns(
                is_algebra_2=pl.when(
                    (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("II"))
                )
                .then(pl.lit(1))
                .otherwise(0)
            )
            .with_columns(
                is_algebra=pl.when(
                    (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Algebra"))
                )
                .then(pl.lit(1))
                .otherwise(0)
            )
            .with_columns(
                is_science=pl.when(
                    (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Science"))
                    | (
                        pl.col("ASSESSMENT_NAME")
                        .cast(pl.String)
                        .str.contains("Phy Set")
                    )
                    | (
                        pl.col("ASSESSMENT_NAME")
                        .cast(pl.String)
                        .str.contains("Living Environment")
                    )
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
            )
        )
        .with_columns(
            is_combined=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Combined")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_eight=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("8")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_seven=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("7")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_six=pl.when(pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("6"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_five=pl.when(pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("5"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_four=pl.when(pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("4"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_three=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("3")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_history=pl.when(
                pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("History")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        )
        .with_columns(
            is_physics=pl.when(
                (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Physics"))
            )
            .then(pl.lit(1))
            .otherwise(0)
        )
        .with_columns(
            is_chem=pl.when(
                (pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("Chemistry"))
            )
            .then(pl.lit(1))
            .otherwise(0)
        )
    )

    # === Estimated SE: per-row, using district mean as p_est and the row's own N_STUDENTS ===
    district_p_est = X_train.group_by("DISTRICT").agg(
        pl.col("PERCENT_PROFICIENT").mean().alias("district_p_est")
    )
    X_train = X_train.join(district_p_est, on="DISTRICT", how="left").with_columns(
        estimated_se=(
            (
                pl.col("district_p_est")
                * (100 - pl.col("district_p_est"))
                / pl.max_horizontal(pl.col("N_STUDENTS"), pl.lit(1))
            ).sqrt()
            / 100
        ),
        log1p_n_students=pl.col("N_STUDENTS").log1p(),
    )
    X_pred = X_pred.join(district_p_est, on="DISTRICT", how="left").with_columns(
        estimated_se=(
            (
                pl.col("district_p_est")
                * (100 - pl.col("district_p_est"))
                / pl.max_horizontal(pl.col("N_STUDENTS"), pl.lit(1))
            ).sqrt()
            / 100
        ),
        log1p_n_students=pl.col("N_STUDENTS").log1p(),
    )

    # === Drop helper column and the target before saving ===
    X_train = (
        X_train.drop("district_p_est")
        .drop("PERCENT_PROFICIENT")
        .fill_nan(0)
        .fill_null(0)
    )
    X_pred = X_pred.drop("district_p_est").fill_nan(0).fill_null(0)
    X_train = X_train.select(pl.exclude("^.*right$"))
    X_pred = X_pred.select(pl.exclude("^.*right$"))
    X_pred_id = test.select(pl.col("ASSESSMENT_ID"))

    # autofeat columns
    X_train = X_train.with_columns(
        [
            pl.col("PERCENT_ECONOMICALLY_DISADVANTAGED")
            .pow(2)
            .alias("PERCENT_ECONOMICALLY_DISADVANTAGED__2"),
            pl.col("PERCENT_HOMELESS").pow(2).alias("PERCENT_HOMELESS__2"),
            pl.col("PERCENT_HOMELESS").pow(3).alias("PERCENT_HOMELESS__3"),
            pl.col("PERCENT_OF_STUDENTS_SUSPENDED")
            .pow(2)
            .alias("PERCENT_OF_STUDENTS_SUSPENDED__2"),
            pl.col("PERCENT_ENGLISH_LANGUAGE_LEANERS")
            .pow(2)
            .alias("PERCENT_ENGLISH_LANGUAGE_LEANERS__2"),
            pl.col("GRADE_2_AVERAGE_CLASS_SIZE")
            .pow(2)
            .alias("GRADE_2_AVERAGE_CLASS_SIZE__2"),
            pl.col("student_teacher_ratio").pow(3).alias("student_teacher_ratio__3"),
            pl.col("fed_funding_per_school")
            .sqrt()
            .alias("sqrt_fed_funding_per_school"),
            pl.col("county_median_reduced_lunch")
            .exp()
            .alias("exp_county_median_reduced_lunch"),
            pl.col("num_tests_for_school").exp().alias("exp_num_tests_for_school"),
            pl.col("NUMBER_OF_SOCIAL_WORKERS")
            .pow(3)
            .alias("NUMBER_OF_SOCIAL_WORKERS__3"),
            pl.col("median_district_percent_dropout")
            .pow(3)
            .alias("median_district_percent_dropout__3"),
            pl.col("N_STUDENTS").pow(2).alias("N_STUDENTS__2"),
            pl.when(
                pl.col("N_PUPILS").is_not_null()
                & pl.col("N_PUPILS").gt(0)
                & pl.col("N_STUDENTS").is_not_null()
            )
            .then(pl.col("N_STUDENTS").truediv(pl.col("N_PUPILS")))
            .otherwise(pl.lit(0))
            .alias("size_of_group"),
        ]
    ).with_columns(
        size_of_3=pl.when(
            (pl.col("is_three").eq(1))
            & pl.col("GRADE_03").is_not_null()
            & pl.col("GRADE_03").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_03")))
        .otherwise(pl.lit(0)),
        size_of_4=pl.when(
            (pl.col("is_four").eq(1))
            & pl.col("GRADE_04").is_not_null()
            & pl.col("GRADE_04").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_04")))
        .otherwise(pl.lit(0)),
        size_of_5=pl.when(
            (pl.col("is_five").eq(1))
            & pl.col("GRADE_05").is_not_null()
            & pl.col("GRADE_05").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_05")))
        .otherwise(pl.lit(0)),
        size_of_6=pl.when(
            (pl.col("is_six").eq(1))
            & pl.col("GRADE_06").is_not_null()
            & pl.col("GRADE_06").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_06")))
        .otherwise(pl.lit(0)),
        size_of_7=pl.when(
            (pl.col("is_seven").eq(1))
            & pl.col("GRADE_07").is_not_null()
            & pl.col("GRADE_07").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_07")))
        .otherwise(pl.lit(0)),
        size_of_8=pl.when(
            (pl.col("is_eight").eq(1))
            & pl.col("GRADE_08").is_not_null()
            & pl.col("GRADE_08").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_08")))
        .otherwise(pl.lit(0)),
        size_of_regent=pl.when(
            (pl.col("is_regent").eq(1))
            & pl.col("GRADE_09").is_not_null()
            & pl.col("GRADE_10").is_not_null()
            & pl.col("GRADE_11").is_not_null()
            & pl.col("GRADE_12").is_not_null()
            & (
                pl.col("GRADE_09")
                .add(pl.col("GRADE_10"))
                .add(pl.col("GRADE_11"))
                .add(pl.col("GRADE_12"))
            ).gt(0)
        )
        .then(
            pl.col("N_STUDENTS").truediv(
                pl.col("GRADE_09")
                .add(pl.col("GRADE_10"))
                .add(pl.col("GRADE_11"))
                .add(pl.col("GRADE_12"))
            )
        )
        .otherwise(pl.lit(0)),
        high_need_regents=pl.when(
            (pl.col("is_high_need").eq(1)) & (pl.col("is_regent").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        nyc_regents=pl.when((pl.col("is_nyc").eq(1)) & (pl.col("is_regent").eq(1)))
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        is_large_city_regents=pl.when(
            (pl.col("is_large_city").eq(1)) & (pl.col("is_regent").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        is_average_needs_regent=pl.when(
            (pl.col("DISTRICT_TYPE").eq("Average Need")) & (pl.col("is_regent").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        high_need_common_core=pl.when(
            (pl.col("is_high_need").eq(1)) & (pl.col("is_common_core").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        is_all_students_regent=pl.when(
            (pl.col("SUBGROUP_NAME").eq("All Students")) & (pl.col("is_regent").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        double_high_needs=pl.when(
            (pl.col("is_high_need").eq(1))
            & (pl.col("SUBGROUP_NAME").eq("Economically Disadvantaged"))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        is_history=pl.when(
            pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("History")
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
    )

    X_pred = X_pred.with_columns(
        [
            pl.col("PERCENT_ECONOMICALLY_DISADVANTAGED")
            .pow(2)
            .alias("PERCENT_ECONOMICALLY_DISADVANTAGED__2"),
            pl.col("PERCENT_HOMELESS").pow(2).alias("PERCENT_HOMELESS__2"),
            pl.col("PERCENT_HOMELESS").pow(3).alias("PERCENT_HOMELESS__3"),
            pl.col("PERCENT_OF_STUDENTS_SUSPENDED")
            .pow(2)
            .alias("PERCENT_OF_STUDENTS_SUSPENDED__2"),
            pl.col("PERCENT_ENGLISH_LANGUAGE_LEANERS")
            .pow(2)
            .alias("PERCENT_ENGLISH_LANGUAGE_LEANERS__2"),
            pl.col("GRADE_2_AVERAGE_CLASS_SIZE")
            .pow(2)
            .alias("GRADE_2_AVERAGE_CLASS_SIZE__2"),
            pl.col("student_teacher_ratio").pow(3).alias("student_teacher_ratio__3"),
            pl.col("fed_funding_per_school")
            .sqrt()
            .alias("sqrt_fed_funding_per_school"),
            pl.col("county_median_reduced_lunch")
            .exp()
            .alias("exp_county_median_reduced_lunch"),
            pl.col("num_tests_for_school").exp().alias("exp_num_tests_for_school"),
            pl.col("NUMBER_OF_SOCIAL_WORKERS")
            .pow(3)
            .alias("NUMBER_OF_SOCIAL_WORKERS__3"),
            pl.col("median_district_percent_dropout")
            .pow(3)
            .alias("median_district_percent_dropout__3"),
            pl.col("N_STUDENTS").pow(2).alias("N_STUDENTS__2"),
            pl.when(
                pl.col("N_PUPILS").is_not_null()
                & pl.col("N_PUPILS").gt(0)
                & pl.col("N_STUDENTS").is_not_null()
            )
            .then(pl.col("N_STUDENTS").truediv(pl.col("N_PUPILS")))
            .otherwise(pl.lit(0))
            .alias("size_of_group"),
        ]
    ).with_columns(
        size_of_3=pl.when(
            (pl.col("is_three").eq(1))
            & pl.col("GRADE_03").is_not_null()
            & pl.col("GRADE_03").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_03")))
        .otherwise(pl.lit(0)),
        size_of_4=pl.when(
            (pl.col("is_four").eq(1))
            & pl.col("GRADE_04").is_not_null()
            & pl.col("GRADE_04").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_04")))
        .otherwise(pl.lit(0)),
        size_of_5=pl.when(
            (pl.col("is_five").eq(1))
            & pl.col("GRADE_05").is_not_null()
            & pl.col("GRADE_05").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_05")))
        .otherwise(pl.lit(0)),
        size_of_6=pl.when(
            (pl.col("is_six").eq(1))
            & pl.col("GRADE_06").is_not_null()
            & pl.col("GRADE_06").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_06")))
        .otherwise(pl.lit(0)),
        size_of_7=pl.when(
            (pl.col("is_seven").eq(1))
            & pl.col("GRADE_07").is_not_null()
            & pl.col("GRADE_07").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_07")))
        .otherwise(pl.lit(0)),
        size_of_8=pl.when(
            (pl.col("is_eight").eq(1))
            & pl.col("GRADE_08").is_not_null()
            & pl.col("GRADE_08").gt(0)
        )
        .then(pl.col("N_STUDENTS").truediv(pl.col("GRADE_08")))
        .otherwise(pl.lit(0)),
        size_of_regent=pl.when(
            (pl.col("is_regent").eq(1))
            & pl.col("GRADE_09").is_not_null()
            & pl.col("GRADE_10").is_not_null()
            & pl.col("GRADE_11").is_not_null()
            & pl.col("GRADE_12").is_not_null()
            & (
                pl.col("GRADE_09")
                .add(pl.col("GRADE_10"))
                .add(pl.col("GRADE_11"))
                .add(pl.col("GRADE_12"))
            ).gt(0)
        )
        .then(
            pl.col("N_STUDENTS").truediv(
                pl.col("GRADE_09")
                .add(pl.col("GRADE_10"))
                .add(pl.col("GRADE_11"))
                .add(pl.col("GRADE_12"))
            )
        )
        .otherwise(pl.lit(0)),
        is_history=pl.when(
            pl.col("ASSESSMENT_NAME").cast(pl.String).str.contains("History")
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        high_need_regents=pl.when(
            (pl.col("is_high_need").eq(1)) & (pl.col("is_regent").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        nyc_regents=pl.when((pl.col("is_nyc").eq(1)) & (pl.col("is_regent").eq(1)))
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        is_large_city_regents=pl.when(
            (pl.col("is_large_city").eq(1)) & (pl.col("is_regent").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        is_average_needs_regent=pl.when(
            (pl.col("DISTRICT_TYPE").eq("Average Need")) & (pl.col("is_regent").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        is_all_students_regent=pl.when(
            (pl.col("SUBGROUP_NAME").eq("All Students")) & (pl.col("is_regent").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        high_need_common_core=pl.when(
            (pl.col("is_high_need").eq(1)) & (pl.col("is_common_core").eq(1))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
        double_high_needs=pl.when(
            (pl.col("is_high_need").eq(1))
            & (pl.col("SUBGROUP_NAME").eq("Economically Disadvantaged"))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0)),
    )
    demo_cols = [
        ("PERCENT_ECONOMICALLY_DISADVANTAGED", "percent_economically_disadvantaged"),
        ("PERCENT_ENGLISH_LANGUAGE_LEANERS", "percent_english_language_leaners"),
        ("PERCENT_WITH_DISABILITIES", "percent_with_disabilities"),
        ("PERCENT_BLACK", "percent_black"),
        ("PERCENT_HISPANIC", "percent_hispanic"),
        ("PERCENT_WHITE", "percent_white"),
        ("PERCENT_HOMELESS", "percent_homeless"),
        ("PERCENT_OF_STUDENTS_SUSPENDED", "percent_of_students_suspended"),
    ]

    levels = ["district", "county", "region", "macro_region", "state"]

    ratio_exprs = [
        # funding composition
        (
            pl.col("FEDERAL_FUNDING_PER_PUPIL")
            / (pl.col("rough_total_funding_per_pupil") + 1e-6)
        ).alias("federal_share_of_funding"),
        (
            pl.col("LOCAL_FUNDING_PER_PUPIL")
            / (pl.col("rough_total_funding_per_pupil") + 1e-6)
        ).alias("local_share_of_funding"),
        # attendance vs geographic peer
        (
            pl.col("ATTENDANCE_RATE")
            / (pl.col("district_level_median_attendance_rate") + 1e-6)
        ).alias("attendance_vs_district"),
        (
            pl.col("ATTENDANCE_RATE") / (pl.col("county_median_attendance_rate") + 1e-6)
        ).alias("attendance_vs_county"),
        (
            pl.col("ATTENDANCE_RATE")
            / (pl.col("region_level_median_attendance_rate") + 1e-6)
        ).alias("attendance_vs_region"),
        # turnover vs peer
        (
            pl.col("TEACHER_TURNOVER_RATE")
            / (pl.col("district_level_median_teacher_turnover_rate") + 1e-6)
        ).alias("turnover_vs_district"),
        (
            pl.col("TEACHER_TURNOVER_RATE")
            / (pl.col("county_median_teacher_turnover_rate") + 1e-6)
        ).alias("turnover_vs_county"),
        # staffing ratios
        (pl.col("N_PUPILS") / (pl.col("NUMBER_OF_COUNSELORS") + 1e-6)).alias(
            "pupils_per_counselor"
        ),
        (pl.col("N_PUPILS") / (pl.col("NUMBER_OF_SOCIAL_WORKERS") + 1e-6)).alias(
            "pupils_per_social_worker"
        ),
        # school size vs peer
        (
            pl.col("N_PUPILS")
            / (pl.col("district_level_median_number_of_pupils") + 1e-6)
        ).alias("school_size_vs_district"),
        (pl.col("N_PUPILS") / (pl.col("county_median_number_of_pupils") + 1e-6)).alias(
            "school_size_vs_county"
        ),
        # assessed fraction of school
        (pl.col("N_STUDENTS") / (pl.col("N_PUPILS") + 1e-6)).alias(
            "assessed_fraction_of_school"
        ),
        # free lunch vs peer
        (
            pl.col("PERCENT_FREE_LUNCH")
            / (pl.col("district_type_level_median_percent_free_lunch") + 1e-6)
        ).alias("free_lunch_vs_district_type"),
        (
            pl.col("PERCENT_FREE_LUNCH")
            / (pl.col("county_median_percent_free_lunch") + 1e-6)
        ).alias("free_lunch_vs_county"),
        # funding vs geographic peer
        (
            pl.col("rough_total_funding_per_pupil")
            / (pl.col("county_median_number_of_pupils") + 1e-6)
        ).alias("funding_vs_county_size"),
        (
            pl.col("total_funding_per_district")
            / (pl.col("total_funding_per_county") + 1e-6)
        ).alias("district_share_of_county_funding"),
    ]

    # demographic vs peer ratios across all levels
    for raw_col, slug in demo_cols:
        for level in levels:
            ratio_exprs.append(
                (pl.col(raw_col) / (pl.col(f"median_{level}_{slug}") + 1e-6)).alias(
                    f"{raw_col}_vs_{level}"
                )
            )

    X_train = X_train.with_columns(ratio_exprs).with_columns(
        is_econ_dis=pl.when(pl.col("SUBGROUP_NAME").eq("Economically Disadvantaged"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )
    X_pred = X_pred.with_columns(ratio_exprs).with_columns(
        is_econ_dis=pl.when(pl.col("SUBGROUP_NAME").eq("Economically Disadvantaged"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )

    X_train = X_train.join(
        macro_region_pupil_share, on="MACRO_REGION", how="left"
    ).with_columns(inverse_mr_share=1 - pl.col("macro_region_share_of_state_pupils"))
    X_pred = X_pred.join(
        macro_region_pupil_share, on="MACRO_REGION", how="left"
    ).with_columns(inverse_mr_share=1 - pl.col("macro_region_share_of_state_pupils"))

    X_train = X_train.join(
        district_pupil_share, on="DISTRICT", how="left"
    ).with_columns(inverse_district_share=1 - pl.col("district_share_of_state_pupils"))
    X_pred = X_pred.join(district_pupil_share, on="DISTRICT", how="left").with_columns(
        inverse_district_share=1 - pl.col("district_share_of_state_pupils")
    )

    X_train = X_train.join(
        district_type_pupil_share, on="DISTRICT_TYPE", how="left"
    ).with_columns(
        inverse_district_type_share=1 - pl.col("district_type_share_of_state_pupils")
    )
    X_pred = X_pred.join(
        district_type_pupil_share, on="DISTRICT_TYPE", how="left"
    ).with_columns(
        inverse_district_type_share=1 - pl.col("district_type_share_of_state_pupils")
    )

    X_train = X_train.join(county_pupil_share, on="COUNTY", how="left").with_columns(
        inverse_county_type_share=1 - pl.col("county_share_of_state_pupils")
    )
    X_pred = X_pred.join(county_pupil_share, on="COUNTY", how="left").with_columns(
        inverse_county_type_share=1 - pl.col("county_share_of_state_pupils")
    )

    X_train = X_train.join(region_pupil_share, on="REGION", how="left").with_columns(
        inverse_region_type_share=1 - pl.col("region_share_of_state_pupils")
    )
    X_pred = X_pred.join(region_pupil_share, on="REGION", how="left").with_columns(
        inverse_region_type_share=1 - pl.col("region_share_of_state_pupils")
    )

    X_train = cross_county_dtype(X_train)
    X_train = cross_county_assessment(X_train)
    X_train = cross_dtype_assessment(X_train)
    X_train = cross_dtype_subgroup(X_train)
    X_train = cross_assessment_subgroup(X_train)
    X_train = cross_dtype_region(X_train)

    X_pred = cross_county_dtype(X_pred)
    X_pred = cross_county_assessment(X_pred)
    X_pred = cross_dtype_assessment(X_pred)
    X_pred = cross_dtype_subgroup(X_pred)
    X_pred = cross_assessment_subgroup(X_pred)
    X_pred = cross_dtype_region(X_pred)

    X_train, X_pred = add_geo_graph_features(X_train, X_pred, y_train)
    X_train, X_pred = add_target_cross_features(X_train, X_pred, y_train)
    if cached:
        X_pred.write_parquet("X_pred.parquet")
        X_train.write_parquet("X_train.parquet")
        y_train.write_parquet("y_train.parquet")
        X_pred_id.write_parquet("X_pred_id.parquet")

        return None
    else:
        return X_pred, X_train, y_train, X_pred_id


if __name__ == "__main__":
    get_data()
