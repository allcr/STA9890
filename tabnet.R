library(tidyverse)
library(tidymodels)
library(tune)
library(yardstick)
library(dplyr)
library(parsnip)
library(torch)
install_torch()
library(tabnet)


set.seed(8675309)
torch_set_num_threads(parallel::detectCores())

X_train <- read.csv("X_train.csv")
y_train <- read.csv("y_train.csv")
X_pred <- read.csv("X_pred.csv")
X_pred_id <- read.csv("X_pred_id.csv")

factor_cols <- function(df) {
  df |> mutate(
    ASSESSMENT_NAME = factor(ASSESSMENT_NAME),
    COUNTY = factor(COUNTY),
    DISTRICT = factor(DISTRICT),
    SCHOOL = factor(SCHOOL),
    DISTRICT_TYPE = factor(DISTRICT_TYPE),
    REGION = factor(REGION),
    ASSESSMENT_ID = factor(ASSESSMENT_ID),
    SUBGROUP_NAME = factor(SUBGROUP_NAME),
    total_schools_per_district = as.numeric(total_schools_per_district),
    total_schools_per_county = as.numeric(total_schools_per_county),
    total_districts_per_county = as.numeric(total_districts_per_county),
    total_district_types_per_county = as.numeric(total_district_types_per_county),
    total_districts_per_region = as.numeric(total_districts_per_region),
    total_schools_per_region = as.numeric(total_schools_per_region)
  )
}

X_train <- factor_cols(X_train)
X_pred <- factor_cols(X_pred)
train <- bind_cols(y_train, X_train)


rec <- recipe(PERCENT_PROFICIENT ~ ., data = train) |>
  step_YeoJohnson(all_numeric(), -all_outcomes()) |>
  step_normalize(all_numeric(), -all_outcomes())


model_tune <- tabnet(
  epochs = 1500,
  learn_rate = tune(),
  early_stopping_tolerance = 1e-4,
  penalty = tune(),
  batch_size = tune(),
  num_steps = tune(),
  attention_width = tune(),
  momentum = tune()
) |>
  set_engine("torch", num_workers = parallel::detectCores() - 1) |>
  set_mode("regression")

wf <- workflow() |>
  add_recipe(rec) |>
  add_model(model_tune)

tabnet_grid <- grid_space_filling(
  learn_rate(range = c(-3, -1)),
  num_steps(range = c(3, 10)),
  attention_width(range = c(8, 64)),
  momentum(range = c(0.01, 0.4)),
  batch_size(range = c(8, 12)),
  dials::penalty(range = c(-4, -1)),
  size = 50
)

folds <- vfold_cv(train, v = 5, strata = PERCENT_PROFICIENT)


tune_results <- tune_grid(
  wf,
  resamples = folds,
  grid = tabnet_grid,
  metrics = metric_set(rmse),
  control = control_grid(
    save_pred     = TRUE,
    verbose       = TRUE,
    save_workflow = TRUE
  )
)

saveRDS(tune_results, "tabnet_tune_results.rds")

show_best(tune_results, metric = "rmse", n = 10)
best_params <- select_best(tune_results, metric = "rmse")


pretrain <- tabnet_pretrain(
  rec, train,
  epochs = 1500,
  early_stopping_tolerance = 1e-4,
  valid_split = 0.1,
  learn_rate = best_params$learn_rate * 2
)


final_model <- tabnet(
  epochs = 1500,
  early_stopping_tolerance = 1e-4,
  learn_rate = best_params$learn_rate,
  penalty = best_params$penalty,
  batch_size = best_params$batch_size,
  num_steps = best_params$num_steps,
  attention_width = best_params$attention_width,
  momentum = best_params$momentum,
  tabnet_model = pretrain
) |>
  set_engine("torch", num_workers = parallel::detectCores() - 1) |>
  set_mode("regression")

final_wf <- workflow() |>
  add_recipe(rec) |>
  add_model(final_model)

final_fit <- final_wf |> fit(data = train)

saveRDS(final_fit, "tabnet_final_fit.rds")

preds <- predict(final_fit, new_data = X_train) |> bind_cols(y_train)
rmse(preds, truth = PERCENT_PROFICIENT, estimate = .pred)

y_pred <- predict(final_fit, new_data = X_pred)

bind_cols(X_pred_id, y_pred) |>
  rename(PERCENT_PROFICIENT = .pred) |>
  write.csv("submission_tabnet.csv", row.names = FALSE)
