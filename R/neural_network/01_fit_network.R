# Fit a single-hidden-layer network for churn.
# Size and weight decay are chosen on a validation slice of the training
# data. The test set from R/01_prepare_data.R is not used until the end.

if (!exists("telco_train") || !exists("telco_test") || !exists("test_id")) {
  source(here::here("R", "01_prepare_data.R"))
}

library(nnet)
source(here::here("R", "neural_network", "nn_lib.R"))
source(here::here("R", "compare_methods", "prediction_io.R"))

stopifnot(identical(as.integer(rownames(telco_test)), as.integer(test_id)))

# Copies: telco_train$Churn stays 0/1 for the linear and logistic fits.
nn_train <- as_churn_factor(telco_train)
nn_test <- as_churn_factor(telco_test)

# 20% of the training rows. Seed 2027 is not the test-split seed.
set.seed(2027)
val_n <- floor(0.20 * nrow(nn_train))
nn_val_id <- sample(seq_len(nrow(nn_train)), size = val_n)
nn_fit_slice <- nn_train[-nn_val_id, , drop = FALSE]
nn_val_slice <- nn_train[nn_val_id, , drop = FALSE]

nn_sizes <- c(0, 2, 5, 10)
nn_decays <- c(0.05, 0.5, 2)
nn_maxit <- 400
nn_tune_starts <- 3
nn_final_starts <- 5

message(
  "Network tuning: ", nrow(nn_fit_slice), " fitting rows, ",
  nrow(nn_val_slice), " validation rows. Test set is held out (",
  nrow(nn_test), " rows)."
)

nn_tune <- system.time(
  nn_sel <- select_network(
    train_raw = nn_fit_slice,
    val_raw = nn_val_slice,
    sizes = nn_sizes,
    decays = nn_decays,
    n_starts = nn_tune_starts,
    maxit = nn_maxit
  )
)
nn_selection <- nn_sel$table
# Size 0 is a skip-layer logistic regression: the benchmark the hidden
# layer has to beat. The network we keep has at least one hidden unit.
# Among those, the rule is the same: lowest validation Brier, then fewer
# units, then larger decay.
nn_benchmark <- nn_selection[1, ]
nn_hidden <- nn_selection[nn_selection$size > 0, ]
if (nrow(nn_hidden) == 0L) stop("The tuning grid has no hidden-layer network.")
nn_chosen <- nn_hidden[1, ]
message("Tuning elapsed (seconds): ", round(nn_tune[["elapsed"]], 1))
print(nn_selection, digits = 4, row.names = FALSE)
message(
  "Best row on the grid (may have no hidden layer): size = ", nn_benchmark$size,
  ", decay = ", nn_benchmark$decay,
  ", validation Brier = ", round(nn_benchmark$val_brier, 4),
  " (SE ", round(nn_benchmark$val_brier_se, 4), ")."
)
message(
  "Network taken forward: size = ", nn_chosen$size,
  ", decay = ", nn_chosen$decay,
  ", validation Brier = ", round(nn_chosen$val_brier, 4),
  " (SE ", round(nn_chosen$val_brier_se, 4), ")."
)

nn_refit <- refit_network(
  train_raw = nn_train,
  size = nn_chosen$size,
  decay = nn_chosen$decay,
  n_starts = nn_final_starts,
  maxit = nn_maxit
)
nn_fit <- nn_refit$fit
nn_spec <- nn_refit$spec
nn_train_scaled <- nn_refit$train_scaled
nn_test_scaled <- apply_center_scale(nn_test, nn_spec)

message(
  "Final starts (training criterion, lower is better): ",
  paste(round(nn_refit$criteria, 1), collapse = ", "),
  " | kept start ", nn_refit$start_kept,
  " | converged = ", nn_fit$convergence == 0
)
message(
  "Inputs ", nn_fit$n[1],
  ", hidden ", nn_fit$n[2],
  ", output ", nn_fit$n[3],
  ", weights ", length(nn_fit$wts)
)

nn_train_p <- nnet_probability(nn_fit, nn_train_scaled)
nn_test_p <- nnet_probability(nn_fit, nn_test_scaled)
nn_y_train <- as.integer(telco_train$Churn)
nn_y_test <- as.integer(telco_test$Churn)

save_test_predictions(
  method_id = "03_neural_network",
  method_label = "Neural network",
  y = nn_y_test,
  p = nn_test_p,
  test_row = test_id,
  split = churn_split_info(telco, telco_train),
  detail = sprintf(
    "nnet, %d hidden unit%s, decay %s",
    nn_chosen$size,
    if (nn_chosen$size == 1) "" else "s",
    format(nn_chosen$decay, trim = TRUE, scientific = FALSE)
  )
)
