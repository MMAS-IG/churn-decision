# Compare classification methods on one shared test set.
#
# This program does not fit models. At the start it loads every
# output/test_predictions/*.rds file written by save_test_predictions().
# To add a method, score the same test rows and save another file, then
# source this script again. Nothing else in this file needs to change.
#
#   source("R/01_prepare_data.R")
#   source("R/compare_methods/prediction_io.R")
#   p_hat <- predict(my_model, newdata = telco_test, type = "response")
#   save_test_predictions(
#     method_id = "04_my_method",
#     method_label = "My method",
#     y = telco_test$Churn,
#     p = p_hat,
#     test_row = test_id,
#     split = churn_split_info(telco, telco_train),
#     detail = "one line on what was fit"
#   )
#   source("R/compare_methods/01_compare_on_test.R")
#
# `p` must be a score for which larger values mean "more likely to churn".
# Probability metrics (Brier, log loss, calibration) treat `p` as an
# estimate of P(Churn = 1). A method that only ranks can still be compared
# on AUC; if its scores are not probabilities, Brier and log loss will
# show that.

library(here)
library(dplyr)
library(ggplot2)
library(pROC)

source(here("R", "compare_methods", "prediction_io.R"))

brier_score <- function(y, p) {
  mean((as.numeric(p) - as.numeric(y))^2)
}

# Same clip as the network notes: scores outside (0, 1) stay visible in
# n_outside_01, but the average log loss itself has to be finite.
log_loss <- function(y, p, eps = 1e-6) {
  p <- pmin(pmax(as.numeric(p), eps), 1 - eps)
  y <- as.numeric(y)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

theme_set(theme_minimal(base_size = 12))
dir.create(here("figures"), showWarnings = FALSE)
dir.create(here("output", "test_predictions"), showWarnings = FALSE)

methods <- load_test_predictions(here("output", "test_predictions"))

y_test <- methods[[1]]$y
test_row <- methods[[1]]$test_row
train_prevalence <- methods[[1]]$split$train_prevalence
n_test <- length(y_test)

message(
  "Loaded ", length(methods), " method(s) on ", n_test,
  " shared test rows (split seed ", methods[[1]]$split$seed,
  ", training prevalence ", round(train_prevalence, 4), ")."
)

classification_at <- function(y, p, threshold) {
  yhat <- as.integer(p >= threshold)
  tp <- sum(y == 1L & yhat == 1L)
  tn <- sum(y == 0L & yhat == 0L)
  fp <- sum(y == 0L & yhat == 1L)
  fn <- sum(y == 1L & yhat == 0L)
  n <- length(y)
  data.frame(
    accuracy = (tp + tn) / n,
    sensitivity = if ((tp + fn) > 0) tp / (tp + fn) else NA_real_,
    specificity = if ((tn + fp) > 0) tn / (tn + fp) else NA_real_,
    precision = if ((tp + fp) > 0) tp / (tp + fp) else NA_real_,
    tp = tp, fp = fp, fn = fn, tn = tn
  )
}

# One shared bootstrap sample so differences between methods are paired.
set.seed(2028)
B <- 2000L
boot_idx <- replicate(B, sample.int(n_test, n_test, replace = TRUE))

brier_bootstrap <- function(y, p, idx) {
  se <- (as.numeric(p) - as.numeric(y))^2
  colMeans(matrix(se[idx], nrow = length(y)))
}

score_method <- function(obj) {
  y <- obj$y
  p <- obj$p
  roc_obj <- tryCatch(
    roc(response = y, predictor = p, levels = c(0, 1), direction = "<", quiet = TRUE),
    error = function(e) NULL
  )
  if (is.null(roc_obj)) {
    auc_hat <- NA_real_
    auc_lo <- NA_real_
    auc_hi <- NA_real_
  } else {
    auc_hat <- as.numeric(auc(roc_obj))
    auc_ci <- tryCatch(
      as.numeric(ci.auc(roc_obj, method = "delong", quiet = TRUE)),
      error = function(e) c(NA_real_, NA_real_, NA_real_)
    )
    auc_lo <- auc_ci[1]
    auc_hi <- auc_ci[3]
  }
  brier_hat <- brier_score(y, p)
  brier_boot <- brier_bootstrap(y, p, boot_idx)
  brier_ci <- quantile(brier_boot, c(0.025, 0.975), names = FALSE)
  at_half <- classification_at(y, p, 0.5)
  at_prev <- classification_at(y, p, train_prevalence)
  data.frame(
    method_id = obj$method_id,
    method_label = obj$method_label,
    detail = obj$detail,
    n = length(y),
    p_min = min(p),
    p_max = max(p),
    n_outside_01 = sum(p < 0 | p > 1),
    mean_p = mean(p),
    brier = brier_hat,
    brier_ci_low = brier_ci[1],
    brier_ci_high = brier_ci[2],
    log_loss = log_loss(y, p),
    auc = auc_hat,
    auc_ci_low = auc_lo,
    auc_ci_high = auc_hi,
    accuracy_05 = at_half$accuracy,
    sensitivity_05 = at_half$sensitivity,
    specificity_05 = at_half$specificity,
    precision_05 = at_half$precision,
    accuracy_prevalence = at_prev$accuracy,
    sensitivity_prevalence = at_prev$sensitivity,
    specificity_prevalence = at_prev$specificity,
    precision_prevalence = at_prev$precision,
    stringsAsFactors = FALSE
  )
}

comparison_scores <- do.call(rbind, lapply(methods, score_method))
rownames(comparison_scores) <- NULL

brier_ref <- brier_score(y_test, rep(train_prevalence, n_test))
log_loss_ref <- log_loss(y_test, rep(train_prevalence, n_test))
comparison_scores$brier_skill <- 1 - comparison_scores$brier / brier_ref

message(
  "Reference forecast (constant training prevalence) on the test set: ",
  "Brier ", round(brier_ref, 4),
  ", log loss ", round(log_loss_ref, 4),
  ". Always predicting stay: accuracy ", round(mean(y_test == 0), 3), "."
)
message(
  "Log loss clips scores to (1e-6, 1 - 1e-6). ",
  "n_outside_01 counts scores outside [0, 1] before that clip."
)

# Paired Brier differences, method A minus method B. Negative means A
# has the smaller (better) Brier on this test set.
pair_rows <- list()
pair_k <- 0L
if (length(methods) >= 2L) {
  for (i in seq_len(length(methods) - 1L)) {
    for (j in seq.int(i + 1L, length(methods))) {
      pair_k <- pair_k + 1L
      se_diff <- (methods[[i]]$p - y_test)^2 - (methods[[j]]$p - y_test)^2
      boot <- colMeans(matrix(se_diff[boot_idx], nrow = n_test))
      ci <- quantile(boot, c(0.025, 0.975), names = FALSE)
      pair_rows[[pair_k]] <- data.frame(
        method_a = methods[[i]]$method_label,
        method_b = methods[[j]]$method_label,
        brier_a_minus_b = mean(se_diff),
        ci_low = ci[1],
        ci_high = ci[2],
        stringsAsFactors = FALSE
      )
    }
  }
}
comparison_brier_diff <- if (pair_k == 0L) {
  data.frame(
    method_a = character(),
    method_b = character(),
    brier_a_minus_b = numeric(),
    ci_low = numeric(),
    ci_high = numeric()
  )
} else {
  do.call(rbind, pair_rows)
}

comparison_classes <- comparison_scores %>%
  select(
    method_label, detail,
    accuracy_05, sensitivity_05, specificity_05, precision_05,
    accuracy_prevalence, sensitivity_prevalence,
    specificity_prevalence, precision_prevalence
  )

pred_dir <- here("output", "test_predictions")
write.csv(comparison_scores, file.path(pred_dir, "comparison_scores.csv"), row.names = FALSE)
write.csv(comparison_classes, file.path(pred_dir, "comparison_classes.csv"), row.names = FALSE)
write.csv(comparison_brier_diff, file.path(pred_dir, "comparison_brier_diff.csv"), row.names = FALSE)

message("Probability scores (test set):")
print(comparison_scores %>% select(
  method_label, n_outside_01, brier, brier_ci_low, brier_ci_high,
  brier_skill, log_loss, auc, auc_ci_low, auc_ci_high
), digits = 4, row.names = FALSE)
message("Decisions at 0.5 and at the training prevalence:")
print(comparison_classes, digits = 4, row.names = FALSE)
if (nrow(comparison_brier_diff) > 0) {
  message("Paired Brier difference (A minus B) with a 95% bootstrap interval:")
  print(comparison_brier_diff, digits = 4, row.names = FALSE)
}

# --- figures ----------------------------------------------------------

method_levels <- comparison_scores$method_label
known_colours <- c(
  "Linear probability" = "#7b3294",
  "Logistic regression" = "#018571",
  "Neural network" = "#d95f02"
)
extra_colours <- c("#e7298a", "#66a61e", "#7570b3", "#e6ab02", "#1b9e77", "#a6761d", "#666666")
missing_labels <- setdiff(method_levels, names(known_colours))
if (length(missing_labels) > length(extra_colours)) {
  extra_colours <- rep_len(extra_colours, length(missing_labels))
}
compare_colours <- c(known_colours, stats::setNames(extra_colours[seq_along(missing_labels)], missing_labels))
compare_colours <- compare_colours[method_levels]

roc_parts <- lapply(methods, function(obj) {
  roc_obj <- roc(
    response = obj$y, predictor = obj$p,
    levels = c(0, 1), direction = "<", quiet = TRUE
  )
  data.frame(
    method = obj$method_label,
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities
  )
})
comparison_roc <- bind_rows(roc_parts)
comparison_roc$method <- factor(comparison_roc$method, levels = method_levels)

compare_roc <- ggplot(comparison_roc, aes(x = fpr, y = tpr, colour = method)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_line(linewidth = 1) +
  coord_equal() +
  scale_colour_manual(values = compare_colours) +
  labs(
    x = "False positive rate (1 - specificity)",
    y = "True positive rate (sensitivity)",
    colour = NULL,
    title = "ROC curves on the shared test set"
  )
print(compare_roc)
ggsave(here("figures", "compare_roc_test.png"), compare_roc, width = 6.8, height = 5.6)

calibration_df <- function(y, p, model, n_bins = 10) {
  br <- unique(quantile(p, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
  if (length(br) < 3) {
    br <- seq(min(p), max(p), length.out = 5)
  }
  cuts <- cut(p, breaks = br, include.lowest = TRUE)
  data.frame(y = y, p = p, bin = cuts, model = model) %>%
    group_by(model, bin) %>%
    summarise(mean_pred = mean(p), mean_obs = mean(y), n = n(), .groups = "drop")
}
comparison_calibration <- bind_rows(lapply(methods, function(obj) {
  calibration_df(obj$y, obj$p, obj$method_label)
}))
comparison_calibration$model <- factor(comparison_calibration$model, levels = method_levels)

compare_cal <- ggplot(comparison_calibration, aes(x = mean_pred, y = mean_obs, colour = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_line() +
  geom_point(size = 2.3) +
  scale_colour_manual(values = compare_colours) +
  labs(
    x = "Mean predicted value",
    y = "Observed churn rate",
    colour = NULL,
    title = "Calibration on the shared test set"
  )
print(compare_cal)
ggsave(here("figures", "compare_calibration_test.png"), compare_cal, width = 7.4, height = 5.4)

brier_plot_df <- comparison_scores
brier_plot_df$method_label <- factor(brier_plot_df$method_label, levels = rev(method_levels))
compare_brier <- ggplot(brier_plot_df, aes(x = brier, y = method_label, colour = method_label)) +
  geom_vline(xintercept = brier_ref, linetype = "dashed", colour = "grey40") +
  geom_errorbar(aes(xmin = brier_ci_low, xmax = brier_ci_high), orientation = "y", width = 0.15, linewidth = 0.7) +
  geom_point(size = 2.8) +
  scale_colour_manual(values = compare_colours, guide = "none") +
  labs(
    x = "Brier score on the test set (95% bootstrap interval)",
    y = NULL,
    title = "Lower Brier is better",
    subtitle = "Dashed line: constant forecast at the training prevalence"
  )
print(compare_brier)
ggsave(here("figures", "compare_brier_test.png"), compare_brier, width = 7.4, height = 4.2)

comparison_summary <- comparison_scores
