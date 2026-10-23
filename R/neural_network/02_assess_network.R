# Test-set and training-set checks for the fitted network.
# Run after R/neural_network/01_fit_network.R.

if (!exists("nn_fit") || !exists("nn_test_p")) {
  source(here::here("R", "neural_network", "01_fit_network.R"))
}

library(ggplot2)
library(dplyr)
library(pROC)

theme_set(theme_minimal(base_size = 12))
dir.create(here::here("figures"), showWarnings = FALSE)

confusion_at <- function(y, p, threshold = 0.5) {
  yhat <- as.integer(p >= threshold)
  tp <- sum(y == 1 & yhat == 1)
  tn <- sum(y == 0 & yhat == 0)
  fp <- sum(y == 0 & yhat == 1)
  fn <- sum(y == 1 & yhat == 0)
  n <- length(y)
  data.frame(
    threshold = threshold,
    accuracy = (tp + tn) / n,
    sensitivity = ifelse((tp + fn) > 0, tp / (tp + fn), NA_real_),
    specificity = ifelse((tn + fp) > 0, tn / (tn + fp), NA_real_),
    precision = ifelse((tp + fp) > 0, tp / (tp + fp), NA_real_),
    tp = tp,
    fp = fp,
    fn = fn,
    tn = tn
  )
}

roc_safe <- function(y, p) {
  roc(response = y, predictor = p, levels = c(0, 1), direction = "<", quiet = TRUE)
}

nn_holdout <- data.frame(
  metric = c("Brier", "Log loss", "AUC"),
  train = c(
    brier_score(nn_y_train, nn_train_p),
    log_loss(nn_y_train, nn_train_p),
    as.numeric(auc(roc_safe(nn_y_train, nn_train_p)))
  ),
  test = c(
    brier_score(nn_y_test, nn_test_p),
    log_loss(nn_y_test, nn_test_p),
    as.numeric(auc(roc_safe(nn_y_test, nn_test_p)))
  )
)
print(nn_holdout, digits = 4, row.names = FALSE)

thr_rate <- mean(telco_train$Churn)
nn_confusion <- rbind(
  cbind(cut = "0.5", confusion_at(nn_y_test, nn_test_p, 0.5)),
  cbind(cut = "training prevalence", confusion_at(nn_y_test, nn_test_p, thr_rate))
)
print(nn_confusion, digits = 4, row.names = FALSE)
message(
  "Test predictions outside [0, 1]: ",
  sum(nn_test_p < 0 | nn_test_p > 1),
  ". Range: ",
  paste(round(range(nn_test_p), 4), collapse = " to "),
  "."
)
message(
  "Always predicting 'stay' on the test set has accuracy ",
  round(mean(nn_y_test == 0), 3), "."
)

# --- figures ----------------------------------------------------------

nn_selection$chosen <- nn_selection$size == nn_chosen$size &
  nn_selection$decay == nn_chosen$decay
nn_selection$size_f <- factor(nn_selection$size)
nn_selection$decay_f <- factor(nn_selection$decay)
p_nn_select <- ggplot(
  nn_selection,
  aes(x = size_f, y = val_brier, group = decay_f, colour = decay_f)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.6) +
  geom_point(
    data = nn_selection[nn_selection$chosen, ],
    aes(x = size_f, y = val_brier),
    inherit.aes = FALSE,
    shape = 1,
    size = 6,
    colour = "grey10"
  ) +
  scale_colour_manual(
    values = c("0.05" = "#fdb863", "0.5" = "#e66101", "2" = "#5e3c99"),
    name = "Weight decay"
  ) +
  labs(
    x = "Hidden units (0 = skip-layer logistic)",
    y = "Validation Brier score",
    title = "Validation Brier by hidden-layer size and weight decay"
  )
print(p_nn_select)
ggsave(here::here("figures", "nn_selection_brier.png"), p_nn_select, width = 7.5, height = 4.6)

p_nn_hist <- ggplot(data.frame(fit = nn_test_p), aes(x = fit)) +
  geom_histogram(bins = 40, fill = "#d95f02", colour = "white") +
  geom_vline(xintercept = c(0, 1), linetype = "dashed", colour = "grey20") +
  coord_cartesian(xlim = c(0, 1)) +
  labs(
    x = "Predicted P(churn = 1)",
    y = "Customers (test set)",
    title = "Neural network scores stay inside (0, 1)"
  )
print(p_nn_hist)
ggsave(here::here("figures", "nn_test_histogram.png"), p_nn_hist, width = 7, height = 4.5)

roc_nn <- roc_safe(nn_y_test, nn_test_p)
nn_roc_df <- data.frame(
  fpr = 1 - roc_nn$specificities,
  tpr = roc_nn$sensitivities
)
p_nn_roc <- ggplot(nn_roc_df, aes(x = fpr, y = tpr)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_line(linewidth = 1, colour = "#d95f02") +
  coord_equal() +
  labs(
    x = "False positive rate (1 - specificity)",
    y = "True positive rate (sensitivity)",
    title = sprintf("Test ROC for the network (AUC = %.3f)", as.numeric(auc(roc_nn)))
  )
print(p_nn_roc)
ggsave(here::here("figures", "nn_roc_test.png"), p_nn_roc, width = 6.2, height = 5.4)

calibration_df <- function(y, p, n_bins = 10) {
  br <- unique(quantile(p, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
  if (length(br) < 3) {
    br <- seq(min(p), max(p), length.out = 5)
  }
  cuts <- cut(p, breaks = br, include.lowest = TRUE)
  data.frame(y = y, p = p, bin = cuts) %>%
    group_by(bin) %>%
    summarise(mean_pred = mean(p), mean_obs = mean(y), n = n(), .groups = "drop")
}
nn_cal <- calibration_df(nn_y_test, nn_test_p)
p_nn_cal <- ggplot(nn_cal, aes(x = mean_pred, y = mean_obs)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_line(colour = "#d95f02") +
  geom_point(size = 2.4, colour = "#d95f02") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    x = "Mean predicted P(churn)",
    y = "Observed churn rate",
    title = "Calibration of the network on the test set"
  )
print(p_nn_cal)
ggsave(here::here("figures", "nn_calibration_test.png"), p_nn_cal, width = 6.5, height = 5.2)

# Profile: same construction as the logistic notes. Other covariates sit
# at the training median (numeric) or mode (factor). TotalCharges tracks
# tenure * MonthlyCharges before scaling, so the network does not see a
# long tenure paired with a new customer's bill.
mode_factor <- function(x) {
  tab <- sort(table(x), decreasing = TRUE)
  factor(names(tab)[1], levels = levels(x))
}
base_row <- telco_train[1, , drop = FALSE]
num_vars <- setdiff(names(telco_train)[vapply(telco_train, is.numeric, logical(1))], "Churn")
fac_vars <- names(telco_train)[vapply(telco_train, is.factor, logical(1))]
for (v in num_vars) base_row[[v]] <- median(telco_train[[v]])
for (v in fac_vars) base_row[[v]] <- mode_factor(telco_train[[v]])

make_profile <- function(contract, tenure_seq = 0:72) {
  out <- base_row[rep(1, length(tenure_seq)), ]
  out$Contract <- factor(contract, levels = levels(telco_train$Contract))
  out$tenure <- tenure_seq
  out$TotalCharges <- out$tenure * out$MonthlyCharges
  out
}
nn_prof <- rbind(make_profile("Month-to-month"), make_profile("Two year"))
nn_prof$Contract <- factor(nn_prof$Contract, levels = c("Month-to-month", "Two year"))
nn_prof_scaled <- apply_center_scale(as_churn_factor(nn_prof), nn_spec)
nn_prof$neural_network <- nnet_probability(nn_fit, nn_prof_scaled)

if (!exists("lpm")) {
  lpm <- lm(churn_formula, data = telco_train)
}
if (!exists("logit_mod")) {
  logit_mod <- glm(churn_formula, data = telco_train, family = binomial(link = "logit"))
}
nn_prof$linear_probability <- as.numeric(predict(lpm, newdata = nn_prof))
nn_prof$logistic <- as.numeric(predict(logit_mod, newdata = nn_prof, type = "response"))

prof_long <- tidyr::pivot_longer(
  nn_prof,
  cols = c("linear_probability", "logistic", "neural_network"),
  names_to = "model",
  values_to = "pred"
)
prof_long$model <- factor(
  prof_long$model,
  levels = c("linear_probability", "logistic", "neural_network"),
  labels = c("Linear probability", "Logistic regression", "Neural network")
)
p_nn_profile <- ggplot(prof_long, aes(x = tenure, y = pred, colour = model, linetype = Contract)) +
  geom_hline(yintercept = c(0, 1), linetype = "dotted") +
  geom_line(linewidth = 0.9) +
  scale_colour_manual(values = c(
    "Linear probability" = "#7b3294",
    "Logistic regression" = "#018571",
    "Neural network" = "#d95f02"
  )) +
  labs(
    x = "Tenure (months)",
    y = "Predicted churn probability",
    colour = NULL,
    linetype = "Contract",
    title = "Same typical profile, three mean functions"
  )
print(p_nn_profile)
ggsave(here::here("figures", "nn_profile_contract_tenure.png"), p_nn_profile, width = 8, height = 4.8)
