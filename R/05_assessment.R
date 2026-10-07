# Hold-out assessment of the LPM and the logistic model.

if (!exists("logit_mod") || !exists("lpm")) {
  source(here::here("R", "04_logistic_regression.R"))
}

library(ggplot2)
library(dplyr)
library(pROC)

theme_set(theme_minimal(base_size = 12))
dir.create(here::here("figures"), showWarnings = FALSE)

lpm_test_p <- predict(lpm, newdata = telco_test)
logit_test_p <- predict(logit_mod, newdata = telco_test, type = "response")
y_test <- telco_test$Churn

message(
  "Test LPM predictions outside [0, 1]: ",
  sum(lpm_test_p < 0 | lpm_test_p > 1), " / ", length(lpm_test_p)
)
message("Test logistic predictions outside [0, 1]: ", sum(logit_test_p < 0 | logit_test_p > 1))

brier <- function(y, p) mean((p - y)^2)
message("Brier (test) LPM: ", round(brier(y_test, lpm_test_p), 4))
message("Brier (test) logistic: ", round(brier(y_test, logit_test_p), 4))

confusion_at <- function(y, p, threshold = 0.5) {
  yhat <- as.integer(p >= threshold)
  tp <- sum(y == 1 & yhat == 1)
  tn <- sum(y == 0 & yhat == 0)
  fp <- sum(y == 0 & yhat == 1)
  fn <- sum(y == 1 & yhat == 0)
  n <- length(y)
  list(
    threshold = threshold,
    table = matrix(
      c(tn, fp, fn, tp),
      nrow = 2,
      byrow = TRUE,
      dimnames = list(observed = c("0", "1"), predicted = c("0", "1"))
    ),
    accuracy = (tp + tn) / n,
    sensitivity = ifelse((tp + fn) > 0, tp / (tp + fn), NA_real_),
    specificity = ifelse((tn + fp) > 0, tn / (tn + fp), NA_real_),
    precision = ifelse((tp + fp) > 0, tp / (tp + fp), NA_real_)
  )
}

thr_half <- 0.5
thr_rate <- mean(telco_train$Churn)

metrics <- list(
  lpm_0.5 = confusion_at(y_test, lpm_test_p, thr_half),
  logit_0.5 = confusion_at(y_test, logit_test_p, thr_half),
  lpm_base = confusion_at(y_test, lpm_test_p, thr_rate),
  logit_base = confusion_at(y_test, logit_test_p, thr_rate)
)
print(metrics)

naive_acc <- mean(y_test == 0)
message("Naive accuracy (always predict 0): ", round(naive_acc, 3))

roc_lpm <- roc(y_test, lpm_test_p, quiet = TRUE)
roc_logit <- roc(y_test, logit_test_p, quiet = TRUE)
message("AUC LPM: ", round(as.numeric(auc(roc_lpm)), 3))
message("AUC logistic: ", round(as.numeric(auc(roc_logit)), 3))

roc_df <- rbind(
  data.frame(
    model = "Linear probability",
    fpr = 1 - roc_lpm$specificities,
    tpr = roc_lpm$sensitivities
  ),
  data.frame(
    model = "Logistic",
    fpr = 1 - roc_logit$specificities,
    tpr = roc_logit$sensitivities
  )
)
p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, colour = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_line(linewidth = 1) +
  coord_equal() +
  scale_colour_manual(values = c("Linear probability" = "#7b3294", "Logistic" = "#018571")) +
  labs(
    x = "False positive rate (1 - specificity)",
    y = "True positive rate (sensitivity)",
    colour = NULL,
    title = "ROC curves on the test set"
  )
print(p_roc)
ggsave(here::here("figures", "roc_test.png"), p_roc, width = 6.5, height = 5.5)

calibration_df <- function(y, p, model, n_bins = 10) {
  br <- unique(quantile(p, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
  if (length(br) < 3) {
    br <- seq(min(p, na.rm = TRUE), max(p, na.rm = TRUE), length.out = 5)
  }
  cuts <- cut(p, breaks = br, include.lowest = TRUE)
  data.frame(y = y, p = p, bin = cuts, model = model) %>%
    group_by(model, bin) %>%
    summarise(
      mean_pred = mean(p),
      mean_obs = mean(y),
      n = n(),
      .groups = "drop"
    )
}

cal <- rbind(
  calibration_df(y_test, lpm_test_p, "Linear probability"),
  calibration_df(y_test, logit_test_p, "Logistic")
)
p_cal <- ggplot(cal, aes(x = mean_pred, y = mean_obs, colour = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(size = 2.4, alpha = 0.9) +
  geom_line() +
  scale_colour_manual(values = c(
    "Linear probability" = "#7b3294",
    "Logistic" = "#018571"
  )) +
  labs(
    x = "Mean predicted value",
    y = "Observed churn rate",
    colour = NULL,
    title = "Calibration on the test set (equal-count bins)"
  )
print(p_cal)
ggsave(here::here("figures", "calibration_test.png"), p_cal, width = 7.5, height = 5.5)

# Binned residual plot for the logistic model (training).
n_bin <- 20
logit_res <- data.frame(
  p = fitted(logit_mod),
  r = telco_train$Churn - fitted(logit_mod)
)
logit_res$bin <- cut(
  logit_res$p,
  breaks = unique(quantile(logit_res$p, probs = seq(0, 1, length.out = n_bin + 1))),
  include.lowest = TRUE
)
binned <- logit_res %>%
  group_by(bin) %>%
  summarise(p = mean(p), r = mean(r), n = n(), .groups = "drop")
se <- sqrt(binned$p * (1 - binned$p) / binned$n)
p_binres <- ggplot(binned, aes(x = p, y = r)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_ribbon(aes(ymin = -2 * se, ymax = 2 * se), alpha = 0.15, fill = "#018571") +
  geom_point(colour = "#018571") +
  labs(
    x = "Fitted P(churn)",
    y = "Mean residual (observed - fitted)",
    title = "Binned residuals for the logistic model (training set)"
  )
print(p_binres)
ggsave(here::here("figures", "logit_binned_residuals.png"), p_binres, width = 7, height = 4.5)

# Influence: largest Cook's distances (training).
cooks <- cooks.distance(logit_mod)
top_infl <- order(cooks, decreasing = TRUE)[1:8]
infl_tbl <- cbind(
  telco_train[top_infl, c("tenure", "Contract", "MonthlyCharges", "Churn")],
  cook = cooks[top_infl],
  p_hat = fitted(logit_mod)[top_infl]
)
print(infl_tbl)

p_cook <- ggplot(data.frame(i = seq_along(cooks), cook = cooks), aes(x = i, y = cook)) +
  geom_point(alpha = 0.25, colour = "#018571") +
  geom_hline(yintercept = 4 / length(cooks), linetype = "dashed", colour = "grey30") +
  labs(
    x = "Observation index (training)",
    y = "Cook's distance",
    title = "Influence on the logistic fit"
  )
print(p_cook)
ggsave(here::here("figures", "logit_cooks.png"), p_cook, width = 7, height = 4.5)
