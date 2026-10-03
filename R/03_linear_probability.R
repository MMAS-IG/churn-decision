# Linear probability model: OLS of a 0/1 churn indicator on all predictors.

if (!exists("telco_train")) {
  source(here::here("R", "01_prepare_data.R"))
}

library(ggplot2)
library(dplyr)
library(broom)
library(car)
library(lmtest)

theme_set(theme_minimal(base_size = 12))
dir.create(here::here("figures"), showWarnings = FALSE)

lpm <- lm(churn_formula, data = telco_train)
print(summary(lpm))

aliased <- alias(lpm)
if (!is.null(aliased$Complete)) {
  message("Aliased (dropped) coefficients:")
  print(aliased$Complete)
} else {
  message("No perfectly collinear coefficients after recoding nested service levels.")
}

print(vif(lpm))

lpm_train_fit <- fitted(lpm)
n_out <- sum(lpm_train_fit < 0 | lpm_train_fit > 1)
message(
  "LPM fitted values outside [0, 1]: ", n_out, " / ", length(lpm_train_fit),
  " (", round(100 * n_out / length(lpm_train_fit), 1), "%)."
)
print(range(lpm_train_fit))

print(bptest(lpm))

p_hist_lpm <- ggplot(data.frame(fit = lpm_train_fit), aes(x = fit)) +
  geom_histogram(bins = 40, fill = "#7b3294", colour = "white") +
  geom_vline(xintercept = c(0, 1), linetype = "dashed", colour = "grey20") +
  labs(
    x = "Fitted value (OLS)",
    y = "Customers (training set)",
    title = "Linear probability model: fitted values are not probabilities"
  )
print(p_hist_lpm)
ggsave(here::here("figures", "lpm_fitted_histogram.png"), p_hist_lpm, width = 7, height = 4.5)

lpm_diag <- data.frame(
  fitted = lpm_train_fit,
  resid = resid(lpm),
  churn = telco_train$Churn
)
p_resid_lpm <- ggplot(lpm_diag, aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.15, colour = "#7b3294") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    x = "Fitted value",
    y = "Residual",
    title = "LPM residuals: two bands (Y = 0 and Y = 1), not Gaussian noise"
  )
print(p_resid_lpm)
ggsave(here::here("figures", "lpm_residuals.png"), p_resid_lpm, width = 7, height = 4.5)

# Univariate illustration: tenure only, so the fitted line is easy to plot.
lpm_tenure <- lm(Churn ~ tenure, data = telco_train)
tenure_grid <- data.frame(tenure = seq(0, 72, by = 1))
tenure_grid$lpm <- predict(lpm_tenure, newdata = tenure_grid)

p_tenure_lpm <- ggplot(telco_train, aes(x = tenure, y = Churn)) +
  geom_jitter(height = 0.04, alpha = 0.08, colour = "grey30") +
  geom_line(data = tenure_grid, aes(x = tenure, y = lpm), colour = "#7b3294", linewidth = 1) +
  geom_hline(yintercept = c(0, 1), linetype = "dotted") +
  labs(
    x = "Tenure (months)",
    y = "Churn (0/1)",
    title = "OLS on a binary outcome can leave the [0, 1] interval"
  )
print(p_tenure_lpm)
ggsave(here::here("figures", "lpm_tenure_line.png"), p_tenure_lpm, width = 7, height = 4.5)

lpm_tidy <- tidy(lpm, conf.int = TRUE)
print(lpm_tidy)
