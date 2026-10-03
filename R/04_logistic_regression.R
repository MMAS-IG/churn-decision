# Logistic regression for churn: binomial GLM with logit link.

if (!exists("telco_train") || !exists("lpm")) {
  source(here::here("R", "03_linear_probability.R"))
}

library(ggplot2)
library(dplyr)
library(tidyr)
library(broom)

theme_set(theme_minimal(base_size = 12))
dir.create(here::here("figures"), showWarnings = FALSE)

logit_mod <- glm(churn_formula, data = telco_train, family = binomial(link = "logit"))
print(summary(logit_mod))

logit_null <- glm(Churn ~ 1, data = telco_train, family = binomial)
print(anova(logit_null, logit_mod, test = "Chisq"))

logit_tidy <- tidy(logit_mod, conf.int = TRUE, exponentiate = FALSE)
logit_or <- tidy(logit_mod, conf.int = TRUE, exponentiate = TRUE)
print(logit_or)

ll_mod <- as.numeric(logLik(logit_mod))
ll_null <- as.numeric(logLik(logit_null))
mcfadden <- 1 - (ll_mod / ll_null)
message("McFadden pseudo-R^2: ", round(mcfadden, 3))
message(
  "AIC (logistic): ", round(AIC(logit_mod), 1),
  " | AIC (LPM, not comparable): ", round(AIC(lpm), 1)
)

logit_train_p <- fitted(logit_mod)
message(
  "Logistic fitted probabilities outside [0, 1]: ",
  sum(logit_train_p < 0 | logit_train_p > 1)
)
print(range(logit_train_p))

# Average partial effects: E[p(1-p)] * beta, comparable to LPM slopes.
ape_scale <- mean(logit_train_p * (1 - logit_train_p))
common_terms <- intersect(names(coef(lpm)), names(coef(logit_mod)))
ape_cmp <- data.frame(
  term = common_terms,
  lpm = as.numeric(coef(lpm)[common_terms]),
  logit_ape = as.numeric(coef(logit_mod)[common_terms]) * ape_scale
)
print(round(ape_scale, 4))
print(ape_cmp[ape_cmp$term %in% c(
  "tenure", "ContractOne year", "ContractTwo year",
  "PaperlessBillingYes", "PaymentMethodElectronic check",
  "OnlineSecurityYes", "TechSupportYes"
), ])

p_hist_logit <- ggplot(data.frame(fit = logit_train_p), aes(x = fit)) +
  geom_histogram(bins = 40, fill = "#018571", colour = "white") +
  geom_vline(xintercept = c(0, 1), linetype = "dashed", colour = "grey20") +
  coord_cartesian(xlim = c(0, 1)) +
  labs(
    x = "Fitted P(churn = 1)",
    y = "Customers (training set)",
    title = "Logistic fitted values stay inside (0, 1)"
  )
print(p_hist_logit)
ggsave(here::here("figures", "logit_fitted_histogram.png"), p_hist_logit, width = 7, height = 4.5)

# Univariate logistic vs LPM on tenure.
logit_tenure <- glm(Churn ~ tenure, data = telco_train, family = binomial)
tenure_grid <- data.frame(tenure = seq(0, 72, by = 1))
tenure_grid$lpm <- predict(lm(Churn ~ tenure, data = telco_train), newdata = tenure_grid)
tenure_grid$logit <- predict(logit_tenure, newdata = tenure_grid, type = "response")
tenure_long <- pivot_longer(
  tenure_grid,
  cols = c("lpm", "logit"),
  names_to = "model",
  values_to = "pred"
)
tenure_long$model <- ifelse(
  tenure_long$model == "lpm",
  "Linear probability",
  "Logistic"
)

p_tenure_both <- ggplot() +
  geom_jitter(
    data = telco_train,
    aes(x = tenure, y = Churn),
    height = 0.04,
    alpha = 0.06,
    colour = "grey30"
  ) +
  geom_line(
    data = tenure_long,
    aes(x = tenure, y = pred, colour = model),
    linewidth = 1
  ) +
  geom_hline(yintercept = c(0, 1), linetype = "dotted") +
  scale_colour_manual(values = c("Linear probability" = "#7b3294", "Logistic" = "#018571")) +
  labs(
    x = "Tenure (months)",
    y = "P(churn) / churn indicator",
    colour = NULL,
    title = "Same predictor, two mean functions"
  )
print(p_tenure_both)
ggsave(here::here("figures", "tenure_lpm_vs_logit.png"), p_tenure_both, width = 7.5, height = 4.5)

# Profile: vary tenure for two contract types; other covariates at training modes / medians.
mode_factor <- function(x) {
  tab <- sort(table(x), decreasing = TRUE)
  factor(names(tab)[1], levels = levels(x))
}

base_row <- telco_train[1, , drop = FALSE]
num_vars <- names(telco_train)[vapply(telco_train, is.numeric, logical(1))]
num_vars <- setdiff(num_vars, "Churn")
fac_vars <- names(telco_train)[vapply(telco_train, is.factor, logical(1))]
for (v in num_vars) base_row[[v]] <- median(telco_train[[v]])
for (v in fac_vars) base_row[[v]] <- mode_factor(telco_train[[v]])

make_profile <- function(contract, tenure_seq = 0:72) {
  out <- base_row[rep(1, length(tenure_seq)), ]
  out$Contract <- factor(contract, levels = levels(telco_train$Contract))
  out$tenure <- tenure_seq
  # Keep the billing identity roughly consistent when tenure moves.
  out$TotalCharges <- out$tenure * out$MonthlyCharges
  out
}

prof <- rbind(
  make_profile("Month-to-month"),
  make_profile("Two year")
)
prof$lpm <- predict(lpm, newdata = prof)
prof$logit <- predict(logit_mod, newdata = prof, type = "response")
prof$Contract <- factor(prof$Contract, levels = c("Month-to-month", "Two year"))

prof_long <- pivot_longer(
  prof,
  cols = c("lpm", "logit"),
  names_to = "model",
  values_to = "pred"
)
prof_long$model <- ifelse(
  prof_long$model == "lpm",
  "Linear probability",
  "Logistic"
)

p_profile <- ggplot(prof_long, aes(x = tenure, y = pred, colour = Contract, linetype = model)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = c(0, 1), linetype = "dotted") +
  scale_colour_manual(values = c("Month-to-month" = "#d7191c", "Two year" = "#2c7bb6")) +
  labs(
    x = "Tenure (months)",
    y = "Predicted churn probability",
    linetype = "Model",
    title = "Full-model profiles: LPM can go negative; logistic stays in (0, 1)"
  )
print(p_profile)
ggsave(here::here("figures", "profile_contract_tenure.png"), p_profile, width = 8, height = 4.8)

# Two concrete accounts (same services, different contract and tenure).
two_accounts <- rbind(
  make_profile("Month-to-month", tenure_seq = 1),
  make_profile("Two year", tenure_seq = 60)
)
two_accounts$account <- c("New, month-to-month", "Long tenure, two-year")
two_accounts$lpm <- predict(lpm, newdata = two_accounts)
two_accounts$logit <- predict(logit_mod, newdata = two_accounts, type = "response")
two_accounts$logit_link <- predict(logit_mod, newdata = two_accounts, type = "link")
print(two_accounts[, c("account", "tenure", "Contract", "lpm", "logit", "logit_link")])

# Odds-ratio forest plot for a readable subset of coefficients.
or_focus <- c(
  "tenure",
  "SeniorCitizenYes",
  "ContractOne year",
  "ContractTwo year",
  "InternetServiceFiber optic",
  "InternetServiceNo",
  "PaperlessBillingYes",
  "PaymentMethodElectronic check",
  "OnlineSecurityYes",
  "TechSupportYes"
)
or_plot_df <- logit_or %>%
  filter(term %in% or_focus) %>%
  mutate(term = factor(term, levels = rev(or_focus)))

p_or <- ggplot(or_plot_df, aes(x = estimate, y = term)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40") +
  geom_errorbar(aes(xmin = conf.low, xmax = conf.high), orientation = "y", width = 0.2, colour = "#018571") +
  geom_point(size = 2.4, colour = "#018571") +
  scale_x_log10() +
  labs(
    x = "Odds ratio (log scale) with 95% CI",
    y = NULL,
    title = "Selected logistic odds ratios (training set)"
  )
print(p_or)
ggsave(here::here("figures", "odds_ratios.png"), p_or, width = 8, height = 5)
