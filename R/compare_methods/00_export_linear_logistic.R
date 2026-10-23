# Refit the linear probability model and the logistic regression on the
# shared training split, and store their test-set scores.
# This does not redraw the diagnostic figures in R/03–R/05.
# The same two files are also written at the end of R/05_assessment.R.

# Always rebuild the split from set.seed(2026), even if a session already
# holds a training frame. The saved scores then match R/01_prepare_data.R.
source(here::here("R", "01_prepare_data.R"))

source(here::here("R", "compare_methods", "prediction_io.R"))

lpm <- lm(churn_formula, data = telco_train)
logit_mod <- glm(churn_formula, data = telco_train, family = binomial(link = "logit"))

lpm_test_p <- as.numeric(predict(lpm, newdata = telco_test))
logit_test_p <- as.numeric(predict(logit_mod, newdata = telco_test, type = "response"))
y_test <- as.integer(telco_test$Churn)
split_info <- churn_split_info(telco, telco_train)

stopifnot(identical(as.integer(rownames(telco_test)), as.integer(test_id)))

save_test_predictions(
  method_id = "01_linear_probability",
  method_label = "Linear probability",
  y = y_test,
  p = lpm_test_p,
  test_row = test_id,
  split = split_info,
  detail = "OLS linear probability model"
)
save_test_predictions(
  method_id = "02_logistic",
  method_label = "Logistic regression",
  y = y_test,
  p = logit_test_p,
  test_row = test_id,
  split = split_info,
  detail = "Binomial GLM, logit link"
)
