# Reproduce the full analysis from the project root
# (or after opening churn-decision.Rproj).

library(here)
source(here("R", "01_prepare_data.R"))
source(here("R", "02_explore.R"))
source(here("R", "03_linear_probability.R"))
source(here("R", "04_logistic_regression.R"))
source(here("R", "05_assessment.R"))
