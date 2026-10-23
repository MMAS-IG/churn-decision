# Fit and assess the churn network from the project root.
# Test-set scores are written to output/test_predictions/03_neural_network.rds.

library(here)
source(here("R", "neural_network", "01_fit_network.R"))
source(here("R", "neural_network", "02_assess_network.R"))
