# Prepare the IBM Telco customer churn data for linear and logistic models.
# Run from the project root (or open churn-decision.Rproj).

library(here)
library(dplyr)

data_path <- here("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
telco_raw <- read.csv(
  data_path,
  stringsAsFactors = FALSE,
  na.strings = c("", " ", "NA")
)

# TotalCharges is blank for 11 customers with tenure = 0 (no bill yet).
n_blank_total <- sum(is.na(telco_raw$TotalCharges))
telco_raw$TotalCharges[is.na(telco_raw$TotalCharges)] <- 0

# Nested levels: "No phone service" / "No internet service" repeat
# PhoneService / InternetService and create perfectly collinear dummies.
# All original columns are kept; redundant labels are recoded to "No".
phone_addon <- "MultipleLines"
internet_addons <- c(
  "OnlineSecurity", "OnlineBackup", "DeviceProtection",
  "TechSupport", "StreamingTV", "StreamingMovies"
)

telco_raw[[phone_addon]] <- ifelse(
  telco_raw[[phone_addon]] == "No phone service",
  "No",
  telco_raw[[phone_addon]]
)
for (v in internet_addons) {
  telco_raw[[v]] <- ifelse(
    telco_raw[[v]] == "No internet service",
    "No",
    telco_raw[[v]]
  )
}

telco <- telco_raw %>%
  mutate(
    Churn = as.integer(Churn == "Yes"),
    SeniorCitizen = factor(
      SeniorCitizen,
      levels = c(0, 1),
      labels = c("No", "Yes")
    )
  ) %>%
  mutate(across(where(is.character), factor)) %>%
  select(-customerID)

# Explicit reference levels for later coefficient reading.
telco$Contract <- relevel(telco$Contract, ref = "Month-to-month")
telco$InternetService <- relevel(telco$InternetService, ref = "DSL")
telco$PaymentMethod <- relevel(
  telco$PaymentMethod,
  ref = "Bank transfer (automatic)"
)
telco$gender <- relevel(telco$gender, ref = "Female")

stopifnot(!anyNA(telco))
stopifnot(all(telco$Churn %in% c(0L, 1L)))

set.seed(2026)
n <- nrow(telco)
test_frac <- 0.30
test_id <- sample(seq_len(n), size = floor(test_frac * n))
telco_train <- telco[-test_id, , drop = FALSE]
telco_test <- telco[test_id, , drop = FALSE]

churn_formula <- Churn ~ .

message(
  "Prepared ", nrow(telco), " rows; ",
  n_blank_total, " TotalCharges values set to 0; ",
  "train n = ", nrow(telco_train), ", test n = ", nrow(telco_test), "."
)
