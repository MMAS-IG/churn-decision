############################################################
# TELCO CUSTOMER CHURN ANALYSIS
# Models and Methods in Applied Statistics
#
# Objective:
# Develop and evaluate an interpretable statistical model
# for predicting customer churn.
############################################################


############################################################
# 1. LOAD REQUIRED PACKAGES
############################################################

library(rsample)
library(pROC)


############################################################
# 2. IMPORT THE DATA
############################################################

telco <- read.csv("C:/Users/LENOVO/Desktop/Models and Medhods in Applied Statistics/Group Work - 2026-2027/Data/WA_Fn-UseC_-Telco-Customer-Churn.csv", stringsAsFactors = FALSE)



############################################################
# 3. INITIAL DATA INSPECTION
############################################################

# Check dimensions
dim(telco)

# Examine variable names
names(telco)

# Inspect variable types and structure
str(telco)

# Preview the first observations
head(telco)

# Check missing values
colSums(is.na(telco))

# Check for duplicated rows
sum(duplicated(telco))

# Check whether customer IDs are unique
length(unique(telco$customerID))
nrow(telco)


############################################################
# 4. RESPONSE VARIABLE AND CLASS DISTRIBUTION
############################################################

# Number of customers in each churn category
table(telco$Churn)

# Churn proportions
prop.table(table(telco$Churn))

# Churn percentages
round(100 * prop.table(table(telco$Churn)),2)

# Visualize the class distribution
barplot(
  table(telco$Churn),
  main = "Distribution of Customer Churn",
  xlab = "Churn",
  ylab = "Number of Customers")


############################################################
# 5. EXPLORATORY DATA ANALYSIS
############################################################

# ----- Contract and Churn -----

contract_churn <- table(telco$Contract, telco$Churn)

contract_churn

round(100 * prop.table(contract_churn, margin = 1), 2)

barplot(
  t(prop.table(contract_churn, margin = 1)),
  beside = FALSE,
  legend.text = TRUE,
  main = "Churn Proportions by Contract Type",
  xlab = "Contract Type",
  ylab = "Proportion")


# ----- Internet Service and Churn -----

internet_churn <- table(telco$InternetService, telco$Churn)

internet_churn

round(100 * prop.table(internet_churn, margin = 1), 2)

barplot(
  t(prop.table(internet_churn, margin = 1)),
  beside = FALSE,
  legend.text = TRUE,
  main = "Churn Proportions by Internet Service",
  xlab = "Internet Service Type",
  ylab = "Proportion")


# ----- Payment Method and Churn -----

payment_churn <- table(telco$PaymentMethod, telco$Churn)

payment_churn

round(100 * prop.table(payment_churn, margin = 1), 2)

barplot(
  t(prop.table(payment_churn, margin = 1)),
  beside = FALSE,
  legend.text = TRUE,
  main = "Churn Proportions by Payment Method",
  xlab = "Payment Method Type",
  ylab = "Proportion")

# ----- Tech Support and Churn -----

techsupport_churn <- table(telco$TechSupport, telco$Churn)

techsupport_churn

round(100 * prop.table(techsupport_churn, margin = 1), 2)

barplot(
  t(prop.table(techsupport_churn, margin = 1)),
  beside = FALSE,
  legend.text = TRUE,
  main = "Churn Proportions by Tech Support",
  xlab = "Tech Support",
  ylab = "Proportion")

# ----- Tenure -----

summary(telco$tenure)

aggregate(
  tenure ~ Churn,
  data = telco,
  FUN = summary
)

boxplot(
  tenure ~ Churn,
  data = telco,
  main = "Tenure by Churn Status",
  xlab = "Churn",
  ylab = "Tenure (Months)"
)


# ----- Monthly Charges -----

aggregate(
  MonthlyCharges ~ Churn,
  data = telco,
  FUN = summary
)

boxplot(
  MonthlyCharges ~ Churn,
  data = telco,
  main = "Monthly Charges by Churn Status",
  xlab = "Churn",
  ylab = "Monthly Charges"
)


# ----- Total Charges -----

aggregate(
  TotalCharges ~ Churn,
  data = telco,
  FUN = summary
)

boxplot(
  TotalCharges ~ Churn,
  data = telco,
  main = "Total Charges by Churn Status",
  xlab = "Churn",
  ylab = "Total Charges"
)


# ----- Relationships among numerical predictors -----

numeric_vars <- telco[c("tenure", "MonthlyCharges", "TotalCharges")]

cor(numeric_vars, use = "complete.obs")

plot(
  telco$tenure,
  telco$TotalCharges,
  xlab = "Tenure (Months)",
  ylab = "Total Charges",
  main = "Tenure versus Total Charges"
)


############################################################
# 6. INVESTIGATE MISSING TOTAL CHARGES
############################################################

# Inspect observations with missing TotalCharges
telco[is.na(telco$TotalCharges), c("customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn")]



# Verify whether all missing TotalCharges observations
# correspond to zero tenure
table(telco$tenure[is.na(telco$TotalCharges)])


############################################################
# 7. DATA PREPROCESSING
############################################################

# The missing TotalCharges observations correspond to
# customers with zero tenure. TotalCharges is therefore
# set to zero as an explicit data-preparation assumption.
telco$TotalCharges[is.na(telco$TotalCharges) & telco$tenure == 0] <- 0


# Convert SeniorCitizen from a numerical code
# to a categorical variable
telco$SeniorCitizen <- factor(telco$SeniorCitizen, levels = c(0, 1), labels = c("No", "Yes"))


# Convert categorical variables to factors
categorical_vars <- c(
  "gender",
  "Partner",
  "Dependents",
  "PhoneService",
  "MultipleLines",
  "InternetService",
  "OnlineSecurity",
  "OnlineBackup",
  "DeviceProtection",
  "TechSupport",
  "StreamingTV",
  "StreamingMovies",
  "Contract",
  "PaperlessBilling",
  "PaymentMethod",
  "Churn"
)

telco[categorical_vars] <- lapply(telco[categorical_vars], factor)


# Verify preprocessing
str(telco)

colSums(is.na(telco))

levels(telco$Churn)

levels(telco$SeniorCitizen)

dim(telco)


############################################################
# 8. TRAIN-TEST SPLIT
############################################################

# Use a fixed seed to make the split reproducible
set.seed(123)

# Create a stratified 80/20 train-test split
# so that the Churn distribution is preserved
telco_split <- initial_split(
  telco,
  prop = 0.80,
  strata = Churn
)

train_data <- training(telco_split)
test_data <- testing(telco_split)


# Check sample sizes
dim(train_data)
dim(test_data)


# Verify Churn distributions
round(100 * prop.table(table(train_data$Churn)), 2)

round(100 * prop.table(table(test_data$Churn)), 2)


############################################################
# 9. MAJORITY-CLASS BASELINE
############################################################

# Identify the majority class using the training data
table(train_data$Churn)


# Predict the majority class ("No") for every test customer
baseline_predictions <- factor(
  rep("No", nrow(test_data)),
  levels = levels(test_data$Churn)
)


# Calculate baseline accuracy
baseline_accuracy <- mean(
  baseline_predictions == test_data$Churn
)

baseline_accuracy


############################################################
# 10. HANDLE STRUCTURAL REDUNDANCY IN SERVICE VARIABLES
############################################################

# Some original service variables contain structural
# redundancy. For example, "No internet service" is
# determined by InternetService = "No".
#
# Create binary indicators for whether each service
# is present, while retaining InternetService separately.

train_data$MultipleLines_binary <- factor(
  ifelse(
    train_data$MultipleLines == "Yes",
    "Yes",
    "No"
  ),
  levels = c("No", "Yes")
)

test_data$MultipleLines_binary <- factor(
  ifelse(
    test_data$MultipleLines == "Yes",
    "Yes",
    "No"
  ),
  levels = c("No", "Yes")
)


internet_service_vars <- c(
  "OnlineSecurity",
  "OnlineBackup",
  "DeviceProtection",
  "TechSupport",
  "StreamingTV",
  "StreamingMovies"
)


# Apply the same deterministic recoding rule
# to the training and test datasets
for (v in internet_service_vars) {
  
  new_name <- paste0(v, "_binary")
  
  train_data[[new_name]] <- factor(
    ifelse(
      train_data[[v]] == "Yes",
      "Yes",
      "No"
    ),
    levels = c("No", "Yes")
  )
  
  test_data[[new_name]] <- factor(
    ifelse(
      test_data[[v]] == "Yes",
      "Yes",
      "No"
    ),
    levels = c("No", "Yes")
  )
}


############################################################
# 11. FIT THE LOGISTIC REGRESSION MODEL
############################################################

churn_model <- glm(
  Churn ~ gender +
    SeniorCitizen +
    Partner +
    Dependents +
    tenure +
    PhoneService +
    MultipleLines_binary +
    InternetService +
    OnlineSecurity_binary +
    OnlineBackup_binary +
    DeviceProtection_binary +
    TechSupport_binary +
    StreamingTV_binary +
    StreamingMovies_binary +
    Contract +
    PaperlessBilling +
    PaymentMethod +
    MonthlyCharges +
    TotalCharges,
  data = train_data,
  family = binomial(link = "logit")
)


# Examine model results
summary(churn_model)


# Check for exact linear dependencies
alias(churn_model)


############################################################
# 12. ODDS RATIOS AND CONFIDENCE INTERVALS
############################################################

# Calculate odds ratios and Wald 95% confidence intervals
OR_table <- cbind(
  Odds_Ratio = exp(coef(churn_model)),
  exp(confint.default(churn_model))
)

round(
  OR_table,
  3
)


# Interpret TotalCharges on a more meaningful scale:
# odds ratio associated with a 100-unit increase
exp(100 * coef(churn_model)["TotalCharges"])


############################################################
# 13. TEST-SET PREDICTED PROBABILITIES
############################################################

# Generate predicted probabilities of Churn = "Yes"
test_prob <- predict(
  churn_model,
  newdata = test_data,
  type = "response"
)


# Examine predicted probabilities
head(test_prob)

summary(test_prob)


############################################################
# 14. CLASSIFICATION USING A 0.50 THRESHOLD
############################################################

test_pred <- factor(
  ifelse(
    test_prob >= 0.50,
    "Yes",
    "No"
  ),
  levels = c("No", "Yes")
)


############################################################
# 15. CONFUSION MATRIX
############################################################

conf_matrix <- table(Actual = test_data$Churn, Predicted = test_pred)

conf_matrix


# Extract confusion matrix components
TN <- conf_matrix["No", "No"]
FP <- conf_matrix["No", "Yes"]
FN <- conf_matrix["Yes", "No"]
TP <- conf_matrix["Yes", "Yes"]


############################################################
# 16. CLASSIFICATION PERFORMANCE
############################################################

accuracy <- (TP + TN) / sum(conf_matrix)

sensitivity <- TP / (TP + FN)

specificity <- TN / (TN + FP)

precision <- TP / (TP + FP)

f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)


# Display evaluation metrics
evaluation_metrics <- c(
  Accuracy = accuracy,
  Sensitivity = sensitivity,
  Specificity = specificity,
  Precision = precision,
  F1 = f1_score,
  Baseline_Accuracy = baseline_accuracy
)

round(evaluation_metrics,3)


############################################################
# 17. ROC CURVE AND AUC
############################################################

roc_obj <- roc(
  response = test_data$Churn,
  predictor = test_prob,
  levels = c("No", "Yes"),
  direction = "<"
)


# Calculate AUC
auc(roc_obj)


# Plot ROC curve
plot(
  roc_obj,
  main = "ROC Curve for Logistic Regression Model"
)

# Add no-discrimination reference line
abline(
  a = 0,
  b = 1,
  lty = 2
)


############################################################
# 18. FINAL RESULTS TO REPORT
############################################################

# Confusion matrix
conf_matrix

# Main predictive performance measures
round(
  evaluation_metrics,
  3
)

# AUC
auc(roc_obj)

# Odds ratios and confidence intervals
round(
  OR_table,
  3
)