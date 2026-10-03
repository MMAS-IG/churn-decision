# Descriptive look at churn and the main account / service variables.

if (!exists("telco")) {
  source(here::here("R", "01_prepare_data.R"))
}

library(dplyr)
library(ggplot2)
library(scales)

theme_set(theme_minimal(base_size = 12))
dir.create(here::here("figures"), showWarnings = FALSE)

churn_rate <- mean(telco$Churn)
message("Overall churn rate: ", percent(churn_rate, accuracy = 0.1))

print(table(Churn = factor(telco$Churn, labels = c("No", "Yes"))))
print(round(prop.table(table(Churn = factor(telco$Churn, labels = c("No", "Yes")))), 3))

churn_by_contract <- telco %>%
  group_by(Contract) %>%
  summarise(n = n(), churn_rate = mean(Churn), .groups = "drop")
print(churn_by_contract)

churn_by_internet <- telco %>%
  group_by(InternetService) %>%
  summarise(n = n(), churn_rate = mean(Churn), .groups = "drop")
print(churn_by_internet)

churn_by_pay <- telco %>%
  group_by(PaymentMethod) %>%
  summarise(n = n(), churn_rate = mean(Churn), .groups = "drop")
print(churn_by_pay)

p_contract <- ggplot(telco, aes(x = Contract, fill = factor(Churn, labels = c("No", "Yes")))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = c("#2c7bb6", "#d7191c"), name = "Churn") +
  labs(x = "Contract", y = "Share of customers", title = "Churn mix by contract type")
print(p_contract)
ggsave(here::here("figures", "churn_by_contract.png"), p_contract, width = 7, height = 4.5)

p_tenure <- ggplot(telco, aes(x = tenure, fill = factor(Churn, labels = c("No", "Yes")))) +
  geom_histogram(binwidth = 3, position = "identity", alpha = 0.55) +
  scale_fill_manual(values = c("#2c7bb6", "#d7191c"), name = "Churn") +
  labs(x = "Tenure (months)", y = "Customers", title = "Tenure distribution by churn")
print(p_tenure)
ggsave(here::here("figures", "tenure_by_churn.png"), p_tenure, width = 7, height = 4.5)

p_charges <- ggplot(telco, aes(x = MonthlyCharges, fill = factor(Churn, labels = c("No", "Yes")))) +
  geom_density(alpha = 0.45) +
  scale_fill_manual(values = c("#2c7bb6", "#d7191c"), name = "Churn") +
  labs(x = "Monthly charges", y = "Density", title = "Monthly charges by churn")
print(p_charges)
ggsave(here::here("figures", "monthly_charges_by_churn.png"), p_charges, width = 7, height = 4.5)
