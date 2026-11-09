rm(list = ls())

# 1. Importation et exploration des données

data <- read.csv(file.choose(), stringsAsFactors = TRUE)

dim(data)
str(data)
summary(data)

# Répartition des clients qui partent et qui restent
table(data$Churn)
round(prop.table(table(data$Churn)) * 100, 2)


# 2. Préparation des données

# Conversion de TotalCharges en variable numérique
data$TotalCharges <- as.numeric(trimws(as.character(data$TotalCharges)))

# Variable cible : 1 si le client part, 0 sinon
data$Churn_num <- ifelse(data$Churn == "Yes", 1, 0)

# Suppression des observations incomplètes pour le modèle
variables <- c("Churn_num", "tenure", "MonthlyCharges",
               "TotalCharges", "Contract", "InternetService")

data_model <- data[complete.cases(data[, variables]), ]


# 3. Régression linéaire multiple

fit_lm <- lm(Churn_num ~ tenure + MonthlyCharges + TotalCharges +
               Contract + InternetService,
             data = data_model)

summary(fit_lm)


# 4. Prédictions

data_model$Predicted_Prob <- predict(fit_lm, newdata = data_model)

# Classe prédite avec un seuil de 0.5
data_model$Predicted_Class <- ifelse(
  data_model$Predicted_Prob >= 0.5, 1, 0
)

head(data_model[, c("Churn", "Churn_num",
                    "Predicted_Prob", "Predicted_Class")])


# 5. Evaluation du modèle

conf_matrix <- table(
  Reel = data_model$Churn_num,
  Predit = data_model$Predicted_Class
)

print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

cat("Taux de bonnes predictions :",
    round(accuracy * 100, 2), "%\n")


# 6. Graphique des probabilités prédites

boxplot(Predicted_Prob ~ Churn,
        data = data_model,
        col = c("lightblue", "salmon"),
        main = "Probabilites predites selon le churn reel",
        xlab = "Churn reel",
        ylab = "Probabilite predite",
        ylim = c(0, 1))

abline(h = 0.5, lty = 2, col = "red", lwd = 2)

legend("topleft",
       legend = "Seuil de classification = 0.5",
       col = "red",
       lty = 2,
       bty = "n")


# 7. Graphique de la regression lineaire multiple

# On fixe les autres variables a leur valeur moyenne ou la plus frequente
new_data <- data.frame(
  tenure = mean(data_model$tenure),
  MonthlyCharges = seq(
    min(data_model$MonthlyCharges),
    max(data_model$MonthlyCharges),
    length.out = 100
  ),
  TotalCharges = mean(data_model$TotalCharges),
  Contract = factor(
    names(which.max(table(data_model$Contract))),
    levels = levels(data_model$Contract)
  ),
  InternetService = factor(
    names(which.max(table(data_model$InternetService))),
    levels = levels(data_model$InternetService)
  )
)

# Predictions du modele multiple
predictions <- predict(fit_lm, newdata = new_data)

# Nuage de points et droite de regression
plot(data_model$MonthlyCharges, data_model$Churn_num,
     pch = 16,
     col = rgb(0, 0, 1, 0.2),
     xlab = "Monthly Charges",
     ylab = "Churn (0 = No, 1 = Yes)",
     main = "Regression lineaire multiple")

lines(new_data$MonthlyCharges, predictions,
      col = "red", lwd = 3)

legend("topleft",
       legend = "Droite estimee par le modele multiple",
       col = "red", lwd = 3, bty = "n")


