
regression_csv <- function(fichier) {

  data <- read.csv(fichier)

  # Extraire les colonnes 2, 3, 4 et 5
  data <- data[, c(2, 3, 4, 5)]
  print("1")

  # Variable dépendante : Partner (No = 0, Yes = 1)
  data$Partner <- ifelse(data$Partner == "Yes", 1,
                         ifelse(data$Partner == "No", 0, NA))

  # Variables explicatives catégorielles
  data$gender <- as.factor(data$gender)
  data$Dependents <- as.factor(data$Dependents)

  # Supprimer les lignes incomplètes
  data <- na.omit(data)

  # Régression linéaire multiple
  modele <- lm(Partner ~ gender + SeniorCitizen + Dependents,
               data = data)

  summary(modele)
}