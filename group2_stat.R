install.packages('readr')
install.packages(c("tidyverse", "caret", "pROC", "car", "broom"), dependencies = TRUE)

library(tidyverse) # Manipulation de données (dplyr) et graphiques (ggplot2)
library(caret)     # Séparation train/test et matrice de confusion
library(pROC)      # Courbe ROC et score AUC
library(car)       # Test de multicolonnalité (VIF)
library(broom)     # Tableaux de résultats propres (Odds Ratios)
library(readr)
library(pROC)


# 1. On recharge le fichier pour effacer l'erreur (remplace par ton chemin)
churn_table <- read.csv("C:/Users/kine/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Nettoyage : suppression de l'ID, conversion en 0/1, et retrait des valeurs manquantes
churn_table$customerID <- NULL
churn_table$Churn <- ifelse(churn_table$Churn == "Yes", 1, 0)
churn_table <- na.omit(churn_table)

# 3. Séparation Train / Test
set.seed(123)
index <- sample(1:nrow(churn_table), size = 0.7 * nrow(churn_table))
donnees_entrainement <- churn_table[index, ]
donnees_test <- churn_table[-index, ]

# 4. Modèle de régression
modele_churn <- glm(Churn ~ ., data = donnees_entrainement, family = binomial)

# 5. Prédictions et courbe ROC
library(pROC)
probabilites_test <- predict(modele_churn, newdata = donnees_test, type = "response")
courbe <- roc(donnees_test$Churn, probabilites_test)

# 6. Affichage du graphique et du score
plot(courbe, col = "blue", lwd = 2, main = "Courbe ROC - Prédiction du Churn")
auc(courbe)