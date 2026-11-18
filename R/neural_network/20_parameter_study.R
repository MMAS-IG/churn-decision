# Étude des paramètres du réseau de neurones (nnet) : size, decay, rang.
#
# Consigne du prof :
#   1. identifier les paramètres importants,
#   2. évaluer les plages de valeurs à utiliser,
#   3. faire tourner l'algorithme sur ces plages et comparer les résultats.
#
# Découpage : seed 123 (celle du groupe 1), split 80/20 stratifié sur Churn.
#
# À placer dans R/neural_network/ et à lancer depuis la racine du projet :
#   source(here::here("R", "neural_network", "20_parameter_study.R"),
#          local = new.env())
#
# Les résultats sont écrits dans output/parameter_study/.
# Rien n'est écrit dans output/test_predictions/ : ce découpage 80/20 est
# différent du 70/30 du cours, le comparateur du cours refuserait ces fichiers.

library(here)
library(rsample)
library(nnet)
library(dplyr)
library(ggplot2)
library(pROC)

## ----- Réglage de la taille de l'étude -----
## QUICK = TRUE : petite grille, quelques minutes. À lancer d'abord pour vérifier
## que tout marche. QUICK = FALSE : grille complète pour les résultats finaux.
QUICK <- FALSE

if (QUICK) {
  sizes  <- c(0, 2, 5, 10)
  decays <- c(0.05, 0.5, 2)
  rangs  <- c(0.1, 0.5, 2)
  n_starts <- 2
  final_starts <- 3
} else {
  sizes  <- c(0, 1, 2, 5, 10, 15)
  decays <- c(0, 0.05, 0.5, 2, 5)
  rangs  <- c(0.1, 0.5, 1, 2)
  n_starts <- 3
  final_starts <- 5
}
maxit <- 400

## ----- Données et code du cours -----
if (!exists("telco")) source(here("R", "01_prepare_data.R"))
source(here("R", "neural_network", "nn_lib.R"))

out_dir <- here("output", "parameter_study")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

## ----- 1. Découpage 80/20 stratifié, seed 123 -----
set.seed(123)
telco_split <- initial_split(telco, prop = 0.80, strata = Churn)
train_data  <- training(telco_split)
test_data   <- testing(telco_split)
message("Train : ", nrow(train_data), " lignes, taux de départ ",
        round(mean(train_data$Churn), 3))
message("Test  : ", nrow(test_data), " lignes, taux de départ ",
        round(mean(test_data$Churn), 3))

nn_train <- as_churn_factor(train_data)
nn_test  <- as_churn_factor(test_data)

# 20 % du train pour choisir les réglages (comme dans le cours, graine 2027).
set.seed(2027)
val_id <- sample(seq_len(nrow(nn_train)), size = floor(0.20 * nrow(nn_train)))
fit_slice <- nn_train[-val_id, , drop = FALSE]
val_slice <- nn_train[val_id, , drop = FALSE]

spec  <- predictor_center_scale(fit_slice)
fit_s <- apply_center_scale(fit_slice, spec)
val_s <- apply_center_scale(val_slice, spec)
y_fit <- as.integer(fit_s$Churn == "Yes")
y_val <- as.integer(val_s$Churn == "Yes")

## ----- 2. Grille size x decay x rang (évaluée sur la validation) -----
grid <- expand.grid(size = sizes, decay = decays, rang = rangs)
message("Grille : ", nrow(grid), " combinaisons, ", n_starts, " départs chacune.")
rows <- vector("list", nrow(grid))
t0 <- Sys.time()
for (k in seq_len(nrow(grid))) {
  g <- grid[k, ]
  starts <- lapply(seq_len(n_starts), function(s) {
    fit_nnet_once(fit_s, size = g$size, decay = g$decay,
                  seed = 1000L * s + k, maxit = maxit, rang = g$rang)
  })
  crit <- vapply(starts, function(f) f$value, numeric(1))
  fit  <- starts[[which.min(crit)]]
  p_fit <- nnet_probability(fit, fit_s)
  p_val <- nnet_probability(fit, val_s)
  rows[[k]] <- data.frame(
    size = g$size, decay = g$decay, rang = g$rang,
    n_weights = length(fit$wts),
    converged = fit$convergence == 0,
    train_brier = brier_score(y_fit, p_fit),
    val_brier = brier_score(y_val, p_val),
    val_brier_se = brier_se(y_val, p_val),
    val_log_loss = log_loss(y_val, p_val)
  )
  message(sprintf("[%d/%d] size=%g decay=%g rang=%g  val Brier=%.4f",
                  k, nrow(grid), g$size, g$decay, g$rang, rows[[k]]$val_brier))
}
message("Durée de la grille : ",
        round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), " min")

res <- bind_rows(rows)
res <- res[order(res$val_brier, res$size, -res$decay), ]
rownames(res) <- NULL
best_any    <- res[1, ]                      # peut avoir size = 0
best_hidden <- res[res$size > 0, ][1, ]      # meilleur réseau avec couche cachée

# Combinaisons indiscernables de la meilleure (dans 1 erreur-type).
res$within_1se <- res$val_brier <= best_any$val_brier + best_any$val_brier_se
write.csv(res, file.path(out_dir, "grid_results.csv"), row.names = FALSE)

cat("\n=== 10 meilleures combinaisons (validation) ===\n")
print(head(res, 10), digits = 4, row.names = FALSE)
cat("\nMeilleure ligne de la grille : size =", best_any$size,
    ", decay =", best_any$decay, ", rang =", best_any$rang,
    ", Brier val =", round(best_any$val_brier, 4), "\n")
cat("Meilleur réseau (>= 1 unité) : size =", best_hidden$size,
    ", decay =", best_hidden$decay, ", rang =", best_hidden$rang,
    ", Brier val =", round(best_hidden$val_brier, 4), "\n")
cat("Réglages non convergés :", sum(!res$converged), "sur", nrow(res), "\n")

## Effet de chaque paramètre : meilleur et moyen Brier de validation par valeur
marginal <- bind_rows(lapply(c("size", "decay", "rang"), function(v) {
  res %>%
    group_by(level = .data[[v]]) %>%
    summarise(best_val_brier = min(val_brier),
              mean_val_brier = mean(val_brier), .groups = "drop") %>%
    mutate(parameter = v, .before = 1)
}))
cat("\n=== Effet de chaque paramètre ===\n")
print(as.data.frame(marginal), digits = 4, row.names = FALSE)
write.csv(marginal, file.path(out_dir, "effect_of_each_parameter.csv"), row.names = FALSE)

## Plages de valeurs qui ne se distinguent pas du meilleur réglage
ok <- res[res$within_1se & res$size > 0, ]
ranges <- data.frame(
  parameter = c("size", "decay", "rang"),
  tested = c(paste(sizes, collapse = ", "), paste(decays, collapse = ", "),
             paste(rangs, collapse = ", ")),
  within_1se_min = c(min(ok$size), min(ok$decay), min(ok$rang)),
  within_1se_max = c(max(ok$size), max(ok$decay), max(ok$rang))
)
cat("\n=== Valeurs testées et plages indiscernables du meilleur (1 erreur-type) ===\n")
print(ranges, row.names = FALSE)
write.csv(ranges, file.path(out_dir, "useful_ranges.csv"), row.names = FALSE)

## ----- 3. Figures -----
g1 <- ggplot(res, aes(factor(decay), factor(size), fill = val_brier)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.3f", val_brier)), size = 2.6) +
  facet_wrap(~rang, labeller = label_both) +
  scale_fill_viridis_c(direction = -1) +
  labs(x = "decay", y = "size", fill = "Brier\nvalidation",
       title = "Brier de validation selon size, decay et rang") +
  theme_minimal(base_size = 11)
ggsave(file.path(out_dir, "heatmap_val_brier.png"), g1, width = 9, height = 5.5, dpi = 150)

g2 <- ggplot(res, aes(size, val_brier, colour = factor(decay), group = decay)) +
  geom_line() + geom_point() +
  facet_wrap(~rang, labeller = label_both) +
  labs(x = "size (unités cachées)", y = "Brier de validation", colour = "decay",
       title = "Effet de size, pour chaque decay et chaque rang") +
  theme_minimal(base_size = 11)
ggsave(file.path(out_dir, "val_brier_vs_size.png"), g2, width = 9, height = 5.5, dpi = 150)

## ----- 4. Réajustement sur tout le train et comparaison sur le test -----
refit_with <- function(train_raw, size, decay, rang, n_starts, maxit) {
  sp <- predictor_center_scale(train_raw)
  tr <- apply_center_scale(train_raw, sp)
  fits <- lapply(seq_len(n_starts), function(s) {
    fit_nnet_once(tr, size = size, decay = decay, seed = 20000L + s,
                  maxit = maxit, rang = rang)
  })
  best <- which.min(vapply(fits, function(f) f$value, numeric(1)))
  list(fit = fits[[best]], spec = sp)
}
predict_test <- function(m) {
  nnet_probability(m$fit, apply_center_scale(nn_test, m$spec))
}

y_test <- test_data$Churn   # 0/1

lpm     <- lm(Churn ~ ., data = train_data)
logit   <- glm(Churn ~ ., data = train_data, family = binomial)
p_lpm   <- as.numeric(predict(lpm, test_data))
p_logit <- as.numeric(predict(logit, test_data, type = "response"))

nn_best   <- refit_with(nn_train, best_hidden$size, best_hidden$decay,
                        best_hidden$rang, final_starts, maxit)
nn_course <- refit_with(nn_train, 10, 2, 0.5, final_starts, maxit)  # réglage du cours
p_best    <- predict_test(nn_best)
p_course  <- predict_test(nn_course)

metrics <- function(label, p) {
  r <- roc(y_test, p, levels = c(0, 1), direction = "<", quiet = TRUE)
  data.frame(model = label,
             brier = brier_score(y_test, p),
             log_loss = log_loss(y_test, p),
             auc = as.numeric(auc(r)),
             sens_05 = mean(p[y_test == 1] >= 0.5),
             spec_05 = mean(p[y_test == 0] < 0.5))
}
test_table <- bind_rows(
  metrics("Linear probability", p_lpm),
  metrics("Logistic regression", p_logit),
  metrics(sprintf("NN course (size 10, decay 2, rang 0.5)"), p_course),
  metrics(sprintf("NN best (size %g, decay %g, rang %g)",
                  best_hidden$size, best_hidden$decay, best_hidden$rang), p_best)
)
cat("\n=== Résultats sur le test (80/20, seed 123) ===\n")
print(test_table, digits = 4, row.names = FALSE)
write.csv(test_table, file.path(out_dir, "test_comparison.csv"), row.names = FALSE)

# Différences de Brier appariées (bootstrap, 2000 rééchantillons du test).
# delta > 0 : le premier modèle a le plus grand Brier, donc est moins bon.
set.seed(2028)
idx <- replicate(2000, sample.int(length(y_test), replace = TRUE))
boot_diff <- function(a_label, pa, b_label, pb) {
  d <- (pa - y_test)^2 - (pb - y_test)^2
  ci <- quantile(colMeans(matrix(d[idx], nrow = length(y_test))), c(0.025, 0.975))
  data.frame(a = a_label, b = b_label, delta = mean(d),
             ci_low = ci[[1]], ci_high = ci[[2]])
}
diffs <- bind_rows(
  boot_diff("Logistic", p_logit, "NN best", p_best),
  boot_diff("NN course", p_course, "NN best", p_best),
  boot_diff("Logistic", p_logit, "NN course", p_course)
)
cat("\n=== Différences de Brier (IC bootstrap 95 %) ===\n")
print(diffs, digits = 4, row.names = FALSE)
write.csv(diffs, file.path(out_dir, "test_brier_differences.csv"), row.names = FALSE)

cat("\nFichiers écrits dans :", out_dir, "\n")
