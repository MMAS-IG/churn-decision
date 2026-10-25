# Train/test scores and figures for the tree methods.
# Run after R/trees/01_fit_trees.R.

if (!exists("tree_fit") || !exists("tree_test_p")) {
  source(here::here("R", "trees", "01_fit_trees.R"))
}

library(ggplot2)
library(dplyr)
library(rpart.plot)

theme_set(theme_minimal(base_size = 12))
dir.create(here::here("figures"), showWarnings = FALSE)

# The drawn tree is the pruned fit on all training rows.
print(tree_fit)
rpart.plot(
  tree_fit,
  type = 4,
  extra = 1,
  under = TRUE,
  fallen.leaves = TRUE,
  box.palette = "RdBu",
  main = "Pruned classification tree on the training rows"
)

tree_colours <- c(
  "Classification tree" = "#e7298a",
  "Bagging" = "#66a61e",
  "Random forest" = "#7570b3",
  "Boosting" = "#e6ab02",
  "Logistic regression" = "#018571"
)

p_tree_cp <- ggplot(tree_selection_plot, aes(x = n_leaf, y = val_brier)) +
  geom_line() +
  geom_point() +
  geom_point(
    data = tree_chosen,
    shape = 1,
    size = 6,
    colour = "grey10"
  ) +
  labs(
    x = "Number of leaves",
    y = expression(B[val]),
    title = "Validation Brier along the cost-complexity path"
  )
print(p_tree_cp)
ggsave(here::here("figures", "tree_cp_brier.png"), p_tree_cp, width = 7.2, height = 4.4)

p_bag <- ggplot(bag_path_plot, aes(x = M, y = oob_brier)) +
  geom_line(colour = tree_colours[["Bagging"]]) +
  geom_point(
    data = bag_chosen,
    aes(x = M, y = oob_brier),
    shape = 1,
    size = 6,
    colour = "grey10"
  ) +
  labs(
    x = "Number of trees M",
    y = expression(B[oob]),
    title = "Bagging: out-of-bag Brier of the vote share"
  )
print(p_bag)
ggsave(here::here("figures", "bagging_brier_path.png"), p_bag, width = 7.2, height = 4.4)

rf_path_plot$mtry_f <- factor(rf_path_plot$mtry)
p_rf <- ggplot(rf_path_plot, aes(x = M, y = oob_brier, colour = mtry_f)) +
  geom_line() +
  geom_point(
    data = rf_chosen,
    aes(x = M, y = oob_brier),
    inherit.aes = FALSE,
    shape = 1,
    size = 6,
    colour = "grey10"
  ) +
  labs(
    x = "Number of trees M",
    y = expression(B[oob]),
    colour = expression(m[try]),
    title = "Random forest: out-of-bag Brier of the vote share"
  )
print(p_rf)
ggsave(here::here("figures", "rf_brier_path.png"), p_rf, width = 7.5, height = 4.6)

boost_path_plot$depth_f <- factor(boost_path_plot$depth)
boost_path_plot$shrink_f <- factor(boost_path_plot$shrinkage)
p_boost <- ggplot(
  boost_path_plot,
  aes(x = M, y = val_brier, colour = shrink_f)
) +
  geom_line() +
  geom_point(
    data = boost_chosen,
    aes(x = M, y = val_brier),
    inherit.aes = FALSE,
    shape = 1,
    size = 6,
    colour = "grey10"
  ) +
  facet_wrap(~ depth_f, labeller = label_both) +
  labs(
    x = "Number of trees M",
    y = expression(B[val]),
    colour = "Shrinkage",
    title = "Boosting: validation Brier"
  )
print(p_boost)
ggsave(here::here("figures", "boosting_brier_path.png"), p_boost, width = 8, height = 4.6)

gini_bar <- function(imp, method) {
  d <- data.frame(
    covariate = rownames(imp),
    gini = imp[, "MeanDecreaseGini"],
    method = method,
    stringsAsFactors = FALSE
  )
  d <- d[order(d$gini), ]
  d$covariate <- factor(d$covariate, levels = d$covariate)
  ggplot(d, aes(x = gini, y = covariate, fill = method)) +
    geom_col() +
    scale_fill_manual(values = tree_colours, name = NULL) +
    labs(
      x = expression(Gini[j]),
      y = NULL,
      title = paste("Gini decrease:", method)
    )
}

bag_gini <- importance(bag_fit)
p_gini_bag <- gini_bar(bag_gini, "Bagging")
print(p_gini_bag)
ggsave(here::here("figures", "bagging_gini_importance.png"), p_gini_bag, width = 7.2, height = 5.2)

rf_imp <- importance(rf_fit, scale = FALSE)
p_gini_rf <- gini_bar(rf_imp, "Random forest")
print(p_gini_rf)
ggsave(here::here("figures", "rf_gini_importance.png"), p_gini_rf, width = 7.2, height = 5.2)

imp_perm <- data.frame(
  covariate = rownames(rf_imp),
  Ij = rf_imp[, "MeanDecreaseAccuracy"],
  method = "Random forest",
  stringsAsFactors = FALSE
)
imp_perm <- imp_perm[order(imp_perm$Ij), ]
imp_perm$covariate <- factor(imp_perm$covariate, levels = imp_perm$covariate)
p_imp <- ggplot(imp_perm, aes(x = Ij, y = covariate, fill = method)) +
  geom_col() +
  scale_fill_manual(values = tree_colours, name = NULL) +
  labs(
    x = expression(I[j]),
    y = NULL,
    title = "Permutation importance: Random forest"
  )
print(p_imp)
ggsave(here::here("figures", "rf_importance.png"), p_imp, width = 7.2, height = 5.2)

calibration_df <- function(y, p, model, n_bins = 10) {
  br <- unique(quantile(p, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
  if (length(br) < 3) br <- seq(min(p), max(p), length.out = 5)
  cuts <- cut(p, breaks = br, include.lowest = TRUE)
  data.frame(y = y, p = p, bin = cuts, model = model) |>
    dplyr::group_by(model, bin) |>
    dplyr::summarise(mean_pred = mean(p), mean_obs = mean(y), n = dplyr::n(), .groups = "drop")
}
tree_cal <- dplyr::bind_rows(
  calibration_df(y_test, tree_test_p, "Classification tree"),
  calibration_df(y_test, bag_test_p, "Bagging"),
  calibration_df(y_test, rf_test_p, "Random forest"),
  calibration_df(y_test, boost_test_p, "Boosting")
)
tree_cal$model <- factor(tree_cal$model, levels = names(tree_colours)[1:4])
p_cal <- ggplot(tree_cal, aes(x = mean_pred, y = mean_obs, colour = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_line() +
  geom_point(size = 2) +
  scale_colour_manual(values = tree_colours) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    x = "Mean predicted probability in the bin",
    y = "Observed churn rate",
    colour = NULL,
    title = "Calibration on the test set"
  )
print(p_cal)
ggsave(here::here("figures", "tree_calibration_test.png"), p_cal, width = 7.4, height = 5.2)

mode_factor <- function(x) {
  tab <- sort(table(x), decreasing = TRUE)
  factor(names(tab)[1], levels = levels(x))
}
base_row <- telco_train[1, , drop = FALSE]
num_vars <- setdiff(names(telco_train)[vapply(telco_train, is.numeric, logical(1))], "Churn")
fac_vars <- names(telco_train)[vapply(telco_train, is.factor, logical(1))]
for (v in num_vars) base_row[[v]] <- median(telco_train[[v]])
for (v in fac_vars) base_row[[v]] <- mode_factor(telco_train[[v]])

make_profile <- function(contract, tenure_seq = 0:72) {
  out <- base_row[rep(1, length(tenure_seq)), ]
  out$Contract <- factor(contract, levels = levels(telco_train$Contract))
  out$tenure <- tenure_seq
  out$TotalCharges <- out$tenure * out$MonthlyCharges
  out
}
tree_prof <- rbind(make_profile("Month-to-month"), make_profile("Two year"))
prof_fac <- as_churn_factor(tree_prof)
prof_num <- as_churn_numeric(tree_prof)
tree_prof$tree <- rpart_probability(tree_fit, prof_fac)
tree_prof$bagging <- vote_share_at(bag_fit, prof_fac, bag_chosen$M)
tree_prof$forest <- vote_share_at(rf_fit, prof_fac, rf_chosen$M)
tree_prof$boosting <- gbm_probability(boost_fit, prof_num, boost_chosen$M)
if (!exists("logit_mod")) {
  logit_mod <- glm(Churn ~ ., data = telco_train, family = binomial(link = "logit"))
}
tree_prof$logistic <- as.numeric(predict(logit_mod, newdata = tree_prof, type = "response"))
tree_prof$Contract <- factor(
  as.character(tree_prof$Contract),
  levels = c("Month-to-month", "Two year")
)

prof_long <- tidyr::pivot_longer(
  tree_prof,
  cols = c("tree", "bagging", "forest", "boosting", "logistic"),
  names_to = "model",
  values_to = "pred"
)
prof_long$model <- factor(
  prof_long$model,
  levels = c("tree", "bagging", "forest", "boosting", "logistic"),
  labels = c(
    "Classification tree", "Bagging", "Random forest",
    "Boosting", "Logistic regression"
  )
)
p_prof <- ggplot(prof_long, aes(x = tenure, y = pred, colour = model, linetype = Contract)) +
  geom_hline(yintercept = c(0, 1), linetype = "dotted") +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = tree_colours) +
  labs(
    x = "Tenure (months)",
    y = "Predicted P(churn = 1)",
    colour = NULL,
    linetype = "Contract",
    title = "Profile at a fixed training row"
  )
print(p_prof)
ggsave(here::here("figures", "tree_profile_contract_tenure.png"), p_prof, width = 8, height = 4.8)

brier_defined <- function(y, p) {
  ok <- !is.na(p)
  brier_score(y[ok], p[ok])
}

tree_holdout <- data.frame(
  method = c("Classification tree", "Bagging", "Random forest", "Boosting"),
  train_brier = c(
    brier_score(y_train, tree_train_p),
    brier_score(y_train, bag_train_p),
    brier_score(y_train, rf_train_p),
    brier_score(y_train, boost_train_p)
  ),
  oob_brier = c(
    NA_real_,
    brier_defined(y_train, bag_oob_p),
    brier_defined(y_train, rf_oob_p),
    NA_real_
  ),
  test_brier = c(
    brier_score(y_test, tree_test_p),
    brier_score(y_test, bag_test_p),
    brier_score(y_test, rf_test_p),
    brier_score(y_test, boost_test_p)
  ),
  test_log_loss = c(
    log_loss(y_test, tree_test_p),
    log_loss(y_test, bag_test_p),
    log_loss(y_test, rf_test_p),
    log_loss(y_test, boost_test_p)
  ),
  stringsAsFactors = FALSE
)
print(tree_holdout, digits = 4, row.names = FALSE)

confusion_at <- function(y, p, threshold) {
  yhat <- as.integer(p >= threshold)
  tp <- sum(y == 1 & yhat == 1)
  tn <- sum(y == 0 & yhat == 0)
  fp <- sum(y == 0 & yhat == 1)
  fn <- sum(y == 1 & yhat == 0)
  n_rows <- length(y)
  data.frame(
    threshold = threshold,
    accuracy = (tp + tn) / n_rows,
    sensitivity = if ((tp + fn) > 0) tp / (tp + fn) else NA_real_,
    specificity = if ((tn + fp) > 0) tn / (tn + fp) else NA_real_,
    precision = if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  )
}

train_prevalence <- mean(y_train)
tree_confusion <- rbind(
  cbind(method = "Classification tree", confusion_at(y_test, tree_test_p, 0.5)),
  cbind(method = "Bagging", confusion_at(y_test, bag_test_p, 0.5)),
  cbind(method = "Random forest", confusion_at(y_test, rf_test_p, 0.5)),
  cbind(method = "Boosting", confusion_at(y_test, boost_test_p, 0.5)),
  cbind(method = "Classification tree", confusion_at(y_test, tree_test_p, train_prevalence)),
  cbind(method = "Bagging", confusion_at(y_test, bag_test_p, train_prevalence)),
  cbind(method = "Random forest", confusion_at(y_test, rf_test_p, train_prevalence)),
  cbind(method = "Boosting", confusion_at(y_test, boost_test_p, train_prevalence))
)
print(tree_confusion, digits = 4, row.names = FALSE)
