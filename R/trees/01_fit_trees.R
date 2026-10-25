# Classification tree, bagging, random forest, and gradient boosting.
# The tree and the boosting triple are chosen on a validation slice.
# Bagging and the random forest are grown on all training rows;
# their number of trees, and mtry, are chosen by the out-of-bag Brier score.
# The test set from R/01_prepare_data.R is not used to choose those values.

if (!exists("telco_train") || !exists("telco_test") || !exists("test_id")) {
  source(here::here("R", "01_prepare_data.R"))
}

library(rpart)
library(randomForest)
library(gbm)
source(here::here("R", "trees", "tree_lib.R"))
source(here::here("R", "compare_methods", "prediction_io.R"))

stopifnot(identical(as.integer(rownames(telco_test)), as.integer(test_id)))

train_fac <- as_churn_factor(telco_train)
test_fac <- as_churn_factor(telco_test)
p_cov <- ncol(train_fac) - 1L

# Same validation draw as the network: 20% of the training rows, seed 2027.
set.seed(2027)
val_n <- floor(0.20 * nrow(train_fac))
tree_val_id <- sample(seq_len(nrow(train_fac)), size = val_n)
fit_fac <- train_fac[-tree_val_id, , drop = FALSE]
val_fac <- train_fac[tree_val_id, , drop = FALSE]
y_fit <- as.integer(telco_train$Churn[-tree_val_id])
y_val <- as.integer(telco_train$Churn[tree_val_id])
y_train <- as.integer(telco_train$Churn)
y_test <- as.integer(telco_test$Churn)

message(
  "Tree tuning: ", nrow(fit_fac), " fitting rows, ",
  nrow(val_fac), " validation rows, ",
  p_cov, " covariates. Test set is held out (",
  nrow(test_fac), " rows)."
)

# --- one tree: cost-complexity path, scored by validation Brier ----------

tree_cp_floor <- 0.001
tree_grown <- rpart(
  Churn ~ .,
  data = fit_fac,
  method = "class",
  control = rpart.control(cp = tree_cp_floor, xval = 0)
)
tree_cp_values <- sort(unique(tree_grown$cptable[, "CP"]), decreasing = TRUE)
tree_selection <- do.call(rbind, lapply(tree_cp_values, function(cp) {
  pruned <- prune(tree_grown, cp = cp)
  p_val <- rpart_probability(pruned, val_fac)
  data.frame(
    cp = cp,
    nsplit = nrow(pruned$frame[pruned$frame$var == "<leaf>", ]) - 1L,
    n_leaf = sum(pruned$frame$var == "<leaf>"),
    val_brier = brier_score(y_val, p_val),
    val_brier_se = brier_se(y_val, p_val),
    stringsAsFactors = FALSE
  )
}))
tree_selection <- tree_selection[order(tree_selection$val_brier, -tree_selection$cp), ]
rownames(tree_selection) <- NULL
tree_chosen <- tree_selection[1, ]

# --- bagging: one forest on all training rows, OOB path in M -------------

bag_ntree <- 500L
set.seed(31000)
bag_fit <- randomForest(
  Churn ~ .,
  data = train_fac,
  ntree = bag_ntree,
  mtry = p_cov,
  replace = TRUE,
  sampsize = nrow(train_fac),
  keep.inbag = TRUE,
  importance = FALSE
)
bag_oob_path <- oob_vote_share_path(bag_fit, train_fac)
bag_path <- data.frame(
  M = seq_len(bag_ntree),
  mtry = p_cov,
  oob_brier = brier_along_oob(bag_oob_path, y_train)
)
bag_path <- bag_path[order(bag_path$oob_brier, bag_path$M), ]
rownames(bag_path) <- NULL
bag_chosen <- bag_path[1, ]
bag_chosen$oob_brier_se <- brier_se_defined(
  y_train, bag_oob_path[, bag_chosen$M]
)
bag_oob_p <- bag_oob_path[, bag_chosen$M]
bag_train_p <- vote_share_at(bag_fit, train_fac, bag_chosen$M)
bag_test_p <- vote_share_at(bag_fit, test_fac, bag_chosen$M)

# --- random forest: m_try < p, same OOB vote share -----------------------

rf_mtry_grid <- c(2L, 4L, 6L, 8L, 12L)
rf_ntree <- 500L
rf_path_list <- vector("list", length(rf_mtry_grid))
rf_oob_by_mtry <- list()
rf_fit_by_mtry <- list()
for (i in seq_along(rf_mtry_grid)) {
  mtry <- rf_mtry_grid[[i]]
  set.seed(32000L + mtry)
  message("Forest tuning mtry = ", mtry, ", ntree = ", rf_ntree)
  fit_m <- randomForest(
    Churn ~ .,
    data = train_fac,
    ntree = rf_ntree,
    mtry = mtry,
    replace = TRUE,
    sampsize = nrow(train_fac),
    keep.inbag = TRUE,
    importance = TRUE
  )
  oob_path <- oob_vote_share_path(fit_m, train_fac)
  rf_oob_by_mtry[[as.character(mtry)]] <- oob_path
  rf_fit_by_mtry[[as.character(mtry)]] <- fit_m
  rf_path_list[[i]] <- data.frame(
    M = seq_len(rf_ntree),
    mtry = mtry,
    oob_brier = brier_along_oob(oob_path, y_train)
  )
}
rf_path <- do.call(rbind, rf_path_list)
rownames(rf_path) <- NULL
rf_ranked <- rf_path[order(rf_path$oob_brier, rf_path$M, rf_path$mtry), ]
rownames(rf_ranked) <- NULL
rf_chosen <- rf_ranked[1, ]
rf_fit <- rf_fit_by_mtry[[as.character(rf_chosen$mtry)]]
rf_oob_p <- rf_oob_by_mtry[[as.character(rf_chosen$mtry)]][, rf_chosen$M]
rf_chosen$oob_brier_se <- brier_se_defined(y_train, rf_oob_p)
rf_train_p <- vote_share_at(rf_fit, train_fac, rf_chosen$M)
rf_test_p <- vote_share_at(rf_fit, test_fac, rf_chosen$M)
rf_fit_by_mtry <- NULL

# --- boosting: bernoulli loss, grid of depth and shrinkage ---------------

boost_depths <- c(1L, 2L, 3L)
boost_shrinkages <- c(0.01, 0.05, 0.1)
boost_ntree <- 1000L
boost_M_grid <- seq.int(25L, boost_ntree, by = 25L)
boost_bag_fraction <- 0.5
boost_min_node <- 10L
fit_num <- as_churn_numeric(fit_fac)
val_num <- as_churn_numeric(val_fac)

boost_path_list <- list()
boost_p_store <- list()
k <- 0L
for (depth in boost_depths) {
  for (shrinkage in boost_shrinkages) {
    k <- k + 1L
    seed <- 33000L + 100L * depth + as.integer(round(1000 * shrinkage))
    set.seed(seed)
    message(
      "Boosting tuning depth = ", depth,
      ", shrinkage = ", shrinkage,
      ", n.trees = ", boost_ntree
    )
    fit_b <- gbm(
      Churn ~ .,
      data = fit_num,
      distribution = "bernoulli",
      n.trees = boost_ntree,
      interaction.depth = depth,
      shrinkage = shrinkage,
      bag.fraction = boost_bag_fraction,
      n.minobsinnode = boost_min_node,
      cv.folds = 0,
      n.cores = 1,
      verbose = FALSE
    )
    p_mat <- gbm_probability_matrix(fit_b, val_num, boost_M_grid)
    key <- paste(depth, shrinkage, sep = "_")
    boost_p_store[[key]] <- p_mat
    boost_path_list[[k]] <- data.frame(
      M = boost_M_grid,
      depth = depth,
      shrinkage = shrinkage,
      val_brier = brier_along_trees(p_mat, y_val)
    )
  }
}
boost_path <- do.call(rbind, boost_path_list)
rownames(boost_path) <- NULL
boost_ranked <- boost_path[order(
  boost_path$val_brier, boost_path$M, boost_path$depth, boost_path$shrinkage
), ]
rownames(boost_ranked) <- NULL
boost_chosen <- boost_ranked[1, ]
boost_key <- paste(boost_chosen$depth, boost_chosen$shrinkage, sep = "_")
boost_col <- match(boost_chosen$M, boost_M_grid)
boost_chosen$val_brier_se <- brier_se(y_val, boost_p_store[[boost_key]][, boost_col])

message("Chosen tree cp = ", signif(tree_chosen$cp, 4),
        ", leaves = ", tree_chosen$n_leaf,
        ", validation Brier = ", round(tree_chosen$val_brier, 4))
message("Chosen bagging M = ", bag_chosen$M,
        ", OOB Brier = ", round(bag_chosen$oob_brier, 4))
message("Chosen forest mtry = ", rf_chosen$mtry,
        ", M = ", rf_chosen$M,
        ", OOB Brier = ", round(rf_chosen$oob_brier, 4))
message("Chosen boosting depth = ", boost_chosen$depth,
        ", shrinkage = ", boost_chosen$shrinkage,
        ", M = ", boost_chosen$M,
        ", validation Brier = ", round(boost_chosen$val_brier, 4))

# --- refit the tree and the boosting fit on all training rows ------------
# Bagging and the random forest above are already grown on those rows.

tree_fit <- prune(
  rpart(
    Churn ~ .,
    data = train_fac,
    method = "class",
    control = rpart.control(cp = tree_cp_floor, xval = 0)
  ),
  cp = tree_chosen$cp
)
tree_train_p <- rpart_probability(tree_fit, train_fac)
tree_test_p <- rpart_probability(tree_fit, test_fac)

train_num <- as_churn_numeric(train_fac)
test_num <- as_churn_numeric(test_fac)
set.seed(43000)
boost_fit <- gbm(
  Churn ~ .,
  data = train_num,
  distribution = "bernoulli",
  n.trees = boost_chosen$M,
  interaction.depth = boost_chosen$depth,
  shrinkage = boost_chosen$shrinkage,
  bag.fraction = boost_bag_fraction,
  n.minobsinnode = boost_min_node,
  cv.folds = 0,
  n.cores = 1,
  verbose = FALSE
)
boost_train_p <- gbm_probability(boost_fit, train_num, boost_chosen$M)
boost_test_p <- gbm_probability(boost_fit, test_num, boost_chosen$M)

split_info <- churn_split_info(telco, telco_train)

save_test_predictions(
  method_id = "04_classification_tree",
  method_label = "Classification tree",
  y = y_test,
  p = tree_test_p,
  test_row = test_id,
  split = split_info,
  detail = sprintf(
    "rpart class, cp %s, %d leaves",
    format(tree_chosen$cp, digits = 4, scientific = FALSE),
    sum(tree_fit$frame$var == "<leaf>")
  )
)
save_test_predictions(
  method_id = "05_bagging",
  method_label = "Bagging",
  y = y_test,
  p = bag_test_p,
  test_row = test_id,
  split = split_info,
  detail = sprintf(
    "randomForest mtry %d, %d trees, vote share",
    p_cov, bag_chosen$M
  )
)
save_test_predictions(
  method_id = "06_random_forest",
  method_label = "Random forest",
  y = y_test,
  p = rf_test_p,
  test_row = test_id,
  split = split_info,
  detail = sprintf(
    "randomForest mtry %d, %d trees, vote share",
    rf_chosen$mtry, rf_chosen$M
  )
)
save_test_predictions(
  method_id = "07_boosting",
  method_label = "Boosting",
  y = y_test,
  p = boost_test_p,
  test_row = test_id,
  split = split_info,
  detail = sprintf(
    "gbm bernoulli, depth %d, shrinkage %s, %d trees",
    boost_chosen$depth,
    format(boost_chosen$shrinkage, trim = TRUE, scientific = FALSE),
    boost_chosen$M
  )
)

bag_path_plot <- bag_path[order(bag_path$M), ]
rf_path_plot <- rf_path[order(rf_path$mtry, rf_path$M), ]
boost_path_plot <- boost_path[order(boost_path$depth, boost_path$shrinkage, boost_path$M), ]
tree_selection_plot <- tree_selection[order(tree_selection$cp, decreasing = TRUE), ]

cat("Cost-complexity path on the validation slice:\n")
print(tree_selection_plot, digits = 4, row.names = FALSE)
cat("Bagging out-of-bag Brier at M = 1, 50, 100, 250, 500:\n")
print(
  bag_path_plot[bag_path_plot$M %in% c(1L, 50L, 100L, 250L, 500L), ],
  digits = 4, row.names = FALSE
)
rf_best_by_mtry <- do.call(rbind, lapply(split(rf_path, rf_path$mtry), function(d) {
  d[order(d$oob_brier, d$M), ][1, ]
}))
cat("Best M inside each mtry, by out-of-bag Brier:\n")
print(rf_best_by_mtry, digits = 4, row.names = FALSE)
boost_best_by_pair <- do.call(rbind, lapply(
  split(boost_path, interaction(boost_path$depth, boost_path$shrinkage, drop = TRUE)),
  function(d) d[order(d$val_brier, d$M), ][1, ]
))
cat("Best M inside each (depth, shrinkage):\n")
print(boost_best_by_pair, digits = 4, row.names = FALSE)
