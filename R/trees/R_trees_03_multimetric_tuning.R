# ==============================================================================
# GROUP 4 — MULTI-METRIC OOB TUNING & TEST VALIDATION (STEPS A, B, C)
# Replicates exact seeds & grids from R/trees/01_fit_trees.R without overwriting
# existing .rds prediction files.
# ==============================================================================

if (!exists("telco_train") || !exists("telco_test") || !exists("test_id")) {
  source(here::here("R", "01_prepare_data.R"))
}

library(randomForest)
source(here::here("R", "trees", "tree_lib.R"))

stopifnot(identical(as.integer(rownames(telco_test)), as.integer(test_id)))

train_fac  <- as_churn_factor(telco_train)
test_fac   <- as_churn_factor(telco_test)
p_cov      <- ncol(train_fac) - 1L
y_train    <- as.integer(telco_train$Churn)
y_test     <- as.integer(telco_test$Churn)
prev_train <- mean(y_train) # ~0.262 training prevalence threshold

cat(sprintf("Training rows: %d | Test rows: %d | Predictors: %d | Train prevalence: %.4f\n",
            nrow(train_fac), nrow(test_fac), p_cov, prev_train))

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS (Vectorized OOB metrics + Test evaluation)
# ------------------------------------------------------------------------------

fast_auc <- function(y, p) {
  ok <- !is.na(p)
  y_ok <- y[ok]
  p_ok <- p[ok]
  n1 <- sum(y_ok == 1L)
  n0 <- sum(y_ok == 0L)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  r <- rank(p_ok, ties.method = "average")
  (sum(r[y_ok == 1L]) - as.numeric(n1) * (n1 + 1) / 2) / (as.numeric(n1) * as.numeric(n0))
}

compute_oob_metrics_along_M <- function(oob_mat, y, prev_thresh, eps = 1e-15) {
  # oob_mat: n_train x ntree matrix of cumulative OOB vote shares
  p_clip <- pmin(pmax(oob_mat, eps), 1 - eps)
  pos_idx <- (y == 1L)
  neg_idx <- (y == 0L)
  
  oob_brier    <- colMeans((oob_mat - y)^2, na.rm = TRUE)
  oob_logloss  <- colMeans(-(y * log(p_clip) + (1 - y) * log(1 - p_clip)), na.rm = TRUE)
  oob_auc      <- apply(oob_mat, 2L, function(p) fast_auc(y, p))
  oob_acc_05   <- colMeans((oob_mat >= 0.50) == pos_idx, na.rm = TRUE)
  oob_sens_pr  <- colMeans(oob_mat[pos_idx, , drop = FALSE] >= prev_thresh, na.rm = TRUE)
  oob_spec_pr  <- colMeans(oob_mat[neg_idx, , drop = FALSE] <  prev_thresh, na.rm = TRUE)
  oob_bal_pr   <- 0.5 * (oob_sens_pr + oob_spec_pr)
  
  data.frame(
    M              = seq_len(ncol(oob_mat)),
    oob_brier      = oob_brier,
    oob_logloss    = oob_logloss,
    oob_auc        = oob_auc,
    oob_acc_05     = oob_acc_05,
    oob_sens_prev  = oob_sens_pr,
    oob_spec_prev  = oob_spec_pr,
    oob_balacc_pr  = oob_bal_pr
  )
}

select_multimetric_winners <- function(path_df, min_M = 50L) {
  # Filter to M >= min_M so all observations have stable OOB estimates
  d <- path_df[path_df$M >= min_M, , drop = FALSE]
  rownames(d) <- NULL
  
  # Compute multi-metric average rank across the 5 core criteria (lower rank = better)
  d$avg_rank <- (
    rank(d$oob_brier,      ties.method = "average") +
      rank(d$oob_logloss,    ties.method = "average") +
      rank(-d$oob_auc,       ties.method = "average") +
      rank(-d$oob_acc_05,    ties.method = "average") +
      rank(-d$oob_sens_prev, ties.method = "average")
  ) / 5
  
  pick_first <- function(ord_idx, label) {
    row <- d[ord_idx[1L], , drop = FALSE]
    row$criterion <- label
    row
  }
  
  winners <- rbind(
    pick_first(order(d$oob_brier,      d$M, -d$nodesize, d$samp_frac), "1_OOB_Brier_min"),
    pick_first(order(d$oob_logloss,    d$M, -d$nodesize, d$samp_frac), "2_OOB_LogLoss_min"),
    pick_first(order(-d$oob_auc,       d$M, -d$nodesize, d$samp_frac), "3_OOB_AUC_max"),
    pick_first(order(-d$oob_acc_05,    d$M, -d$nodesize, d$samp_frac), "4_OOB_Acc05_max"),
    pick_first(order(-d$oob_sens_prev, -d$oob_spec_prev, d$M),         "5_OOB_SensPrev_max"),
    pick_first(order(-d$oob_balacc_pr, d$M, -d$nodesize, d$samp_frac), "6_OOB_BalAccPrev_max"),
    pick_first(order(d$avg_rank,       d$oob_brier, d$M),              "7_Compromise_AvgRank")
  )
  rownames(winners) <- NULL
  winners
}

evaluate_winners_on_test <- function(winners_df, fit_store, test_df, y_test, prev_thresh, eps = 1e-15) {
  res_list <- vector("list", nrow(winners_df))
  for (i in seq_len(nrow(winners_df))) {
    w <- winners_df[i, ]
    fit_obj <- fit_store[[w$model_key]]
    p_test  <- vote_share_at(fit_obj, test_df, w$M)
    p_clip  <- pmin(pmax(p_test, eps), 1 - eps)
    
    pred_05   <- (p_test >= 0.50)
    pred_prev <- (p_test >= prev_thresh)
    y_pos     <- (y_test == 1L)
    y_neg     <- (y_test == 0L)
    
    tp_05 <- sum(pred_05 & y_pos)
    fp_05 <- sum(pred_05 & y_neg)
    
    res_list[[i]] <- data.frame(
      criterion      = w$criterion,
      mtry           = w$mtry,
      nodesize       = w$nodesize,
      samp_frac      = w$samp_frac,
      M              = w$M,
      test_brier     = mean((p_test - y_test)^2),
      test_logloss   = mean(-(y_test * log(p_clip) + (1 - y_test) * log(1 - p_clip))),
      test_auc       = fast_auc(y_test, p_test),
      test_acc_05    = mean(pred_05 == y_pos),
      test_sens_05   = mean(pred_05[y_pos]),
      test_prec_05   = if ((tp_05 + fp_05) > 0) tp_05 / (tp_05 + fp_05) else NA_real_,
      test_sens_prev = mean(pred_prev[y_pos]),
      test_spec_prev = mean(!pred_prev[y_neg]),
      test_balacc_pr = 0.5 * (mean(pred_prev[y_pos]) + mean(!pred_prev[y_neg])),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, res_list)
}

# ==============================================================================
# PART 1: BAGGING MULTI-METRIC TUNING (12 configs x 500 trees)
# ==============================================================================
message("\n=== RUNNING PART 1: BAGGING MULTI-METRIC TUNING ===")

bag_ntree          <- 500L
bag_nodesize_grid  <- c(1L, 10L, 15L, 50L)
bag_samp_frac_grid <- c(0.6, 0.8, 1.0)
bag_grid           <- expand.grid(nodesize = bag_nodesize_grid, samp_frac = bag_samp_frac_grid)

bag_mm_list   <- vector("list", nrow(bag_grid))
bag_fit_store <- list()

for (i in seq_len(nrow(bag_grid))) {
  n_size <- bag_grid$nodesize[i]
  s_frac <- bag_grid$samp_frac[i]
  s_size <- floor(s_frac * nrow(train_fac))
  
  seed_val <- if (s_frac == 1.0) {
    31000L + n_size
  } else {
    31000L + n_size + as.integer(round(100 * s_frac))
  }
  set.seed(seed_val)
  message("Bagging [", i, "/", nrow(bag_grid), "] nodesize = ", n_size, ", samp_frac = ", s_frac)
  
  fit_bag <- randomForest(
    Churn ~ .,
    data = train_fac,
    ntree = bag_ntree,
    mtry = p_cov,
    nodesize = n_size,
    replace = TRUE,
    sampsize = s_size,
    keep.inbag = TRUE,
    importance = FALSE
  )
  
  oob_mat <- oob_vote_share_path(fit_bag, train_fac)
  key <- paste(n_size, s_frac, sep = "_")
  bag_fit_store[[key]] <- fit_bag
  
  m_df <- compute_oob_metrics_along_M(oob_mat, y_train, prev_train)
  m_df$mtry      <- p_cov
  m_df$nodesize  <- n_size
  m_df$samp_frac <- s_frac
  m_df$model_key <- key
  bag_mm_list[[i]] <- m_df
}

bag_mm_path <- do.call(rbind, bag_mm_list)

# STEP B (Bagging): Winners per metric & Compromise
bag_oob_winners <- select_multimetric_winners(bag_mm_path, min_M = 50L)

# STEP C (Bagging): Held-out Test Set Evaluation
bag_test_eval <- evaluate_winners_on_test(bag_oob_winners, bag_fit_store, test_fac, y_test, prev_train)

# Free Bagging models from RAM before fitting RF
rm(bag_fit_store, bag_mm_list)
gc(verbose = FALSE)

# ==============================================================================
# PART 2: RANDOM FOREST MULTI-METRIC TUNING (18 configs x 500 trees)
# ==============================================================================
message("\n=== RUNNING PART 2: RANDOM FOREST MULTI-METRIC TUNING ===")

rf_ntree         <- 500L
rf_mtry_grid     <- c(2L, 4L, 6L, 8L, 12L, 15L)
rf_nodesize_grid <- c(1L, 10L, 50L)
rf_grid          <- expand.grid(mtry = rf_mtry_grid, nodesize = rf_nodesize_grid)

rf_mm_list   <- vector("list", nrow(rf_grid))
rf_fit_store <- list()

for (i in seq_len(nrow(rf_grid))) {
  mtry_val   <- rf_grid$mtry[i]
  n_size_val <- rf_grid$nodesize[i]
  
  set.seed(32000L + mtry_val + n_size_val)
  message("Random Forest [", i, "/", nrow(rf_grid), "] mtry = ", mtry_val, ", nodesize = ", n_size_val)
  
  fit_m <- randomForest(
    Churn ~ .,
    data = train_fac,
    ntree = rf_ntree,
    mtry = mtry_val,
    nodesize = n_size_val,
    replace = TRUE,
    sampsize = nrow(train_fac),
    keep.inbag = TRUE,
    importance = TRUE
  )
  
  oob_mat <- oob_vote_share_path(fit_m, train_fac)
  key <- paste(mtry_val, n_size_val, sep = "_")
  rf_fit_store[[key]] <- fit_m
  
  m_df <- compute_oob_metrics_along_M(oob_mat, y_train, prev_train)
  m_df$mtry      <- mtry_val
  m_df$nodesize  <- n_size_val
  m_df$samp_frac <- 1.0
  m_df$model_key <- key
  rf_mm_list[[i]] <- m_df
}

rf_mm_path <- do.call(rbind, rf_mm_list)

# STEP B (RF): Winners per metric & Compromise
rf_oob_winners <- select_multimetric_winners(rf_mm_path, min_M = 50L)

# STEP C (RF): Held-out Test Set Evaluation
rf_test_eval <- evaluate_winners_on_test(rf_oob_winners, rf_fit_store, test_fac, y_test, prev_train)

rm(rf_fit_store, rf_mm_list)
gc(verbose = FALSE)

# ==============================================================================
# PRINT SUMMARY TABLES & SAVE CSVs TO output/
# ==============================================================================
out_dir <- here::here("output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(bag_oob_winners, file.path(out_dir, "multimetric_bagging_oob_winners.csv"), row.names = FALSE)
write.csv(bag_test_eval,   file.path(out_dir, "multimetric_bagging_test_eval.csv"),   row.names = FALSE)
write.csv(rf_oob_winners,  file.path(out_dir, "multimetric_rf_oob_winners.csv"),      row.names = FALSE)
write.csv(rf_test_eval,    file.path(out_dir, "multimetric_rf_test_eval.csv"),        row.names = FALSE)

cols_oob <- c("criterion", "mtry", "nodesize", "samp_frac", "M",
              "oob_brier", "oob_logloss", "oob_auc", "oob_acc_05", "oob_sens_prev", "oob_spec_prev")

cat("\n==============================================================================\n")
cat("BAGGING — STEP B: OPTIMAL CONFIGURATIONS BY OOB METRIC (M >= 50)\n")
cat("==============================================================================\n")
print(bag_oob_winners[, cols_oob], digits = 4, row.names = FALSE)

cat("\n==============================================================================\n")
cat("BAGGING — STEP C: OUT-OF-SAMPLE TEST SET PERFORMANCE OF SELECTED MODELS\n")
cat("==============================================================================\n")
print(bag_test_eval, digits = 4, row.names = FALSE)

cat("\n==============================================================================\n")
cat("RANDOM FOREST — STEP B: OPTIMAL CONFIGURATIONS BY OOB METRIC (M >= 50)\n")
cat("==============================================================================\n")
print(rf_oob_winners[, cols_oob], digits = 4, row.names = FALSE)

cat("\n==============================================================================\n")
cat("RANDOM FOREST — STEP C: OUT-OF-SAMPLE TEST SET PERFORMANCE OF SELECTED MODELS\n")
cat("==============================================================================\n")
print(rf_test_eval, digits = 4, row.names = FALSE)

message("\nDone! All 4 summary CSV tables saved in: ", out_dir)