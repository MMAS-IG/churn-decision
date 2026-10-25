# Helpers for a classification tree, bagging, a random forest, and gradient boosting.
# No data are read and no model is fit until a caller asks.

brier_score <- function(y, p) {
  mean((as.numeric(p) - as.numeric(y))^2)
}

brier_se <- function(y, p) {
  err <- (as.numeric(p) - as.numeric(y))^2
  stats::sd(err) / sqrt(length(err))
}

log_loss <- function(y, p, eps = 1e-6) {
  p <- pmin(pmax(as.numeric(p), eps), 1 - eps)
  y <- as.numeric(y)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

# randomForest and rpart classification use a two-level factor.
# The second level is the churn event.
as_churn_factor <- function(df) {
  out <- df
  out$Churn <- factor(out$Churn, levels = c(0, 1), labels = c("No", "Yes"))
  out
}

# gbm with distribution = "bernoulli" requires a numeric 0/1 response.
as_churn_numeric <- function(df) {
  out <- df
  if (is.factor(out$Churn)) {
    out$Churn <- as.integer(out$Churn == "Yes")
  } else {
    out$Churn <- as.integer(out$Churn)
  }
  out
}

# Leaf probability of the second class from an rpart classification tree.
rpart_probability <- function(fit, newdata) {
  as.numeric(predict(fit, newdata = newdata, type = "prob")[, "Yes"])
}

# Vote share: fraction of trees whose predicted class is Yes.
# predict(..., type = "prob") returns that fraction for a randomForest fit.
rf_probability <- function(fit, newdata) {
  as.numeric(predict(fit, newdata = newdata, type = "prob")[, "Yes"])
}

# Cumulative vote share after 1, 2, ..., M trees.
# Column m is the fraction of the first m trees that vote Yes.
vote_share_path <- function(fit, newdata) {
  votes <- predict(fit, newdata = newdata, predict.all = TRUE)$individual
  yes <- votes == "Yes"
  counts <- t(apply(yes, 1L, cumsum))
  m_seq <- seq_len(ncol(counts))
  sweep(counts, 2L, m_seq, "/")
}

# Vote share of the first n_trees trees.
vote_share_at <- function(fit, newdata, n_trees) {
  as.numeric(vote_share_path(fit, newdata)[, n_trees])
}

# Out-of-bag vote share of the first 1, ..., ntree trees.
# fit$inbag[i, b] is the number of times training row i was drawn
# into the bootstrap sample of tree b. A row is out of bag when that
# count is 0. Column m is the Yes vote share among those of the first
# m trees that left the row out. The entry is NA when none of them did.
oob_vote_share_path <- function(fit, data) {
  if (is.null(fit$inbag)) {
    stop("oob_vote_share_path requires keep.inbag = TRUE")
  }
  votes <- predict(fit, newdata = data, predict.all = TRUE)$individual
  oob <- fit$inbag == 0
  yes_oob <- (votes == "Yes") & oob
  c_yes <- t(apply(yes_oob, 1L, cumsum))
  c_n <- t(apply(oob, 1L, cumsum))
  p <- matrix(NA_real_, nrow = nrow(c_yes), ncol = ncol(c_yes))
  ok <- c_n > 0
  p[ok] <- c_yes[ok] / c_n[ok]
  p
}

# Brier score of an OOB path. Column m averages only rows that are
# out of bag for at least one of the first m trees.
brier_along_oob <- function(p_mat, y) {
  y <- as.numeric(y)
  vapply(seq_len(ncol(p_mat)), function(m) {
    ok <- !is.na(p_mat[, m])
    mean((p_mat[ok, m] - y[ok])^2)
  }, numeric(1))
}

brier_se_defined <- function(y, p) {
  ok <- !is.na(p)
  brier_se(y[ok], p[ok])
}

brier_along_trees <- function(p_mat, y) {
  y <- as.numeric(y)
  colMeans((p_mat - y)^2)
}

# sigma(F_M) from a bernoulli gbm fit, for each requested number of trees.
gbm_probability <- function(fit, newdata, n_trees) {
  as.numeric(predict(fit, newdata = newdata, n.trees = n_trees, type = "response"))
}

gbm_probability_matrix <- function(fit, newdata, n_trees) {
  predict(fit, newdata = newdata, n.trees = n_trees, type = "response")
}
