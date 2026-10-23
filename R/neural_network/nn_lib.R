# Helpers for a single-hidden-layer classifier fit with nnet.
# No data are read and no model is fit until a caller asks.

brier_score <- function(y, p) {
  mean((as.numeric(p) - as.numeric(y))^2)
}

# Standard error of the mean Brier score on one sample. Used to see
# whether two validation scores differ by more than noise.
brier_se <- function(y, p) {
  err <- (as.numeric(p) - as.numeric(y))^2
  stats::sd(err) / sqrt(length(err))
}

# Mean Bernoulli log loss. Probabilities are clipped only so that a
# score sitting on 0 or 1 (or, for the linear probability model, outside
# [0, 1]) does not make the average infinite. The clip is not a model.
log_loss <- function(y, p, eps = 1e-6) {
  p <- pmin(pmax(as.numeric(p), eps), 1 - eps)
  y <- as.numeric(y)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

# Training mean and sd of the numeric predictors. Factors are left for
# model.matrix inside nnet(), which applies the same treatment contrasts
# as lm() and glm() because the reference levels were set in preparation.
predictor_center_scale <- function(df) {
  vars <- setdiff(names(df)[vapply(df, is.numeric, logical(1))], "Churn")
  center <- vapply(df[vars], mean, numeric(1))
  scale <- vapply(df[vars], stats::sd, numeric(1))
  scale[scale == 0] <- 1
  list(vars = vars, center = center, scale = scale)
}

apply_center_scale <- function(df, spec) {
  out <- df
  for (v in spec$vars) {
    out[[v]] <- (out[[v]] - spec$center[[v]]) / spec$scale[[v]]
  }
  out
}

as_churn_factor <- function(df) {
  out <- df
  out$Churn <- factor(out$Churn, levels = c(0, 1), labels = c("No", "Yes"))
  out
}

# Second factor level is the event: nnet.formula codes it as 1 and
# turns on entropy fitting, so predict(..., type = "raw") is P(Churn = Yes).
nnet_probability <- function(fit, newdata) {
  as.numeric(stats::predict(fit, newdata = newdata, type = "raw"))
}

# One BFGS start. size = 0 with skip = TRUE is a logistic regression in
# the same inputs (no hidden layer), used only as a baseline inside the
# tuning grid. Decay is applied by nnet to every weight, biases included.
fit_nnet_once <- function(data, size, decay, seed, maxit = 400, rang = 0.5) {
  set.seed(seed)
  nnet::nnet(
    Churn ~ .,
    data = data,
    size = size,
    decay = decay,
    maxit = maxit,
    rang = rang,
    skip = isTRUE(size == 0),
    trace = FALSE,
    MaxNWts = 10000
  )
}

# Choose size and decay on a validation slice. Random starts are ranked
# by the training criterion (fit$value, which includes the decay penalty).
# The validation slice is used only after that choice, and only to pick
# the architecture. `train_raw` / `val_raw` stay on the original scale;
# centering uses the training slice alone.
select_network <- function(
    train_raw,
    val_raw,
    sizes,
    decays,
    n_starts,
    maxit = 400
) {
  spec <- predictor_center_scale(train_raw)
  train <- apply_center_scale(train_raw, spec)
  val <- apply_center_scale(val_raw, spec)
  y_tr <- as.integer(train$Churn == "Yes")
  y_val <- as.integer(val$Churn == "Yes")

  rows <- list()
  k <- 0L
  for (size in sizes) {
    for (decay in decays) {
      k <- k + 1L
      message("Tuning size = ", size, ", decay = ", decay, " (", n_starts, " starts)")
      starts <- vector("list", n_starts)
      for (s in seq_len(n_starts)) {
        seed <- 100000L + as.integer(size) * 1000L +
          as.integer(round(decay * 100)) * 10L + as.integer(s)
        starts[[s]] <- fit_nnet_once(
          train, size = size, decay = decay, seed = seed, maxit = maxit
        )
      }
      criteria <- vapply(starts, function(f) f$value, numeric(1))
      best <- which.min(criteria)
      fit <- starts[[best]]
      p_tr <- nnet_probability(fit, train)
      p_val <- nnet_probability(fit, val)
      rows[[k]] <- data.frame(
        size = size,
        decay = decay,
        n_inputs = fit$n[1],
        n_hidden = fit$n[2],
        n_weights = length(fit$wts),
        converged = fit$convergence == 0,
        start_kept = best,
        train_brier = brier_score(y_tr, p_tr),
        val_brier = brier_score(y_val, p_val),
        val_brier_se = brier_se(y_val, p_val),
        val_log_loss = log_loss(y_val, p_val),
        stringsAsFactors = FALSE
      )
    }
  }

  table <- do.call(rbind, rows)
  rownames(table) <- NULL
  # Prefer a lower validation Brier, then a smaller network, then more decay.
  ord <- order(table$val_brier, table$size, -table$decay)
  table <- table[ord, ]
  rownames(table) <- NULL
  list(table = table, best = table[1, ])
}

# Refit the chosen architecture on the full training set. Centering is
# recomputed on that full training set and must be reused for the test set.
refit_network <- function(train_raw, size, decay, n_starts, maxit = 400) {
  spec <- predictor_center_scale(train_raw)
  train <- apply_center_scale(train_raw, spec)
  fits <- lapply(seq_len(n_starts), function(s) {
    fit_nnet_once(
      train,
      size = size,
      decay = decay,
      seed = 20000L + as.integer(s),
      maxit = maxit
    )
  })
  criteria <- vapply(fits, function(f) f$value, numeric(1))
  best <- which.min(criteria)
  list(
    fit = fits[[best]],
    spec = spec,
    train_scaled = train,
    start_kept = best,
    criteria = criteria
  )
}
