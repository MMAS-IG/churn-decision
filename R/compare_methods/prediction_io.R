# Save and load test-set predictions for one shared churn split.
# Every classification method writes one .rds file. The comparison
# program reads those files and does not refit anything.

churn_split_info <- function(telco, telco_train) {
  if (is.factor(telco_train$Churn)) {
    stop("telco_train$Churn must be coded 0/1, not as a factor.")
  }
  list(
    seed = 2026L,
    test_frac = 0.30,
    n_rows = nrow(telco),
    n_train = nrow(telco_train),
    train_prevalence = unname(mean(telco_train$Churn))
  )
}

save_test_predictions <- function(
    method_id,
    method_label,
    y,
    p,
    test_row,
    split,
    detail = "",
    dir = here::here("output", "test_predictions")
) {
  method_id <- as.character(method_id)
  if (!grepl("^[A-Za-z0-9_]+$", method_id)) {
    stop(
      "method_id must contain only letters, digits, and underscores ",
      "(it is used as the file name)."
    )
  }

  y <- as.integer(unname(y))
  p <- as.numeric(unname(p))
  test_row <- as.integer(unname(test_row))

  if (length(y) != length(p) || length(y) != length(test_row)) {
    stop("y, p, and test_row must have the same length.")
  }
  if (length(y) == 0L) stop("y is empty.")
  if (anyNA(y) || anyNA(p) || anyNA(test_row)) {
    stop("y, p, and test_row must not contain NA.")
  }
  if (any(!is.finite(p))) stop("p must be finite.")
  if (!all(y %in% c(0L, 1L))) stop("y must be coded 0/1.")
  if (anyDuplicated(test_row)) stop("test_row contains duplicate rows.")

  required <- c("seed", "test_frac", "n_rows", "n_train", "train_prevalence")
  if (!is.list(split) || !all(required %in% names(split))) {
    stop("split is missing one of: ", paste(required, collapse = ", "), ".")
  }

  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  obj <- list(
    schema = 1L,
    method_id = method_id,
    method_label = as.character(method_label),
    detail = as.character(detail),
    y = y,
    p = p,
    test_row = test_row,
    split = list(
      seed = as.integer(split$seed),
      test_frac = as.numeric(split$test_frac),
      n_rows = as.integer(split$n_rows),
      n_train = as.integer(split$n_train),
      train_prevalence = as.numeric(split$train_prevalence)
    )
  )
  path <- file.path(dir, paste0(method_id, ".rds"))
  saveRDS(obj, path)
  message("Saved test predictions: ", path)
  invisible(obj)
}

load_test_predictions <- function(dir = here::here("output", "test_predictions")) {
  files <- sort(list.files(dir, pattern = "\\.rds$", full.names = TRUE))
  if (length(files) == 0L) {
    stop(
      "No .rds files in ", dir, ".\n",
      "Export scores first, for example:\n",
      "  source(\"R/compare_methods/00_export_linear_logistic.R\")\n",
      "  source(\"R/neural_network/00_run_all.R\")"
    )
  }

  objs <- lapply(files, readRDS)
  for (i in seq_along(objs)) {
    obj <- objs[[i]]
    stem <- sub("\\.rds$", "", basename(files[[i]]))
    if (!is.list(obj) || is.null(obj$method_id) || is.null(obj$y) ||
        is.null(obj$p) || is.null(obj$test_row) || is.null(obj$split)) {
      stop(files[[i]], " is not a test-prediction file written by save_test_predictions().")
    }
    if (!identical(stem, obj$method_id)) {
      stop("File name ", stem, " does not match method_id ", obj$method_id, ".")
    }
    objs[[i]] <- obj
  }

  # Canonical order is the sorted test-row id, so a method may save rows
  # in a different order and still be compared.
  ref <- objs[[1]]
  ref_rows <- sort(ref$test_row)

  aligned <- lapply(objs, function(obj) {
    if (!setequal(obj$test_row, ref_rows)) {
      stop(
        obj$method_id,
        " was not scored on the same test rows as ",
        ref$method_id, "."
      )
    }
    if (!identical(obj$split$seed, ref$split$seed) ||
        !identical(obj$split$n_rows, ref$split$n_rows) ||
        !identical(obj$split$n_train, ref$split$n_train)) {
      stop(obj$method_id, " does not carry the same split description as ", ref$method_id, ".")
    }
    if (!isTRUE(all.equal(obj$split$train_prevalence, ref$split$train_prevalence))) {
      stop(obj$method_id, " has a different training prevalence from ", ref$method_id, ".")
    }
    ord <- match(ref_rows, obj$test_row)
    y <- obj$y[ord]
    if (!identical(y, ref$y[match(ref_rows, ref$test_row)])) {
      stop(
        obj$method_id,
        " has different observed outcomes on the shared test rows."
      )
    }
    obj$y <- y
    obj$p <- obj$p[ord]
    obj$test_row <- ref_rows
    obj
  })

  aligned
}
