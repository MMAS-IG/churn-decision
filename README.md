# churn-decision

Predicting IBM Telco customer churn with a linear probability model and a logistic regression, in R.

Data: [`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`](data/WA_Fn-UseC_-Telco-Customer-Churn.csv), from
[Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Document

Open [`from-linear-to-logistic-churn.qmd`](from-linear-to-logistic-churn.qmd) in RStudio or Positron and run the chunks, or render it:

```bash
quarto render from-linear-to-logistic-churn.qmd
```

The narrative covers the data, why OLS on a 0/1 churn flag is the wrong mean function, logistic regression (odds ratios, fitted probabilities, average partial effects), and standard checks (deviance, hold-out confusion tables, Brier score, ROC, calibration, binned residuals).

## Scripts

Same analysis, as scripts to source from the project root (or after opening `churn-decision.Rproj`):

```r
source("R/00_run_all.R")
```

| File | Role |
| --- | --- |
| `R/01_prepare_data.R` | Read, recode nested service labels, train/test split |
| `R/02_explore.R` | Univariate churn views |
| `R/03_linear_probability.R` | OLS linear probability model |
| `R/04_logistic_regression.R` | Binomial GLM, odds ratios, profiles |
| `R/05_assessment.R` | Test-set assessment |

Requires R packages: `here`, `dplyr`, `ggplot2`, `tidyr`, `scales`, `broom`, `car`, `lmtest`, `pROC`.
