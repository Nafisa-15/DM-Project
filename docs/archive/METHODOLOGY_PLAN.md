# Methodology (ORIGINAL PLAN, archived 2026-10-01)

> Archived copy of the plan written before any code ran, kept as the pre-registered design. The executed methodology, with results for every step, is in [../METHODOLOGY.md](../METHODOLOGY.md). Links below are relative to docs/ and may not resolve from this folder.

Step-by-step plan. Each step lists what to do and a **done-when** condition. Data facts are in [DATA_AUDIT.md](DATA_AUDIT.md).

> **Status: this is the PLAN, written before any code ran.** It is kept unchanged below as the pre-registered design; it shows which rules were fixed before any results existed. Where the executed project differs, the table below applies; the reasons are in [DECISIONS_LOG.md](DECISIONS_LOG.md). The results as run are in [MODEL_JUSTIFICATION.md](MODEL_JUSTIFICATION.md); the progress log is `PROGRESS.md`.

### Plan vs as-run (updated 2026-10-01)
| Plan (below) | As run | Why |
|---|---|---|
| Five feature sets | FS-Primary and FS-PM25 only | Lean scope ([PROJECT_RULES.md](PROJECT_RULES.md)). FS-PM25 is the direct H1 test. |
| Steps 2B, 8, 13; Holm correction; leave-one-city-out; Poisson GLM; VIF | Skipped | Lean scope. Each is listed as a limitation. |
| Notebooks (`notebooks/*.ipynb`) | Scripts in `src/` with a FAST/FULL mode | Reproducibility (D1) |
| Evaluation rules in `EVALUATION.md` | That file was never created. The tie rule (RMSE gap < 0.05) and spike definition are in [PROJECT_RULES.md](PROJECT_RULES.md). | — |
| Ridge alpha grid {0, 0.01, 0.1, 1, 10} | As planned, widened once if the best is 10. Logistic C is searched on a log grid {1e-3 … 1e3}. Numeric inputs standardised. | D11 |
| Tree grid with `ccp_alpha` ∈ {0, 1e-4, 1e-3, 1e-2} | Same depth and leaf grid; `ccp_alpha` candidates taken from the training data's pruning path (CART) | D11: the planned values ignored the data's scale |
| SVM grid C {0.1–100}, epsilon {0.1, 0.5, 1}, gamma {scale, 0.01, 0.1} | Two-phase random search: Phase A 40×5-fold on 10k rows; Phase B 20×5-fold on 20k rows. C 0.01–1000, gamma 1e-4–1 or scale, epsilon 0.05–3.5. | D7, D8: wider log ranges (Hsu et al.); epsilon tied to the noise level (Cherkassky & Ma) |
| SVC with Platt scaling | No `probability=True`; `decision_function` scores are used for PR-AUC and ROC-AUC | Laptop compute limit. The AUC metrics only need rankings, not probabilities. |
| SVM learning curve at 5k/10k/20k | 5k/10k/20k/40k; final models on 20k because the curve is flat | D7, D10 |
| MLP in PyTorch/Keras with dropout | sklearn `MLPRegressor`/`MLPClassifier`: L2 `alpha` instead of dropout, early stopping, 5 seeds | Allowed packages ([PROJECT_RULES.md](PROJECT_RULES.md)): no torch. sklearn has no dropout; L2 is the regulariser it offers. |
| Original compute rules: random search (n_iter 10–30) for the tree; SVM n_iter=10, cv=3 on 5k–8k rows | Overridden (2026-10-01). SVM: at the user's request for thorough training, the two-phase search above. Tree: a full grid, proposed by the assistant in the fix prompt the user ran (fits are cheap, no sampling luck). | For classification, small searches risked under-tuning: with 10 draws, about a 1-in-4 chance (SVC) or 1-in-3 (tree) of ending well below the best PR-AUC. For regression, the landscape is flat and the extra search changed nothing. [PROJECT_RULES.md](PROJECT_RULES.md) now states the rule this implies (Bergstra & Bengio 2012): n_iter ≥ 10 for regression, ≥ 30 for classification, full grids for cheap models, 5-fold CV throughout. |
| Step 10: F-test for the city × PM2.5 interaction | Wald χ² test with HC3 robust SEs: χ²(7) = 15.7, p = 0.028 | Count variance grows with the mean (variance/mean = 1.7), so robust SEs are used; with robust covariance statsmodels gives a Wald χ² test, not an F test (D17) |
| Step 10: per-city models | Each tuned by 5-fold CV on its own city's training rows | Reusing the pooled tree's settings overfit small cities and biased the comparison (D16) |
| Permutation importance: 30 repeats | 5 repeats; 3,000-row test subsample for SVM and MLP | Laptop compute limit ([PROJECT_RULES.md](PROJECT_RULES.md)) |

## Changes from the Original Proposal

The central idea is unchanged: predict respiratory hospital admissions from air quality data and compare models across cities.

| Original proposal | Final project | Reason |
|---|---|---|
| Linear Regression, Random Forest, Prophet | Linear Regression, Decision Tree, SVM, MLP (Random Forest optional) | Instructor's model list. Prophet needs a real time series, which the data lacks. |
| Datasets 2 and 3 (BreathWatch, Global Environmental Intelligence) | Dropped. An optional real air quality dataset can be used for a context check (Step 2B) | No shared key or admissions outcome, so merging was unjustified. |
| Filter to Delhi (high pollution) and London (low pollution) | All 8 cities used. Delhi and London remain a focal pair. | In the main dataset both have the same pollution level. |
| Lag features, `day_of_week`, `month` | Removed | Dates are meaningless (see DATA_AUDIT.md). |
| Handle missing values | Replaced by a validation step | The dataset has no missing values. We assert that instead of assuming it. |
| Random Forest feature importance | Permutation importance on the test set | Model-agnostic and less biased than impurity importance. |
| MAE, RMSE, R² | Same, plus Poisson deviance and classification metrics | The outcome is a count, and early warning is a classification problem. |
| Conclusions about pollution and health in Delhi vs London | Conclusions about model behaviour on this benchmark | The data is synthetic. |

---

## Step 0. Project setup
- Create the repository layout ([REPRODUCIBILITY.md](REPRODUCIBILITY.md)) and a pinned `requirements.txt`.
- Set one global seed (`SEED = 42`) in `src/config.py`.
- Keep the raw CSV in `data/raw/` and never modify it.

**Done when:** a fresh clone can run Step 1 with one command.

## Step 1. Load and validate
`src/data.py` provides `load_raw()` with assertions:
- 12 expected columns and dtypes, 88,489 rows
- no missing values, no duplicate rows
- `city` has exactly 8 values, `population_density` exactly 3
- pollutant columns ≥ 0, `aqi` in [0, 500], `humidity` in [0, 100], admissions non-negative integers
- admissions ≤ hospital capacity

**Done when:** all assertions pass and a report is saved to `results/tables/01_validation.csv`.

## Step 2. Data audit and exploratory analysis
Reproduce every number in [DATA_AUDIT.md](DATA_AUDIT.md) in `notebooks/01_audit.ipynb`. Required outputs:
- distributions of all numeric variables and of the target
- correlation heatmap and PM2.5 vs admissions plot
- mean admissions by PM2.5 decile with confidence bands
- per-city summaries and sample sizes
- date-gap and lag-autocorrelation analysis showing there is no time series
- variance inflation factors (VIF)

**Done when:** each audit finding has a figure or table behind it.

## Step 2B. Real-data reference check (optional, context only)
**Status: optional.** Skipping it does not affect any other step, and it can be done at any time after Step 2.

**Purpose:** the original proposal claimed that Delhi is a high-pollution city and London a low-pollution one. That claim cannot be tested in the main dataset, where both cities have the same pollution levels. A real dataset shows what real air quality data looks like, and documents why the benchmark differs. If skipped, the report states only that the claim cannot be tested with this file.

**Dataset choice.** Use **one** real-world air quality dataset. Pick it only if it meets all of these:
- real calendar dates covering at least one year
- includes Delhi (and London or another low-pollution city if available)
- includes PM2.5 and AQI columns
- documented source (monitoring stations or a government/open-data API)
- source URL, license and SHA-256 recorded in `data/README_DATA.md`

**What to compute** (one table, same statistics for both datasets):

| Check | Why |
|---|---|
| Mean and median PM2.5 and AQI per city | Does Delhi differ from a low-pollution city? |
| Correlation between AQI and PM2.5 | Real AQI is derived from pollutants, so a strong link is expected. The main dataset shows 0.003. |
| Lag-1 autocorrelation of PM2.5 | Real air quality is persistent day to day. The main dataset shows about 0. |
| Monthly (seasonal) pattern of PM2.5 | Real cities show seasonality. The main dataset does not. |
| Share of days with PM2.5 above the main dataset's maximum (109.9 µg/m³) | Shows whether the benchmark covers real pollution ranges |

**What this step does not do:**
- The datasets are not merged, and no admissions labels are transferred.
- The trained models are not applied to real Delhi or London data. The benchmark's PM2.5 range (0–110) may not cover real Delhi values, and there is no ground truth to check predictions against.

**Done when:** the comparison table in [RESULTS_AND_CHECKLIST.md](RESULTS_AND_CHECKLIST.md) is filled and one paragraph of interpretation is written. If the chosen file fails the criteria above, choose another. Nothing else in the project depends on this step.

## Step 3. Target and spike label
- **Regression target:** `hospital_admissions`.
- **Spike label:** `spike = 1` if admissions ≥ the 90th percentile of the **training set** (13 admissions on this data, about 11.7% of days). The threshold is computed on training data only and then applied unchanged to the test set.

**Done when:** the threshold is stored in `results/tables/03_spike_threshold.json`.

## Step 4. Feature sets
Fixed now, before any results.

| ID | Features | Use |
|---|---|---|
| **FS-Primary** | `aqi, pm2_5, pm10, no2, o3, temperature, humidity` + one-hot `city` | Main comparison (the proposal's features) |
| FS-PM25 | `pm2_5` only | Ablation, tests H1 |
| FS-NoAQI | FS-Primary without `aqi` | Ablation, removes the redundant index |
| FS-Pollutants | `aqi, pm2_5, pm10, no2, o3` | Ablation, weather removed |
| FS-Extended | FS-Primary + `hospital_capacity` + one-hot `population_density` | Sensitivity analysis |

`hospital_capacity` is excluded from the primary set because it describes hospital supply, not environmental exposure. `population_density` is a random per-row attribute here. `date` is never used. Per-city models drop the `city` columns.

## Step 5. Train/test split
- Split once, **80% train / 20% test**, stratified by `city × spike`. Expected sizes: about 70,791 train and 17,698 test rows.
- Save row indices to `data/processed/split_indices.json`.
- **Rules for every later step:**
  1. The test set is used **once**, in Step 9.
  2. Scalers and encoders are fitted inside a scikit-learn `Pipeline` on training folds only.
  3. The spike threshold and any decision threshold come from training data only.
  4. Random row splitting is valid here because the audit shows rows are independent. It would be invalid for real time series.

**Done when:** a unit test confirms no row is in both splits and city/class proportions match.

## Step 6. Baselines

| Baseline | Purpose |
|---|---|
| Mean predictor | Zero-skill reference (R² ≈ 0) |
| City-mean predictor | Does city identity carry information? |
| Simple linear regression on `pm2_5` only | The strongest simple model. Anything complex must beat it. |
| Prevalence classifier | Reference for spike detection |

## Step 7. Models, tuning and ablations

### 7.1 Model roster

| # | Model | Regression | Spike classification | Notes |
|---|---|---|---|---|
| M1 | Linear Regression | OLS | Logistic Regression | Also a Poisson GLM as a count-appropriate variant |
| M2 | Decision Tree | `DecisionTreeRegressor` | `DecisionTreeClassifier` | Depth and pruning tuned by CV. The classifier yields readable rules. |
| M3 | SVM | `SVR` (RBF) | `SVC` (RBF, Platt scaling) | Standardised inputs, see 7.3 |
| M4 | Deep Neural Network | MLP, 2–3 hidden layers | Same network, sigmoid output | PyTorch or Keras, Adam, early stopping |
| M5 (optional) | Random Forest | `RandomForestRegressor` | `RandomForestClassifier` | Kept from the original proposal as an extension |

### 7.2 Hyperparameter search
Search on the **training split only** with 5-fold CV (stratified by `city × spike`). Grids are fixed in advance:

| Model | Search space |
|---|---|
| Linear / Poisson | ridge `alpha` ∈ {0, 0.01, 0.1, 1, 10} |
| Decision Tree | `max_depth` ∈ {2, 3, 4, 5, 6, 8, 10, None}; `min_samples_leaf` ∈ {1, 5, 20, 50, 100, 200}; `ccp_alpha` ∈ {0, 1e-4, 1e-3, 1e-2} |
| SVM | `C` ∈ {0.1, 1, 10, 100}; `epsilon` ∈ {0.1, 0.5, 1.0} (SVR); `gamma` ∈ {scale, 0.01, 0.1} |
| MLP | hidden layers ∈ {(32), (64,32), (128,64,32)}; dropout ∈ {0, 0.1, 0.3}; learning rate ∈ {1e-3, 3e-4}; weight decay ∈ {0, 1e-4}; batch 256; up to 200 epochs, early stopping (patience 15) on a 10% internal validation split |
| Random Forest | `n_estimators` = 300; `max_depth` ∈ {5, 10, None}; `min_samples_leaf` ∈ {1, 20, 50} |

Rules:
- Selection metric: CV RMSE (regression), CV PR-AUC (classification).
- If the best setting sits on a grid edge, extend the grid once and record it.
- The MLP is stochastic, so train the final configuration with **5 seeds** and report mean ± SD.

### 7.3 SVM compute cap
Kernel SVMs scale roughly quadratically with data size. Tune and train on a **stratified subsample of at most 20,000 training rows**. Run a learning-curve check at 5,000, 10,000 and 20,000 rows. If CV RMSE is flat between 10,000 and 20,000, the cap is justified. If time allows, train once on the full training set as a check.

### 7.4 Neural network guardrails
- Standardise inputs (fit on train). MSE loss for regression, binary cross-entropy for classification.
- Keep the network small. One real signal and about 70,000 rows do not need a large network.
- Save training and validation loss curves for the report.

### 7.5 Ablations
Repeat the final comparison on FS-PM25, FS-NoAQI, FS-Pollutants and FS-Extended. This is the direct test of H1 and shows whether noise features hurt flexible models.

## Step 8. Cross-validated comparison (development only)
Run 5-fold CV repeated 3 times on the training split for every tuned model. Record RMSE, MAE and R² per fold. This checks stability and is **not** the final result.

## Step 9. Final test-set evaluation (one time)
- Refit each tuned model on the full training split and predict the test split **once**. Clip regression predictions at 0.
- Report RMSE, MAE, R² and Poisson deviance with **95% bootstrap CIs** (1,000 resamples).
- Compute **paired bootstrap** differences against Linear Regression and the PM2.5-only baseline.
- Produce residual, predicted-vs-actual and calibration-by-PM2.5-decile plots.

**Sanity alarm:** PM2.5 alone caps R² near 0.154. **Any model with test R² above 0.20 triggers a leakage audit** before anything is reported.

## Step 10. City-level analysis (RQ2)
1. **Pooled vs per-city:** for the best model and for Linear Regression, fit one model per city and compare with the pooled model on each city's test rows. Report weighted and macro averages with CIs.
2. **Slope heterogeneity:** fit OLS `admissions ~ pm2_5 * city`, run an F-test for the interaction, report each city's PM2.5 slope with a 95% CI, and test Delhi vs London directly. Apply **Holm correction**.
3. **Leave-one-city-out:** for each city, train on the other 7 cities' training rows and evaluate on the held-out city's test rows. Compare with the pooled model.
4. Treat **São Paulo (1,748 rows) and Cairo (2,700 rows)** with caution because their intervals are wide.

## Step 11. Spike classification (RQ4)
- Train M1–M4 as classifiers with the same splits and search protocol.
- Choose each decision threshold on **cross-validated training predictions** (maximise F1), never on the test set.
- Report **PR-AUC** (primary), ROC-AUC, precision, recall, F1, balanced accuracy and the confusion matrix against the prevalence baseline.
- Sensitivity check with `class_weight="balanced"`.
- Print the decision tree's rules as the interpretable early-warning output.

## Step 12. Interpretation (RQ3)
- Permutation importance on the test set (30 repeats) for the best regression model and the MLP.
- Linear Regression coefficients with CIs, and the Decision Tree structure.
- Partial dependence of PM2.5 for each model, to see whether non-linear models learned more than a straight line.
- Read importance of correlated features at group level (VIF from Step 2).

## Step 13. Robustness checks

| Check | What it shows |
|---|---|
| Repeat Steps 5–9 with 5 split seeds (cheap models) | The ranking is not an accident of one split |
| Winsorise PM2.5 at the 1st and 99th percentile | Results do not depend on extreme values |
| Drop rows with 0 admissions | Results do not depend on zero cases |
| SVM on the full training set (if feasible) | The 20,000-row cap hid nothing |
| Poisson GLM vs OLS | Count assumptions do not change the conclusion |

## Step 14. Reporting
Fill the tables in [RESULTS_AND_CHECKLIST.md](RESULTS_AND_CHECKLIST.md), export figures to `results/figures/`, and write the report and slides from the results. Put the synthetic-data disclosure on the first page of each.

---

## References
- Wong, T. W., et al. (1999). Air pollution and hospital admissions for respiratory and cardiovascular diseases in Hong Kong. *Occupational and Environmental Medicine*, 56(10), 679–683.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273–297.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52, 239–281.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6, 65–70.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
