# Methodology (as executed)

The complete methodology of the project, step by step: what we did, why (with the literature behind each choice), what we found, and where the evidence is. Every number comes from a file in `results/` produced by a FULL run.

- **Original plan:** written before any code ran, kept unchanged in [archive/METHODOLOGY_PLAN.md](archive/METHODOLOGY_PLAN.md). The section *Changes from the plan* at the end lists every difference.
- **Fixed rules:** [PROJECT_RULES.md](PROJECT_RULES.md).
- **Every decision with its reasoning:** [DECISIONS_LOG.md](DECISIONS_LOG.md) (D1–D17).
- **Results tables:** [RESULTS_AND_CHECKLIST.md](RESULTS_AND_CHECKLIST.md).

> **The dataset is synthetic** (Step 2). Every conclusion is about how models behave on this benchmark, not about real health effects.

---

## Overview

| Step | Name | Status | One-line result |
|---|---|---|---|
| 0 | Setup | Done | Python 3.12 environment, pinned packages, seed 42, FAST/FULL modes |
| 1 | Load and validate | Done | 88,489 rows × 12 columns; all 11 checks pass |
| 2 | Data audit | Done | One signal (PM2.5, r = 0.39); no time series; identical cities; synthetic |
| 2B | Real-data check | Skipped | – |
| 3 | Target and spike label | Done | Spike = admissions ≥ 13 (training 90th percentile); 11.7% of days |
| 4 | Feature sets | Done | FS-Primary (8 inputs) and FS-PM25 (1 input) |
| 5 | Train/test split | Done | 70,791 / 17,698 rows, stratified by city × spike |
| 6 | Baselines | Done | PM2.5-only line: CV RMSE 3.417 (R² 0.154); prevalence PR-AUC 0.117 |
| 7 | Models and tuning | Done | Every model ties at CV RMSE ≈ 3.41–3.42 |
| 8 | Repeated CV | Skipped | – |
| 9 | Final test evaluation | Done | Every model ties with the PM2.5 line (test RMSE 3.409–3.425); H1 supported |
| 10 | City analysis | Done | Per-city models never beat pooled; slopes 0.090–0.106 |
| 11 | Spike classification | Done | F1 ≈ 0.31 for every model vs 0.21 for flagging every day |
| 12 | Interpretation | Done | Only PM2.5 matters; its effect is a 10 µg/m³ staircase |
| 13 | Robustness checks | Skipped | – |
| 14 | Reporting | Report done; slides deferred | `reports/REPORT.md` |

**Research questions.**
- **RQ1:** which model predicts admissions best?
- **RQ2:** do cities differ?
- **RQ3:** which features matter?
- **RQ4:** can spike days be flagged in advance?
- **H1:** PM2.5 alone predicts about as well as all features.

---

## Step 0. Setup
**Goal:** make every result reproducible.

**What was done:**
- **Environment:** Python 3.12 in a project `.venv` created with `uv`, with package versions pinned in `requirements.txt`: pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, statsmodels 0.15.0, matplotlib 3.11.2.
- **Seed:** one global `SEED = 42` (`src/config.py`) for the split, folds, subsamples and models.
- **Scripts, not notebooks.** Every script has two modes:
  - **FAST:** a small sample, written only to `results_fast/` with a `_FAST` suffix, used only to check that code runs.
  - **FULL:** report-grade results, written to `results/`.
- **Guards:** every table has a `mode` column. `src/check_results.py` fails if FAST data ever reaches `results/`, and the final evaluation refuses to run unless that check passes.

**Literature:** scikit-learn (Pedregosa et al., 2011).

**Result:** every number in this document can be regenerated with the commands in [PROJECT_RULES.md](PROJECT_RULES.md), "Running on another computer".

## Step 1. Load and validate
**Goal:** confirm the data is what we think it is, instead of assuming it.

**What was done:** `src/data.py` `load_raw()` checks 11 conditions on every load:
- 12 expected columns and 88,489 rows
- no missing values and no duplicate rows
- 8 cities and 3 population-density levels
- pollutants non-negative, AQI within 0–500, humidity within 0–100
- admissions are non-negative integers and never exceed hospital capacity

**Result:** all 11 checks pass, so no cleaning was needed. The only negative values are temperatures down to −5 °C, which are plausible. The dataset file's SHA-256 is recorded in [PROJECT_RULES.md](PROJECT_RULES.md).

**Evidence:** `results/tables/01_validation.csv`.

## Step 2. Data audit
**Goal:** find out what the data can and cannot support before modelling.

**What was done:** `src/audit.py` produced:
- distributions of all 9 numeric variables
- correlations with admissions (Pearson and Spearman)
- mean admissions by PM2.5 decile, with 95% CIs
- per-city summaries and per-city PM2.5 correlations
- date coverage and gap analysis
- lag-1 autocorrelation per city

**Findings:**

| Finding | Evidence | Consequence |
|---|---|---|
| **No time series.** All 88,489 dates are unique, one per calendar day from 2020-01-01 to 2262-04-10, with cities scattered randomly across them. Lag-1 autocorrelation is about 0 in every city (−0.013 to 0.045). | `02_date_gaps.csv`, `02_lag_autocorrelation.csv` | Rows are independent, so a random split is valid. Prophet and lag features were dropped. |
| **One signal.** PM2.5 correlates with admissions at r = 0.392 (Spearman 0.384). Every other variable has \|r\| < 0.005. | `02_target_correlations.csv` | The best achievable R² is about r² ≈ 0.154. |
| **A steady rise.** Mean admissions rise from 5.56 to 10.64 across PM2.5 deciles. | `02_pm25_deciles.csv`, `02_pm25_deciles.png` | Supports the use of linear models. |
| **Identical cities.** Mean PM2.5 is 35.0–35.4, mean admissions 7.95–8.11 and mean AQI about 249 in every city. Per-city PM2.5 correlation ranges from 0.36 to 0.42. | `02_city_means.csv` | The "high- vs low-pollution city" framing from the proposal is unsupported. |
| **AQI is unrelated to PM2.5** (r = 0.003), and AQI is spread evenly over 0–499. | `02_correlation_heatmap.png` | In real data AQI is calculated from pollutant levels, so this is a sign of generated data. |
| **Count outcome:** mean 8.05, variance 13.80, 2% zeros, maximum 25. | `02_numeric_summary.csv` | Predictions are clipped at 0, and Poisson deviance is reported. |
| **Unbalanced cities:** from Delhi (26,465 rows) down to São Paulo (1,748). | `02_city_means.csv` | Stratified split; small cities flagged. |

**Literature (context for the data):** real studies of PM2.5 and respiratory admissions (Wong et al., 1999; Dominici et al., 2006) find smooth, roughly linear relationships with time structure such as lags and seasonality (WHO, 2021). This data has none of that time structure.

**Conclusion:** the data behaves like a controlled benchmark: one nearly linear signal plus noise. Step 12 later showed the signal is an exact staircase. Full details are in [DATA_AUDIT.md](DATA_AUDIT.md).

## Step 2B. Real-data reference check: skipped
Skipped under the lean scope. The report states that the Delhi-vs-London pollution claim cannot be tested with this file.

## Step 3. Target and spike label
**What was done:**
- **Regression target:** `hospital_admissions`.
- **Spike label:** admissions ≥ the 90th percentile of the **training set**, which is 13. 11.66% of training days are spikes.
- The threshold is stored once and loaded by every later script, never recomputed. An early SVM script recomputed it on its own subsample; this was fixed (D4).

**Literature:** computing any cut-off on test data leaks information into the model (Hastie et al., 2009, §7.10.2).

**Evidence:** `results/tables/03_spike_threshold.json`.

## Step 4. Feature sets
**What was done:**

| ID | Inputs | Purpose |
|---|---|---|
| FS-Primary | `aqi, pm2_5, pm10, no2, o3, temperature, humidity` + one-hot `city` | Main comparison: the proposal's features |
| FS-PM25 | `pm2_5` only | Tests H1 |

`date` is never used (Step 2). `hospital_capacity` is excluded as a hospital-supply variable, and `population_density` because it is assigned randomly per row. The three planned extra feature sets were dropped under the lean scope.

## Step 5. Train/test split
**What was done:**
- One 80/20 split, stratified by city × spike, seed 42: **70,791 training rows and 17,698 test rows**.
- The row indices are saved in `data/processed/split_indices.json`.
- `src/test_split.py` confirms that no row is in both sets and that city and spike proportions match.
- **Rule:** the test set is read only by `src/final_evaluation.py`, after every choice has been fixed.

**Literature:** a random split is valid only when rows are independent (Step 2). Real daily data would need a chronological split.

**Caveat:** the stratification used a provisional 90th-percentile cutoff computed on all rows. This only decides which rows go to which side; the spike threshold itself comes from training rows (D3).

## Step 6. Baselines
**Goal:** give every model a reference to beat.

**What was done:** 5-fold CV on training rows, using the **same folds as every later model** (stratified by city × spike, seed 42), so comparisons are paired.

**Result:**

| Baseline | CV RMSE | CV MAE | CV R² | CV PR-AUC |
|---|---|---|---|---|
| Mean predictor | 3.716 | 2.949 | 0.000 | – |
| City mean | 3.716 | 2.949 | 0.000 | – |
| PM2.5-only linear | **3.417** | 2.711 | **0.154** | – |
| Prevalence | – | – | – | 0.117 (ROC-AUC 0.5) |

City identity adds nothing. The PM2.5 line reaches the ceiling the audit predicted.

**Evidence:** `06_baselines.csv`.

## Step 7. Models and tuning
**Common protocol** (rules in [PROJECT_RULES.md](PROJECT_RULES.md)):
- 5-fold CV on the shared folds, training rows only.
- Selection by CV RMSE (regression) and CV PR-AUC (classification). PR-AUC is preferred for a rare positive class (Saito & Rehmsmeier, 2015).
- Preprocessing inside Pipelines.
- **Search size:** n_iter is chosen so that 1 − (1 − p)^n ≥ 0.95, where p is the measured share of good settings (Bergstra & Bengio, 2012). That gives ≥ 10 draws for regression and ≥ 30 for classification; cheap models use full grids.
- **Edge rule:** a range whose best value sits on its edge is widened once, and the widening is recorded.

### 7.1 Linear Regression (Ridge) and Logistic Regression (D11)
- **Setup:** numeric inputs standardised, because the L2 penalty depends on each feature's units (Hoerl & Kennard, 1970; Hastie et al., 2009, §3.4).
- **Grids:** Ridge alpha {0, 0.01, 0.1, 1, 10}, widened once to {30 … 1000}. Logistic C {1e-3 … 1e3}, widened to 1e-4.
- **Result:**
  - CV RMSE **3.4176** (FS-Primary), **3.4171** (FS-PM25). Logistic PR-AUC **0.2396** for both, with 0 convergence warnings.
  - RMSE is flat for every alpha from 0 to 1000 (3.41759–3.41763), so shrinkage has nothing to correct.
  - With PM2.5 alone, PR-AUC is identical at every C, as ranking theory predicts (Fawcett, 2006).
  - The collinear one-hot city columns at alpha = 0 (the "dummy variable trap") were harmless.

### 7.2 Decision Tree (D11)
- **Setup:** CART with cost-complexity pruning (Breiman et al., 1984).
- **Grid:** a full grid of depth {2, 3, 4, 5, 6, 8, 10, none} × minimum leaf size {1, 5, 20, 50, 100, 200}, widened once to {500, 1000, 2000} because the classifier's best leaf size sat at 200. Pruning strengths were taken from the training data's pruning path.
- **Result:**
  - CV RMSE **3.4077 / 3.4075** (depth 4).
  - PR-AUC **0.2274 / 0.2323**, best at a leaf size of 500, inside the widened range. Large leaves help ranking because each leaf's spike rate is estimated from more rows (Provost & Domingos, 2003).
  - The tree splits only on PM2.5.

### 7.3 SVM, RBF kernel (D7–D10)
- **Setup:** SVM training cost grows faster than the square of the row count (Chang & Lin, 2011); measured here as n^1.9 for SVR and n^2.1 for SVC. So the search runs on subsamples, coarse first and then fine (Hsu, Chang & Lin, 2003):
  - **Phase A:** 40 random settings × 5 folds on 10k rows.
  - **Phase B:** 20 settings × 5 folds on 20k rows, in a narrower window around the Phase A best.
- **Search ranges:** C from 0.01 to 1000 and gamma from 1e-4 to 1 (or "scale"), both on a log scale. Epsilon from 0.05 to 3.5, with the top set to about one noise SD (Cherkassky & Ma, 2004).
- **Learning curve:** 5k → 40k rows, 3 folds. The final fit uses 20k rows, because the gain from 20k to 40k was smaller than the fold-to-fold noise in all 4 cases (at most 0.004 RMSE, against an SD of 0.017).
- **SVC scores:** taken from `decision_function`. There's no probability calibration because of the compute limit.
- **Result:**
  - Fair CV (fixed settings, full-training folds, 20k rows): RMSE **3.419 / 3.417**, PR-AUC **0.239 / 0.239**. Identical to Linear/Logistic.
  - The tuning scores (3.411, 0.251) are optimistic, being the best of about 60 candidates (Cawley & Talbot, 2010), so they were not used for comparison.
  - CV chose a small gamma (0.001–0.005), which makes the RBF SVM nearly linear (Keerthi & Lin, 2003).
  - CV RMSE is flat across epsilon.
  - Range widenings: FS-Primary SVC C (low); FS-PM25 SVR epsilon (high); FS-PM25 SVC C (high) and gamma (low).
- **Runtime:** 1 h 16 min.

### 7.4 MLP (D13)
- **Setup:** sklearn MLP (Goodfellow et al., 2016) with standardised inputs, the Adam optimiser and batches of 256, for up to 200 passes over the data.
  - **Search:** random over {(32), (64, 32), (128, 64, 32)} networks, L2 alpha from 1e-6 to 0.1 (in place of dropout, which sklearn lacks) and learning rate; 20 draws for regression and 30 for classification, each widened once on the learning rate.
  - **Regressor:** validation early stopping.
  - **Classifier:** stops on training loss, because sklearn's classifier early stopping watches accuracy, which can't guide training at 11.7% prevalence.
  - **Seeds:** the chosen settings were retrained with 5 seeds.
- **Result:**
  - 5-seed CV RMSE **3.4232 / 3.4155**, PR-AUC **0.2352 / 0.2396**. The spread across seeds is 0.0011 or less.
  - The final models stopped after 25–65 passes.
- **Runtime:** 1 h 45 min.

### 7.5 Step 7 summary
Every model ties with the PM2.5-only baseline in CV. The searches were made larger after we measured that 10 random draws carried roughly a 1-in-4 risk of under-tuning the classifiers.

**Evidence:** `07_*` tables and figures.

## Step 8. Repeated CV: skipped
Skipped under the lean scope. Its purpose, checking stability, is covered by:
- fold SDs from every 5-fold CV
- the MLP's 5-seed reruns
- the SVM learning curve across folds
- the 1,000-resample bootstrap CIs in Step 9

## Step 9. Final test evaluation, used once (D14)
**What was done:**
- **Guards:** `check_results.py` had to pass first, and the script was dry-run on training rows only. That dry run caught a bug in which the tree and MLP classifier results overwrote their regression results.
- **Models:** Linear/Logistic/Tree refitted with their stored settings on all training rows; the saved SVM (20k rows) and MLP (5 seeds) loaded.
- **Prediction:** the 17,698 test rows predicted once, with regression predictions clipped at 0. The predictions were saved, so later steps don't predict the test set again.
- **Metrics:** RMSE, MAE, R² and Poisson deviance; PR-AUC and ROC-AUC. 95% CIs come from 1,000 bootstrap resamples (Efron & Tibshirani, 1993), and comparisons use the same resampled rows for both models (paired).
- **Verdict rule:** two models differ only if the RMSE gap is ≥ 0.05 and the paired CI excludes 0.
- **Leakage alarm:** set at R² > 0.20.

**Result:**

| Model (FS-Primary) | Test RMSE [95% CI] | R² | Δ vs Linear [95% CI] | Verdict |
|---|---|---|---|---|
| Mean predictor | 3.714 [3.675, 3.751] | 0.000 | +0.294 | worse |
| PM2.5-only line | 3.419 [3.384, 3.460] | 0.152 | −0.0004 [−0.0010, 0.0003] | tie |
| Linear (Ridge) | 3.420 [3.384, 3.461] | 0.152 | – | – |
| Decision Tree | 3.409 [3.374, 3.449] | 0.157 | −0.011 [−0.015, −0.006] | tie (real but 5× below the margin) |
| SVR | 3.420 [3.385, 3.462] | 0.152 | +0.0007 [−0.0004, 0.0019] | tie |
| MLP (5 seeds) | 3.425 [3.390, 3.466] | 0.150 | +0.005 [0.003, 0.007] | tie |

- **H1** (FS-Primary vs FS-PM25, test): every regression model ties (largest gap +0.007, MLP). The tree's predictions are identical for both feature sets.
- **Spike ranking:** PR-AUC is Logistic 0.249 = SVC 0.249 = MLP 0.247 (0.249 with PM2.5 alone), Tree 0.238, against prevalence 0.117. ROC-AUC is about 0.71.
- **No leakage alarm** (highest R² 0.157).

**Evidence:** `09_test_*.csv`, `09_*.png`. The full tables are in [RESULTS_AND_CHECKLIST.md](RESULTS_AND_CHECKLIST.md), Tables 1–3.

## Step 10. City analysis (D16–D17)
### 10.1 Pooled vs per-city models (test)
- **Models compared:** Linear, and the regression model that was best in CV (the Decision Tree; chosen on CV, not on the test set).
- **Per-city models:** tuned by 5-fold CV on **each city's own training rows**. A dry run showed that reusing the pooled tree's settings overfit small cities and biased the comparison (D16).
- **Result:** per-city models never beat the pooled model in any city.
  - Weighted over all test rows: Linear Δ = +0.0005 [−0.0007, 0.0017]; Tree Δ = +0.0065 [0.0031, 0.0102]. Both ties.
  - Pooling helps the tree slightly, because each split is estimated from more data.

### 10.2 PM2.5 slope by city (training data)
- **Model:** OLS `admissions ~ pm2_5 * city`, with HC3 robust standard errors because count variance grows with the mean (MacKinnon & White, 1985).
- **Result:**
  - The slopes range from 0.090 (São Paulo) to 0.106 (London); pooled 0.0990 [0.0973, 0.1007].
  - The test of whether slopes differ by city gives Wald χ²(7) = 15.7, p = 0.028. London vs Delhi: +0.0075, p = 0.028. Both unadjusted; Holm correction was dropped.
- **Reading:** small differences that are detectable with 70k rows. They may partly come from fitting a straight line to a staircase (Step 12), and they don't improve prediction (10.1). São Paulo and Cairo have the widest CIs.

**Evidence:** `10_*` tables, `10_city_slopes.png`. Full tables: Tables 4–5 in [RESULTS_AND_CHECKLIST.md](RESULTS_AND_CHECKLIST.md).

## Step 11. Spike classification (D17)
- **What was done:**
  - **Thresholds:** each model's decision threshold maximises F1 on **out-of-fold training** predictions (5 folds; 5 seeds averaged for the MLP; the SVM's own 20k-row sample for the SVC). It is then applied once to the saved test scores.
  - **Sensitivity check:** Logistic with `class_weight="balanced"`.
  - **Interpretable output:** spike rates by PM2.5 band, taken from the tuned PM2.5-only tree.
- **Result (test):**
  - F1 is about 0.31 for every model (Logistic 0.314, Tree 0.311, MLP 0.313), against 0.209 for flagging every day. Precision is about 0.22 and recall about 0.52, with about 28% of days flagged.
  - The SVC flags 46% of days (F1 0.295): its raw decision-function scale shifts between the CV models and the final model, a known weakness of thresholding uncalibrated SVM scores.
  - Class weighting changes nothing (F1 0.317).
  - Spike rate climbs in steps with PM2.5: about 3% below 20 µg/m³, 6% for 20–30, 10% for 30–40, 15% for 40–50, 22% for 50–60 and 30–44% above 60.

**Evidence:** `11_*` tables. Full table: Table 6 in [RESULTS_AND_CHECKLIST.md](RESULTS_AND_CHECKLIST.md).

## Step 12. Interpretation (D15, D17)
- **What was done:**
  - permutation importance on the test set: 5 repeats, the tree on all rows and the MLP on a 3,000-row subsample (Breiman, 2001)
  - partial dependence of predictions on PM2.5, on training data (Friedman, 2001)
  - standardised OLS coefficients with HC3 CIs
  - the tree's structure
  - a check on training data of which shape (line, quadratic, spline, bins, isotonic, steps) best describes PM2.5 → admissions
- **Result:**
  - **Permutation importance:** shuffling PM2.5 raises RMSE by 0.595 (tree) and 0.568 (MLP). Every other input changes it by ≤ 0.003.
  - **OLS:** PM2.5 = 1.46 admissions per SD [1.435, 1.485]. Every other CI includes 0, except one city term (Beijing, p = 0.03), about what chance gives across 13 tests.
  - **The regression tree** uses only PM2.5.
  - **The PM2.5 effect is a staircase:** mean admissions ≈ 5.05 + 0.99 × floor(PM2.5 / 10).
    - It beats a straight line in CV (RMSE 3.4065 vs 3.4171) and matches the tree (3.4077).
    - Only a band width of 10 works (8, 12 and 15 are all worse).
    - The tree's splits fall at 10.05, 19.95, 29.95, 39.95, 49.95 and 59.95.
  - **This explains the tree's small, real test advantage:** trees represent step functions exactly (Breiman et al., 1984). The partial dependence plot shows the tree tracing the steps while Linear, SVR and MLP draw the same straight line.
- **Note:** the staircase was found after Step 9 and on training data only. No model was rebuilt and re-scored on the test set.

**Evidence:** `12_*` tables and figures. Full table: Table 7 in [RESULTS_AND_CHECKLIST.md](RESULTS_AND_CHECKLIST.md).

## Step 13. Robustness checks: skipped
Skipped under the lean scope. The SVM's subsample cap is still justified by its learning curve (7.3).

## Step 14. Reporting
- **Report:** [../reports/REPORT.md](../reports/REPORT.md), with the synthetic-data disclosure first.
- **Slides:** deferred.

---

## Answers to the research questions
- **RQ1:** no model beats the others in practice. All tie at R² ≈ 0.15, the ceiling set by one variable with r = 0.39. The tree is detectably but negligibly better (−0.011 RMSE), because the data's effect is a staircase.
- **H1: supported.** PM2.5 alone is as good as all features, for every model.
- **RQ2:** cities share one relationship. Per-city models don't help; the small slope differences are detectable but practically irrelevant.
- **RQ3:** PM2.5 is the only input that matters.
- **RQ4:** spike days can be flagged better than chance (PR-AUC 0.25 vs 0.12; F1 0.31 vs 0.21), but the warning is weak.

## Literature used in this methodology
| Topic | Source | Used in |
|---|---|---|
| ML library | Pedregosa et al. (2011) | All steps |
| Leakage; CV; bias–variance; model complexity | Hastie, Tibshirani & Friedman (2009) | Steps 3, 5, 7, discussion |
| Ridge / L2 penalty | Hoerl & Kennard (1970) | 7.1 |
| ROC and ranking invariance | Fawcett (2006) | 7.1, 9, 11 |
| PR-AUC for imbalanced classes | Saito & Rehmsmeier (2015) | 7, 11 |
| CART and cost-complexity pruning | Breiman et al. (1984) | 7.2, 12 |
| Probability trees for ranking | Provost & Domingos (2003) | 7.2 |
| SVM | Cortes & Vapnik (1995); Chang & Lin (2011) | 7.3 |
| SVM tuning procedure | Hsu, Chang & Lin (2003) | 7.3 |
| SVR epsilon and noise | Cherkassky & Ma (2004) | 7.3 |
| RBF SVM approaching linear at small gamma | Keerthi & Lin (2003) | 7.3, 12 |
| Random search budget | Bergstra & Bengio (2012) | 7 |
| Optimism of tuning scores | Cawley & Talbot (2010) | 7.3, 9 |
| Neural networks, early stopping, L2 | Goodfellow, Bengio & Courville (2016) | 7.4 |
| Bootstrap CIs | Efron & Tibshirani (1993) | 9, 10, 11 |
| Robust (HC3) standard errors | MacKinnon & White (1985) | 10.2, 12 |
| Permutation importance | Breiman (2001) | 12 |
| Partial dependence | Friedman (2001) | 12 |
| Simple models often suffice | Holte (1993); Hand (2006); Christodoulou et al. (2019) | Discussion |
| PM2.5 and respiratory admissions | Wong et al. (1999); Dominici et al. (2006); WHO (2021) | Step 2, limits of the data |

Full references with links: [../reports/REPORT.md](../reports/REPORT.md#references).

## Changes from the plan
| Plan | As run | Why |
|---|---|---|
| Five feature sets | Two (FS-Primary, FS-PM25) | Lean scope; FS-PM25 is the H1 test |
| Steps 2B, 8, 13; Holm; leave-one-city-out; Poisson GLM; VIF | Skipped | Lean scope; each stated as a limitation |
| Notebooks | Scripts with FAST/FULL modes | Reproducibility (D1) |
| Ridge grid {0 … 10} | Same, widened once to 1000; inputs standardised | Edge rule; L2 penalty depends on units (D11) |
| Tree `ccp_alpha` ∈ {0, 1e-4, 1e-3, 1e-2} | Candidates from the training pruning path; leaf grid widened to 2000 | Planned values ignored the data's scale; edge rule (D11) |
| SVM grid C {0.1–100}, epsilon {0.1, 0.5, 1}, gamma {scale, 0.01, 0.1}; Platt scaling | Two-phase random search over wider log ranges; epsilon tied to noise; no Platt scaling | Literature (Hsu et al.; Cherkassky & Ma); compute limit (D7, D8) |
| SVM learning curve 5k/10k/20k | 5k/10k/20k/40k, flat; final 20k | D7, D10 |
| MLP in PyTorch/Keras with dropout | sklearn MLP with L2 alpha, 5 seeds | Allowed packages; sklearn has no dropout (D13) |
| Original compute rule: random search n_iter 10–30; SVM n_iter = 10, cv = 3 | Search size from 1 − (1 − p)^n ≥ 0.95; 5-fold CV throughout | 10 draws risked under-tuning the classifiers (D7, PROJECT_RULES) |
| Interaction F-test | Wald χ² test with HC3 robust SEs | Count variance grows with the mean (D17) |
| Per-city models | Tuned on each city's own data | Fair comparison (D16) |
| Permutation importance: 30 repeats | 5 repeats; 3,000-row subsample for the MLP | Compute limit |
| Test evaluation as one step | Step 9 plus the test-side parts of Steps 10–12, all in `final_evaluation.py`, with every setting fixed first | The test set stays in one script (D14, D16) |
