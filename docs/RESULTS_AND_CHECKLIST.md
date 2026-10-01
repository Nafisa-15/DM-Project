# Results Tables and Checklist

Templates for the final (test-set) results, plus the project checklist.

> **Status: updated 2026-10-01 to the lean scope.** Templates for skipped analyses (Poisson GLM, extra feature sets, leave-one-city-out, Holm, repeated CV, robustness) were removed. They are listed in [METHODOLOGY.md](METHODOLOGY.md) "Changes from the plan". Cross-validation numbers (training data only) are in [MODEL_JUSTIFICATION.md](MODEL_JUSTIFICATION.md). The tables below hold the one-time test evaluation (Steps 9–12); all are filled.

## Results tables (fill from Step 9 onward)

**Table 1. Test-set performance (95% bootstrap CI, 1,000 resamples).** Filled 2026-10-01 from `results/tables/09_test_regression.csv`. Test set: 17,698 rows, used once. No leakage alarm (highest R² 0.157).

| Model | Feature set | RMSE [95% CI] | MAE | R² | Poisson deviance |
|---|---|---|---|---|---|
| Mean predictor | – | 3.7135 [3.675, 3.751] | 2.946 | 0.000 | 1.914 |
| PM2.5-only linear | FS-PM25 | 3.4193 [3.384, 3.460] | 2.717 | 0.152 | 1.654 |
| Linear Regression (Ridge) | FS-Primary | 3.4197 [3.384, 3.461] | 2.717 | 0.152 | 1.655 |
| Decision Tree | FS-Primary | 3.4089 [3.374, 3.449] | 2.691 | 0.157 | 1.645 |
| SVM (SVR) | FS-Primary | 3.4204 [3.385, 3.462] | 2.718 | 0.152 | 1.655 |
| MLP (5-seed average) | FS-Primary | 3.4245 [3.390, 3.466] | 2.721 | 0.150 | 1.659 |

**Table 2. Paired RMSE differences vs Linear Regression (FS-Primary).** From `09_test_paired_regression.csv`, which also compares every model with the PM2.5 baseline.

| Model | ΔRMSE | 95% bootstrap CI | Verdict (tie if gap < 0.05 or the CI includes 0) |
|---|---|---|---|
| PM2.5-only linear | −0.0004 | [−0.0010, 0.0003] | tie |
| Decision Tree | −0.0108 | [−0.0152, −0.0062] | tie (the gap is real but far below the 0.05 margin) |
| SVR | +0.0007 | [−0.0004, 0.0019] | tie |
| MLP | +0.0048 | [0.0028, 0.0067] | tie (real but far below the margin) |
| Mean predictor | +0.2938 | [0.2710, 0.3134] | worse |

**Table 3. H1: FS-Primary vs FS-PM25 (test).** From `09_test_h1.csv`.

| Model | FS-Primary | FS-PM25 | Difference [95% CI] | Verdict |
|---|---|---|---|---|
| Linear (RMSE) | 3.4197 | 3.4193 | +0.0004 [−0.0003, 0.0010] | tie |
| Tree (RMSE) | 3.4089 | 3.4089 | 0.0000 (identical: the tree splits only on PM2.5) | tie |
| SVR (RMSE) | 3.4204 | 3.4192 | +0.0012 [0.0001, 0.0023] | tie |
| MLP (RMSE) | 3.4245 | 3.4171 | +0.0074 [0.0050, 0.0097] | tie |
| Logistic (PR-AUC) | 0.2492 | 0.2490 | +0.0002 [−0.0006, 0.0010] | tie |
| Tree classifier (PR-AUC) | 0.2382 | 0.2437 | −0.0055 [−0.0089, −0.0021] | FS-PM25 better |
| SVC (PR-AUC) | 0.2493 | 0.2490 | +0.0003 [−0.0006, 0.0011] | tie |
| MLP (PR-AUC) | 0.2474 | 0.2490 | −0.0016 [−0.0035, 0.0001] | tie |

**Table 4. Per-city results: pooled vs per-city model (test RMSE).** From `10_test_pooled_vs_city.csv`. Each per-city model is tuned by CV on that city's own training rows (`10_city_params.csv`).

| City | n (test) | Linear: pooled / per-city | Tree: pooled / per-city | Verdict |
|---|---|---|---|---|
| Beijing | 4,413 | 3.443 / 3.444 | 3.428 / 3.432 | tie |
| Cairo * | 540 | 3.547 / 3.547 | 3.538 / 3.565 | tie |
| Delhi | 5,293 | 3.440 / 3.439 | 3.428 / 3.428 | tie |
| London | 1,397 | 3.305 / 3.303 | 3.296 / 3.304 | tie |
| Los Angeles | 1,800 | 3.415 / 3.419 | 3.416 / 3.423 | tie |
| Mexico City | 2,676 | 3.363 / 3.365 | 3.348 / 3.364 | tie |
| São Paulo * | 350 | 3.483 / 3.486 | 3.487 / 3.523 | tie |
| Tokyo | 1,229 | 3.430 / 3.428 | 3.426 / 3.430 | tie |
| **Weighted (all rows)** | 17,698 | 3.420 / 3.420 (Δ +0.0005 [−0.0007, 0.0017]) | 3.409 / 3.415 (Δ +0.0065 [0.0031, 0.0102]) | tie |
| **Macro (city mean)** | 17,698 | 3.428 / 3.429 | 3.421 / 3.434 (Δ +0.013) | tie |

\* small sample. Per-city models never beat the pooled model. For the tree, pooling helps slightly, because each split is estimated from more rows.

**Table 5. PM2.5 slope by city (OLS `admissions ~ pm2_5 * city` on training rows, HC3 robust SEs).** From `10_city_slopes.csv` and `10_interaction_test.json`.

| City | n (train) | Slope | 95% CI |
|---|---|---|---|
| London | 5,588 | 0.1058 | [0.0998, 0.1117] |
| Beijing | 17,651 | 0.1011 | [0.0978, 0.1044] |
| Los Angeles | 7,203 | 0.1009 | [0.0957, 0.1061] |
| Tokyo | 4,918 | 0.0998 | [0.0933, 0.1063] |
| Delhi | 21,172 | 0.0982 | [0.0951, 0.1014] |
| Cairo * | 2,160 | 0.0939 | [0.0838, 0.1040] |
| Mexico City | 10,701 | 0.0938 | [0.0893, 0.0982] |
| São Paulo * | 1,398 | 0.0904 | [0.0783, 0.1026] |
| **Pooled** | 70,791 | 0.0990 | [0.0973, 0.1007] |

- **Do the slopes differ by city?** Wald χ²(7) = 15.7, p = 0.028.
- **Delhi vs London** (the one planned comparison): London's slope is higher by 0.0075, p = 0.028.
- The p-values are unadjusted (Holm was dropped). The differences are statistically detectable with 70k rows but small: the largest slope is 17% above the smallest. They may partly come from the staircase shape, since a straight-line slope fitted to steps depends on where each city's PM2.5 values fall relative to the step edges. They don't help prediction (Table 4).

**Table 6. Spike classification (FS-Primary, test set).** PR-AUC and ROC-AUC come from Step 9 (`09_test_classification.csv`). The threshold-based columns come from Step 11 (`11_test_thresholded.csv`): each threshold maximises F1 on out-of-fold training scores (`11_thresholds.csv`) and is then applied to the saved test scores.

| Model | PR-AUC [95% CI] | ROC-AUC | Days flagged | Precision | Recall | F1 [95% CI] | Balanced accuracy |
|---|---|---|---|---|---|---|---|
| Prevalence (flag every day) | 0.1167 [0.112, 0.122] | 0.500 | 100% | 0.117 | 1.000 | 0.209 [0.201, 0.217] | 0.500 |
| Logistic Regression | 0.2492 [0.232, 0.266] | 0.713 | 26% | 0.227 | 0.511 | 0.314 [0.300, 0.329] | 0.640 |
| Decision Tree | 0.2382 [0.223, 0.253] | 0.708 | 28% | 0.221 | 0.524 | 0.311 [0.296, 0.324] | 0.640 |
| SVM (decision_function) | 0.2493 [0.232, 0.266] | 0.713 | 46% | 0.185 | 0.728 | 0.295 [0.284, 0.307] | 0.652 |
| MLP (5-seed average) | 0.2474 [0.231, 0.264] | 0.711 | 29% | 0.221 | 0.539 | 0.313 [0.299, 0.328] | 0.644 |
| Logistic, `class_weight="balanced"` (sensitivity check) | – | – | 30% | 0.221 | 0.558 | 0.317 [0.303, 0.331] | 0.649 |

- **All models reach F1 ≈ 0.31,** against 0.21 for flagging every day. As an early-warning rule: flag about 28% of days to catch about half of the spike days, and roughly 1 flagged day in 4.5 is a real spike.
- **SVC flags more days (46%).** Its threshold was set on CV models trained on 16k-row folds of the SVM's 20k-row sample, and the decision-function scale shifts slightly when the model is refit on all 20k rows. This is a limitation of thresholding raw SVM scores (no Platt scaling, per PROJECT_RULES.md).
- **Class weighting changes nothing** (F1 0.317 vs 0.314): the threshold absorbs the weighting, because ranking quality is the same.
- **With PM2.5 alone,** the SVC and MLP flag exactly the same days, since both rank days by PM2.5.
- **Early-warning rule:** the spike rate by PM2.5 band, from the tuned PM2.5-only tree on training rows (`11_tree_pm25_bands.csv`), shows the staircase: about 0.03 below 20 µg/m³, 0.06 for 20–30, 0.10 for 30–40, 0.15 for 40–50, 0.22 for 50–60 and 0.30–0.44 above 60.

Paired vs Logistic (`09_test_paired_classification.csv`): SVC tie (+0.0001); MLP −0.0018 [−0.0037, −0.0001]; Tree −0.0109 [−0.0162, −0.0063]. There is no pre-set PR-AUC margin, so these small "worse" verdicts are statistically detectable but practically small. Every model is about 2.1× the prevalence baseline.

**Table 7. Permutation importance on the test set** (increase in RMSE when the column is shuffled; 5 repeats; `12_permutation_importance.csv`)

| Feature | Decision Tree (CV-best; 17,698 rows) | MLP (3,000-row subsample) |
|---|---|---|
| pm2_5 | **0.595** (± 0.007) | **0.568** (± 0.030) |
| humidity | 0.000 | 0.003 |
| temperature | 0.000 | 0.003 |
| o3 | 0.000 | 0.001 |
| city | 0.000 | 0.001 |
| aqi | 0.000 | 0.001 |
| pm10 | 0.000 | 0.000 |
| no2 | 0.000 | −0.001 |

- **Only PM2.5 matters.** The tree never splits on anything else (`12_tree_features_used.json`).
- The MLP's tiny non-zero values (≤ 0.003) are noise it fitted from the extra inputs.
- **Linear coefficients agree** (`12_linear_coefficients.csv`, standardised inputs): PM2.5 = 1.46 admissions per SD [1.435, 1.485]. Every other variable's CI includes 0, except one city term (Beijing, p = 0.03), which is about what chance gives across 13 unadjusted tests.
- **The partial dependence plot** (`12_partial_dependence.png`) shows the tree tracing the 10 µg/m³ staircase, while Linear, SVR and MLP fit the best straight line through it.

**Figures:**
- **Done:** target distribution; numeric distributions; correlation heatmap; PM2.5 vs admissions; admissions by PM2.5 decile; SVM learning curves.
- **Also done:** MLP loss curves; test predicted vs actual, residuals, decile calibration and RMSE CIs (`09_*`); per-city slope forest plot (`10_city_slopes.png`); tree diagram (`12_tree_structure.png`); partial dependence (`12_partial_dependence.png`); permutation importance (`12_permutation_importance.png`).
- **Not made:** PR curves (PR-AUC values are in Table 6).

## Project checklist

- [x] **Step 0:** repository, environment, seed
- [x] **Step 1:** loader with validation assertions passing (`01_validation.csv`)
- [x] **Step 2:** audit reproduces every number in DATA_AUDIT.md (verified 2026-10-01). VIF skipped.
- [ ] ~~Step 2B~~ skipped (lean scope)
- [x] **Step 3:** spike threshold 13 from training rows only
- [x] **Step 4:** FS-Primary and FS-PM25 fixed in `config.py`
- [x] **Step 5:** stratified split saved; split check passing
- [x] **Step 6:** baselines evaluated with 5-fold CV
- [x] **Step 7:** all four models tuned (FULL); SVM learning curve; MLP 5 seeds
- [ ] ~~Step 8~~ skipped (lean scope)
- [x] **Step 9:** single test-set evaluation with bootstrap CIs (2026-10-01; test predictions saved for Steps 10–12)
- [x] **Step 10:** pooled vs per-city (per-city models tuned on their own city's data); PM2.5 slope per city
- [x] **Step 11:** spike classification (thresholds from out-of-fold training scores), class-weight check, tree early-warning bands
- [x] **Step 12:** permutation importance, partial dependence, linear coefficients, tree structure
- [ ] ~~Step 13~~ skipped (lean scope)
- [x] **Step 14 (report):** `reports/REPORT.md`, synthetic-data disclosure at the top
- [ ] **Step 14 (slides):** deferred by the user
- [x] `src/check_results.py` (passed before Step 9)
