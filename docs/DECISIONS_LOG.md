# Decisions Log

Every choice that affects the results: what we chose, why, what it changed, and where the evidence is. Every number here can be checked in the file named in the **Evidence** column. The run logs in `results/logs/` give the date and time of every FULL run.

Status on 2026-10-01: Steps 0–7 and 9–12 are done (FULL). Step 14: the report is in `reports/REPORT.md`; slides deferred.

---

## Data and setup

### D1. Scripts plus a FAST/FULL mode switch
- **Choice:** Python scripts in `src/`, not notebooks. Every script has a FAST mode (small sample, writes only to `results_fast/` with `_FAST` in each file name) and a FULL mode (all rows, writes to `results/`). Every table has a `mode` column.
- **Why:** reproducibility. A debug run can never be mistaken for a reported result.
- **Effect on results:** none directly. It guarantees that every number in `results/` came from a FULL run.
- **Evidence:** `src/config.py`; the `mode` column in every table in `results/tables/`.

### D2. Lean scope
- **Choice:** two feature sets only: FS-Primary (7 environmental variables + city) and FS-PM25 (PM2.5 alone). Skipped: repeated CV, robustness checks, Holm correction, leave-one-city-out, Poisson GLM, VIF, and the real-data check (Step 2B).
- **Why:** course time budget. FS-PM25 against FS-Primary is the direct test of H1.
- **Effect on results:** fewer ablations. The H1 comparison is unaffected.
- **Evidence:** [PROJECT_RULES.md](PROJECT_RULES.md) "Scope"; `FEATURE_SETS` in `src/config.py`.

### D3. Random 80/20 split, stratified by city × spike
- **Choice:** one split, seed 42: 70,791 train rows and 17,698 test rows.
- **Why:** a random split is only valid if rows are independent. The audit (D5) shows they are: every date is unique, and lag-1 autocorrelation is about 0 in every city. If the rows formed a real time series, this split would leak future information into training.
- **Caveat (for honesty):** the strata used a provisional 90th-percentile cutoff computed on all rows. This only decides which rows go to which split; the spike threshold itself (D4) comes from training rows only.
- **Evidence:** `data/processed/split_indices.json`; `src/prepare.py`; `src/test_split.py` (checks that no row is in both splits).

### D4. Spike threshold from training rows only
- **Choice:** a spike is admissions ≥ 13, the 90th percentile of the training set. Prevalence is 11.7%. Every later script loads 13 from the stored file; none recomputes it.
- **Why:** computing a threshold on test rows leaks test information.
- **Fix made:** the first SVM script recomputed the threshold on its own subsample. It now loads the stored value.
- **Evidence:** `results/tables/03_spike_threshold.json`; the `spike_prevalence` column in `06_baselines.csv`.

### D5. Completing the audit (Step 2)
- **Choice:** added distributions of every numeric variable, a PM2.5-vs-admissions plot, a PM2.5 decile table with 95% CIs, per-city correlations and date-gap analysis. VIF was skipped (lean scope).
- **Findings:**
  - PM2.5 has r = 0.392 with admissions (Spearman 0.384). Every other variable has |r| < 0.005.
  - The per-city PM2.5 correlation ranges from 0.36 to 0.42.
  - Mean admissions rise steadily across PM2.5 deciles, from 5.56 to 10.64.
  - All 88,489 dates are unique, from 2020-01-01 to 2262-04-10: exactly one row per calendar day across the whole file, with cities scattered over those days. There is no time series.
- **Effect on results:** this explains the ceiling every model hits. One linear signal with r = 0.39 means R² ≈ 0.154 at best.
- **Evidence:** `results/tables/02_*.csv`, `results/figures/02_*.png`.

### D6. Baselines scored with CV on the same folds
- **Choice:** the baselines are scored by 5-fold CV on training rows only, using the same city × spike folds (seed 42) as every tuned model.
- **Why:** the first version saved only descriptive numbers (the mean, the PM2.5 slope), so there was nothing to compare models against. Using the same folds makes the comparisons paired and fair.
- **Result:**

  | Baseline | CV RMSE | CV R² | PR-AUC |
  |---|---|---|---|
  | Mean | 3.716 | 0.000 | – |
  | City mean | 3.716 | 0.000 | – |
  | PM2.5 linear | 3.417 | 0.154 | – |
  | Prevalence | – | – | 0.117 (ROC-AUC 0.5) |

- **Evidence:** `results/tables/06_baselines.csv`.

### D12. Tie rule and selection metrics were fixed before any results
- **Choice:** RMSE differences under 0.05 count as ties. Regression is selected on CV RMSE; classification on CV PR-AUC.
- **Why:** set in [PROJECT_RULES.md](PROJECT_RULES.md) and METHODOLOGY before any model was trained, so they can't have been chosen to favour a result. PR-AUC is the standard metric for a rare positive class (11.7%) because it focuses on the positive class, where ROC-AUC can look optimistic.
- **Note:** 0.05 is a project convention, not a statistical test. Step 9 adds bootstrap confidence intervals for a formal check.

---

## SVM (Step 7)

### D7. Kernel SVM on a subsample with a two-phase search
- **Choice:**
  - **Phase A:** 40 random candidates, 5-fold CV, on 10k training rows.
  - **Phase B:** 20 candidates in a narrower window around the Phase A best, 5-fold CV, on 20k rows.
  - Then a learning curve at 5k, 10k, 20k and 40k rows, and a final fit.
- **Why:**
  - Kernel SVM training time grows faster than the square of the row count (LIBSVM, Chang & Lin 2011). Measured on this laptop: time ≈ n^1.9 for SVR and n^2.1 for SVC. Tuning on all 70k rows is not feasible.
  - A coarse log-scale search followed by a finer one is the recommended SVM procedure (Hsu, Chang & Lin, *A Practical Guide to Support Vector Classification*).
  - Random search is more efficient than a grid for the same budget (Bergstra & Bengio 2012).
- **Override (approved by the user):** the original compute rule said n_iter=10, cv=3 on 5k–8k rows (history in [PROJECT_RULES.md](PROJECT_RULES.md)). The user asked for a thorough search, so the budget above replaced it.
- **Evidence:** `src/tune_svm.py`; `results/tables/07_svm_phaseA_cv.csv` and `07_svm_phaseB_cv.csv`; `results/logs/tune_svm.progress.txt`.

### D8. SVM tuning rules brought in line with the literature
Every change was made before the FULL run whose results we report:

| Rule | Before | After | Basis |
|---|---|---|---|
| Learning curve | Refit once per metric (3× the work) | One fit per fold and size; all metrics from it | Efficiency only; no effect on results |
| Refit after search | Extra unused refit | `refit=False` | Efficiency only |
| gamma = "scale" in Phase B | Searched the full range again | Kept at "scale" | Consistency with coarse-to-fine |
| Epsilon range | 0.05–2.0, no stated basis; Phase B could drift past it without a record | 0.05–3.5. The top ≈ one noise SD (3.417 = PM2.5 baseline CV RMSE). Widened past the range only through the recorded edge rule. | Cherkassky & Ma (2004): epsilon should scale with the noise level |
| Epsilon reference | none | Cherkassky & Ma value 3σ√(ln n / n) logged (≈ 0.25). Reference only, never used for selection. | Cherkassky & Ma (2004) |
| When to fit on more rows | Gain ≥ 0.01 RMSE / 0.005 PR-AUC (made-up numbers) | Gain from 20k→40k must exceed the fold-to-fold SD, i.e. be larger than CV noise | Standard: a difference smaller than CV variability can't be distinguished from noise |
| Edge rule | C and gamma only | C, gamma and epsilon, each note naming its parameter | METHODOLOGY 7.2: "if best is on a grid edge, extend once and record" |

- **Conventions, not literature:** the 10%-of-range edge zone, the ×10 widening (×1.5 for epsilon) and the ×/÷4 Phase B window. All were fixed before the FULL run.

### D9. The first FULL SVM run was stopped and relaunched
- **What happened:** the first FULL run was stopped during its first search. It had written no results. The epsilon fix (D8) was applied and FULL was relaunched.
- **Why it matters for defensibility:** no FULL result had been seen when the rule changed. The change was prompted by an inconsistency (epsilon could pass its range without a record), which a FAST test run exposed. FAST numbers are never reported.
- **Evidence:** `results/logs/tune_svm.progress.txt` starts at 01:35 (the relaunch); the earlier, stopped run left no results.

### D10. Reporting fair CV scores, not tuning scores
- **Choice:** for model comparison we use the SVM's learning-curve scores (chosen settings held fixed, 3-fold CV on the full training set, at 20k rows), not its Phase B tuning scores.
- **Why:**
  - A tuning score is the best of about 60 candidates scored on the same folds. Taking the maximum of noisy scores is optimistic ("selection bias"; Cawley & Talbot 2010).
  - The SVM tuning scores were also measured on the 20k-row SVM sample, not the full-training folds the other models used.
- **Proof it matters:** with PM2.5 as the only feature, any model whose score only rises with PM2.5 ranks rows identically, so all such models must get the same PR-AUC.
  - The SVC tuning score was 0.251. Its fair score is 0.239, and Logistic gets 0.240.
  - The 0.251 was selection noise.
- **Step 9 settles it:** the test set is used once, with no selection, so its numbers carry no such bias.
- **Evidence:** `07_svm_best.csv` (`cv_*` columns = tuning scores) and `07_svm_learning_curve.csv` (`train_size = 20000` = fair scores).

**SVM results (FULL):**

| | Tuning score | Fair score | Linear/Logistic | PM2.5 baseline |
|---|---|---|---|---|
| SVR FS-Primary, RMSE | 3.411 | 3.419 | 3.418 | 3.417 |
| SVR FS-PM25, RMSE | 3.410 | 3.417 | 3.417 | 3.417 |
| SVC FS-Primary, PR-AUC | 0.251 | 0.239 | 0.240 | 0.117 |
| SVC FS-PM25, PR-AUC | 0.251 | 0.239 | 0.240 | 0.117 |

- **Learning curve:** flat. From 5k to 40k rows the scores change by at most 0.004 RMSE and 0.004 PR-AUC, while the fold-to-fold SD is about 0.017 and 0.005. All final models use 20k rows.
- **Chosen settings:**
  - Both SVRs use a small gamma (0.001–0.005), so the model is close to a straight line.
  - The FS-Primary SVC has C = 0.006; 19,991 of its 20,000 training rows are support vectors, the textbook sign that the classes overlap heavily.
  - Epsilon: CV picked 3.40 and 2.73, but RMSE barely changes with epsilon (Phase A, FS-PM25: best 3.398, median 3.403).
- **Range widenings (recorded):** FS-Primary SVC C (low); FS-PM25 SVR epsilon (high); FS-PM25 SVC C (high) and gamma (low).

---

## Linear, Logistic and Decision Tree (Step 7)

### D11. Linear/Tree tuning brought in line with the literature
| Problem | Fix | Basis |
|---|---|---|
| Ridge and Logistic ran on unscaled inputs. The L2 penalty then shrinks each coefficient differently depending on its units, which also caused Logistic convergence warnings. | StandardScaler on numeric columns inside the pipeline. Trees get no scaler. | Hoerl & Kennard 1970; Hastie, Tibshirani & Friedman, *ESL* §3.4 |
| Ridge alpha was searched over 1e-5 to 10, not the planned grid, and the best value (6.6) sat near the edge without widening | Full grid over alpha {0, 0.01, 0.1, 1, 10}, widened once if the best is 10. Logistic C over {1e-3 … 1e3}, same rule. | METHODOLOGY 7.2 |
| Tree `ccp_alpha` range (0 to 0.01) had no connection to the data's scale | Candidates taken from `cost_complexity_pruning_path` on training rows, crossed with a full depth × leaf-size grid | Breiman et al. 1984 (CART) |
| Threshold was recomputed | Loaded from `03_spike_threshold.json` | D4 |
| FAST mode wrote to `results/` | Now writes only to `results_fast/` | D1 |

- **Edge rule for the tree (added after the first FULL run):** the first FULL run showed the classifier's best `min_samples_leaf` at the grid maximum (200), with PR-AUC still rising, and the edge rule had only been coded for alpha and C.
  - Added the same rule for the tree: leaf size widened once to {500, 1000, 2000}; depth to 1 if the best were 2.
  - Natural bounds need no widening: `min_samples_leaf=1`, `ccp_alpha=0` and `max_depth=None`. The largest pruning candidate already prunes nearly to a single split.
  - After widening, the optimum is inside the range (500), and PR-AUC falls by 2000.
  - This is the rule fixed in METHODOLOGY 7.2 before any results, now applied consistently. Larger leaves help ranking because each leaf's spike rate is estimated from more rows (Provost & Domingos 2003).
- **Status:** done. The final FULL run (08:41–09:32) had no errors and 0 convergence warnings.
- **Final Linear/Tree CV results** (FS-Primary / FS-PM25):
  - RMSE: Linear 3.4176 / 3.4171; Tree 3.4077 / 3.4075. A tie: the gap is 0.010, below 0.05.
  - PR-AUC: Logistic 0.2396 / 0.2396; Tree 0.2274 / 0.2323 (before the fix: 0.219 / 0.217).
- **Collinear city columns (dummy variable trap): checked, harmless.**
  - With alpha = 0 on FS-Primary, the 8 one-hot city columns plus the intercept are collinear, which caused the `LinAlgWarning` in the error log.
  - RMSE is flat for every alpha from 0 to 1000 (3.41759–3.41763), so predictions were unaffected. The "best" alpha of 300 wins by 0.00004, which is noise.
  - No fix needed. (`convergence_warnings` counts only ConvergenceWarnings, not this one.)
- **Logistic with PM2.5 alone:** PR-AUC is identical (0.23959) at every C, as theory predicts. With one feature, every positive slope ranks the rows the same way. The "C_low" note just reflects how ties are broken.

---

## MLP (Step 7)

### D13. MLP design
- **Choice:**
  - sklearn MLP with standardised inputs and the Adam optimiser in batches of 256, for up to 200 passes over the data.
  - Random search over the METHODOLOGY network sizes, the L2 penalty `alpha` (1e-6 to 0.1) and the learning rate. 20 candidates for regression and 30 for classification, 5 folds; each range widened once if its best value sits at an edge.
  - The chosen settings re-run with 5 seeds (mean ± SD), then final fits on all training rows with 5 seeds.
- **When training stops:**
  - **Regressor:** validation early stopping (Goodfellow et al. 2016, §7.8).
  - **Classifier:** sklearn's classifier early stopping watches validation *accuracy*, which can't guide training at 11.7% prevalence. It therefore stops on training log-loss, sklearn's default, with L2 tuned by CV.
- **Result:**
  - 5-seed CV RMSE 3.4232 / 3.4155, PR-AUC 0.2352 / 0.2396. The spread across seeds is 0.0011 or less.
  - Final models stopped after 25–65 passes.
  - The learning-rate range was widened once in all 4 searches. The FS-Primary classifier's best rate (0.01) sits at the widened edge; the extend-once rule allows no further widening. The possible gain is at most about 0.004, because FS-PM25 already reaches the 0.2396 ceiling.
- **Evidence:** `07_mlp_best.csv`, `07_mlp_cv_results.csv`, `07_mlp_loss_curves.csv`, `07_mlp_loss_curves.png`.

## Final test evaluation (Step 9)

### D14. The test set was used once, after every setting was fixed
- **Before running:**
  - `check_results.py` had to pass, so no FAST data could reach the report.
  - The script was dry-run on training rows only, which caught a bug where the tree and MLP regression results were overwritten by their classifiers. It was fixed before the real run.
- **What the run did:**
  - Refitted Linear/Logistic/Tree with their stored settings on all training rows. Loaded the SVM models (20k rows) and the MLP models (5 seeds).
  - Predicted the 17,698 test rows once, clipping regression predictions at 0.
  - 1,000 paired bootstrap resamples for every CI and difference.
  - Saved the test predictions so Steps 10–12 reuse them instead of predicting again.
- **Verdict rule (fixed in advance):** two models differ only if the RMSE gap is at least 0.05 **and** the paired CI excludes 0. PR-AUC has no pre-set margin, and none was added after seeing results.
- **Results:**
  - **Regression:** every model ties with the PM2.5-only baseline (3.4193); no leakage alarm (highest R² 0.157).
  - **The tree is the only detectable difference:** 0.011 lower RMSE than Linear, CI [−0.015, −0.006]. The same gap appeared in CV (0.010), so it is real, probably a slight non-linearity the steps capture (see the decile plot). It is 5× smaller than the tie margin, so it has no practical weight.
  - **H1 holds:** FS-PM25 ties with FS-Primary for every regression model. The tree's predictions are identical for both feature sets: it never splits on anything but PM2.5.
  - **Classification:** Logistic, SVC and the PM2.5-only MLP get the same PR-AUC (0.249), as predicted: any model whose score rises with PM2.5 ranks the rows the same way. The tree is slightly lower (0.238–0.244). Every model reaches about 2.1× the prevalence baseline (0.117).
- **Evidence:** `results/tables/09_test_*.csv`, `09_test_run.json`, `results/figures/09_*.png`, `results/logs/final_evaluation.*.txt`.

### D15. Why the tree is slightly better: the data's PM2.5 effect is a staircase (found after Step 9, training data only)
- **What we checked:** the tree's small but real RMSE advantage (−0.011 on test, the same in CV) needed an explanation, not an assumption. We compared shapes for the PM2.5–admissions relationship with 5-fold CV on training rows only.
- **Finding:**
  - Mean admissions ≈ 5.05 + 0.99 × floor(PM2.5 / 10): about +1 admission per full 10 µg/m³ band, and flat inside each band.
  - This one feature reaches CV RMSE 3.4065, against 3.4171 for a straight line, matching the tree (3.4077).
  - Adding a straight-line term on top of it adds nothing.
  - Only a band width of 10 works: widths of 8, 12 and 15 give 3.423–3.438.
  - The depth-4 tree places its splits at 10.05, 19.95, 29.95, 39.95, 49.95 and 59.95.
  - Mean admissions in 20 PM2.5 bins swing above and below the straight line by about ±0.25 (4–5 standard errors).
- **What it means:**
  - **For the tree:** trees represent step functions exactly and smooth functions only roughly (Breiman et al. 1984; Hastie et al., *ESL* §9.2), so a tree should win here, and only by the small amount the steps add over a line. Its advantage is genuine, explained, and still 5× below the tie margin.
  - **For the data:** real concentration–response curves are smooth and roughly linear with no threshold (WHO 2021 air quality guidelines). Steps at round numbers are another sign the data was generated by a script.
- **What we did not do:**
  - We did not change any model or retest. Building a staircase model now and scoring it on the test set would reuse the test set and make the result look better than it is.
  - This is reported as an explanation found on training data. Step 12's partial dependence plots should show the steps directly.
- **Evidence:** reproducible from the CV on training rows described above; figure `02_pm25_deciles.png` (wiggles) and the tree splits.

## Steps 10–12

### D16. Training-only parts and test parts were kept apart
- **Choice:** everything that needs no test data runs in `analysis_train.py` on training rows only:
  - city slopes
  - per-city tuning
  - spike thresholds
  - partial dependence
  - coefficients
  - the tree diagram

  Only the three parts that need test predictions run inside `final_evaluation.py` (PROJECT_RULES.md: the test set only in the final script):
  - pooled vs per-city
  - thresholded spike metrics
  - permutation importance

  Step 9 was not recomputed; the thresholded metrics reuse the saved Step 9 test scores.
- **Why:** every setting and threshold was fixed before the test was touched, so nothing on the test set could steer a choice.
- **Dry runs** on training rows caught two problems before any test use:
  1. **sklearn API:** the MLP wrapper needed a `fit` method.
  2. **A fairness bias:** per-city models were reusing the pooled tree's settings, including a minimum leaf size of 1. On one city's data that overfits, which unfairly penalised per-city models (+0.14 to +0.50 RMSE in the dry run). Fixed by tuning each city's model by CV on that city's own training rows (`10_city_params.csv`; chosen depths 2–3, alpha 10–30, all inside the grid).

### D17. Step 10–12 choices and findings
- **Pooled vs per-city:** done for Linear and for the regression model that was best in **CV** (the Decision Tree), so the choice did not use the test set.
  - Per-city models never beat the pooled one. Linear: weighted Δ +0.0005 [−0.0007, 0.0017]. Tree: +0.0065 [0.0031, 0.0102]. Both ties.
  - The cities share one relationship, so pooling simply gives each model more data.
- **City slopes:** OLS `admissions ~ pm2_5 * city` on training rows, with HC3 robust SEs because count variance grows with the mean (variance/mean = 1.7).
  - Wald χ²(7) = 15.7, p = 0.028; London vs Delhi +0.0075, p = 0.028. Both unadjusted; Holm was dropped.
  - Read as small, detectable differences (slopes 0.090–0.106) that may partly come from fitting a line to a staircase (D15). They don't help prediction.
  - **Correction made:** the first output labelled the χ² statistic as "F". With robust covariance statsmodels reports a Wald χ² test, and the output was regenerated with the right label.
- **Spike thresholds:** F1-maximising threshold on out-of-fold training scores (5 folds; 5 seeds averaged for the MLP; the SVM's own 20k-row sample for the SVC), then applied once to the test scores.
  - Test F1 ≈ 0.31 for every model, against 0.21 for flagging every day.
  - The SVC flags more days (46%) because its raw decision-function scale shifts between the 16k-row CV models and the 20k-row final model. That's a known weakness of thresholding raw SVM scores without Platt scaling, which the compute rules exclude (PROJECT_RULES.md).
  - Class weighting changes nothing (F1 0.317 vs 0.314).
- **Interpretation:**
  - **Permutation importance (test):** PM2.5 adds 0.59 RMSE when shuffled; every other input adds about 0.
  - **The tree splits only on PM2.5.**
  - **Standardised OLS:** only PM2.5 is clearly non-zero (1.46 per SD). One city term passes at 5% out of 13 tests, about what chance gives.
  - **Partial dependence:** shows the staircase (D15). The tree follows it; Linear, SVR and MLP draw the same straight line.
- **Evidence:** `results/tables/10_*`, `11_*`, `12_*`; figures `10_city_slopes.png`, `12_*.png`; logs `analysis_train*.txt`, `final_evaluation_steps10_12.*.txt`.

---

## Where each claim is recorded
| Claim | File |
|---|---|
| Row counts and validation | `results/tables/01_validation.csv` |
| PM2.5 is the only signal | `02_target_correlations.csv`, `02_pm25_deciles.csv`, `02_pm25_vs_admissions.png` |
| No time series | `02_date_gaps.csv`, `02_lag_autocorrelation.csv` |
| Cities don't differ | `02_city_means.csv`; the city-mean baseline in `06_baselines.csv` |
| Split and threshold | `data/processed/split_indices.json`, `03_spike_threshold.json` |
| Baselines | `06_baselines.csv` |
| Linear/Tree CV | `07_best_linear_tree.csv`, `07_cv_results_linear_tree.csv` |
| SVM CV, learning curve, settings | `07_svm_best.csv`, `07_svm_learning_curve.csv`, `07_svm_phase*_cv.csv` |
| MLP CV, seeds, loss curves | `07_mlp_best.csv`, `07_mlp_cv_results.csv`, `07_mlp_loss_curves.csv` |
| Final test results (Step 9) | `09_test_regression.csv`, `09_test_classification.csv`, `09_test_paired_*.csv`, `09_test_h1.csv` |
| Run history | `results/logs/*.txt` |
