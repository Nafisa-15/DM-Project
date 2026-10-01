# Model Justification

For each model: why it's in the study, what the literature predicts on data like ours, how it was tuned, and what we found. The decision numbers (D1, D2, …) point to [DECISIONS_LOG.md](DECISIONS_LOG.md).

## What the data allows (the lens for every result)
- **One signal:** admissions rise linearly with PM2.5 (r = 0.392). Every other variable has |r| < 0.005, and city adds nothing (city-mean baseline RMSE 3.716 = overall-mean baseline 3.716).
- **Ceiling:** when the signal is linear with r ≈ 0.39, the best any model can do is R² ≈ r² ≈ 0.154, i.e. RMSE ≈ 3.417 (the PM2.5-only linear baseline). No model, however flexible, can explain noise that is independent of the inputs.
- **So the expected result is a tie.** A flexible model can at most match the straight line; its extra flexibility can only fit noise, which cross-validation penalises. Seeing a tie is the correct outcome here, not a failure. A model that clearly beat the ceiling would point to leakage (METHODOLOGY Step 9 sanity alarm: test R² > 0.20).
- **The data is synthetic** (unique dates spanning 2020–2262, identical city profiles). Conclusions are about how models behave on this benchmark, not about real health effects.

---

## Baselines
- **Why they're needed:** a score means nothing without a reference. The mean predictor is zero skill; the PM2.5-only line is the strongest simple model and the bar every complex model must clear (METHODOLOGY Step 6).
- **Results (5-fold CV, train only):**
  - RMSE: mean 3.716, city mean 3.716, PM2.5 linear 3.417 (R² 0.154).
  - Spike prevalence PR-AUC: 0.117 (the PR-AUC of random guessing equals the prevalence).

## M1. Linear Regression (Ridge) and Logistic Regression
- **Why:** least squares is the best linear unbiased estimator under the Gauss–Markov assumptions, and the audit shows the signal is linear. This is the model that matches the data, so it's the reference.
- **Tuning basis:**
  - An L2 penalty (Hoerl & Kennard 1970) shrinks coefficients against noise features.
  - The penalty depends on each feature's units, so inputs are standardised inside the pipeline (Hastie, Tibshirani & Friedman, *ESL* §3.4).
  - Grids: alpha {0, 0.01, 0.1, 1, 10} and C {1e-3 … 1e3}, widened once if the best value sits at an edge (D11).
- **Expectation:** Ridge ≈ OLS here. With 70k rows and 15 inputs, shrinkage has little to correct.
- **Result:** CV RMSE 3.4176 (FS-Primary) and 3.4171 (FS-PM25): a tie with the PM2.5 baseline (3.417). Logistic PR-AUC is 0.2396 for both. No convergence warnings after scaling.
- **Theory confirmed:**
  - RMSE is flat across alpha from 0 to 1000, so shrinkage has nothing to correct.
  - With PM2.5 alone, PR-AUC is identical at every C, because any positive slope gives the same ranking.
  - The collinear city columns at alpha = 0 had no effect on predictions (D11).

## M2. Decision Tree (CART)
- **Why:** it is interpretable and can model non-linearity and interactions without being told about them (Breiman et al. 1984). METHODOLOGY also uses its rules as the early-warning output.
- **Tuning basis:**
  - Cost-complexity pruning is CART's own method for controlling overfitting.
  - Pruning candidates come from the training data's pruning path, crossed with a grid over depth and minimum leaf size (D11).
  - Trees are scale-invariant, so they need no scaling.
- **Expectation:** a tree approximates a straight line with steps. On a linear signal it can at best tie with linear regression, and deep trees fit noise.
- **Result:**
  - CV RMSE 3.4077 / 3.4075, a tie with Linear (gap 0.010 < 0.05). RMSE stops improving at depth 4.
  - The tree's slightly lower CV RMSE could be a small real curve or could come from picking the best of 384 grid points. Step 9 decides; we don't assume either.
- **Classifier:**
  - PR-AUC 0.2274 / 0.2323 with large leaves (500 rows), after the leaf-size range was widened (D11).
  - Still below Logistic (0.2396), as theory predicts: the ideal ranking rises smoothly with PM2.5, and a tree can only approximate it with steps. Larger leaves give more stable leaf rates and better ranking (Provost & Domingos 2003), but not a smooth curve.

## M3. Support Vector Machine (SVR / SVC, RBF kernel)
- **Why:** a margin-based, kernel method (Cortes & Vapnik 1995). The RBF kernel can represent smooth non-linear shapes, so it tests whether anything non-linear exists.
- **Tuning basis:**
  - A log-scale coarse-to-fine search over C and gamma (Hsu, Chang & Lin).
  - An epsilon range tied to the noise level (Cherkassky & Ma 2004).
  - Standardised inputs.
  - Training on a subsample, because the cost grows faster than the square of the row count (Chang & Lin 2011); a learning curve checks that the cap loses nothing (D7, D8).
- **Expectation:** if the truth is linear, CV should push the RBF kernel toward near-linear behaviour: a small gamma, which makes the kernel very wide.
- **Result:**
  - Both SVRs use a small gamma (0.001–0.005): near-linear, as predicted.
  - Fair CV scores (D10): RMSE 3.419 / 3.417 and PR-AUC 0.239 / 0.239. These are identical to Linear/Logistic.
  - The learning curve is flat from 5k to 40k rows, so the 20k cap is justified.
  - The FS-Primary SVC uses almost all its training points as support vectors (19,991 of 20,000), which is expected when the classes overlap heavily.
- **Why the tuning score (0.251) isn't reported:** see D10. It is the best of about 60 candidates and so is inflated by selection.

## M4. Neural network (MLP), done (`src/tune_mlp.py`)
**Result:**
- 5-seed CV RMSE 3.4232 (FS-Primary) / 3.4155 (FS-PM25); PR-AUC 0.2352 / 0.2396. Spread across seeds ≤ 0.0011.
- Test RMSE 3.4245 / 3.4171: a tie.
- The MLP is slightly worse with all features than with PM2.5 alone (+0.007 RMSE), a small cost of fitting the 7 noise inputs.
- It rediscovered the straight line: its decile curve sits on Linear's (`09_calibration_pm25_deciles.png`). That was the expectation, confirmed.

Design:
Design, grounded in the literature so we don't assume answers:
- **Setup:** sklearn `MLPRegressor` / `MLPClassifier`, standardised inputs, the Adam optimiser in batches of 256, up to 200 passes over the data, and an L2 penalty (`alpha`, 1e-6 to 0.1) in place of dropout, which sklearn lacks (Goodfellow, Bengio & Courville 2016, ch. 7–8). Random search with 20 candidates for regression and 30 for classification, 5-fold CV on the same folds, edge rule applied once.
- **When training stops:**
  - **Regressor:** early stopping on a 10% validation split from the training fold, the standard approach (§7.8).
  - **Classifier:** sklearn's classifier early stopping watches validation *accuracy*, which can't guide training at 11.7% prevalence (always predicting "no spike" already scores 88%). The classifier therefore stops on training log-loss (sklearn's default), with CV-tuned L2 controlling overfitting.
  - The largest classifier network can hit the 200-pass cap, which raises sklearn "maximum iterations" warnings. That is the plan's cap, not an error.
- **Size:** small networks (METHODOLOGY 7.2 hidden layers {(32), (64,32), (128,64,32)}). One signal and 70k rows don't need a large network; a larger one only has more room to fit noise.
- **Stability:** the MLP is stochastic, so the final settings are trained with 5 seeds and reported as mean ± SD (METHODOLOGY 7.2). One lucky seed isn't a result.
- **Training rows:** MLP cost grows about linearly with the row count, so, unlike the SVM, it can use all 70,791 training rows.
- **Expectation:** it should rediscover the straight line, landing near RMSE 3.42. If it clearly beats the others, check for leakage first, before celebrating.

---

## H1: answered on the test set (Step 9)
H1: PM2.5 alone predicts about as well as all features. **Supported.**
- On the test set (1,000 paired bootstrap resamples), FS-PM25 ties with FS-Primary for every regression model. The largest gap is +0.007 RMSE (MLP), 7× below the 0.05 margin.
- The tree splits only on PM2.5, so its predictions are identical for both feature sets.
- For spike ranking, the extra features never help. Logistic, SVC and MLP are unchanged; the tree gets slightly worse with them (−0.006 PR-AUC).
- All test RMSEs (3.409–3.425) sit at the predicted ceiling (R² ≈ 0.15). No model escapes it, and no leakage alarm fired. Full tables: [RESULTS_AND_CHECKLIST.md](RESULTS_AND_CHECKLIST.md).
