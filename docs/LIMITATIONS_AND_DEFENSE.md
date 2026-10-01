# Limitations and Defense

Threats to validity and how each is handled, questions we expect, and ethical scope.

> **Status: written as a plan before any code ran; updated 2026-10-01 to match the lean scope.** Rows marked **Not done (lean scope)** were planned but dropped. They are now limitations to state openly, not safeguards we claim. General questions about the data and design are answered here. Questions about the specific choices we made (SVM subsample, epsilon, tuning scores vs fair scores) are in [DEFENSE_QA.md](DEFENSE_QA.md).

## Threats to Validity and Corner Cases

| Issue | How it is handled |
|---|---|
| **Synthetic data** | Disclosed on the first page of the README, report and slides. Conclusions are limited to model behaviour on this benchmark. No health claims. |
| **Invalid `date` column** | Dropped. No temporal features, lag features or Prophet. Independence is verified before random splitting. |
| **Redundant AQI** | Kept in the primary set (it was in the proposal). The FS-NoAQI ablation is **Not done (lean scope)**. FS-PM25 vs FS-Primary shows that AQI and the other extra features add nothing. |
| **Data leakage** | Pipelines fit on training folds only. Test set used once. Spike threshold from training data. R² > 0.20 triggers an audit. |
| **Test-set overuse** | Model choice from CV only. One final test pass. |
| **City imbalance** (1,748 to 26,465 rows) | Stratified splits, per-city CIs, weighted and macro averages, small cities flagged. Leave-one-city-out is **Not done (lean scope)**. |
| **Multiple comparisons** | Holm correction is **Not done (lean scope)**. Per-city tests are reported with unadjusted CIs and read cautiously; with 8 cities, about one false positive at 5% would be unsurprising. |
| **Tiny effect sizes** | Pre-set tie rule (RMSE gap < 0.05, [PROJECT_RULES.md](PROJECT_RULES.md)) plus bootstrap CIs in Step 9. Significance is not confused with practical importance. |
| **Count outcome, zeros, overdispersion** | Predictions clipped at 0; Poisson deviance reported in Step 9. The Poisson GLM is **Not done (lean scope)**. |
| **Outliers** | PM2.5 up to 109.9 is kept (no evidence of error). The winsorised check (Step 13) is **Not done (lean scope)**. |
| **Class imbalance in spike task** | Stratification, PR-AUC, CV-chosen threshold, class-weight sensitivity check. |
| **Decision tree over-fitting** | Depth and pruning tuned by CV. Compare train and CV error. |
| **SVM cost** | 20,000-row cap. The learning curve at 5k–40k is flat (done, D7/D10 in DECISIONS_LOG.md). |
| **Tuning scores are optimistic** (best of many candidates) | Model comparison uses fixed-setting CV scores and the single test evaluation, not the best tuning score (D10). |
| **Neural network variance** | 5 seeds, mean ± SD, learning curves, small architecture. |
| **Correlated features distort importance** | VIF is **Not done (lean scope)**. The audit shows the features are nearly uncorrelated (AQI–PM2.5 r = 0.003), so this risk is low here. Importance is read at group level. |
| **`population_density` random per row** | Excluded from the primary set. |
| **`hospital_capacity` as a predictor** | Excluded from the primary set (supply-side variable). Sensitivity only. |
| **Real reference dataset (optional Step 2B)** | **Not done (lean scope).** The report says the Delhi-vs-London claim can't be tested with this file. |
| **Benchmark range vs real conditions** | Main-data PM2.5 stops at 109.9 µg/m³. Real Delhi may exceed this, so the models are not applied to real data. |
| **Ecological, non-causal design** | Framed as prediction only. No causal language. |
| **Software version drift** | Versions pinned, seeds fixed. |
| **Dataset changes on Kaggle** | Record the download date and SHA-256 hash in `data/README_DATA.md`. |

## Anticipated Questions

**Why is the project still valid with synthetic data?**
A data mining project is judged on its method: data auditing, leakage control, fair model comparison and honest interpretation. Synthetic data with a known structure is a legitimate benchmark if it is labelled and no real-world claims are made. Documenting that the data is synthetic is part of the analysis.

**Why not use Prophet or another time-series model?**
The `date` field is a row counter with a random city per row, and autocorrelation is about zero. There is no time series to model.

**Why is Delhi no longer described as "high pollution" in the main analysis?**
Because in the main dataset Delhi and London have the same average PM2.5 (35.1 vs 35.4). If the optional real-data check (Step 2B) is done, it shows real Delhi levels for context. Otherwise the report states that the claim cannot be tested with this file. Either way, the models are not applied to real data.

**Why not merge the real air quality data with the synthetic admissions?**
There is no shared key or date alignment, and admissions would be invented. That would create fake real-world evidence.

**Why are R² values so low?**
Only PM2.5 carries signal and it explains about 15% of the variance. The rest is noise from the data generation.

**Why does the deep learning model not win?**
The true relationship is close to a straight line in one variable. A flexible model has nothing extra to learn. Reporting this is the point of a fair comparison.

**Why a random split when the data has dates?**
The dates are not real and autocorrelation is zero, so rows are independent. Real time-series data would need a chronological split, and the report says so.

**Why was hospital capacity not used?**
It describes hospital supply, not environmental exposure, and it has no relationship with admissions here (correlation 0.002). It stays in the sensitivity analysis.

**How do you know the best model is really best?**
A model is called best only if it beats the others by more than the pre-set margin (RMSE gap ≥ 0.05) and the bootstrap CI of the paired difference (Step 9) excludes zero. Otherwise the result is a tie and the simpler model is preferred. (Holm adjustment was dropped in the lean scope.)

## Ethics and Responsible Use

- The models are **not** for clinical, operational or policy use. They would need real hospital and monitoring data, temporal validation and expert review first.
- No personal or patient-level data is involved. The main dataset is aggregate and synthetic.
- Results must not be quoted as evidence about air pollution in any real city.
- Cite the dataset sources and respect their licenses.
