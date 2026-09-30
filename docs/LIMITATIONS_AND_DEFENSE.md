# Limitations and Defense

Threats to validity and how each is handled, questions we expect, and ethical scope.

## Threats to Validity and Corner Cases

| Issue | How it is handled |
|---|---|
| **Synthetic data** | Disclosed on the first page of the README, report and slides. Conclusions are limited to model behaviour on this benchmark. No health claims. |
| **Invalid `date` column** | Dropped. No temporal features, lag features or Prophet. Independence is verified before random splitting. |
| **Redundant AQI** | Kept in the primary set (it was in the proposal) and ablated in FS-NoAQI. |
| **Data leakage** | Pipelines fit on training folds only. Test set used once. Spike threshold from training data. R² > 0.20 triggers an audit. |
| **Test-set overuse** | Model choice from CV only. One final test pass. |
| **City imbalance** (1,748 to 26,465 rows) | Stratified splits, per-city CIs, weighted and macro averages, small cities flagged. |
| **Multiple comparisons** | Holm correction across cities and models. |
| **Tiny effect sizes** | Equivalence margin and tie-breaking rule. Significance is not confused with practical importance. |
| **Count outcome, zeros, overdispersion** | Poisson GLM as a secondary model, Poisson deviance reported, predictions clipped at 0. |
| **Outliers** | PM2.5 up to 109.9 is kept (no evidence of error). Winsorised sensitivity check. |
| **Class imbalance in spike task** | Stratification, PR-AUC, CV-chosen threshold, class-weight sensitivity check. |
| **Decision tree over-fitting** | Depth and pruning tuned by CV. Compare train and CV error. |
| **SVM cost** | 20,000-row cap with learning-curve check. |
| **Neural network variance** | 5 seeds, mean ± SD, learning curves, small architecture. |
| **Correlated features distort importance** | VIF check and group-level interpretation. |
| **`population_density` random per row** | Excluded from the primary set. |
| **`hospital_capacity` as a predictor** | Excluded from the primary set (supply-side variable). Sensitivity only. |
| **Real reference dataset quality (optional Step 2B)** | Chosen by fixed criteria. Used for context only. Not merged with the main data and not used for prediction. |
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
A model is called best only if it beats the others by more than a pre-set margin with a Holm-adjusted CI. Otherwise the result is a tie and the simpler model is preferred.

## Ethics and Responsible Use

- The models are **not** for clinical, operational or policy use. They would need real hospital and monitoring data, temporal validation and expert review first.
- No personal or patient-level data is involved. The main dataset is aggregate and synthetic.
- Results must not be quoted as evidence about air pollution in any real city.
- Cite the dataset sources and respect their licenses.
