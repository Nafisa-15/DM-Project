# Results Tables and Checklist

Empty templates to fill during the project.

## Results Tables

**Table 1. Test-set performance, FS-Primary (95% bootstrap CI)**

| Model | RMSE | MAE | R² | Poisson deviance |
|---|---|---|---|---|
| Mean predictor | | | | |
| PM2.5-only linear | | | | |
| Linear Regression | | | | |
| Poisson GLM | | | | |
| Decision Tree | | | | |
| SVM (SVR) | | | | |
| MLP (mean ± SD, 5 seeds) | | | | |
| Random Forest (optional) | | | | |

**Table 2. Paired RMSE differences vs Linear Regression** (Holm-adjusted)

| Model | ΔRMSE | 95% CI | Verdict (better / equivalent / worse) |
|---|---|---|---|

**Table 3. Feature-set ablation** (RMSE per model × feature set)

| Model | FS-Primary | FS-PM25 | FS-NoAQI | FS-Pollutants | FS-Extended |
|---|---|---|---|---|---|

**Table 4. Per-city results** (pooled vs per-city vs leave-one-city-out)

| City | n (test) | Pooled RMSE | Per-city RMSE | LOCO RMSE |
|---|---|---|---|---|

**Table 5. PM2.5 slope by city**

| City | Slope | 95% CI | Holm-adjusted p (vs pooled) |
|---|---|---|---|

Include a Delhi vs London row.

**Table 6. Spike classification**

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Balanced accuracy |
|---|---|---|---|---|---|---|
| Prevalence baseline | | | | | | |
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| SVM | | | | | | |
| MLP | | | | | | |

**Table 7. Permutation importance** (best model and MLP)

| Feature | Best model | MLP |
|---|---|---|

**Table 8 (optional). Real-data reference check.** See [DATA_AUDIT.md](DATA_AUDIT.md).

**Figures:** target distribution; correlation heatmap; admissions by PM2.5 decile; predicted vs actual; residuals; learning curves (SVM, MLP); per-city slope forest plot; PR curves; decision-tree diagram; partial dependence of PM2.5; (optional) real vs synthetic PM2.5 distribution.

## Project Checklist

- [ ] **Step 0:** repository, environment, seed
- [ ] **Step 1:** loader with validation assertions passing
- [ ] **Step 2:** audit notebook reproduces every number in DATA_AUDIT.md
- [ ] **Step 2B (optional):** real reference dataset chosen by the criteria, comparison table filled
- [ ] **Step 3:** target and spike threshold defined (train only)
- [ ] **Step 4:** feature sets fixed in `config.py`
- [ ] **Step 5:** stratified split saved, leakage unit test passing
- [ ] **Step 6:** baselines evaluated
- [ ] **Step 7:** models tuned, SVM learning-curve check, MLP seeds, ablations
- [ ] **Step 8:** repeated CV comparison
- [ ] **Step 9:** single test-set evaluation with bootstrap CIs
- [ ] **Step 10:** per-city, interaction test, leave-one-city-out
- [ ] **Step 11:** spike classification
- [ ] **Step 12:** permutation importance, partial dependence
- [ ] **Step 13:** robustness checks
- [ ] **Step 14:** report and slides with the synthetic-data disclosure on the first page
