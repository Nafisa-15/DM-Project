# Data Audit

> **Status: verified on 2026-10-01.** This file was written before the pipeline existed. Every number below was then recomputed from the CSV, and all of them match. They are reproduced by `src/audit.py` (outputs: `results/tables/02_*.csv`, `results/figures/02_*.png`). The city PM2.5 slopes below use all rows; the formal per-city analysis, using training data and CIs, is METHODOLOGY Step 10.

## Source

- **Name:** Global Air Quality and Respiratory Health Outcomes Dataset
- **Host:** Kaggle, https://www.kaggle.com/datasets/tfisthis/global-air-quality-and-respiratory-health-outcomes
- **License and terms:** check the dataset page and copy the stated license into `data/README_DATA.md` before publishing the repository.

## Schema

88,489 rows, 12 columns, no missing values, no duplicate rows.

| Column | Type | Description | Role |
|---|---|---|---|
| `city` | categorical (8) | Delhi, Beijing, Mexico City, Los Angeles, London, Tokyo, Cairo, São Paulo | Grouping variable, one-hot feature in pooled models |
| `date` | string | Sequential daily counter | **Dropped** |
| `aqi` | int | Air Quality Index, 0–499 | Feature |
| `pm2_5` | float | PM2.5 (µg/m³), 0–109.9 | Feature |
| `pm10` | float | PM10 (µg/m³) | Feature |
| `no2` | float | NO₂ | Feature |
| `o3` | float | O₃ | Feature |
| `temperature` | float | °C, −5 to 40 | Feature |
| `humidity` | int | %, 20–94 | Feature |
| `population_density` | categorical | Rural / Suburban / Urban | Sensitivity analysis only |
| `hospital_capacity` | int | 50–1,999 | Sensitivity analysis only |
| `hospital_admissions` | int | Daily admissions, 0–25 | **Target** |

## Findings: why the data is treated as synthetic

| Finding | Evidence | Consequence |
|---|---|---|
| `date` is a row counter | Consecutive rows are exactly 1 day apart for all 88,488 steps, from 2020-01-01 to 2262-04-10, with a random city per row. Within one city, rows are 3 to 50 days apart on average (Delhi 3.3, São Paulo 50.5). | No time series. Prophet and lag features are excluded. |
| No temporal dependence | Lag-1 autocorrelation of admissions is −0.003 in row order and between −0.013 and 0.045 within each city. PM2.5 lag-1 is −0.001. | Rows are treated as independent, which justifies random stratified splitting. |
| Cities are statistically identical | Mean PM2.5 is 35.11 (Delhi) vs 35.41 (London). Mean AQI is about 249 in every city. Mean admissions are about 8.0 in every city. | The "high vs low pollution" framing is removed. |
| Only PM2.5 relates to admissions | Correlation with admissions: PM2.5 0.392. AQI, PM10, NO₂, O₃, temperature, humidity and capacity lie between −0.004 and 0.002. | Low performance ceiling (R² ≈ 0.15). |
| Pollutants are mutually independent | AQI–PM2.5 correlation is 0.003. AQI is uniform over 0–499. In real data AQI is calculated from pollutant concentrations. | AQI is kept as a proposal feature and ablated. |
| Effect is linear and consistent | Mean admissions rise from 5.6 (lowest PM2.5 decile) to 10.6 (highest). Fitted slope: 0.0987 admissions per µg/m³ (about +1 admission per +10 µg/m³). City slopes range from 0.090 to 0.106. Delhi is 0.0986, London 0.1056. | Supports H2 and H3. |
| The PM2.5 effect is a staircase, not a line (found 2026-10-01, training data) | Mean admissions ≈ 5.05 + 0.99 × floor(PM2.5/10): flat inside each 10 µg/m³ band, about +1 per band. It beats a straight line in CV (RMSE 3.4065 vs 3.4171); a tree splits exactly at 10, 20, …, 60. | Explains the tree's small, real advantage. Steps at round numbers are another sign of generated data; real concentration–response curves are smooth. |
| `population_density` is random per row | Every city has the same ≈10% / 30% / 60% Rural / Suburban / Urban split. | Not a city property. Excluded from the primary set. |
| Unbalanced sampling | Delhi 26,465; Beijing 22,064; Mexico City 13,377; Los Angeles 9,003; London 6,985; Tokyo 6,147; Cairo 2,700; São Paulo 1,748. | Stratified splits, per-city CIs, caution for small cities. |
| Count outcome, mild overdispersion | Mean 8.05, variance 13.80. 2.0% zeros. Max 25. Admissions never exceed `hospital_capacity`. | Predictions clipped at 0. (The planned Poisson GLM was dropped in the lean scope; this is stated as a limitation.) |
| Negative values are legitimate | The only negatives are temperatures (minimum −5 °C). | No cleaning needed. A validation script asserts it. |

**Conclusion.** The data behaves like a controlled benchmark in which the real signal is one linear feature plus noise. That is useful for testing whether complex models over-fit or add nothing. It is a poor source of real-world epidemiological conclusions, and the project is designed around that.

## Real-data comparison (optional, Methodology Step 2B)

**Status: skipped (lean scope).** The report states that the Delhi-vs-London pollution claim can't be tested with this file. The table is kept in case Step 2B is done later.

Fill only if Step 2B is done, after choosing the real reference dataset. Expected patterns are hypotheses to check, not results.

| Statistic | Main dataset (synthetic) | Real reference dataset | Expected in real data |
|---|---|---|---|
| Delhi mean PM2.5 / AQI | 35.1 / 248.9 | | Clearly higher than a low-pollution city |
| Low-pollution city mean PM2.5 / AQI | London: 35.4 / 248.4 | | Clearly lower than Delhi |
| Correlation of AQI with PM2.5 | 0.003 | | Strongly positive |
| Lag-1 autocorrelation of PM2.5 | −0.001 | | Strongly positive |
| Seasonal pattern of PM2.5 | None | | Visible seasonality |
| Share of days above 109.9 µg/m³ | 0% by construction | | To be measured |
