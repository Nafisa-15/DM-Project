# Project Rules and How to Run

Part 1 lists the fixed rules this project followed: scope, data handling, evaluation, tuning and compute. They were set before results existed, except where a dated change says otherwise. Other documents cite this file. Reasons and evidence for each rule are in [DECISIONS_LOG.md](DECISIONS_LOG.md) (D-numbers).

Part 2 explains how to view the results or rerun everything on another computer.

---

# Part 1. Rules

## Scope (lean)
- **Feature sets:** FS-Primary (`aqi, pm2_5, pm10, no2, o3, temperature, humidity` + one-hot `city`) and FS-PM25 (`pm2_5` only).
- **Done:** Steps 0–7, 9, 10 (pooled vs per-city; PM2.5 slope per city), 11, 12, 14 of [METHODOLOGY.md](METHODOLOGY.md).
- **Skipped:** Step 2B (real-data check), Step 8 (repeated CV), Step 13 (robustness), Holm correction, leave-one-city-out, Poisson GLM, VIF. Each is stated as a limitation.

## Data handling
- `SEED = 42` for every split, fold, subsample and model.
- One 80/20 split, stratified by city × spike. **The test set is read only by `src/final_evaluation.py`**, after every setting, threshold and model choice is fixed on training data.
- The spike threshold (admissions ≥ 13, the training 90th percentile) comes from training rows only. Decision thresholds come from out-of-fold training predictions.
- Scalers and encoders are fitted inside sklearn Pipelines on training folds only.

## Evaluation
- **Selection metrics:** CV RMSE (regression), CV PR-AUC (classification).
- **Tie rule:** two models differ only if the RMSE gap is ≥ 0.05 **and** the paired bootstrap 95% CI (1,000 resamples) excludes 0. No PR-AUC margin was set in advance, and none was added later.
- **Leakage alarm:** any test R² > 0.20 stops reporting until audited. PM2.5 alone caps R² near 0.154.
- Regression predictions are clipped at 0.

## Tuning
- **All searches:** 5-fold CV on the shared city × spike folds, so every model is scored on the same folds (Hastie et al., *ESL* §7.10).
- **Search size** (Bergstra & Bengio 2012): choose `n_iter` so that 1 − (1 − p)^n ≥ 0.95, where p is the measured share of settings scoring within 0.01 of the best.
  - Regression: p ≥ 0.30, so `n_iter` ≥ 10 (20 used).
  - Classification: p ≥ 0.10, so `n_iter` ≥ 30.
  - Cheap models (Ridge/Logistic, Decision Tree) use the full grid.
- **Edge rule:** if a best value sits on a range edge, widen that range once and record it (`edge_note`).
- **SVM:** two-phase random search (40 × 5-fold on 10k rows, then 20 × 5-fold on 20k rows); final fit on 20k rows, justified by a flat learning curve from 5k to 40k (D7–D10).
- **MLP:** sklearn `MLPRegressor` / `MLPClassifier` (no PyTorch or TensorFlow); final setting trained with 5 seeds, reported as mean ± SD.

## Compute limits (project laptop)
- At most 2 parallel jobs.
- SVC uses `decision_function` scores, not `probability=True` (Platt scaling would multiply the cost).
- Permutation importance: 5 repeats; a 3,000-row test subsample for the MLP.
- Allowed packages: pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib.

## FAST / FULL runs
- Every script reads `RUN_MODE`. **FAST** (the default) uses a small sample and writes only to `results_fast/` with a `_FAST` suffix. **FULL** writes report-grade results to `results/`.
- Every table has a `mode` column. `src/check_results.py` fails if `results/` contains any FAST data, and `final_evaluation.py` refuses to run unless it passes.

## Changes to these rules
| Date | Rule | Before | After | Why |
|---|---|---|---|---|
| 2026-10-01 | Search size | `n_iter` 10–30 random search for tree/SVM/MLP; SVM `n_iter=10`, `cv=3` on 5k–8k rows | The search-size rule above | Measured: with 10 draws, classification risked under-tuning (≈1-in-4 chance for the SVC). See D7, D11, and METHODOLOGY "Changes from the plan". |

---

# Part 2. Running on another computer

## You do not need to retrain anything to see the results
Every result is already saved in the project folder: about 26 MB, not counting the environment.

| Folder | Contents |
|---|---|
| `results/tables/` | Every number reported (CSV/JSON) |
| `results/figures/` | Every figure (PNG) |
| `results/models/` | The trained SVM and MLP models (the slow ones) |
| `results/logs/` | The log of every FULL run |
| `data/processed/split_indices.json` | The exact train/test split |

To show the work, open [../reports/REPORT.md](../reports/REPORT.md). In VS Code, `Ctrl+Shift+V` shows it rendered with its figures; on GitHub it renders automatically. Closing VS Code, restarting the computer, or copying the whole folder elsewhere changes none of this.

## What another computer needs
1. **The project folder.** Copy it or clone it from git. `data/raw/` and `.venv/` are not in git (see `.gitignore`).
2. **uv** (Python package manager): `pip install uv` with any Python, or see https://docs.astral.sh/uv/.
3. **The dataset**, only if you want to rerun code. Download `air_quality_health_dataset.csv` from https://www.kaggle.com/datasets/tfisthis/global-air-quality-and-respiratory-health-outcomes into `data/raw/`, then check it's the same file:
   - Windows: `Get-FileHash data\raw\air_quality_health_dataset.csv -Algorithm SHA256`
   - macOS/Linux: `shasum -a 256 data/raw/air_quality_health_dataset.csv`
   - Expected: `70AF8B5F0E94E5D2D4AE5C1E5CFA79C7FC4B39B0517AA6A990214B5A2E4E5E58` (5.4 MB, 88,489 rows). `src/data.py` also checks the shape on every load.

## Set up the environment (about 2 minutes)
```powershell
uv venv --python 3.12                    # creates .venv with Python 3.12 (uv downloads it if missing)
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python -c "import sklearn, pandas; print('ok')"
```
On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python`.

**If the folder was moved:** a `.venv` stops working when its folder moves, because it stores absolute paths. Delete `.venv` and run the two setup lines again. Nothing else is affected.

## Check the code runs (FAST mode, a few minutes in total)
With `RUN_MODE` unset, every script runs in FAST mode on a small sample and writes only to `results_fast/`. It never touches `results/` or the test set.
```powershell
.venv\Scripts\python src\data.py
.venv\Scripts\python src\baselines.py
.venv\Scripts\python src\tune_linear_tree.py
```

## Reproduce every result (FULL mode, about 4.5 hours on the project laptop)
`tune_svm.py`, `tune_mlp.py`, `analysis_train.py` and `final_evaluation.py` **skip any part whose FULL output already exists**, so rerunning them in place reuses the saved results. The other scripts rerun fully and overwrite their outputs with the same numbers (seed 42). For a truly fresh run, move or rename `results/` first.

```powershell
$env:RUN_MODE='FULL'
.venv\Scripts\python src\data.py               # Step 1   seconds
.venv\Scripts\python src\audit.py              # Step 2   ~10 s
.venv\Scripts\python src\prepare.py            # Steps 3-5 seconds (rewrites the same split: seed 42)
.venv\Scripts\python src\test_split.py         # Step 5 check
.venv\Scripts\python src\baselines.py          # Step 6   ~10 s
.venv\Scripts\python src\tune_linear_tree.py   # Step 7   ~50 min
.venv\Scripts\python src\tune_svm.py           # Step 7   ~1 h 15 min
.venv\Scripts\python src\tune_mlp.py           # Step 7   ~1 h 45 min
.venv\Scripts\python src\analysis_train.py     # Steps 10-12 (training side) ~40 min
.venv\Scripts\python src\check_results.py      # must print PASS
.venv\Scripts\python src\final_evaluation.py   # Steps 9-12 (test side) ~3 min; the only script that reads the test set
```
- **Background runs:** for the long scripts, use the background form so a closed terminal doesn't stop them:
  `Start-Process -NoNewWindow -FilePath .\.venv\Scripts\python.exe -ArgumentList 'src\tune_svm.py' -RedirectStandardOutput results\logs\tune_svm.out.txt -RedirectStandardError results\logs\tune_svm.err.txt`
- **Progress:** `results/logs/*.progress.txt` shows how far a run has got. A stopped run resumes from its last finished part when relaunched.
- **Expected agreement:** with the pinned versions in `requirements.txt` and seed 42, a rerun gives the same numbers. Another operating system or CPU can change the last decimal places of MLP and SVM results, but not any conclusion.
- **Laptop limits:** scripts use at most 2 CPU cores (`n_jobs=2`). A faster machine finishes sooner without code changes.

## Repository layout
```
.
├── README.md                  # plain-language overview
├── requirements.txt           # pinned package versions
├── reports/REPORT.md          # final report
├── docs/                      # methodology, rules, audit, decisions, results, defense
│   └── archive/               # the original plan (written before any code)
├── data/
│   ├── raw/                   # the CSV (download; not in git)
│   └── processed/             # split_indices.json
├── src/                       # config, data, audit, prepare, test_split, baselines,
│                              # tune_linear_tree, tune_svm, tune_mlp, analysis_train,
│                              # check_results, final_evaluation
├── results/                   # FULL results: tables, figures, models, logs
└── results_fast/              # FAST test-run output (not in git, never reported)
```
