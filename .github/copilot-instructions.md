# Project: Respiratory admissions vs air quality (Data Mining course)

Plan: docs/METHODOLOGY.md. Overview: README.md.
Other docs in docs/ (DATA_AUDIT, LIMITATIONS_AND_DEFENSE, REPRODUCIBILITY, RESULTS_AND_CHECKLIST) exist. Read them only when I name them.
EVALUATION.md does not exist. Do not look for it.
If any doc conflicts with this file, THIS FILE WINS (for example: scripts not notebooks, sklearn MLP, lean scope below).

## Lean scope (overrides METHODOLOGY.md)
- Feature sets: only FS-Primary and FS-PM25.
- Skip: Step 8 (repeated CV), Step 13 (robustness), Holm correction, leave-one-city-out, Poisson GLM, VIF, Step 2B.
- Keep: Steps 0 to 7, 9, 10 (pooled vs per-city, PM2.5 slope per city), 11, 12, 14.
Read PROGRESS.md at the start of a task to see what is already done.

## Rules
- Do ONLY the stage I name. When done, stop, list files created, and append 3 lines to PROGRESS.md (stage, files, key numbers). Do not refactor earlier stages.
- Run git commands only when I explicitly ask (for example "commit stage 2"). Never push unless I say "push". Never use --force. Never commit data/raw or .venv.
- Write Python scripts in src/, not notebooks. Save tables to results/tables (csv/json) and plots to results/figures (png).
- Never print or read the whole CSV. Use df.head(), df.info(), df.describe() only.
- SEED = 42 in src/config.py. The test set is used once, in the final evaluation script only.
- Fit scalers, encoders and thresholds on train data only, inside sklearn Pipelines.
- Keep code short and plain. No CLI frameworks, no long docstrings, no test suites beyond the split check.
- Print short summaries only (about 20 lines max).
- Environment: create it with `uv venv --python 3.12` in .venv (system Python is 3.14; do not use it). Install with `uv pip install`.
  Allowed packages: pandas numpy scikit-learn scipy statsmodels matplotlib.
- Neural network: sklearn MLPRegressor / MLPClassifier. Do not install torch or tensorflow unless I say so.

## FAST mode (only for checking that code runs)
- src/config.py reads the environment variable RUN_MODE. If it is unset, the mode is FAST (safe default). Only RUN_MODE=FULL gives report-grade results. FAST numbers are never reported.
- FAST=True writes only to results_fast/ and adds "_FAST" to every file name. It never writes to results/.
- Every saved table has a `mode` column ("FAST" or "FULL").
- FAST changes only the row sample size and grid size. Same split logic, preprocessing and metric code. It never touches the test set.
- Print "FAST MODE - NOT FOR REPORT" at the start of every script when FAST=True.
- The final evaluation script and all report figures must assert FAST=False and stop otherwise.
- src/check_results.py scans results/ and fails if any file has mode=FAST or _FAST in its name.
- Full runs: only when I say "launch full run <script>". Launch it in the background and never wait for it:
  `$env:RUN_MODE='FULL'; Start-Process -NoNewWindow -FilePath .\.venv\Scripts\python.exe -ArgumentList 'src\<script>.py' -RedirectStandardOutput results\logs\<script>.out.txt -RedirectStandardError results\logs\<script>.err.txt`
  Then tell me the log path and stop. One full run at a time.

## Windows and slow laptop
- OS is Windows. Run scripts with `.venv\Scripts\python src\<script>.py` (no activation needed).
- Never run a job you expect to take more than 5 minutes in the foreground. Only exception: a full run I asked you to launch (background, as above).
- Use n_jobs=2 at most. Put code that uses multiprocessing under `if __name__ == "__main__":`.
- Write logs to results/logs/*.txt. Print only the last 10 lines to the console.
- SVC: do not use probability=True. Use decision_function scores for PR-AUC and ROC-AUC.
- Permutation importance: 5 repeats, on a 3,000-row test subsample for SVM and MLP.
- Random search instead of full grids for the tree, SVM and MLP (n_iter 10 to 30).
- SVM: RandomizedSearchCV (n_iter=10, cv=3) on a 5k to 8k stratified subsample, then fit the final model on at most 20k rows.

## Facts
- Data: data/raw/air_quality_health_dataset.csv, 88,489 rows, 8 cities, target hospital_admissions.
- Audit: only pm2_5 relates to admissions (r about 0.39). Dates are meaningless (no time series). Data shows strong signs of being synthetic.
- H1: PM2.5 alone predicts about as well as all features.
- Tie rule: models whose RMSE differs by less than 0.05 are reported as tied.
- Spike = admissions >= 90th percentile of the training set.
