import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import FAST, PROJECT_ROOT, RUN_MODE, SEED, SPLIT_PATH
from data import load_raw
from prepare import save_split

CV_FOLDS = 5
FAST_ROWS = 5_000
OUT_TABLES = PROJECT_ROOT / ("results_fast" if FAST else "results") / "tables"
SUFFIX = "_FAST" if FAST else ""
REG_SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
CLF_SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def city_mean_model() -> Pipeline:
    encode = ColumnTransformer([("city", OneHotEncoder(handle_unknown="ignore"), ["city"])])
    return Pipeline([("preprocess", encode), ("model", LinearRegression())])


def baseline_specs() -> list[tuple[str, str, object, list[str]]]:
    return [
        ("mean_predictor", "regression", DummyRegressor(strategy="mean"), ["pm2_5"]),
        ("city_mean_predictor", "regression", city_mean_model(), ["city"]),
        ("pm25_linear_regression", "regression", LinearRegression(), ["pm2_5"]),
        ("prevalence_classifier", "classification", DummyClassifier(strategy="prior"), ["pm2_5"]),
    ]


def score_baselines(train: pd.DataFrame, threshold: float) -> pd.DataFrame:
    spike = (train["hospital_admissions"] >= threshold).astype(int)
    strata = train["city"].astype(str) + "_" + spike.astype(str)
    folds = list(StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(train, strata))
    rows = []
    for name, task, model, features in baseline_specs():
        regression = task == "regression"
        y = train["hospital_admissions"] if regression else spike
        scoring = REG_SCORING if regression else CLF_SCORING
        scores = cross_validate(model, train[features], y, cv=folds, scoring=scoring, n_jobs=2)
        row = {"baseline": name, "task": task, "features": "+".join(features), "cv_rows": len(train)}
        for metric in scoring:
            values = scores[f"test_{metric}"] * (-1 if metric in ("rmse", "mae") else 1)
            row[f"cv_{metric}_mean"] = values.mean()
            row[f"cv_{metric}_sd"] = values.std(ddof=1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["spike_threshold"] = threshold
    table["spike_prevalence"] = spike.mean()
    table["cv_folds"] = CV_FOLDS
    table["mode"] = RUN_MODE
    return table


if __name__ == "__main__":
    if FAST:
        print("FAST MODE - NOT FOR REPORT")
    data = load_raw()
    if not SPLIT_PATH.exists():
        save_split(data)
    train_idx = np.array(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["train_indices"], dtype=int)
    threshold = json.loads((PROJECT_ROOT / "results" / "tables" / "03_spike_threshold.json").read_text())["threshold"]
    train = data.iloc[train_idx].reset_index(drop=True)
    if FAST:
        spike = (train["hospital_admissions"] >= threshold).astype(int)
        strata = train["city"].astype(str) + "_" + spike.astype(str)
        keep, _ = train_test_split(np.arange(len(train)), train_size=FAST_ROWS, random_state=SEED, stratify=strata)
        train = train.iloc[np.sort(keep)].reset_index(drop=True)
    table = score_baselines(train, threshold)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    path = OUT_TABLES / f"06_baselines{SUFFIX}.csv"
    table.to_csv(path, index=False)
    cols = ["baseline", "cv_rmse_mean", "cv_mae_mean", "cv_r2_mean", "cv_pr_auc_mean", "cv_roc_auc_mean"]
    print(table[cols].round(4).to_string(index=False))
    print(f"CV on {len(train):,} training rows; test set not loaded; saved {path.relative_to(PROJECT_ROOT)}")
