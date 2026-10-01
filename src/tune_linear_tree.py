import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import FEATURE_SETS, PROJECT_ROOT, RUN_MODE, SEED
from data import load_raw

FAST = RUN_MODE == "FAST"
SUFFIX = "_FAST" if FAST else ""
OUT_DIR = PROJECT_ROOT / ("results_fast" if FAST else "results")
OUT_TABLES = OUT_DIR / "tables"
OUT_LOGS = OUT_DIR / "logs"
THRESHOLD_PATH = PROJECT_ROOT / "results" / "tables" / "03_spike_threshold.json"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_indices.json"
CV_FOLDS = 3 if FAST else 5


def output_path(name: str) -> Path:
    return OUT_TABLES / f"{name}{SUFFIX}.csv"


def load_threshold() -> float:
    return float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"])


def load_train(data: pd.DataFrame) -> pd.DataFrame:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train = data.iloc[np.array(split["train_indices"], dtype=int)].reset_index(drop=True)
    if not FAST:
        return train
    threshold = load_threshold()
    spike = (train["hospital_admissions"] >= threshold).astype(int)
    strata = train["city"].astype(str) + "_" + spike.astype(str)
    positions, _ = train_test_split(
        np.arange(len(train)), train_size=5_000, random_state=SEED, stratify=strata
    )
    return train.iloc[np.sort(positions)].reset_index(drop=True)


def preprocessor(features: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric = [feature for feature in features if feature != "city"]
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    transformers = [("numeric", numeric_transformer, numeric)]
    if "city" in features:
        transformers.append(("city", OneHotEncoder(handle_unknown="ignore"), ["city"]))
    return ColumnTransformer(transformers=transformers)


def pipeline(model, features: list[str], scale_numeric: bool) -> Pipeline:
    return Pipeline([("preprocess", preprocessor(features, scale_numeric)), ("model", model)])


def folds_for(data: pd.DataFrame, threshold: float) -> list[tuple[np.ndarray, np.ndarray]]:
    spike = (data["hospital_admissions"] >= threshold).astype(int)
    strata = data["city"].astype(str) + "_" + spike.astype(str)
    return list(StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(data, strata))


def pruning_candidates(data: pd.DataFrame, features: list[str], task: str) -> list[float]:
    target = data["hospital_admissions"] if task == "regression" else (data["hospital_admissions"] >= load_threshold()).astype(int)
    estimator = DecisionTreeRegressor(random_state=SEED) if task == "regression" else DecisionTreeClassifier(random_state=SEED)
    transformed = preprocessor(features, scale_numeric=False).fit_transform(data[features], target)
    path = estimator.cost_complexity_pruning_path(transformed, target)
    positive = np.unique(path.ccp_alphas[path.ccp_alphas > 0])
    if len(positive) == 0:
        return []
    midpoints = np.sqrt(positive[:-1] * positive[1:])
    upper = float(midpoints.max()) if len(midpoints) else float(positive.max())
    count = 3 if FAST else 8
    values = np.geomspace(float(positive.min()), upper, num=count)
    return sorted({float(value) for value in values if value > 0})


def linear_space(model: str) -> tuple[list, dict, str]:
    if model == "Linear Regression":
        # Hoerl & Kennard (1970): alpha controls scale-dependent L2 shrinkage; include OLS at alpha=0.
        values = [0, 1, 10] if FAST else [0, 0.01, 0.1, 1, 10]
        return values, {"model__alpha": values}, "rmse"
    # Hastie, Tibshirani & Friedman, ESL sec. 3.4: C controls inverse L2 regularization strength.
    values = [0.01, 1, 100] if FAST else [1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
    return values, {"model__C": values}, "pr_auc"


def tree_space(ccp_values: list[float], task: str) -> dict:
    # Breiman et al. (1984), CART: cost-complexity pruning is selected on impurity-calibrated candidates.
    depths = [3, 5, None] if FAST else [2, 3, 4, 5, 6, 8, 10, None]
    leaves = [20, 100] if FAST else [1, 5, 20, 50, 100, 200]
    values = [0.0] + ccp_values
    parameters = {"model__max_depth": depths, "model__min_samples_leaf": leaves, "model__ccp_alpha": values}
    if task == "classification":
        parameters["model__class_weight"] = [None, "balanced"]
    return parameters


def model_and_space(model: str, task: str, features: list[str], ccp_values: list[float]) -> tuple[Pipeline, dict, str]:
    if model == "Linear Regression":
        return pipeline(Ridge(random_state=SEED), features, True), linear_space(model)[1], "rmse"
    if model == "Logistic Regression":
        return pipeline(LogisticRegression(solver="lbfgs", max_iter=5000, random_state=SEED), features, True), linear_space(model)[1], "pr_auc"
    if task == "regression":
        estimator = DecisionTreeRegressor(random_state=SEED)
        metric = "rmse"
    else:
        estimator = DecisionTreeClassifier(random_state=SEED)
        metric = "pr_auc"
    return pipeline(estimator, features, False), tree_space(ccp_values, task), metric


def extend_linear_grid(model: str, grid: dict, best: pd.Series) -> tuple[dict, bool, str]:
    if model == "Linear Regression":
        values = list(grid["model__alpha"])
        if best["param_model__alpha"] == max(values) and max(values) == 10:
            return {"model__alpha": values + [30, 100, 300, 1000]}, True, "alpha_high"
        return grid, False, ""
    values = list(grid["model__C"])
    best_value = float(best["param_model__C"])
    if best_value == min(values):
        return {"model__C": [min(values) / 10] + values}, True, "C_low"
    if best_value == max(values):
        return {"model__C": values + [max(values) * 10]}, True, "C_high"
    return grid, False, ""


def extend_tree_grid(grid: dict, best: pd.Series) -> tuple[dict, bool, str]:
    # METHODOLOGY 7.2 edge rule. min_samples_leaf=1, ccp_alpha=0 and max_depth=None are natural
    # bounds; the largest ccp candidate already prunes to a stump, so it has no meaningful upper edge.
    extended, notes = dict(grid), []
    leaves = list(grid["model__min_samples_leaf"])
    if float(best["param_model__min_samples_leaf"]) == max(leaves):
        extended["model__min_samples_leaf"] = leaves + [max(leaves) * factor for factor in (2.5, 5, 10)]
        extended["model__min_samples_leaf"] = [int(value) for value in extended["model__min_samples_leaf"]]
        notes.append("leaf_high")
    depths = [value for value in grid["model__max_depth"] if value is not None]
    best_depth = best["param_model__max_depth"]
    if best_depth is not None and not pd.isna(best_depth) and float(best_depth) == min(depths) and min(depths) > 1:
        extended["model__max_depth"] = [min(depths) - 1] + list(grid["model__max_depth"])
        notes.append("depth_low")
    return extended, bool(notes), ",".join(notes)


def run_grid(data: pd.DataFrame, model: str, task: str, feature_set: str, ccp_values: list[float]) -> tuple[pd.DataFrame, pd.Series, int, bool, str]:
    features = FEATURE_SETS[feature_set]
    estimator, grid, metric = model_and_space(model, task, features, ccp_values)
    target = data["hospital_admissions"] if task == "regression" else (data["hospital_admissions"] >= load_threshold()).astype(int)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"} if task == "regression" else {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
    folds = folds_for(data, load_threshold())
    warning_count = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search = GridSearchCV(
            estimator,
            param_grid=grid,
            scoring=scoring,
            refit=False,
            cv=folds,
            n_jobs=2,
            return_train_score=False,
        )
        search.fit(data[features], target)
        warning_count += sum(isinstance(item.message, ConvergenceWarning) for item in caught)
    result = pd.DataFrame(search.cv_results_)
    selection_column = "rank_test_rmse" if task == "regression" else "rank_test_pr_auc"
    best = result.loc[result[selection_column].idxmin()]
    if model in ["Linear Regression", "Logistic Regression"]:
        extended_grid, edge_extended, edge_note = extend_linear_grid(model, grid, best)
    else:
        extended_grid, edge_extended, edge_note = extend_tree_grid(grid, best)
    if edge_extended:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            search = GridSearchCV(estimator, param_grid=extended_grid, scoring=scoring, refit=False, cv=folds, n_jobs=2, return_train_score=False)
            search.fit(data[features], target)
            warning_count += sum(isinstance(item.message, ConvergenceWarning) for item in caught)
        result = pd.DataFrame(search.cv_results_)
        grid = extended_grid
    for metric_name in ["rmse", "mae"]:
        if f"mean_test_{metric_name}" in result:
            result[f"mean_test_{metric_name}"] = -result[f"mean_test_{metric_name}"]
    best = result.loc[result[selection_column].idxmin()]
    result.insert(0, "feature_set", feature_set)
    result.insert(1, "model", model)
    result.insert(2, "task", task)
    result["grid_size"] = int(np.prod([len(values) for values in grid.values()]))
    result["edge_extended"] = edge_extended
    result["edge_note"] = edge_note
    result["ccp_candidates"] = json.dumps(ccp_values)
    result["convergence_warnings"] = warning_count
    result["cv_folds"] = CV_FOLDS
    result["rows_used"] = len(data)
    result["mode"] = RUN_MODE
    return result, best, warning_count, edge_extended, edge_note


def best_record(result: pd.DataFrame, best: pd.Series, model: str, task: str, feature_set: str, ccp_values: list[float], warnings_count: int, edge_extended: bool, edge_note: str) -> dict:
    metrics = ["rmse", "mae", "r2"] if task == "regression" else ["pr_auc", "roc_auc"]
    params = {key.replace("param_model__", ""): value for key, value in best.items() if key.startswith("param_model__")}
    record = {
        "feature_set": feature_set,
        "model": model,
        "task": task,
        "best_params": json.dumps(params, default=str),
        "grid_size": int(result["grid_size"].iloc[0]),
        "edge_extended": edge_extended,
        "edge_note": edge_note,
        "ccp_candidates": json.dumps(ccp_values),
        "convergence_warnings": warnings_count,
        "cv_folds": CV_FOLDS,
        "rows_used": int(result["rows_used"].iloc[0]),
        "mode": RUN_MODE,
    }
    for metric in metrics:
        record[f"cv_{metric}_mean"] = float(best[f"mean_test_{metric}"])
        record[f"cv_{metric}_sd"] = float(best[f"std_test_{metric}"])
    return record


def log_finished(feature_set: str, model: str, task: str, warnings_count: int) -> None:
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    path = OUT_LOGS / f"tune_linear_tree{SUFFIX}.progress.txt"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat(timespec='seconds')}\t{feature_set}\t{model}\t{task}\twarnings={warnings_count}\tmode={RUN_MODE}\n")


if __name__ == "__main__":
    if FAST:
        print("FAST MODE - NOT FOR REPORT")
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    data = load_raw()
    train = load_train(data)
    results = []
    best_rows = []
    for feature_set, features in FEATURE_SETS.items():
        for task, model in [("regression", "Linear Regression"), ("classification", "Logistic Regression")]:
            result, best, warnings_count, edge_extended, edge_note = run_grid(train, model, task, feature_set, [])
            results.append(result)
            best_rows.append(best_record(result, best, model, task, feature_set, [], warnings_count, edge_extended, edge_note))
            log_finished(feature_set, model, task, warnings_count)
            print(f"{feature_set}: {model} ({task}) complete; warnings={warnings_count}")
        for task in ["regression", "classification"]:
            model = "Decision Tree"
            ccp_values = pruning_candidates(train, features, task)
            result, best, warnings_count, edge_extended, edge_note = run_grid(train, model, task, feature_set, ccp_values)
            results.append(result)
            best_rows.append(best_record(result, best, model, task, feature_set, ccp_values, warnings_count, edge_extended, edge_note))
            log_finished(feature_set, model, task, warnings_count)
            print(f"{feature_set}: {model} ({task}) complete; warnings={warnings_count}")
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(output_path("07_cv_results_linear_tree"), index=False)
    pd.DataFrame(best_rows).to_csv(output_path("07_best_linear_tree"), index=False)
    print(f"saved {len(all_results):,} CV rows; {len(best_rows)} searches; test set not loaded")
