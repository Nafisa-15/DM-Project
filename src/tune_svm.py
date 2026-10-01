import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils import check_random_state

from config import FEATURE_SETS, PROJECT_ROOT, RUN_MODE, SEED
from data import load_raw

FAST = RUN_MODE == "FAST"
SUFFIX = "_FAST" if FAST else ""
OUT_DIR = PROJECT_ROOT / ("results_fast" if FAST else "results")
OUT_TABLES = OUT_DIR / "tables"
OUT_FIGURES = OUT_DIR / "figures"
OUT_MODELS = OUT_DIR / "models"
OUT_LOGS = OUT_DIR / "logs"
THRESHOLD_PATH = PROJECT_ROOT / "results" / "tables" / "03_spike_threshold.json"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_indices.json"

PHASE_A_ROWS = 1_500 if FAST else 10_000
PHASE_B_ROWS = 3_000 if FAST else 20_000
PHASE_A_ITER = 4 if FAST else 40
PHASE_B_ITER = 3 if FAST else 20
CV_FOLDS_A = 3 if FAST else 5
CV_FOLDS_B = 3 if FAST else 5
LEARNING_SIZES = [1_000, 2_000, 3_000] if FAST else [5_000, 10_000, 20_000, 40_000]
# epsilon range fixed before the FULL run: upper bound ~ one residual SD (PM2.5 baseline CV RMSE 3.417)
EPS_LOW, EPS_HIGH = 0.05, 3.5


class MixedGamma:
    def __init__(self, low: float = 1e-4, high: float = 1.0):
        self.low = low
        self.high = high

    def rvs(self, random_state=None):
        rng = check_random_state(random_state)
        if rng.rand() < 0.2:
            return "scale"
        return float(10 ** rng.uniform(np.log10(self.low), np.log10(self.high)))


def output_path(name: str, folder: Path = OUT_TABLES) -> Path:
    return folder / f"{name}{SUFFIX}.csv"


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if "mode" not in table or not (table["mode"] == RUN_MODE).all():
        return pd.DataFrame()
    return table


def save_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def append_progress(stage: str, feature_set: str, task: str, note: str = "") -> None:
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    path = OUT_LOGS / f"tune_svm.progress{SUFFIX}.txt"
    stamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\t{stage}\t{feature_set}\t{task}\t{note}\tmode={RUN_MODE}\n")


def make_pipeline(task: str, features: list[str]) -> Pipeline:
    numeric = [feature for feature in features if feature != "city"]
    transformers = [("numeric", StandardScaler(), numeric)]
    if "city" in features:
        transformers.append(("city", OneHotEncoder(handle_unknown="ignore"), ["city"]))
    preprocess = ColumnTransformer(transformers=transformers)
    model = SVR(cache_size=2000) if task == "regression" else SVC(cache_size=2000)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def metric_specs(task: str) -> tuple[dict, str, list[str]]:
    if task == "regression":
        return (
            {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
            "rmse",
            ["rmse", "mae", "r2"],
        )
    return ({"pr_auc": "average_precision", "roc_auc": "roc_auc"}, "pr_auc", ["pr_auc", "roc_auc"])


def phase_a_distributions(task: str) -> dict:
    values = {
        "model__C": loguniform(0.01, 1000),
        "model__gamma": MixedGamma(),
    }
    if task == "regression":
        values["model__epsilon"] = uniform(EPS_LOW, EPS_HIGH - EPS_LOW)
    else:
        values["model__class_weight"] = [None, "balanced"]
    return values


def load_train(data: pd.DataFrame) -> pd.DataFrame:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return data.iloc[np.array(split["train_indices"], dtype=int)].reset_index(drop=True)


def load_threshold() -> float:
    return float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"])


def strata_for(data: pd.DataFrame, threshold: float) -> pd.Series:
    spike = (data["hospital_admissions"] >= threshold).astype(int)
    return data["city"].astype(str) + "_" + spike.astype(str)


def nested_samples(train: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = strata_for(train, threshold)
    phase_b_positions, _ = train_test_split(
        np.arange(len(train)), train_size=PHASE_B_ROWS, random_state=SEED, stratify=strata
    )
    phase_b_positions = np.sort(phase_b_positions)
    phase_b = train.iloc[phase_b_positions].reset_index(drop=True)
    phase_a_positions, _ = train_test_split(
        np.arange(len(phase_b)),
        train_size=PHASE_A_ROWS,
        random_state=SEED,
        stratify=strata.iloc[phase_b_positions].reset_index(drop=True),
    )
    return train.iloc[phase_b_positions[np.sort(phase_a_positions)]].reset_index(drop=True), phase_b


def cv_splits(data: pd.DataFrame, threshold: float, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    return list(splitter.split(data, strata_for(data, threshold)))


def clean_params(params: dict) -> dict:
    return {name: (value.item() if isinstance(value, np.generic) else value) for name, value in params.items()}


def score_columns(table: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    for metric in ["rmse", "mae"]:
        if metric in metrics:
            table[f"mean_test_{metric}"] = -table[f"mean_test_{metric}"]
    return table


def search_key(table: pd.DataFrame) -> set[str]:
    if table.empty:
        return set()
    return set(table["feature_set"] + "|" + table["task"])


def run_search(data: pd.DataFrame, feature_set: str, task: str, distributions: dict, iterations: int, folds: int, stage: str) -> pd.DataFrame:
    scoring, refit_metric, metrics = metric_specs(task)
    features = FEATURE_SETS[feature_set]
    target = data["hospital_admissions"] if task == "regression" else (data["hospital_admissions"] >= load_threshold()).astype(int)
    search = RandomizedSearchCV(
        make_pipeline(task, features),
        param_distributions=distributions,
        n_iter=iterations,
        scoring=scoring,
        refit=False,
        cv=cv_splits(data, load_threshold(), folds),
        random_state=SEED,
        n_jobs=2,
        return_train_score=False,
    )
    search.fit(data[features], target)
    result = score_columns(pd.DataFrame(search.cv_results_), metrics)
    result.insert(0, "feature_set", feature_set)
    result.insert(1, "task", task)
    result.insert(2, "stage", stage)
    result["rows_used"] = len(data)
    result["n_iter"] = iterations
    result["cv"] = folds
    result["refit_metric"] = refit_metric
    result["mode"] = RUN_MODE
    return result


def best_row(table: pd.DataFrame, feature_set: str, task: str) -> pd.Series:
    subset = table[(table["feature_set"] == feature_set) & (table["task"] == task)]
    metric = "rmse" if task == "regression" else "pr_auc"
    return subset.loc[subset[f"rank_test_{metric}"].idxmin()]


def value_from_row(row: pd.Series, name: str):
    value = row.get(f"param_model__{name}")
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value.item() if isinstance(value, np.generic) else value


def edge_range(best: float, low: float, high: float) -> tuple[float, float, bool, str]:
    span = np.log10(high) - np.log10(low)
    low_edge = np.log10(best) - np.log10(low) <= 0.1 * span
    high_edge = np.log10(high) - np.log10(best) <= 0.1 * span
    lower, upper, notes = best / 4, best * 4, []
    if low_edge:
        lower = low / 10
        notes.append("low")
    if high_edge:
        upper = high * 10
        notes.append("high")
    return lower, upper, bool(notes), ",".join(notes)


def epsilon_range(best: float) -> tuple[float, float, bool, str]:
    span = EPS_HIGH - EPS_LOW
    lower, upper = max(0.5 * best, EPS_LOW), min(1.5 * best, EPS_HIGH)
    notes = []
    if best - EPS_LOW <= 0.1 * span:
        lower = 0.01
        notes.append("eps_low")
    if EPS_HIGH - best <= 0.1 * span:
        upper = 1.5 * EPS_HIGH
        notes.append("eps_high")
    return lower, upper, bool(notes), ",".join(notes)


def phase_b_distributions(task: str, row: pd.Series) -> tuple[dict, bool, str]:
    best_c = float(value_from_row(row, "C"))
    c_low, c_high, c_extended, c_note = edge_range(best_c, 0.01, 1000)
    best_gamma = value_from_row(row, "gamma")
    gamma_extended = False
    gamma_note = ""
    if isinstance(best_gamma, str):
        gamma_values = [best_gamma]
    else:
        gamma_low, gamma_high, gamma_extended, gamma_note = edge_range(float(best_gamma), 1e-4, 1)
        gamma_values = loguniform(gamma_low, gamma_high)
    distributions = {"model__C": loguniform(c_low, c_high), "model__gamma": gamma_values}
    eps_extended, eps_note = False, ""
    if task == "regression":
        eps_low, eps_high, eps_extended, eps_note = epsilon_range(float(value_from_row(row, "epsilon")))
        distributions["model__epsilon"] = uniform(eps_low, eps_high - eps_low)
    else:
        distributions["model__class_weight"] = [value_from_row(row, "class_weight")]
    c_note = ",".join(f"C_{side}" for side in c_note.split(",") if side)
    gamma_note = ",".join(f"gamma_{side}" for side in gamma_note.split(",") if side)
    notes = ";".join(note for note in [c_note, gamma_note, eps_note] if note)
    return distributions, c_extended or gamma_extended or eps_extended, notes


def best_params_from_row(row: pd.Series, task: str) -> dict:
    params = {"model__C": float(value_from_row(row, "C")), "model__gamma": value_from_row(row, "gamma")}
    if task == "regression":
        params["model__epsilon"] = float(value_from_row(row, "epsilon"))
    else:
        params["model__class_weight"] = value_from_row(row, "class_weight")
    return params


def curve_point(full_train: pd.DataFrame, fit_idx: np.ndarray, val_idx: np.ndarray, size: int, feature_set: str, task: str, params: dict, threshold: float) -> dict:
    features = FEATURE_SETS[feature_set]
    fold_train = full_train.iloc[fit_idx]
    keep, _ = train_test_split(
        np.arange(len(fold_train)), train_size=size, random_state=SEED, stratify=strata_for(fold_train, threshold)
    )
    fit_rows = fold_train.iloc[np.sort(keep)]
    validation = full_train.iloc[val_idx]
    model = make_pipeline(task, features).set_params(**params)
    if task == "regression":
        model.fit(fit_rows[features], fit_rows["hospital_admissions"])
        predicted = model.predict(validation[features])
        actual = validation["hospital_admissions"]
        return {
            "train_size": size,
            "rmse": root_mean_squared_error(actual, predicted),
            "mae": mean_absolute_error(actual, predicted),
            "r2": r2_score(actual, predicted),
        }
    model.fit(fit_rows[features], (fit_rows["hospital_admissions"] >= threshold).astype(int))
    scores = model.decision_function(validation[features])
    actual = (validation["hospital_admissions"] >= threshold).astype(int)
    return {"train_size": size, "pr_auc": average_precision_score(actual, scores), "roc_auc": roc_auc_score(actual, scores)}


def run_learning_curve(full_train: pd.DataFrame, feature_set: str, task: str, params: dict, threshold: float) -> pd.DataFrame:
    folds = cv_splits(full_train, threshold, 3)
    points = Parallel(n_jobs=2)(
        delayed(curve_point)(full_train, fit_idx, val_idx, size, feature_set, task, params, threshold)
        for fit_idx, val_idx in folds
        for size in LEARNING_SIZES
    )
    long = pd.DataFrame(points).melt(id_vars="train_size", var_name="metric", value_name="score")
    summary = long.groupby(["metric", "train_size"])["score"].agg(cv_mean="mean", cv_sd="std").reset_index()
    summary.insert(0, "feature_set", feature_set)
    summary.insert(1, "task", task)
    summary["rows_used"] = len(full_train)
    summary["mode"] = RUN_MODE
    return summary


def update_curve_plot(curves: pd.DataFrame, task: str) -> None:
    metric = "rmse" if task == "regression" else "pr_auc"
    subset = curves[(curves["task"] == task) & (curves["metric"] == metric)]
    if subset.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for feature_set, group in subset.groupby("feature_set"):
        axis.plot(group["train_size"], group["cv_mean"], marker="o", label=feature_set)
    axis.set(xlabel="Training rows", ylabel=metric, title=f"SVM learning curve ({task})")
    axis.legend()
    figure.tight_layout()
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_FIGURES / f"07_svm_learning_curve_{task}{SUFFIX}.png", dpi=150)
    plt.close(figure)


def timing_pilot(train: pd.DataFrame, threshold: float) -> None:
    if not FAST:
        return
    # FULL settings, C=100 on FS-Primary: a deliberately pessimistic per-fit cost
    features = FEATURE_SETS["FS-Primary"]
    strata = strata_for(train, threshold)
    total = 0.0
    for task in ["regression", "classification"]:
        seconds = {}
        for size in [2_000, 4_000]:
            keep, _ = train_test_split(np.arange(len(train)), train_size=size, random_state=SEED, stratify=strata)
            pilot = train.iloc[np.sort(keep)]
            target = pilot["hospital_admissions"] if task == "regression" else (pilot["hospital_admissions"] >= threshold).astype(int)
            params = {"model__C": 100.0, "model__gamma": 0.1}
            params.update({"model__epsilon": 0.5} if task == "regression" else {"model__class_weight": None})
            model = make_pipeline(task, features).set_params(**params)
            started = time.perf_counter()
            model.fit(pilot[features], target)
            seconds[size] = time.perf_counter() - started
        power = np.log(seconds[4_000] / seconds[2_000]) / np.log(2)
        cost = lambda rows: seconds[4_000] * (rows / 4_000) ** power
        per_feature_set = (
            40 * 5 * cost(0.8 * 10_000)
            + 20 * 5 * cost(0.8 * 20_000)
            + 3 * sum(cost(size) for size in [5_000, 10_000, 20_000, 40_000])
            + cost(20_000)
        )
        total += 2 * per_feature_set / 2
        print(f"pilot {task}: {seconds[2_000]:.1f}s at 2k, {seconds[4_000]:.1f}s at 4k (time ~ n^{power:.2f})")
    print(f"projected FULL run (C=100 everywhere, n_jobs=2): about {total / 3600:.1f} hours")


def final_summary(row: pd.Series, task: str, feature_set: str, decision: str, rows: int, support: int, seconds: float) -> dict:
    metrics = ["rmse", "mae", "r2"] if task == "regression" else ["pr_auc", "roc_auc"]
    summary = {
        "feature_set": feature_set,
        "task": task,
        "best_params": json.dumps(clean_params(best_params_from_row(row, task)), default=str),
        "cv_rows": row["rows_used"],
        "n_iter": row["n_iter"],
        "cv": row["cv"],
        "edge_extended": row.get("edge_extended", False),
        "edge_note": row.get("edge_note", ""),
        "final_fit_rows": rows,
        "learning_curve_decision": decision,
        "n_support_vectors": support,
        "fit_seconds": seconds,
        "mode": RUN_MODE,
    }
    for metric in metrics:
        summary[f"cv_{metric}_mean"] = row.get(f"mean_test_{metric}")
        summary[f"cv_{metric}_sd"] = row.get(f"std_test_{metric}")
    if task == "regression":
        # reference only (not used for selection): Cherkassky & Ma (2004) eps = 3*sigma*sqrt(ln n / n),
        # sigma = noise SD estimated by the PM2.5 baseline CV RMSE
        sigma = float(pd.read_csv(PROJECT_ROOT / "results" / "tables" / "06_baselines.csv").set_index("baseline").loc["pm25_linear_regression", "cv_rmse_mean"])
        n = int(0.8 * row["rows_used"])
        summary["epsilon_cherkassky_ref"] = 3 * sigma * np.sqrt(np.log(n) / n)
        summary["noise_sd_ref"] = sigma
    return summary


if __name__ == "__main__":
    if FAST:
        print("FAST MODE - NOT FOR REPORT")
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    data = load_raw()
    train = load_train(data)
    threshold = load_threshold()
    phase_a, phase_b = nested_samples(train, threshold)
    phase_a_path = output_path("07_svm_phaseA_cv")
    phase_b_path = output_path("07_svm_phaseB_cv")
    curve_path = output_path("07_svm_learning_curve")
    best_path = output_path("07_svm_best")
    phase_a_table = load_table(phase_a_path)
    phase_b_table = load_table(phase_b_path)
    curve_table = load_table(curve_path)
    best_table = load_table(best_path)
    specs = [(feature_set, task) for feature_set in FEATURE_SETS for task in ["regression", "classification"]]

    timing_pilot(train, threshold)

    for feature_set, task in specs:
        if f"{feature_set}|{task}" in search_key(phase_a_table):
            continue
        result = run_search(phase_a, feature_set, task, phase_a_distributions(task), PHASE_A_ITER, CV_FOLDS_A, "phaseA")
        phase_a_table = pd.concat([phase_a_table, result], ignore_index=True)
        save_table(phase_a_table, phase_a_path)
        append_progress("phaseA", feature_set, task)
        print(f"phaseA complete: {feature_set} {task}")

    for feature_set, task in specs:
        if f"{feature_set}|{task}" in search_key(phase_b_table):
            continue
        row_a = best_row(phase_a_table, feature_set, task)
        distributions, edge_extended, edge_note = phase_b_distributions(task, row_a)
        result = run_search(phase_b, feature_set, task, distributions, PHASE_B_ITER, CV_FOLDS_B, "phaseB")
        result["edge_extended"] = edge_extended
        result["edge_note"] = edge_note
        phase_b_table = pd.concat([phase_b_table, result], ignore_index=True)
        save_table(phase_b_table, phase_b_path)
        append_progress("phaseB", feature_set, task, f"edge_extended={edge_extended}; {edge_note}")
        print(f"phaseB complete: {feature_set} {task}; edge extension={edge_extended} {edge_note}")

    for feature_set, task in specs:
        if f"{feature_set}|{task}" in search_key(curve_table):
            continue
        row_b = best_row(phase_b_table, feature_set, task)
        result = run_learning_curve(train, feature_set, task, best_params_from_row(row_b, task), threshold)
        curve_table = pd.concat([curve_table, result], ignore_index=True)
        save_table(curve_table, curve_path)
        update_curve_plot(curve_table, task)
        append_progress("learning_curve", feature_set, task)
        print(f"curve complete: {feature_set} {task}")

    for feature_set, task in specs:
        model_path = OUT_MODELS / f"07_svm_{task}_{feature_set}{SUFFIX}.joblib"
        if f"{feature_set}|{task}" in search_key(best_table) and model_path.exists():
            continue
        row_b = best_row(phase_b_table, feature_set, task)
        curve = curve_table[(curve_table["feature_set"] == feature_set) & (curve_table["task"] == task)]
        refit_metric = "rmse" if task == "regression" else "pr_auc"
        if FAST:
            final_rows, final_train = PHASE_B_ROWS, phase_b
            decision = f"FAST fixed at {final_rows} rows"
        else:
            # more data is used only if the 20k -> 40k gain is larger than fold-to-fold CV noise
            metric_rows = curve[curve["metric"] == refit_metric].set_index("train_size")
            at_20, at_40 = metric_rows.loc[20_000, "cv_mean"], metric_rows.loc[40_000, "cv_mean"]
            improvement = at_20 - at_40 if task == "regression" else at_40 - at_20
            cutoff = float(metric_rows.loc[[20_000, 40_000], "cv_sd"].max())
            use_full = improvement > cutoff
            final_rows, final_train = (len(train), train) if use_full else (PHASE_B_ROWS, phase_b)
            decision = f"{'FULL' if use_full else '20k'}; improvement={improvement:.4f}; cutoff={cutoff}"
        model = make_pipeline(task, FEATURE_SETS[feature_set]).set_params(**best_params_from_row(row_b, task))
        target = final_train["hospital_admissions"] if task == "regression" else (final_train["hospital_admissions"] >= threshold).astype(int)
        started = time.perf_counter()
        model.fit(final_train[FEATURE_SETS[feature_set]], target)
        seconds = time.perf_counter() - started
        support = len(model.named_steps["model"].support_)
        OUT_MODELS.mkdir(parents=True, exist_ok=True)
        dump(model, model_path)
        best_table = pd.concat(
            [best_table, pd.DataFrame([final_summary(row_b, task, feature_set, decision, final_rows, support, seconds)])],
            ignore_index=True,
        )
        save_table(best_table, best_path)
        append_progress("final_fit", feature_set, task)
        print(f"final fit complete: {feature_set} {task}; {final_rows:,} rows; {support:,} support vectors")

    print(f"saved {OUT_TABLES.relative_to(PROJECT_ROOT)} and {OUT_FIGURES.relative_to(PROJECT_ROOT)}")
    print("test set not loaded")
