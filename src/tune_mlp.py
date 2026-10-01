import json
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FAST, FEATURE_SETS, PROJECT_ROOT, RUN_MODE, SEED
from data import load_raw

SUFFIX = "_FAST" if FAST else ""
OUT_DIR = PROJECT_ROOT / ("results_fast" if FAST else "results")
OUT_TABLES, OUT_FIGURES, OUT_MODELS, OUT_LOGS = (OUT_DIR / name for name in ["tables", "figures", "models", "logs"])
THRESHOLD_PATH = PROJECT_ROOT / "results" / "tables" / "03_spike_threshold.json"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_indices.json"

FAST_ROWS = 5_000
CV_FOLDS = 3 if FAST else 5
# regression landscape is flat on this data; classification varies more, so it gets more draws
N_ITER = {"regression": 3 if FAST else 20, "classification": 3 if FAST else 30}
MAX_EPOCHS = 30 if FAST else 200
SEEDS = [SEED, SEED + 1] if FAST else [SEED + offset for offset in range(5)]
REG_SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
CLF_SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def make_pipeline(task: str, features: list[str]) -> Pipeline:
    # Goodfellow et al. (2016) ch. 8: gradient training needs standardised inputs
    numeric = [feature for feature in features if feature != "city"]
    transformers = [("numeric", StandardScaler(), numeric)]
    if "city" in features:
        transformers.append(("city", OneHotEncoder(handle_unknown="ignore"), ["city"]))
    common = dict(solver="adam", batch_size=256, max_iter=MAX_EPOCHS, n_iter_no_change=15, random_state=SEED)
    if task == "regression":
        # validation early stopping (Goodfellow et al. 2016 sec. 7.8); 10% of the training fold
        model = MLPRegressor(early_stopping=True, validation_fraction=0.1, **common)
    else:
        # sklearn's classifier early stopping monitors validation ACCURACY, which is uninformative at
        # 11.7% prevalence; stop on training log-loss instead and regularise with L2 (alpha)
        model = MLPClassifier(early_stopping=False, tol=1e-4, **common)
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", model)])


ALPHA_RANGE = (1e-6, 1e-1)
LEARNING_RATES = [3e-4, 1e-3, 3e-3]


def search_space(alpha_range: tuple = ALPHA_RANGE, rates: list = LEARNING_RATES) -> dict:
    # METHODOLOGY 7.2 architectures; sklearn has no dropout, so L2 alpha (weight decay) is the regulariser
    return {
        "model__hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)],
        "model__alpha": loguniform(*alpha_range),
        "model__learning_rate_init": rates,
    }


def load_threshold() -> float:
    return float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"])


def strata_for(data: pd.DataFrame, threshold: float) -> pd.Series:
    spike = (data["hospital_admissions"] >= threshold).astype(int)
    return data["city"].astype(str) + "_" + spike.astype(str)


def target_for(data: pd.DataFrame, task: str, threshold: float) -> pd.Series:
    if task == "regression":
        return data["hospital_admissions"]
    return (data["hospital_admissions"] >= threshold).astype(int)


def load_table(name: str) -> pd.DataFrame:
    path = OUT_TABLES / f"{name}{SUFFIX}.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    return table if "mode" in table and (table["mode"] == RUN_MODE).all() else pd.DataFrame()


def save_table(table: pd.DataFrame, name: str) -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLES / f"{name}{SUFFIX}.csv", index=False)


def log_progress(stage: str, feature_set: str, task: str, note: str = "") -> None:
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with (OUT_LOGS / f"tune_mlp.progress{SUFFIX}.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\t{stage}\t{feature_set}\t{task}\t{note}\tmode={RUN_MODE}\n")


def done(table: pd.DataFrame, feature_set: str, task: str) -> bool:
    return not table.empty and ((table["feature_set"] == feature_set) & (table["task"] == task)).any()


def one_search(train: pd.DataFrame, folds: list, feature_set: str, task: str, threshold: float, space: dict, n_iter: int, stage: str) -> pd.DataFrame:
    features = FEATURE_SETS[feature_set]
    scoring = REG_SCORING if task == "regression" else CLF_SCORING
    search = RandomizedSearchCV(
        make_pipeline(task, features),
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        refit=False,
        cv=folds,
        random_state=SEED,
        n_jobs=2,
        return_train_score=False,
    )
    search.fit(train[features], target_for(train, task, threshold))
    result = pd.DataFrame(search.cv_results_)
    for metric in ["rmse", "mae"]:
        if f"mean_test_{metric}" in result:
            result[f"mean_test_{metric}"] = -result[f"mean_test_{metric}"]
    result.insert(0, "feature_set", feature_set)
    result.insert(1, "task", task)
    result.insert(2, "stage", stage)
    result["param_model__hidden_layer_sizes"] = result["param_model__hidden_layer_sizes"].astype(str)
    result["n_iter"] = n_iter
    result["cv_folds"] = CV_FOLDS
    result["rows_used"] = len(train)
    result["mode"] = RUN_MODE
    return result


def run_search(train: pd.DataFrame, folds: list, feature_set: str, task: str, threshold: float) -> pd.DataFrame:
    result = one_search(train, folds, feature_set, task, threshold, search_space(), N_ITER[task], "main")
    note = edge_note(best_params(result, feature_set, task))
    if not note:
        return result
    # METHODOLOGY 7.2 edge rule: widen once on the edge side, search again, keep the best of both
    low, high = ALPHA_RANGE
    low = low / 100 if "alpha_low" in note else low
    high = min(high * 10, 1.0) if "alpha_high" in note else high
    rates = [1e-4, *LEARNING_RATES, 1e-2] if "lr_edge" in note else LEARNING_RATES
    extra = one_search(train, folds, feature_set, task, threshold, search_space((low, high), rates), max(N_ITER[task] // 2, 2), f"edge:{note}")
    for column in ["rank_test_rmse", "rank_test_pr_auc"]:
        if column in result:
            metric = column.replace("rank_test_", "mean_test_")
            combined = pd.concat([result, extra], ignore_index=True)
            combined[column] = combined[metric].rank(method="min", ascending=(metric == "mean_test_rmse")).astype(int)
            return combined
    return pd.concat([result, extra], ignore_index=True)


def best_params(search_table: pd.DataFrame, feature_set: str, task: str) -> dict:
    subset = search_table[(search_table["feature_set"] == feature_set) & (search_table["task"] == task)]
    rank = "rank_test_rmse" if task == "regression" else "rank_test_pr_auc"
    row = subset.loc[subset[rank].idxmin()]
    return {
        "model__hidden_layer_sizes": tuple(json.loads(row["param_model__hidden_layer_sizes"].replace("(", "[").replace(",)", "]").replace(")", "]"))),
        "model__alpha": float(row["param_model__alpha"]),
        "model__learning_rate_init": float(row["param_model__learning_rate_init"]),
    }


def edge_note(params: dict) -> str:
    # METHODOLOGY 7.2 edge rule, recorded; alpha's range is continuous, so flag the outer 10% (log scale)
    notes = []
    span = np.log10(1e-1) - np.log10(1e-6)
    if np.log10(params["model__alpha"]) - np.log10(1e-6) <= 0.1 * span:
        notes.append("alpha_low")
    if np.log10(1e-1) - np.log10(params["model__alpha"]) <= 0.1 * span:
        notes.append("alpha_high")
    if params["model__learning_rate_init"] in (min(LEARNING_RATES), max(LEARNING_RATES)):
        notes.append("lr_edge")
    return ",".join(notes)


def search_note(search_table: pd.DataFrame, feature_set: str, task: str) -> str:
    stages = search_table[(search_table["feature_set"] == feature_set) & (search_table["task"] == task)]["stage"]
    extended = [stage.split(":", 1)[1] for stage in stages.unique() if stage.startswith("edge:")]
    return extended[0] if extended else ""


def seed_stability(train: pd.DataFrame, folds: list, feature_set: str, task: str, params: dict, threshold: float, note: str) -> dict:
    # METHODOLOGY 7.2: the MLP is stochastic, so the chosen setting is re-run with 5 seeds (mean +/- SD).
    # These fixed-setting scores are the "fair" CV numbers for comparison with the other models.
    features = FEATURE_SETS[feature_set]
    scoring = REG_SCORING if task == "regression" else CLF_SCORING
    per_seed = []
    for seed in SEEDS:
        model = make_pipeline(task, features).set_params(**params, model__random_state=seed)
        scores = cross_validate(model, train[features], target_for(train, task, threshold), cv=folds, scoring=scoring, n_jobs=2)
        per_seed.append({metric: scores[f"test_{metric}"].mean() * (-1 if metric in ("rmse", "mae") else 1) for metric in scoring})
    frame = pd.DataFrame(per_seed)
    record = {"feature_set": feature_set, "task": task, "best_params": json.dumps({k.replace("model__", ""): v for k, v in params.items()}, default=str)}
    for metric in scoring:
        record[f"seed_cv_{metric}_mean"] = frame[metric].mean()
        record[f"seed_cv_{metric}_sd"] = frame[metric].std(ddof=1)
    record["edge_extended"] = bool(note)
    record["edge_note"] = note
    record["n_seeds"] = len(SEEDS)
    record["n_iter"] = N_ITER[task]
    record["cv_folds"] = CV_FOLDS
    record["rows_used"] = len(train)
    record["mode"] = RUN_MODE
    return record


def final_fits(train: pd.DataFrame, feature_set: str, task: str, params: dict, threshold: float) -> pd.DataFrame:
    features = FEATURE_SETS[feature_set]
    rows = []
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        model = make_pipeline(task, features).set_params(**params, model__random_state=seed)
        started = time.perf_counter()
        model.fit(train[features], target_for(train, task, threshold))
        seconds = time.perf_counter() - started
        net = model.named_steps["model"]
        dump(model, OUT_MODELS / f"07_mlp_{task}_{feature_set}_seed{seed}{SUFFIX}.joblib")
        validation = getattr(net, "validation_scores_", None) or [np.nan] * len(net.loss_curve_)
        for epoch, (loss, score) in enumerate(zip(net.loss_curve_, validation), start=1):
            rows.append({"feature_set": feature_set, "task": task, "seed": seed, "epoch": epoch, "train_loss": loss,
                         "validation_score": score, "epochs_run": net.n_iter_, "fit_seconds": seconds, "mode": RUN_MODE})
    return pd.DataFrame(rows)


def plot_loss_curves(curves: pd.DataFrame) -> None:
    if curves.empty:
        return
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, task in zip(axes, ["regression", "classification"]):
        subset = curves[curves["task"] == task]
        for (feature_set, seed), group in subset.groupby(["feature_set", "seed"]):
            axis.plot(group["epoch"], group["train_loss"], alpha=0.7, label=f"{feature_set} seed {seed}")
        axis.set(title=f"MLP training loss ({task})", xlabel="Epoch", ylabel="Training loss")
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(OUT_FIGURES / f"07_mlp_loss_curves{SUFFIX}.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    if FAST:
        print("FAST MODE - NOT FOR REPORT")
    data = load_raw()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train = data.iloc[np.array(split["train_indices"], dtype=int)].reset_index(drop=True)
    threshold = load_threshold()
    if FAST:
        keep, _ = train_test_split(np.arange(len(train)), train_size=FAST_ROWS, random_state=SEED, stratify=strata_for(train, threshold))
        train = train.iloc[np.sort(keep)].reset_index(drop=True)
    folds = list(StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(train, strata_for(train, threshold)))
    specs = [(feature_set, task) for feature_set in FEATURE_SETS for task in ["regression", "classification"]]

    search_table = load_table("07_mlp_cv_results")
    for feature_set, task in specs:
        if done(search_table, feature_set, task):
            continue
        search_table = pd.concat([search_table, run_search(train, folds, feature_set, task, threshold)], ignore_index=True)
        save_table(search_table, "07_mlp_cv_results")
        log_progress("search", feature_set, task)
        print(f"search complete: {feature_set} {task}")

    best_table = load_table("07_mlp_best")
    for feature_set, task in specs:
        if done(best_table, feature_set, task):
            continue
        params = best_params(search_table, feature_set, task)
        record = seed_stability(train, folds, feature_set, task, params, threshold, search_note(search_table, feature_set, task))
        best_table = pd.concat([best_table, pd.DataFrame([record])], ignore_index=True)
        save_table(best_table, "07_mlp_best")
        log_progress("seeds", feature_set, task, record["edge_note"])
        print(f"seed check complete: {feature_set} {task}")

    curve_table = load_table("07_mlp_loss_curves")
    for feature_set, task in specs:
        if done(curve_table, feature_set, task):
            continue
        curves = final_fits(train, feature_set, task, best_params(search_table, feature_set, task), threshold)
        curve_table = pd.concat([curve_table, curves], ignore_index=True)
        save_table(curve_table, "07_mlp_loss_curves")
        plot_loss_curves(curve_table)
        log_progress("final_fits", feature_set, task)
        print(f"final fits complete: {feature_set} {task}")

    cols = [c for c in best_table.columns if c.startswith("seed_cv_") and (c.endswith("rmse_mean") or c.endswith("pr_auc_mean"))]
    print(best_table[["feature_set", "task", *cols, "edge_note"]].round(4).to_string(index=False))
    print(f"trained on {len(train):,} training rows; test set not loaded")
