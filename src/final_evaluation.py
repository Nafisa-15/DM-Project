import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from config import FAST, FEATURE_SETS, PROJECT_ROOT, RUN_MODE, SEED, SPLIT_PATH
from data import load_raw
from tune_linear_tree import model_and_space, pipeline

N_BOOT = 1_000
TIE_MARGIN = 0.05  # tie rule (docs/PROJECT_RULES.md), fixed before any results
LEAK_R2 = 0.20  # METHODOLOGY Step 9 sanity alarm (PM2.5 alone caps R2 near 0.154)
COLORS = {"Linear": "#2a78d6", "Tree": "#eb6834", "SVM": "#1baf7a", "MLP": "#eda100"}  # dataviz palette, validated
INK, MUTED = "#0b0b0b", "#898781"


def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value
    return value


def tuned_params(tables_dir: Path, suffix: str) -> dict:
    best = pd.read_csv(tables_dir / f"07_best_linear_tree{suffix}.csv")
    return {(r.model, r.task, r.feature_set): {f"model__{k}": clean(v) for k, v in json.loads(r.best_params).items()} for r in best.itertuples()}


def frozen_models(train: pd.DataFrame, threshold: float, models_dir: Path, tables_dir: Path, suffix: str) -> list[dict]:
    # every setting was fixed on training data (Step 7); nothing here looks at the test rows
    params = tuned_params(tables_dir, suffix)
    y_reg = train["hospital_admissions"]
    y_clf = (y_reg >= threshold).astype(int)
    entries = [
        {"name": "Mean predictor", "family": "Baseline", "task": "regression", "fs": "-", "features": ["pm2_5"], "models": [DummyRegressor().fit(train[["pm2_5"]], y_reg)]},
        {"name": "PM2.5-only linear", "family": "Baseline", "task": "regression", "fs": "FS-PM25", "features": ["pm2_5"], "models": [LinearRegression().fit(train[["pm2_5"]], y_reg)]},
        {"name": "Prevalence", "family": "Baseline", "task": "classification", "fs": "-", "features": ["pm2_5"], "models": [DummyClassifier(strategy="prior").fit(train[["pm2_5"]], y_clf)]},
    ]
    for fs, features in FEATURE_SETS.items():
        for model, task, family in [("Linear Regression", "regression", "Linear"), ("Logistic Regression", "classification", "Linear"),
                                    ("Decision Tree", "regression", "Tree"), ("Decision Tree", "classification", "Tree")]:
            pipe = model_and_space(model, task, features, [])[0].set_params(**params[(model, task, fs)])
            pipe.fit(train[features], y_reg if task == "regression" else y_clf)
            entries.append({"name": model, "family": family, "task": task, "fs": fs, "features": features, "models": [pipe]})
        for task in ["regression", "classification"]:
            svm = load(models_dir / f"07_svm_{task}_{fs}{suffix}.joblib")
            entries.append({"name": "SVR" if task == "regression" else "SVC", "family": "SVM", "task": task, "fs": fs, "features": features, "models": [svm]})
            seeds = sorted(models_dir.glob(f"07_mlp_{task}_{fs}_seed*{suffix}.joblib"))
            seeds = [path for path in seeds if suffix or "_FAST" not in path.name]
            entries.append({"name": "MLP", "family": "MLP", "task": task, "fs": fs, "features": features, "models": [load(path) for path in seeds]})
    return entries


def predictions(entry: dict, test: pd.DataFrame) -> list[np.ndarray]:
    X = test[entry["features"]]
    if entry["task"] == "regression":
        return [np.clip(model.predict(X), 0, None) for model in entry["models"]]  # METHODOLOGY: clip at 0
    scores = []
    for model in entry["models"]:
        scores.append(model.decision_function(X) if hasattr(model, "decision_function") and not hasattr(model, "predict_proba") else model.predict_proba(X)[:, 1])
    return scores


def regression_draws(y: np.ndarray, preds: list[np.ndarray], idx: np.ndarray) -> dict:
    # metric per bootstrap draw, averaged over seeds for the MLP; row 0 = the full test set
    out = {"rmse": [], "mae": [], "r2": [], "poisson_dev": []}
    full = np.arange(len(y))[None, :]
    rows = np.vstack([full, idx])
    sst = y[rows].var(axis=1)
    for p in preds:
        se, ae = (y - p) ** 2, np.abs(y - p)
        pc = np.maximum(p, 1e-6)  # deviance needs a positive mean
        dev = 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / pc), 0) - (y - pc))
        mse = se[rows].mean(axis=1)
        out["rmse"].append(np.sqrt(mse))
        out["mae"].append(ae[rows].mean(axis=1))
        out["r2"].append(1 - mse / sst)
        out["poisson_dev"].append(dev[rows].mean(axis=1))
    return {metric: np.mean(values, axis=0) for metric, values in out.items()}


def classification_draws(y: np.ndarray, scores: list[np.ndarray], idx: np.ndarray) -> dict:
    rows = np.vstack([np.arange(len(y))[None, :], idx])
    out = {"pr_auc": [], "roc_auc": []}
    for s in scores:
        out["pr_auc"].append([average_precision_score(y[r], s[r]) for r in rows])
        out["roc_auc"].append([roc_auc_score(y[r], s[r]) for r in rows])
    return {metric: np.mean(values, axis=0) for metric, values in out.items()}


def summarise(draws: np.ndarray) -> tuple[float, float, float]:
    return float(draws[0]), float(np.percentile(draws[1:], 2.5)), float(np.percentile(draws[1:], 97.5))


def verdict(delta: float, low: float, high: float, lower_is_better: bool, margin: float | None) -> str:
    if (low <= 0 <= high) or (margin is not None and abs(delta) < margin):
        return "tie"
    better = delta < 0 if lower_is_better else delta > 0
    return "better" if better else "worse"


def evaluate(train: pd.DataFrame, test: pd.DataFrame, threshold: float, models_dir: Path, tables_dir: Path, out_dir: Path,
             in_suffix: str, out_suffix: str, n_boot: int) -> dict:
    entries = frozen_models(train, threshold, models_dir, tables_dir, in_suffix)
    y = test["hospital_admissions"].to_numpy(dtype=float)
    spike = (test["hospital_admissions"] >= threshold).astype(int).to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(test), size=(n_boot, len(test)), dtype=np.int32)
    pred_frame = test[["city", "pm2_5", "hospital_admissions"]].assign(spike=spike)
    draws = {}
    for entry in entries:
        key = (entry["task"], entry["name"], entry["fs"])
        preds = predictions(entry, test)
        draws[key] = regression_draws(y, preds, idx) if entry["task"] == "regression" else classification_draws(spike, preds, idx)
        entry["mean_prediction"] = np.mean(preds, axis=0)
        pred_frame[f"{entry['task'][:3]}|{entry['name']}|{entry['fs']}"] = entry["mean_prediction"]
        for seed_number, p in enumerate(preds if len(preds) > 1 else []):
            pred_frame[f"{entry['task'][:3]}|{entry['name']}|{entry['fs']}|seed{seed_number}"] = p

    def table(task: str, metrics: list[str]) -> pd.DataFrame:
        rows = []
        for entry in entries:
            if entry["task"] != task:
                continue
            row = {"model": entry["name"], "family": entry["family"], "feature_set": entry["fs"], "n_models": len(entry["models"])}
            for metric in metrics:
                row[metric], row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = summarise(draws[(task, entry["name"], entry["fs"])][metric])
            rows.append(row)
        return pd.DataFrame(rows)

    reg = table("regression", ["rmse", "mae", "r2", "poisson_dev"])
    reg["leakage_alarm"] = reg["r2"] > LEAK_R2
    clf = table("classification", ["pr_auc", "roc_auc"])

    def paired(task: str, metric: str, references: list[tuple[str, str]], lower_is_better: bool, margin: float | None) -> pd.DataFrame:
        rows = []
        for entry in entries:
            if entry["task"] != task:
                continue
            for ref_name, ref_fs in references:
                key, ref = (task, entry["name"], entry["fs"]), (task, ref_name, ref_fs)
                if key == ref:
                    continue
                # paired bootstrap: both models scored on the same resampled rows
                delta, low, high = summarise(draws[key][metric] - draws[ref][metric])
                rows.append({"model": entry["name"], "feature_set": entry["fs"], "reference": f"{ref_name} | {ref_fs}", "metric": metric,
                             "delta": delta, "ci_low": low, "ci_high": high, "verdict": verdict(delta, low, high, lower_is_better, margin)})
        return pd.DataFrame(rows)

    reg_pairs = paired("regression", "rmse", [("Linear Regression", "FS-Primary"), ("PM2.5-only linear", "FS-PM25")], True, TIE_MARGIN)
    clf_pairs = paired("classification", "pr_auc", [("Logistic Regression", "FS-Primary"), ("Prevalence", "-")], False, None)
    h1_rows = []
    for task, metric, lower in [("regression", "rmse", True), ("classification", "pr_auc", False)]:
        for name in sorted({e["name"] for e in entries if e["task"] == task and e["fs"] == "FS-Primary"}):
            primary, pm25 = draws[(task, name, "FS-Primary")][metric], draws[(task, name, "FS-PM25")][metric]
            delta, low, high = summarise(primary - pm25)
            h1_rows.append({"model": name, "task": task, "metric": metric, "fs_primary": float(primary[0]), "fs_pm25": float(pm25[0]),
                            "delta_primary_minus_pm25": delta, "ci_low": low, "ci_high": high,
                            "verdict": verdict(delta, low, high, lower, TIE_MARGIN if task == "regression" else None)})
    h1 = pd.DataFrame(h1_rows)

    tables_out = out_dir / "tables"
    tables_out.mkdir(parents=True, exist_ok=True)
    meta = {"mode": RUN_MODE, "n_test": len(test), "n_train": len(train), "n_boot": n_boot, "seed": SEED}
    outputs = {"09_test_regression": reg, "09_test_classification": clf, "09_test_paired_regression": reg_pairs,
               "09_test_paired_classification": clf_pairs, "09_test_h1": h1, "09_test_predictions": pred_frame}
    for name, frame in outputs.items():
        frame.assign(mode=RUN_MODE).to_csv(tables_out / f"{name}{out_suffix}.csv", index=False)
    (tables_out / f"09_test_run{out_suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    make_figures(entries, test, y, out_dir / "figures", out_suffix, reg)
    return {"regression": reg, "classification": clf, "h1": h1}


def make_figures(entries: list[dict], test: pd.DataFrame, y: np.ndarray, figures_dir: Path, suffix: str, reg: pd.DataFrame) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    main = [e for e in entries if e["task"] == "regression" and e["fs"] == "FS-Primary"]

    # calibration by PM2.5 decile: mean actual (with 95% CI) vs mean prediction per model
    decile = pd.qcut(test["pm2_5"], 10, labels=False) + 1
    actual = pd.DataFrame({"d": decile, "y": y}).groupby("d")["y"].agg(["mean", "std", "size"])
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.errorbar(actual.index, actual["mean"], yerr=1.96 * actual["std"] / np.sqrt(actual["size"]), fmt="o", color=INK,
                  markersize=5, capsize=3, label="Actual (mean, 95% CI)", zorder=3)
    for entry in main:
        means = pd.Series(entry["mean_prediction"]).groupby(decile.to_numpy()).mean()
        # lines nearly overlap (all models learn the same trend), so a legend labels them, not direct labels
        axis.plot(means.index, means.values, color=COLORS[entry["family"]], linewidth=2, label=entry["name"])
    axis.set(title="Test set: mean admissions by PM2.5 decile (FS-Primary)", xlabel="PM2.5 decile (test set)", ylabel="Mean daily admissions",
             xticks=range(1, 11))
    axis.grid(axis="y", color="#e4e3df", linewidth=0.8)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(figures_dir / f"09_calibration_pm25_deciles{suffix}.png", dpi=150)
    plt.close(figure)

    # predicted vs actual and residuals, one panel per model (small multiples, shared axes)
    for kind in ["pred_vs_actual", "residuals"]:
        figure, axes = plt.subplots(1, len(main), figsize=(3.2 * len(main), 3.3), sharex=True, sharey=True)
        for axis, entry in zip(np.atleast_1d(axes), main):
            p = entry["mean_prediction"]
            values = (p, y) if kind == "pred_vs_actual" else (p, y - p)
            axis.hexbin(*values, gridsize=30, cmap="Blues", mincnt=1, linewidths=0)
            if kind == "pred_vs_actual":
                axis.plot([0, 25], [0, 25], color=MUTED, linewidth=1, linestyle="--")
            else:
                axis.axhline(0, color=MUTED, linewidth=1, linestyle="--")
            axis.set_title(entry["name"], fontsize=9, color=INK)
            axis.set_xlabel("Predicted admissions")
        np.atleast_1d(axes)[0].set_ylabel("Actual admissions" if kind == "pred_vs_actual" else "Residual (actual - predicted)")
        figure.suptitle(f"Test set: {'predicted vs actual' if kind == 'pred_vs_actual' else 'residuals'} (FS-Primary)", fontsize=10)
        figure.tight_layout()
        figure.savefig(figures_dir / f"09_{kind}{suffix}.png", dpi=150)
        plt.close(figure)

    # test RMSE with 95% bootstrap CI, both feature sets
    shown = reg[reg["model"] != "Mean predictor"].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    labels = [f"{r.model} ({r.feature_set})" for r in shown.itertuples()]
    for i, r in enumerate(shown.itertuples()):
        color = COLORS.get(r.family, INK)
        axis.errorbar(r.rmse, i, xerr=[[r.rmse - r.rmse_ci_low], [r.rmse_ci_high - r.rmse]], fmt="o", color=color, markersize=6, capsize=3)
    baseline = shown.loc[shown["model"] == "PM2.5-only linear", "rmse"]
    if len(baseline):
        axis.axvspan(baseline.iloc[0] - TIE_MARGIN, baseline.iloc[0] + TIE_MARGIN, color="#f0efec", zorder=0)
        axis.axvline(baseline.iloc[0], color=MUTED, linewidth=1, linestyle="--")
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set(title="Test RMSE with 95% bootstrap CI",
             xlabel="RMSE, lower is better (dashed: PM2.5-only baseline; shaded: ±0.05 tie margin)")
    axis.tick_params(axis="y", labelcolor="#52514e")
    figure.tight_layout()
    figure.savefig(figures_dir / f"09_test_rmse_ci{suffix}.png", dpi=150)
    plt.close(figure)


class SeedAverage:
    # the MLP is reported as the average of its 5 seed models (METHODOLOGY 7.2)
    def __init__(self, models: list):
        self.models = models

    def fit(self, X, y=None):
        return self  # already trained; required by sklearn's permutation_importance API, never called with refitting

    def predict(self, X):
        return np.mean([model.predict(X) for model in self.models], axis=0)


def per_city(train: pd.DataFrame, test: pd.DataFrame, tables_dir: Path, in_suffix: str, n_boot: int) -> pd.DataFrame:
    # Step 10.1: pooled vs per-city models, on each city's test rows. Linear Regression and the CV-best
    # regression model (Decision Tree, chosen on CV in Step 7, not on test). Per-city models drop the city
    # columns and use settings tuned by CV on that city's own training rows (analysis_train.py).
    params = tuned_params(tables_dir, in_suffix)
    city_params = pd.read_csv(tables_dir / f"10_city_params{in_suffix}.csv").set_index(["model", "city"])["best_params"]
    features = FEATURE_SETS["FS-Primary"]
    numeric = [f for f in features if f != "city"]
    rng = np.random.default_rng(SEED)
    rows = []
    for name in ["Linear Regression", "Decision Tree"]:
        setting = params[(name, "regression", "FS-Primary")]
        pooled = model_and_space(name, "regression", features, [])[0].set_params(**setting).fit(train[features], train["hospital_admissions"])
        cities = sorted(test["city"].unique())
        mse_pooled, mse_city, sizes, point = [], [], [], []
        for city in cities:
            tr, te = train[train["city"] == city], test[test["city"] == city]
            local_setting = json.loads(city_params.loc[(name, city)])
            local = model_and_space(name, "regression", numeric, [])[0].set_params(**local_setting).fit(tr[numeric], tr["hospital_admissions"])
            y = te["hospital_admissions"].to_numpy(dtype=float)
            se_p = (y - np.clip(pooled.predict(te[features]), 0, None)) ** 2
            se_c = (y - np.clip(local.predict(te[numeric]), 0, None)) ** 2
            idx = np.vstack([np.arange(len(y))[None, :], rng.integers(0, len(y), size=(n_boot, len(y)))])
            mse_pooled.append(se_p[idx].mean(axis=1))
            mse_city.append(se_c[idx].mean(axis=1))
            sizes.append(len(y))
            delta, low, high = summarise(np.sqrt(mse_city[-1]) - np.sqrt(mse_pooled[-1]))
            rows.append({"model": name, "city": city, "n_train": len(tr), "n_test": len(y), "rmse_pooled": float(np.sqrt(mse_pooled[-1][0])),
                         "rmse_per_city": float(np.sqrt(mse_city[-1][0])), "delta_city_minus_pooled": delta, "ci_low": low, "ci_high": high,
                         "verdict": verdict(delta, low, high, True, TIE_MARGIN)})
        weights = np.array(sizes)[:, None] / sum(sizes)
        for label, pooled_draws, city_draws in [
            ("Weighted (all test rows)", np.sqrt((weights * np.array(mse_pooled)).sum(axis=0)), np.sqrt((weights * np.array(mse_city)).sum(axis=0))),
            ("Macro (mean of 8 cities)", np.sqrt(np.array(mse_pooled)).mean(axis=0), np.sqrt(np.array(mse_city)).mean(axis=0)),
        ]:
            delta, low, high = summarise(city_draws - pooled_draws)
            rows.append({"model": name, "city": label, "n_train": len(train), "n_test": sum(sizes), "rmse_pooled": float(pooled_draws[0]),
                         "rmse_per_city": float(city_draws[0]), "delta_city_minus_pooled": delta, "ci_low": low, "ci_high": high,
                         "verdict": verdict(delta, low, high, True, TIE_MARGIN)})
    return pd.DataFrame(rows)


def thresholded(train: pd.DataFrame, test: pd.DataFrame, threshold: float, tables_dir: Path, in_suffix: str, predictions: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    # Step 11: thresholds chosen on out-of-fold TRAINING scores (analysis_train.py) applied to the saved test scores
    cuts = pd.read_csv(tables_dir / f"11_thresholds{in_suffix}.csv")
    y = (test["hospital_admissions"] >= threshold).astype(int).to_numpy()
    params = tuned_params(tables_dir, in_suffix)
    balanced = pipeline(LogisticRegression(solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=SEED), FEATURE_SETS["FS-Primary"], True)
    balanced.set_params(**params[("Logistic Regression", "classification", "FS-Primary")])
    balanced.fit(train[FEATURE_SETS["FS-Primary"]], (train["hospital_admissions"] >= threshold).astype(int))
    idx = np.vstack([np.arange(len(y))[None, :], np.random.default_rng(SEED).integers(0, len(y), size=(n_boot, len(y)))])
    rows = []
    for cut in cuts.itertuples():
        if cut.model == "Logistic Regression (balanced)":
            scores = balanced.predict_proba(test[FEATURE_SETS["FS-Primary"]])[:, 1]
        else:
            scores = predictions[f"cla|{cut.model}|{cut.feature_set}"].to_numpy()
        flag = (scores >= cut.threshold).astype(int)
        tp, fp = (flag & y)[idx].sum(axis=1), (flag & (1 - y))[idx].sum(axis=1)
        fn, tn = ((1 - flag) & y)[idx].sum(axis=1), ((1 - flag) & (1 - y))[idx].sum(axis=1)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / np.maximum(tp + fn, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        balanced_acc = (recall + tn / np.maximum(tn + fp, 1)) / 2
        row = {"model": cut.model, "feature_set": cut.feature_set, "threshold": cut.threshold, "flagged_share": float(flag.mean()),
               "tp": int(tp[0]), "fp": int(fp[0]), "fn": int(fn[0]), "tn": int(tn[0])}
        for metric, draws in [("precision", precision), ("recall", recall), ("f1", f1), ("balanced_accuracy", balanced_acc)]:
            row[metric], row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = summarise(draws)
        rows.append(row)
    return pd.DataFrame(rows)


def importance(test: pd.DataFrame, train: pd.DataFrame, models_dir: Path, tables_dir: Path, in_suffix: str) -> pd.DataFrame:
    # Step 12: permutation importance on the test set, 5 repeats (docs/PROJECT_RULES.md); the CV-best model (Decision
    # Tree) on all test rows, the MLP on a 3,000-row test subsample. Columns are permuted raw, so city is one group.
    params = tuned_params(tables_dir, in_suffix)
    features = FEATURE_SETS["FS-Primary"]
    tree = model_and_space("Decision Tree", "regression", features, [])[0].set_params(**params[("Decision Tree", "regression", "FS-Primary")])
    tree.fit(train[features], train["hospital_admissions"])
    seeds = sorted(models_dir.glob(f"07_mlp_regression_FS-Primary_seed*{in_suffix}.joblib"))
    mlp = SeedAverage([load(p) for p in seeds if in_suffix or "_FAST" not in p.name])
    sub = test.sample(n=min(3_000, len(test)), random_state=SEED)

    def neg_rmse(estimator, X, y):
        return -float(np.sqrt(np.mean((y - np.clip(estimator.predict(X), 0, None)) ** 2)))

    rows = []
    for name, model, rows_used in [("Decision Tree", tree, test), ("MLP", mlp, sub)]:
        result = permutation_importance(model, rows_used[features], rows_used["hospital_admissions"].to_numpy(dtype=float),
                                        scoring=neg_rmse, n_repeats=5, random_state=SEED, n_jobs=1)
        for feature, mean, sd in zip(features, result.importances_mean, result.importances_std):
            rows.append({"model": name, "feature": feature, "rmse_increase_mean": float(mean), "rmse_increase_sd": float(sd),
                         "n_rows": len(rows_used), "n_repeats": 5})
    return pd.DataFrame(rows)


def importance_figure(imp: pd.DataFrame, figures_dir: Path, suffix: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for axis, (name, group) in zip(axes, imp.groupby("model", sort=False)):
        group = group.sort_values("rmse_increase_mean")
        family = "Tree" if name == "Decision Tree" else "MLP"
        axis.barh(group["feature"], group["rmse_increase_mean"], xerr=group["rmse_increase_sd"], color=COLORS[family], height=0.6,
                  error_kw={"ecolor": MUTED, "linewidth": 1})
        axis.axvline(0, color=MUTED, linewidth=1)
        axis.set_title(f"{name} (n = {int(group['n_rows'].iloc[0]):,} test rows)", fontsize=9, color=INK)
        axis.set_xlabel("RMSE increase when shuffled")
        axis.tick_params(axis="y", labelcolor="#52514e")
    figure.suptitle("Permutation importance on the test set (5 repeats, FS-Primary)", fontsize=10)
    figure.tight_layout()
    figure.savefig(figures_dir / f"12_permutation_importance{suffix}.png", dpi=150)
    plt.close(figure)


def evaluate_extra(train: pd.DataFrame, test: pd.DataFrame, threshold: float, models_dir: Path, tables_dir: Path, out_dir: Path,
                   in_suffix: str, out_suffix: str, n_boot: int) -> dict:
    tables_out, figures_out = out_dir / "tables", out_dir / "figures"
    predictions = pd.read_csv(tables_out / f"09_test_predictions{out_suffix}.csv")
    done = {}
    for name, make in [("10_test_pooled_vs_city", lambda: per_city(train, test, tables_dir, in_suffix, n_boot)),
                       ("11_test_thresholded", lambda: thresholded(train, test, threshold, tables_dir, in_suffix, predictions, n_boot)),
                       ("12_permutation_importance", lambda: importance(test, train, models_dir, tables_dir, in_suffix))]:
        path = tables_out / f"{name}{out_suffix}.csv"
        if path.exists():
            continue
        frame = make()
        frame.assign(mode=RUN_MODE).to_csv(path, index=False)
        if name == "12_permutation_importance":
            importance_figure(frame, figures_out, out_suffix)
        done[name] = frame
    return done


if __name__ == "__main__":
    assert not FAST, "final_evaluation.py uses the test set and runs in FULL mode only (docs/PROJECT_RULES.md)"
    from check_results import find_problems

    problems = find_problems()
    assert not problems, f"check_results failed: {problems[:5]}"
    data = load_raw()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train = data.iloc[np.array(split["train_indices"], dtype=int)].reset_index(drop=True)
    test = data.iloc[np.array(split["test_indices"], dtype=int)].reset_index(drop=True)
    threshold = float(json.loads((PROJECT_ROOT / "results" / "tables" / "03_spike_threshold.json").read_text())["threshold"])
    results_dir = PROJECT_ROOT / "results"
    if not (results_dir / "tables" / "09_test_regression.csv").exists():
        out = evaluate(train, test, threshold, results_dir / "models", results_dir / "tables", results_dir, "", "", N_BOOT)
        reg, clf = out["regression"], out["classification"]
        print(reg[["model", "feature_set", "rmse", "rmse_ci_low", "rmse_ci_high", "r2"]].round(4).to_string(index=False))
        print(clf[["model", "feature_set", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high"]].round(4).to_string(index=False))
        if reg["leakage_alarm"].any():
            print("LEAKAGE ALARM: test R2 > 0.20 - audit before reporting anything")
    else:
        print("Step 9 outputs exist - not recomputed")
    assert (results_dir / "tables" / "11_thresholds.csv").exists(), "run analysis_train.py (FULL) first: Step 11 thresholds come from training data"
    extra = evaluate_extra(train, test, threshold, results_dir / "models", results_dir / "tables", results_dir, "", "", N_BOOT)
    for name, frame in extra.items():
        print(f"{name}: {len(frame)} rows")
    print(f"test set: {len(test):,} rows; all settings and thresholds were fixed on training data before this run")
