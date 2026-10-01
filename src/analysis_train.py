import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.tree import export_text, plot_tree

from config import FAST, FEATURE_SETS, PROJECT_ROOT, RUN_MODE, SEED, SPLIT_PATH
from data import load_raw
from final_evaluation import COLORS, INK, MUTED, tuned_params
from tune_linear_tree import model_and_space, pipeline, pruning_candidates
from tune_mlp import make_pipeline as mlp_pipeline
from tune_svm import make_pipeline as svm_pipeline
from tune_svm import nested_samples

# Steps 10.2, 11 (thresholds) and 12 (PDP, coefficients, tree) - TRAINING rows only; the test set is not loaded
SUFFIX = "_FAST" if FAST else ""
OUT_DIR = PROJECT_ROOT / ("results_fast" if FAST else "results")
TABLES, FIGURES = OUT_DIR / "tables", OUT_DIR / "figures"
CV_FOLDS = 3 if FAST else 5
SMALL_CITIES = ["São Paulo", "Cairo"]  # METHODOLOGY Step 10.4: wide intervals
NUMERIC = [f for f in FEATURE_SETS["FS-Primary"] if f != "city"]


def save(frame: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.assign(mode=RUN_MODE).to_csv(TABLES / f"{name}{SUFFIX}.csv", index=False)


def style() -> None:
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def city_slopes(train: pd.DataFrame) -> None:
    # Step 10.2: OLS admissions ~ pm2_5 * city with HC3 robust SEs (count outcome, variance/mean ~1.7)
    model = smf.ols("hospital_admissions ~ pm2_5 * C(city, Treatment('Delhi'))", data=train).fit(cov_type="HC3")
    names = list(model.params.index)
    inter = [n for n in names if n.startswith("pm2_5:")]
    R = np.zeros((len(inter), len(names)))
    for row, name in enumerate(inter):
        R[row, names.index(name)] = 1
    wald = model.wald_test(R, scalar=True)
    rows = []
    for city in sorted(train["city"].unique()):
        contrast = np.zeros(len(names))
        contrast[names.index("pm2_5")] = 1
        term = f"pm2_5:C(city, Treatment('Delhi'))[T.{city}]"
        if term in names:
            contrast[names.index(term)] = 1
        test = model.t_test(contrast)
        low, high = np.asarray(test.conf_int()).ravel()
        diff_p = float(model.pvalues[term]) if term in names else np.nan
        rows.append({"city": city, "n_train": int((train["city"] == city).sum()), "slope": float(np.asarray(test.effect).ravel()[0]),
                     "ci_low": low, "ci_high": high, "p_vs_delhi_unadjusted": diff_p, "small_sample_flag": city in SMALL_CITIES})
    slopes = pd.DataFrame(rows)
    pooled = smf.ols("hospital_admissions ~ pm2_5", data=train).fit(cov_type="HC3")
    save(slopes, "10_city_slopes")
    london = f"pm2_5:C(city, Treatment('Delhi'))[T.London]"
    # with robust (HC3) covariance statsmodels reports a Wald chi-square test, not an F test
    summary = {"interaction_wald_chi2": float(np.asarray(wald.statistic).ravel()[0]), "df": int(len(inter)),
               "interaction_p": float(wald.pvalue), "test": "Wald chi-square, HC3", "delhi_vs_london_diff": float(model.params[london]),
               "delhi_vs_london_p_unadjusted": float(model.pvalues[london]), "pooled_slope": float(pooled.params["pm2_5"]),
               "pooled_ci": [float(v) for v in pooled.conf_int().loc["pm2_5"]], "cov_type": "HC3", "mode": RUN_MODE}
    (TABLES / f"10_interaction_test{SUFFIX}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    style()
    figure, axis = plt.subplots(figsize=(7, 4.2))
    order = slopes.sort_values("slope").reset_index(drop=True)
    for i, r in order.iterrows():
        axis.errorbar(r.slope, i, xerr=[[r.slope - r.ci_low], [r.ci_high - r.slope]], fmt="o", color=COLORS["Linear"], capsize=3, markersize=5)
    axis.axvline(summary["pooled_slope"], color=MUTED, linestyle="--", linewidth=1)
    axis.set_yticks(range(len(order)), [f"{c} (n={n:,}){' *' if f else ''}" for c, n, f in zip(order.city, order.n_train, order.small_sample_flag)])
    axis.tick_params(axis="y", labelcolor="#52514e")
    axis.set(title=f"PM2.5 slope by city, 95% CI (interaction p = {summary['interaction_p']:.2f})",
             xlabel="Admissions per µg/m³ (dashed: pooled slope; * small sample)")
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"10_city_slopes{SUFFIX}.png", dpi=150)
    plt.close(figure)


def per_city_tuning(train: pd.DataFrame, threshold: float) -> None:
    # Step 10.1 fairness: a per-city model is tuned on that city's own training rows (5-fold CV), so the
    # pooled-vs-per-city comparison is not biased by settings tuned for the pooled data
    numeric = [f for f in FEATURE_SETS["FS-Primary"] if f != "city"]
    rows = []
    for city in sorted(train["city"].unique()):
        local = train[train["city"] == city].reset_index(drop=True)
        spike = (local["hospital_admissions"] >= threshold).astype(int)
        folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(local, spike))
        ccp = pruning_candidates(local, numeric, "regression")
        grids = {
            "Linear Regression": {"model__alpha": [0, 0.01, 0.1, 1, 10, 30, 100, 300, 1000]},
            "Decision Tree": {"model__max_depth": [2, 3, 4, 5, 6, 8, 10, None], "model__min_samples_leaf": [1, 5, 20, 50, 100, 200, 500, 1000],
                              "model__ccp_alpha": [0.0] + ccp},
        }
        for name, grid in grids.items():
            search = GridSearchCV(model_and_space(name, "regression", numeric, [])[0], grid, scoring="neg_root_mean_squared_error",
                                  cv=folds, n_jobs=2, refit=False)
            search.fit(local[numeric], local["hospital_admissions"])
            best = {k: (None if v is None else (float(v) if isinstance(v, (float, np.floating)) else int(v))) for k, v in search.best_params_.items()}
            rows.append({"model": name, "city": city, "n_train": len(local), "best_params": json.dumps(best), "cv_rmse": -search.best_score_,
                         "grid_size": len(search.cv_results_["params"])})
    save(pd.DataFrame(rows), "10_city_params")


def f1_threshold(y: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)[:-1]
    best = int(np.argmax(f1))
    return {"threshold": float(thresholds[best]), "oof_f1": float(f1[best]), "oof_precision": float(precision[best]),
            "oof_recall": float(recall[best]), "oof_pr_auc": float(average_precision_score(y, scores))}


def oof_scores(model, X: pd.DataFrame, y: pd.Series, folds: list) -> np.ndarray:
    # SVC has no predict_proba (probability=False, docs/PROJECT_RULES.md): its threshold lives on the decision_function scale
    if hasattr(model, "predict_proba"):
        return cross_val_predict(model, X, y, cv=folds, method="predict_proba", n_jobs=2)[:, 1]
    return cross_val_predict(model, X, y, cv=folds, method="decision_function", n_jobs=2)


def spike_thresholds(train: pd.DataFrame, threshold: float, folds: list) -> None:
    # Step 11: decision threshold = max F1 on out-of-fold training predictions, never on the test set
    params = tuned_params(TABLES, SUFFIX)
    svm_best = pd.read_csv(TABLES / f"07_svm_best{SUFFIX}.csv").set_index(["feature_set", "task"])
    mlp_best = pd.read_csv(TABLES / f"07_mlp_best{SUFFIX}.csv").set_index(["feature_set", "task"])
    y = (train["hospital_admissions"] >= threshold).astype(int)
    _, svm_sample = nested_samples(train, threshold)
    y_svm = (svm_sample["hospital_admissions"] >= threshold).astype(int)
    strata = svm_sample["city"].astype(str) + "_" + y_svm.astype(str)
    svm_folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(svm_sample, strata))
    rows = [{"model": "Prevalence", "feature_set": "-", "score": "constant", "threshold": float("-inf"), "oof_f1": 2 * y.mean() / (1 + y.mean()),
             "oof_precision": float(y.mean()), "oof_recall": 1.0, "oof_pr_auc": float(y.mean()), "rows_used": len(train)}]
    for fs, features in FEATURE_SETS.items():
        for name in ["Logistic Regression", "Decision Tree"]:
            model = model_and_space(name, "classification", features, [])[0].set_params(**params[(name, "classification", fs)])
            rows.append({"model": name, "feature_set": fs, "score": "probability", **f1_threshold(y, oof_scores(model, train[features], y, folds)), "rows_used": len(train)})
        svc = svm_pipeline("classification", features).set_params(**json.loads(svm_best.loc[(fs, "classification"), "best_params"]))
        rows.append({"model": "SVC", "feature_set": fs, "score": "decision_function",
                     **f1_threshold(y_svm, oof_scores(svc, svm_sample[features], y_svm, svm_folds)), "rows_used": len(svm_sample)})
        mlp_params = {f"model__{k}": (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(mlp_best.loc[(fs, "classification"), "best_params"]).items()}
        seeds = int(mlp_best.loc[(fs, "classification"), "n_seeds"])
        mlp_scores = np.mean([oof_scores(mlp_pipeline("classification", features).set_params(**mlp_params, model__random_state=SEED + s),
                                         train[features], y, folds) for s in range(seeds)], axis=0)
        rows.append({"model": "MLP", "feature_set": fs, "score": f"probability (mean of {seeds} seeds)", **f1_threshold(y, mlp_scores), "rows_used": len(train)})
    # METHODOLOGY Step 11 sensitivity check: class_weight="balanced" (Logistic, FS-Primary)
    balanced = pipeline(LogisticRegression(solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=SEED), FEATURE_SETS["FS-Primary"], True)
    balanced.set_params(**params[("Logistic Regression", "classification", "FS-Primary")])
    rows.append({"model": "Logistic Regression (balanced)", "feature_set": "FS-Primary", "score": "probability",
                 **f1_threshold(y, oof_scores(balanced, train[FEATURE_SETS["FS-Primary"]], y, folds)), "rows_used": len(train)})
    save(pd.DataFrame(rows), "11_thresholds")

    # early-warning output: the tuned PM2.5-only tree's leaves as PM2.5 bands with spike rates
    tree = model_and_space("Decision Tree", "classification", ["pm2_5"], [])[0].set_params(**params[("Decision Tree", "classification", "FS-PM25")])
    tree.fit(train[["pm2_5"]], y)
    (TABLES / f"11_tree_rules{SUFFIX}.txt").write_text(export_text(tree.named_steps["model"], feature_names=["pm2_5"], decimals=2), encoding="utf-8")
    leaf = tree.named_steps["model"].apply(tree.named_steps["preprocess"].transform(train[["pm2_5"]]))
    bands = train.assign(leaf=leaf, spike=y).groupby("leaf").agg(pm2_5_from=("pm2_5", "min"), pm2_5_to=("pm2_5", "max"),
                                                                 rows=("spike", "size"), spike_rate=("spike", "mean"))
    save(bands.sort_values("pm2_5_from").reset_index(drop=True), "11_tree_pm25_bands")


def interpretation(train: pd.DataFrame) -> None:
    # Step 12 (train side): partial dependence of PM2.5, linear coefficients, tree structure
    params = tuned_params(TABLES, SUFFIX)
    features = FEATURE_SETS["FS-Primary"]
    y = train["hospital_admissions"]
    linear = model_and_space("Linear Regression", "regression", features, [])[0].set_params(**params[("Linear Regression", "regression", "FS-Primary")]).fit(train[features], y)
    tree = model_and_space("Decision Tree", "regression", features, [])[0].set_params(**params[("Decision Tree", "regression", "FS-Primary")]).fit(train[features], y)
    svr = load(OUT_DIR / "models" / f"07_svm_regression_FS-Primary{SUFFIX}.joblib")
    mlps = [load(p) for p in sorted((OUT_DIR / "models").glob(f"07_mlp_regression_FS-Primary_seed*{SUFFIX}.joblib")) if SUFFIX or "_FAST" not in p.name]
    keep, _ = train_test_split(np.arange(len(train)), train_size=min(2_000, len(train) - 1), random_state=SEED)
    background = train.iloc[keep][features]
    grid = np.linspace(*np.percentile(train["pm2_5"], [1, 99]), 60)
    curves = {"Linear": [linear], "Tree": [tree], "SVM": [svr], "MLP": mlps}
    rows = []
    for family, models in curves.items():
        for value in grid:
            X = background.assign(pm2_5=value)
            rows.append({"model": family, "pm2_5": value, "partial_dependence": float(np.mean([m.predict(X).mean() for m in models]))})
    pdp = pd.DataFrame(rows)
    save(pdp, "12_partial_dependence")
    bands = train.assign(band=np.floor(train["pm2_5"] / 10) * 10).groupby("band")["hospital_admissions"].agg(["mean", "size"])
    bands = bands[bands["size"] >= 100]

    style()
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for start, r in bands.iterrows():
        axis.hlines(r["mean"], start, start + 10, color=INK, linewidth=1.2, label="Training mean per 10 µg/m³ band" if start == bands.index[0] else None)
    names = {"Linear": "Linear Regression", "Tree": "Decision Tree", "SVM": "SVR", "MLP": "MLP"}
    for family, group in pdp.groupby("model", sort=False):
        axis.plot(group["pm2_5"], group["partial_dependence"], color=COLORS[family], linewidth=2, label=names[family])
    axis.set(title="Partial dependence of predicted admissions on PM2.5 (FS-Primary, training data)",
             xlabel="PM2.5 (µg/m³)", ylabel="Predicted daily admissions", xlim=(grid[0], grid[-1]))
    axis.grid(axis="y", color="#e4e3df", linewidth=0.8)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURES / f"12_partial_dependence{SUFFIX}.png", dpi=150)
    plt.close(figure)

    # linear coefficients with 95% CIs: OLS on standardised numeric inputs + city (Delhi reference), HC3
    z = train.copy()
    z[NUMERIC] = (z[NUMERIC] - z[NUMERIC].mean()) / z[NUMERIC].std()
    ols = smf.ols("hospital_admissions ~ " + " + ".join(NUMERIC) + " + C(city, Treatment('Delhi'))", data=z).fit(cov_type="HC3")
    ci = ols.conf_int()
    coefs = pd.DataFrame({"term": ols.params.index, "coef": ols.params.values, "ci_low": ci[0].values, "ci_high": ci[1].values, "p_value": ols.pvalues.values})
    coefs["term"] = coefs["term"].str.replace("C(city, Treatment('Delhi'))[T.", "city: ", regex=False).str.replace("]", "", regex=False)
    save(coefs, "12_linear_coefficients")

    figure, axis = plt.subplots(figsize=(12, 6))
    names_out = list(tree.named_steps["preprocess"].get_feature_names_out())
    names_out = [n.split("__", 1)[1] for n in names_out]
    plot_tree(tree.named_steps["model"], feature_names=names_out, filled=False, impurity=False, proportion=False,
              rounded=True, fontsize=6, ax=axis, precision=2)
    axis.set_title("Regression tree (FS-Primary, tuned: depth 4, pruned) - every split is on PM2.5", fontsize=10)
    figure.tight_layout()
    figure.savefig(FIGURES / f"12_tree_structure{SUFFIX}.png", dpi=150)
    plt.close(figure)
    used = sorted({names_out[f] for f in tree.named_steps["model"].tree_.feature if f >= 0})
    (TABLES / f"12_tree_features_used{SUFFIX}.json").write_text(json.dumps({"features_used": used, "mode": RUN_MODE}), encoding="utf-8")


if __name__ == "__main__":
    if FAST:
        print("FAST MODE - NOT FOR REPORT")
    data = load_raw()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train = data.iloc[np.array(split["train_indices"], dtype=int)].reset_index(drop=True)
    threshold = float(json.loads((PROJECT_ROOT / "results" / "tables" / "03_spike_threshold.json").read_text())["threshold"])
    if FAST:
        strata = train["city"].astype(str) + "_" + (train["hospital_admissions"] >= threshold).astype(str)
        keep, _ = train_test_split(np.arange(len(train)), train_size=5_000, random_state=SEED, stratify=strata)
        train = train.iloc[np.sort(keep)].reset_index(drop=True)
    strata = train["city"].astype(str) + "_" + (train["hospital_admissions"] >= threshold).astype(str)
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(train, strata))
    # each part is skipped if its output already exists, so a rerun only adds what is missing
    for output, step, run in [("10_city_slopes", "step 10.2 city slopes", lambda: city_slopes(train)),
                              ("10_city_params", "step 10.1 per-city tuning", lambda: per_city_tuning(train, threshold)),
                              ("11_thresholds", "step 11 thresholds", lambda: spike_thresholds(train, threshold, folds)),
                              ("12_linear_coefficients", "step 12 PDP, coefficients, tree", lambda: interpretation(train))]:
        if (TABLES / f"{output}{SUFFIX}.csv").exists():
            print(f"{step}: exists, skipped")
            continue
        run()
        print(f"{step}: done")
    t = pd.read_csv(TABLES / f"11_thresholds{SUFFIX}.csv")
    print(t[["model", "feature_set", "threshold", "oof_f1", "oof_pr_auc"]].round(4).to_string(index=False))
    print(f"training rows only ({len(train):,}); test set not loaded")
