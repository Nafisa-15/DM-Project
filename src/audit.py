import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FAST, PROJECT_ROOT, RUN_MODE
from data import load_raw

OUT_DIR = PROJECT_ROOT / ("results_fast" if FAST else "results")
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"
SUFFIX = "_FAST" if FAST else ""
NUMERIC = ["aqi", "pm2_5", "pm10", "no2", "o3", "temperature", "humidity", "hospital_capacity", "hospital_admissions"]


def out(folder: Path, name: str) -> Path:
    stem, ext = name.rsplit(".", 1)
    return folder / f"{stem}{SUFFIX}.{ext}"


def save_target_distribution(data: pd.DataFrame) -> Path:
    path = out(FIGURES_DIR, "02_target_distribution.png")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(data["hospital_admissions"], bins=25, color="#1f6f8b", edgecolor="white")
    axis.set(title="Hospital admissions distribution", xlabel="Hospital admissions", ylabel="Days")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_correlation_heatmap(data: pd.DataFrame) -> Path:
    numeric = data.select_dtypes(include="number")
    correlation = numeric.corr()
    path = out(FIGURES_DIR, "02_correlation_heatmap.png")
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation.columns)), correlation.columns)
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_title("Numeric feature correlation")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_pm25_deciles(data: pd.DataFrame) -> Path:
    deciles = pd.qcut(data["pm2_5"], 10, labels=False, duplicates="drop") + 1
    grouped = data.assign(pm25_decile=deciles).groupby("pm25_decile", observed=True)["hospital_admissions"].agg(
        mean="mean", count="size", standard_error=lambda values: values.std(ddof=1) / np.sqrt(len(values))
    )
    path = out(FIGURES_DIR, "02_pm25_deciles.png")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.errorbar(
        grouped.index,
        grouped["mean"],
        yerr=1.96 * grouped["standard_error"],
        marker="o",
        color="#d95f02",
        capsize=3,
    )
    axis.set(title="Admissions by PM2.5 decile", xlabel="PM2.5 decile", ylabel="Mean hospital admissions")
    axis.set_xticks(grouped.index)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_city_means(data: pd.DataFrame) -> Path:
    result = (
        data.groupby("city", as_index=False)
        .agg(
            rows=("hospital_admissions", "size"),
            mean_hospital_admissions=("hospital_admissions", "mean"),
            sd_hospital_admissions=("hospital_admissions", "std"),
            mean_pm2_5=("pm2_5", "mean"),
            sd_pm2_5=("pm2_5", "std"),
            mean_aqi=("aqi", "mean"),
        )
        .assign(mode=RUN_MODE)
    )
    result["share_of_rows"] = result["rows"] / len(data)
    result["r_pm25_admissions"] = result["city"].map(
        data.groupby("city").apply(lambda g: g["pm2_5"].corr(g["hospital_admissions"]), include_groups=False)
    )
    path = out(TABLES_DIR, "02_city_means.csv")
    result.to_csv(path, index=False)
    return path


def save_lag_autocorrelation(data: pd.DataFrame) -> Path:
    ordered = data.assign(date=pd.to_datetime(data["date"])).sort_values(["city", "date"])
    rows = []
    for city, group in ordered.groupby("city", sort=True):
        for variable in ["pm2_5", "hospital_admissions"]:
            rows.append(
                {
                    "city": city,
                    "variable": variable,
                    "lag": 1,
                    "autocorrelation": group[variable].autocorr(lag=1),
                    "mode": RUN_MODE,
                }
            )
    path = out(TABLES_DIR, "02_lag_autocorrelation.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_numeric_distributions(data: pd.DataFrame) -> Path:
    path = out(FIGURES_DIR, "02_numeric_distributions.png")
    figure, axes = plt.subplots(3, 3, figsize=(11, 8.5))
    for axis, column in zip(axes.flat, NUMERIC):
        values = data[column]
        whole = values.dtype.kind == "i" or (values % 1 == 0).all()
        bins = np.arange(values.min(), values.max() + 2) - 0.5 if whole and values.nunique() <= 120 else 40
        axis.hist(values, bins=bins, color="#1f6f8b", edgecolor="white", linewidth=0.3)
        axis.set_title(column, fontsize=10)
    figure.suptitle("Distributions of numeric variables")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_numeric_summary(data: pd.DataFrame) -> Path:
    summary = data[NUMERIC].describe().T
    summary["skew"] = data[NUMERIC].skew()
    summary["zero_share"] = (data[NUMERIC] == 0).mean()
    path = out(TABLES_DIR, "02_numeric_summary.csv")
    summary.rename_axis("variable").reset_index().assign(mode=RUN_MODE).to_csv(path, index=False)
    return path


def save_target_correlations(data: pd.DataFrame) -> Path:
    rows = [
        {
            "variable": column,
            "pearson_r": data[column].corr(data["hospital_admissions"]),
            "spearman_rho": data[column].corr(data["hospital_admissions"], method="spearman"),
            "mode": RUN_MODE,
        }
        for column in NUMERIC[:-1]
    ]
    path = out(TABLES_DIR, "02_target_correlations.csv")
    pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False).to_csv(path, index=False)
    return path


def save_pm25_scatter(data: pd.DataFrame) -> Path:
    path = out(FIGURES_DIR, "02_pm25_vs_admissions.png")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    image = axis.hexbin(data["pm2_5"], data["hospital_admissions"], gridsize=45, cmap="Blues", mincnt=1)
    slope, intercept = np.polyfit(data["pm2_5"], data["hospital_admissions"], 1)
    grid = np.linspace(data["pm2_5"].min(), data["pm2_5"].max(), 50)
    axis.plot(grid, intercept + slope * grid, color="#d95f02", label=f"OLS slope {slope:.3f}")
    r = data["pm2_5"].corr(data["hospital_admissions"])
    axis.set(title=f"PM2.5 vs hospital admissions (r = {r:.2f})", xlabel="PM2.5", ylabel="Hospital admissions")
    axis.legend()
    figure.colorbar(image, ax=axis, label="Days")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_pm25_decile_table(data: pd.DataFrame) -> Path:
    deciles = pd.qcut(data["pm2_5"], 10, labels=False, duplicates="drop") + 1
    grouped = data.assign(pm25_decile=deciles).groupby("pm25_decile")
    table = grouped.agg(
        pm2_5_min=("pm2_5", "min"),
        pm2_5_max=("pm2_5", "max"),
        rows=("hospital_admissions", "size"),
        mean_admissions=("hospital_admissions", "mean"),
        sd_admissions=("hospital_admissions", "std"),
    )
    half_width = 1.96 * table["sd_admissions"] / np.sqrt(table["rows"])
    table["ci_low"] = table["mean_admissions"] - half_width
    table["ci_high"] = table["mean_admissions"] + half_width
    path = out(TABLES_DIR, "02_pm25_deciles.csv")
    table.reset_index().assign(mode=RUN_MODE).to_csv(path, index=False)
    return path


def save_date_gaps(data: pd.DataFrame) -> Path:
    dates = pd.to_datetime(data["date"])
    rows = []
    for city, group in data.assign(date=dates).groupby("city"):
        ordered = group["date"].sort_values()
        gaps = ordered.diff().dt.days.dropna()
        rows.append(
            {
                "city": city,
                "rows": len(group),
                "unique_dates": ordered.nunique(),
                "first_date": ordered.iloc[0].date(),
                "last_date": ordered.iloc[-1].date(),
                "span_years": (ordered.iloc[-1] - ordered.iloc[0]).days / 365.25,
                "median_gap_days": gaps.median(),
                "max_gap_days": gaps.max(),
                "share_gap_1_day": (gaps == 1).mean(),
            }
        )
    rows.append(
        {
            "city": "ALL",
            "rows": len(data),
            "unique_dates": dates.nunique(),
            "first_date": dates.min().date(),
            "last_date": dates.max().date(),
            "span_years": (dates.max() - dates.min()).days / 365.25,
            "median_gap_days": dates.sort_values().diff().dt.days.median(),
            "max_gap_days": dates.sort_values().diff().dt.days.max(),
            "share_gap_1_day": (dates.sort_values().diff().dt.days == 1).mean(),
        }
    )
    path = out(TABLES_DIR, "02_date_gaps.csv")
    pd.DataFrame(rows).assign(mode=RUN_MODE).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    if RUN_MODE == "FAST":
        print("FAST MODE - NOT FOR REPORT")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_raw()
    outputs = [
        save_target_distribution(frame),
        save_correlation_heatmap(frame),
        save_pm25_deciles(frame),
        save_city_means(frame),
        save_lag_autocorrelation(frame),
        save_numeric_distributions(frame),
        save_numeric_summary(frame),
        save_target_correlations(frame),
        save_pm25_scatter(frame),
        save_pm25_decile_table(frame),
        save_date_gaps(frame),
    ]
    print(f"audited {len(frame):,} rows")
    for output in outputs:
        print(output.relative_to(Path.cwd()))
