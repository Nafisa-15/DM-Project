import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype, is_string_dtype

from config import FAST, PROJECT_ROOT, RAW_DATA_PATH, RUN_MODE, TABLES_DIR

EXPECTED_COLUMNS = [
    "city",
    "date",
    "aqi",
    "pm2_5",
    "pm10",
    "no2",
    "o3",
    "temperature",
    "humidity",
    "hospital_admissions",
    "population_density",
    "hospital_capacity",
]
POLLUTANT_COLUMNS = ["pm2_5", "pm10", "no2", "o3"]


def load_raw() -> pd.DataFrame:
    data = pd.read_csv(RAW_DATA_PATH)

    assert list(data.columns) == EXPECTED_COLUMNS
    assert data.shape == (88_489, 12)
    assert is_string_dtype(data["city"])
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(data["date"]))
    for column in ["aqi", *POLLUTANT_COLUMNS, "temperature", "humidity", "hospital_admissions", "hospital_capacity"]:
        assert is_numeric_dtype(data[column])
    assert is_string_dtype(data["population_density"])
    assert data.notna().all().all()
    assert not data.duplicated().any()
    assert data["city"].nunique() == 8
    assert data["population_density"].nunique() == 3
    assert (data[POLLUTANT_COLUMNS] >= 0).all().all()
    assert data["aqi"].between(0, 500).all()
    assert data["humidity"].between(0, 100).all()
    assert (data["hospital_admissions"] >= 0).all()
    assert is_integer_dtype(data["hospital_admissions"])
    assert (data["hospital_admissions"] <= data["hospital_capacity"]).all()

    return data


def validation_report(data: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("columns", len(data.columns), 12),
        ("rows", len(data), 88_489),
        ("missing_values", int(data.isna().sum().sum()), 0),
        ("duplicate_rows", int(data.duplicated().sum()), 0),
        ("cities", data["city"].nunique(), 8),
        ("population_density_values", data["population_density"].nunique(), 3),
        ("pollutants_nonnegative", True, True),
        ("aqi_in_range", True, True),
        ("humidity_in_range", True, True),
        ("admissions_nonnegative_integer", True, True),
        ("admissions_within_capacity", True, True),
    ]
    return pd.DataFrame(
        [
            {"check": name, "value": value, "expected": expected, "passed": value == expected, "mode": RUN_MODE}
            for name, value, expected in checks
        ]
    )


if __name__ == "__main__":
    if RUN_MODE == "FAST":
        print("FAST MODE - NOT FOR REPORT")
    frame = load_raw()
    out_tables = PROJECT_ROOT / "results_fast" / "tables" if FAST else TABLES_DIR
    report_path = out_tables / f"01_validation{'_FAST' if FAST else ''}.csv"
    out_tables.mkdir(parents=True, exist_ok=True)
    validation_report(frame).to_csv(report_path, index=False)
    print(f"validated {len(frame):,} rows, {len(frame.columns)} columns")
    print(f"saved {report_path.relative_to(PROJECT_ROOT)}")
