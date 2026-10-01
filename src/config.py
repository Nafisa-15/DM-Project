import os
from pathlib import Path

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DATA_DIR / "air_quality_health_dataset.csv"
SPLIT_PATH = PROCESSED_DIR / "split_indices.json"
FEATURE_SETS = {
	"FS-Primary": ["aqi", "pm2_5", "pm10", "no2", "o3", "temperature", "humidity", "city"],
	"FS-PM25": ["pm2_5"],
}
RUN_MODE = os.getenv("RUN_MODE", "FAST").upper()
FAST = RUN_MODE != "FULL"
