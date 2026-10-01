import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import FAST, FEATURE_SETS, PROJECT_ROOT, RUN_MODE, SEED, SPLIT_PATH, TABLES_DIR
from data import load_raw

# FAST never touches the real split or threshold; it writes _FAST copies to results_fast/
SUFFIX = "_FAST" if FAST else ""
OUT_SPLIT = PROJECT_ROOT / "results_fast" / "split_indices_FAST.json" if FAST else SPLIT_PATH
OUT_TABLES = PROJECT_ROOT / "results_fast" / "tables" if FAST else TABLES_DIR


def make_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    provisional_threshold = data["hospital_admissions"].quantile(0.90)
    provisional_spike = (data["hospital_admissions"] >= provisional_threshold).astype(int)
    strata = data["city"].astype(str) + "_" + provisional_spike.astype(str)
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=strata,
    )
    threshold = float(data.iloc[train_idx]["hospital_admissions"].quantile(0.90))
    return np.sort(train_idx), np.sort(test_idx), threshold


def save_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    train_idx, test_idx, threshold = make_split(data)
    OUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "stratified_by": "city x provisional_spike",
        "mode": RUN_MODE,
    }
    OUT_SPLIT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    threshold_path = OUT_TABLES / f"03_spike_threshold{SUFFIX}.json"
    threshold_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "quantile": 0.90,
                "source": "training_rows_only",
                "train_rows": len(train_idx),
                "mode": RUN_MODE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return train_idx, test_idx, threshold


if __name__ == "__main__":
    if RUN_MODE == "FAST":
        print("FAST MODE - NOT FOR REPORT")
    frame = load_raw()
    train_idx, test_idx, threshold = save_split(frame)
    print(f"threshold {threshold:g} from {len(train_idx):,} training rows")
    print(f"split {len(train_idx):,} train / {len(test_idx):,} test")
    print(f"feature sets: {', '.join(FEATURE_SETS)}")
    print(f"saved {OUT_SPLIT.relative_to(PROJECT_ROOT)}")
