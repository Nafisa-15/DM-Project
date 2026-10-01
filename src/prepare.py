import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import FEATURE_SETS, PROCESSED_DIR, RUN_MODE, SEED, SPLIT_PATH, TABLES_DIR
from data import load_raw


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
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "stratified_by": "city x provisional_spike",
        "mode": RUN_MODE,
    }
    SPLIT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    threshold_path = TABLES_DIR / "03_spike_threshold.json"
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
    print(f"saved {SPLIT_PATH.relative_to(Path.cwd())}")
