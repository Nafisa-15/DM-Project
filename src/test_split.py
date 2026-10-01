import json

import numpy as np

from config import RAW_DATA_PATH, SPLIT_PATH


def test_split_is_disjoint_and_complete() -> None:
    import pandas as pd

    row_count = len(pd.read_csv(RAW_DATA_PATH, usecols=["city"]))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_idx = np.array(split["train_indices"])
    test_idx = np.array(split["test_indices"])
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.union1d(train_idx, test_idx)) == row_count
    assert len(train_idx) == round(row_count * 0.80)
    assert len(test_idx) == row_count - len(train_idx)


if __name__ == "__main__":
    test_split_is_disjoint_and_complete()
    print("split check passed: disjoint, complete, and 80/20")
