# Reproducibility

## Repository Layout

```
.
├── README.md
├── requirements.txt
├── docs/                          # this documentation
├── data/
│   ├── raw/                       # original CSVs, never modified
│   ├── processed/                 # split_indices.json, cleaned tables
│   └── README_DATA.md             # source URLs, licenses, download dates, SHA-256
├── notebooks/
│   ├── 01_audit.ipynb             # Steps 1-2 (and optional 2B)
│   ├── 02_baselines_models.ipynb  # Steps 3-8
│   ├── 03_final_evaluation.ipynb  # Step 9
│   ├── 04_city_analysis.ipynb     # Step 10
│   ├── 05_spike_task.ipynb        # Step 11
│   └── 06_interpretation.ipynb    # Steps 12-13
├── src/
│   ├── config.py                  # seed, paths, feature sets, grids
│   ├── data.py                    # loading and validation assertions
│   ├── features.py                # pipelines and encoders
│   ├── models.py                  # model factories (LR, DT, SVM, MLP, RF)
│   ├── evaluate.py                # metrics, bootstrap CIs, paired tests
│   └── plots.py
├── tests/
│   └── test_split_and_leakage.py
├── results/
│   ├── tables/
│   └── figures/
└── reports/
    ├── final_report.pdf
    └── slides.pdf
```

Do not commit large binaries. If a CSV is too large, provide download instructions instead.

## How to Reproduce

1. Clone the repository and create a fresh environment (Python 3.10+).
2. `pip install -r requirements.txt` (pinned versions of numpy, pandas, scikit-learn, statsmodels, scipy, matplotlib, seaborn, and PyTorch or TensorFlow).
3. Download the main dataset (link in the README) and, only if doing the optional Step 2B, the real reference dataset into `data/raw/`. Record source, license, download date and SHA-256 for each in `data/README_DATA.md`.
4. Run the notebooks in numerical order, or the equivalent scripts.
5. All randomness is controlled by `SEED = 42` (split, CV folds, model initialisation, bootstrap). MLP results are averaged over seeds 42–46.
