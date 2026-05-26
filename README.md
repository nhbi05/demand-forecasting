# Demand Forecasting — Unified Ensemble

A demand-forecasting system that combines four models — LSTM, GRU, Random Forest, and Gaussian Process — into a calibrated weighted ensemble. All models share a common preprocessing contract so their predictions can be combined safely on `(date, item_id)`.

## Authoritative documents

| Document | Purpose |
|---|---|
| [docs/ensemble-contract.md](docs/ensemble-contract.md) | **Source of truth** for the data interface every model must follow before submitting predictions to the ensemble |
| [docs/preprocessing-architecture.md](docs/preprocessing-architecture.md) | How `src/preprocessing.py` is organized |
| [docs/lstm-improvement-roadmap.md](docs/lstm-improvement-roadmap.md) | LSTM tuning history and decisions |
| [SETUP.md](SETUP.md) | Environment setup and data preparation |
| [QUICK_START.md](QUICK_START.md) | Fastest path to reproducing ensemble results (no training required) |

## What the system does

- Loads ~7.4M raw sales transactions from `sales.csv`
- Aggregates to daily product-level demand: 100 products, 761 days (2022-08-28 → 2024-09-26)
- Trains four models on a shared train / val / test split (80/20 with a 60-day validation tail)
- Combines individual predictions into a final forecast using inverse-RMSE-weighted averaging

## Project layout

```
demand-forecasting/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_LSTM.ipynb              # PyTorch LSTM
│   ├── 03_RNN.ipynb               # PyTorch GRU (despite the filename)
│   ├── 04_Random_Forest.ipynb
│   ├── 05_Gaussian_(1).ipynb
│   └── 06_Ensemble.ipynb
│
├── src/
│   ├── preprocessing.py           # Shared preprocessing contract
│   ├── ensemble.py                # EnsembleForecaster
│   └── models/
│       ├── lstm_model.py          # PyTorch LSTMForecaster
│       ├── rnn_model.py           # Legacy TensorFlow code — unused
│       ├── random_forest_model.py # sklearn RandomForestRegressor
│       └── gaussian_model.py      # sklearn per-product GaussianProcessRegressor
│
├── data/
│   ├── processed/                 # Artifacts from prepare_data() — gitignored
│   └── predictions/               # *_test_predictions.csv — tracked in git
│
├── models/                        # Saved model state (.pt, .json)
├── scripts/                       # build_lstm_notebook.py
├── docs/                          # Authoritative technical docs
└── requirements.txt
```

## Current ensemble results

Measured on the second half of the test window (2024-07-12 → 2024-09-24), 5,775 product-day rows.

| Model | RMSE | R² |
|---|---:|---:|
| LSTM (PyTorch) | 64.29 | 0.981 |
| GRU (PyTorch) | 65.61 | 0.981 |
| Random Forest | 86.20 | 0.966 |
| Gaussian Process | 172.54 | 0.866 |
| Simple-average ensemble | 85.43 | 0.967 |
| **Inverse-RMSE-weighted ensemble** | **74.90** | **0.975** |

Calibrated weights on the first half of the test window:

| Model | Weight |
|---|---:|
| LSTM | 0.299 |
| GRU | 0.293 |
| Random Forest | 0.257 |
| Gaussian Process | 0.150 |

The weighted ensemble does not beat the best individual model on RMSE — this is expected when combining models of unequal quality. It does reduce per-product variance compared with any single model.

See [docs/ensemble-fixes.md](docs/ensemble-fixes.md) for the rationale behind the inverse-RMSE switch (previously R²-based, which produced near-uniform weights).

## Tech stack

- Python 3.10 or 3.11
- PyTorch 2.0.1 (LSTM, GRU)
- scikit-learn 1.3.0 (Random Forest, Gaussian Process)
- pandas, numpy, matplotlib, seaborn, Jupyter

Exact versions in [requirements.txt](requirements.txt).

## Getting started

- New to the repo and just want to see results: [QUICK_START.md](QUICK_START.md)
- Setting up to train models locally: [SETUP.md](SETUP.md)
- Modifying or adding a model: read [docs/ensemble-contract.md](docs/ensemble-contract.md) first
