# LSTM Demand Forecasting — Design Spec

> Historical v1 spec. The team-wide product universe has since been updated in
> [`docs/ensemble-contract.md`](../../ensemble-contract.md) to top 100 products
> by total positive demand with at least 730 active sales days. Use the ensemble
> contract as the current source of truth for product/date alignment.

**Date:** 2026-05-02
**Owner:** the user (LSTM only)
**Team scope (out of this spec):** RNN, Random Forest, Gaussian Process, ensemble integration

## 1. Goal

Build a PyTorch LSTM that predicts the next **30 days** of daily demand for each of the top-50 products, given the past 60 days of sales. The trained model's test-set predictions feed the team's ensemble in Sub-project D. The notebook also serves as the user's introduction to LSTMs.

## 2. Data

- **Source:** `sales.csv` at the project root (7,432,685 rows, 380 MB).
- **Columns:** `date, item_id, quantity, price_base, sum_total, store_id` (plus an unnamed index column to be dropped).
- **Date range:** 2022-08-28 → 2024-09-26 (761 days).
- **Unique products:** 28,182 across 4 stores.
- **Quirks:**
  - Negative `quantity` values exist (returns) — kept and netted into daily totals.
  - Outliers up to `quantity = 4,952` per row exist — capped at the 99.5th percentile per product.
  - Product activity is highly skewed: median product has only 72 days with sales; only 2,334 products have ≥600 days of history.

## 3. Forecasting problem

| Property | Value |
|---|---|
| Forecasting unit | One series per `item_id` (summed across the 4 stores) |
| Time granularity | Daily |
| Horizon | 30 days ahead (multi-step, direct multi-output) |
| Lookback | 60 days |
| Products forecasted | Top 50 by total quantity, restricted to those with sales on all 761 days |
| Use case | Per-product inventory restocking decisions |

## 4. Architectural decision

A **single global PyTorch LSTM with a learned product embedding** trains across all 50 products at once. Each training example is a `(60-day input window, 30-day target window)` pair for one product. The model receives the product's identity through an 8-dim embedding, concatenated with the per-day quantity and calendar features.

Rejected alternatives:
- *One LSTM per product:* 50× the training/maintenance cost, no cross-product transfer of patterns.
- *Seq2seq encoder-decoder:* more powerful but materially harder to debug — out of scope for a first LSTM.

## 5. ML lifecycle stages

The work is organized as **six sections inside one notebook (`notebooks/02_LSTM.ipynb`)**, mapping 1:1 to the standard ML cycle.

### 5.1 Stage 1 — Data preparation

Logic lives in `src/preprocessing.py` (framework-agnostic — returns `pandas.DataFrame` and `numpy.ndarray`, importable by the team's sklearn-based notebooks).

Steps:
1. Load `sales.csv`, drop the unnamed index column, parse `date` as datetime.
2. Group by `(item_id, date)`, sum `quantity` (returns net into the daily total).
3. Identify the **top 50 products** that satisfy: (a) sales recorded on all 761 days, and (b) highest total quantity.
4. Cap each product's daily quantity at its own 99.5th percentile.
5. Reindex to a complete date grid per product (no missing dates).

Outputs (after Stage 3 also runs):
- `data/processed/daily.csv` — long-format: `(date, item_id, quantity, day_of_week, day_of_month, month, quantity_scaled)`
- `data/processed/scalers.pkl` — `dict[item_id, MinMaxScaler]` for inverse scaling
- `data/processed/splits.json` — date boundaries for `train_start, train_end, val_start, val_end, test_start, test_end`

### 5.2 Stage 2 — EDA

LSTM-specific exploration of the cleaned daily data, in the notebook only. Plots:
- Per-product time series for 6 representative products (highest, median, lowest among the top 50)
- Aggregate weekly seasonality (mean quantity by day-of-week)
- Aggregate monthly seasonality (mean quantity by month)
- Distribution of quantities before vs after outlier capping
- Sparsity check: count of zero-demand days per product

The general/raw-data EDA stays in `notebooks/01_EDA.ipynb` (separate, shared with team).

### 5.3 Stage 3 — Feature engineering

In notebook + reused via `src/preprocessing.py`:

1. **Calendar features** — cyclical encoding (sin and cos pairs) for `day_of_week`, `day_of_month`, `month`. Six numeric features per day.
2. **Per-product MinMax scaling** of `quantity` to [0, 1]. Save scalers to `scalers.pkl`.
3. **Time-based 80/20 train/test split** (no random shuffling):
   - **Test (20%) = last 152 days** (held out, untouched until final eval).
   - **Train + val pool (80%) = first 609 days.**
     - Internal validation = last 60 days of the pool (for early stopping and tuning).
     - Internal training = first 549 days.
4. **Sliding-window sequence generation** — produce `(60-day input, 30-day target)` pairs per product, stride = 1.

### 5.4 Stage 4 — Model selection and training

Implementation in `src/models/lstm_model.py` (PyTorch). Imported and exercised in the notebook.

**Inputs per example:**

| Tensor | Shape | Meaning |
|---|---|---|
| `past_quantity` | `(60, 1)` | scaled daily quantity for the past 60 days |
| `calendar` | `(60, 6)` | sin/cos cyclical features for those 60 days |
| `product_idx` | `(1,)` | int 0–49 |

**Forward pass:**

```
product_idx ─> nn.Embedding(50, 8) ─> (8,) ─repeat 60 times─> (60, 8)
past_quantity (60,1) + calendar (60,6) + product_embed (60,8) ─> concat ─> (60, 15)
                                          │
                                          ▼
                          nn.LSTM(input_size=15, hidden_size=64, num_layers=2, dropout=0.2)
                                          │
                                          ▼
                              h_n[-1]  → (64,)
                                          │
                                          ▼
                              nn.Linear(64, 30) → ŷ_next30 (30,)
```

**Defaults (locked for v1, swept in Stage 6):**

| Knob | Default |
|---|---|
| `lookback_days` | 60 |
| `hidden_size` | 64 |
| `num_layers` | 2 |
| `dropout` | 0.2 |
| `embedding_dim` | 8 |
| `batch_size` | 64 |
| `learning_rate` | 1e-3 |
| Optimizer | Adam |
| Loss | MSE |
| Output activation | none (clip negatives at inference) |
| Max epochs | 100 |
| Early stopping patience | 10 (on val loss) |
| Random seed | 42 |

**Educational scaffolding** — markdown cells before code in this stage explain:
1. What problem LSTMs solve vs vanilla RNNs (forgetting → memory).
2. The three gates (forget / input / output) with one diagram.
3. Hidden state vs cell state — the LSTM's "scratchpad" and "long-term memory."
4. Why we use an embedding for `product_idx`.
5. Why a sliding window of 60 → 30 and how the dataset class generates it.
6. One worked example: pick `b0d24502fb66`, show one input window numerically, trace it through every layer.

### 5.5 Stage 5 — Evaluation

1. Generate predictions on the **152-day test set** using **5 non-overlapping 30-day evaluation windows** (covering 150 of the 152 test days; the trailing 2 days are unused). Each window's 60-day input comes from the actual data preceding it — not from the model's own earlier predictions.
   - Window 1: input = last 60 days of `train + val`; target = test days 0–29.
   - Windows 2–5: input = the 60 actual days preceding each target window; target = test days 30–59, 60–89, 90–119, 120–149.
2. **Inverse-scale** predictions to real units using `scalers.pkl`. Clip at 0.
3. **Metrics, computed both per-product and aggregated:**
   - RMSE (penalizes big misses)
   - MAE (interpretable: "off by N units/day on average")
   - MAPE (skipping days where actual = 0)
4. **Compare against two baselines** — required to know the LSTM is earning its keep:
   - **Naive:** ŷ_t = y_{t-1}
   - **Seasonal naive:** ŷ_t = y_{t-7} (same weekday last week)
5. **Diagnostic plots:**
   - Predicted vs actual time series for 4–6 representative products
   - Residuals distribution (bias check)
   - Per-product MAPE bar chart
   - Error vs forecast horizon (does day-30 error >> day-1 error?)
6. **Save predictions for the ensemble** to `data/predictions/lstm_test_predictions.csv` with columns `(date, item_id, predicted_quantity, actual_quantity)`.

### 5.6 Stage 6 — Fine-tuning

**Approach:** explicit grid, one knob at a time (not full Cartesian product), driven by val RMSE.

**Grid (one knob varied at a time, others held at default):**

| Knob | Values to sweep | New trainings (excluding baseline) |
|---|---|---|
| `hidden_size` | 32, 64, 128 | 2 (32, 128 — 64 is baseline) |
| `num_layers` | 1, 2 | 1 (1 — 2 is baseline) |
| `dropout` | 0.1, 0.3 | 2 (both new — 0.2 baseline not in grid) |
| `lookback_days` | 30, 60, 90 | 2 (30, 90 — 60 is baseline) |

Total: **1 baseline + 7 sweep trainings = 8 distinct trainings**, plus 1 final retrain on train+val combined → **9 trainings total**.

**Procedure:**
1. For each configuration, train on the train slice, evaluate on val. Log val RMSE.
2. Select the configuration with the lowest val RMSE.
3. **Retrain that configuration on train + val combined** for the same number of epochs that worked best.
4. Final test-set evaluation — this is the headline number reported.
5. Save:
   - `models/lstm_final.pt` (state dict)
   - `models/lstm_config.json` (architecture hyperparameters for reload)
   - `data/predictions/lstm_test_predictions.csv` (overwrites Stage 5 version with the tuned model's predictions)
6. Notebook closes with a short markdown write-up: which knob mattered most, what surprised you, what's next.

## 6. Deliverables

```
notebooks/
├── 01_EDA.ipynb            (general raw-data exploration — populate as part of Stage 1; shared with team)
└── 02_LSTM.ipynb           ★ all 6 stages as sections, top to bottom

src/
├── preprocessing.py        framework-agnostic; team's sklearn notebooks import this
└── models/
    └── lstm_model.py       PyTorch model class + train/predict utilities

data/
├── processed/
│   ├── daily.csv
│   ├── scalers.pkl
│   └── splits.json
└── predictions/
    └── lstm_test_predictions.csv

models/
├── lstm_final.pt
└── lstm_config.json

requirements.txt            drop tensorflow==2.13.0; keep torch==2.0.1
```

## 7. Replaced files

The following files in the existing scaffolding are **replaced**, not extended:

- `src/preprocessing.py` — current version uses synthetic `np.random.randn` data. Replace with a real implementation reading `sales.csv`.
- `src/models/lstm_model.py` — current version uses TensorFlow/Keras (`Sequential`, `LSTM`, `Dense`). Replace with PyTorch.
- `notebooks/01_EDA.ipynb` and `notebooks/02_LSTM.ipynb` — current versions are template scaffolding. Replace with real implementations.

The other model files (`rnn_model.py`, `random_forest_model.py`, `gaussian_model.py`) are out of scope for this spec — the team owns those.

## 8. Out of scope

- RNN model (team).
- Random Forest model (team).
- Gaussian Process model (team).
- Ensemble combination of all four models (team, Sub-project D).
- Real-time inference / API serving.
- Per-store forecasting (we sum across stores).
- Forecasting beyond the top-50 products.
- Inclusion of `price_base` as a feature (deferred to v2 — quantity + calendar + product embedding only for v1).

## 9. Reproducibility

- `random_seed = 42` set for `numpy`, `torch`, and `torch.cuda` if used.
- Exact split dates committed to `data/processed/splits.json`.
- Final model architecture hyperparameters committed to `models/lstm_config.json`.

## 10. Hardware assumptions

- Default training: CPU is sufficient for the v1 model (~50 products × ~500 train days × 60 lookback ≈ small dataset).
- Stage 6 grid search: 10 trainings will take ~2–6 hours on CPU; substantially less on GPU. No GPU is required.

## 11. Definition of done

1. `01_EDA.ipynb` runs end-to-end with real `sales.csv` and produces meaningful raw-data plots.
2. `02_LSTM.ipynb` runs end-to-end and reports test-set RMSE/MAE/MAPE that beats both naive baselines.
3. `data/predictions/lstm_test_predictions.csv` exists and matches `data/processed/splits.json` test boundaries — ready for the team's ensemble.
4. `models/lstm_final.pt` and `models/lstm_config.json` exist and can be reloaded successfully in a fresh kernel.
5. `requirements.txt` no longer contains `tensorflow`.
6. The notebook contains the LSTM educational explainers listed in 5.4.
