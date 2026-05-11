# Ensemble Contract

This document is the non-negotiable rule for every model notebook and every
agentic coding tool working in this repository.

## Short version

Models may use different internal features and training code, but every model
that contributes to `notebooks/06_Ensemble.ipynb` must output predictions for
the same:

- `item_id`
- `date`
- target definition
- test window
- real-unit scale

Different internal preprocessing is allowed. Different final prediction rows
are not allowed.

## Why this matters

The ensemble combines predictions by joining on:

```text
date + item_id
```

For one prediction row, the ensemble expects this shape:

```text
date        item_id     actual_quantity  lstm_pred  rnn_pred  rf_pred  gp_pred
2024-04-28  product_A   94               82         80        88       90
```

Then the ensemble can safely compute:

```text
ensemble_pred = average(lstm_pred, rnn_pred, rf_pred, gp_pred)
```

If one model predicts a different product or a different date, there is no row
to average. The ensemble either drops rows, averages only partial models, or
produces misleading results.

## Shared artifacts

All ensemble-facing notebooks must use the canonical artifacts created by
`src/preprocessing.py`:

```text
data/processed/daily.csv
data/processed/splits.json
data/processed/selected_products.json
data/processed/scalers.pkl
data/processed/feature_scalers.pkl
data/processed/feature_columns.json
data/processed/outlier_caps.json
data/processed/metadata.json
```

The most important files are:

| File | Purpose |
|---|---|
| `selected_products.json` | The exact products every ensemble model must forecast |
| `splits.json` | The exact train / validation / test date boundaries |
| `scalers.pkl` | Per-product quantity scalers for inverse-scaling predictions |
| `feature_columns.json` | The shared columns available for model inputs |
| `daily.csv` | The canonical processed daily data |

## Current shared benchmark

For the first team-wide benchmark, the shared product/day rule is:

```text
Select the top 100 products by total positive demand,
among products with at least 730 active sales days.
```

Definitions:

- `active sales day`: a product-date where `positive_quantity > 0`
- `total positive demand`: sum of `positive_quantity`, not net quantity
- missing selected product-date rows are filled with zero observed demand
- the date range remains the full 761-day dataset range

In `src/preprocessing.py`, this is represented by:

```python
n_products = 100
min_active_days = 730
ranking_col = "positive_quantity"
```

This benchmark was chosen from the actual `sales.csv` distribution because it
doubles the LSTM/RNN training windows compared with top 50, keeps zero-filled
rows very low, and stays more manageable for Random Forest and Gaussian Process
than top 250 or top 500.

## Required final prediction schema

Every ensemble-facing model must save predictions with this schema:

```text
date,item_id,predicted_quantity,actual_quantity
```

Required properties:

- `date` must be in the shared test window from `splits.json`.
- `item_id` must come from `selected_products.json`.
- `predicted_quantity` must be in real demand units, not scaled units.
- `actual_quantity` must use the same target definition as every other model.
- The file should contain the same rows as every other ensemble model.

Recommended output paths:

```text
data/predictions/lstm_test_predictions.csv
data/predictions/rnn_test_predictions.csv
data/predictions/random_forest_test_predictions.csv
data/predictions/gaussian_test_predictions.csv
```

## What models may change

Each model may freely change its internal feature engineering.

Examples:

| Model | Allowed internal features |
|---|---|
| LSTM | sequences, embeddings, price features, transaction features |
| RNN | sequences, recurrent hidden state, calendar features |
| Random Forest | lag features, rolling means, day-of-week, price features |
| Gaussian Process | time index, seasonal inputs, product-level features |

These differences are fine because they happen before the final prediction file.

## What models must not change independently

Individual notebooks must not independently change:

- selected product list
- train / validation / test boundaries
- target definition
- final prediction scale
- final prediction row set
- outlier cap policy
- scaler fitting policy

If any of those need to change, update `src/preprocessing.py`, regenerate the
shared artifacts, rerun the affected models, and document the change.

## Good example

All models use:

```text
selected_products.json:
product_A
product_B
product_C

splits.json:
test_start = 2024-04-28
test_end   = 2024-09-26
```

LSTM can train on sequences. Random Forest can train on lag features. Gaussian
can train on a time index. But all final prediction files still contain:

```text
2024-04-28,product_A,...
2024-04-28,product_B,...
2024-04-28,product_C,...
2024-04-29,product_A,...
...
```

This is safe for the ensemble.

## Bad example

LSTM predicts:

```text
product_A
product_B
product_C
product_D
```

Random Forest predicts:

```text
product_A
product_B
product_E
```

Gaussian predicts:

```text
product_A
product_F
```

The only product all models share is `product_A`. The ensemble loses coverage
and cannot use most predictions.

## How to load the shared contract

Use this pattern in every notebook:

```python
from src import preprocessing

artifacts = preprocessing.load_artifacts("data/processed")
daily = artifacts["daily"]
splits = artifacts["splits"]
selected_products = artifacts["selected_products"]
scalers = artifacts["scalers"]
feature_columns = artifacts["feature_columns"]
```

Then add model-specific features after loading `daily`.

## Checklist before submitting predictions to the ensemble

Before a model's predictions are used in `06_Ensemble.ipynb`, verify:

- The model read `selected_products.json`.
- The model read `splits.json`.
- The prediction file has columns `date,item_id,predicted_quantity,actual_quantity`.
- Predictions are inverse-scaled to real units.
- Dates are within `splits["test_start"]` and `splits["test_end"]`.
- Product count matches `len(selected_products)`.
- `(date, item_id)` pairs are unique.
- The row set matches the other ensemble prediction files.

## Agentic coding tool rule

Claude Code, Codex, Gemini, Copilot, and any other coding agent must follow
this rule:

> Do not optimize a model by changing the ensemble-facing product list, split
> dates, target definition, or output schema inside one notebook. Model-specific
> experimentation is allowed only if the final prediction file still conforms
> to this contract.

If a requested change would violate this contract, stop and ask the user before
editing.
