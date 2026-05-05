# Why we keep one shared `preprocessing.py` for all four models

> A short note on why splitting `preprocessing.py` into one file per model would break our ensemble — and what to do instead when a model needs its own features.

For the stricter ensemble-facing rules, read
[`docs/ensemble-contract.md`](ensemble-contract.md). That file defines the
prediction schema, shared artifacts, and agent rules that every model must
respect before contributing to `notebooks/06_Ensemble.ipynb`.

## The proposal

Replace the single `src/preprocessing.py` with four model-specific files:

```
src/preprocessing_lstm.py
src/preprocessing_rnn.py
src/preprocessing_random_forest.py
src/preprocessing_gaussian.py
```

The reasoning behind it is fair: different models need different feature engineering, and one shared file feels like it's trying to be everything to everyone.

## Why it doesn't work for our project

The four models are not independent. They feed into a single ensemble in
`notebooks/06_Ensemble.ipynb`, which averages four sets of predictions. For that
average to be meaningful, **every model must forecast the same test days, for
the same products, in the same scale**. Right now the files under
`data/processed/` are the single source of truth that make this true — every
notebook reads them.

Four separate preprocessing files would inevitably diverge in subtle ways. Each
one would independently decide:

### 1. Which products to forecast

`select_top_products(n=100, min_active_days=730)` runs at one moment in time.
If a teammate later experiments with `min_active_days=600` in their own file,
they get a different set of products. Now the LSTM is forecasting product A and
the Random Forest is forecasting product B. The ensemble has nothing to average.

### 2. Where train and test split

If one file does 80/20 and another does 75/25, the LSTM's test set spans days
`[610..761]` while RF's spans `[571..761]`. They predict different days. The
ensemble can't combine them.

### 3. How to scale

If each file fits its own `MinMaxScaler` on its own slice of the data, the
inverse-scaling in each notebook produces predictions in slightly different
real-unit scales. Errors compound when averaging.

### 4. Which day is "today"

If the LSTM uses cutoff `2024-04-27` and the RF uses `2024-04-28`, the four
notebooks are forecasting non-comparable horizons.

The four predictions arrive at `06_Ensemble.ipynb` and either crash with a
shape mismatch or — worse — silently produce garbage that *looks* fine.

## The real distinction: shared steps vs model-specific steps

The instinct that "different models need different things" is correct. But
"different things" is overwhelmingly about how to **shape the already-prepared
data** into model inputs, not about **what the prepared data is**.

| Step | Shared by all 4 models? |
|---|---|
| Load `sales.csv` | yes |
| Aggregate transactions to daily per product | yes |
| Pick top-N products | yes — must be same products |
| Missing-value handling | yes — must be same raw-data policy |
| Fill missing product-date rows | yes — missing rows mean zero observed sales |
| Outlier capping | yes — train-only caps, must be same data |
| Calendar features (sin/cos for dow/dom/month) | yes — cheap, all models can use |
| Train / val / test split dates | **must be identical or ensemble breaks** |
| Per-product MinMax scaling | yes — train-only fits, same scalers for inverse comparison |
| Sliding-window sequence generation | no — only LSTM and RNN need this |
| 7-day, 14-day lag features | no — RF and GP benefit; LSTM doesn't |
| Rolling 7-day mean / std | no — RF / GP-specific |
| Holiday flag, weather, etc. | optional — share if all four want it |

The first shared rows account for most of the preprocessing work, and they have
to be identical across all four models. Splitting them across four files is
busywork that introduces alignment bugs without any real upside.

The last four rows are model-specific. Those can — and should — live in each
model's notebook.

## The architecture we're using (already set up)

```
src/preprocessing.py            shared canonical pipeline
                                writes data/processed/{daily.csv,
                                                       scalers.pkl,
                                                       feature_scalers.pkl,
                                                       splits.json,
                                                       selected_products.json,
                                                       outlier_caps.json,
                                                       feature_columns.json,
                                                       metadata.json}

notebooks/02_LSTM.ipynb          imports from src.preprocessing,
                                 does sliding-window in the notebook
notebooks/03_RNN.ipynb           same idea
notebooks/04_Random_Forest.ipynb imports from src.preprocessing,
                                 adds lag / rolling features in the notebook
notebooks/05_Gaussian.ipynb      same idea
notebooks/06_Ensemble.ipynb      reads predictions from all four, averages
```

The shared `preprocessing.py` already has unit tests covering every
function. Splitting it would multiply the tests by four for no benefit, and
nothing in the shared pipeline is a barrier to model-specific feature
engineering.

## What the shared preprocessing file now owns

`src/preprocessing.py` is the canonical source of truth for:

- Raw schema validation.
- Missing-value handling.
- Daily aggregation from raw transactions.
- Product selection.
- Missing product-date row filling.
- Train / validation / test split dates.
- Train-only outlier caps.
- Train-only per-product scalers.
- Shared calendar and continuous features.
- Artifact loading helpers for all notebooks.
- Optional lag / rolling helpers for flat models.

The current shared benchmark is:

```text
top 100 products by total positive demand
with at least 730 active sales days
```

This is intentionally not a per-model choice. If the team later changes this
benchmark, regenerate the shared artifacts and rerun all ensemble-facing model
notebooks against the new `selected_products.json` and `splits.json`.

The important leakage rule is:

> Anything that learns a statistic from the data must fit on the training
> period only, then apply to validation and test.

That is why outlier caps and scalers are saved as artifacts.

## How to do model-specific feature engineering correctly

Inside each model's notebook, after importing the shared output, add whatever
features the model needs. This pattern is exactly what we want.

### Random Forest example (in `04_Random_Forest.ipynb`)

```python
import pandas as pd
from src import preprocessing

# Use the shared canonical data and artifacts — same as every model does
artifacts = preprocessing.load_artifacts("data/processed")
daily = artifacts["daily"]
splits = artifacts["splits"]

# RF-specific feature engineering, only in this notebook:
daily = preprocessing.create_lag_features(
    daily,
    value_col="quantity_scaled",
    lags=(1, 7, 14, 30),
    rolling_windows=(7, 14, 30),
)

slices = preprocessing.slice_by_split(daily, splits)

# ... train RF on these flat features ...
```

### Gaussian Process example (in `05_Gaussian.ipynb`)

```python
import pandas as pd
from src import preprocessing

artifacts = preprocessing.load_artifacts("data/processed")
daily = artifacts["daily"]
splits = artifacts["splits"]

# GP-specific kernel inputs would go here
# (e.g. the time index as a continuous feature for an RBF kernel)
daily["t"] = (daily["date"] - daily["date"].min()).dt.days
slices = preprocessing.slice_by_split(daily, splits)
```

In both cases:

- The shared `daily.csv`, `scalers.pkl`, `feature_scalers.pkl`, and JSON
  artifacts are unchanged.
- The model-specific transformations are visible and editable in one place.
- The ensemble in `06_Ensemble.ipynb` still gets aligned predictions because
  all four models read the same `splits.json` and selected product list.

## Summary

- One shared `preprocessing.py` is what makes the ensemble safe to average.
- "Different models need different features" is a real concern — but the right
  fix is to add those features in each model's own notebook, not to fork the
  preprocessing module.
- Anything that has to be identical across models stays in `preprocessing.py`.
  Anything specific to one model stays in that model's notebook.
- The contract everyone respects is the full `data/processed/` artifact set,
  especially `splits.json`, `selected_products.json`, `scalers.pkl`, and
  `feature_columns.json`.
