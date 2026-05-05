# Why we keep one shared `preprocessing.py` for all four models

> A short note on why splitting `preprocessing.py` into one file per model would break our ensemble — and what to do instead when a model needs its own features.

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
the same products, in the same scale**. Right now `data/processed/splits.json`
is the single source of truth that makes this true — every notebook reads it.

Four separate preprocessing files would inevitably diverge in subtle ways. Each
one would independently decide:

### 1. Which products to forecast

`select_top_products(n=50, full_history_days=761)` runs at one moment in time.
If a teammate later experiments with `full_history_days=730` in their own file,
they get a different set of 50 products. Now the LSTM is forecasting product A
and the Random Forest is forecasting product B. The ensemble has nothing to
average.

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
| Outlier capping | yes — must be same data |
| Calendar features (sin/cos for dow/dom/month) | yes — cheap, all models can use |
| Train / val / test split dates | **must be identical or ensemble breaks** |
| Per-product MinMax scaling | yes — same scalers for inverse comparison |
| Sliding-window sequence generation | no — only LSTM and RNN need this |
| 7-day, 14-day lag features | no — RF and GP benefit; LSTM doesn't |
| Rolling 7-day mean / std | no — RF / GP-specific |
| Holiday flag, weather, etc. | optional — share if all four want it |

The first seven rows account for most of the preprocessing work, and they have
to be identical across all four models. Splitting them across four files is
busywork that introduces alignment bugs without any real upside.

The last four rows are model-specific. Those can — and should — live in each
model's notebook.

## The architecture we're using (already set up)

```
src/preprocessing.py            shared canonical pipeline (the 7 things above)
                                writes data/processed/{daily.csv,
                                                       scalers.pkl,
                                                       splits.json}

notebooks/02_LSTM.ipynb          imports from src.preprocessing,
                                 does sliding-window in the notebook
notebooks/03_RNN.ipynb           same idea
notebooks/04_Random_Forest.ipynb imports from src.preprocessing,
                                 adds lag / rolling features in the notebook
notebooks/05_Gaussian.ipynb      same idea
notebooks/06_Ensemble.ipynb      reads predictions from all four, averages
```

The shared `preprocessing.py` already has 17 unit tests covering every
function. Splitting it would multiply the tests by four for no benefit, and
nothing in the shared pipeline is a barrier to model-specific feature
engineering.

## How to do model-specific feature engineering correctly

Inside each model's notebook, after importing the shared output, add whatever
features the model needs. This pattern is exactly what we want.

### Random Forest example (in `04_Random_Forest.ipynb`)

```python
from src import preprocessing
import pandas as pd

# Use the shared canonical data — same as the LSTM and RNN do
daily = pd.read_csv("data/processed/daily.csv", parse_dates=["date"])

# RF-specific feature engineering, only in this notebook:
daily["sales_lag_7"]    = daily.groupby("item_id")["quantity_scaled"].shift(7)
daily["sales_lag_14"]   = daily.groupby("item_id")["quantity_scaled"].shift(14)
daily["rolling_mean_7"] = (
    daily.groupby("item_id")["quantity_scaled"]
         .transform(lambda s: s.rolling(7).mean())
)

# ... train RF on these flat features ...
```

### Gaussian Process example (in `05_Gaussian.ipynb`)

```python
from src import preprocessing
import pandas as pd

daily = pd.read_csv("data/processed/daily.csv", parse_dates=["date"])

# GP-specific kernel inputs would go here
# (e.g. the time index as a continuous feature for an RBF kernel)
daily["t"] = (daily["date"] - daily["date"].min()).dt.days
```

In both cases:

- The shared `daily.csv`, `scalers.pkl`, and `splits.json` are unchanged.
- The model-specific transformations are visible and editable in one place.
- The ensemble in `06_Ensemble.ipynb` still gets aligned predictions because
  all four models read the same `splits.json` for their train / test
  boundaries.

## Summary

- One shared `preprocessing.py` is what makes the ensemble safe to average.
- "Different models need different features" is a real concern — but the right
  fix is to add those features in each model's own notebook, not to fork the
  preprocessing module.
- Anything that has to be identical across models stays in `preprocessing.py`.
  Anything specific to one model stays in that model's notebook.
- The contract everyone respects is `data/processed/splits.json`.
