# LSTM improvement roadmap after shared preprocessing

This is the next phase after stabilizing `src/preprocessing.py` as the shared
source of truth.

## Current baseline

The executed LSTM notebook produced:

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Naive | 172.26 | 68.12 | 31.92% |
| Tuned LSTM | 97.97 | 47.10 | 24.08% |
| Seasonal naive | 71.75 | 39.89 | 24.71% |

The LSTM clearly beats naive, but seasonal naive still wins on RMSE and MAE.
The next LSTM phase should focus on beating seasonal naive without overfitting.

## What the improved preprocessing unlocks

The original LSTM saw only:

- `quantity_scaled`
- six calendar features
- product embedding

The shared preprocessing now also exposes:

- `positive_quantity_scaled`
- `returns_quantity_scaled`
- `price_base_scaled`
- `sum_total_scaled`
- `transaction_count_scaled`
- `store_count_scaled`
- `had_sales`

Those features give the LSTM more business signal without each notebook
inventing its own preprocessing.

The next LSTM run should use the shared team benchmark:

```text
top 100 products by total positive demand
with at least 730 active sales days
```

That benchmark gives roughly 46,000 LSTM train windows for a 60-day lookback and
30-day horizon, compared with roughly 23,000 windows in the original top-50 run.

## Recommended order

### 1. Add price and transaction features to the LSTM input

Use `feature_columns.json` or `preprocessing.DEFAULT_SHARED_FEATURE_COLS` when
calling `create_sequences()`.

Reason: price changes and transaction intensity are signals that seasonal naive
cannot see. This is the most direct way for the LSTM to learn something beyond
weekly repetition.

### 2. Treat product-universe expansion as a later shared benchmark experiment

After every model has run once on the top-100 / 730-active-day benchmark, try
controlled shared benchmark runs:

- top 150 with `min_active_days >= 700`
- top 250 with `min_active_days >= 600`
- top 500 with `min_active_days >= 500`

Reason: the top-50 run gives the LSTM only about 26k training windows. More
products create more sequence examples and more pattern diversity for the
embedding. Do not make these changes inside only the LSTM notebook; benchmark
changes must regenerate the shared artifacts for every ensemble-facing model.

### 3. Train against a seasonal-residual target

Instead of predicting demand directly:

```text
residual = actual_demand - seasonal_naive_prediction
final_prediction = seasonal_naive_prediction + lstm_residual_prediction
```

Reason: seasonal naive already captures the weekly pattern very well. The LSTM
should focus on the part seasonal naive cannot explain.

### 4. Regularize before making the model bigger

Try:

- `weight_decay=1e-4`
- dropout `0.3` or `0.4`
- embedding dimension `4`
- hidden size `32` or `64`
- keep early stopping

Reason: the first run overfit quickly. Bigger LSTMs are likely to memorize
training quirks unless the product universe and features improve first.

### 5. Try MAE or Huber loss

Compare MSE against:

- `torch.nn.L1Loss()`
- `torch.nn.SmoothL1Loss()`

Reason: retail demand has spikes. MSE chases spikes aggressively and can hurt
typical-day accuracy. Huber is a good middle ground.

### 6. Report the same baselines every time

Every new run should report:

- RMSE
- MAE
- MAPE
- naive baseline
- seasonal-naive baseline

Reason: the LSTM is only genuinely improved if it beats the strongest simple
baseline, not just the previous LSTM.
