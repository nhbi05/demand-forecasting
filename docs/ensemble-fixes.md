# Ensemble Bug Report & Fixes

## What Was Wrong

### 1. Wrong Product Averaging (Critical)

The notebook extracted predictions as raw arrays using `.values` and averaged them positionally — row 0 with row 0, row 1 with row 1, etc.

The problem is that RF and Gaussian sorted their prediction files in a different product order than LSTM and GRU. So the ensemble was averaging LSTM's forecast for `product_A` with RF's forecast for a completely different `product_B`.

**Evidence:** A proper merge on `(date, item_id)` only found 11,550 matching rows — not the 15,000 the notebook claimed to be working with.

**Effect:** RF appeared to have R² = -0.85 (terrible). After the fix it scores R² = 0.97. Its predictions were never bad — they were just being matched to the wrong products.

---

### 2. Gaussian Had 200 Extra Rows

Gaussian's prediction file had 15,200 rows ending on `2024-09-26`, while the other three models had 15,000 rows ending on `2024-09-24`.

The notebook handled this with a `min_length` truncation — silently chopping off the 200 extra rows rather than aligning by date.

**Effect:** Those 200 Gaussian rows were quietly dropped with no warning. The contract violation went undetected.

---

### 3. Data Leakage in Weight Calibration

The weighted ensemble called `adapt_weights(y_true=y_test)` — fitting the model weights using test set labels — then immediately evaluated on that same test set.

```python
# old code
weighted_pred = weighted_ensemble.predict(predictions_dict, y_true=y_test)
weighted_results = weighted_ensemble.evaluate(y_test, weighted_pred)
```

**Effect:** The reported weighted ensemble R² of 0.9797 was optimistic. Weights were tuned to the exact data being measured.

---

### 4. MAPE Was Broken

All MAPE values were around 10¹⁶ — effectively meaningless. The cause was 31 zero-quantity rows in `actual_quantity`. `mean_absolute_percentage_error` divides by the actual value, so zeros cause division-by-zero and the metric explodes.

This affected every model's MAPE and the ensemble's, making the metric useless for comparison.

---

## What Was Fixed

### Fix 1 — Proper `(date, item_id)` Merge

**File:** `notebooks/06_Ensemble.ipynb`

Replaced positional `.values` extraction with an explicit merge on `(date, item_id)`, as the ensemble contract requires. Each model's predictions are now guaranteed to be for the same product on the same date before averaging.

```python
# new code
merged = (
    lstm_df
    .rename(columns={'predicted_quantity': 'lstm_pred', 'actual_quantity': 'actual'})
    .merge(gru_df.rename(columns={'predicted_quantity': 'gru_pred'})[['date','item_id','gru_pred']], on=['date','item_id'])
    .merge(rf_df.rename(columns={'predicted_quantity': 'rf_pred'})[['date','item_id','rf_pred']], on=['date','item_id'])
    .merge(gaussian_df.rename(columns={'predicted_quantity': 'gaussian_pred'})[['date','item_id','gaussian_pred']], on=['date','item_id'])
    .sort_values(['date','item_id'])
    .reset_index(drop=True)
)
```

The Gaussian row mismatch is handled automatically — only rows present in all four files are kept.

---

### Fix 2 — Calibration / Evaluation Split

**File:** `notebooks/06_Ensemble.ipynb`

The test window is now split in half. The first half calibrates the ensemble weights; the second half is held out for evaluation. Weights are never fit on data that is used to report accuracy.

```python
dates    = np.sort(merged['date'].unique())
mid_date = dates[len(dates) // 2]

cal_mask  = merged['date'].values < mid_date   # fit weights here
eval_mask = ~cal_mask                          # report metrics here

weighted_ensemble.adapt_weights(y_cal, cal_preds)
weighted_pred = weighted_ensemble.weighted_average(eval_preds)
```

Individual model metrics and the simple ensemble are also evaluated on the held-out eval window for a consistent comparison.

---

### Fix 3 — MAPE Zero Masking

**Files:** `src/ensemble.py`, `notebooks/06_Ensemble.ipynb`

Zero-quantity rows are now masked before computing MAPE in both the `EnsembleForecaster.evaluate()` method and the individual model metrics cell.

```python
nonzero = y_true != 0
mape = mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero]) * 100
```

---

## Results Before and After

| Model | R² Before | R² After |
|---|---|---|
| LSTM | 0.979 | 0.981 |
| GRU | 0.978 | 0.981 |
| Random Forest | -0.855 | **0.966** |
| Gaussian | -0.447 | 0.866 |
| Simple Ensemble | 0.628 | 0.967 |
| Weighted Ensemble | 0.980\* | 0.968 |

\* Previous weighted ensemble R² was inflated by data leakage and wrong product matching.

The simple ensemble improvement (0.628 → 0.967) is entirely explained by fix 1 — it was averaging predictions for the wrong products before.

---

## Remaining Issue to Be Aware Of

The merge only finds **77 of the expected 100 products** with predictions from all four models. The other 23 are silently dropped by the merge. Check which model is missing which products — each individual model should cover all 100 products from `selected_products.json`.
