# Quick Start

The fastest path to seeing ensemble results. Skips training entirely — uses the per-model prediction CSVs already tracked in this repo.

## What you need

- Python 3.10 or 3.11
- `pip install -r requirements.txt`

The raw `sales.csv` and trained model weights are **not** required for this path. Every model has already saved its test predictions to `data/predictions/`, and the ensemble notebook only reads those CSVs.

## Steps

```bash
pip install -r requirements.txt
jupyter notebook notebooks/06_Ensemble.ipynb
```

Then "Run All Cells".

You should see:

- Individual model metrics (LSTM, GRU, Random Forest, Gaussian Process)
- Simple-average ensemble results
- Inverse-RMSE-weighted ensemble results with calibrated weights
- Comparison plots

Expected weighted-ensemble result: RMSE ≈ 74.9, R² ≈ 0.975 on the evaluation half of the test window.

## What the notebook does

1. Loads the four `*_test_predictions.csv` files from [data/predictions/](data/predictions/)
2. Merges them on `(date, item_id)` — the schema defined in [docs/ensemble-contract.md](docs/ensemble-contract.md)
3. Splits the test window 50/50: first half calibrates ensemble weights, second half is held out for evaluation
4. Reports both simple-average and inverse-RMSE-weighted ensemble metrics

## Want to re-train a model?

Re-training requires the raw `sales.csv` and a Python environment with the model's dependencies. See [SETUP.md](SETUP.md).

## Want to add a model?

Read [docs/ensemble-contract.md](docs/ensemble-contract.md) first. Your model's prediction CSV must follow the shared schema (`date, item_id, predicted_quantity, actual_quantity`) and cover the same `(date, item_id)` set as the other models, or the ensemble will drop rows on the merge.
