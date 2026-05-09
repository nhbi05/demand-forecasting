"""One-time builder for notebooks/02_LSTM.ipynb (v2).

This script regenerates 02_LSTM.ipynb against the shared preprocessing
contract. Run: python scripts/build_lstm_notebook.py

v2 differences from v1:
  - Loads canonical artifacts via preprocessing.load_artifacts()
  - Uses the full DEFAULT_SHARED_FEATURE_COLS (14 features incl. price,
    transaction_count, returns, etc.) — not just calendar + quantity
  - Trains against seasonal-naive residuals: predict (actual - seasonal_naive),
    add the baseline back at inference. Forces the model to learn what
    seasonal-naive cannot
  - Regularization: dropout 0.3, embed_dim 4, weight_decay 1e-4, hidden 64
  - Always reports naive + seasonal-naive baselines
"""

import nbformat as nbf

CELLS = []


def md(s):
    CELLS.append(("md", s))


def code(s):
    CELLS.append(("code", s))


# ============== Title ==============
md("""# 02 - LSTM Demand Forecasting (v2)

End-to-end PyTorch LSTM on the shared **top-100 / 730-active-day** universe.

This is v2 - improvements over v1:

1. **Richer features** - quantity, returns, price, sum_total, transaction_count, store_count, had_sales, plus calendar (14 total)
2. **Residual training** - model predicts `actual - seasonal_naive`, not `actual`
3. **Regularization** - dropout 0.3, embed 4, weight_decay 1e-4
4. **Always reports baselines** - naive + seasonal-naive comparison

Ensemble contract: predictions saved to `data/predictions/lstm_test_predictions.csv` with the canonical schema `date,item_id,predicted_quantity,actual_quantity`. Same product list and test window as every other ensemble model.""")

# ============== SECTION 1 ==============
md("""## 1. Data preparation

Loads the shared artifacts written by `src.preprocessing.prepare_data()`.
Regenerates them if they don't exist yet.""")

code("""import sys, os
# Make project root importable when this notebook is run from notebooks/
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src import preprocessing
from src.models import lstm_model

torch.manual_seed(42)
np.random.seed(42)""")

code("""# Regenerate the shared artifacts only if they're missing.
PROCESSED_DIR = "data/processed"
required = ["daily.csv", "splits.json", "selected_products.json",
            "scalers.pkl", "feature_columns.json"]
needs_regen = any(not (Path(PROCESSED_DIR) / f).exists() for f in required)

if needs_regen:
    print("Regenerating shared preprocessing artifacts...")
    preprocessing.prepare_data(csv_path="sales.csv", outdir=PROCESSED_DIR)
else:
    print("Shared artifacts already present - using them.")""")

code("""# Load the canonical shared artifacts
artifacts = preprocessing.load_artifacts(PROCESSED_DIR)
daily = artifacts["daily"]
splits = artifacts["splits"]
selected_products = artifacts["selected_products"]
scalers = artifacts["scalers"]
feature_columns = artifacts["feature_columns"]
metadata = artifacts.get("metadata", {})

print(f"Products: {len(selected_products)}")
print(f"Daily rows: {len(daily):,}")
print(f"Benchmark: {metadata.get('benchmark_rule', 'n/a')}")
print(f"Splits: {splits}")
print()
print(f"Shared feature columns ({len(feature_columns['shared_default'])}):")
for col in feature_columns['shared_default']:
    print(f"  - {col}")""")

# ============== SECTION 2 ==============
md("""## 2. EDA (LSTM-specific)

Quick checks on the v2 dataset - top 100 products instead of 50, and the
new feature columns we're about to feed into the model.""")

code("""totals = daily.groupby("item_id")["quantity"].sum().sort_values(ascending=False)
representatives = [
    totals.index[0],          # highest volume
    totals.index[len(totals) // 2],  # median volume
    totals.index[-1],         # lowest of top 100
]
print("Representative products (top, median, bottom of top-100):")
print(representatives)""")

code("""fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, item_id in zip(axes, representatives):
    sub = daily[daily["item_id"] == item_id].sort_values("date")
    ax.plot(sub["date"], sub["quantity"])
    ax.set_title(f"item {item_id}")
    ax.set_ylabel("quantity")
plt.tight_layout(); plt.show()""")

code("""train_end = pd.Timestamp(splits["train_end"])
train_rows = daily[daily["date"] <= train_end]
scaled_cols = feature_columns["scaled"]
print("Scaled feature ranges in training period:")
print(train_rows[scaled_cols].describe().loc[["min", "max", "mean"]].T.round(3))""")

code("""prod = representatives[0]
sub = daily[daily["item_id"] == prod].sort_values("date").set_index("date")
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(sub.index, sub["quantity"], label="demand", color="#2E86AB")
axes[0].set_title(f"{prod} - demand vs. price")
axes[0].set_ylabel("quantity")
axes[1].plot(sub.index, sub["price_base"], label="price", color="#A23B72")
axes[1].set_ylabel("price_base"); axes[1].set_xlabel("date")
plt.tight_layout(); plt.show()""")

# ============== SECTION 3 ==============
md("""## 3. Feature engineering

Three things happen here:

1. Build sequences with the **richer 14-feature input** instead of just calendar + quantity.
2. Compute the **seasonal-naive baseline for every horizon position** so we can derive residual targets.
3. Wrap everything in DataLoaders.""")

code("""LOOKBACK = 60
HORIZON = 30
PERIOD = 7   # weekly seasonality
BATCH_SIZE = 64

slices = preprocessing.slice_by_split(daily, splits, lookback_days=LOOKBACK)
train_slice, val_slice, test_slice = slices["train"], slices["val"], slices["test"]

product_to_idx = {p: i for i, p in enumerate(sorted(selected_products))}
N_PRODUCTS = len(product_to_idx)
SHARED_FEATURES = feature_columns["shared_default"]
N_FEATURES = len(SHARED_FEATURES)

print(f"train slice: {len(train_slice):,} rows ({train_slice['date'].min()} -> {train_slice['date'].max()})")
print(f"val slice:   {len(val_slice):,} rows ({val_slice['date'].min()} -> {val_slice['date'].max()})")
print(f"test slice:  {len(test_slice):,} rows ({test_slice['date'].min()} -> {test_slice['date'].max()})")
print(f"input feature width per timestep: {N_FEATURES}")""")

code("""X_qty_tr, X_feat_tr, idx_tr, y_tr = preprocessing.create_sequences(
    train_slice, product_to_idx,
    lookback=LOOKBACK, horizon=HORIZON,
    feature_cols=SHARED_FEATURES,
    target_col="quantity_scaled",
)
X_qty_va, X_feat_va, idx_va, y_va = preprocessing.create_sequences(
    val_slice, product_to_idx,
    lookback=LOOKBACK, horizon=HORIZON,
    feature_cols=SHARED_FEATURES,
    target_col="quantity_scaled",
)
X_qty_te, X_feat_te, idx_te, y_te = preprocessing.create_sequences(
    test_slice, product_to_idx,
    lookback=LOOKBACK, horizon=HORIZON,
    feature_cols=SHARED_FEATURES,
    target_col="quantity_scaled",
)

print(f"train windows: X_qty {X_qty_tr.shape}, X_feat {X_feat_tr.shape}, y {y_tr.shape}")
print(f"val windows:   X_qty {X_qty_va.shape}, X_feat {X_feat_va.shape}, y {y_va.shape}")
print(f"test windows:  X_qty {X_qty_te.shape}, X_feat {X_feat_te.shape}, y {y_te.shape}")""")

code("""# Seasonal-naive baseline per (window, horizon-day): repeat the last 7 days of input.
# Same definition as lstm_model.seasonal_naive_baseline().
def make_seasonal_baseline(X_qty, horizon=HORIZON, period=PERIOD):
    baselines = np.zeros((X_qty.shape[0], horizon), dtype=np.float32)
    for h in range(horizon):
        baselines[:, h] = X_qty[:, -period + (h % period), 0]
    return baselines

seasonal_tr = make_seasonal_baseline(X_qty_tr)
seasonal_va = make_seasonal_baseline(X_qty_va)
seasonal_te = make_seasonal_baseline(X_qty_te)

y_residual_tr = (y_tr - seasonal_tr).astype(np.float32)
y_residual_va = (y_va - seasonal_va).astype(np.float32)
y_residual_te = (y_te - seasonal_te).astype(np.float32)

print(f"y absolute target - mean {y_tr.mean():.4f}, std {y_tr.std():.4f}")
print(f"y residual target - mean {y_residual_tr.mean():.4f}, std {y_residual_tr.std():.4f}")
print()
print("If residual std << absolute std, seasonal-naive is doing most of the work")
print("and the LSTM only has to model the leftover signal.")""")

code("""train_ds = lstm_model.LSTMDataset(X_qty_tr, X_feat_tr, idx_tr, y_residual_tr)
val_ds   = lstm_model.LSTMDataset(X_qty_va, X_feat_va, idx_va, y_residual_va)
test_ds  = lstm_model.LSTMDataset(X_qty_te, X_feat_te, idx_te, y_residual_te)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"train batches: {len(train_loader)}")
print(f"val batches:   {len(val_loader)}")
print(f"test batches:  {len(test_loader)}")""")

# ============== SECTION 4 ==============
md("""## 4. Model selection & training

Same `LSTMForecaster` architecture but with v2 hyperparameters tuned for
the new setup:

- `embed_dim = 4` (was 8) - less memorization capacity
- `hidden_size = 64` (unchanged)
- `dropout = 0.3` (was 0.2)
- `weight_decay = 1e-4` (NEW - L2 reg, not used in v1)
- `n_calendar = 14` (the 14 shared feature columns, was 6 just calendar)

Loss is MSE on the residual target. The model output is the predicted
residual; we add the seasonal baseline back at inference.""")

code("""HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3              # v2: stronger regularization
EMBED_DIM = 4              # v2: smaller, less overfit-prone
WEIGHT_DECAY = 1e-4        # v2: L2 regularization
EPOCHS = 100
LR = 1e-3
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"training on: {DEVICE}")

model = lstm_model.LSTMForecaster(
    n_products=N_PRODUCTS,
    embed_dim=EMBED_DIM,
    lookback=LOOKBACK,
    horizon=HORIZON,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    n_calendar=N_FEATURES,
)
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"trainable parameters: {n_params:,}")""")

code("""trained_model, best_val, best_epoch, history = lstm_model.train_model(
    model, train_loader, val_loader,
    epochs=EPOCHS, lr=LR, patience=PATIENCE,
    weight_decay=WEIGHT_DECAY,
    device=DEVICE, verbose=True,
)
print()
print(f"best val MSE (on residual target): {best_val:.6f} at epoch {best_epoch}")""")

code("""fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history["train_loss"], label="train")
ax.plot(history["val_loss"], label="val")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE on residual")
ax.set_title("v2 training curve (residual target, regularized)")
ax.legend(); plt.show()""")

# ============== SECTION 5 ==============
md("""## 5. Evaluation

Five non-overlapping 30-day test windows per product. For each window:

1. Get the model's predicted **residual** (output is in scaled-residual space).
2. Add the seasonal-naive baseline (also in scaled space) to recover absolute scaled prediction.
3. Inverse-scale to real units.
4. Clip at 0 (no negative demand).
5. Compute metrics and compare against naive + seasonal-naive baselines.""")

code("""test_start = pd.Timestamp(splits["test_start"])
test_end = pd.Timestamp(splits["test_end"])

eval_rows = []
for item_id, idx in product_to_idx.items():
    series = daily[daily["item_id"] == item_id].sort_values("date").reset_index(drop=True)
    test_first_idx_arr = series.index[series["date"] == test_start]
    if len(test_first_idx_arr) == 0:
        continue
    test_first_idx = test_first_idx_arr[0]
    for w in range(5):
        target_start = test_first_idx + w * HORIZON
        target_end = target_start + HORIZON
        if target_end > len(series):
            break
        input_start = target_start - LOOKBACK
        if input_start < 0:
            break
        eval_rows.append({
            "item_id": item_id,
            "prod_idx": idx,
            "X_qty": series.loc[input_start:target_start - 1, "quantity_scaled"].values.reshape(-1, 1),
            "X_feat": series.loc[input_start:target_start - 1, SHARED_FEATURES].values,
            "y_scaled": series.loc[target_start:target_end - 1, "quantity_scaled"].values,
            "y_real": series.loc[target_start:target_end - 1, "quantity"].values,
            "target_dates": series.loc[target_start:target_end - 1, "date"].values,
        })
print(f"total eval windows: {len(eval_rows)}")""")

code("""X_qty_eval = np.stack([r["X_qty"]    for r in eval_rows]).astype(np.float32)
X_feat_eval = np.stack([r["X_feat"]    for r in eval_rows]).astype(np.float32)
idx_eval   = np.array([r["prod_idx"] for r in eval_rows], dtype=np.int64)
y_eval_scaled = np.stack([r["y_scaled"] for r in eval_rows]).astype(np.float32)
y_eval_real   = np.stack([r["y_real"]   for r in eval_rows]).astype(np.float32)
item_ids_eval = np.array([r["item_id"] for r in eval_rows])

seasonal_eval = make_seasonal_baseline(X_qty_eval)
y_residual_eval = y_eval_scaled - seasonal_eval

eval_ds_for_model = lstm_model.LSTMDataset(X_qty_eval, X_feat_eval, idx_eval, y_residual_eval)
eval_loader = DataLoader(eval_ds_for_model, batch_size=BATCH_SIZE, shuffle=False)
preds_residual_scaled = lstm_model.predict(trained_model, eval_loader, device=DEVICE)

# Add seasonal baseline back to recover absolute scaled prediction
preds_scaled = preds_residual_scaled + seasonal_eval

preds_real = lstm_model.inverse_scale_predictions(preds_scaled, item_ids_eval, scalers)
preds_real = np.clip(preds_real, 0.0, None)

print(f"preds_real shape: {preds_real.shape}")""")

code("""m_lstm = lstm_model.compute_metrics(preds_real, y_eval_real)

past_real = np.zeros_like(X_qty_eval[:, :, 0])
for i, item_id in enumerate(item_ids_eval):
    past_real[i] = scalers[item_id].inverse_transform(
        X_qty_eval[i, :, 0].reshape(-1, 1)
    ).flatten()

naive    = lstm_model.naive_baseline(past_real, HORIZON)
seasonal = lstm_model.seasonal_naive_baseline(past_real, HORIZON, period=PERIOD)

m_naive = lstm_model.compute_metrics(naive, y_eval_real)
m_seas  = lstm_model.compute_metrics(seasonal, y_eval_real)

print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
print("-" * 55)
print(f"{'Naive':<20} {m_naive['rmse']:>10.2f} {m_naive['mae']:>10.2f} {m_naive['mape']:>9.2f}%")
print(f"{'Seasonal-naive':<20} {m_seas['rmse']:>10.2f} {m_seas['mae']:>10.2f} {m_seas['mape']:>9.2f}%")
print(f"{'LSTM v2':<20} {m_lstm['rmse']:>10.2f} {m_lstm['mae']:>10.2f} {m_lstm['mape']:>9.2f}%")
print()
print("v1 reference (top-50, calendar-only features, no residual):")
print(f"{'  LSTM v1':<20} {97.97:>10.2f} {47.10:>10.2f} {24.08:>9.2f}%")""")

code("""plot_items = list(representatives) + [list(product_to_idx)[10]]
fig, axes = plt.subplots(len(plot_items), 1, figsize=(12, 9), sharex=False)
for ax, item_id in zip(axes, plot_items):
    mask = item_ids_eval == item_id
    actuals = np.concatenate([r["y_real"] for r in eval_rows if r["item_id"] == item_id])
    preds   = np.concatenate(preds_real[mask])
    ax.plot(actuals, label="actual")
    ax.plot(preds, label="LSTM v2", linestyle="--")
    ax.set_title(f"item {item_id}")
    ax.legend()
plt.tight_layout(); plt.show()""")

code("""residuals_real = (preds_real - y_eval_real).flatten()
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.histplot(residuals_real, bins=80, ax=axes[0])
axes[0].set_title("v2 residuals (predicted - actual)")
axes[0].axvline(0, color="black", linestyle=":")
axes[0].set_xlabel("residual")

per_day_rmse = np.sqrt(((preds_real - y_eval_real) ** 2).mean(axis=0))
axes[1].plot(np.arange(1, HORIZON + 1), per_day_rmse, marker="o")
axes[1].set_xlabel("forecast horizon (days ahead)")
axes[1].set_ylabel("RMSE")
axes[1].set_title("v2 error growth across horizon")

plt.tight_layout(); plt.show()
print(f"residual mean: {residuals_real.mean():.3f} (~ 0 = unbiased)")""")

code("""Path("data/predictions").mkdir(parents=True, exist_ok=True)

records = []
for r, p in zip(eval_rows, preds_real):
    for d_idx in range(HORIZON):
        records.append({
            "date": pd.Timestamp(r["target_dates"][d_idx]),
            "item_id": r["item_id"],
            "predicted_quantity": float(p[d_idx]),
            "actual_quantity": float(r["y_real"][d_idx]),
        })
predictions_df = pd.DataFrame(records)
out_path = "data/predictions/lstm_test_predictions.csv"
predictions_df.to_csv(out_path, index=False)
print(f"saved {len(predictions_df):,} rows to {out_path}")
print(f"products: {predictions_df['item_id'].nunique()}, "
      f"date range: {predictions_df['date'].min().date()} -> {predictions_df['date'].max().date()}")""")

# ============== SECTION 6 ==============
md("""## 6. Save final model

For v2 the regularized run is the final model. Skip the grid search - it
was the v1 path. If you want to sweep `dropout`, `embed_dim`, or
`weight_decay`, do that as a follow-up experiment, not in this notebook.""")

code("""Path("models").mkdir(exist_ok=True)
torch.save(trained_model.state_dict(), "models/lstm_final.pt")
with open("models/lstm_config.json", "w") as f:
    json.dump({
        "version": "v2",
        "n_products": N_PRODUCTS,
        "embed_dim": EMBED_DIM,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "n_calendar": N_FEATURES,
        "feature_cols": SHARED_FEATURES,
        "target_mode": "residual_against_seasonal_naive",
        "period": PERIOD,
        "best_epoch": int(best_epoch),
        "test_metrics": m_lstm,
        "baselines": {"naive": m_naive, "seasonal_naive": m_seas},
    }, f, indent=2)
print("saved models/lstm_final.pt and models/lstm_config.json")""")

md("""### Notes on the v2 result

When interpreting the metrics above:

- The model trained on residuals - its raw output is *not* a demand prediction; it's the *correction* to add on top of seasonal-naive. The reconstruction (`prediction = seasonal_baseline + model_output`) happens at inference.
- If `LSTM v2` RMSE / MAE is below `Seasonal-naive`, the LSTM is genuinely adding signal beyond weekly seasonality. That was the goal.
- If the LSTM is now WORSE than seasonal-naive, the model is hurting more than it's helping - try `weight_decay=1e-3`, `dropout=0.4`, or revert to MSE on absolute targets.
- v1 numbers (top-50 products, calendar-only features, no residual): RMSE 97.97, MAE 47.10, MAPE 24.08%. v2 results are NOT directly comparable because the product universe and feature set are both different, but the relative position to seasonal-naive *is* the right comparison.""")


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
        for k, s in CELLS
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    out = "notebooks/02_LSTM.ipynb"
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {out} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
