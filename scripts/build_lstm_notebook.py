"""One-time builder for notebooks/02_LSTM.ipynb (v3).

This script regenerates 02_LSTM.ipynb against the shared preprocessing
contract. Run: python scripts/build_lstm_notebook.py

v3 differences from v2 (post-mortem driven):
  - Direct prediction (revert from residual-against-seasonal-naive).
    Residual training in v2 produced a systematic -34.9 bias on test;
    val window was too short to validate residual stability, and the
    LSTM's tiny learned residuals inverse-scaled into large negative bias
    for high-volume products.
  - Drop two features from the 14 shared columns:
      * returns_quantity_scaled (all zeros in train period -> OOD noise at test)
      * positive_quantity_scaled (literal duplicate of quantity_scaled)
    Final feature count: 12.
  - Huber loss (SmoothL1Loss) instead of MSE - more robust to spike days.
  - Early-stop patience 5 (was 10); val curve flattened by epoch ~11.
  - Keeps top-100 universe, hidden 64, layers 2, dropout 0.3, embed 4,
    weight_decay 1e-4.
"""

import nbformat as nbf

CELLS = []


def md(s):
    CELLS.append(("md", s))


def code(s):
    CELLS.append(("code", s))


# ============== Title ==============
md("""# 02 - LSTM Demand Forecasting (v3)

End-to-end PyTorch LSTM on the shared **top-100 / 730-active-day** universe.

v3 is a course-correction from v2. The v2 residual-training approach
underperformed seasonal-naive on test by a wide margin (test RMSE 167.6 vs
seasonal-naive 54.7) because the 2-month val window couldn't validate that
the residual signal was stable into the 5-month test window. v3 reverts to
direct prediction but keeps the other v2 gains.

**What changed from v2:**

1. **Direct prediction** - model output is `y_scaled` (not `y - seasonal_naive`)
2. **Cleaner feature set** (12 cols) - dropped `returns_quantity_scaled` (all
   zeros in train -> OOD noise at test) and `positive_quantity_scaled` (literal
   duplicate of `quantity_scaled`)
3. **Huber loss** (SmoothL1Loss) - more robust to demand spikes than MSE
4. **Early-stop patience 5** (was 10) - v2's val curve plateaued by epoch ~11
5. Keeps: top-100, hidden 64, layers 2, dropout 0.3, embed 4, weight_decay 1e-4

Ensemble contract: predictions saved to `data/predictions/lstm_test_predictions.csv` with the canonical schema `date,item_id,predicted_quantity,actual_quantity`. Same product list and test window as every other ensemble model.""")

# ============== Colab bootstrap ==============
md("""## 0. Environment setup (Colab)

**Local runs:** skip this cell - it's a no-op outside Colab.

**Colab runs:** this cell clones the repo (fresh, fast working tree), mounts
your Google Drive, and symlinks `sales.csv` + `data/processed/` from Drive
into the clone. Prerequisite: the project is already on Drive at:

```
MyDrive/demand-forecasting/
  sales.csv
  data/
    processed/      <-- daily.csv, scalers.pkl, feature_columns.json, ...
```

If your Drive layout is different, edit `DRIVE_DATA_DIR` below.""")

code("""# === Colab bootstrap (no-op when running locally) ===
import os, sys, subprocess

def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

if _in_colab():
    REPO_URL = "https://github.com/nhbi05/demand-forecasting.git"
    BRANCH = "feature/lstm-design"
    REPO_DIR = "/content/demand-forecasting"
    # Points at the project folder on Drive (containing sales.csv + data/).
    DRIVE_DATA_DIR = "/content/drive/MyDrive/demand-forecasting"

    if not os.path.exists(REPO_DIR):
        subprocess.run(
            ["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, REPO_DIR],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=False)

    from google.colab import drive  # noqa: E402
    drive.mount("/content/drive")

    src_csv = os.path.join(DRIVE_DATA_DIR, "sales.csv")
    src_processed = os.path.join(DRIVE_DATA_DIR, "data", "processed")
    dst_csv = os.path.join(REPO_DIR, "sales.csv")
    dst_processed = os.path.join(REPO_DIR, "data", "processed")
    os.makedirs(os.path.join(REPO_DIR, "data"), exist_ok=True)

    for src, dst in [(src_csv, dst_csv), (src_processed, dst_processed)]:
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Expected {src} on Drive. Upload it (or fix DRIVE_DATA_DIR).")
        if os.path.lexists(dst) and not os.path.samefile(dst, src):
            if os.path.islink(dst) or os.path.isfile(dst):
                os.remove(dst)
        if not os.path.lexists(dst):
            os.symlink(src, dst)

    os.chdir(REPO_DIR)
    print(f"Repo:       {REPO_DIR}")
    print(f"Data root:  {DRIVE_DATA_DIR}")
    print(f"sales.csv:  {os.path.exists(dst_csv)}")
    print(f"processed:  {os.path.exists(dst_processed)}")
else:
    print("Not running in Colab - skipping bootstrap.")""")

# ============== SECTION 1 ==============
md("""## 1. Data preparation

Loads the shared artifacts written by `src.preprocessing.prepare_data()`.
Regenerates them if they don't exist yet.""")

code("""import sys, os
# Ensure CWD is the repo root and that the root is on sys.path so `from src
# import ...` works whether launched from notebooks/, repo root, or Colab.
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())

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

Quick checks on the dataset - top 100 products and the scaled feature
ranges in the training period.""")

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

Two things happen here:

1. Build sequences with the **12-feature input** (dropped `returns_quantity_scaled` which is all zeros in train, and `positive_quantity_scaled` which duplicates `quantity_scaled`).
2. Wrap everything in DataLoaders.

The model output is the direct scaled quantity for the next 30 days. We inverse-scale at evaluation time.""")

code("""LOOKBACK = 60
HORIZON = 30
PERIOD = 7   # weekly seasonality (used for the seasonal-naive baseline only)
BATCH_SIZE = 64

slices = preprocessing.slice_by_split(daily, splits, lookback_days=LOOKBACK)
train_slice, val_slice, test_slice = slices["train"], slices["val"], slices["test"]

product_to_idx = {p: i for i, p in enumerate(sorted(selected_products))}
N_PRODUCTS = len(product_to_idx)

# v3: drop two of the 14 shared features. Contract is unchanged - we just
# choose which subset *this* model consumes.
DROPPED_FEATURES = {"returns_quantity_scaled", "positive_quantity_scaled"}
SHARED_FEATURES = [c for c in feature_columns["shared_default"]
                   if c not in DROPPED_FEATURES]
N_FEATURES = len(SHARED_FEATURES)

print(f"train slice: {len(train_slice):,} rows ({train_slice['date'].min()} -> {train_slice['date'].max()})")
print(f"val slice:   {len(val_slice):,} rows ({val_slice['date'].min()} -> {val_slice['date'].max()})")
print(f"test slice:  {len(test_slice):,} rows ({test_slice['date'].min()} -> {test_slice['date'].max()})")
print(f"using {N_FEATURES} of {len(feature_columns['shared_default'])} shared feature columns")
print(f"dropped: {sorted(DROPPED_FEATURES)}")""")

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

code("""# Quick diagnostic: training target distribution.
print(f"y_scaled target - mean {y_tr.mean():.4f}, std {y_tr.std():.4f}, "
      f"min {y_tr.min():.4f}, max {y_tr.max():.4f}")""")

code("""train_ds = lstm_model.LSTMDataset(X_qty_tr, X_feat_tr, idx_tr, y_tr)
val_ds   = lstm_model.LSTMDataset(X_qty_va, X_feat_va, idx_va, y_va)
test_ds  = lstm_model.LSTMDataset(X_qty_te, X_feat_te, idx_te, y_te)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"train batches: {len(train_loader)}")
print(f"val batches:   {len(val_loader)}")
print(f"test batches:  {len(test_loader)}")""")

# ============== SECTION 4 ==============
md("""## 4. Model selection & training

Same `LSTMForecaster` architecture with the v3 hyperparameters:

- `embed_dim = 4`, `hidden_size = 64`, `num_layers = 2`
- `dropout = 0.3`, `weight_decay = 1e-4`
- `n_calendar = 12` (the 12 features after dropping returns/positive)
- **Huber loss** (`SmoothL1Loss`) - L2 near zero, L1 in the tails. Less
  punishment for big spike days, which helped bias the v2 model
- **Patience = 5** - v2's val curve plateaued by epoch ~11; tighter early
  stop catches the right epoch instead of overfitting through noise

The target is `y_scaled` directly; we inverse-scale predictions at evaluation.""")

code("""HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
EMBED_DIM = 4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
LR = 1e-3
PATIENCE = 5               # v3: tighter than v2's 10
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
    loss_fn=torch.nn.SmoothL1Loss(),   # v3: Huber instead of MSE
    device=DEVICE, verbose=True,
)
print()
print(f"best val Huber loss: {best_val:.6f} at epoch {best_epoch}")""")

code("""fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history["train_loss"], label="train")
ax.plot(history["val_loss"], label="val")
ax.set_xlabel("epoch"); ax.set_ylabel("Huber loss")
ax.set_title("v3 training curve (direct prediction, Huber loss)")
ax.legend(); plt.show()""")

# ============== SECTION 5 ==============
md("""## 5. Evaluation

Five non-overlapping 30-day test windows per product. For each window:

1. Run the model. Output is in **scaled** space directly (no residual reconstruction).
2. Inverse-scale to real units, per product.
3. Clip at 0 (no negative demand).
4. Compute metrics and compare against naive + seasonal-naive baselines.""")

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

eval_ds_for_model = lstm_model.LSTMDataset(X_qty_eval, X_feat_eval, idx_eval, y_eval_scaled)
eval_loader = DataLoader(eval_ds_for_model, batch_size=BATCH_SIZE, shuffle=False)
preds_scaled = lstm_model.predict(trained_model, eval_loader, device=DEVICE)

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
print(f"{'LSTM v3':<20} {m_lstm['rmse']:>10.2f} {m_lstm['mae']:>10.2f} {m_lstm['mape']:>9.2f}%")
print()
print("Historical reference points (different setups, NOT apples-to-apples):")
print(f"{'  LSTM v1 (top-50)':<20} {97.97:>10.2f} {47.10:>10.2f} {24.08:>9.2f}%")
print(f"{'  LSTM v2 (residual)':<20} {167.63:>10.2f} {50.24:>10.2f} {34.59:>9.2f}%")""")

code("""plot_items = list(representatives) + [list(product_to_idx)[10]]
fig, axes = plt.subplots(len(plot_items), 1, figsize=(12, 9), sharex=False)
for ax, item_id in zip(axes, plot_items):
    mask = item_ids_eval == item_id
    actuals = np.concatenate([r["y_real"] for r in eval_rows if r["item_id"] == item_id])
    preds   = np.concatenate(preds_real[mask])
    ax.plot(actuals, label="actual")
    ax.plot(preds, label="LSTM v3", linestyle="--")
    ax.set_title(f"item {item_id}")
    ax.legend()
plt.tight_layout(); plt.show()""")

code("""residuals_real = (preds_real - y_eval_real).flatten()
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.histplot(residuals_real, bins=80, ax=axes[0])
axes[0].set_title("v3 residuals (predicted - actual)")
axes[0].axvline(0, color="black", linestyle=":")
axes[0].set_xlabel("residual")

per_day_rmse = np.sqrt(((preds_real - y_eval_real) ** 2).mean(axis=0))
axes[1].plot(np.arange(1, HORIZON + 1), per_day_rmse, marker="o")
axes[1].set_xlabel("forecast horizon (days ahead)")
axes[1].set_ylabel("RMSE")
axes[1].set_title("v3 error growth across horizon")

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

Save the trained weights and a config snapshot. The config lists the exact
feature set and hyperparameters so the artifacts are self-describing.""")

code("""Path("models").mkdir(exist_ok=True)
torch.save(trained_model.state_dict(), "models/lstm_final.pt")
with open("models/lstm_config.json", "w") as f:
    json.dump({
        "version": "v3",
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
        "dropped_from_shared_default": sorted(DROPPED_FEATURES),
        "target_mode": "direct_scaled_quantity",
        "loss": "SmoothL1Loss",
        "patience": PATIENCE,
        "best_epoch": int(best_epoch),
        "test_metrics": m_lstm,
        "baselines": {"naive": m_naive, "seasonal_naive": m_seas},
    }, f, indent=2)
print("saved models/lstm_final.pt and models/lstm_config.json")""")

md("""### Reading the v3 results

What to look for in the metric table above:

- **Match or beat seasonal-naive on RMSE.** That's the bar. Seasonal-naive captured ~32% of target variance for free; v3 needs to capture more of the remaining 68% than seasonal-naive captures of nothing.
- **Residual mean near 0.** v2 had a -34.9 bias from the residual-training mechanic. With direct prediction, this should land near 0; any systematic offset points to a remaining issue (scaler drift, distribution shift, etc.).
- **Per-horizon RMSE growth is gentle.** A steep climb from day 1 to day 30 means the model is over-anchored to the input window and not learning forward dynamics.

If v3 still loses to seasonal-naive on RMSE, the next moves are (in order):

1. Stacked output: `final = alpha * lstm + (1-alpha) * seasonal_naive`, with alpha fit on val. Cheapest insurance.
2. Add a 30-day rolling-mean feature so the model can track level shifts across the train/test boundary.
3. Switch from MinMax to `log(quantity + 1)` per-product, eliminating the scale-amplification of small biases for high-volume products.""")


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
