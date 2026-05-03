"""Framework-agnostic preprocessing for demand forecasting.

All functions operate on pandas DataFrames and numpy arrays.
No PyTorch / TensorFlow / Keras imports — so the team's sklearn-based
model notebooks can import this module without conflict.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_raw_sales(csv_path: str = "sales.csv") -> pd.DataFrame:
    """Load the raw sales CSV, drop the unnamed index column, parse dates."""
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Sum `quantity` per (item_id, date) across all stores.

    Negative quantities (returns) are kept and net into the daily total —
    a return of 5 on a day that also had 8 sales becomes net demand of 3.
    """
    return df.groupby(["item_id", "date"], as_index=False)["quantity"].sum()


def select_top_products(daily: pd.DataFrame, n: int = 50,
                        full_history_days: int = 761) -> list[str]:
    """Return the IDs of the top-N products by total quantity, restricted to
    those that had at least one sale on every day in the full date range."""
    counts = daily.groupby("item_id")["date"].nunique()
    eligible = counts[counts >= full_history_days].index
    totals = (
        daily[daily["item_id"].isin(eligible)]
        .groupby("item_id")["quantity"]
        .sum()
    )
    return totals.nlargest(n).index.tolist()


def cap_outliers(daily: pd.DataFrame, percentile: float = 99.5) -> pd.DataFrame:
    """Winsorize each product's quantity at its own percentile.

    Values above the cap are replaced by the cap. Values below are unchanged.
    """
    out = daily.copy()
    caps = (
        daily.groupby("item_id")["quantity"]
        .transform(lambda s: s.quantile(percentile / 100.0))
    )
    out["quantity"] = np.minimum(out["quantity"], caps)
    return out


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical sin/cos encodings of day-of-week, day-of-month, month.

    Cyclical encoding tells the model that, e.g., Sunday (6) and Monday (0)
    are adjacent — a plain integer encoding wouldn't.
    """
    out = daily.copy()
    dow = out["date"].dt.dayofweek          # 0..6
    dom = out["date"].dt.day                # 1..31
    month = out["date"].dt.month            # 1..12

    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["dom_sin"] = np.sin(2 * np.pi * (dom - 1) / 31.0)
    out["dom_cos"] = np.cos(2 * np.pi * (dom - 1) / 31.0)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    return out


def split_time_based(daily: pd.DataFrame, val_days: int = 60,
                     test_frac: float = 0.20) -> dict:
    """Time-based 80/20 train/test split with a `val_days` slice carved out
    of the train portion (used for early stopping and tuning).

    Returns a dict of ISO date strings: train_start, train_end, val_start,
    val_end, test_start, test_end.
    """
    dates = sorted(pd.to_datetime(daily["date"].unique()))
    n = len(dates)

    test_size = int(n * test_frac)
    train_plus_val_size = n - test_size
    train_size = train_plus_val_size - val_days

    if train_size < 1:
        raise ValueError(
            f"Not enough data: {n} days with val_days={val_days}, "
            f"test_frac={test_frac} leaves {train_size} train days."
        )

    return {
        "train_start": dates[0].strftime("%Y-%m-%d"),
        "train_end":   dates[train_size - 1].strftime("%Y-%m-%d"),
        "val_start":   dates[train_size].strftime("%Y-%m-%d"),
        "val_end":     dates[train_size + val_days - 1].strftime("%Y-%m-%d"),
        "test_start":  dates[train_size + val_days].strftime("%Y-%m-%d"),
        "test_end":    dates[-1].strftime("%Y-%m-%d"),
    }


def scale_per_product(daily: pd.DataFrame, train_end_date) -> tuple[pd.DataFrame, dict]:
    """Fit a MinMaxScaler per product on data up to and including
    `train_end_date`, then transform all data (including val and test).

    Fitting on train-only prevents leakage: the scaler doesn't "see" future
    values when computing min/max.

    Returns:
        (scaled_df, scalers) where scaled_df has a new column `quantity_scaled`
        and scalers is `{item_id: MinMaxScaler}` for inverse-scaling later.
    """
    train_end_date = pd.to_datetime(train_end_date)
    out = daily.copy().sort_values(["item_id", "date"]).reset_index(drop=True)
    out["quantity_scaled"] = 0.0
    scalers: dict = {}

    for item_id, group in out.groupby("item_id"):
        train_mask = group["date"] <= train_end_date
        train_qty = group.loc[train_mask, "quantity"].values.reshape(-1, 1)
        if len(train_qty) == 0:
            raise ValueError(f"Product {item_id} has no training rows.")
        scaler = MinMaxScaler()
        scaler.fit(train_qty)
        scalers[item_id] = scaler
        out.loc[group.index, "quantity_scaled"] = scaler.transform(
            group["quantity"].values.reshape(-1, 1)
        ).flatten()

    return out, scalers


CALENDAR_COLS = ["dow_sin", "dow_cos", "dom_sin", "dom_cos", "month_sin", "month_cos"]


def create_sequences(
    daily: pd.DataFrame,
    product_to_idx: dict[str, int],
    lookback: int = 60,
    horizon: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding-window (input, target) examples per product.

    For each product's contiguous time series, slide a window of size
    `lookback` and predict the next `horizon` days. Stride = 1.

    Args:
        daily: must be sorted by (item_id, date) and contain `quantity_scaled`
            plus the six CALENDAR_COLS.
        product_to_idx: maps item_id (str) to integer index for embedding.
        lookback: number of past days used as model input.
        horizon: number of future days predicted.

    Returns:
        X_qty:    (N, lookback, 1)
        X_cal:    (N, lookback, 6)
        prod_idx: (N,)  int64
        y:        (N, horizon)
    """
    df = daily.sort_values(["item_id", "date"]).reset_index(drop=True)

    X_qty_list, X_cal_list, idx_list, y_list = [], [], [], []

    for item_id, group in df.groupby("item_id"):
        if item_id not in product_to_idx:
            continue
        qty = group["quantity_scaled"].values
        cal = group[CALENDAR_COLS].values
        n = len(qty)
        max_start = n - lookback - horizon + 1
        if max_start <= 0:
            continue
        pidx = product_to_idx[item_id]
        for start in range(max_start):
            X_qty_list.append(qty[start:start + lookback].reshape(-1, 1))
            X_cal_list.append(cal[start:start + lookback])
            idx_list.append(pidx)
            y_list.append(qty[start + lookback:start + lookback + horizon])

    X_qty = np.asarray(X_qty_list, dtype=np.float32)
    X_cal = np.asarray(X_cal_list, dtype=np.float32)
    prod_idx = np.asarray(idx_list, dtype=np.int64)
    y = np.asarray(y_list, dtype=np.float32)
    return X_qty, X_cal, prod_idx, y


def prepare_data(
    csv_path: str = "sales.csv",
    n_products: int = 50,
    full_history_days: int = 761,
    val_days: int = 60,
    test_frac: float = 0.20,
    outlier_pct: float = 99.5,
    outdir: str = "data/processed",
) -> dict:
    """End-to-end preprocessing.

    Pipeline:
        load → aggregate daily → select top-N → cap outliers
              → add calendar features → time split → per-product scaling
              → write daily.csv, scalers.pkl, splits.json

    Returns the split-boundaries dict.
    """
    raw = load_raw_sales(csv_path)
    daily = aggregate_daily(raw)
    top_ids = select_top_products(daily, n=n_products,
                                  full_history_days=full_history_days)
    daily = daily[daily["item_id"].isin(top_ids)].copy()
    daily = cap_outliers(daily, percentile=outlier_pct)
    daily = add_calendar_features(daily)

    splits = split_time_based(daily, val_days=val_days, test_frac=test_frac)
    daily, scalers = scale_per_product(daily, splits["train_end"])

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "daily.csv", index=False)
    with open(out / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    with open(out / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    return splits
