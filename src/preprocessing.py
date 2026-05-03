"""Framework-agnostic preprocessing for demand forecasting.

All functions operate on pandas DataFrames and numpy arrays.
No PyTorch / TensorFlow / Keras imports — so the team's sklearn-based
model notebooks can import this module without conflict.
"""

from __future__ import annotations

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
