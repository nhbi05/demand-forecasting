"""Framework-agnostic preprocessing for demand forecasting.

All functions operate on pandas DataFrames and numpy arrays.
No PyTorch / TensorFlow / Keras imports — so the team's sklearn-based
model notebooks can import this module without conflict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
