"""Framework-agnostic preprocessing for demand forecasting.

All functions operate on pandas DataFrames and numpy arrays.
No PyTorch / TensorFlow / Keras imports — so the team's sklearn-based
model notebooks can import this module without conflict.
"""

from __future__ import annotations

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
