"""Shared preprocessing contract for the demand-forecasting project.

This module is intentionally framework-agnostic: pandas, numpy, and sklearn
only. All notebooks should start from the artifacts written here, then add
model-specific feature engineering in their own notebooks.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


RAW_REQUIRED_COLUMNS = [
    "date",
    "item_id",
    "quantity",
    "price_base",
    "sum_total",
    "store_id",
]
CRITICAL_RAW_COLUMNS = ["date", "item_id", "quantity"]
OPTIONAL_RAW_COLUMNS = ["price_base", "sum_total", "store_id"]

CALENDAR_COLS = ["dow_sin", "dow_cos", "dom_sin", "dom_cos", "month_sin", "month_cos"]
BASE_TARGET_COL = "quantity"
DEFAULT_SCALE_COLUMNS = [
    "quantity",
    "positive_quantity",
    "returns_quantity",
    "price_base",
    "sum_total",
    "transaction_count",
    "store_count",
]
DEFAULT_SHARED_FEATURE_COLS = CALENDAR_COLS + [
    "quantity_scaled",
    "positive_quantity_scaled",
    "returns_quantity_scaled",
    "price_base_scaled",
    "sum_total_scaled",
    "transaction_count_scaled",
    "store_count_scaled",
    "had_sales",
]
DEFAULT_N_PRODUCTS = 100
DEFAULT_MIN_ACTIVE_DAYS = 730
DEFAULT_PRODUCT_RANK_COL = "positive_quantity"


def validate_raw_sales(df: pd.DataFrame,
                       required_columns: Iterable[str] = RAW_REQUIRED_COLUMNS) -> None:
    """Validate that the raw sales frame contains the shared input schema."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Raw sales data is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Raw sales data is empty.")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's raw-data missing-value policy.

    Rows missing date, item_id, or quantity cannot be recovered and are dropped.
    Missing price is filled by item median, then global median, then 0. Missing
    sum_total is recomputed as quantity * price_base. Missing store_id is kept as
    a sentinel store so transaction counts are preserved.

    The function adds `*_was_missing` indicator columns so downstream notebooks
    can inspect how much imputation happened.
    """
    validate_raw_sales(df)
    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    before = len(out)
    out = out.dropna(subset=CRITICAL_RAW_COLUMNS).copy()
    out.attrs["dropped_critical_missing_rows"] = before - len(out)

    for col in ["quantity", "price_base", "sum_total"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["quantity"]).copy()
    out.attrs["dropped_quantity_parse_rows"] = before - len(out)

    out["price_base_was_missing"] = out["price_base"].isna()
    item_price_median = out.groupby("item_id")["price_base"].transform("median")
    global_price_median = out["price_base"].median()
    if pd.isna(global_price_median):
        global_price_median = 0.0
    out["price_base"] = (
        out["price_base"]
        .fillna(item_price_median)
        .fillna(global_price_median)
        .fillna(0.0)
    )

    out["sum_total_was_missing"] = out["sum_total"].isna()
    out["sum_total"] = out["sum_total"].fillna(out["quantity"] * out["price_base"])

    out["store_id_was_missing"] = out["store_id"].isna()
    out["store_id"] = out["store_id"].fillna("__missing_store__")
    return out


def load_raw_sales(csv_path: str = "sales.csv") -> pd.DataFrame:
    """Load, validate, clean, and type-normalize the raw sales CSV."""
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    validate_raw_sales(df)
    return handle_missing_values(df)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction rows to one row per product per day.

    `quantity` remains net demand for backward compatibility: returns are kept
    as negative quantities and net into the daily total. Additional shared
    features preserve useful signal for every model:

    - `positive_quantity`: sum of positive sales only
    - `returns_quantity`: absolute returned quantity
    - `price_base`: daily mean price
    - `sum_total`: daily revenue / transaction total
    - `transaction_count`: number of raw rows
    - `store_count`: number of stores represented that day
    - `had_sales`: 1 when positive_quantity > 0, otherwise 0
    """
    if not {"item_id", "date", "quantity"}.issubset(df.columns):
        raise ValueError("Daily aggregation requires item_id, date, and quantity columns.")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["positive_quantity"] = work["quantity"].clip(lower=0)
    work["returns_quantity"] = (-work["quantity"].clip(upper=0)).astype(float)

    agg_spec: dict[str, tuple[str, str]] = {
        "quantity": ("quantity", "sum"),
        "positive_quantity": ("positive_quantity", "sum"),
        "returns_quantity": ("returns_quantity", "sum"),
        "transaction_count": ("quantity", "size"),
    }
    if "price_base" in work.columns:
        agg_spec["price_base"] = ("price_base", "mean")
        agg_spec["price_base_median"] = ("price_base", "median")
    if "sum_total" in work.columns:
        agg_spec["sum_total"] = ("sum_total", "sum")
    if "store_id" in work.columns:
        agg_spec["store_count"] = ("store_id", "nunique")
    if "price_base_was_missing" in work.columns:
        agg_spec["price_base_missing_count"] = ("price_base_was_missing", "sum")
    if "sum_total_was_missing" in work.columns:
        agg_spec["sum_total_missing_count"] = ("sum_total_was_missing", "sum")
    if "store_id_was_missing" in work.columns:
        agg_spec["store_id_missing_count"] = ("store_id_was_missing", "sum")

    daily = work.groupby(["item_id", "date"], as_index=False).agg(**agg_spec)

    defaults = {
        "price_base": 0.0,
        "price_base_median": 0.0,
        "sum_total": 0.0,
        "store_count": 0,
        "price_base_missing_count": 0,
        "sum_total_missing_count": 0,
        "store_id_missing_count": 0,
    }
    for col, value in defaults.items():
        if col not in daily.columns:
            daily[col] = value

    daily["had_sales"] = (daily["positive_quantity"] > 0).astype(int)
    return daily.sort_values(["item_id", "date"]).reset_index(drop=True)


def select_top_products(
    daily: pd.DataFrame,
    n: int | None = DEFAULT_N_PRODUCTS,
    full_history_days: int | None = None,
    min_history_days: int | None = None,
    min_active_days: int | None = DEFAULT_MIN_ACTIVE_DAYS,
    volume_col: str = DEFAULT_PRODUCT_RANK_COL,
) -> list[str]:
    """Return product IDs ordered by demand volume after activity filters.

    The default team benchmark is top 100 products by `positive_quantity`,
    restricted to products with at least 730 active sales days. An active sales
    day means `positive_quantity > 0`.

    `full_history_days` and `min_history_days` are kept for backward
    compatibility with the older row-count filter. Pass `n=None` to keep every
    eligible product.
    """
    if min_history_days is None:
        min_history_days = full_history_days

    product_groups = daily.groupby("item_id")
    row_days = product_groups["date"].nunique()
    eligible = set(row_days.index)
    if min_history_days is not None:
        eligible &= set(row_days[row_days >= min_history_days].index)

    if min_active_days is not None:
        if "positive_quantity" in daily.columns:
            active_days = product_groups["positive_quantity"].apply(lambda s: int((s > 0).sum()))
        else:
            active_days = product_groups[volume_col].apply(lambda s: int((s > 0).sum()))
        eligible &= set(active_days[active_days >= min_active_days].index)

    totals = (
        daily[daily["item_id"].isin(sorted(eligible))]
        .groupby("item_id")[volume_col]
        .sum()
        .sort_values(ascending=False)
    )
    if n is not None:
        totals = totals.head(n)
    return totals.index.tolist()


def complete_product_date_grid(
    daily: pd.DataFrame,
    product_ids: Iterable[str] | None = None,
    dates: Iterable | None = None,
) -> pd.DataFrame:
    """Add explicit zero-sale rows for missing product-date combinations.

    This is critical once the project expands beyond products that sold every
    day. A missing raw transaction for a selected product-date means zero
    observed demand, not "unknown row".
    """
    if product_ids is None:
        product_ids = sorted(daily["item_id"].unique())
    else:
        product_ids = list(product_ids)

    if dates is None:
        dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    else:
        dates = pd.to_datetime(list(dates))

    base_index = pd.MultiIndex.from_product(
        [product_ids, dates],
        names=["item_id", "date"],
    )
    out = (
        daily.set_index(["item_id", "date"])
        .reindex(base_index)
        .reset_index()
        .sort_values(["item_id", "date"])
        .reset_index(drop=True)
    )

    zero_fill_cols = [
        "quantity",
        "positive_quantity",
        "returns_quantity",
        "sum_total",
        "transaction_count",
        "store_count",
        "price_base_missing_count",
        "sum_total_missing_count",
        "store_id_missing_count",
    ]
    for col in zero_fill_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    price_cols = [col for col in ["price_base", "price_base_median"] if col in out.columns]
    for col in price_cols:
        global_median = out[col].median()
        if pd.isna(global_median):
            global_median = 0.0
        out[col] = (
            out.groupby("item_id")[col]
            .transform(lambda s: s.ffill().bfill())
            .fillna(global_median)
            .fillna(0.0)
        )

    out["had_sales"] = (out["positive_quantity"] > 0).astype(int)
    count_cols = [
        "transaction_count",
        "store_count",
        "price_base_missing_count",
        "sum_total_missing_count",
        "store_id_missing_count",
        "had_sales",
    ]
    for col in count_cols:
        if col in out.columns:
            out[col] = out[col].astype(int)
    return out


def split_time_based(daily: pd.DataFrame, val_days: int = 60,
                     test_frac: float = 0.20) -> dict:
    """Time-based 80/20 train/test split with a validation tail."""
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
        "train_end": dates[train_size - 1].strftime("%Y-%m-%d"),
        "val_start": dates[train_size].strftime("%Y-%m-%d"),
        "val_end": dates[train_size + val_days - 1].strftime("%Y-%m-%d"),
        "test_start": dates[train_size + val_days].strftime("%Y-%m-%d"),
        "test_end": dates[-1].strftime("%Y-%m-%d"),
    }


def fit_outlier_caps(
    daily: pd.DataFrame,
    train_end_date,
    columns: Iterable[str] = (BASE_TARGET_COL,),
    percentile: float = 99.5,
) -> dict:
    """Fit per-product upper caps using train rows only."""
    train_end_date = pd.to_datetime(train_end_date)
    train = daily[daily["date"] <= train_end_date]
    caps: dict[str, dict[str, float]] = {}
    for col in columns:
        col_caps = (
            train.groupby("item_id")[col]
            .quantile(percentile / 100.0)
            .astype(float)
            .to_dict()
        )
        caps[col] = col_caps
    return caps


def apply_outlier_caps(daily: pd.DataFrame, caps: dict) -> pd.DataFrame:
    """Apply caps returned by `fit_outlier_caps` to all splits."""
    out = daily.copy()
    for col, col_caps in caps.items():
        cap_series = out["item_id"].map(col_caps)
        out[col] = np.minimum(out[col], cap_series.fillna(out[col]))
    return out


def cap_outliers(
    daily: pd.DataFrame,
    percentile: float = 99.5,
    train_end_date=None,
    columns: Iterable[str] = (BASE_TARGET_COL,),
    return_caps: bool = False,
):
    """Winsorize each product's columns at its own percentile.

    If `train_end_date` is supplied, caps are fit on train rows only to avoid
    leakage. Without it, behavior matches the original v1 helper and fits on
    the provided frame.
    """
    if train_end_date is None:
        caps = {}
        for col in columns:
            caps[col] = (
                daily.groupby("item_id")[col]
                .quantile(percentile / 100.0)
                .astype(float)
                .to_dict()
            )
    else:
        caps = fit_outlier_caps(daily, train_end_date, columns=columns, percentile=percentile)

    out = apply_outlier_caps(daily, caps)
    if return_caps:
        return out, caps
    return out


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encodings of day-of-week, day-of-month, and month."""
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"])
    dow = out["date"].dt.dayofweek
    dom = out["date"].dt.day
    month = out["date"].dt.month

    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["dom_sin"] = np.sin(2 * np.pi * (dom - 1) / 31.0)
    out["dom_cos"] = np.cos(2 * np.pi * (dom - 1) / 31.0)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    return out


def scale_per_product(
    daily: pd.DataFrame,
    train_end_date,
    value_col: str = BASE_TARGET_COL,
    output_col: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit a MinMaxScaler per product on train rows, then transform all rows."""
    train_end_date = pd.to_datetime(train_end_date)
    output_col = output_col or f"{value_col}_scaled"
    out = daily.copy().sort_values(["item_id", "date"]).reset_index(drop=True)
    out[output_col] = 0.0
    scalers: dict = {}

    for item_id, group in out.groupby("item_id"):
        train_mask = group["date"] <= train_end_date
        train_values = group.loc[train_mask, value_col].values.reshape(-1, 1)
        if len(train_values) == 0:
            raise ValueError(f"Product {item_id} has no training rows.")
        scaler = MinMaxScaler()
        scaler.fit(train_values)
        scalers[item_id] = scaler
        out.loc[group.index, output_col] = scaler.transform(
            group[value_col].values.reshape(-1, 1)
        ).flatten()

    return out, scalers


def scale_feature_columns(
    daily: pd.DataFrame,
    train_end_date,
    columns: Iterable[str] = DEFAULT_SCALE_COLUMNS,
) -> tuple[pd.DataFrame, dict]:
    """Scale multiple continuous columns per product using train-only fits."""
    out = daily.copy()
    all_scalers: dict[str, dict] = {}
    for col in columns:
        if col not in out.columns:
            continue
        out, scalers = scale_per_product(out, train_end_date, value_col=col)
        all_scalers[col] = scalers
    return out, all_scalers


def create_sequences(
    daily: pd.DataFrame,
    product_to_idx: dict[str, int],
    lookback: int = 60,
    horizon: int = 30,
    feature_cols: Iterable[str] | None = None,
    target_col: str = "quantity_scaled",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding-window examples per product.

    Default output remains backward-compatible with v1:
    X_qty is `(N, lookback, 1)`, X_cal is `(N, lookback, 6)`, prod_idx is
    `(N,)`, and y is `(N, horizon)`.

    Pass `feature_cols=DEFAULT_SHARED_FEATURE_COLS` to give an RNN/LSTM the
    richer shared features produced by `prepare_data`.
    """
    feature_cols = list(feature_cols) if feature_cols is not None else CALENDAR_COLS
    df = daily.sort_values(["item_id", "date"]).reset_index(drop=True)

    X_qty_list, X_feature_list, idx_list, y_list = [], [], [], []

    for item_id, group in df.groupby("item_id"):
        if item_id not in product_to_idx:
            continue
        qty = group[target_col].values
        features = group[feature_cols].values
        n = len(qty)
        max_start = n - lookback - horizon + 1
        if max_start <= 0:
            continue
        pidx = product_to_idx[item_id]
        for start in range(max_start):
            X_qty_list.append(qty[start:start + lookback].reshape(-1, 1))
            X_feature_list.append(features[start:start + lookback])
            idx_list.append(pidx)
            y_list.append(qty[start + lookback:start + lookback + horizon])

    X_qty = np.asarray(X_qty_list, dtype=np.float32)
    X_features = np.asarray(X_feature_list, dtype=np.float32)
    prod_idx = np.asarray(idx_list, dtype=np.int64)
    y = np.asarray(y_list, dtype=np.float32)
    return X_qty, X_features, prod_idx, y


def create_lag_features(
    daily: pd.DataFrame,
    value_col: str = "quantity_scaled",
    lags: Iterable[int] = (1, 7, 14, 30),
    rolling_windows: Iterable[int] = (7, 14, 30),
) -> pd.DataFrame:
    """Reusable flat lag/rolling features for RF/GP-style notebooks."""
    out = daily.sort_values(["item_id", "date"]).copy()
    grouped = out.groupby("item_id")[value_col]
    for lag in lags:
        out[f"{value_col}_lag_{lag}"] = grouped.shift(lag)
    for window in rolling_windows:
        shifted = grouped.shift(1)
        out[f"{value_col}_rolling_mean_{window}"] = (
            shifted.groupby(out["item_id"]).transform(lambda s: s.rolling(window).mean())
        )
        out[f"{value_col}_rolling_std_{window}"] = (
            shifted.groupby(out["item_id"]).transform(lambda s: s.rolling(window).std())
        )
    return out


def slice_by_split(
    daily: pd.DataFrame,
    splits: dict,
    lookback_days: int = 0,
) -> dict[str, pd.DataFrame]:
    """Return train/val/test slices using the canonical split boundaries.

    `lookback_days` includes pre-split context for sequence models while keeping
    target dates aligned to the same val/test windows.
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    train_end = pd.Timestamp(splits["train_end"])
    val_start = pd.Timestamp(splits["val_start"])
    val_end = pd.Timestamp(splits["val_end"])
    test_start = pd.Timestamp(splits["test_start"])
    test_end = pd.Timestamp(splits["test_end"])

    delta = pd.Timedelta(days=lookback_days)
    return {
        "train": df[(df["date"] >= pd.Timestamp(splits["train_start"])) & (df["date"] <= train_end)].copy(),
        "val": df[(df["date"] >= val_start - delta) & (df["date"] <= val_end)].copy(),
        "test": df[(df["date"] >= test_start - delta) & (df["date"] <= test_end)].copy(),
    }


def load_processed_data(outdir: str = "data/processed") -> pd.DataFrame:
    """Load the canonical processed daily data."""
    return pd.read_csv(Path(outdir) / "daily.csv", parse_dates=["date"])


def load_artifacts(outdir: str = "data/processed") -> dict:
    """Load preprocessing artifacts written by `prepare_data`."""
    out = Path(outdir)
    artifacts: dict = {}
    artifacts["daily"] = load_processed_data(str(out))
    for name in ["splits", "selected_products", "outlier_caps", "feature_columns", "metadata"]:
        path = out / f"{name}.json"
        if path.exists():
            with open(path) as f:
                artifacts[name] = json.load(f)
    for name in ["scalers", "feature_scalers"]:
        path = out / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                artifacts[name] = pickle.load(f)
    return artifacts


def prepare_data(
    csv_path: str = "sales.csv",
    n_products: int | None = DEFAULT_N_PRODUCTS,
    full_history_days: int | None = None,
    min_history_days: int | None = None,
    min_active_days: int | None = DEFAULT_MIN_ACTIVE_DAYS,
    ranking_col: str = DEFAULT_PRODUCT_RANK_COL,
    val_days: int = 60,
    test_frac: float = 0.20,
    outlier_pct: float = 99.5,
    fill_missing_dates: bool = True,
    scale_columns: Iterable[str] = DEFAULT_SCALE_COLUMNS,
    outdir: str = "data/processed",
) -> dict:
    """End-to-end shared preprocessing.

    Writes:
        daily.csv, scalers.pkl, feature_scalers.pkl, splits.json,
        selected_products.json, outlier_caps.json, feature_columns.json,
        metadata.json

    Returns the split-boundaries dict for backward compatibility.
    """
    raw = load_raw_sales(csv_path)
    daily = aggregate_daily(raw)
    selected_products = select_top_products(
        daily,
        n=n_products,
        full_history_days=full_history_days,
        min_history_days=min_history_days,
        min_active_days=min_active_days,
        volume_col=ranking_col,
    )
    if not selected_products:
        raise ValueError("No products matched the requested history/product filters.")

    daily = daily[daily["item_id"].isin(selected_products)].copy()
    if fill_missing_dates:
        daily = complete_product_date_grid(daily, product_ids=selected_products)

    splits = split_time_based(daily, val_days=val_days, test_frac=test_frac)

    cap_columns = [col for col in ["quantity", "positive_quantity", "returns_quantity"] if col in daily.columns]
    daily, outlier_caps = cap_outliers(
        daily,
        percentile=outlier_pct,
        train_end_date=splits["train_end"],
        columns=cap_columns,
        return_caps=True,
    )
    daily = add_calendar_features(daily)
    daily, feature_scalers = scale_feature_columns(daily, splits["train_end"], columns=scale_columns)
    quantity_scalers = feature_scalers.get("quantity", {})

    feature_columns = {
        "calendar": CALENDAR_COLS,
        "scaled": [f"{col}_scaled" for col in scale_columns if f"{col}_scaled" in daily.columns],
        "shared_default": [col for col in DEFAULT_SHARED_FEATURE_COLS if col in daily.columns],
        "target": "quantity_scaled",
    }
    metadata = {
        "csv_path": csv_path,
        "n_raw_rows": int(len(raw)),
        "n_products": None if n_products is None else int(n_products),
        "n_selected_products": int(len(selected_products)),
        "full_history_days": full_history_days,
        "min_history_days": min_history_days,
        "min_active_days": min_active_days,
        "ranking_col": ranking_col,
        "benchmark_rule": (
            f"top {n_products} products by {ranking_col} with "
            f"at least {min_active_days} active sales days"
        ),
        "fill_missing_dates": bool(fill_missing_dates),
        "outlier_percentile": float(outlier_pct),
        "scale_columns": list(scale_columns),
        "date_min": daily["date"].min().strftime("%Y-%m-%d"),
        "date_max": daily["date"].max().strftime("%Y-%m-%d"),
        "n_daily_rows": int(len(daily)),
    }

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "daily.csv", index=False)
    with open(out / "scalers.pkl", "wb") as f:
        pickle.dump(quantity_scalers, f)
    with open(out / "feature_scalers.pkl", "wb") as f:
        pickle.dump(feature_scalers, f)
    json_payloads = {
        "splits.json": splits,
        "selected_products.json": selected_products,
        "outlier_caps.json": outlier_caps,
        "feature_columns.json": feature_columns,
        "metadata.json": metadata,
    }
    for filename, payload in json_payloads.items():
        with open(out / filename, "w") as f:
            json.dump(payload, f, indent=2)

    return splits
