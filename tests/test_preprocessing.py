"""Tests for src/preprocessing.py."""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import preprocessing


def _clean_test_dir(name: str) -> Path:
    path = Path(".test_tmp_preprocessing") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_validate_raw_sales_rejects_missing_required_columns():
    raw = pd.DataFrame({"date": ["2024-01-01"], "item_id": ["a"]})

    with pytest.raises(ValueError, match="missing required columns"):
        preprocessing.validate_raw_sales(raw)


def test_load_raw_sales_drops_unnamed_index_parses_dates_and_handles_missing():
    workdir = _clean_test_dir("load_raw_sales")
    csv = workdir / "sales.csv"
    csv.write_text(
        ",date,item_id,quantity,price_base,sum_total,store_id\n"
        "0,2023-08-04,abc,1.0,5.0,5.0,1\n"
        "1,2023-08-05,abc,2.0,,10.0,1\n"
        "2,2023-08-06,abc,,5.0,10.0,1\n"
    )

    df = preprocessing.load_raw_sales(str(csv))

    assert "Unnamed: 0" not in df.columns
    assert df["date"].dtype.kind == "M"
    assert len(df) == 2  # row with missing quantity is unrecoverable
    assert df["price_base"].isna().sum() == 0
    assert "price_base_was_missing" in df.columns
    assert df["price_base_was_missing"].sum() == 1


def test_handle_missing_values_recomputes_missing_sum_total_and_store_sentinel():
    raw = pd.DataFrame({
        "date": ["2024-01-01"],
        "item_id": ["a"],
        "quantity": [3.0],
        "price_base": [2.5],
        "sum_total": [np.nan],
        "store_id": [np.nan],
    })

    out = preprocessing.handle_missing_values(raw)

    assert out["sum_total"].iloc[0] == pytest.approx(7.5)
    assert out["store_id"].iloc[0] == "__missing_store__"
    assert out["sum_total_was_missing"].iloc[0]
    assert out["store_id_was_missing"].iloc[0]


def test_aggregate_daily_sums_quantities_and_preserves_shared_features():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01"]),
        "item_id": ["a", "a", "a", "b"],
        "quantity": [3.0, 2.0, -5.0, 1.0],
        "price_base": [10.0, 20.0, 30.0, 5.0],
        "sum_total": [30.0, 40.0, -150.0, 5.0],
        "store_id": [1, 2, 1, 1],
    })

    daily = preprocessing.aggregate_daily(raw)

    a_jan1 = daily[(daily["item_id"] == "a") & (daily["date"] == pd.Timestamp("2024-01-01"))].iloc[0]
    assert a_jan1["quantity"] == 5.0
    assert a_jan1["positive_quantity"] == 5.0
    assert a_jan1["returns_quantity"] == 0.0
    assert a_jan1["price_base"] == pytest.approx(15.0)
    assert a_jan1["transaction_count"] == 2
    assert a_jan1["store_count"] == 2
    assert a_jan1["had_sales"] == 1

    a_jan2 = daily[(daily["item_id"] == "a") & (daily["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert a_jan2["quantity"] == -5.0
    assert a_jan2["returns_quantity"] == 5.0
    assert a_jan2["had_sales"] == 0


def test_select_top_products_filters_to_active_sales_days_then_top_n():
    days = pd.date_range("2024-01-01", "2024-01-04")
    rows = []
    for d in days:
        rows.append({"item_id": "a", "date": d, "quantity": 25, "positive_quantity": 25})
        rows.append({"item_id": "b", "date": d, "quantity": 50, "positive_quantity": 50})
    for d in days[:2]:
        rows.append({"item_id": "c", "date": d, "quantity": 999, "positive_quantity": 999})

    chosen = preprocessing.select_top_products(
        pd.DataFrame(rows), n=1, min_active_days=4
    )

    assert chosen == ["b"]


def test_select_top_products_can_keep_all_eligible_products_and_rank_by_positive_quantity():
    days = pd.date_range("2024-01-01", periods=3)
    daily = pd.DataFrame(
        {
            "item_id": ["a", "a", "a", "b", "b", "b"],
            "date": list(days) * 2,
            "quantity": [1, 1, 1, 5, 5, 5],
            "positive_quantity": [1, 1, 1, 5, 5, 5],
        }
    )

    chosen = preprocessing.select_top_products(daily, n=None, min_active_days=3)

    assert chosen == ["b", "a"]


def test_select_top_products_keeps_legacy_history_filter_when_requested():
    days = pd.date_range("2024-01-01", periods=4)
    daily = pd.DataFrame({
        "item_id": ["a"] * 4 + ["b"] * 2,
        "date": list(days) + list(days[:2]),
        "quantity": [5, 5, 5, 5, 100, 100],
        "positive_quantity": [5, 5, 5, 5, 100, 100],
    })

    chosen = preprocessing.select_top_products(
        daily,
        n=2,
        min_history_days=4,
        min_active_days=None,
    )

    assert chosen == ["a"]


def test_complete_product_date_grid_fills_missing_rows_as_zero_sales():
    daily = pd.DataFrame({
        "item_id": ["a", "a"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
        "quantity": [4.0, 6.0],
        "positive_quantity": [4.0, 6.0],
        "returns_quantity": [0.0, 0.0],
        "price_base": [10.0, 12.0],
        "price_base_median": [10.0, 12.0],
        "sum_total": [40.0, 72.0],
        "transaction_count": [1, 1],
        "store_count": [1, 1],
    })

    out = preprocessing.complete_product_date_grid(daily, product_ids=["a"])
    missing = out[out["date"] == pd.Timestamp("2024-01-02")].iloc[0]

    assert len(out) == 3
    assert missing["quantity"] == 0.0
    assert missing["transaction_count"] == 0
    assert missing["store_count"] == 0
    assert missing["had_sales"] == 0
    assert missing["price_base"] == pytest.approx(10.0)


def test_cap_outliers_can_fit_caps_on_train_only():
    days = pd.date_range("2024-01-01", periods=6)
    daily = pd.DataFrame({
        "item_id": ["a"] * 6,
        "date": days,
        "quantity": [1.0, 2.0, 3.0, 4.0, 1000.0, 2000.0],
    })

    capped, caps = preprocessing.cap_outliers(
        daily,
        percentile=100,
        train_end_date=pd.Timestamp("2024-01-04"),
        return_caps=True,
    )

    assert caps["quantity"]["a"] == 4.0
    assert capped["quantity"].max() == 4.0


def test_cap_outliers_old_behavior_still_caps_per_product():
    days = pd.date_range("2024-01-01", periods=101)
    daily = pd.DataFrame({
        "item_id": ["a"] * 101,
        "date": days,
        "quantity": list(range(1, 101)) + [99999.0],
    })

    capped = preprocessing.cap_outliers(daily, percentile=99.5)

    expected = daily["quantity"].quantile(0.995)
    assert capped["quantity"].max() == pytest.approx(expected)


def test_add_calendar_features_creates_six_columns_and_cycles_weekday():
    daily = pd.DataFrame({
        "item_id": ["a", "a"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
        "quantity": [1.0, 1.0],
    })

    out = preprocessing.add_calendar_features(daily)

    assert set(preprocessing.CALENDAR_COLS).issubset(out.columns)
    assert out["dow_sin"].iloc[0] == pytest.approx(out["dow_sin"].iloc[1])
    assert out["dow_cos"].iloc[0] == pytest.approx(out["dow_cos"].iloc[1])


def test_split_time_based_returns_correct_boundaries():
    days = pd.date_range("2022-08-28", "2024-09-26")
    daily = pd.DataFrame({"item_id": ["a"] * len(days), "date": days, "quantity": 1.0})

    boundaries = preprocessing.split_time_based(daily, val_days=60, test_frac=0.20)

    test_size = int(len(days) * 0.20)
    train_plus_val = len(days) - test_size
    train_size = train_plus_val - 60
    assert boundaries["train_start"] == "2022-08-28"
    assert boundaries["test_end"] == "2024-09-26"
    assert pd.Timestamp(boundaries["train_end"]) == days[train_size - 1]
    assert pd.Timestamp(boundaries["val_start"]) == days[train_size]
    assert pd.Timestamp(boundaries["val_end"]) == days[train_size + 60 - 1]
    assert pd.Timestamp(boundaries["test_start"]) == days[train_size + 60]


def test_split_time_based_rejects_too_few_days():
    daily = pd.DataFrame({
        "item_id": ["a"] * 30,
        "date": pd.date_range("2024-01-01", periods=30),
        "quantity": 1.0,
    })

    with pytest.raises(ValueError):
        preprocessing.split_time_based(daily, val_days=60, test_frac=0.20)


def test_scale_per_product_fits_on_train_only():
    days = pd.date_range("2024-01-01", periods=10)
    daily = pd.DataFrame({
        "item_id": ["a"] * 10,
        "date": days,
        "quantity": list(range(10)),
    })

    scaled, scalers = preprocessing.scale_per_product(daily, pd.Timestamp("2024-01-05"))

    train_rows = scaled[scaled["date"] <= pd.Timestamp("2024-01-05")]
    assert train_rows["quantity_scaled"].min() == pytest.approx(0.0)
    assert train_rows["quantity_scaled"].max() == pytest.approx(1.0)
    assert scaled[scaled["date"] > pd.Timestamp("2024-01-05")]["quantity_scaled"].min() > 1.0
    assert "a" in scalers


def test_scale_feature_columns_scales_multiple_columns_independently():
    days = pd.date_range("2024-01-01", periods=4)
    daily = pd.DataFrame({
        "item_id": ["a"] * 4,
        "date": days,
        "quantity": [1, 2, 3, 4],
        "price_base": [10, 20, 30, 40],
    })

    scaled, scalers = preprocessing.scale_feature_columns(
        daily,
        train_end_date=pd.Timestamp("2024-01-04"),
        columns=["quantity", "price_base"],
    )

    assert {"quantity_scaled", "price_base_scaled"}.issubset(scaled.columns)
    assert set(scalers) == {"quantity", "price_base"}
    assert scaled["quantity_scaled"].max() == pytest.approx(1.0)
    assert scaled["price_base_scaled"].max() == pytest.approx(1.0)


def test_create_sequences_yields_default_shapes():
    days = pd.date_range("2024-01-01", periods=20)
    daily = pd.DataFrame({
        "item_id": ["a"] * 20,
        "date": days,
        "quantity_scaled": np.linspace(0, 1, 20),
        "dow_sin": np.zeros(20), "dow_cos": np.ones(20),
        "dom_sin": np.zeros(20), "dom_cos": np.ones(20),
        "month_sin": np.zeros(20), "month_cos": np.ones(20),
    })

    X_qty, X_cal, prod_idx, y = preprocessing.create_sequences(
        daily, {"a": 0}, lookback=5, horizon=3
    )

    assert X_qty.shape == (13, 5, 1)
    assert X_cal.shape == (13, 5, 6)
    assert prod_idx.shape == (13,)
    assert y.shape == (13, 3)


def test_create_sequences_accepts_richer_feature_columns():
    days = pd.date_range("2024-01-01", periods=10)
    daily = pd.DataFrame({
        "item_id": ["a"] * 10,
        "date": days,
        "quantity_scaled": np.arange(10, dtype=float),
        "price_base_scaled": np.linspace(0, 1, 10),
        "had_sales": 1,
    })

    X_qty, X_feat, _, y = preprocessing.create_sequences(
        daily,
        {"a": 0},
        lookback=3,
        horizon=2,
        feature_cols=["price_base_scaled", "had_sales"],
    )

    assert list(X_qty[0, :, 0]) == [0.0, 1.0, 2.0]
    assert list(y[0]) == [3.0, 4.0]
    assert X_feat.shape == (6, 3, 2)


def test_create_lag_features_uses_past_values_only():
    days = pd.date_range("2024-01-01", periods=5)
    daily = pd.DataFrame({
        "item_id": ["a"] * 5,
        "date": days,
        "quantity_scaled": [10, 20, 30, 40, 50],
    })

    out = preprocessing.create_lag_features(
        daily,
        value_col="quantity_scaled",
        lags=(1, 2),
        rolling_windows=(2,),
    )

    assert pd.isna(out["quantity_scaled_lag_1"].iloc[0])
    assert out["quantity_scaled_lag_1"].iloc[2] == 20
    assert out["quantity_scaled_lag_2"].iloc[3] == 20
    assert out["quantity_scaled_rolling_mean_2"].iloc[3] == pytest.approx((20 + 30) / 2)


def test_slice_by_split_returns_aligned_periods_with_lookback():
    days = pd.date_range("2024-01-01", periods=10)
    daily = pd.DataFrame({"item_id": ["a"] * 10, "date": days, "quantity": 1.0})
    splits = {
        "train_start": "2024-01-01",
        "train_end": "2024-01-05",
        "val_start": "2024-01-06",
        "val_end": "2024-01-07",
        "test_start": "2024-01-08",
        "test_end": "2024-01-10",
    }

    slices = preprocessing.slice_by_split(daily, splits, lookback_days=2)

    assert slices["train"]["date"].min() == pd.Timestamp("2024-01-01")
    assert slices["val"]["date"].min() == pd.Timestamp("2024-01-04")
    assert slices["test"]["date"].min() == pd.Timestamp("2024-01-06")


def test_prepare_data_writes_shared_artifacts_and_fills_missing_product_dates():
    workdir = _clean_test_dir("prepare_data")
    days = pd.date_range("2024-01-01", periods=20)
    rows = []
    for d in days:
        rows.append({"date": d, "item_id": "a", "quantity": 10.0,
                     "price_base": 1.0, "sum_total": 10.0, "store_id": 1})
    # Product b misses one date but is still eligible with min_history_days=19.
    for d in days.delete(5):
        rows.append({"date": d, "item_id": "b", "quantity": 5.0,
                     "price_base": 2.0, "sum_total": 10.0, "store_id": 1})
    csv = workdir / "sales.csv"
    pd.DataFrame(rows).to_csv(csv, index_label="")

    outdir = workdir / "processed"
    splits = preprocessing.prepare_data(
        csv_path=str(csv),
        n_products=2,
        full_history_days=None,
        min_active_days=19,
        val_days=4,
        test_frac=0.20,
        outdir=str(outdir),
    )

    for name in [
        "daily.csv",
        "scalers.pkl",
        "feature_scalers.pkl",
        "splits.json",
        "selected_products.json",
        "outlier_caps.json",
        "feature_columns.json",
        "metadata.json",
    ]:
        assert (outdir / name).exists()

    with open(outdir / "splits.json") as f:
        assert json.load(f) == splits
    with open(outdir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    assert set(scalers) == {"a", "b"}

    daily = pd.read_csv(outdir / "daily.csv", parse_dates=["date"])
    assert daily.groupby("item_id")["date"].nunique().to_dict() == {"a": 20, "b": 20}
    b_missing = daily[(daily["item_id"] == "b") & (daily["date"] == days[5])].iloc[0]
    assert b_missing["quantity"] == 0.0
    assert b_missing["had_sales"] == 0
    expected_cols = {
        "quantity_scaled",
        "price_base_scaled",
        "transaction_count_scaled",
        "dow_sin",
        "dow_cos",
    }
    assert expected_cols.issubset(daily.columns)

    artifacts = preprocessing.load_artifacts(str(outdir))
    assert set(artifacts["selected_products"]) == {"a", "b"}
    assert "daily" in artifacts and "metadata" in artifacts
    assert artifacts["metadata"]["n_selected_products"] == 2
    assert artifacts["metadata"]["min_active_days"] == 19
    assert artifacts["metadata"]["ranking_col"] == "positive_quantity"


def test_prepare_data_defaults_encode_team_benchmark():
    assert preprocessing.DEFAULT_N_PRODUCTS == 100
    assert preprocessing.DEFAULT_MIN_ACTIVE_DAYS == 730
    assert preprocessing.DEFAULT_PRODUCT_RANK_COL == "positive_quantity"
