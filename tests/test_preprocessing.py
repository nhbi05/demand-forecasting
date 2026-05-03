"""Tests for src/preprocessing.py."""

import io
import pandas as pd
import pytest

from src import preprocessing


def _make_csv(content: str) -> io.StringIO:
    return io.StringIO(content)


def test_load_raw_sales_drops_unnamed_index_and_parses_dates(tmp_path):
    csv = tmp_path / "sales.csv"
    csv.write_text(
        ",date,item_id,quantity,price_base,sum_total,store_id\n"
        "0,2023-08-04,abc,1.0,5.0,5.0,1\n"
        "1,2023-08-05,abc,2.0,5.0,10.0,1\n"
    )

    df = preprocessing.load_raw_sales(str(csv))

    assert "Unnamed: 0" not in df.columns
    assert list(df.columns) == ["date", "item_id", "quantity", "price_base", "sum_total", "store_id"]
    assert df["date"].dtype.kind == "M"  # datetime64
    assert len(df) == 2


def test_aggregate_daily_sums_quantities_per_product_per_day():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01"]),
        "item_id": ["a", "a", "a", "b"],
        "quantity": [3.0, 2.0, 5.0, 1.0],
        "store_id": [1, 2, 1, 1],
    })

    daily = preprocessing.aggregate_daily(raw)

    # 'a' on Jan 1 across stores → 5; 'a' on Jan 2 → 5; 'b' on Jan 1 → 1
    assert len(daily) == 3
    a_jan1 = daily[(daily["item_id"] == "a") & (daily["date"] == pd.Timestamp("2024-01-01"))]
    assert a_jan1["quantity"].iloc[0] == 5.0


def test_aggregate_daily_keeps_negative_returns_in_net():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "item_id": ["a", "a"],
        "quantity": [8.0, -5.0],  # 8 sold, 5 returned
        "store_id": [1, 1],
    })

    daily = preprocessing.aggregate_daily(raw)

    assert daily["quantity"].iloc[0] == 3.0  # net demand


def test_select_top_products_filters_to_full_history_then_top_n():
    # 4 days of data total
    days = pd.date_range("2024-01-01", "2024-01-04")
    rows = []
    # 'a' has all 4 days, totals 100
    for d in days:
        rows.append({"item_id": "a", "date": d, "quantity": 25})
    # 'b' has all 4 days, totals 200 → highest, but we filter top-1
    for d in days:
        rows.append({"item_id": "b", "date": d, "quantity": 50})
    # 'c' has only 2 days — should be excluded for sparse history
    for d in days[:2]:
        rows.append({"item_id": "c", "date": d, "quantity": 999})  # would be top by volume

    daily = pd.DataFrame(rows)

    chosen = preprocessing.select_top_products(daily, n=1, full_history_days=4)

    assert chosen == ["b"]


def test_select_top_products_returns_n_when_more_eligible_exist():
    days = pd.date_range("2024-01-01", "2024-01-04")
    rows = []
    for item, qty in [("a", 10), ("b", 50), ("c", 30)]:
        for d in days:
            rows.append({"item_id": item, "date": d, "quantity": qty})

    daily = pd.DataFrame(rows)

    chosen = preprocessing.select_top_products(daily, n=2, full_history_days=4)

    assert chosen == ["b", "c"]  # ordered by total quantity desc


def test_cap_outliers_caps_at_per_product_percentile():
    # Product 'a' has values 1..100; 99.5th percentile ≈ 99.505
    days = pd.date_range("2024-01-01", periods=100)
    rows = [{"item_id": "a", "date": d, "quantity": float(i + 1)}
            for i, d in enumerate(days)]
    rows.append({"item_id": "a", "date": days[-1] + pd.Timedelta(days=1),
                 "quantity": 99999.0})  # huge outlier

    daily = pd.DataFrame(rows)
    capped = preprocessing.cap_outliers(daily, percentile=99.5)

    # The 99999 value must be reduced to the 99.5th percentile of 'a'
    expected_cap = daily[daily["item_id"] == "a"]["quantity"].quantile(0.995)
    assert capped["quantity"].max() == pytest.approx(expected_cap)


def test_cap_outliers_does_not_cross_products():
    # Product 'a' is small-scale, 'b' is huge-scale; capping 'a' shouldn't
    # be influenced by 'b'.
    rows = (
        [{"item_id": "a", "date": pd.Timestamp(f"2024-01-{d:02d}"), "quantity": 1.0}
         for d in range(1, 21)]
        + [{"item_id": "b", "date": pd.Timestamp(f"2024-01-{d:02d}"), "quantity": 1000.0}
           for d in range(1, 21)]
    )
    daily = pd.DataFrame(rows)
    capped = preprocessing.cap_outliers(daily, percentile=99.5)

    assert capped[capped["item_id"] == "a"]["quantity"].max() == 1.0
    assert capped[capped["item_id"] == "b"]["quantity"].max() == 1000.0


def test_add_calendar_features_creates_six_columns():
    daily = pd.DataFrame({
        "item_id": ["a", "a"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),  # both Mondays
        "quantity": [1.0, 1.0],
    })

    out = preprocessing.add_calendar_features(daily)

    expected_cols = {"dow_sin", "dow_cos", "dom_sin", "dom_cos", "month_sin", "month_cos"}
    assert expected_cols.issubset(out.columns)


def test_add_calendar_features_same_weekday_yields_same_dow_encoding():
    daily = pd.DataFrame({
        "item_id": ["a", "a"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),  # both Mondays
        "quantity": [1.0, 1.0],
    })

    out = preprocessing.add_calendar_features(daily)

    assert out["dow_sin"].iloc[0] == pytest.approx(out["dow_sin"].iloc[1])
    assert out["dow_cos"].iloc[0] == pytest.approx(out["dow_cos"].iloc[1])


def test_split_time_based_returns_correct_boundaries():
    days = pd.date_range("2022-08-28", "2024-09-26")  # 761 days
    daily = pd.DataFrame({
        "item_id": ["a"] * len(days),
        "date": days,
        "quantity": 1.0,
    })

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
        "quantity": list(range(10)),  # 0..9
    })
    train_end = pd.Timestamp("2024-01-05")  # train rows 0..4 → values 0..4

    scaled, scalers = preprocessing.scale_per_product(daily, train_end)

    # Train range = 0..4; min=0, max=4 → train values scale to 0..1
    train_rows = scaled[scaled["date"] <= train_end]
    assert train_rows["quantity_scaled"].min() == pytest.approx(0.0)
    assert train_rows["quantity_scaled"].max() == pytest.approx(1.0)

    # Post-train values 5..9 will scale > 1 (extrapolation), proving fit was train-only
    post_train = scaled[scaled["date"] > train_end]
    assert post_train["quantity_scaled"].min() > 1.0


def test_scale_per_product_independent_per_product():
    days = pd.date_range("2024-01-01", periods=4)
    daily = pd.DataFrame({
        "item_id": ["a"] * 4 + ["b"] * 4,
        "date": list(days) * 2,
        "quantity": [10, 20, 30, 40] + [100, 200, 300, 400],
    })
    train_end = pd.Timestamp("2024-01-04")  # all rows are train

    scaled, scalers = preprocessing.scale_per_product(daily, train_end)

    a_max = scaled[scaled["item_id"] == "a"]["quantity_scaled"].max()
    b_max = scaled[scaled["item_id"] == "b"]["quantity_scaled"].max()
    assert a_max == pytest.approx(1.0)
    assert b_max == pytest.approx(1.0)
    assert "a" in scalers and "b" in scalers
