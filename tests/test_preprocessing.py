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
