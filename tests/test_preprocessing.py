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
