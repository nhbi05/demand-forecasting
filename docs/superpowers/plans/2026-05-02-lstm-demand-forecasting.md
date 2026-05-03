# LSTM Demand Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch LSTM that forecasts the next 30 days of daily demand for the top-50 products from `sales.csv`, plus a framework-agnostic preprocessing pipeline the team's other models reuse.

**Architecture:** Single global LSTM (`hidden_size=64`, 2 layers) with a learned 8-dim product embedding. Each example is a `(60-day input window, 30-day target window)` for one product. Direct multi-output forecast via a `Linear(64, 30)` head. Per-product MinMax scaling, time-based 80/20 train/test split with a 60-day val slice carved out of train.

**Tech Stack:** Python 3.10+, PyTorch (model + training), pandas + numpy + scikit-learn (preprocessing), pytest (tests), matplotlib + seaborn (plots), Jupyter (notebooks).

**Spec:** [docs/superpowers/specs/2026-05-02-lstm-demand-forecasting-design.md](../specs/2026-05-02-lstm-demand-forecasting-design.md)

---

## File structure (locked before tasks)

| File | Status | Responsibility |
|---|---|---|
| `requirements.txt` | modify | drop `tensorflow`, add `pytest`, `pyarrow` not needed |
| `src/__init__.py` | modify | remove auto-import of `ensemble` (team's module); keep `preprocessing` only |
| `src/models/__init__.py` | modify | empty / pass-through — don't auto-import other-team modules with TF deps |
| `src/preprocessing.py` | replace | framework-agnostic data pipeline; pure pandas/numpy; importable by team |
| `src/models/lstm_model.py` | replace | PyTorch `LSTMForecaster`, `LSTMDataset`, train/predict/evaluate utilities |
| `tests/__init__.py` | create | empty marker |
| `tests/test_preprocessing.py` | create | unit tests for every preprocessing function |
| `tests/test_lstm_model.py` | create | smoke + shape tests for the PyTorch model |
| `notebooks/01_EDA.ipynb` | replace | raw-data exploration (shared with team) |
| `notebooks/02_LSTM.ipynb` | replace | the 6 ML lifecycle stages as sections |
| `data/processed/` | created at runtime | `daily.csv`, `scalers.pkl`, `splits.json` |
| `data/predictions/` | created at runtime | `lstm_test_predictions.csv` |
| `models/` | created at runtime | `lstm_final.pt`, `lstm_config.json` |

**Out of scope (team's territory; do not modify):** `src/ensemble.py`, `src/models/rnn_model.py`, `src/models/random_forest_model.py`, `src/models/gaussian_model.py`, `notebooks/03_RNN.ipynb`, `notebooks/04_Random_Forest.ipynb`, `notebooks/05_Gaussian.ipynb`, `notebooks/06_Ensemble.ipynb`.

---

## Task 1: Project setup and dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `src/__init__.py`
- Modify: `src/models/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1.1 — Modify `requirements.txt`**

Replace the entire file with:

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
torch==2.0.1
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
ipython==8.14.0
scipy==1.11.1
pytest==7.4.0
nbconvert==7.7.3
```

(Removed: `tensorflow==2.13.0`, `torchvision==0.15.2`. Added: `pytest`, `nbconvert`.)

- [ ] **Step 1.2 — Modify `src/__init__.py`**

Replace the file with:

```python
"""
Demand Forecasting — LSTM sub-project (user's scope).

The `ensemble` module belongs to the team and is intentionally not
imported here to avoid pulling in their dependencies during preprocessing.
"""

from . import preprocessing

__version__ = "0.1.0"
```

- [ ] **Step 1.3 — Modify `src/models/__init__.py`**

Replace the file with:

```python
"""Model implementations.

Empty by design: every other model module (rnn_model, random_forest_model,
gaussian_model) currently still imports TensorFlow at module load time, which
breaks if TF isn't installed. Each consumer should import the specific module
they need, e.g.:

    from src.models.lstm_model import LSTMForecaster
"""
```

- [ ] **Step 1.4 — Create `tests/__init__.py`** (empty file marking the directory as a package)

Create the file with no content (zero bytes).

- [ ] **Step 1.5 — Install dependencies and verify**

Run:
```
pip install -r requirements.txt
python -c "import torch, pandas, numpy, sklearn, pytest; print('ok')"
```
Expected output: `ok`

- [ ] **Step 1.6 — Commit**

```bash
git add requirements.txt src/__init__.py src/models/__init__.py tests/__init__.py
git commit -m "chore: switch to PyTorch-only deps and reset model __init__"
```

---

## Task 2: `preprocessing.load_raw_sales()` — load and tidy `sales.csv`

**Files:**
- Create: `src/preprocessing.py` (initial version with this function only)
- Test: `tests/test_preprocessing.py` (initial version with this test only)

- [ ] **Step 2.1 — Write the failing test**

Create `tests/test_preprocessing.py`:

```python
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
```

- [ ] **Step 2.2 — Run test to verify it fails**

Run: `pytest tests/test_preprocessing.py::test_load_raw_sales_drops_unnamed_index_and_parses_dates -v`

Expected: FAIL with `ModuleNotFoundError` or `AttributeError: module 'src.preprocessing' has no attribute 'load_raw_sales'`.

- [ ] **Step 2.3 — Write the minimal implementation**

Create `src/preprocessing.py`:

```python
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
```

- [ ] **Step 2.4 — Run test, verify pass**

Run: `pytest tests/test_preprocessing.py::test_load_raw_sales_drops_unnamed_index_and_parses_dates -v`

Expected: PASS.

- [ ] **Step 2.5 — Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): load raw sales CSV"
```

---

## Task 3: `preprocessing.aggregate_daily()` — sum quantities to daily per product

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 3.1 — Write the failing test**

Append to `tests/test_preprocessing.py`:

```python
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
```

- [ ] **Step 3.2 — Verify both tests fail**

Run: `pytest tests/test_preprocessing.py -v -k aggregate`
Expected: 2 FAIL with `AttributeError: ... 'aggregate_daily'`.

- [ ] **Step 3.3 — Implement**

Append to `src/preprocessing.py`:

```python
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Sum `quantity` per (item_id, date) across all stores.

    Negative quantities (returns) are kept and net into the daily total —
    a return of 5 on a day that also had 8 sales becomes net demand of 3.
    """
    return df.groupby(["item_id", "date"], as_index=False)["quantity"].sum()
```

- [ ] **Step 3.4 — Verify pass**

Run: `pytest tests/test_preprocessing.py -v -k aggregate`
Expected: 2 PASS.

- [ ] **Step 3.5 — Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): aggregate transactions to daily quantity per product"
```

---

## Task 4: `preprocessing.select_top_products()` — pick top-N with full history

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 4.1 — Write the failing test**

Append to `tests/test_preprocessing.py`:

```python
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
```

- [ ] **Step 4.2 — Verify failing**

Run: `pytest tests/test_preprocessing.py -v -k select_top`
Expected: 2 FAIL.

- [ ] **Step 4.3 — Implement**

Append to `src/preprocessing.py`:

```python
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
```

- [ ] **Step 4.4 — Verify pass**

Run: `pytest tests/test_preprocessing.py -v -k select_top`
Expected: 2 PASS.

- [ ] **Step 4.5 — Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): select top-N products with full history"
```

---

## Task 5: `preprocessing.cap_outliers()` — winsorize per product at 99.5th pct

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 5.1 — Write the failing test**

Append to `tests/test_preprocessing.py`:

```python
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
```

- [ ] **Step 5.2 — Verify failing**

Run: `pytest tests/test_preprocessing.py -v -k cap_outliers`
Expected: 2 FAIL.

- [ ] **Step 5.3 — Implement**

Append to `src/preprocessing.py`:

```python
import numpy as np  # add at top of file if not already present


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
```

If `import numpy as np` isn't already at the top of `src/preprocessing.py`, add it there now.

- [ ] **Step 5.4 — Verify pass**

Run: `pytest tests/test_preprocessing.py -v -k cap_outliers`
Expected: 2 PASS.

- [ ] **Step 5.5 — Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): cap outliers at per-product 99.5th percentile"
```

---

## Task 6: `preprocessing.add_calendar_features()` — sin/cos cyclical features

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 6.1 — Write the failing test**

Append to `tests/test_preprocessing.py`:

```python
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
```

- [ ] **Step 6.2 — Verify failing**

Run: `pytest tests/test_preprocessing.py -v -k calendar`
Expected: 2 FAIL.

- [ ] **Step 6.3 — Implement**

Append to `src/preprocessing.py`:

```python
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
```

- [ ] **Step 6.4 — Verify pass**

Run: `pytest tests/test_preprocessing.py -v -k calendar`
Expected: 2 PASS.

- [ ] **Step 6.5 — Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): add cyclical calendar features"
```

---

## Task 7: `preprocessing.split_time_based()` — 80/20 split with internal val

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 7.1 — Write the failing test**

Append to `tests/test_preprocessing.py`:

```python
def test_split_time_based_returns_correct_boundaries():
    # 100 days; expect test = 20%, val = 60 days, train = remainder
    # but 60 val on 100 days leaves only 20 for train, so we use a smaller val.
    # Use the actual project size: 761 days.
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
```

- [ ] **Step 7.2 — Verify failing**

Run: `pytest tests/test_preprocessing.py -v -k split_time`
Expected: 2 FAIL.

- [ ] **Step 7.3 — Implement**

Append to `src/preprocessing.py`:

```python
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
```

- [ ] **Step 7.4 — Verify pass**

Run: `pytest tests/test_preprocessing.py -v -k split_time`
Expected: 2 PASS.

- [ ] **Step 7.5 — Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): time-based 80/20 split with internal val"
```

---

## Task 8: `preprocessing.scale_per_product()` — fit on train, transform all

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 8.1 — Write the failing test**

Append to `tests/test_preprocessing.py`:

```python
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
```

- [ ] **Step 8.2 — Verify failing**

Run: `pytest tests/test_preprocessing.py -v -k scale_per_product`
Expected: 2 FAIL.

- [ ] **Step 8.3 — Implement**

Append to `src/preprocessing.py`:

```python
from sklearn.preprocessing import MinMaxScaler


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
```

- [ ] **Step 8.4 — Verify pass**

Run: `pytest tests/test_preprocessing.py -v -k scale_per_product`
Expected: 2 PASS.

- [ ] **Step 8.5 — Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): fit per-product MinMax scalers on train only"
```

---

## Task 9: `preprocessing.create_sequences()` — sliding-window (input, target) windows

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 9.1 — Write the failing test**

Append to `tests/test_preprocessing.py`:

```python
import numpy as np


def test_create_sequences_yields_correct_shapes():
    # Single product, 20 days. lookback=5, horizon=3 → windows = 20 - 5 - 3 + 1 = 13
    days = pd.date_range("2024-01-01", periods=20)
    daily = pd.DataFrame({
        "item_id": ["a"] * 20,
        "date": days,
        "quantity_scaled": np.linspace(0, 1, 20),
        "dow_sin": np.zeros(20), "dow_cos": np.ones(20),
        "dom_sin": np.zeros(20), "dom_cos": np.ones(20),
        "month_sin": np.zeros(20), "month_cos": np.ones(20),
    })
    product_to_idx = {"a": 0}

    X_qty, X_cal, prod_idx, y = preprocessing.create_sequences(
        daily, product_to_idx, lookback=5, horizon=3
    )

    assert X_qty.shape == (13, 5, 1)
    assert X_cal.shape == (13, 5, 6)
    assert prod_idx.shape == (13,)
    assert y.shape == (13, 3)


def test_create_sequences_input_and_target_are_consecutive():
    # Use distinct values so we can check ordering.
    days = pd.date_range("2024-01-01", periods=10)
    daily = pd.DataFrame({
        "item_id": ["a"] * 10,
        "date": days,
        "quantity_scaled": np.arange(10, dtype=float),
        "dow_sin": np.zeros(10), "dow_cos": np.zeros(10),
        "dom_sin": np.zeros(10), "dom_cos": np.zeros(10),
        "month_sin": np.zeros(10), "month_cos": np.zeros(10),
    })
    X_qty, _, _, y = preprocessing.create_sequences(
        daily, {"a": 0}, lookback=3, horizon=2
    )

    # First window: input = [0,1,2], target = [3,4]
    assert list(X_qty[0, :, 0]) == [0.0, 1.0, 2.0]
    assert list(y[0]) == [3.0, 4.0]


def test_create_sequences_handles_multiple_products_independently():
    # Two products, no cross-bleeding.
    days = pd.date_range("2024-01-01", periods=8)
    daily = pd.concat([
        pd.DataFrame({
            "item_id": ["a"] * 8, "date": days,
            "quantity_scaled": np.arange(8, dtype=float) + 100,
            "dow_sin": 0, "dow_cos": 0, "dom_sin": 0, "dom_cos": 0,
            "month_sin": 0, "month_cos": 0,
        }),
        pd.DataFrame({
            "item_id": ["b"] * 8, "date": days,
            "quantity_scaled": np.arange(8, dtype=float) + 200,
            "dow_sin": 0, "dow_cos": 0, "dom_sin": 0, "dom_cos": 0,
            "month_sin": 0, "month_cos": 0,
        }),
    ]).reset_index(drop=True)

    X_qty, _, prod_idx, y = preprocessing.create_sequences(
        daily, {"a": 0, "b": 1}, lookback=3, horizon=2
    )

    # 8 days, lookback 3, horizon 2 → 4 windows per product → 8 total
    assert X_qty.shape[0] == 8
    a_windows = X_qty[prod_idx == 0]
    b_windows = X_qty[prod_idx == 1]
    assert a_windows[:, :, 0].min() >= 100
    assert b_windows[:, :, 0].min() >= 200
```

- [ ] **Step 9.2 — Verify failing**

Run: `pytest tests/test_preprocessing.py -v -k create_sequences`
Expected: 3 FAIL.

- [ ] **Step 9.3 — Implement**

Append to `src/preprocessing.py`:

```python
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
```

- [ ] **Step 9.4 — Verify pass**

Run: `pytest tests/test_preprocessing.py -v -k create_sequences`
Expected: 3 PASS.

- [ ] **Step 9.5 — Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): sliding-window sequence generation"
```

---

## Task 10: `preprocessing.prepare_data()` — orchestrator + disk write

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 10.1 — Write the failing test**

Append to `tests/test_preprocessing.py`:

```python
import json
import pickle


def test_prepare_data_writes_three_artifact_files(tmp_path):
    # Build a tiny but full-shaped CSV: 2 products × 100 days each.
    days = pd.date_range("2024-01-01", periods=100)
    rows = []
    for d in days:
        rows.append({"date": d, "item_id": "a", "quantity": 10.0,
                     "price_base": 1.0, "sum_total": 10.0, "store_id": 1})
        rows.append({"date": d, "item_id": "b", "quantity": 5.0,
                     "price_base": 1.0, "sum_total": 5.0, "store_id": 1})
    csv = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(csv, index_label="")

    outdir = tmp_path / "processed"
    splits = preprocessing.prepare_data(
        csv_path=str(csv),
        n_products=2,
        full_history_days=100,
        val_days=10,
        test_frac=0.20,
        outdir=str(outdir),
    )

    assert (outdir / "daily.csv").exists()
    assert (outdir / "scalers.pkl").exists()
    assert (outdir / "splits.json").exists()

    # splits.json is parseable and matches the returned dict
    with open(outdir / "splits.json") as f:
        on_disk = json.load(f)
    assert on_disk == splits

    # scalers.pkl loads and contains both items
    with open(outdir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    assert set(scalers) == {"a", "b"}

    # daily.csv has the expected columns
    daily = pd.read_csv(outdir / "daily.csv", parse_dates=["date"])
    expected_cols = {"date", "item_id", "quantity", "quantity_scaled",
                     "dow_sin", "dow_cos", "dom_sin", "dom_cos",
                     "month_sin", "month_cos"}
    assert expected_cols.issubset(daily.columns)
```

- [ ] **Step 10.2 — Verify failing**

Run: `pytest tests/test_preprocessing.py -v -k prepare_data`
Expected: 1 FAIL.

- [ ] **Step 10.3 — Implement**

Append to `src/preprocessing.py`:

```python
import json
import pickle
from pathlib import Path


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
```

- [ ] **Step 10.4 — Verify pass**

Run: `pytest tests/test_preprocessing.py -v -k prepare_data`
Expected: 1 PASS. Run the full suite to confirm nothing regressed:
`pytest tests/test_preprocessing.py -v`
Expected: all PASS.

- [ ] **Step 10.5 — Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): prepare_data orchestrator writes daily.csv, scalers.pkl, splits.json"
```

> **Checkpoint after Task 10:** the data layer is complete. Confirm:
> ```bash
> pytest tests/test_preprocessing.py -v   # all green
> ```

---

## Task 11: `LSTMDataset` — PyTorch Dataset wrapping the prepared arrays

**Files:**
- Create: `src/models/lstm_model.py` (initial version with this class only)
- Create: `tests/test_lstm_model.py`

- [ ] **Step 11.1 — Write the failing test**

Create `tests/test_lstm_model.py`:

```python
"""Tests for src/models/lstm_model.py."""

import numpy as np
import torch
import pytest

from src.models import lstm_model


def _dummy_arrays(n=4, lookback=5, horizon=3, n_calendar=6):
    rng = np.random.default_rng(0)
    X_qty = rng.standard_normal((n, lookback, 1)).astype(np.float32)
    X_cal = rng.standard_normal((n, lookback, n_calendar)).astype(np.float32)
    prod_idx = rng.integers(0, 2, size=n).astype(np.int64)
    y = rng.standard_normal((n, horizon)).astype(np.float32)
    return X_qty, X_cal, prod_idx, y


def test_lstm_dataset_len_and_indexing():
    X_qty, X_cal, prod_idx, y = _dummy_arrays(n=4)
    ds = lstm_model.LSTMDataset(X_qty, X_cal, prod_idx, y)

    assert len(ds) == 4
    sample = ds[0]
    assert isinstance(sample, tuple) and len(sample) == 4

    past_qty, calendar, p, target = sample
    assert past_qty.shape == (5, 1) and past_qty.dtype == torch.float32
    assert calendar.shape == (5, 6) and calendar.dtype == torch.float32
    assert p.shape == () and p.dtype == torch.int64
    assert target.shape == (3,) and target.dtype == torch.float32
```

- [ ] **Step 11.2 — Verify failing**

Run: `pytest tests/test_lstm_model.py::test_lstm_dataset_len_and_indexing -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError`.

- [ ] **Step 11.3 — Implement**

Create `src/models/lstm_model.py`:

```python
"""PyTorch LSTM forecaster for the demand-forecasting project.

Components:
    LSTMDataset       — wraps the numpy arrays produced by preprocessing.
    LSTMForecaster    — the model: LSTM body + product embedding + linear head.
    train_model       — training loop with early stopping.
    predict           — batch inference.
    inverse_scale     — undo per-product MinMax scaling on predictions.
    compute_metrics   — RMSE / MAE / MAPE.
    naive_baseline / seasonal_naive_baseline — sanity-check baselines.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    """Wraps `(X_qty, X_cal, prod_idx, y)` numpy arrays as a PyTorch Dataset.

    Each item is a 4-tuple of tensors:
        (past_quantity, calendar, product_idx, target)
    matching the model's forward-pass argument names.
    """

    def __init__(self, X_qty: np.ndarray, X_cal: np.ndarray,
                 prod_idx: np.ndarray, y: np.ndarray):
        self.X_qty = torch.from_numpy(X_qty).float()
        self.X_cal = torch.from_numpy(X_cal).float()
        self.prod_idx = torch.from_numpy(prod_idx).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X_qty.shape[0]

    def __getitem__(self, i: int):
        return self.X_qty[i], self.X_cal[i], self.prod_idx[i], self.y[i]
```

- [ ] **Step 11.4 — Verify pass**

Run: `pytest tests/test_lstm_model.py -v`
Expected: 1 PASS.

- [ ] **Step 11.5 — Commit**

```bash
git add src/models/lstm_model.py tests/test_lstm_model.py
git commit -m "feat(lstm): LSTMDataset wrapping prepared arrays"
```

---

## Task 12: `LSTMForecaster` — the PyTorch model

**Files:**
- Modify: `src/models/lstm_model.py`
- Modify: `tests/test_lstm_model.py`

- [ ] **Step 12.1 — Write the failing test**

Append to `tests/test_lstm_model.py`:

```python
def test_forecaster_forward_output_shape():
    # batch=4, lookback=5, horizon=3, 2 products, 6 calendar dims
    model = lstm_model.LSTMForecaster(
        n_products=2, embed_dim=4, lookback=5, horizon=3,
        hidden_size=8, num_layers=2, dropout=0.0, n_calendar=6,
    )
    past_qty = torch.zeros(4, 5, 1)
    calendar = torch.zeros(4, 5, 6)
    prod_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    out = model(past_qty, calendar, prod_idx)

    assert out.shape == (4, 3)


def test_forecaster_embedding_actually_distinguishes_products():
    """Different product indices should produce different outputs (with non-trivial
    embeddings and otherwise-identical inputs)."""
    torch.manual_seed(0)
    model = lstm_model.LSTMForecaster(
        n_products=2, embed_dim=4, lookback=5, horizon=3,
        hidden_size=8, num_layers=1, dropout=0.0, n_calendar=6,
    )
    past_qty = torch.zeros(2, 5, 1)
    calendar = torch.zeros(2, 5, 6)
    out_a = model(past_qty, calendar, torch.tensor([0, 0], dtype=torch.long))
    out_b = model(past_qty, calendar, torch.tensor([1, 1], dtype=torch.long))

    assert not torch.allclose(out_a, out_b), "Embedding must influence output"
```

- [ ] **Step 12.2 — Verify failing**

Run: `pytest tests/test_lstm_model.py -v -k forecaster`
Expected: 2 FAIL with `AttributeError`.

- [ ] **Step 12.3 — Implement**

Append to `src/models/lstm_model.py`:

```python
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Single global LSTM with a learned product embedding.

    Inputs (per batch):
        past_qty: (B, L, 1)         — scaled daily quantity for the past L days
        calendar: (B, L, n_cal)     — sin/cos cyclical features
        prod_idx: (B,)              — int64, 0..n_products-1

    Output:
        forecast: (B, horizon)      — direct multi-output prediction
    """

    def __init__(
        self,
        n_products: int = 50,
        embed_dim: int = 8,
        lookback: int = 60,
        horizon: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_calendar: int = 6,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon

        self.product_embed = nn.Embedding(n_products, embed_dim)

        input_size = 1 + n_calendar + embed_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, past_qty: torch.Tensor, calendar: torch.Tensor,
                prod_idx: torch.Tensor) -> torch.Tensor:
        embed = self.product_embed(prod_idx)            # (B, embed_dim)
        L = past_qty.shape[1]
        embed_seq = embed.unsqueeze(1).expand(-1, L, -1)  # (B, L, embed_dim)
        x = torch.cat([past_qty, calendar, embed_seq], dim=-1)  # (B, L, F)

        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                            # (B, hidden_size)
        return self.head(last_hidden)                    # (B, horizon)
```

- [ ] **Step 12.4 — Verify pass**

Run: `pytest tests/test_lstm_model.py -v -k forecaster`
Expected: 2 PASS.

- [ ] **Step 12.5 — Commit**

```bash
git add src/models/lstm_model.py tests/test_lstm_model.py
git commit -m "feat(lstm): LSTMForecaster with product embedding"
```

---

## Task 13: `train_model()` — training loop with early stopping

**Files:**
- Modify: `src/models/lstm_model.py`
- Modify: `tests/test_lstm_model.py`

- [ ] **Step 13.1 — Write the failing test**

Append to `tests/test_lstm_model.py`:

```python
from torch.utils.data import DataLoader


def test_train_model_returns_trained_model_and_history():
    torch.manual_seed(0)
    np.random.seed(0)
    X_qty, X_cal, prod_idx, y = _dummy_arrays(n=16, lookback=5, horizon=3)
    train_ds = lstm_model.LSTMDataset(X_qty[:12], X_cal[:12], prod_idx[:12], y[:12])
    val_ds   = lstm_model.LSTMDataset(X_qty[12:], X_cal[12:], prod_idx[12:], y[12:])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4)

    model = lstm_model.LSTMForecaster(
        n_products=2, embed_dim=4, lookback=5, horizon=3,
        hidden_size=8, num_layers=1, dropout=0.0,
    )

    trained, best_val, best_epoch, history = lstm_model.train_model(
        model, train_loader, val_loader, epochs=3, lr=1e-2, patience=10,
    )

    assert isinstance(trained, lstm_model.LSTMForecaster)
    assert isinstance(best_val, float)
    assert isinstance(best_epoch, int)
    assert 1 <= best_epoch <= 3
    assert "train_loss" in history and "val_loss" in history
    assert len(history["train_loss"]) == 3 and len(history["val_loss"]) == 3
    # best_epoch must point to the epoch with the lowest val loss
    assert history["val_loss"][best_epoch - 1] == min(history["val_loss"])


def test_train_model_early_stops_when_val_does_not_improve():
    """Patience=0 should stop after the first epoch (since improvement check fails)."""
    torch.manual_seed(0)
    X_qty, X_cal, prod_idx, y = _dummy_arrays(n=8, lookback=5, horizon=3)
    train_ds = lstm_model.LSTMDataset(X_qty[:6], X_cal[:6], prod_idx[:6], y[:6])
    val_ds   = lstm_model.LSTMDataset(X_qty[6:], X_cal[6:], prod_idx[6:], y[6:])
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader   = DataLoader(val_ds,   batch_size=2)

    model = lstm_model.LSTMForecaster(
        n_products=2, embed_dim=2, lookback=5, horizon=3,
        hidden_size=4, num_layers=1, dropout=0.0,
    )
    _, _, _, history = lstm_model.train_model(
        model, train_loader, val_loader, epochs=20, lr=1e-2, patience=0,
    )

    # With patience=0, should stop very early (≤ 2 epochs typically).
    assert len(history["val_loss"]) <= 5
```

- [ ] **Step 13.2 — Verify failing**

Run: `pytest tests/test_lstm_model.py -v -k train_model`
Expected: 2 FAIL.

- [ ] **Step 13.3 — Implement**

Append to `src/models/lstm_model.py`:

```python
import copy


def train_model(
    model: LSTMForecaster,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
    verbose: bool = True,
):
    """Train the LSTM with MSE loss, Adam, and early stopping on val loss.

    Returns:
        (best_model, best_val_loss, best_epoch, history)

    `best_epoch` is the 1-indexed epoch number at which `best_val_loss`
    was achieved. Use this — not `len(history["val_loss"])` — when retraining
    on train+val combined, otherwise you'll overshoot by the patience window.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_total, train_n = 0.0, 0
        for past_qty, calendar, prod_idx, y in train_loader:
            past_qty = past_qty.to(device)
            calendar = calendar.to(device)
            prod_idx = prod_idx.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(past_qty, calendar, prod_idx)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            train_total += loss.item() * y.size(0)
            train_n += y.size(0)
        train_loss = train_total / max(train_n, 1)

        # ---- val ----
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for past_qty, calendar, prod_idx, y in val_loader:
                past_qty = past_qty.to(device)
                calendar = calendar.to(device)
                prod_idx = prod_idx.to(device)
                y = y.to(device)
                preds = model(past_qty, calendar, prod_idx)
                loss = loss_fn(preds, y)
                val_total += loss.item() * y.size(0)
                val_n += y.size(0)
        val_loss = val_total / max(val_n, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        # ---- early stopping ----
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(best was epoch {best_epoch} @ val {best_val:.4f}).")
                break

    model.load_state_dict(best_state)
    return model, best_val, best_epoch, history
```

- [ ] **Step 13.4 — Verify pass**

Run: `pytest tests/test_lstm_model.py -v -k train_model`
Expected: 2 PASS.

- [ ] **Step 13.5 — Commit**

```bash
git add src/models/lstm_model.py tests/test_lstm_model.py
git commit -m "feat(lstm): training loop with MSE + early stopping"
```

---

## Task 14: `predict()`, `inverse_scale_predictions()`, `compute_metrics()`, baselines

**Files:**
- Modify: `src/models/lstm_model.py`
- Modify: `tests/test_lstm_model.py`

- [ ] **Step 14.1 — Write the failing test**

Append to `tests/test_lstm_model.py`:

```python
def test_predict_returns_array_with_correct_shape():
    torch.manual_seed(0)
    X_qty, X_cal, prod_idx, y = _dummy_arrays(n=6, lookback=5, horizon=3)
    ds = lstm_model.LSTMDataset(X_qty, X_cal, prod_idx, y)
    loader = DataLoader(ds, batch_size=2)

    model = lstm_model.LSTMForecaster(
        n_products=2, embed_dim=2, lookback=5, horizon=3,
        hidden_size=4, num_layers=1, dropout=0.0,
    )

    preds = lstm_model.predict(model, loader)

    assert preds.shape == (6, 3)
    assert preds.dtype == np.float32


def test_inverse_scale_predictions_roundtrip():
    from sklearn.preprocessing import MinMaxScaler
    sc_a = MinMaxScaler().fit(np.array([[0.0], [10.0]]))
    sc_b = MinMaxScaler().fit(np.array([[0.0], [100.0]]))
    scalers = {"a": sc_a, "b": sc_b}

    # Two products' predictions in scaled space
    preds_scaled = np.array([[0.5, 1.0], [0.5, 1.0]], dtype=np.float32)
    item_ids = np.array(["a", "b"])

    preds_real = lstm_model.inverse_scale_predictions(preds_scaled, item_ids, scalers)

    # 'a' max=10 → 0.5→5, 1.0→10; 'b' max=100 → 0.5→50, 1.0→100
    assert preds_real[0, 0] == pytest.approx(5.0)
    assert preds_real[0, 1] == pytest.approx(10.0)
    assert preds_real[1, 0] == pytest.approx(50.0)
    assert preds_real[1, 1] == pytest.approx(100.0)


def test_compute_metrics_zero_error_case():
    preds = np.array([[1.0, 2.0], [3.0, 4.0]])
    targets = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = lstm_model.compute_metrics(preds, targets)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["mape"] == pytest.approx(0.0)


def test_compute_metrics_skips_zero_targets_for_mape():
    preds = np.array([[10.0, 5.0]])
    targets = np.array([[0.0, 5.0]])  # first target is 0 → skipped for MAPE
    m = lstm_model.compute_metrics(preds, targets)
    # MAPE only counts the second pair: |10-5|? No, |5-5|/5 = 0
    assert m["mape"] == pytest.approx(0.0)


def test_naive_baseline_repeats_last_observed_value():
    past_qty = np.array([[1.0, 2.0, 3.0]])  # one example, lookback 3
    horizon = 4
    out = lstm_model.naive_baseline(past_qty, horizon)
    assert out.shape == (1, 4)
    assert np.allclose(out, 3.0)


def test_seasonal_naive_baseline_uses_same_weekday_last_week():
    # past 14 days: last 7 are days "8..14"
    past_qty = np.arange(1, 15, dtype=float).reshape(1, 14)
    horizon = 7
    out = lstm_model.seasonal_naive_baseline(past_qty, horizon)
    assert out.shape == (1, 7)
    # The seasonal-naive forecast for the next 7 days copies the most recent 7.
    assert np.allclose(out[0], np.arange(8, 15))
```

- [ ] **Step 14.2 — Verify failing**

Run: `pytest tests/test_lstm_model.py -v -k "predict or inverse or metrics or baseline"`
Expected: 6 FAIL.

- [ ] **Step 14.3 — Implement**

Append to `src/models/lstm_model.py`:

```python
def predict(model: LSTMForecaster, loader, device: str = "cpu") -> np.ndarray:
    """Run the model on a DataLoader and return predictions as a numpy array.

    Output shape: (N, horizon).
    """
    model = model.to(device).eval()
    out_chunks = []
    with torch.no_grad():
        for past_qty, calendar, prod_idx, _y in loader:
            past_qty = past_qty.to(device)
            calendar = calendar.to(device)
            prod_idx = prod_idx.to(device)
            preds = model(past_qty, calendar, prod_idx)
            out_chunks.append(preds.cpu().numpy())
    return np.concatenate(out_chunks, axis=0).astype(np.float32)


def inverse_scale_predictions(preds_scaled: np.ndarray,
                              item_ids: np.ndarray,
                              scalers: dict) -> np.ndarray:
    """Inverse the per-product MinMax scaling on a (N, horizon) array.

    `item_ids` is parallel to the rows of `preds_scaled` and identifies which
    product each row belongs to.
    """
    out = np.zeros_like(preds_scaled, dtype=np.float32)
    for i, item_id in enumerate(item_ids):
        scaler = scalers[item_id]
        row = preds_scaled[i].reshape(-1, 1)
        out[i] = scaler.inverse_transform(row).flatten().astype(np.float32)
    return out


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """RMSE, MAE, MAPE computed across all (window, day) pairs.

    MAPE skips entries where target == 0 (would be undefined).
    """
    preds = np.asarray(preds, dtype=np.float64).flatten()
    targets = np.asarray(targets, dtype=np.float64).flatten()
    err = preds - targets

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    nonzero = targets != 0
    if nonzero.sum() == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs(err[nonzero] / targets[nonzero])) * 100.0)

    return {"rmse": rmse, "mae": mae, "mape": mape}


def naive_baseline(past_qty_2d: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast that just repeats the last observed value `horizon` times.

    Args:
        past_qty_2d: (N, lookback) — last column is "today".
        horizon: number of days to forecast.
    """
    last = past_qty_2d[:, -1:]
    return np.tile(last, (1, horizon)).astype(np.float32)


def seasonal_naive_baseline(past_qty_2d: np.ndarray, horizon: int,
                            period: int = 7) -> np.ndarray:
    """Forecast = same weekday last period.

    For days t..t+horizon-1, predict y_{t-period}..y_{t+horizon-1-period}.
    Requires past_qty_2d to have at least `period` columns; if `horizon`
    exceeds `period`, the pattern repeats.
    """
    n, look = past_qty_2d.shape
    if look < period:
        raise ValueError(f"past_qty has {look} cols < period {period}")
    out = np.zeros((n, horizon), dtype=np.float32)
    for h in range(horizon):
        out[:, h] = past_qty_2d[:, -period + (h % period)]
    return out
```

- [ ] **Step 14.4 — Verify pass**

Run: `pytest tests/test_lstm_model.py -v`
Expected: all PASS (10 tests in this file by now).

- [ ] **Step 14.5 — Commit**

```bash
git add src/models/lstm_model.py tests/test_lstm_model.py
git commit -m "feat(lstm): predict, inverse-scale, metrics, naive baselines"
```

> **Checkpoint after Task 14:** the model layer is complete. Confirm:
> ```bash
> pytest -v   # all preprocessing + lstm_model tests green
> ```

---

## Task 15: Replace `notebooks/01_EDA.ipynb` with raw-data exploration

**Files:**
- Replace: `notebooks/01_EDA.ipynb`

This is a notebook task — verification is "run all cells without errors and check the plots look sensible." There are no unit tests, but the notebook should run end-to-end via `nbconvert`.

- [ ] **Step 15.1 — Write the notebook content**

Build a notebook with the following cells (markdown headings + code). Use Jupyter or `jupytext` to author it; the cell sequence below is the spec.

**Markdown cell 1 — Title and goal:**
```
# 01 — EDA: Raw Sales Data

Shared exploration of `sales.csv` for the team. Each model notebook
(`02_LSTM.ipynb`, `03_RNN.ipynb`, etc.) builds on what's documented here.
```

**Code cell 2 — Imports and data load:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import preprocessing

df = preprocessing.load_raw_sales("sales.csv")
print(df.shape)
df.head()
```

**Markdown cell 3 — Section: shape and date range.**

**Code cell 4 — Basic profile:**
```python
print(f"rows: {len(df):,}")
print(f"date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"days covered: {df['date'].nunique()}")
print(f"unique items: {df['item_id'].nunique():,}")
print(f"unique stores: {df['store_id'].nunique()}")
print()
print("nulls per column:")
print(df.isna().sum())
```

**Markdown cell 5 — Section: quantity distribution and returns.**

**Code cell 6:**
```python
print(df["quantity"].describe(percentiles=[.5, .75, .9, .99]))
n_returns = (df["quantity"] < 0).sum()
print(f"\nnegative-quantity rows (returns): {n_returns:,} "
      f"({100*n_returns/len(df):.2f}%)")
```

**Code cell 7 — quantity histogram (clipped):**
```python
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["quantity"].clip(-10, 100), bins=80, ax=ax)
ax.set_title("Quantity per transaction (clipped to [-10, 100] for readability)")
ax.set_xlabel("quantity"); ax.set_ylabel("count")
plt.show()
```

**Markdown cell 8 — Section: per-product activity.**

**Code cell 9:**
```python
daily = preprocessing.aggregate_daily(df)
per_item = (
    daily.groupby("item_id")
         .agg(days_with_sales=("date", "nunique"),
              total_qty=("quantity", "sum"))
         .sort_values("total_qty", ascending=False)
)

print("days_with_sales distribution:")
print(per_item["days_with_sales"].describe(percentiles=[.5, .9, .99]).round(1))

print("\nProducts with full 761-day history:",
      (per_item["days_with_sales"] >= 761).sum())

per_item.head(10)
```

**Code cell 10 — top-10 quantity bar chart:**
```python
fig, ax = plt.subplots(figsize=(10, 4))
per_item.head(10)["total_qty"].plot(kind="bar", ax=ax)
ax.set_title("Top 10 products by total quantity (Aug 2022 – Sep 2024)")
ax.set_ylabel("total quantity"); ax.set_xlabel("item_id")
plt.tight_layout(); plt.show()
```

**Markdown cell 11 — Section: aggregate daily volume.**

**Code cell 12:**
```python
total_daily = daily.groupby("date")["quantity"].sum()
fig, ax = plt.subplots(figsize=(12, 4))
total_daily.plot(ax=ax)
ax.set_title("Total daily quantity (all products, all stores)")
ax.set_ylabel("quantity"); ax.set_xlabel("date")
plt.show()
```

**Markdown cell 13 — Section: weekly seasonality.**

**Code cell 14:**
```python
total_daily_df = total_daily.reset_index()
total_daily_df["dow"] = total_daily_df["date"].dt.dayofweek
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=total_daily_df, x="dow", y="quantity", ax=ax)
ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
ax.set_title("Total daily quantity by day-of-week")
plt.show()
```

**Markdown cell 15 — Summary takeaways:**
```
**Key findings used downstream by 02_LSTM.ipynb:**

- 761 days of data, ~28k unique items, 4 stores.
- Most products are sparse (median ~72 days of sales) — only the heavy-volume head
  has the full history needed for a 30-day LSTM forecast.
- Negative quantities (returns) are present and are netted into daily totals.
- Clear weekly seasonality at the aggregate level → calendar features will help.
```

- [ ] **Step 15.2 — Run the notebook end-to-end**

Run: `jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb --output 01_EDA.ipynb`
Expected: completes without errors. Open the notebook and visually confirm the four plots (quantity histogram, top-10 bar, aggregate time series, dow boxplot) render correctly.

- [ ] **Step 15.3 — Commit**

```bash
git add notebooks/01_EDA.ipynb
git commit -m "feat(notebooks): replace 01_EDA template with real raw-data exploration"
```

---

## Task 16: `02_LSTM.ipynb` Section 1+2 — Data preparation and EDA

**Files:**
- Create/replace: `notebooks/02_LSTM.ipynb` (initial version with these two sections only)

- [ ] **Step 16.1 — Build the first sections of the notebook**

**Markdown cell 1 — title:**
```
# 02 — LSTM Demand Forecasting

End-to-end PyTorch LSTM that predicts the next 30 days of daily demand
for the top 50 products. Six sections, one per ML lifecycle stage:

1. Data preparation
2. EDA (LSTM-specific)
3. Feature engineering
4. Model selection & training
5. Evaluation
6. Fine-tuning
```

**Markdown cell 2 — Section 1 heading:**
```
## 1. Data preparation

Calls `src.preprocessing.prepare_data()` end-to-end. This writes:

- `data/processed/daily.csv`
- `data/processed/scalers.pkl`
- `data/processed/splits.json`

The team's RNN/RF/GP notebooks will read the same files.
```

**Code cell 3 — imports:**
```python
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src import preprocessing
from src.models import lstm_model

torch.manual_seed(42)
np.random.seed(42)
```

**Code cell 4 — run prepare_data:**
```python
splits = preprocessing.prepare_data(
    csv_path="sales.csv",
    n_products=50,
    full_history_days=761,
    val_days=60,
    test_frac=0.20,
    outdir="data/processed",
)
splits
```

**Code cell 5 — load the processed data back:**
```python
daily = pd.read_csv("data/processed/daily.csv", parse_dates=["date"])
with open("data/processed/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

print(f"top-50 products kept: {daily['item_id'].nunique()}")
print(f"days per product:     {daily.groupby('item_id')['date'].nunique().min()}")
print(f"split boundaries:     {splits}")
daily.head()
```

**Markdown cell 6 — Section 2 heading:**
```
## 2. EDA (LSTM-specific)

Now that we've filtered to the top 50 products and added calendar features,
we look at *this* dataset (not the raw 28k-product `sales.csv`) — what
patterns will the LSTM need to learn?
```

**Code cell 7 — pick representative products:**
```python
totals = daily.groupby("item_id")["quantity"].sum().sort_values(ascending=False)
representatives = [totals.index[0], totals.index[24], totals.index[-1]]  # top, median, bottom of top-50
representatives
```

**Code cell 8 — time series for representatives:**
```python
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, item_id in zip(axes, representatives):
    sub = daily[daily["item_id"] == item_id].sort_values("date")
    ax.plot(sub["date"], sub["quantity"])
    ax.set_title(f"item {item_id}")
    ax.set_ylabel("quantity")
plt.tight_layout(); plt.show()
```

**Code cell 9 — weekly seasonality across all top-50:**
```python
daily["dow"] = daily["date"].dt.dayofweek
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=daily, x="dow", y="quantity", ax=ax)
ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
ax.set_title("Top-50 products: quantity by day-of-week")
plt.show()
```

**Code cell 10 — sanity-check scaling:**
```python
print("quantity_scaled — should be in [0, 1] for training rows:")
train_end = pd.Timestamp(splits["train_end"])
train_rows = daily[daily["date"] <= train_end]
print(train_rows["quantity_scaled"].describe()[["min", "max", "mean"]])
```

- [ ] **Step 16.2 — Execute the notebook**

Run: `jupyter nbconvert --to notebook --execute notebooks/02_LSTM.ipynb --output 02_LSTM.ipynb`
Expected: completes without errors. **Note:** this step is slow (~30s–2 min for `prepare_data` over 7M rows). On a tight machine you can subsample `sales.csv` for development; restore for the final run.

- [ ] **Step 16.3 — Commit**

```bash
git add notebooks/02_LSTM.ipynb
git commit -m "feat(notebooks): 02_LSTM sections 1-2 (data prep + EDA)"
```

---

## Task 17: `02_LSTM.ipynb` Section 3 — Feature engineering and DataLoaders

**Files:**
- Modify: `notebooks/02_LSTM.ipynb`

- [ ] **Step 17.1 — Append Section 3 cells**

**Markdown cell — Section 3 heading:**
```
## 3. Feature engineering

What we're doing in this section:

1. Build a `product_to_idx` mapping (string item IDs → 0..49 integers for the embedding).
2. Slice `daily.csv` into the three time periods (train, val, test) using `splits.json`.
3. For each slice, run `create_sequences()` to generate `(60-day input, 30-day target)` pairs.
4. Wrap each slice in an `LSTMDataset` and a PyTorch `DataLoader`.
```

**Code cell — product index map:**
```python
product_to_idx = {pid: i for i, pid in enumerate(sorted(daily["item_id"].unique()))}
idx_to_product = {i: pid for pid, i in product_to_idx.items()}
len(product_to_idx)
```

**Code cell — slice into train / val / test:**
```python
train_end = pd.Timestamp(splits["train_end"])
val_end   = pd.Timestamp(splits["val_end"])

# For sequence generation each slice needs LOOKBACK days of context BEFORE its
# first target day, so val and test ranges include the lookback "lead-in".
LOOKBACK = 60
HORIZON = 30

train_slice = daily[daily["date"] <= train_end].copy()
val_slice   = daily[(daily["date"] >  train_end - pd.Timedelta(days=LOOKBACK))
                    & (daily["date"] <= val_end)].copy()
test_slice  = daily[(daily["date"] >  val_end   - pd.Timedelta(days=LOOKBACK))].copy()

print(f"train rows: {len(train_slice):,}  ({train_slice['date'].min()} → {train_slice['date'].max()})")
print(f"val rows:   {len(val_slice):,}    ({val_slice['date'].min()} → {val_slice['date'].max()})")
print(f"test rows:  {len(test_slice):,}   ({test_slice['date'].min()} → {test_slice['date'].max()})")
```

**Code cell — create sequences:**
```python
X_qty_tr, X_cal_tr, idx_tr, y_tr = preprocessing.create_sequences(
    train_slice, product_to_idx, lookback=LOOKBACK, horizon=HORIZON)
X_qty_va, X_cal_va, idx_va, y_va = preprocessing.create_sequences(
    val_slice,   product_to_idx, lookback=LOOKBACK, horizon=HORIZON)
X_qty_te, X_cal_te, idx_te, y_te = preprocessing.create_sequences(
    test_slice,  product_to_idx, lookback=LOOKBACK, horizon=HORIZON)

print(f"train windows: {X_qty_tr.shape}")
print(f"val windows:   {X_qty_va.shape}")
print(f"test windows:  {X_qty_te.shape}")
```

**Code cell — wrap into DataLoaders:**
```python
BATCH_SIZE = 64

train_ds = lstm_model.LSTMDataset(X_qty_tr, X_cal_tr, idx_tr, y_tr)
val_ds   = lstm_model.LSTMDataset(X_qty_va, X_cal_va, idx_va, y_va)
test_ds  = lstm_model.LSTMDataset(X_qty_te, X_cal_te, idx_te, y_te)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print("train batches:", len(train_loader))
print("val batches:  ", len(val_loader))
print("test batches: ", len(test_loader))
```

- [ ] **Step 17.2 — Execute and verify shapes**

Re-execute the notebook (`jupyter nbconvert --to notebook --execute notebooks/02_LSTM.ipynb --output 02_LSTM.ipynb`). Expected: all the print statements show non-zero windows in each split.

- [ ] **Step 17.3 — Commit**

```bash
git add notebooks/02_LSTM.ipynb
git commit -m "feat(notebooks): 02_LSTM section 3 (feature engineering + dataloaders)"
```

---

## Task 18: `02_LSTM.ipynb` Section 4 — Model, training, and LSTM teaching material

**Files:**
- Modify: `notebooks/02_LSTM.ipynb`

- [ ] **Step 18.1 — Append teaching markdown cells**

**Markdown cell — Section 4 heading:**
```
## 4. Model selection & training

Before we train, here's a plain-language tour of what an LSTM is and why
we're using one. If you already know LSTMs, you can skip the next six
sub-sections and jump to "Build the model."
```

**Markdown cell — what problem LSTMs solve:**
```
### What problem do LSTMs solve?

A regular ("vanilla") RNN reads a sequence one element at a time, updating
an internal hidden state. The trouble is that vanilla RNNs **forget**:
information from the start of the sequence gets diluted as new elements
arrive, so they struggle to connect "what happened 60 days ago" to "what
will happen tomorrow."

LSTMs (Long Short-Term Memory networks, Hochreiter & Schmidhuber, 1997)
add a separate **cell state** that flows through time with minimal
modification, so information can persist over long ranges. In addition,
three learned "gates" decide what to forget, what to add, and what to
output at each step.
```

**Markdown cell — the three gates:**
```
### The three gates

At every timestep, an LSTM cell computes three "gates" — each one a
sigmoid-activated linear transformation of the previous hidden state and
the current input:

| Gate    | Question it answers                                  |
|---------|------------------------------------------------------|
| Forget  | What of the previous cell state should I drop?       |
| Input   | What new information should I write into the cell?   |
| Output  | Of the cell's contents, what do I expose now?        |

The math (PyTorch handles it for us):

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   # forget
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   # input
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   # output

c_t = f_t * c_{t-1} + i_t * g_t       # new cell state
h_t = o_t * tanh(c_t)                 # new hidden state
```
```

**Markdown cell — hidden vs cell state:**
```
### Hidden state vs cell state

Two pieces of "memory" travel through the LSTM:

- **Cell state `c_t`** — the LSTM's long-term memory. Information written
  here can survive for many timesteps because the only operations that
  touch it are element-wise (no matrix multiplications that would dilute
  it).
- **Hidden state `h_t`** — the LSTM's short-term, exposed scratchpad. This
  is what the next layer reads.

After 60 timesteps, we take the **final hidden state** (`h_n[-1]` in
PyTorch — the top layer's hidden state from the last timestep) and feed it
into a `Linear` layer that produces our 30-day forecast.
```

**Markdown cell — why a product embedding:**
```
### Why an embedding for `product_idx`?

We have 50 products. We could one-hot encode them (a 50-dim vector with
49 zeros and a single 1) — but that wastes parameters and treats every
product as equally distant from every other.

An **embedding** is a small lookup table: each product gets its own
learned vector (we use 8 dimensions). Products with similar demand
patterns will, after training, end up with similar vectors. The model
then concatenates the embedding with the per-day quantity and calendar
features at every timestep, so the LSTM "knows which product it's
looking at" throughout the entire sequence.
```

**Markdown cell — sliding windows explanation:**
```
### Why a 60→30 sliding window?

Each training example is a `(60-day input, 30-day target)` pair. We
slide that window across the time series with stride 1, generating many
examples per product:

```
                day index
   0 ─────────── 59 60 ─────────── 89    ← example 1
       1 ─────────── 60 61 ─────────── 90  ← example 2
                ...
```

This gives the model many opportunities to learn the
"given the last 60 days, what do the next 30 look like?" mapping. The
choice of 60 (lookback) and 30 (horizon) is a hyperparameter — we'll
sweep it in Section 6.
```

**Markdown cell — Build the model:**
```
### Build the model
```

**Code cell — instantiate model:**
```python
N_PRODUCTS = len(product_to_idx)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
EMBED_DIM = 8

model = lstm_model.LSTMForecaster(
    n_products=N_PRODUCTS,
    embed_dim=EMBED_DIM,
    lookback=LOOKBACK,
    horizon=HORIZON,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    n_calendar=6,
)
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"trainable parameters: {n_params:,}")
```

**Code cell — single-example trace (the worked example from the spec):**
```python
# Pull one example out of the train set and walk it through the layers manually,
# printing the shape after each step. This is the "what is the LSTM doing?" demo.
past_qty, calendar, prod_idx, target = train_ds[0]
past_qty = past_qty.unsqueeze(0)   # (1, 60, 1)
calendar = calendar.unsqueeze(0)   # (1, 60, 6)
prod_idx = prod_idx.unsqueeze(0)   # (1,)

print("Input shapes:")
print(f"  past_qty:  {past_qty.shape}")
print(f"  calendar:  {calendar.shape}")
print(f"  prod_idx:  {prod_idx.shape}")

embed = model.product_embed(prod_idx)
print(f"\nproduct_embed output: {embed.shape}")

embed_seq = embed.unsqueeze(1).expand(-1, LOOKBACK, -1)
combined = torch.cat([past_qty, calendar, embed_seq], dim=-1)
print(f"combined input to LSTM: {combined.shape}")

with torch.no_grad():
    _, (h_n, _) = model.lstm(combined)
    last_hidden = h_n[-1]
    forecast = model.head(last_hidden)
print(f"final hidden state:     {last_hidden.shape}")
print(f"forecast (30 days):     {forecast.shape}")
```

**Markdown cell — Train:**
```
### Train

We train for up to 100 epochs with early stopping. On CPU this takes
~5–15 minutes; on GPU, a fraction of that.
```

**Code cell — train:**
```python
EPOCHS = 100
LR = 1e-3
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"training on: {DEVICE}")

trained_model, best_val, history = lstm_model.train_model(
    model, train_loader, val_loader,
    epochs=EPOCHS, lr=LR, patience=PATIENCE,
    device=DEVICE, verbose=True,
)
print(f"\nbest val MSE: {best_val:.4f}")
```

**Code cell — plot training curves:**
```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history["train_loss"], label="train")
ax.plot(history["val_loss"], label="val")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
ax.set_title("Training curve"); ax.legend()
plt.show()
```

- [ ] **Step 18.2 — Execute and verify**

Re-execute the notebook. Expected: training curve trends down on both train and val (val should plateau before train, where early stopping kicks in).

- [ ] **Step 18.3 — Commit**

```bash
git add notebooks/02_LSTM.ipynb
git commit -m "feat(notebooks): 02_LSTM section 4 (LSTM teaching + training)"
```

---

## Task 19: `02_LSTM.ipynb` Section 5 — Evaluation

**Files:**
- Modify: `notebooks/02_LSTM.ipynb`

- [ ] **Step 19.1 — Append Section 5 cells**

**Markdown cell — Section 5 heading:**
```
## 5. Evaluation

We evaluate on the **5 non-overlapping 30-day test windows** described in
the spec. For each window, we feed the actual 60-day past and ask the
model for 30 days ahead. We then:

1. Inverse-scale predictions back to real units.
2. Compute RMSE / MAE / MAPE per product and aggregated.
3. Compare against naive and seasonal-naive baselines.
4. Save predictions to `data/predictions/lstm_test_predictions.csv` for the team's ensemble.
```

**Code cell — build the 5 non-overlapping eval windows:**
```python
# For each of the 50 products, build exactly 5 (input, target) windows that
# tile the 152 test days non-overlappingly with horizon=30 (5 * 30 = 150 used).
test_start = pd.Timestamp(splits["test_start"])
test_end   = pd.Timestamp(splits["test_end"])

eval_rows = []          # list of dicts to assemble eval batches
for item_id, idx in product_to_idx.items():
    series = daily[daily["item_id"] == item_id].sort_values("date").reset_index(drop=True)
    test_first_idx = series.index[series["date"] == test_start][0]
    for w in range(5):
        target_start = test_first_idx + w * HORIZON
        target_end = target_start + HORIZON          # exclusive
        if target_end > len(series):
            break
        input_start = target_start - LOOKBACK
        if input_start < 0:
            break
        eval_rows.append({
            "item_id": item_id,
            "prod_idx": idx,
            "X_qty":  series.loc[input_start:target_start - 1, "quantity_scaled"].values.reshape(-1, 1),
            "X_cal":  series.loc[input_start:target_start - 1,
                                 ["dow_sin","dow_cos","dom_sin","dom_cos","month_sin","month_cos"]].values,
            "y_scaled": series.loc[target_start:target_end - 1, "quantity_scaled"].values,
            "y_real":   series.loc[target_start:target_end - 1, "quantity"].values,
            "target_dates": series.loc[target_start:target_end - 1, "date"].values,
        })
print(f"total eval windows: {len(eval_rows)} (expected {50*5} = 250)")
```

**Code cell — generate model predictions:**
```python
X_qty_eval = np.stack([r["X_qty"]    for r in eval_rows]).astype(np.float32)
X_cal_eval = np.stack([r["X_cal"]    for r in eval_rows]).astype(np.float32)
idx_eval   = np.array([r["prod_idx"] for r in eval_rows], dtype=np.int64)
y_eval_scaled = np.stack([r["y_scaled"] for r in eval_rows]).astype(np.float32)
y_eval_real   = np.stack([r["y_real"]   for r in eval_rows]).astype(np.float32)
item_ids_eval = np.array([r["item_id"] for r in eval_rows])

eval_ds = lstm_model.LSTMDataset(X_qty_eval, X_cal_eval, idx_eval, y_eval_scaled)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

preds_scaled = lstm_model.predict(trained_model, eval_loader, device=DEVICE)
preds_real   = lstm_model.inverse_scale_predictions(preds_scaled, item_ids_eval, scalers)
preds_real   = np.clip(preds_real, 0.0, None)   # demand can't be negative

print("preds_real shape:", preds_real.shape, "  (expected (250, 30))")
```

**Code cell — metrics:**
```python
m_lstm = lstm_model.compute_metrics(preds_real, y_eval_real)
print("LSTM:", m_lstm)

# Baselines: use the last 60 days of input (which we have in X_qty_eval).
# We need them in REAL units, so use the unscaled past quantities.
# Pull "real" past_qty by inverse-scaling X_qty_eval per product.
past_real = np.zeros_like(X_qty_eval[:, :, 0])
for i, item_id in enumerate(item_ids_eval):
    past_real[i] = scalers[item_id].inverse_transform(
        X_qty_eval[i, :, 0].reshape(-1, 1)
    ).flatten()

naive = lstm_model.naive_baseline(past_real, HORIZON)
seasonal = lstm_model.seasonal_naive_baseline(past_real, HORIZON, period=7)

m_naive = lstm_model.compute_metrics(naive, y_eval_real)
m_seas  = lstm_model.compute_metrics(seasonal, y_eval_real)
print("Naive:", m_naive)
print("Seasonal-naive:", m_seas)
```

**Code cell — diagnostic plots (predicted vs actual for 4 reps):**
```python
plot_items = [representatives[0], representatives[1], representatives[2],
              list(product_to_idx)[10]]    # one extra
fig, axes = plt.subplots(len(plot_items), 1, figsize=(12, 9), sharex=False)
for ax, item_id in zip(axes, plot_items):
    mask = item_ids_eval == item_id
    # Concatenate the 5 30-day windows in chronological order
    actuals = np.concatenate([r["y_real"] for r in eval_rows if r["item_id"] == item_id])
    preds   = np.concatenate(preds_real[mask])
    ax.plot(actuals, label="actual")
    ax.plot(preds, label="LSTM", linestyle="--")
    ax.set_title(f"item {item_id}")
    ax.legend()
plt.tight_layout(); plt.show()
```

**Code cell — residuals histogram:**
```python
residuals = (preds_real - y_eval_real).flatten()
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(residuals, bins=80, ax=ax)
ax.set_title("Residuals (predicted - actual), all (window, day) pairs")
ax.set_xlabel("residual"); ax.axvline(0, color="black", linestyle=":")
plt.show()
print(f"residual mean: {residuals.mean():.3f}  (≈ 0 = unbiased)")
```

**Code cell — error by horizon day:**
```python
per_day_rmse = np.sqrt(((preds_real - y_eval_real) ** 2).mean(axis=0))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(1, HORIZON + 1), per_day_rmse, marker="o")
ax.set_xlabel("forecast horizon (days ahead)")
ax.set_ylabel("RMSE")
ax.set_title("Error growth across the 30-day horizon")
plt.show()
```

**Code cell — save predictions for the ensemble:**
```python
from pathlib import Path
Path("data/predictions").mkdir(parents=True, exist_ok=True)

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
predictions_df.to_csv("data/predictions/lstm_test_predictions.csv", index=False)
print(f"saved {len(predictions_df):,} rows to data/predictions/lstm_test_predictions.csv")
```

- [ ] **Step 19.2 — Execute notebook end-to-end**

Run: `jupyter nbconvert --to notebook --execute notebooks/02_LSTM.ipynb --output 02_LSTM.ipynb`
Expected: completes without errors. Verify in the output:
- LSTM RMSE < both baseline RMSEs (otherwise the LSTM isn't earning its keep — go back and check Section 4).
- Predicted-vs-actual plots roughly track the actual lines.
- `data/predictions/lstm_test_predictions.csv` has 50 × 150 = 7,500 rows.

- [ ] **Step 19.3 — Commit**

```bash
git add notebooks/02_LSTM.ipynb
git commit -m "feat(notebooks): 02_LSTM section 5 (evaluation + baselines + saved preds)"
```

---

## Task 20: `02_LSTM.ipynb` Section 6 — Fine-tuning grid search

**Files:**
- Modify: `notebooks/02_LSTM.ipynb`

- [ ] **Step 20.1 — Append Section 6 cells**

**Markdown cell — Section 6 heading:**
```
## 6. Fine-tuning

One-knob-at-a-time grid (8 distinct trainings + 1 final retrain on
train+val combined). Each configuration trains from scratch on the train
slice; we pick the configuration with the lowest val RMSE.

Total trainings: 9. Expect ~30–60 min on CPU; faster on GPU.
```

**Code cell — define grid:**
```python
DEFAULTS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lookback": 60,
}

CONFIGS = [DEFAULTS.copy()]  # baseline counts as run #1
for hs in [32, 128]:
    CONFIGS.append({**DEFAULTS, "hidden_size": hs})
for nl in [1]:
    CONFIGS.append({**DEFAULTS, "num_layers": nl})
for dr in [0.1, 0.3]:
    CONFIGS.append({**DEFAULTS, "dropout": dr})
for lb in [30, 90]:
    CONFIGS.append({**DEFAULTS, "lookback": lb})

len(CONFIGS), CONFIGS[:3]
```

**Code cell — function to build sequences for an arbitrary lookback:**
```python
def build_loaders_for_lookback(lookback):
    """Reusable: re-create train/val data with a different lookback."""
    train_slice2 = daily[daily["date"] <= train_end].copy()
    val_slice2   = daily[(daily["date"] >  train_end - pd.Timedelta(days=lookback))
                         & (daily["date"] <= val_end)].copy()
    Xq_tr, Xc_tr, ix_tr, yy_tr = preprocessing.create_sequences(
        train_slice2, product_to_idx, lookback=lookback, horizon=HORIZON)
    Xq_va, Xc_va, ix_va, yy_va = preprocessing.create_sequences(
        val_slice2,   product_to_idx, lookback=lookback, horizon=HORIZON)
    tl = DataLoader(lstm_model.LSTMDataset(Xq_tr, Xc_tr, ix_tr, yy_tr),
                    batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(lstm_model.LSTMDataset(Xq_va, Xc_va, ix_va, yy_va),
                    batch_size=BATCH_SIZE)
    return tl, vl
```

**Code cell — grid search:**
```python
import time

results = []
for i, cfg in enumerate(CONFIGS):
    print(f"\n=== Config {i+1}/{len(CONFIGS)}: {cfg} ===")
    t0 = time.time()
    torch.manual_seed(42)

    tl, vl = build_loaders_for_lookback(cfg["lookback"])
    m = lstm_model.LSTMForecaster(
        n_products=N_PRODUCTS,
        embed_dim=EMBED_DIM,
        lookback=cfg["lookback"],
        horizon=HORIZON,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        n_calendar=6,
    )
    m, best_val, best_epoch, hist = lstm_model.train_model(
        m, tl, vl, epochs=EPOCHS, lr=LR, patience=PATIENCE,
        device=DEVICE, verbose=False,
    )
    epochs_run = len(hist["val_loss"])
    elapsed = time.time() - t0
    print(f"   best val MSE: {best_val:.4f} | best epoch: {best_epoch} "
          f"| epochs run: {epochs_run} | {elapsed:.0f}s")
    results.append({**cfg, "val_mse": best_val,
                    "best_epoch": best_epoch, "epochs_run": epochs_run})

results_df = pd.DataFrame(results).sort_values("val_mse")
results_df
```

**Code cell — pick winner and retrain on train+val:**
```python
winner = results_df.iloc[0].to_dict()
print(f"winner: {winner}")

# Combine train+val data with the winner's lookback.
combined_slice = daily[daily["date"] <= val_end].copy()
Xq_c, Xc_c, ix_c, yy_c = preprocessing.create_sequences(
    combined_slice, product_to_idx,
    lookback=int(winner["lookback"]), horizon=HORIZON,
)
combined_loader = DataLoader(
    lstm_model.LSTMDataset(Xq_c, Xc_c, ix_c, yy_c),
    batch_size=BATCH_SIZE, shuffle=True,
)
print(f"combined train+val windows: {Xq_c.shape}")

torch.manual_seed(42)
final_model = lstm_model.LSTMForecaster(
    n_products=N_PRODUCTS,
    embed_dim=EMBED_DIM,
    lookback=int(winner["lookback"]),
    horizon=HORIZON,
    hidden_size=int(winner["hidden_size"]),
    num_layers=int(winner["num_layers"]),
    dropout=float(winner["dropout"]),
    n_calendar=6,
)

# Retrain for the BEST EPOCH count from the sweep, not the total epochs run.
# `epochs_run` includes the patience-window of non-improving epochs after the
# best validation loss — using it would overshoot the optimum.
WINNER_EPOCHS = int(winner["best_epoch"])
final_model = final_model.to(DEVICE)
opt = torch.optim.Adam(final_model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()
for ep in range(1, WINNER_EPOCHS + 1):
    final_model.train()
    total, n = 0.0, 0
    for past_qty, calendar, prod_idx, y in combined_loader:
        past_qty, calendar, prod_idx, y = (
            past_qty.to(DEVICE), calendar.to(DEVICE), prod_idx.to(DEVICE), y.to(DEVICE))
        opt.zero_grad()
        preds = final_model(past_qty, calendar, prod_idx)
        loss = loss_fn(preds, y)
        loss.backward(); opt.step()
        total += loss.item() * y.size(0); n += y.size(0)
    print(f"final-model epoch {ep:3d} | train MSE {total/n:.4f}")
```

**Code cell — re-evaluate on test set with the final tuned model:**
```python
# Rebuild eval rows with the winner's lookback.
WINNER_LOOKBACK = int(winner["lookback"])

eval_rows_final = []
for item_id, idx in product_to_idx.items():
    series = daily[daily["item_id"] == item_id].sort_values("date").reset_index(drop=True)
    test_first_idx = series.index[series["date"] == test_start][0]
    for w in range(5):
        target_start = test_first_idx + w * HORIZON
        target_end = target_start + HORIZON
        if target_end > len(series): break
        input_start = target_start - WINNER_LOOKBACK
        if input_start < 0: break
        eval_rows_final.append({
            "item_id": item_id,
            "prod_idx": idx,
            "X_qty":  series.loc[input_start:target_start - 1, "quantity_scaled"].values.reshape(-1, 1),
            "X_cal":  series.loc[input_start:target_start - 1,
                                 ["dow_sin","dow_cos","dom_sin","dom_cos","month_sin","month_cos"]].values,
            "y_scaled": series.loc[target_start:target_end - 1, "quantity_scaled"].values,
            "y_real":   series.loc[target_start:target_end - 1, "quantity"].values,
            "target_dates": series.loc[target_start:target_end - 1, "date"].values,
        })

X_qty_f = np.stack([r["X_qty"]    for r in eval_rows_final]).astype(np.float32)
X_cal_f = np.stack([r["X_cal"]    for r in eval_rows_final]).astype(np.float32)
idx_f   = np.array([r["prod_idx"] for r in eval_rows_final], dtype=np.int64)
y_f_scaled = np.stack([r["y_scaled"] for r in eval_rows_final]).astype(np.float32)
y_f_real   = np.stack([r["y_real"]   for r in eval_rows_final]).astype(np.float32)
items_f    = np.array([r["item_id"] for r in eval_rows_final])

eval_loader_final = DataLoader(
    lstm_model.LSTMDataset(X_qty_f, X_cal_f, idx_f, y_f_scaled),
    batch_size=BATCH_SIZE, shuffle=False,
)
preds_f_scaled = lstm_model.predict(final_model, eval_loader_final, device=DEVICE)
preds_f_real   = np.clip(
    lstm_model.inverse_scale_predictions(preds_f_scaled, items_f, scalers),
    0.0, None,
)
m_final = lstm_model.compute_metrics(preds_f_real, y_f_real)
print("FINAL TUNED LSTM:", m_final)
print("Naive baseline:  ", m_naive)
print("Seasonal naive:  ", m_seas)
```

**Code cell — save final model and predictions:**
```python
import json
Path("models").mkdir(exist_ok=True)

torch.save(final_model.state_dict(), "models/lstm_final.pt")
with open("models/lstm_config.json", "w") as f:
    json.dump({
        "n_products": N_PRODUCTS,
        "embed_dim":  EMBED_DIM,
        "lookback":   WINNER_LOOKBACK,
        "horizon":    HORIZON,
        "hidden_size": int(winner["hidden_size"]),
        "num_layers":  int(winner["num_layers"]),
        "dropout":     float(winner["dropout"]),
        "n_calendar":  6,
        "winner_best_epoch": WINNER_EPOCHS,
    }, f, indent=2)

# Overwrite the test predictions with the tuned model's output.
records_final = []
for r, p in zip(eval_rows_final, preds_f_real):
    for d_idx in range(HORIZON):
        records_final.append({
            "date": pd.Timestamp(r["target_dates"][d_idx]),
            "item_id": r["item_id"],
            "predicted_quantity": float(p[d_idx]),
            "actual_quantity": float(r["y_real"][d_idx]),
        })
pd.DataFrame(records_final).to_csv("data/predictions/lstm_test_predictions.csv", index=False)
print("saved models/lstm_final.pt, models/lstm_config.json, data/predictions/lstm_test_predictions.csv")
```

**Markdown cell — closing write-up template:**
```
### What we learned

(Fill in after the grid runs:)

- Which knob mattered most? (Compare val MSE across configs.)
- What surprised you?
- What would you try next? (E.g., a longer horizon, more products,
  Bayesian optimization, adding price as a feature.)
```

- [ ] **Step 20.2 — Execute notebook end-to-end**

Run: `jupyter nbconvert --to notebook --execute notebooks/02_LSTM.ipynb --output 02_LSTM.ipynb --ExecutePreprocessor.timeout=7200`

Expected: completes without errors. The full grid takes ~30–90 minutes depending on hardware; the timeout flag (`7200` seconds = 2 hours) gives headroom.

- [ ] **Step 20.3 — Commit**

```bash
git add notebooks/02_LSTM.ipynb
git commit -m "feat(notebooks): 02_LSTM section 6 (grid-search fine-tuning + final model)"
```

---

## Task 21: Final integration check (definition of done)

**Files:** none modified — verification only.

- [ ] **Step 21.1 — Run all unit tests once more**

Run: `pytest -v`
Expected: all green (preprocessing + lstm_model suites).

- [ ] **Step 21.2 — Verify deliverable artifacts exist**

Run:
```bash
ls -la data/processed/daily.csv data/processed/scalers.pkl data/processed/splits.json \
       data/predictions/lstm_test_predictions.csv \
       models/lstm_final.pt models/lstm_config.json
```
Expected: every file listed.

- [ ] **Step 21.3 — Verify the saved model can be reloaded in a fresh process**

Run:
```bash
python -c "
import json, torch
from src.models.lstm_model import LSTMForecaster
cfg = json.load(open('models/lstm_config.json'))
m = LSTMForecaster(**{k:v for k,v in cfg.items() if k != 'winner_best_epoch'})
m.load_state_dict(torch.load('models/lstm_final.pt', map_location='cpu'))
m.eval()
print('reload OK; param count:', sum(p.numel() for p in m.parameters()))
"
```
Expected: `reload OK; param count: <some number>`.

- [ ] **Step 21.4 — Verify predictions CSV matches `splits.json`**

Run:
```bash
python -c "
import json, pandas as pd
splits = json.load(open('data/processed/splits.json'))
preds = pd.read_csv('data/predictions/lstm_test_predictions.csv', parse_dates=['date'])
assert preds['date'].min() >= pd.Timestamp(splits['test_start']), 'predictions start before test_start'
assert preds['date'].max() <= pd.Timestamp(splits['test_end']),   'predictions end after test_end'
assert preds['item_id'].nunique() == 50,                          'wrong product count'
print(f'OK: {len(preds):,} rows, {preds[\"item_id\"].nunique()} products, '
      f'{preds[\"date\"].min().date()} → {preds[\"date\"].max().date()}')
"
```
Expected: `OK: 7500 rows, 50 products, ...`.

- [ ] **Step 21.5 — Verify `requirements.txt` has no `tensorflow`**

Run:
```bash
grep -i tensorflow requirements.txt && echo "FAIL: tensorflow still listed" || echo "OK"
```
Expected: `OK`.

- [ ] **Step 21.6 — Final commit (if any incidental changes accumulated)**

Run `git status`. If anything is staged or unstaged, commit it with a message like `chore: cleanup`. Otherwise no commit needed.

---

## Self-review (what was checked)

**Spec coverage:**
- §5.1 Data preparation → Tasks 2–5, 10 ✓
- §5.2 EDA → Tasks 15 (general), 16 (LSTM-specific in §2 of 02_LSTM) ✓
- §5.3 Feature engineering → Tasks 6, 7, 8, 9, 17 ✓
- §5.4 Model selection & training → Tasks 11, 12, 13, 18 ✓
- §5.5 Evaluation → Task 19 ✓
- §5.6 Fine-tuning → Task 20 ✓
- §6 Deliverables → all paths covered in Tasks 1, 10, 19, 20 ✓
- §7 Replaced files → handled in Tasks 1 (init.py), 2 (preprocessing.py), 11 (lstm_model.py), 15 (01_EDA), 16–20 (02_LSTM) ✓
- §11 Definition of done → Task 21 ✓

**Placeholders:** none (all code blocks are concrete; the closing markdown cell in §6 is template prose for the user to fill in *after* the grid runs — the grid itself runs deterministically).

**Type/name consistency:** function names, argument names, return shapes are consistent across `preprocessing.py` ↔ `lstm_model.py` ↔ notebook (verified: `create_sequences` returns `(X_qty, X_cal, prod_idx, y)`, `LSTMDataset` and `LSTMForecaster.forward` consume the same names).
