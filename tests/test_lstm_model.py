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
