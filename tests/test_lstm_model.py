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
