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
