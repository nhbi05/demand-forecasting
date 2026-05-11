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
    """Patience=0 with a divergent learning rate forces val loss to worsen
    after the first epoch, so training stops early."""
    torch.manual_seed(0)
    np.random.seed(0)
    X_qty, X_cal, prod_idx, y = _dummy_arrays(n=8, lookback=5, horizon=3)
    train_ds = lstm_model.LSTMDataset(X_qty[:6], X_cal[:6], prod_idx[:6], y[:6])
    val_ds   = lstm_model.LSTMDataset(X_qty[6:], X_cal[6:], prod_idx[6:], y[6:])
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader   = DataLoader(val_ds,   batch_size=2)

    model = lstm_model.LSTMForecaster(
        n_products=2, embed_dim=2, lookback=5, horizon=3,
        hidden_size=4, num_layers=1, dropout=0.0,
    )
    # lr=10 is wildly too high; weights blow up and val loss diverges fast.
    _, _, _, history = lstm_model.train_model(
        model, train_loader, val_loader, epochs=20, lr=10.0, patience=0,
        verbose=False,
    )

    # With patience=0 and a divergent setup, training must stop well before 20.
    assert len(history["val_loss"]) < 20


def test_train_model_accepts_weight_decay_and_custom_loss():
    """v2 knobs: weight_decay (L2 reg) and a swap-in loss function."""
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

    # Use Huber loss + weight decay; just confirm it runs and returns the
    # 4-tuple with sensible types.
    trained, best_val, best_epoch, history = lstm_model.train_model(
        model, train_loader, val_loader,
        epochs=2, lr=1e-2, patience=10,
        weight_decay=1e-4,
        loss_fn=torch.nn.SmoothL1Loss(),
        verbose=False,
    )
    assert isinstance(best_val, float)
    assert isinstance(best_epoch, int)
    assert len(history["train_loss"]) == 2


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
    # MAPE only counts the second pair: |5-5|/5 = 0
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
