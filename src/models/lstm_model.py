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

import copy

import numpy as np
import torch
import torch.nn as nn
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


class LSTMForecaster(nn.Module):
    """Single global LSTM with a learned product embedding.

    Inputs (per batch):
        past_qty: (B, L, 1)         - scaled daily quantity for the past L days
        calendar: (B, L, n_cal)     - sin/cos cyclical features
        prod_idx: (B,)              - int64, 0..n_products-1

    Output:
        forecast: (B, horizon)      - direct multi-output prediction
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


def train_model(
    model: LSTMForecaster,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
    verbose: bool = True,
    weight_decay: float = 0.0,
    loss_fn: torch.nn.Module | None = None,
):
    """Train the LSTM with Adam + early stopping on val loss.

    Args:
        weight_decay: L2 regularization on Adam (default 0.0 — disabled).
            Use 1e-4 for mild regularization that works well with this model.
        loss_fn: torch loss module instance. Defaults to MSELoss. Pass
            `torch.nn.SmoothL1Loss()` (Huber) or `torch.nn.L1Loss()` (MAE)
            to be more robust to spike-day outliers.

    Returns:
        (best_model, best_val_loss, best_epoch, history)

    `best_epoch` is the 1-indexed epoch number at which `best_val_loss`
    was achieved. Use this — not `len(history["val_loss"])` — when retraining
    on train+val combined, otherwise you'll overshoot by the patience window.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_fn is None:
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
