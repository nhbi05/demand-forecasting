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
