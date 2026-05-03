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
