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
