"""
Data preprocessing module - ensures all models receive identical input
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_synthetic_data(n_samples=500, n_features=10, random_state=42):
    """
    Load or generate synthetic demand data
    Returns: X, y with consistent shape for all models
    """
    np.random.seed(random_state)
    
    # Generate synthetic time-series demand data
    X = np.random.randn(n_samples, n_features)
    # Create some trend
    trend = np.linspace(100, 200, n_samples)
    y = trend + np.random.randn(n_samples) * 10
    
    return X, y


def preprocess_data(X, y, test_size=0.2, seq_length=10, random_state=42):
    """
    Unified preprocessing for all models
    
    Args:
        X: input features (n_samples, n_features)
        y: target values (n_samples,)
        test_size: fraction for test set
        seq_length: sequence length for LSTM/RNN
        random_state: for reproducibility
    
    Returns:
        dict with standardized outputs for all models
    """
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features to [0, 1]
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target to [0, 1]
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.fit_transform(y_test.reshape(-1, 1)).flatten()
    
    # Create sequences for deep learning models
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
    
    return {
        # For traditional ML models (Random Forest, Gaussian)
        'X_train_flat': X_train_scaled,
        'X_test_flat': X_test_scaled,
        'y_train_flat': y_train_scaled,
        'y_test_flat': y_test_scaled,
        
        # For deep learning models (LSTM, RNN)
        'X_train_seq': X_train_seq,
        'X_test_seq': X_test_seq,
        'y_train_seq': y_train_seq,
        'y_test_seq': y_test_seq,
        
        # Scalers for inverse transformation
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        
        # Metadata
        'n_features': X.shape[1],
        'seq_length': seq_length,
        'test_indices': len(y_test_scaled)
    }


def inverse_scale_predictions(predictions, scaler_y):
    """Inverse scale predictions back to original range"""
    return scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
