"""
Feature engineering utilities for demand forecasting
"""

import pandas as pd
import numpy as np


# Define feature and target columns globally
TARGET_COL = 'qty_sold'

FEATURE_COLS = [
    # Temporal features (6)
    'year',
    'month',
    'day_of_month',
    'day_of_week',
    'quarter',
    'is_weekend',
    
    # Categorical features (2)
    'store_id',
    'item_id',
    
    # Lag features (4) - previous demand
    'lag_1',
    'lag_7',
    'lag_14',
    'lag_30',
    
    # Rolling statistics (3)
    'rolling_mean_7',
    'rolling_std_7',
    'rolling_mean_30',
]


def build_features(df):
    """
    Build engineered features from raw demand data.
    
    Args:
        df: DataFrame with columns [store_id, item_id, date, qty_sold]
    
    Returns:
        pd.DataFrame: DataFrame with engineered features + target column
                      Total: 13 features + 1 target = 14 columns
    
    Features created:
        Temporal (6): year, month, day_of_month, day_of_week, quarter, is_weekend
        Categorical (2): store_id, item_id
        Lag (4): lag_1, lag_7, lag_14, lag_30
        Rolling (3): rolling_mean_7, rolling_std_7, rolling_mean_30
    """
    
    df = df.copy()
    
    # --- TEMPORAL FEATURES ---
    # Extract temporal components from date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Saturday or Sunday
    
    # --- LAG FEATURES ---
    # Create lagged demand values (previous days' quantities)
    # For each store-item combination, lag the quantity sold
    df = df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)
    
    for lag_days in [1, 7, 14, 30]:
        df[f'lag_{lag_days}'] = df.groupby(['store_id', 'item_id'])[TARGET_COL].shift(lag_days)
    
    # --- ROLLING STATISTICS ---
    # Calculate rolling average and std dev over different windows
    for col in [f'rolling_mean_{window}' for window in [7, 30]] + [f'rolling_std_{window}' for window in [7]]:
        window = int(col.split('_')[-1])
        stat_type = 'mean' if 'mean' in col else 'std'
        
        if stat_type == 'mean':
            df[col] = df.groupby(['store_id', 'item_id'])[TARGET_COL].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        else:  # std
            df[col] = df.groupby(['store_id', 'item_id'])[TARGET_COL].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
    
    # Fill NaN values in lag and rolling features with 0 or group mean
    lag_cols = [col for col in df.columns if 'lag_' in col]
    rolling_cols = [col for col in df.columns if 'rolling_' in col]
    
    for col in lag_cols + rolling_cols:
        # Fill with group mean first, then with 0
        df[col] = df.groupby(['store_id', 'item_id'])[col].transform(
            lambda x: x.fillna(x.mean())
        )
        df[col].fillna(0, inplace=True)
    
    # Select only required columns (features + target)
    selected_cols = FEATURE_COLS + [TARGET_COL]
    df_final = df[selected_cols].copy()
    
    # Final data quality checks
    assert not df_final.isnull().any().any(), "NaN values found in features"
    assert len(df_final) > 0, "No records after feature engineering"
    
    return df_final


def train_val_test_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Perform temporal train-validation-test split to prevent data leakage.
    
    Splits data chronologically to ensure:
    - Training set comes before validation set
    - Validation set comes before test set
    - No future data leakage into training
    
    Args:
        df: DataFrame with engineered features (must have features + target)
        train_ratio: Fraction for training (default: 0.70)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
    
    Returns:
        Tuple[X_train, X_val, X_test, y_train, y_val, y_test]:
            - X_* : Feature DataFrames (excluding target)
            - y_* : Target Series (qty_sold)
    
    Raises:
        AssertionError: If ratios don't sum to 1.0
    """
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split temporally (chronological order)
    train_indices = slice(0, train_end)
    val_indices = slice(train_end, val_end)
    test_indices = slice(val_end, n)
    
    # Extract features and target
    X_train = df.iloc[train_indices][FEATURE_COLS]
    y_train = df.iloc[train_indices][TARGET_COL]
    
    X_val = df.iloc[val_indices][FEATURE_COLS]
    y_val = df.iloc[val_indices][TARGET_COL]
    
    X_test = df.iloc[test_indices][FEATURE_COLS]
    y_test = df.iloc[test_indices][TARGET_COL]
    
    # Verify splits
    assert len(X_train) > 0, "Empty training set"
    assert len(X_val) > 0, "Empty validation set"
    assert len(X_test) > 0, "Empty test set"
    assert len(X_train) + len(X_val) + len(X_test) == n, "Split mismatch"
    
    return X_train, X_val, X_test, y_train, y_val, y_test
