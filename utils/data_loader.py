"""
Data loading utilities for demand forecasting
Loads processed data with columns: store_id, item_id, date, qty_sold
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_processed():
    """
    Load preprocessed demand data.
    
    Returns:
        pd.DataFrame: DataFrame with columns [store_id, item_id, date, qty_sold]
                      - store_id: Store identifier (categorical)
                      - item_id: Item identifier (categorical)
                      - date: Transaction date (datetime)
                      - qty_sold: Daily quantity sold (numeric, target variable)
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    
    # Define potential data paths (check multiple locations)
    data_paths = [
        Path('data/processed_demand.csv'),
        Path('data/demand_data.csv'),
        Path('processed_demand.csv'),
    ]
    
    # Try to load from available path
    data_path = None
    for path in data_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        # Create synthetic data for demonstration if no file exists
        print("⚠ No processed data file found. Generating synthetic demand data for demonstration...")
        df = _generate_synthetic_data()
        return df
    
    # Load the data
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Error loading data from {data_path}: {e}")
    
    # Validate required columns
    required_cols = ['store_id', 'item_id', 'date', 'qty_sold']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date for temporal operations
    df = df.sort_values('date').reset_index(drop=True)
    
    # Validate data types and ranges
    if df['qty_sold'].dtype not in ['int64', 'float64']:
        df['qty_sold'] = pd.to_numeric(df['qty_sold'], errors='coerce')
    
    if df['qty_sold'].isnull().any():
        print(f"⚠ Found {df['qty_sold'].isnull().sum()} null values in qty_sold. Removing...")
        df = df.dropna(subset=['qty_sold'])
    
    # Remove negative quantities (invalid demand)
    if (df['qty_sold'] < 0).any():
        print(f"⚠ Removing {(df['qty_sold'] < 0).sum()} rows with negative quantities")
        df = df[df['qty_sold'] >= 0]
    
    return df


def _generate_synthetic_data(n_records=10000, n_stores=5, n_items=20, seed=42):
    """
    Generate synthetic demand data for testing/demonstration.
    
    Args:
        n_records: Number of records to generate
        n_stores: Number of unique stores
        n_items: Number of unique items
        seed: Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Synthetic demand data
    """
    np.random.seed(seed)
    
    # Generate date range
    start_date = pd.Timestamp('2022-01-01')
    dates = pd.date_range(start=start_date, periods=n_records, freq='D')
    
    # Generate store and item IDs with realistic distribution
    store_ids = np.random.choice(range(1, n_stores + 1), size=n_records)
    item_ids = np.random.choice(range(1, n_items + 1), size=n_records)
    
    # Generate demand with realistic patterns
    # Base demand with weekly seasonality and trend
    base_demand = 50 + np.arange(n_records) * 0.01  # Slight upward trend
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_records) / 7)  # Weekly pattern
    noise = np.random.normal(0, 10, n_records)  # Random noise
    qty_sold = np.maximum(base_demand + seasonality + noise, 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'store_id': store_ids,
        'item_id': item_ids,
        'date': dates,
        'qty_sold': qty_sold
    })
    
    print(f"✓ Generated {len(df):,} synthetic demand records")
    print(f"  - Stores: {df['store_id'].nunique()}")
    print(f"  - Items: {df['item_id'].nunique()}")
    print(f"  - Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  - Avg daily demand: {df['qty_sold'].mean():.2f} units")
    
    return df
