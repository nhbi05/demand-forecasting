"""
Evaluation metrics and result saving utilities for demand forecasting
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred, model_name='Model'):
    """
    Calculate comprehensive evaluation metrics for predictions.
    
    Args:
        y_true: Actual values (array-like)
        y_pred: Predicted values (array-like)
        model_name: Name of the model (for logging)
    
    Returns:
        dict: Dictionary containing:
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - mape: Mean Absolute Percentage Error
            - r2: R² Score
            - model_name: Model identifier
    
    Metrics:
        MAE: Average absolute difference (same units as target)
        RMSE: Penalizes large errors more heavily
        MAPE: Percentage-based error (robust to scale)
        R²: Proportion of variance explained (0-1 scale)
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE: Avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf
    
    r2 = r2_score(y_true, y_pred)
    
    results = {
        'model_name': model_name,
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
    }
    
    return results


def save_results(results, output_path='results/model_results.json'):
    """
    Save evaluation metrics to JSON file.
    
    Args:
        results: Dictionary of metrics from evaluate()
        output_path: Path to save JSON file
    
    Returns:
        None
    """
    
    # Create parent directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read existing results if file exists
    if Path(output_path).exists():
        try:
            with open(output_path, 'r') as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, IOError):
            all_results = []
    else:
        all_results = []
    
    # Append new results
    if isinstance(all_results, dict):
        all_results = [all_results]
    
    all_results.append(results)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\nMetrics saved to {output_path}")
    print(f"  Model: {results.get('model_name', 'Unknown')}")
    print(f"  MAE:  {results.get('mae', 'N/A'):.4f}")
    print(f"  RMSE: {results.get('rmse', 'N/A'):.4f}")
    print(f"  MAPE: {results.get('mape', 'N/A'):.2f}%")
    print(f"  R²:   {results.get('r2', 'N/A'):.4f}")
