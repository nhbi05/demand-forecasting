#!/usr/bin/env python3
"""
Random Forest Base Model Training Pipeline
==========================================

Multi-store, multi-item daily demand forecasting using Random Forest regressor.
This is the base layer in a stacking ensemble architecture.

Input: Processed demand data (store_id, item_id, date, qty_sold)
Output: 
  - Trained Random Forest model (pickle)
  - Test predictions CSV (y_true, y_pred) - CRITICAL for ensemble layer
  - Model metrics JSON
  - Feature importance analysis

Author: Demand Forecasting System
Date: 2024
"""

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# ============================================================================
# IMPORTS: Shared Utilities
# ============================================================================

# Add parent directory to path to import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.data_loader import load_processed
from utils.feature_engineering import build_features, train_val_test_split, FEATURE_COLS, TARGET_COL
from utils.metrics import evaluate, save_results


# ============================================================================
# CONFIGURATION
# ============================================================================

# Hyperparameter tuning grid
PARAM_GRID = {
    'n_estimators': [100, 200, 300],      # Number of trees
    'max_depth': [15, 25, 35],            # Tree depth (prevent overfitting)
    'min_samples_leaf': [2, 5, 10]        # Min samples per leaf (regularization)
}

# Output paths
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results' / 'metrics'
MODEL_DIR = Path(__file__).parent
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_DIR / 'random_forest_model.pkl'
PREDICTIONS_SAVE_PATH = RESULTS_DIR / 'random_forest_predictions.csv'
METRICS_SAVE_PATH = Path(__file__).parent.parent.parent / 'results' / 'model_results.json'
TUNING_RESULTS_PATH = MODEL_DIR / 'tuning_results.json'


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """
    Load and validate processed demand data.
    
    Returns:
        pd.DataFrame: Raw demand data with columns (store_id, item_id, date, qty_sold)
    """
    print("\n[1/6] Loading processed demand data...")
    
    try:
        df_raw = load_processed()
        print(f"✓ Data loaded successfully: {df_raw.shape[0]:,} records, {df_raw.shape[1]} columns")
        print(f"  Columns: {list(df_raw.columns)}")
        print(f"  Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")
        print(f"  Unique stores: {df_raw['store_id'].nunique()}")
        print(f"  Unique items: {df_raw['item_id'].nunique()}")
        
        return df_raw
    
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


# ============================================================================
# STEP 2: BUILD FEATURES
# ============================================================================

def engineer_features(df_raw):
    """
    Build engineered features using shared utility.
    
    Creates 13 features:
      - Temporal: year, month, day_of_month, day_of_week, quarter, is_weekend
      - Categorical: store_id, item_id
      - Lag: lag_1, lag_7, lag_14, lag_30
      - Rolling: rolling_mean_7, rolling_std_7, rolling_mean_30
    
    Args:
        df_raw (pd.DataFrame): Raw demand data
        
    Returns:
        pd.DataFrame: Feature matrix with target column
    """
    print("\n[2/6] Building engineered features...")
    
    try:
        df_features = build_features(df_raw)
        print(f"✓ Features built: {df_features.shape[0]:,} samples, {df_features.shape[1]} columns")
        print(f"  Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
        print(f"  Target column: {TARGET_COL}")
        
        # Check for missing values
        missing_count = df_features.isnull().sum().sum()
        if missing_count == 0:
            print(f"✓ No missing values detected")
        else:
            print(f"⚠ {missing_count} missing values detected - will be handled during split")
        
        return df_features
    
    except Exception as e:
        print(f"✗ Error building features: {e}")
        raise


# ============================================================================
# STEP 3: TRAIN-VAL-TEST SPLIT
# ============================================================================

def split_data(df_features):
    """
    Perform temporal train-validation-test split.
    
    Strategy: Temporal ordering (70% train / 15% val / 15% test)
    to prevent data leakage from future to past predictions.
    
    Args:
        df_features (pd.DataFrame): Feature matrix
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n[3/6] Performing temporal train-validation-test split...")
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features)
        
        total_samples = len(df_features)
        print(f"✓ Data split completed:")
        print(f"  Train set: {len(X_train):,} samples ({len(X_train)/total_samples*100:5.1f}%)")
        print(f"  Val set:   {len(X_val):,} samples ({len(X_val)/total_samples*100:5.1f}%)")
        print(f"  Test set:  {len(X_test):,} samples ({len(X_test)/total_samples*100:5.1f}%)")
        print(f"  Total:     {total_samples:,} samples")
        
        # Statistics
        print(f"\n  Target variable statistics:")
        print(f"    Train - Mean: {y_train.mean():8.2f}, Std: {y_train.std():8.2f}")
        print(f"    Val   - Mean: {y_val.mean():8.2f}, Std: {y_val.std():8.2f}")
        print(f"    Test  - Mean: {y_test.mean():8.2f}, Std: {y_test.std():8.2f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    except Exception as e:
        print(f"✗ Error in train-val-test split: {e}")
        raise


# ============================================================================
# STEP 4: HYPERPARAMETER TUNING
# ============================================================================

def tune_hyperparameters(X_train, X_val, y_train, y_val):
    """
    Perform hyperparameter tuning using GridSearchCV on validation set.
    
    Tunes:
      - n_estimators: [100, 200, 300] - number of trees
      - max_depth: [15, 25, 35] - tree depth
      - min_samples_leaf: [2, 5, 10] - minimum samples per leaf
    
    Total combinations: 27 hyperparameter sets
    Evaluation: 3-fold cross-validation, scoring on MAE
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        y_train (pd.Series): Training target
        y_val (pd.Series): Validation target
        
    Returns:
        tuple: (best_params, best_val_mae, grid_search_results)
    """
    print("\n[4/6] Tuning hyperparameters using GridSearchCV...")
    
    n_combinations = np.prod([len(v) for v in PARAM_GRID.values()])
    print(f"  Grid size: {n_combinations} combinations × 3-fold CV = {n_combinations * 3} model fits")
    print(f"  Scoring metric: Negative MAE (Mean Absolute Error)")
    print(f"  Validation set size: {len(X_val):,} samples")
    print(f"  This may take 1-2 minutes...\n")
    
    try:
        # Initialize base Random Forest
        rf_base = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        
        # Setup GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=PARAM_GRID,
            scoring='neg_mean_absolute_error',  # MAE scoring (neg for maximization)
            cv=3,                                # 3-fold cross-validation
            n_jobs=-1,                          # Parallel processing
            verbose=0
        )
        
        # Run hyperparameter tuning
        start_time = time.time()
        grid_search.fit(X_val, y_val)
        elapsed = time.time() - start_time
        
        # Extract best results
        best_params = grid_search.best_params_
        best_val_mae = -grid_search.best_score_  # Convert back to positive MAE
        
        print(f"✓ Hyperparameter tuning completed in {elapsed:.2f} seconds")
        print(f"\n  Best hyperparameters found:")
        for param, value in best_params.items():
            print(f"    {param:20s}: {value}")
        print(f"\n  Best validation MAE: {best_val_mae:.4f}")
        
        # Store all tuning results
        tuning_results = {
            'best_params': best_params,
            'best_validation_mae': float(best_val_mae),
            'all_results': pd.DataFrame(grid_search.cv_results_).to_dict()
        }
        
        return best_params, best_val_mae, tuning_results
    
    except Exception as e:
        print(f"✗ Error during hyperparameter tuning: {e}")
        raise


# ============================================================================
# STEP 5: TRAIN FINAL MODEL
# ============================================================================

def train_final_model(X_train, X_val, y_train, y_val, best_params):
    """
    Train final Random Forest model on combined training + validation set
    using best hyperparameters from tuning.
    
    This maximizes training data while using hyperparameters optimized on
    independent validation set (prevents overfitting to training set).
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        y_train (pd.Series): Training target
        y_val (pd.Series): Validation target
        best_params (dict): Best hyperparameters from tuning
        
    Returns:
        RandomForestRegressor: Trained final model
    """
    print("\n[5/6] Training final model with best hyperparameters...")
    
    try:
        # Combine train + val for final training
        X_train_full = pd.concat([X_train, X_val], ignore_index=False)
        y_train_full = pd.concat([y_train, y_val], ignore_index=False)
        
        print(f"  Training on combined set: {len(X_train_full):,} samples")
        print(f"  Hyperparameters: {best_params}")
        
        # Create and train final model
        start_time = time.time()
        
        rf_final = RandomForestRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        rf_final.fit(X_train_full, y_train_full)
        
        elapsed = time.time() - start_time
        print(f"✓ Final model trained in {elapsed:.2f} seconds")
        print(f"  Model type: RandomForestRegressor")
        print(f"  Total trees: {rf_final.n_estimators}")
        print(f"  Total features: {rf_final.n_features_in_}")
        
        return rf_final
    
    except Exception as e:
        print(f"✗ Error training final model: {e}")
        raise


# ============================================================================
# STEP 6: EVALUATE & SAVE
# ============================================================================

def evaluate_and_save(model, X_test, y_test, best_params, best_val_mae):
    """
    Evaluate model on test set and save all outputs.
    
    Outputs:
      1. Test predictions CSV (CRITICAL for ensemble) - results/metrics/random_forest_predictions.csv
      2. Trained model pickle - models/random_forest/random_forest_model.pkl
      3. Model metrics JSON - results/model_results.json
      4. Tuning results JSON - models/random_forest/tuning_results.json
      5. Feature importances printed and saved
    
    Args:
        model (RandomForestRegressor): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        best_params (dict): Best hyperparameters used
        best_val_mae (float): Best validation MAE from tuning
    """
    print("\n[6/6] Evaluating model and saving artifacts...")
    
    try:
        # =====================================================================
        # PREDICTIONS & METRICS
        # =====================================================================
        
        # Generate predictions on test set
        y_pred_test = model.predict(X_test)
        
        # Evaluate using shared utility
        results = evaluate(y_test, y_pred_test, model_name="Random Forest")
        
        print(f"\n✓ Test Set Performance:")
        print(f"  MAE  (Mean Absolute Error):     {results['mae']:10.4f} units")
        print(f"  RMSE (Root Mean Squared Error): {results['rmse']:10.4f} units")
        print(f"  MAPE (Mean Absolute % Error):   {results['mape']:10.4f}%")
        print(f"  R²   (Coefficient of Determination): {results['r2']:10.4f}")
        
        # Calculate residuals
        residuals = y_test - y_pred_test
        print(f"\n  Residual Statistics:")
        print(f"    Mean:       {residuals.mean():10.4f}")
        print(f"    Std Dev:    {residuals.std():10.4f}")
        print(f"    Min:        {residuals.min():10.4f}")
        print(f"    Max:        {residuals.max():10.4f}")
        
        # =====================================================================
        # SAVE PREDICTIONS (CRITICAL FOR ENSEMBLE)
        # =====================================================================
        
        predictions_df = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': y_pred_test
        })
        
        predictions_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)
        print(f"\n✓ Predictions saved to: {PREDICTIONS_SAVE_PATH}")
        print(f"  Shape: {predictions_df.shape}")
        print(f"  Columns: ['y_true', 'y_pred']")
        print(f"  Note: These predictions will be used as input features for ensemble layer")
        
        # =====================================================================
        # SAVE MODEL
        # =====================================================================
        
        joblib.dump(model, str(MODEL_SAVE_PATH))
        print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")
        
        # =====================================================================
        # SAVE METRICS
        # =====================================================================
        
        save_results(results)
        print(f"✓ Metrics saved to: {METRICS_SAVE_PATH}")
        
        # =====================================================================
        # SAVE TUNING RESULTS
        # =====================================================================
        
        tuning_results = {
            'best_params': best_params,
            'best_validation_mae': float(best_val_mae),
            'test_metrics': {
                'mae': float(results['mae']),
                'rmse': float(results['rmse']),
                'mape': float(results['mape']),
                'r2': float(results['r2'])
            },
            'data_info': {
                'n_test_samples': int(len(X_test)),
                'n_features': int(len(FEATURE_COLS)),
                'features': FEATURE_COLS
            }
        }
        
        with open(TUNING_RESULTS_PATH, 'w') as f:
            json.dump(tuning_results, f, indent=2)
        print(f"✓ Tuning results saved to: {TUNING_RESULTS_PATH}")
        
        # =====================================================================
        # FEATURE IMPORTANCES
        # =====================================================================
        
        print_feature_importances(model)
        
        return y_pred_test
    
    except Exception as e:
        print(f"✗ Error during evaluation and saving: {e}")
        raise


def print_feature_importances(model):
    """
    Print top 15 feature importances from trained model.
    
    Args:
        model (RandomForestRegressor): Trained model
    """
    print(f"\n{'='*60}")
    print(f"{'Feature Importance Analysis (Top 15)':^60}")
    print(f"{'='*60}\n")
    
    try:
        # Get importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]  # Sort descending
        
        # Print top 15
        print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12} {'%':<8}")
        print("-" * 52)
        
        top_n = min(15, len(FEATURE_COLS))
        for rank in range(top_n):
            feature_idx = indices[rank]
            feature_name = FEATURE_COLS[feature_idx]
            importance = importances[feature_idx]
            percentage = (importance / importances.sum()) * 100
            
            print(f"{rank+1:<6} {feature_name:<25} {importance:<12.6f} {percentage:<8.2f}%")
        
        cumsum = importances[indices[:top_n]].sum() / importances.sum() * 100
        print(f"\nCumulative importance (top 15): {cumsum:.2f}%")
    
    except Exception as e:
        print(f"⚠ Error printing feature importances: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline:
    1. Load processed data
    2. Build engineered features
    3. Temporal train-val-test split
    4. Hyperparameter tuning on validation set
    5. Train final model on train+val with best hyperparameters
    6. Evaluate on test set and save artifacts
    """
    
    print("\n" + "="*70)
    print("RANDOM FOREST BASE MODEL TRAINING PIPELINE".center(70))
    print("Demand Forecasting - Multi-store, Multi-item".center(70))
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Step 1: Load data
        df_raw = load_data()
        
        # Step 2: Build features
        df_features = engineer_features(df_raw)
        
        # Step 3: Train-val-test split
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_features)
        
        # Step 4: Hyperparameter tuning
        best_params, best_val_mae, tuning_results = tune_hyperparameters(
            X_train, X_val, y_train, y_val
        )
        
        # Step 5: Train final model
        rf_final = train_final_model(X_train, X_val, y_train, y_val, best_params)
        
        # Step 6: Evaluate and save
        y_pred = evaluate_and_save(rf_final, X_test, y_test, best_params, best_val_mae)
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ PIPELINE COMPLETED SUCCESSFULLY".center(70))
        print(f"{'='*70}")
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        print(f"\nKey output files:")
        print(f"  • {PREDICTIONS_SAVE_PATH}")
        print(f"  • {MODEL_SAVE_PATH}")
        print(f"  • {METRICS_SAVE_PATH}")
        print(f"\nEnsemble Integration:")
        print(f"  Load {PREDICTIONS_SAVE_PATH} as input features for LSTM/RNN/GRU/NFTU")
        print(f"\n")
    
    except Exception as e:
        print(f"\n✗ PIPELINE FAILED: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
