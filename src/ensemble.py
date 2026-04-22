"""
Ensemble Module - Combines predictions from all models into a unified forecast
This is the core of the system - all models feed into this
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class EnsembleForecaster:
    """
    Unified ensemble that combines predictions from:
    - LSTM
    - RNN
    - Random Forest
    - Gaussian Process
    """
    
    def __init__(self, method='weighted_average', weights=None):
        """
        Args:
            method: 'simple_average', 'weighted_average', or 'meta_learner'
            weights: dict with keys ['lstm', 'rnn', 'random_forest', 'gaussian']
                    If None and method='weighted_average', uses equal weights
        """
        self.method = method
        self.weights = weights or {
            'lstm': 0.25,
            'rnn': 0.25,
            'random_forest': 0.25,
            'gaussian': 0.25
        }
        self.performance_scores = {}
        
    def simple_average(self, predictions_dict):
        """Average all predictions equally"""
        predictions = np.array([
            predictions_dict['lstm'],
            predictions_dict['rnn'],
            predictions_dict['random_forest'],
            predictions_dict['gaussian']
        ])
        return np.mean(predictions, axis=0)
    
    def weighted_average(self, predictions_dict):
        """Weighted average based on model weights"""
        ensemble_pred = (
            self.weights['lstm'] * predictions_dict['lstm'] +
            self.weights['rnn'] * predictions_dict['rnn'] +
            self.weights['random_forest'] * predictions_dict['random_forest'] +
            self.weights['gaussian'] * predictions_dict['gaussian']
        )
        return ensemble_pred
    
    def adapt_weights_from_performance(self, y_true, predictions_dict):
        """
        Adapt weights based on individual model performance (R² score)
        Higher performing models get higher weights
        """
        models = ['lstm', 'rnn', 'random_forest', 'gaussian']
        scores = {}
        
        for model_name in models:
            r2 = r2_score(y_true, predictions_dict[model_name])
            scores[model_name] = max(0, r2)  # Ensure non-negative
        
        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            self.weights = {k: v / total for k, v in scores.items()}
        
        self.performance_scores = scores
        return self.weights
    
    def predict(self, predictions_dict, y_true=None):
        """
        Combine predictions from all models
        
        Args:
            predictions_dict: {
                'lstm': array,
                'rnn': array,
                'random_forest': array,
                'gaussian': array
            }
            y_true: optional, for weight adaptation
        
        Returns:
            ensemble_predictions: array with same shape as individual predictions
        """
        # Verify all predictions have same shape
        shapes = {k: v.shape for k, v in predictions_dict.items()}
        assert len(set(shapes.values())) == 1, \
            f"All models must output same shape. Got: {shapes}"
        
        # Adapt weights if validation data provided
        if y_true is not None and self.method == 'weighted_average':
            self.adapt_weights_from_performance(y_true, predictions_dict)
        
        # Combine predictions
        if self.method == 'simple_average':
            ensemble_pred = self.simple_average(predictions_dict)
        elif self.method == 'weighted_average':
            ensemble_pred = self.weighted_average(predictions_dict)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return ensemble_pred
    
    def evaluate(self, y_true, ensemble_pred, individual_preds=None):
        """Evaluate ensemble performance"""
        mse = mean_squared_error(y_true, ensemble_pred)
        mae = mean_absolute_error(y_true, ensemble_pred)
        r2 = r2_score(y_true, ensemble_pred)
        
        results = {
            'ensemble_mse': mse,
            'ensemble_mae': mae,
            'ensemble_r2': r2,
            'weights': self.weights,
            'individual_scores': self.performance_scores
        }
        
        if individual_preds is not None:
            results['individual_r2'] = {
                model: r2_score(y_true, preds)
                for model, preds in individual_preds.items()
            }
        
        return results


def combine_predictions(predictions_dict, method='weighted_average', weights=None, 
                       y_true=None):
    """
    Simple utility function to combine predictions
    
    Args:
        predictions_dict: dict with model predictions
        method: 'simple_average' or 'weighted_average'
        weights: optional weights dict
        y_true: optional for performance-based weighting
    
    Returns:
        ensemble_predictions: combined forecast
    """
    ensemble = EnsembleForecaster(method=method, weights=weights)
    return ensemble.predict(predictions_dict, y_true=y_true)
