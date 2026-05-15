"""
Ensemble Module - Combines predictions from all models into a unified forecast
This is the core of the system - all models feed into this
"""

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


class EnsembleForecaster:

    def __init__(self, method='simple_average', weights=None):

        """
        method:
            - simple_average
            - weighted_average
        """

        self.method = method

        self.weights = weights or {
            'lstm': 0.25,
            'gru': 0.25,
            'rf': 0.25,
            'gaussian': 0.25
        }

    def simple_average(self, predictions_dict):

        predictions = np.array([
            predictions_dict['lstm'],
            predictions_dict['gru'],
            predictions_dict['rf'],
            predictions_dict['gaussian']
        ])

        return np.mean(predictions, axis=0)

    def weighted_average(self, predictions_dict):

        ensemble_pred = (
            self.weights['lstm'] * predictions_dict['lstm'] +
            self.weights['gru'] * predictions_dict['gru'] +
            self.weights['rf'] * predictions_dict['rf'] +
            self.weights['gaussian'] * predictions_dict['gaussian']
        )

        return ensemble_pred

    def adapt_weights(self, y_true, predictions_dict):

        """
        Adapt weights using R² score
        Better models receive larger weights
        """

        scores = {}

        for model_name, preds in predictions_dict.items():

            r2 = r2_score(y_true, preds)

            scores[model_name] = max(r2, 0)

        total = sum(scores.values())

        if total > 0:

            self.weights = {
                k: v / total
                for k, v in scores.items()
            }

        return self.weights

    def predict(self, predictions_dict, y_true=None):

        shapes = [v.shape for v in predictions_dict.values()]

        if len(set(shapes)) != 1:
            raise ValueError(
                f"Prediction shapes do not match: {shapes}"
            )

        if self.method == 'weighted_average' and y_true is not None:
            self.adapt_weights(y_true, predictions_dict)

        if self.method == 'simple_average':
            return self.simple_average(predictions_dict)

        elif self.method == 'weighted_average':
            return self.weighted_average(predictions_dict)

        else:
            raise ValueError("Invalid ensemble method")

    def evaluate(self, y_true, y_pred):

        mse = mean_squared_error(y_true, y_pred)

        mae = mean_absolute_error(y_true, y_pred)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_true, y_pred)

        mape = (
            mean_absolute_percentage_error(
                y_true,
                y_pred
            ) * 100
        )

        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }