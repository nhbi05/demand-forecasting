"""
Gaussian Model for demand forecasting (using Gaussian Process)
Input: (n_samples, n_features) - flattened
Output: predictions with shape (n_test_samples,)
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GaussianModel:
    def __init__(self, alpha=1e-6, random_state=42):
        self.alpha = alpha
        self.random_state = random_state
        self.model = None
        
    def build_model(self):
        """Build Gaussian Process with RBF kernel"""
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=True,
            random_state=self.random_state,
            n_restarts_optimizer=10
        )
        return self.model
    
    def train_and_predict(self, X_train, y_train, X_test):
        """
        Train Gaussian Process and make predictions
        
        Args:
            X_train: (n_train, n_features) - flattened
            y_train: (n_train,)
            X_test: (n_test, n_features) - flattened
        
        Returns:
            predictions: (n_test,) - IMPORTANT: must match X_test.shape[0]
        """
        # Build and train
        self.build_model()
        self.model.fit(X_train, y_train)
        
        # Predict (returns mean predictions)
        predictions, _ = self.model.predict(X_test, return_std=True)
        
        return predictions


def train_and_predict_gaussian(X_train, y_train, X_test, alpha=1e-6):
    """
    Utility function for Gaussian Process training and prediction
    
    Returns predictions with same length as X_test
    """
    gp = GaussianModel(alpha=alpha)
    return gp.train_and_predict(X_train, y_train, X_test)
