"""
Random Forest Model for demand forecasting
Input: (n_samples, n_features) - flattened
Output: predictions with shape (n_test_samples,)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        
    def build_model(self):
        """Build Random Forest"""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        return self.model
    
    def train_and_predict(self, X_train, y_train, X_test):
        """
        Train Random Forest and make predictions
        
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
        
        # Predict
        predictions = self.model.predict(X_test)
        
        return predictions


def train_and_predict_random_forest(X_train, y_train, X_test, 
                                    n_estimators=100, max_depth=15):
    """
    Utility function for Random Forest training and prediction
    
    Returns predictions with same length as X_test
    """
    rf = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth)
    return rf.train_and_predict(X_train, y_train, X_test)
