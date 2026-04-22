"""
LSTM Model for demand forecasting
Input: (batch, seq_length, n_features)
Output: predictions with shape (n_test_samples,)
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class LSTMModel:
    def __init__(self, seq_length, n_features, epochs=50, batch_size=16, verbose=0):
        self.seq_length = seq_length
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        
    def build_model(self):
        """Build LSTM architecture"""
        self.model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.seq_length, self.n_features),
                 return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Sigmoid for [0,1] range
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return self.model
    
    def train_and_predict(self, X_train, y_train, X_test):
        """
        Train LSTM model and make predictions
        
        Args:
            X_train: (n_train_seq, seq_length, n_features)
            y_train: (n_train_seq,)
            X_test: (n_test_seq, seq_length, n_features)
        
        Returns:
            predictions: (n_test_samples,) - IMPORTANT: must match X_test.shape[0]
        """
        # Build and train
        self.build_model()
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=0.1
        )
        
        # Predict
        predictions = self.model.predict(X_test, verbose=0).flatten()
        
        return predictions


def train_and_predict_lstm(X_train, y_train, X_test, seq_length=10, n_features=10, 
                           epochs=50, verbose=0):
    """
    Utility function for LSTM training and prediction
    
    Returns predictions with same length as X_test
    """
    lstm = LSTMModel(seq_length, n_features, epochs=epochs, verbose=verbose)
    return lstm.train_and_predict(X_train, y_train, X_test)
