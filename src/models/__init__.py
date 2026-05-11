"""Model implementations.

Empty by design: every other model module (rnn_model, random_forest_model,
gaussian_model) currently still imports TensorFlow at module load time, which
breaks if TF isn't installed. Each consumer should import the specific module
they need, e.g.:

    from src.models.lstm_model import LSTMForecaster
"""
