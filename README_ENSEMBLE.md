# Demand Forecasting System - Unified Ensemble

A professional, production-ready demand forecasting system that combines **LSTM**, **RNN**, **Random Forest**, and **Gaussian Process** models into a unified ensemble for accurate predictions.

## 🎯 Key Features

✅ **Unified Ensemble Architecture** - All 4 models work together, not independently  
✅ **Identical Input/Output** - All models trained on same data, same prediction shape  
✅ **Multiple Ensemble Methods** - Simple averaging, weighted averaging, performance-based adaptation  
✅ **Comprehensive Analysis** - Individual model metrics + ensemble performance comparison  
✅ **Production Ready** - Clean module structure, well-documented, scalable  

## 📁 Project Structure

```
demand-forecasting/
├── notebooks/                    # Jupyter notebooks for experimentation
│   ├── 01_EDA.ipynb              # Exploratory data analysis
│   ├── 02_LSTM.ipynb             # LSTM model training & prediction
│   ├── 03_RNN.ipynb              # RNN model training & prediction
│   ├── 04_Random_Forest.ipynb    # Random Forest training & prediction
│   ├── 05_Gaussian.ipynb         # Gaussian Process training & prediction
│   └── 06_Ensemble.ipynb         # Unified ensemble system
│
├── src/                          # Reusable Python modules
│   ├── __init__.py
│   ├── preprocessing.py          # Data loading & unified preprocessing
│   ├── ensemble.py               # Ensemble combination logic
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py         # LSTM implementation
│       ├── rnn_model.py          # RNN implementation
│       ├── random_forest_model.py # Random Forest implementation
│       └── gaussian_model.py     # Gaussian Process implementation
│
├── data/                         # Data storage (optional)
├── scripts/                      # Standalone scripts (future)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks in Order

The system is designed to be run sequentially:

1. **01_EDA.ipynb** - Understand your data
2. **02_LSTM.ipynb** - Train LSTM model
3. **03_RNN.ipynb** - Train RNN model
4. **04_Random_Forest.ipynb** - Train Random Forest
5. **05_Gaussian.ipynb** - Train Gaussian Process
6. **06_Ensemble.ipynb** - Combine all predictions

Each notebook generates predictions with **identical shape** and **same scale**.

### 3. Use in Python

```python
from src import preprocessing
from src.models import lstm_model, rnn_model
from src.ensemble import EnsembleForecaster

# Load and preprocess data
X, y = preprocessing.load_synthetic_data()
data = preprocessing.preprocess_data(X, y)

# Get predictions from all models
lstm_pred = lstm_model.train_and_predict(
    data['X_train_seq'], data['y_train_seq'], data['X_test_seq']
)
# ... train other models similarly

# Combine into ensemble
ensemble = EnsembleForecaster(method='weighted_average')
ensemble_pred = ensemble.predict(predictions_dict)
```

## 🔧 Core Modules

### `preprocessing.py`
- **`load_synthetic_data()`** - Generate or load demand data
- **`preprocess_data()`** - Unified preprocessing (scaling, sequencing, splitting)
- **`inverse_scale_predictions()`** - Convert predictions back to original scale

### Model Modules
Each model implements the same interface:
```python
train_and_predict(X_train, y_train, X_test) -> predictions
```

**LSTM & RNN** (deep learning):
- Input: (batch, seq_length, n_features)
- Output: (n_test_samples,)
- Training: 50 epochs, early stopping

**Random Forest & Gaussian** (traditional ML):
- Input: (n_samples, n_features) - flattened
- Output: (n_test_samples,)
- Both standardized to work with same data format

### `ensemble.py`
- **`EnsembleForecaster`** - Combines model predictions
  - Simple averaging: Equal weights
  - Weighted averaging: Based on R² scores
  - Performance adaptation: Dynamic weight adjustment

## 📊 Example Output

```
INDIVIDUAL MODEL PERFORMANCE
========================================

LSTM:
  RMSE: 0.0542  |  MAE: 0.0385  |  R²: 0.8234  |  MAPE: 5.23%

RNN:
  RMSE: 0.0589  |  MAE: 0.0428  |  R²: 0.7891  |  MAPE: 5.87%

Random Forest:
  RMSE: 0.0478  |  MAE: 0.0352  |  R²: 0.8512  |  MAPE: 4.76%

Gaussian:
  RMSE: 0.0501  |  MAE: 0.0368  |  R²: 0.8367  |  MAPE: 4.95%


ENSEMBLE PERFORMANCE
========================================

Simple Average Ensemble:
  RMSE: 0.0425  |  MAE: 0.0298  |  R²: 0.8756  |  MAPE: 4.03%
  Improvement: +2.86% vs best individual model

Weighted Average Ensemble:
  RMSE: 0.0418  |  MAE: 0.0289  |  R²: 0.8823  |  MAPE: 3.91%
  Improvement: +3.65% vs best individual model
  
Weights:
  LSTM: 0.22
  RNN: 0.18
  Random Forest: 0.35
  Gaussian: 0.25
```

## 🎓 How the Ensemble Works

### Data Flow
```
Raw Data
   ↓
[Unified Preprocessing]
   ├─→ (scaled) → LSTM → predictions ┐
   ├─→ (scaled) → RNN → predictions  ├─→ [Ensemble] → Final Forecast
   ├─→ (scaled) → RF → predictions   │
   └─→ (scaled) → GP → predictions  ┘
```

### Prediction Combination
**Simple Average:**
```
ensemble_pred = (lstm + rnn + rf + gp) / 4
```

**Weighted Average:**
```
weights = {
    'lstm': 0.22,
    'rnn': 0.18,
    'random_forest': 0.35,
    'gaussian': 0.25
}
ensemble_pred = Σ(weight_i × pred_i)
```

Weights are calculated based on individual model R² scores on validation data.

## 🔍 Key Design Principles

1. **Unified Preprocessing** - All models receive identical scaled data
2. **Identical Output Shape** - All predictions have shape (n_test_samples,)
3. **Transparent Combination** - Ensemble logic is explicit and interpretable
4. **Performance Tracking** - Individual + ensemble metrics compared
5. **Scalability** - Easy to add new models to the ensemble

## 📈 Extending the System

### Add a New Model

1. Create `src/models/new_model.py`:
```python
def train_and_predict(X_train, y_train, X_test):
    # Train on X_train, y_train
    model = ...
    model.fit(X_train, y_train)
    # Return predictions with shape (len(X_test),)
    return model.predict(X_test)
```

2. Update `06_Ensemble.ipynb` to include the new model:
```python
new_pred_scaled = new_model.train_and_predict(...)
predictions_dict['new_model'] = new_pred_scaled
```

3. The ensemble automatically adapts weights!

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras (LSTM, RNN)
- **Traditional ML**: scikit-learn (Random Forest, Gaussian Process)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Notebooks**: Jupyter

## 📝 License

MIT License - Free for academic and commercial use

## 📞 Support

For issues or questions about the unified ensemble system, check:
- Individual model notebooks for model-specific details
- `06_Ensemble.ipynb` for ensemble logic
- `src/preprocessing.py` for data handling

---

**Built with ❤️ for production demand forecasting**
