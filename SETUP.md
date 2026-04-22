# Setup Instructions

## Prerequisites

- Python 3.8+
- pip or conda

## Installation

### Option 1: Using pip

```bash
# Install all dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create virtual environment
conda create -n demand-forecasting python=3.10

# Activate environment
conda activate demand-forecasting

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "import tensorflow; import torch; import sklearn; print('✓ All packages installed')"
```

## Run the Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to `notebooks/` folder

3. Open and run notebooks in order:
   - `01_EDA.ipynb`
   - `02_LSTM.ipynb`
   - `03_RNN.ipynb`
   - `04_Random_Forest.ipynb`
   - `05_Gaussian.ipynb`
   - `06_Ensemble.ipynb`

## Troubleshooting

**TensorFlow GPU Issues:**
```bash
# For CPU-only (faster install):
pip install tensorflow-cpu
```

**Memory Issues:**
- Reduce batch size in notebooks (default: 16)
- Reduce number of epochs (default: 50)

**Port Already in Use:**
```bash
jupyter notebook --port 8889
```

---

For more details, see README_ENSEMBLE.md
