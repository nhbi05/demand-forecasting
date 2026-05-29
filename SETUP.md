# Setup

Environment setup, data preparation, and how to run the system end-to-end. For the no-training path, see [QUICK_START.md](QUICK_START.md).

## Prerequisites

- Python 3.10 or 3.11 (the pinned versions in `requirements.txt` do not support 3.12+)
- `pip` or `conda`
- A CUDA-capable GPU is recommended for full LSTM/GRU training but not required. CPU is fine for the ensemble notebook and for small experiments.

## Install dependencies

Using `pip`:

```bash
pip install -r requirements.txt
```

Using `conda`:

```bash
conda create -n demand-forecasting python=3.10
conda activate demand-forecasting
pip install -r requirements.txt
```

## Verify the install

```bash
python -c "import numpy, pandas, sklearn, torch, matplotlib; print('OK')"
```

## Data preparation

The raw `sales.csv` (~7.4M rows) is gitignored. Place it at the project root, then materialize the shared processed artifacts:

```bash
python -c "from src.preprocessing import prepare_data; prepare_data('sales.csv')"
```

This writes `data/processed/`:

```
data/processed/
├── daily.csv               # Aggregated daily demand: 100 products × 761 days
├── splits.json             # train / val / test date boundaries
├── selected_products.json  # The 100 product IDs every model must forecast
├── scalers.pkl             # Per-product MinMax scalers for inverse-scaling
├── feature_scalers.pkl     # Per-product scalers for every continuous feature
├── feature_columns.json    # Shared feature column lists
├── outlier_caps.json       # Per-product 99.5th percentile caps
└── metadata.json           # Run parameters and dataset summary
```

These artifacts are the single source of truth for every model. Notebooks should load them via:

```python
from src.preprocessing import load_artifacts
artifacts = load_artifacts("data/processed")
```

Read [docs/ensemble-contract.md](docs/ensemble-contract.md) before training or modifying a model.

## Running the notebooks

```bash
jupyter notebook
```

<<<<<<< HEAD
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
=======
Open `notebooks/` and run in order:

1. `01_EDA.ipynb` — Exploratory data analysis
2. `02_LSTM.ipynb` — PyTorch LSTM training and predictions
3. `03_RNN.ipynb` — PyTorch GRU training and predictions
4. `04_Random_Forest.ipynb` — sklearn Random Forest
5. `05_Gaussian_(1).ipynb` — sklearn Gaussian Process, one model per product
6. `06_Ensemble.ipynb` — Combines all four models

Each model writes its predictions to `data/predictions/{model}_test_predictions.csv`. The ensemble notebook reads those CSVs — it does not re-train.

## Training cost

| Notebook | Typical wall-clock | Notes |
|---|---|---|
| `02_LSTM.ipynb` | Minutes on GPU, hours on CPU | Colab GPU recommended. Weights saved to `models/lstm_final.pt`. |
| `03_RNN.ipynb` | Similar to LSTM | Weights saved to `models/gru_final.pt`. |
| `04_Random_Forest.ipynb` | Under a minute | CPU only. |
| `05_Gaussian_(1).ipynb` | A few minutes | One per-product GP × 100 products; CPU. |
| `06_Ensemble.ipynb` | Seconds | Pure CSV math, no model loaded. |

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run notebooks from the repo root, or keep the `sys.path.append("..")` line at the top of each notebook.

**Jupyter port already in use**
>>>>>>> 70e3b2b08b7d0b2630531e85edc9e2083f627093
```bash
jupyter notebook --port 8889
```

<<<<<<< HEAD
---

For more details, see README_ENSEMBLE.md
=======
**Out-of-memory during LSTM/GRU training**
Reduce `batch_size` or `epochs` in the relevant notebook cell. Defaults are tuned for a 16 GB GPU.
>>>>>>> 70e3b2b08b7d0b2630531e85edc9e2083f627093
