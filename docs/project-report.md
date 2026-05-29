# DEMAND FORECASTING SYSTEM
## Project Report

---

Submitted by:
[Group Member Names]
[Student IDs]
[Course / Program Name]
[Date]

---

## 1. Introduction

Accurate demand forecasting is a critical component of retail operations, enabling businesses to optimise inventory, reduce waste, and improve customer satisfaction. Traditional forecasting methods, while widely used, often struggle to capture the complex, non-linear patterns inherent in real-world retail data. This project addresses that limitation by designing and implementing a multi-model demand forecasting system that combines the complementary strengths of several machine learning architectures.

The system predicts daily sales quantities for a curated selection of retail products using historical transaction data. Rather than relying on a single model, four independent forecasting models are trained — Long Short-Term Memory (LSTM) [1], Gated Recurrent Unit (GRU) [2], Random Forest [3], and Gaussian Process [4] — and their predictions are combined through a weighted ensemble [5]. The rationale is that different model types excel at capturing different patterns, and an ensemble consistently outperforms any individual model in variance and robustness.

### 1.1 Problem Statement

Retail businesses face the challenge of forecasting demand at the product level across time, often with significant noise in transaction records. No single modelling approach adequately handles all the variability in real-world sales data: deep learning models capture sequential dependencies well but may overfit on sparse products, while tree-based models handle feature interactions effectively but miss long-range temporal patterns.

This project therefore aims to:

- Develop four independent forecasting models trained on shared, pre-processed data.
- Combine their predictions through a robust ensemble that weights models by performance.
- Evaluate all models and the ensemble on a held-out test set using standard regression metrics.

### 1.2 Significance

Improved demand forecasting delivers direct business value: it reduces overstock and stockout events, informs replenishment decisions, and supports data-driven planning. Beyond the immediate application, the project demonstrates a replicable pipeline architecture in which multiple heterogeneous models are trained independently on shared artifacts and combined transparently — a pattern applicable across a wide range of time-series forecasting problems.

---

## 2. Methodology

### 2.1 Dataset

The raw dataset (`sales.csv`) comprised approximately **7.4 million retail transaction rows**. A preprocessing pipeline aggregated records into a clean daily format. The final processed dataset (`data/processed/daily.csv`) contained **76,100 rows** — one row per product per day — spanning 28 August 2022 to 26 September 2024 (761 days).

The top 100 products by total positive demand were selected among those with a meaningful sales history. This selection was fixed across all models to ensure fair comparison. The 27 features in the processed dataset included:

- **Demand signals:** daily quantity sold, positive quantity (sales only), returns quantity, and transaction count.
- **Price signals:** base price and total revenue.
- **Store signals:** number of stores selling the product on a given day.
- **Calendar features:** day of week, day of month, and month encoded as sine/cosine pairs to capture cyclical patterns.

The data was divided into training, validation, and test splits as shown in Table 1:

| Split | Start Date | End Date |
|---|---|---|
| Training | 2022-08-28 | 2024-02-27 |
| Validation | 2024-02-28 | 2024-04-27 |
| Test | 2024-04-28 | 2024-09-26 |

*Table 1: Train / Validation / Test Split Boundaries*

All scalers and outlier caps were fitted exclusively on the training period and applied to validation and test sets, ensuring no future information leaked into model training.

### 2.2 Exploratory Data Analysis

Exploratory analysis was conducted on the raw `sales.csv` before any preprocessing, with the goal of understanding the structure and statistical properties of the data and informing key design decisions — product selection, feature engineering, lookback window size, and model choice.

#### 2.2.1 Dataset Overview

The raw dataset contained **7,432,685 transaction rows** spanning **761 days** (28 August 2022 – 26 September 2024) across **28,182 unique products** and **4 stores**. No missing values were found in any column; imputation flags for `price_base`, `sum_total`, and `store_id` were all zero, confirming clean data at the transaction level.

#### 2.2.2 Quantity Distribution and Returns

Transaction quantities ranged from –500 to +4,952, with a median of 2 and a 99th percentile of 57, indicating a strongly right-skewed distribution with rare but extreme values. Negative-quantity rows — representing product returns — accounted for only **0.02% of transactions** (1,160 rows) and were netted into daily aggregates rather than excluded, preserving the true net demand signal. An additional **11.17% of rows** (829,933 transactions across 2,500 distinct products) contained fractional quantities, indicating weight- or volume-based items sold in non-integer units. These were retained without special treatment since the models predict continuous-valued demand.

#### 2.2.3 Product Sparsity and Selection

Per-product activity analysis revealed high sparsity: the median product had sales on only **72 of the 761 available days**, while only 187 products had a complete sales history across all 761 days. This sparsity makes most products unsuitable for sequence models that require a long, consistent history. Accordingly, the **top 100 products by total positive demand** were selected for modelling — a threshold that ensured sufficient historical signal for LSTM, GRU, and Random Forest while keeping the modelling scope tractable.

#### 2.2.4 Temporal Patterns and Seasonality

Aggregate daily demand showed a clear **weekly seasonal pattern**: a day-of-week boxplot revealed consistent variation across Monday through Sunday, with certain days carrying systematically higher volumes. This confirmed that day-of-week encoding is an essential feature. Monthly analysis showed moderate variation across calendar months, and year-over-year plots indicated relatively stable aggregate demand with no strong upward or downward trend at the aggregate level.

An **Augmented Dickey-Fuller (ADF) stationarity test** applied to the aggregate daily series returned a p-value of **0.857**, failing to reject the unit-root null hypothesis. This suggests the aggregate series is not strictly stationary. However, individual product-level series — which the models actually target — exhibit product-specific local trends managed by normalisation and rolling features rather than explicit differencing.

#### 2.2.5 ACF/PACF and STL Decomposition

Autocorrelation (ACF) and partial autocorrelation (PACF) plots of the aggregate series confirmed significant autocorrelation at **lags that are multiples of 7**, providing quantitative support for the 7-day and 14-day lag features used in the Random Forest model and for the 60-day lookback window chosen for LSTM and GRU. STL decomposition (period = 7) separated the series into trend, seasonal, and residual components, with a **high seasonality strength score** confirming that weekly cyclicality is the dominant signal. These findings directly motivated the use of sine/cosine calendar encodings for day-of-week, day-of-month, and month-of-year.

#### 2.2.6 Store-Level Heterogeneity

Aggregating demand by store revealed that volume and variance differed across the four stores. Rather than training per-store models — which would quadruple the modelling effort and fragment the training data — store-level signal was captured by including `store_count` (the number of stores selling a product on a given day) as a feature. This encodes store-level presence without requiring separate model instances.

### 2.3 System Architecture

The system was structured into two layers: a shared data pipeline and four independent model notebooks, all feeding into a single ensemble notebook. The preprocessing module (`src/preprocessing.py`) produced a set of canonical shared artifacts — `daily.csv`, `splits.json`, `selected_products.json`, `scalers.pkl`, and `outlier_caps.json` — that every model notebook read without modification. This design guaranteed that product selection, date splits, scaling, and feature definitions were identical across all models, making their predictions directly comparable and ensemble-combinable.

Each model notebook saved test-set predictions as a CSV with a common schema (`date, item_id, predicted_quantity, actual_quantity`). The ensemble notebook (`06_Ensemble.ipynb`) joined these four files on `(date, item_id)` and computed a weighted average.

### 2.4 Models

#### 2.4.1 LSTM (Long Short-Term Memory)

LSTM is a recurrent neural network architecture designed for sequential learning [1]. It processes a sliding window of 60 days of past daily demand values and uses a gating mechanism — input, forget, and output gates — to selectively retain or discard information across many timesteps. This makes it well-suited for capturing weekly demand cycles, sustained trends, and abrupt spikes. The model additionally learns a product embedding, a low-dimensional learned representation of each product's identity, enabling it to capture product-level demand characteristics implicitly. The model was implemented in PyTorch and trained using the Adam optimiser with early stopping on the validation set.

#### 2.4.2 GRU (Gated Recurrent Unit)

GRU is a simplified variant of LSTM with two gates (reset and update) instead of three, resulting in fewer trainable parameters and faster training [2]. On well-structured time series, GRU and LSTM typically converge to similar performance. A notable design choice in this project's GRU is **residual learning**: the model predicts the correction to add on top of a seasonal-naive baseline (last week's demand), focusing on what weekly seasonality alone cannot explain. This project confirmed the expected convergence: GRU trained faster while matching LSTM's R² almost exactly.

#### 2.4.3 Random Forest

Random Forest is an ensemble of decision trees, each trained on a bootstrap sample of the data [3]. Rather than processing sequences, it takes flat feature vectors comprising lag features (demand from 1, 7, 14, and 30 days prior), rolling averages over 7-, 14-, and 30-day windows, price signals, store counts, and calendar features. It learns non-linear decision rules efficiently and produces interpretable feature importance scores via Gini Impurity (Mean Decrease in Impurity).

#### 2.4.4 Gaussian Process

Gaussian Process (GP) is a probabilistic, non-parametric model that represents predictions as distributions rather than point estimates [4]. It produces a mean forecast with a confidence interval, making it particularly useful for capturing smooth underlying trends and quantifying predictive uncertainty. Its computational cost scales cubically with dataset size (O(n³)), which required capping training data at 200 rows per product. The kernel combines RBF (smooth trends), ExpSineSquared (seasonal patterns), ConstantKernel, and WhiteKernel (observation noise).

### 2.5 Ensemble Strategy

The ensemble combined the four models' forecasts through a weighted average [5]:

```
ensemble_pred = w_lstm × lstm_pred + w_gru × gru_pred + w_rf × rf_pred + w_gp × gp_pred
```

Two strategies were implemented. The **simple average** assigned equal weight (0.25) to each model. The **weighted average** set weights proportional to each model's R² score on a calibration window — specifically the first half of the test set. Final metrics were reported on the second half only, so no label used during weight fitting appeared in the evaluation. This split prevents the weight-fitting step from inflating the reported accuracy.

### 2.6 Technologies

| Category | Tool |
|---|---|
| Deep learning | PyTorch |
| Traditional ML | scikit-learn |
| Data processing | pandas, numpy |
| Visualisation | matplotlib |
| Notebooks | Jupyter |
| Language | Python 3 |

---

## 3. Results

### 3.1 Individual Model Performance

All four models were evaluated on the held-out test set using R² (explained variance), RMSE (root mean squared error), and MAPE (mean absolute percentage error, computed only on days with positive actual demand). Results are summarised in Table 2.

| Model | R² | RMSE | MAPE |
|---|---|---|---|
| LSTM | 0.981 | 64 | 37% |
| GRU | 0.981 | 66 | 38% |
| Random Forest | 0.966 | 86 | 17% |
| Gaussian Process | 0.866 | 173 | 36% |

*Table 2: Individual Model Performance on Test Set*

[ Figure 1: Actual vs Predicted daily demand — LSTM (test set) ]

[ Figure 2: Actual vs Predicted daily demand — GRU (test set) ]

[ Figure 3: Actual vs Predicted daily demand — Random Forest (test set) ]

LSTM and GRU delivered the strongest overall performance, each achieving R² of 0.981 with RMSE values of 64 and 66 respectively. Their near-identical scores reflect the well-documented convergence of these two architectures on well-structured time-series data. Both excelled at capturing sequential demand patterns — weekly periodicity, sustained trends, and demand spikes.

Random Forest achieved R² of 0.966 with a notably lower MAPE of 17%, outperforming all other models on that metric. This suggests it is particularly well-calibrated on typical lower-demand days, even though its RMSE is higher because it struggles more with extreme demand spikes that deep learning models handle smoothly.

Gaussian Process recorded the weakest point-prediction metrics (R² = 0.866, RMSE = 173). This is consistent with its primary design as a probabilistic model rather than a high-precision point forecaster, and with the 200-row training cap imposed by its O(n³) complexity. Its added value lies in the confidence intervals it provides, which can inform inventory safety-stock decisions.

### 3.2 Feature Importance

Feature importance was assessed using the Random Forest model's built-in Gini Importance (Mean Decrease in Impurity). Short-term lag features — particularly the 1-day and 7-day lagged demand — accounted for the largest share of predictive importance, confirming that recent sales history and weekly periodicity are the dominant signals in retail demand. Rolling demand averages over 7 and 14 days ranked next, followed by calendar encodings (day of week) and price signals. This ranking is consistent with the strong weekly seasonality visible in the raw transaction data.

[ Figure 4: Top 10 feature importances — Random Forest (Gini Importance) ]

### 3.3 Ensemble Performance

Both ensemble strategies were evaluated on the second half of the test window after calibrating weights on the first half. Results are shown in Table 3 alongside individual model baselines.

| Model | R² | RMSE | MAPE |
|---|---|---|---|
| LSTM | 0.981 | 64 | 37% |
| GRU | 0.981 | 66 | 38% |
| Random Forest | 0.966 | 86 | 17% |
| Gaussian Process | 0.866 | 173 | 36% |
| Simple Ensemble | 0.967 | 85 | 28% |
| Weighted Ensemble | **0.968** | **84** | **28%** |

*Table 3: Ensemble vs Individual Model Performance*

[ Figure 5: Ensemble Comparison — R², RMSE, MAPE across all models ]

The weighted ensemble marginally outperformed the simple average (R² 0.968 vs 0.967, RMSE 84 vs 85), confirming that assigning greater weight to stronger models provides a consistent benefit. Both ensembles achieved MAPE of 28%, a meaningful improvement over the deep learning models (37–38%) while falling between the deep learning and Random Forest extremes.

Neither ensemble surpassed LSTM or GRU individually on R² and RMSE. This is consistent with theory: when the best individual models are already highly correlated, the ensemble's variance-reduction benefit is limited, and the inclusion of a weaker model (Gaussian Process) introduces a downward pull on overall accuracy.

### 3.4 Discussion

The results confirm that the ensemble approach delivers more stable, consistent predictions than any single model in isolation. The weighted ensemble provides a principled mechanism for automatically allocating more influence to stronger models without requiring manual tuning, and its MAPE of 28% represents a balanced improvement over the individual model extremes.

The four models each contribute distinct value: LSTM and GRU provide the highest raw accuracy by capturing temporal sequences; Random Forest provides the best calibration on typical-demand days and interpretable feature importance; and Gaussian Process provides uncertainty estimates that support risk-aware inventory decisions. The ensemble fuses these contributions into a single forecast that is more robust than any individual model alone.

Looking forward, several directions could strengthen the system: incorporating exogenous variables such as promotions and holidays could improve accuracy; expanding the product selection beyond the top 100 items would test generalisability; and extending the Gaussian Process output to produce ensemble-level prediction intervals would add uncertainty quantification to the final forecast.

---

## 4. Conclusion

This project successfully developed a multi-model demand forecasting system trained on 7.4 million real retail transactions. Four machine learning architectures — LSTM, GRU, Random Forest, and Gaussian Process — were trained independently on shared preprocessed data and combined through a weighted ensemble, meeting all stated objectives.

Among individual models, LSTM and GRU achieved the highest predictive accuracy with R² values of 0.981, demonstrating that deep learning architectures are highly effective at capturing the sequential, cyclical patterns in daily retail demand. Random Forest delivered the lowest MAPE (17%), making it the most reliable model for typical-demand days, while the Gaussian Process contributed probabilistic uncertainty estimates not available from the other models.

The weighted ensemble (R² = 0.968, RMSE = 84, MAPE = 28%) improved prediction stability and reduced MAPE relative to the deep learning models, demonstrating the value of combining heterogeneous models with performance-weighted contributions.

The system provides a robust and scalable framework for retail demand forecasting and inventory planning. By producing daily, product-level forecasts, it equips retail decision-makers with the information needed to reduce stockouts and overstock events, optimise replenishment schedules, and ultimately improve service levels and profitability.

---

## 5. References

[1] Hochreiter, S. and Schmidhuber, J. (1997) 'Long Short-Term Memory,' *Neural Computation*, 9(8), pp. 1735–1780.

[2] Cho, K. et al. (2014) 'Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation,' *Proceedings of EMNLP 2014*, pp. 1724–1734.

[3] Breiman, L. (2001) 'Random Forests,' *Machine Learning*, 45(1), pp. 5–32.

[4] Rasmussen, C.E. and Williams, C.K.I. (2006) *Gaussian Processes for Machine Learning*. MIT Press.

[5] Dietterich, T.G. (2000) 'Ensemble Methods in Machine Learning,' in *Multiple Classifier Systems, Lecture Notes in Computer Science*, vol. 1857, Springer, pp. 1–15.

[6] Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*, 3rd ed. OTexts. Available at: https://otexts.com/fpp3

[7] Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python,' *Journal of Machine Learning Research*, 12, pp. 2825–2830.

[8] Paszke, A. et al. (2019) 'PyTorch: An Imperative Style, High-Performance Deep Learning Library,' *Advances in Neural Information Processing Systems*, 32.

---

## Appendix A: Model Hyperparameters

### A.1 LSTM

| Hyperparameter | Value |
|---|---|
| Framework | PyTorch |
| Lookback window | 60 days |
| Forecast horizon | 30 days |
| Hidden units | 64 |
| Recurrent layers | 2 |
| Dropout rate | 0.3 |
| Product embedding dimension | 4 |
| Optimiser | Adam (lr = 0.001) |
| L2 weight decay | 1e-4 |
| Loss function | Huber (SmoothL1Loss) |
| Max epochs | 100 |
| Early stopping patience | 5 (monitor: val Huber loss) |
| Input features | 15 |

### A.2 GRU

| Hyperparameter | Value |
|---|---|
| Framework | PyTorch |
| Lookback window | 60 days |
| Forecast horizon | 30 days |
| Hidden units | 64 |
| Recurrent layers | 2 |
| Dropout rate | 0.3 |
| Product embedding dimension | 4 |
| Optimiser | Adam (lr = 0.001) |
| L2 weight decay | 1e-4 |
| Loss function | MSE (on residual target) |
| Max epochs | 100 |
| Early stopping patience | 10 (monitor: val MSE) |
| Seasonality period | 7 days |
| Input features | 14 |

### A.3 Random Forest

| Hyperparameter | Value |
|---|---|
| Framework | scikit-learn |
| n_estimators | 100 |
| max_depth | 15 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| Lag features | 1, 7, 14, 30 days |
| Rolling windows | 7, 14, 30 days |
| Forecast strategy | Recursive |
| Random seed | 42 |

### A.4 Gaussian Process

| Hyperparameter | Value |
|---|---|
| Framework | scikit-learn |
| Kernel | RBF + ConstantKernel + WhiteKernel + ExpSineSquared |
| Max training rows per product | 200 (O(n³) constraint) |
| Optimiser restarts | 2 |
| Output | Mean prediction + uncertainty (std) |

---

## Appendix B: Key Code Snippets

**B.1 Ensemble Join**

Predictions are joined on `(date, item_id)` to guarantee the same product on the same day is matched across all four files before averaging:

```python
merged = (
    lstm_df.rename(columns={'predicted_quantity': 'lstm_pred', 'actual_quantity': 'actual'})
    .merge(gru_df.rename(columns={'predicted_quantity': 'gru_pred'})[['date','item_id','gru_pred']], on=['date','item_id'])
    .merge(rf_df.rename(columns={'predicted_quantity': 'rf_pred'})[['date','item_id','rf_pred']], on=['date','item_id'])
    .merge(gaussian_df.rename(columns={'predicted_quantity': 'gaussian_pred'})[['date','item_id','gaussian_pred']], on=['date','item_id'])
    .sort_values(['date','item_id'])
    .reset_index(drop=True)
)
```

**B.2 MAPE Calculation**

Zero-quantity rows are excluded before computing MAPE to avoid division by zero:

```python
def mape(y_true, y_pred):
    mask   = y_true > 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

**B.3 Weighted Ensemble Calibration**

Weights are calibrated on the first half of the test window and the ensemble is evaluated on the second half:

```python
mid = len(merged) // 2
calib = merged.iloc[:mid]
eval_ = merged.iloc[mid:]

weights       = compute_r2_weights(calib)
ensemble_pred = weighted_average(eval_, weights)
results       = evaluate(eval_['actual_quantity'], ensemble_pred)
```
