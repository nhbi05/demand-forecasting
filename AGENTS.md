# Agent Instructions

Before editing model notebooks, preprocessing code, or ensemble code, read:

```text
docs/ensemble-contract.md
docs/preprocessing-architecture.md
```

## Non-negotiable ensemble rule

Every model that contributes to `notebooks/06_Ensemble.ipynb` must produce
predictions for the same `date + item_id` rows.

The current shared benchmark is:

```text
top 100 products by total positive demand
with at least 730 active sales days
```

Models may use different internal features, but final ensemble-facing
prediction files must all follow:

```text
date,item_id,predicted_quantity,actual_quantity
```

Do not independently change these inside one model notebook:

- `selected_products.json`
- `splits.json`
- target definition
- final prediction scale
- final prediction schema

If a change requires different products, dates, scalers, or target definition,
update the shared preprocessing contract in `src/preprocessing.py`, regenerate
shared artifacts, and document the change.

## Safe model-specific work

Allowed:

- LSTM / RNN sequence building
- Random Forest lag and rolling features
- Gaussian time-index or kernel features
- model-specific hyperparameter tuning

Not allowed:

- creating per-model product lists for ensemble predictions
- creating per-model train/test date splits for ensemble predictions
- saving scaled predictions as final ensemble predictions
- changing `actual_quantity` meaning in only one model

When uncertain, stop and ask the user.

This file is the canonical repo-level agent instruction. Do not create
additional tool-specific rule files unless the user explicitly asks for them.
