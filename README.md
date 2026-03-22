# Credit Risk ML — Home Credit Default Risk

End-to-end machine learning project built on the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset. Two tasks on the same 300k loan applications across 7 relational tables:
binary default classification and loan amount regression.

## Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Classification | Logistic Regression (baseline) | AUC | 0.7550 |
| Classification | XGBoost default | AUC | 0.7679 |
| Classification | XGBoost tuned | AUC | 0.7799 |
| Classification | XGBoost tuned | KS | 0.42 |
| Regression | Median predictor (baseline) | MAE | 307,119 |
| Regression | XGBoost default | MAE | 29,242 |
| Regression | XGBoost tuned | MAE | 26,569 |
| Regression | XGBoost tuned | R² | 0.9894 |

## Key Techniques

- 7-table data merging with domain-driven aggregations
- Feature engineering: credit-to-income ratios, bureau aggregations, payment behaviour features
- 5-fold TimeSeriesSplit cross-validation — no temporal data leakage
- Optuna hyperparameter tuning (Bayesian optimisation, 25 trials)
- SHAP TreeExplainer — global feature attribution and individual prediction explanations
- Class imbalance handling via scale_pos_weight (8% default rate)
- Log-transform for skewed regression target (skewness 1.24 → -0.34)

## Tech Stack

Python · XGBoost · LightGBM · Scikit-learn · SHAP · Optuna · Pandas · NumPy · Matplotlib · Seaborn

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — Kaggle competition dataset.
7 tables: application, bureau, bureau balance, previous applications,
POS cash balance, credit card balance, installment payments.

## Files

- `home-credit-default-risk.ipynb` — full pipeline: EDA, feature engineering,
  baseline LR, XGBoost CV, Optuna tuning, SHAP global + individual explanations
