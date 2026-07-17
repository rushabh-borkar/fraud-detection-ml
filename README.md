# Credit Card Fraud Detection

An end-to-end machine learning pipeline that detects fraudulent credit card transactions from raw, real-world-style transaction data — including feature engineering, imbalance handling, model comparison, threshold selection, and a deployed FastAPI prediction service.

## Problem Statement

Credit card fraud is rare (well under 1% of transactions) but costly. A naive classifier can hit 99%+ accuracy just by predicting "not fraud" every time — making accuracy a useless metric here. The real challenge is building a model that catches fraud (recall) without flooding a bank's review team or customers with false alarms (precision), and picking an operating point that balances the two deliberately rather than by default.

This project builds that pipeline end-to-end: from raw transaction logs to a deployed API that returns a fraud probability and decision for a new transaction.

## Dataset

**Source:** [Sparkov Credit Card Transaction Simulator](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle)

- ~1.3M training transactions, ~556K test transactions
- Fraud rate: ~0.58%
- Raw fields include: transaction amount, merchant, category, customer/merchant GPS coordinates, customer demographics (age, gender, job), and timestamps

Unlike the commonly-used `mlg-ulb/creditcardfraud` dataset (which ships with anonymized, pre-PCA'd features), this dataset provides **raw, interpretable fields**, making real feature engineering possible and necessary.

## Approach

### 1. Exploratory Data Analysis
Investigated fraud patterns across transaction amount, merchant category, and time before writing any modeling code. Key findings:
- Fraudulent transactions average **~8x higher** transaction amount than legitimate ones
- Fraud clusters at **late-night / early-morning hours** (bimodal distribution), unlike the steady daytime spread of legitimate transactions
- Fraud rate varies meaningfully by merchant category (highest: `shopping_net`, `misc_net`, `grocery_pos`)
- Customer-to-merchant distance and day-of-week showed negligible correlation with fraud

### 2. Feature Engineering
Built four new features from raw fields, each validated against the target before use:

| Feature | Derived From | Signal Strength |
|---|---|---|
| `distance` | Haversine distance between customer and merchant GPS coords | Negligible |
| `age` | Customer DOB vs. transaction timestamp | Weak |
| `hour` | Transaction timestamp | **Strong** |
| `day_of_week` | Transaction timestamp | Negligible |

### 3. Preprocessing
- Dropped identifiers/PII (`cc_num`, names, street address, transaction ID) and high-cardinality categoricals (`merchant`, `city`, `job`, `state`) that added complexity without proportional signal
- One-hot encoded `category`, binary-encoded `gender`

### 4. Imbalance Handling & Modeling
Compared three models, each with class-imbalance correction:

| Model | Technique | Precision (Fraud) | Recall (Fraud) | F1 |
|---|---|---|---|---|
| Logistic Regression | `class_weight='balanced'` | 0.02 | 0.74 | 0.04 |
| Random Forest | `class_weight='balanced'` | 0.90 | 0.78 | 0.83 |
| XGBoost | `scale_pos_weight` (tuned) | 0.62 | 0.85 | 0.72 |

Models were then compared on a **threshold-independent basis** using Average Precision (Precision-Recall AUC):

- Random Forest: **AP = 0.866**
- XGBoost: **AP = 0.870** ← selected as final model

### 5. Threshold Selection
Rather than using the default 0.5 cutoff, the operating threshold was chosen at the point where precision and recall intersect on the PR curve — avoiding both the low-recall "too cautious" zone and the high-recall "too noisy" zone where precision collapses.

**Final threshold: 0.730**

## Final Results

| Metric | Score |
|---|---|
| Precision (fraud) | 0.80 |
| Recall (fraud) | 0.80 |
| F1-score (fraud) | 0.80 |
| False positives | 429 |
| False negatives | 429 |

The model catches 4 out of 5 fraudulent transactions, with 4 out of 5 flagged transactions being genuine fraud — a balanced, defensible operating point for a real-world fraud review system.

### Feature Importance
`amt` (55%) and `hour` (20%) account for ~75% of the model's decision-making — directly confirming the patterns found during manual EDA before any model was trained.

## Project Structure

```
.
├── code.ipynb              # Full pipeline: EDA, feature engineering, modeling, evaluation
├── app.py                  # FastAPI service for real-time fraud prediction
├── fraud_model.pkl         # Trained XGBoost model
├── feature_column.pkl      # Feature column order expected by the model
├── threshold.pkl           # Selected decision threshold (0.730)
├── requirements.txt
└── README.md
```

## How to Run

### 1. Set up environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Explore the pipeline
Open `code.ipynb` in Jupyter to walk through EDA, feature engineering, model training, and evaluation.

### 3. Run the prediction API
```bash
uvicorn app:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for an interactive Swagger UI to test predictions.

**Example request to `/predict`:**
```json
{
  "amt": 850.50,
  "gender": "M",
  "lat": 36.0788,
  "long": -81.1781,
  "city_pop": 3495,
  "merch_lat": 40.5,
  "merch_long": -85.0,
  "category": "shopping_net",
  "dob": "1988-03-09",
  "trans_date_trans_time": "2024-01-01 02:30:00"
}
```

**Example response:**
```json
{
  "fraud_probability": 0.8749,
  "is_fraud": 1,
  "threshold_used": 0.73
}
```
### 4. Open the UI
Open `fraud_ui.html` in your browser while uvicorn is running — 
enter any transaction details and get a real-time fraud risk assessment.

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, FastAPI, uvicorn, matplotlib

## Key Takeaways
- Accuracy is a misleading metric on imbalanced datasets — a 99%-accurate model can still be practically useless
- Manual EDA correctly anticipated the two dominant features (`amt`, `hour`), later confirmed by model feature importance
- Threshold selection is a business decision as much as a technical one — the "best" model depends on what recall/precision tradeoff a real deployment can tolerate
- Comparing models by Average Precision (not a single threshold's classification report) avoids being misled by threshold-specific noise

## Future Improvements
- Test SMOTE against class-weighting on the final model
- Add `state`-level or `job`-based features with proper encoding for high-cardinality categoricals
- Deploy the API publicly (Render/Railway) for a live demo link
- Add authentication and request logging for production-readiness
