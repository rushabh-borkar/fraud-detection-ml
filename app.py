from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="Fraud Detection API")

# Load saved artifacts
model = joblib.load('fraud_model.pkl')
feature_columns = joblib.load('feature_column.pkl')
threshold = joblib.load('threshold.pkl')

# Same haversine function used in training
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 3959 * c

# Define the shape of an incoming transaction request
class Transaction(BaseModel):
    amt: float
    gender: str          # "M" or "F"
    lat: float
    long: float
    city_pop: int
    merch_lat: float
    merch_long: float
    category: str         # e.g. "shopping_net"
    dob: str              # "YYYY-MM-DD"
    trans_date_trans_time: str  # "YYYY-MM-DD HH:MM:SS"

@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()

    # Recreate engineered features
    dob = pd.to_datetime(data['dob'])
    trans_time = pd.to_datetime(data['trans_date_trans_time'])

    distance = haversine_distance(data['lat'], data['long'], data['merch_lat'], data['merch_long'])
    age = (trans_time - dob).days // 365
    hour = trans_time.hour
    day_of_week = trans_time.dayofweek
    gender_encoded = 1 if data['gender'] == 'M' else 0

    # Build a single-row dataframe matching training format
    row = {
        'amt': data['amt'],
        'gender': gender_encoded,
        'lat': data['lat'],
        'long': data['long'],
        'city_pop': data['city_pop'],
        'merch_lat': data['merch_lat'],
        'merch_long': data['merch_long'],
        'distance': distance,
        'age': age,
        'hour': hour,
        'day_of_week': day_of_week,
    }

    # One-hot encode category manually to match training columns
    for col in feature_columns:
        if col.startswith('category_'):
            cat_name = col.replace('category_', '')
            row[col] = 1 if data['category'] == cat_name else 0

    # Ensure correct column order, fill any missing with 0
    input_df = pd.DataFrame([row])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prob = model.predict_proba(input_df)[:, 1][0]
    is_fraud = int(prob >= threshold)

    # New return with risk tiers
    if prob < 0.3:
        risk_tier = "🟢 Low"
        action = "Normal monitoring"
    elif prob < 0.5:
        risk_tier = "🔵 Medium"
        action = "Enhanced transaction monitoring"
    elif prob < 0.73:
        risk_tier = "🟡 High"
        action = "Investigator alert"
    else:
        risk_tier = "🔴 Critical"
        action = "Immediate escalation"

    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": is_fraud,
        "threshold_used": threshold,
        "risk_tier": risk_tier,
        "recommended_action": action
    }
    from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running. POST to /predict with transaction data."}
