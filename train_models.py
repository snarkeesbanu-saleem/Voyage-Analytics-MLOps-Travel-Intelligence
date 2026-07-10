"""
train_models.py
===============
Standalone script to train all three Voyage Analytics models and save them
along with their preprocessors to the models/ directory.

Run: python train_models.py
"""

import os
import sys
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Voyage Analytics — Model Training Script")
print("=" * 60)

# ── Load Raw Data ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading datasets...")
users_df = pd.read_csv(DATA_DIR / "users.csv")
flights_df = pd.read_csv(DATA_DIR / "flights.csv")
hotels_df = pd.read_csv(DATA_DIR / "hotels.csv")
print(f"  Users: {len(users_df):,} rows")
print(f"  Flights: {len(flights_df):,} rows")
print(f"  Hotels: {len(hotels_df):,} rows")

# ── Date Parsing ──────────────────────────────────────────────────────────────
flights_df["date"] = pd.to_datetime(flights_df["date"], errors="coerce")
hotels_df["date"] = pd.to_datetime(hotels_df["date"], errors="coerce")

# ── USER BEHAVIOR AGGREGATION ─────────────────────────────────────────────────
print("\n[2/6] Building user behavior features...")

flight_agg = flights_df.groupby("userCode").agg(
    num_flights=("travelCode", "count"),
    total_flight_spend=("price", "sum"),
    avg_flight_price=("price", "mean"),
    unique_destinations=("to", "nunique"),
    last_flight_date=("date", "max"),
).reset_index()

hotel_agg = hotels_df.groupby("userCode").agg(
    num_hotel_stays=("travelCode", "count"),
    total_hotel_spend=("total", "sum"),
    avg_hotel_price=("price", "mean"),
    avg_stay_duration=("days", "mean"),
).reset_index()

# preferred flight type
pref_flight = (
    flights_df.groupby(["userCode", "flightType"])
    .size()
    .reset_index(name="cnt")
    .sort_values("cnt", ascending=False)
    .drop_duplicates("userCode")[["userCode", "flightType"]]
    .rename(columns={"flightType": "preferred_flight_type"})
)

# preferred agency
pref_agency = (
    flights_df.groupby(["userCode", "agency"])
    .size()
    .reset_index(name="cnt")
    .sort_values("cnt", ascending=False)
    .drop_duplicates("userCode")[["userCode", "agency"]]
    .rename(columns={"agency": "preferred_agency"})
)

# Merge all
df_ml = users_df.rename(columns={"code": "userCode"}).merge(flight_agg, on="userCode", how="left")
df_ml = df_ml.merge(hotel_agg, on="userCode", how="left")
df_ml = df_ml.merge(pref_flight, on="userCode", how="left")
df_ml = df_ml.merge(pref_agency, on="userCode", how="left")

# Fill NAs for users with no flight/hotel data
num_cols = ["num_flights","total_flight_spend","avg_flight_price","unique_destinations",
            "num_hotel_stays","total_hotel_spend","avg_hotel_price","avg_stay_duration"]
df_ml[num_cols] = df_ml[num_cols].fillna(0)
df_ml["preferred_flight_type"] = df_ml["preferred_flight_type"].fillna("economic")
df_ml["preferred_agency"] = df_ml["preferred_agency"].fillna("CloudFy")

# Churn label: no activity in last 180 days
max_date = flights_df["date"].max()
churn_threshold = max_date - pd.Timedelta(days=180)
df_ml["last_activity"] = df_ml["last_flight_date"].fillna(pd.Timestamp("2000-01-01"))
df_ml["churned"] = (df_ml["last_activity"] < churn_threshold).astype(int)
df_ml["days_since_last_activity"] = (max_date - df_ml["last_activity"]).dt.days

print(f"  Combined ML DataFrame: {len(df_ml):,} users")
print(f"  Churn rate: {df_ml['churned'].mean():.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — CHURN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Training Churn Classifier...")

churn_features = [
    "age", "num_flights", "total_flight_spend", "avg_flight_price",
    "unique_destinations", "num_hotel_stays", "total_hotel_spend",
    "avg_hotel_price", "avg_stay_duration", "days_since_last_activity"
]

X_churn = df_ml[churn_features].copy()
y_churn = df_ml["churned"]

# Scale
churn_scaler = StandardScaler()
X_churn_scaled = churn_scaler.fit_transform(X_churn)

X_tr, X_te, y_tr, y_te = train_test_split(X_churn_scaled, y_churn, test_size=0.2, random_state=42, stratify=y_churn)
churn_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
churn_model.fit(X_tr, y_tr)
acc = accuracy_score(y_te, churn_model.predict(X_te))
print(f"  Churn Model Accuracy: {acc:.4f}")

# Save model and preprocessor
joblib.dump(churn_model, MODELS_DIR / "churn_classifier.joblib")
with open(MODELS_DIR / "churn_preprocessor.pkl", "wb") as f:
    pickle.dump({"scaler": churn_scaler, "features": churn_features}, f)
print("  [Saved] churn_classifier.joblib + churn_preprocessor.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — FLIGHT PRICE REGRESSOR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Training Flight Price Regressor...")

# Encode categoricals
le_route_from = LabelEncoder()
le_route_to = LabelEncoder()
le_flight_type = LabelEncoder()
le_agency = LabelEncoder()

fdf = flights_df.dropna(subset=["price", "distance", "time", "flightType", "agency"]).copy()
fdf["from_enc"] = le_route_from.fit_transform(fdf["from"].astype(str))
fdf["to_enc"] = le_route_to.fit_transform(fdf["to"].astype(str))
fdf["flightType_enc"] = le_flight_type.fit_transform(fdf["flightType"].astype(str))
fdf["agency_enc"] = le_agency.fit_transform(fdf["agency"].astype(str))
fdf["month"] = fdf["date"].dt.month.fillna(1)
fdf["day_of_week"] = fdf["date"].dt.dayofweek.fillna(0)

price_features = ["from_enc", "to_enc", "flightType_enc", "agency_enc",
                  "distance", "time", "month", "day_of_week"]

X_price = fdf[price_features]
y_price = fdf["price"]

price_scaler = StandardScaler()
X_price_scaled = price_scaler.fit_transform(X_price)

X_tr, X_te, y_tr, y_te = train_test_split(X_price_scaled, y_price, test_size=0.2, random_state=42)
price_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
price_model.fit(X_tr, y_tr)
r2 = r2_score(y_te, price_model.predict(X_te))
mae = mean_absolute_error(y_te, price_model.predict(X_te))
print(f"  Flight Price Model — R²: {r2:.4f}, MAE: ${mae:,.2f}")

joblib.dump(price_model, MODELS_DIR / "flight_price_model.joblib")
with open(MODELS_DIR / "flight_price_preprocessor.pkl", "wb") as f:
    pickle.dump({
        "scaler": price_scaler,
        "features": price_features,
        "le_from": le_route_from,
        "le_to": le_route_to,
        "le_flight_type": le_flight_type,
        "le_agency": le_agency,
    }, f)
print("  [Saved] flight_price_model.joblib + flight_price_preprocessor.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — GENDER CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Training Gender Classifier...")

le_gender = LabelEncoder()
df_ml_g = df_ml.dropna(subset=["gender"]).copy()
df_ml_g["gender_enc"] = le_gender.fit_transform(df_ml_g["gender"].astype(str))

gender_features = [
    "age", "num_flights", "total_flight_spend", "avg_flight_price",
    "unique_destinations", "num_hotel_stays", "total_hotel_spend",
    "avg_hotel_price", "avg_stay_duration"
]

X_gender = df_ml_g[gender_features]
y_gender = df_ml_g["gender_enc"]

gender_scaler = StandardScaler()
X_gender_scaled = gender_scaler.fit_transform(X_gender)

X_tr, X_te, y_tr, y_te = train_test_split(X_gender_scaled, y_gender, test_size=0.2, random_state=42, stratify=y_gender)
gender_model = LogisticRegression(max_iter=500, random_state=42, C=1.0)
gender_model.fit(X_tr, y_tr)
acc_g = accuracy_score(y_te, gender_model.predict(X_te))
print(f"  Gender Model Accuracy: {acc_g:.4f}")

joblib.dump(gender_model, MODELS_DIR / "gender_classifier.joblib")
with open(MODELS_DIR / "gender_preprocessor.pkl", "wb") as f:
    pickle.dump({
        "scaler": gender_scaler,
        "features": gender_features,
        "le_gender": le_gender,
    }, f)
print("  [Saved] gender_classifier.joblib + gender_preprocessor.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Verifying saved model files...")
model_files = list(MODELS_DIR.glob("*"))
for f in model_files:
    size_kb = f.stat().st_size / 1024
    print(f"   {f.name} ({size_kb:.1f} KB)")

print("\n" + "=" * 60)
print("All 3 models trained and saved successfully!")
print(f"   Model directory: {MODELS_DIR}")
print("=" * 60)
