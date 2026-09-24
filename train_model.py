"""
train_model.py
──────────────
Loads housing_price_dataset.csv, preprocesses features, trains an
XGBoost regression model, and saves the model + preprocessor pipeline
as artifacts inside the models/ folder.

Run once before starting the Flask API or Streamlit app:
    python train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "housing_price_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_pipeline.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Feature sets ─────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "city", "locality_tier", "property_type",
    "floor_category", "facing", "furnishing_status",
    "transaction_type",
]

NUMERIC_FEATURES = [
    "bhk", "bathrooms", "balconies", "built_up_area", "carpet_area",
    "floor_number", "total_floors", "property_age", "parking_spaces",
    "security_score", "maintenance_fee_monthly",
    "distance_to_city_center_km", "distance_to_metro_km",
    "nearby_schools", "nearby_hospitals",
]

BINARY_FEATURES = [
    "gym_available", "swimming_pool", "power_backup", "lift_available",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES
TARGET       = "price_in_lakhs"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop columns not used for prediction
    drop_cols = ["property_id", "locality", "price_category"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna(subset=[TARGET])
    return df


def build_pipeline() -> Pipeline:
    """Build a sklearn Pipeline with preprocessing + XGBoost regressor."""

    numeric_pipe = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipe,     NUMERIC_FEATURES + BINARY_FEATURES),
            ("cat",  categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    model),
    ])


def train():
    print("[*] Loading data ...")
    df = load_data(DATA_PATH)
    print(f"    Rows: {len(df):,}   Columns: {len(df.columns)}")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[*] Training XGBoost pipeline ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred   = pipeline.predict(X_test)
    mae      = mean_absolute_error(y_test, y_pred)
    r2       = r2_score(y_test, y_pred)
    mape     = np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=1))) * 100

    print(f"\n[*] Evaluation on hold-out test set ({len(y_test):,} samples):")
    print(f"    MAE  : Rs.{mae:.2f} lakhs")
    print(f"    R2   : {r2:.4f}")
    print(f"    MAPE : {mape:.2f}%")

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[OK] Model saved -> {MODEL_PATH}")

    # Save metadata (city/tier/type lists for frontend dropdowns)
    df_orig = pd.read_csv(DATA_PATH)
    meta = {
        "cities":             sorted(df_orig["city"].dropna().unique().tolist()),
        "locality_tiers":     sorted(df_orig["locality_tier"].dropna().unique().tolist()),
        "property_types":     sorted(df_orig["property_type"].dropna().unique().tolist()),
        "floor_categories":   sorted(df_orig["floor_category"].dropna().unique().tolist()),
        "facings":            sorted(df_orig["facing"].dropna().unique().tolist()),
        "furnishing_statuses":sorted(df_orig["furnishing_status"].dropna().unique().tolist()),
        "transaction_types":  sorted(df_orig["transaction_type"].dropna().unique().tolist()),
        "metrics": {
            "mae":  round(mae, 2),
            "r2":   round(r2, 4),
            "mape": round(mape, 2),
            "train_samples": len(X_train),
            "test_samples":  len(X_test),
        },
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric":     NUMERIC_FEATURES,
            "binary":      BINARY_FEATURES,
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Metadata saved -> {META_PATH}")


if __name__ == "__main__":
    train()
