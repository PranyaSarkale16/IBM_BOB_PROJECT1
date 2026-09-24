"""
=============================================================================
  HOUSE PRICE PREDICTION — SINGLE-FILE APP
  IBM SkillsBuild Data Analytics with AI Academic Internship
  BharatCares × AICTE

  Everything in one file:
    • Model training logic  (runs automatically on first launch)
    • Prediction logic      (runs inside the app, no HTTP calls)
    • Frontend UI           (Streamlit — 3 pages)

  Run with ONE command:
    streamlit run house_price_prediction_code.py
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# =============================================================================
# SECTION 1 — CONSTANTS & PATHS
# =============================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "housing_price_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_pipeline.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

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
TARGET = "price_in_lakhs"


# =============================================================================
# SECTION 2 — MODEL TRAINING  (called once; result cached to disk)
# =============================================================================

def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]),
             NUMERIC_FEATURES + BINARY_FEATURES),
            ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore",
                                                     sparse_output=False))]),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


def train_and_save():
    """Train the XGBoost pipeline and save model + metadata to disk."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in ["property_id", "locality", "price_category"]
                           if c in df.columns])
    df = df.dropna(subset=[TARGET])

    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=1))) * 100)

    joblib.dump(pipeline, MODEL_PATH)

    df_orig = pd.read_csv(DATA_PATH)
    meta = {
        "cities":              sorted(df_orig["city"].dropna().unique().tolist()),
        "locality_tiers":      sorted(df_orig["locality_tier"].dropna().unique().tolist()),
        "property_types":      sorted(df_orig["property_type"].dropna().unique().tolist()),
        "floor_categories":    sorted(df_orig["floor_category"].dropna().unique().tolist()),
        "facings":             sorted(df_orig["facing"].dropna().unique().tolist()),
        "furnishing_statuses": sorted(df_orig["furnishing_status"].dropna().unique().tolist()),
        "transaction_types":   sorted(df_orig["transaction_type"].dropna().unique().tolist()),
        "metrics": {
            "mae":           round(mae, 2),
            "r2":            round(r2, 4),
            "mape":          round(mape, 2),
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

    return pipeline, meta


# =============================================================================
# SECTION 3 — LOAD MODEL & META  (cached in Streamlit session)
# =============================================================================

@st.cache_resource(show_spinner="Loading model…")
def load_model_and_meta():
    """Load pipeline + metadata from disk. Train first if not found."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        pipeline, meta = train_and_save()
        return pipeline, meta
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return pipeline, meta


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# =============================================================================
# SECTION 4 — PREDICTION HELPERS  (called directly, no HTTP)
# =============================================================================

def predict_price(pipeline, meta: dict, inputs: dict) -> tuple[float, str]:
    """Run prediction inline and return (price_in_lakhs, price_category)."""
    all_features = (
        meta["features"]["categorical"]
        + meta["features"]["numeric"]
        + meta["features"]["binary"]
    )
    row = {col: inputs.get(col) for col in all_features}
    df  = pd.DataFrame([row])
    price = float(pipeline.predict(df)[0])
    price = round(max(price, 1.0), 2)

    if price < 50:    cat = "Budget"
    elif price < 150: cat = "Mid-Range"
    elif price < 300: cat = "Premium"
    else:             cat = "Luxury"

    return price, cat


def price_color(price: float) -> str:
    if price < 50:   return "#22c55e"
    if price < 150:  return "#f59e0b"
    if price < 300:  return "#3b82d4"
    return "#7c5cd8"


# =============================================================================
# SECTION 5 — STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #1e3a5f, #3b82d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subheader { color: #57606a; font-size: 1rem; margin-top: 0; }
    .result-card {
        background: #f0f7ff; border-left: 5px solid #3b82d4;
        border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    }
    .price-big { font-size: 2.6rem; font-weight: 800; color: #1e3a5f; }
    .stButton>button {
        background: #1e3a5f; color: white; border-radius: 6px;
        font-size: 1.1rem; padding: 0.6rem 2.5rem;
    }
    .stButton>button:hover { background: #3b82d4; }
</style>
""", unsafe_allow_html=True)

# ── Load model (trains automatically on first run if model not found) ─────────
pipeline, meta = load_model_and_meta()
m = meta.get("metrics", {})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 House Price Predictor")
    st.divider()
    st.markdown("### Model Info")
    st.metric("R² Score",   str(m.get("r2",   "N/A")))
    st.metric("MAE",        f"Rs. {m.get('mae',  'N/A')} L")
    st.metric("MAPE",       f"{m.get('mape', 'N/A')}%")
    st.metric("Train Data", f"{m.get('train_samples', 0):,}")
    st.divider()
    st.markdown("**Algorithm:** XGBoost Regression")
    st.markdown("**Dataset:** 50,000 Indian properties")
    st.markdown("**Cities:** Bangalore, Delhi, Hyderabad, Mumbai")
    st.divider()
    page = st.radio("Navigate", ["🔮 Predict Price", "📊 Data Explorer", "📈 Model Insights"])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏠 House Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">AI-powered property valuation · IBM SkillsBuild Data Analytics Internship</p>',
            unsafe_allow_html=True)
st.divider()


# =============================================================================
# PAGE 1 — PREDICT PRICE
# =============================================================================
if "Predict" in page:
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.subheader("Property Details")

        c1, c2 = st.columns(2)
        city          = c1.selectbox("City",          meta["cities"])
        locality_tier = c2.selectbox("Locality Tier", meta["locality_tiers"])

        c1, c2, c3 = st.columns(3)
        property_type = c1.selectbox("Property Type", meta["property_types"])
        bhk           = c2.selectbox("BHK", [1, 2, 3, 4, 5, 6])
        bathrooms     = c3.number_input("Bathrooms", min_value=1, max_value=8, value=2)

        c1, c2, c3 = st.columns(3)
        built_up_area = c1.number_input("Built-up Area (sq ft)", 200, 10000, 1200, 50)
        carpet_area   = c2.number_input("Carpet Area (sq ft)", 150, 9000,
                                         int(built_up_area * 0.85), 50)
        balconies     = c3.number_input("Balconies", 0, 5, 1)

        c1, c2, c3 = st.columns(3)
        floor_number   = c1.number_input("Floor Number", 0, 60, 3)
        total_floors   = c2.number_input("Total Floors", 1, 60, 10)
        floor_category = c3.selectbox("Floor Category", meta["floor_categories"])

        c1, c2, c3 = st.columns(3)
        facing            = c1.selectbox("Facing",      meta["facings"])
        furnishing_status = c2.selectbox("Furnishing",  meta["furnishing_statuses"])
        transaction_type  = c3.selectbox("Transaction", meta["transaction_types"])

        c1, c2, c3 = st.columns(3)
        property_age   = c1.number_input("Property Age (yrs)", 0, 50, 3)
        parking_spaces = c2.number_input("Parking Spaces", 0, 5, 1)
        security_score = c3.slider("Security Score", 0.0, 10.0, 6.5, 0.5)

        c1, c2 = st.columns(2)
        dist_center = c1.number_input("Distance to City Center (km)", 0.1, 80.0, 10.0, 0.5)
        dist_metro  = c2.number_input("Distance to Metro (km)", 0.1, 20.0, 2.0, 0.5)

        c1, c2, c3 = st.columns(3)
        nearby_schools   = c1.number_input("Nearby Schools",   0, 10, 2)
        nearby_hospitals = c2.number_input("Nearby Hospitals", 0, 10, 1)
        maintenance_fee  = c3.number_input("Maintenance Fee/mo (Rs)", 500, 20000, 3000, 100)

        st.markdown("**Amenities**")
        a1, a2, a3, a4 = st.columns(4)
        gym_available = int(a1.checkbox("Gym"))
        swimming_pool = int(a2.checkbox("Swimming Pool"))
        power_backup  = int(a3.checkbox("Power Backup", value=True))
        lift_available = int(a4.checkbox("Lift", value=True))

        predict_btn = st.button("Predict Price", use_container_width=True)

    with col_right:
        st.subheader("Prediction Result")

        if predict_btn:
            inputs = {
                "city": city, "locality_tier": locality_tier,
                "property_type": property_type, "bhk": bhk,
                "bathrooms": bathrooms, "balconies": balconies,
                "built_up_area": built_up_area, "carpet_area": carpet_area,
                "floor_number": floor_number, "total_floors": total_floors,
                "floor_category": floor_category, "facing": facing,
                "furnishing_status": furnishing_status,
                "transaction_type": transaction_type,
                "property_age": property_age, "parking_spaces": parking_spaces,
                "security_score": security_score,
                "gym_available": gym_available, "swimming_pool": swimming_pool,
                "power_backup": power_backup, "lift_available": lift_available,
                "maintenance_fee_monthly": maintenance_fee,
                "distance_to_city_center_km": dist_center,
                "distance_to_metro_km": dist_metro,
                "nearby_schools": nearby_schools,
                "nearby_hospitals": nearby_hospitals,
            }
            with st.spinner("Predicting…"):
                price, cat = predict_price(pipeline, meta, inputs)

            st.markdown(f"""
            <div class="result-card">
                <p style="color:#57606a;font-size:.85rem;margin-bottom:0">Estimated Market Value</p>
                <p class="price-big" style="color:{price_color(price)}">
                    Rs. {price:,.2f} Lakhs
                </p>
                <p style="color:#57606a;margin-top:-0.5rem">
                    &asymp; Rs. {price/100:,.2f} Crores &nbsp;|&nbsp;
                    Category: <strong>{cat}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            ppsf = (price * 100000) / built_up_area
            st.metric("Price per Sq Ft", f"Rs. {ppsf:,.0f}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=price,
                title={"text": "Price (Lakhs)"},
                number={"prefix": "Rs. ", "suffix": " L"},
                gauge={
                    "axis": {"range": [0, 600]},
                    "bar":  {"color": price_color(price)},
                    "steps": [
                        {"range": [0,   50],  "color": "#d1fae5"},
                        {"range": [50,  150], "color": "#fef9c3"},
                        {"range": [150, 300], "color": "#dbeafe"},
                        {"range": [300, 600], "color": "#ede9fe"},
                    ],
                },
            ))
            fig.update_layout(height=280, margin=dict(t=30, b=0, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(pd.DataFrame({
                "Feature": ["City", "Type", "BHK", "Built-up Area", "Floor", "Age", "Furnishing"],
                "Value":   [city, property_type, str(bhk), f"{built_up_area} sq ft",
                            f"{floor_number}/{total_floors}", f"{property_age} yrs",
                            furnishing_status],
            }), hide_index=True, use_container_width=True)

        else:
            st.info("Fill in property details on the left and click **Predict Price**.")
            sample = np.random.lognormal(mean=4.8, sigma=0.7, size=300)
            fig = px.histogram(x=sample, nbins=40, title="Sample Price Distribution",
                               color_discrete_sequence=["#3b82d4"],
                               labels={"x": "Price (Lakhs)", "y": "Count"})
            fig.update_layout(height=300, showlegend=False,
                              margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 2 — DATA EXPLORER
# =============================================================================
elif "Explorer" in page:
    st.subheader("Dataset Explorer")
    df = load_dataset()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Properties", f"{len(df):,}")
    k2.metric("Avg Price",        f"Rs. {df['price_in_lakhs'].mean():.1f} L")
    k3.metric("Median Price",     f"Rs. {df['price_in_lakhs'].median():.1f} L")
    k4.metric("Cities",           df["city"].nunique())
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        city_avg = df.groupby("city")["price_in_lakhs"].median().reset_index()
        fig = px.bar(city_avg, x="city", y="price_in_lakhs",
                     title="Median Price by City (Lakhs)",
                     color="price_in_lakhs", color_continuous_scale="Blues",
                     labels={"price_in_lakhs": "Median Price (L)", "city": "City"})
        fig.update_layout(showlegend=False, height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x="property_type", y="price_in_lakhs",
                     title="Price Distribution by Property Type",
                     color="property_type",
                     labels={"price_in_lakhs": "Price (L)", "property_type": "Type"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        sample_df = df.sample(min(3000, len(df)), random_state=1)
        fig = px.scatter(sample_df, x="built_up_area", y="price_in_lakhs",
                         color="city", opacity=0.5,
                         title="Built-up Area vs Price",
                         labels={"built_up_area": "Area (sq ft)", "price_in_lakhs": "Price (L)"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        bhk_avg = df.groupby("bhk")["price_in_lakhs"].median().reset_index()
        fig = px.line(bhk_avg, x="bhk", y="price_in_lakhs", markers=True,
                      title="Median Price by BHK",
                      labels={"price_in_lakhs": "Median Price (L)", "bhk": "BHK"},
                      color_discrete_sequence=["#3b82d4"])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Average Price Heatmap: City × Furnishing")
    pivot = df.pivot_table(values="price_in_lakhs", index="city",
                           columns="furnishing_status", aggfunc="median")
    fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="Blues",
                    labels={"color": "Median Price (L)"}, aspect="auto")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Browse Raw Data"):
        city_filter = st.multiselect("Filter by City", df["city"].unique().tolist(),
                                      default=df["city"].unique().tolist()[:2])
        filtered = df[df["city"].isin(city_filter)] if city_filter else df
        st.dataframe(filtered.head(500), use_container_width=True)


# =============================================================================
# PAGE 3 — MODEL INSIGHTS
# =============================================================================
elif "Insights" in page:
    st.subheader("Model Insights")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score",     str(m.get("r2",   "—")))
    c2.metric("MAE",          f"Rs. {m.get('mae',  '—')} L")
    c3.metric("MAPE",         f"{m.get('mape', '—')}%")
    c4.metric("Test Samples", f"{m.get('test_samples', 0):,}")
    st.divider()

    try:
        xgb_model    = pipeline.named_steps["regressor"]
        preprocessor = pipeline.named_steps["preprocessor"]
        num_names    = meta["features"]["numeric"] + meta["features"]["binary"]
        cat_names    = (preprocessor
                        .named_transformers_["cat"]
                        .named_steps["ohe"]
                        .get_feature_names_out(meta["features"]["categorical"])
                        .tolist())
        feat_names   = num_names + cat_names
        importances  = xgb_model.feature_importances_
        n_show       = min(25, len(feat_names))
        idx          = np.argsort(importances)[::-1][:n_show]

        fi_df = pd.DataFrame({
            "Feature":    [feat_names[i] for i in idx],
            "Importance": [importances[i] for i in idx],
        })
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     title=f"Top {n_show} Feature Importances (XGBoost)",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(yaxis={"autorange": "reversed"},
                          height=600, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render feature importance: {e}")

    st.markdown("### Model Architecture")
    arch = {
        "Algorithm":        "XGBoost Regressor",
        "n_estimators":     600,
        "max_depth":        7,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "Preprocessing":    "StandardScaler (numeric) + OneHotEncoder (categorical)",
        "Train / Test":     "80% / 20%",
    }
    st.dataframe(
        pd.DataFrame({"Parameter": arch.keys(),
                      "Value": [str(v) for v in arch.values()]}),
        hide_index=True, use_container_width=True,
    )
