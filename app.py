import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="CarResale AI | Price & Category Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
    <style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.5);
    }
    .main-title {
        color: #38bdf8;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Prediction Result Container */
    .result-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 2px solid #6366f1;
        border-radius: 16px;
        padding: 28px;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
    }
    
    .price-display {
        font-size: 2.8rem;
        font-weight: 900;
        color: #34d399;
        margin: 10px 0;
    }

    .badge-high {
        background-color: #6366f1;
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    .badge-budget {
        background-color: #f59e0b;
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
    }

    /* Primary Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #0284c7 0%, #2563eb 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 12px;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
        margin-top: 15px;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #0369a1 0%, #1d4ed8 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model_path = 'car_price_models.joblib'
    if not os.path.exists(model_path):
        from train_models import train_and_save_models
        train_and_save_models()
    return joblib.load(model_path)

artifacts = load_model_artifacts()

lr_model = artifacts['lr_model']
log_model = artifacts['log_model']
scaler = artifacts['scaler']
encoder = artifacts.get('encoder')
feature_columns = artifacts['feature_columns']
median_price = artifacts['median_price']
brand_models = artifacts['brand_models']
fuel_types = artifacts['fuel_types']
transmissions = artifacts['transmissions']

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown("""
    <div class="main-header">
        <div class="main-title">🚗 Used Car Resale Price & Category Predictor</div>
        <div class="subtitle">Select the vehicle parameters below and click Predict to view the resale price and logistic price classification.</div>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# VEHICLE CONFIGURATION FORM
# -------------------------------------------------------------
st.subheader("🛠️ Vehicle Details")

# Row 1: Brand & Model Dropdowns
r1_1, r1_2 = st.columns(2)
with r1_1:
    selected_brand = st.selectbox("Car Brand", list(brand_models.keys()), index=0)
with r1_2:
    available_models = brand_models.get(selected_brand, ["Standard Model"])
    selected_model = st.selectbox("Car Model", available_models, index=0)

# Row 2: Manufacturing Year & Kilometers Driven Dropdowns
r2_1, r2_2 = st.columns(2)
with r2_1:
    years_options = list(range(2025, 2007, -1)) # 2025 down to 2008
    selected_year = st.selectbox("Manufacturing Year", years_options, index=5) # Default 2020
    car_age = 2026 - selected_year
with r2_2:
    kms_options = [5000, 10000, 15000, 20000, 30000, 45000, 60000, 75000, 90000, 100000, 120000, 150000, 200000]
    driven_kms = st.selectbox("Kilometers Driven", kms_options, index=5, format_func=lambda x: f"{x:,} km")

# Row 3: Fuel Type & Transmission Dropdowns
r3_1, r3_2 = st.columns(2)
with r3_1:
    fuel_type = st.selectbox("Fuel Type", fuel_types, index=0)
with r3_2:
    transmission = st.selectbox("Transmission", transmissions, index=0)

# Row 4: Engine Size & Mileage Dropdowns
r4_1, r4_2 = st.columns(2)
with r4_1:
    if fuel_type == "Electric":
        engine_size = 0.0
        st.selectbox("Engine Size (L)", ["0.0 L (Electric)"], disabled=True)
    else:
        engine_options = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5]
        engine_size = st.selectbox("Engine Size (L)", engine_options, index=3, format_func=lambda x: f"{x:.1f} L")

with r4_2:
    mileage_options = [8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 24.0, 25.0, 28.0, 30.0]
    mileage = st.selectbox("Mileage (kmpl)", mileage_options, index=7, format_func=lambda x: f"{x:.1f} kmpl")

# Predict Button
predict_btn = st.button("🚀 Predict Car Resale Price", use_container_width=True)

# -------------------------------------------------------------
# PREDICTION OUTPUT AREA (DISPLAYED ONLY ON BUTTON CLICK)
# -------------------------------------------------------------
if predict_btn:
    # 1. Prepare Numerical & Categorical Input DataFrames
    num_input = pd.DataFrame([{
        'Year': selected_year,
        'Driven_KMs': driven_kms,
        'Engine_Size_L': engine_size,
        'Mileage_kmpl': mileage,
        'Car_Age': car_age
    }])
    
    cat_input = pd.DataFrame([{
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Brand': selected_brand
    }])

    # 2. One-Hot Encode using fitted OneHotEncoder
    if encoder is not None:
        encoded_cat = encoder.transform(cat_input)
        full_input = pd.concat([num_input, encoded_cat], axis=1)
        full_input = full_input[feature_columns]
    else:
        input_data = {
            'Year': selected_year,
            'Driven_KMs': driven_kms,
            'Engine_Size_L': engine_size,
            'Mileage_kmpl': mileage,
            'Car_Age': car_age,
            'Fuel_Type': fuel_type,
            'Transmission': transmission,
            'Brand': selected_brand
        }
        input_df = pd.DataFrame([input_data])
        encoded_input = pd.get_dummies(input_df, drop_first=False, dtype=float)
        full_input = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        for col in encoded_input.columns:
            if col in full_input.columns:
                full_input[col] = encoded_input[col].values

    # 3. Linear Regression Prediction (Continuous Price)
    lr_pred_val = lr_model.predict(full_input)[0]
    lr_pred_val = max(50000.0, lr_pred_val)

    # 4. Logistic Regression Prediction (Category Tier & Confidence)
    full_input_scaled = scaler.transform(full_input)
    log_class_pred = log_model.predict(full_input_scaled)[0]
    log_prob_pred = log_model.predict_proba(full_input_scaled)[0][1]

    # Badge HTML
    if log_class_pred == 1:
        badge_html = f'<span class="badge-high">🔥 High Price Tier (Above ₹{median_price/100000:.2f} Lakhs)</span>'
    else:
        badge_html = f'<span class="badge-budget">🏷️ Budget Price Tier (Below ₹{median_price/100000:.2f} Lakhs)</span>'

    # Render Prediction Card
    st.markdown(f"""
        <div class="result-container">
            <div style="text-align: center;">
                <span style="color: #94a3b8; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                    Predicted Car Resale Price
                </span>
                <div class="price-display">
                    ₹{lr_pred_val:,.2f}
                </div>
                <div style="color: #38bdf8; font-size: 1.2rem; font-weight: 700; margin-bottom: 20px;">
                    (~{(lr_pred_val / 100000.0):.2f} Lakhs INR)
                </div>
                <hr style="border: 0; height: 1px; background: rgba(255,255,255,0.15); margin: 20px 0;">
                <div style="margin-bottom: 12px;">
                    <span style="color: #cbd5e1; font-size: 1rem; font-weight: 600;">Logistic Regression Model Classification:</span>
                </div>
                <div style="margin-bottom: 15px;">
                    {badge_html}
                </div>
                <div style="color: #a5b4fc; font-size: 0.95rem; font-weight: 500;">
                    Logistic High-Tier Probability Confidence: <b>{log_prob_pred * 100:.1f}%</b>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("💡 Select the vehicle details above and click **'🚀 Predict Car Resale Price'** to compute the prediction.")
