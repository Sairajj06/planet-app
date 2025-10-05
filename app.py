import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- LOAD THE FINAL, CORRECTED FILES ---
try:
    model = joblib.load('lgbm_exoplanet_model_final.pkl')
    scaler = joblib.load('scaler_final.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'lgbm_exoplanet_model_final.pkl' and 'scaler_final.pkl' are in your GitHub repository.")
    st.stop()

# --- APP TITLE AND DESCRIPTION ---
st.set_page_config(page_title="Exoplanet Classifier", layout="wide")
st.title("🚀 Exoplanet Classification App")
st.write("""
This app predicts the disposition of a Kepler Object of Interest (KOI) using a robustly trained LightGBM model.
Enter the object's properties in the sidebar to get a prediction.
""")

# --- DEFINE THE INPUT FEATURES IN THE SIDEBAR ---
st.sidebar.header("Input Features")

feature_order = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
    'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_sma', 'koi_ror', 'koi_steff', 'koi_slogg', 'koi_srad'
]

user_inputs = {}

user_inputs['koi_period'] = st.sidebar.number_input('Orbital Period (days)', value=9.25, format="%.4f")
user_inputs['koi_depth'] = st.sidebar.number_input('Transit Depth (ppm)', value=350.0, format="%.1f")
user_inputs['koi_duration'] = st.sidebar.number_input('Transit Duration (hours)', value=2.9, format="%.2f")
user_inputs['koi_prad'] = st.sidebar.number_input('Planetary Radius (Earth radii)', value=2.26, format="%.2f")
user_inputs['koi_insol'] = st.sidebar.number_input('Insolation Flux (Earth flux)', value=93.5, format="%.1f")
user_inputs['koi_teq'] = st.sidebar.number_input('Equilibrium Temperature (K)', value=765.0)
user_inputs['koi_impact'] = st.sidebar.number_input('Impact Parameter', value=0.58, format="%.2f")

user_inputs['koi_time0bk'] = 132.6
user_inputs['koi_sma'] = 0.08
user_inputs['koi_ror'] = 0.02
user_inputs['koi_steff'] = 5800.0
user_inputs['koi_slogg'] = 4.5
user_inputs['koi_srad'] = 0.95

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Disposition"):
    input_df = pd.DataFrame([user_inputs], columns=feature_order)
    st.write("### User Input Features:")
    st.dataframe(input_df)

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.write("---")
    st.write("### 🤖 Prediction Result")

    # ✅ FINAL FIX: The model's prediction is already the final string (e.g., 'CONFIRMED').
    # We just need to get the first item from the prediction array.
    result = prediction[0]

    # Display result with color-coded alerts
    if result == 'CONFIRMED':
        st.success(f"The model predicts: **{result}**")
    elif result == 'CANDIDATE':
        st.info(f"The model predicts: **{result}**")
    else:
        st.error(f"The model predicts: **{result}**")

    st.write("### Prediction Confidence")
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=model.classes_,
        index=['Confidence']
    )
    st.dataframe(proba_df.style.format("{:.2%}"))


