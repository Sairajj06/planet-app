import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os

# --- FUNCTION TO DOWNLOAD FILES FROM GITHUB ---
def download_file_from_github(url, local_filename):
    if os.path.exists(local_filename):
        return
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(local_filename, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {local_filename}: {e}")
        st.stop()

# --- DEFINE GITHUB RAW FILE URLS AND LOCAL PATHS ---
# 🚨 IMPORTANT: Replace these with the RAW URLs for your first model
MODEL_URL = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPOSITORY/main/lgbm_exoplanet_model_final.pkl'
SCALER_URL = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPOSITORY/main/scaler_final.pkl'
MODEL_PATH = 'lgbm_exoplanet_model_final.pkl'
SCALER_PATH = 'scaler_final.pkl'

# --- DOWNLOAD AND LOAD FILES ---
download_file_from_github(MODEL_URL, MODEL_PATH)
download_file_from_github(SCALER_URL, SCALER_PATH)

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# --- APP TITLE AND DESCRIPTION ---
st.set_page_config(page_title="Exoplanet Classifier", layout="wide")
st.title("🚀 Exoplanet Classification App")
st.write("This app predicts the disposition of a Kepler Object of Interest (KOI).")


# --- ✅ FIX: DEFINE ALL FEATURE COLUMNS IN THE CORRECT ORDER ---
# This list must exactly match the columns used to train the model.
ALL_FEATURE_COLUMNS = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period',
    'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1',
    'koi_time0bk_err2', 'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
    'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 'koi_depth',
    'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
    'koi_teq', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr',
    'koi_tce_plnt_num', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 'koi_slogg',
    'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2',
    'ra', 'dec', 'koi_kepmag'
]

# --- DEFINE THE INPUT FEATURES IN THE SIDEBAR ---
st.sidebar.header("Adjustable Input Features")
user_inputs = {}
user_inputs['koi_period'] = st.sidebar.number_input('Orbital Period (days)', value=9.25, format="%.4f")
user_inputs['koi_depth'] = st.sidebar.number_input('Transit Depth (ppm)', value=350.0, format="%.1f")
user_inputs['koi_duration'] = st.sidebar.number_input('Transit Duration (hours)', value=2.9, format="%.2f")
user_inputs['koi_prad'] = st.sidebar.number_input('Planetary Radius (Earth radii)', value=2.26, format="%.2f")
user_inputs['koi_insol'] = st.sidebar.number_input('Insolation Flux (Earth flux)', value=93.5, format="%.1f")
user_inputs['koi_teq'] = st.sidebar.number_input('Equilibrium Temperature (K)', value=765.0)
user_inputs['koi_impact'] = st.sidebar.number_input('Impact Parameter', value=0.58, format="%.2f")
user_inputs['koi_model_snr'] = st.sidebar.number_input('Transit Signal-to-Noise', value=25.8, format="%.1f")


# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Disposition"):
    try:
        # Define the mapping from numbers back to class names.
        class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']

        # ✅ FIX: Create a DataFrame with all required columns, initialized to 0.
        input_data = pd.DataFrame(0, index=[0], columns=ALL_FEATURE_COLUMNS)

        # ✅ FIX: Update the DataFrame with the user's specific input values.
        for key, value in user_inputs.items():
            if key in input_data.columns:
                input_data[key] = value

        st.write("### Input Features (showing user-adjustable values):")
        st.dataframe(pd.DataFrame([user_inputs]))

        # Scale the complete DataFrame and make the prediction.
        scaled_input = scaler.transform(input_data)
        prediction_numeric = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        st.write("---")
        st.write("### 🤖 Prediction Result")

        result_name = class_names[prediction_numeric[0]]

        if result_name == 'CONFIRMED':
            st.success(f"The model predicts: **{result_name}**")
        elif result_name == 'CANDIDATE':
            st.info(f"The model predicts: **{result_name}**")
        else:
            st.error(f"The model predicts: **{result_name}**")

        st.write("### Prediction Confidence")
        proba_df = pd.DataFrame(prediction_proba, columns=class_names, index=['Confidence'])
        st.dataframe(proba_df.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"An error occurred: {e}")
