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
# 🚨 IMPORTANT: Replace these with the RAW URLs of your files on GitHub
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
st.write("This app predicts the disposition of a Kepler Object of Interest (KOI) using a pre-trained model.")

# --- DEFINE THE INPUT FEATURES IN THE SIDEBAR ---
st.sidebar.header("Adjustable Input Features")
ui_features = {
    'koi_period': st.sidebar.number_input('Orbital Period (days)', value=9.25, format="%.4f"),
    'koi_depth': st.sidebar.number_input('Transit Depth (ppm)', value=350.0, format="%.1f"),
    'koi_duration': st.sidebar.number_input('Transit Duration (hours)', value=2.9, format="%.2f"),
    'koi_prad': st.sidebar.number_input('Planetary Radius (Earth radii)', value=2.26, format="%.2f"),
    'koi_insol': st.sidebar.number_input('Insolation Flux (Earth flux)', value=93.5, format="%.1f"),
    'koi_teq': st.sidebar.number_input('Equilibrium Temperature (K)', value=765.0),
    'koi_impact': st.sidebar.number_input('Impact Parameter', value=0.58, format="%.2f"),
    'koi_model_snr': st.sidebar.number_input('Transit Signal-to-Noise', value=25.8, format="%.1f")
}

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Disposition"):
    try:
        # ✅ FIX #1: Define the mapping from numbers back to class names.
        # This order is based on how LabelEncoder sorts the names alphabetically.
        class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']

        # Get the number of features the scaler expects.
        n_expected_features = scaler.n_features_in_
        
        # Get user values from the UI.
        user_values = list(ui_features.values())
        
        # Create an array of zeros with the correct shape.
        input_array = np.zeros(n_expected_features)
        
        # Fill the start of the array with user values.
        input_array[:len(user_values)] = user_values
        
        # Reshape for the scaler.
        input_array_reshaped = input_array.reshape(1, -1)
        
        # Display simplified user input.
        st.write("### User Input Features:")
        st.dataframe(pd.DataFrame([ui_features]))

        # Scale the input and make the prediction.
        scaled_input = scaler.transform(input_array_reshaped)
        prediction_numeric = model.predict(scaled_input) # This will be [0], [1], or [2]
        prediction_proba = model.predict_proba(scaled_input)

        st.write("---")
        st.write("### 🤖 Prediction Result")

        # ✅ FIX #2: Use the numeric prediction to look up the text label.
        result_name = class_names[prediction_numeric[0]]

        # Display the final text result with color-coding.
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
        st.error(f"An error occurred during prediction: {e}")
