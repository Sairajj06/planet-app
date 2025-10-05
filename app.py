import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os

# --- CRITICAL: Ensure these file names match what you have on GitHub ---
MODEL_FILENAME = 'lgbm_exoplanet_model_final.pkl'
SCALER_FILENAME = 'scaler_final.pkl'

@st.cache_resource
def load_model_and_scaler(model_url, scaler_url):
    """Downloads and loads the model and scaler, caching them to prevent re-downloading."""
    def download_file(url, filename):
        if not os.path.exists(filename):
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(filename, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                st.error(f"Error downloading {filename}: {e}")
                st.stop()
    
    download_file(model_url, MODEL_FILENAME)
    download_file(scaler_url, SCALER_FILENAME)

    try:
        model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler files: {e}")
        st.stop()

# --- App Configuration ---
st.set_page_config(page_title="Exoplanet Classifier", layout="wide")
st.title("🚀 Exoplanet Classification App")
st.write("This app predicts the disposition of a Kepler Object of Interest (KOI).")

# --- Load Model (this will only run once) ---
# 🚨 IMPORTANT: Replace these with the RAW URLs for your FIRST model
MODEL_URL = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPOSITORY/main/lgbm_exoplanet_model_final.pkl'
SCALER_URL = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPOSITORY/main/scaler_final.pkl'
model, scaler = load_model_and_scaler(MODEL_URL, SCALER_URL)

# --- Sidebar for User Input ---
st.sidebar.header("Input Features")
user_inputs = {
    'koi_period': st.sidebar.number_input('Orbital Period (days)', value=129.9, format="%.4f"),
    'koi_depth': st.sidebar.number_input('Transit Depth (ppm)', value=113.0, format="%.1f"),
    'koi_duration': st.sidebar.number_input('Transit Duration (hours)', value=4.4, format="%.2f"),
    'koi_prad': st.sidebar.number_input('Planetary Radius (Earth radii)', value=1.17, format="%.2f"),
    'koi_insol': st.sidebar.number_input('Insolation Flux (Earth flux)', value=0.22, format="%.1f"),
    'koi_teq': st.sidebar.number_input('Equilibrium Temperature (K)', value=188.0),
    'koi_impact': st.sidebar.number_input('Impact Parameter', value=0.23, format="%.2f"),
    'koi_model_snr': st.sidebar.number_input('Transit Signal-to-Noise', value=25.0, format="%.1f")
}

if st.sidebar.button("Predict Disposition"):
    try:
        class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
        
        # Get the number of features the scaler expects
        n_features = scaler.n_features_in_
        
        # Create a zero array with the correct shape
        input_array = np.zeros(n_features)
        
        # Get user values from the UI
        user_values = list(user_inputs.values())
        
        # Fill the start of the array with user values
        input_array[:len(user_values)] = user_values
        
        # Reshape for the scaler (expects a 2D array)
        input_array_2d = input_array.reshape(1, -1)
        
        # Display simplified user input
        st.write("### User Input Features:")
        st.dataframe(pd.DataFrame([user_inputs]))

        # Scale the data and predict
        scaled_input = scaler.transform(input_array_2d)
        prediction_num = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)
        
        # Convert numeric prediction to class name
        result_name = class_names[prediction_num[0]]

        st.write("---")
        st.write("### 🤖 Prediction Result")

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
