🚀 Exoplanet Prediction App
This project contains a machine learning model and a Streamlit web application to predict the disposition of Kepler Objects of Interest (KOIs) from the NASA Exoplanet Archive. The app classifies an object as CONFIRMED, CANDIDATE, or FALSE POSITIVE based on its astronomical measurements.

The model was trained on the cumulative KOI dataset and uses an XGBoost classifier to achieve high accuracy.

## 📋 Features
Interactive Web App: A user-friendly interface built with Streamlit to get instant predictions.

High-Performance Model: Utilizes a tuned XGBoost model to classify exoplanet candidates.

Dynamic Input: Allows users to adjust key astronomical parameters to see how they affect the prediction.

Clear Results: Provides a direct prediction along with the model's confidence scores for each class.

## ⚙️ How to Use the App
The application is deployed on Streamlit Cloud.

Open the App: Navigate to the Streamlit URL for this project.

Input Features: On the left sidebar, you will find several input fields for the KOI's properties, such as Orbital Period, Transit Depth, and Planetary Radius.

Adjust Values: You can use the default values or enter your own.

Predict: Click the "Predict Disposition" button at the bottom of the sidebar.

View Results: The app will display the final prediction (CONFIRMED, CANDIDATE, or FALSE POSITIVE) and a table showing the model's confidence level for each of the three categories.

## 📁 Repository Contents
app.py: The main Python script that runs the Streamlit web application.

requirements.txt: A list of all the Python libraries required to run the app.

lgbm_exoplanet_model_final.pkl: The pre-trained XGBoost classification model. (Note: Despite the name, this is an XGBoost model).

scaler_final.pkl: The pre-fitted MinMaxScaler object used to normalize the input data before prediction.

nasa-koi-final.ipynb: The Jupyter Notebook containing the complete model training, evaluation, and hyperparameter tuning process.

## 🧠 Model Details
Model Type: XGBoost (Extreme Gradient Boosting) Classifier

Dataset: NASA Exoplanet Archive's cumulative Kepler Objects of Interest (KOI) data.

Target Variable: koi_disposition

Classes:

CONFIRMED: The object is a confirmed exoplanet.

CANDIDATE: The object is a candidate for being an exoplanet but requires further verification.

FALSE POSITIVE: The object is not an exoplanet (e.g., a background star or instrumental noise).
