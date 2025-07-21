import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load the Dataset for Slider Configuration ---
# This loads the data to dynamically set the min, max, and default values for the sliders.
# This fixes the "NameError: name 'data' is not defined"
try:
    data = pd.read_csv('data.csv')
    # Drop non-feature columns to get the correct feature list and ranges
    if 'Unnamed: 32' in data.columns:
        data = data.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)
    else:
        data = data.drop(['id', 'diagnosis'], axis=1)
    feature_names = data.columns.tolist()
except FileNotFoundError:
    st.error("Error: `data.csv` not found. Please make sure it's in the same directory.")
    data = None # Set to None if file is not found to prevent further errors
    feature_names = []


# --- 2. Load the Saved Model and Scaler ---
@st.cache_resource
def load_model_assets():
    """Load the trained model and scaler."""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model_assets()

# --- 3. Set Up the Page ---
st.set_page_config(page_title="Breast Cancer Diagnosis Predictor", layout="wide")
st.title("Breast Cancer Diagnosis Predictor 🩺")
st.markdown("""
This application uses a Random Forest model to predict whether a breast tumor is **Benign** or **Malignant** based on its measurements.
Please input the tumor's features in the sidebar to get a prediction.
""")

# --- 4. Check if All Assets are Loaded ---
if model is None or scaler is None or data is None:
    st.error("🔴 Error: One or more required files (`model.pkl`, `scaler.pkl`, `data.csv`) are missing. Please ensure they are in the correct directory.")
else:
    # --- 5. Create Sidebar for User Input ---
    st.sidebar.header("Tumor Features")
    st.sidebar.markdown("Adjust the sliders to input the tumor measurements.")

    # Use a dictionary to store user inputs
    input_dict = {}
    for feature in feature_names:
        input_dict[feature] = st.sidebar.slider(
            label=feature.replace('_', ' ').title(),
            min_value=float(data[feature].min()),
            max_value=float(data[feature].max()),
            value=float(data[feature].mean()) # Default to the mean value
        )

    # --- 6. Prediction Logic ---
    if st.sidebar.button("Predict Diagnosis"):
        # Convert the dictionary to a DataFrame
        input_df = pd.DataFrame([input_dict])

        # Ensure the column order is the same as during training
        input_df = input_df[feature_names]

        # Scale the user input using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # --- 7. Display the Result ---
        st.subheader("Prediction Result")
        
        if prediction[0] == 1: # Malignant
            st.error("Prediction: **Malignant**")
            st.write(f"Confidence: {prediction_proba[0][1]:.2%}")
    
        else: # Benign
            st.success("Prediction: **Benign**")
            st.write(f"Confidence: {prediction_proba[0][0]:.2%}")
            
        st.subheader("Input Features")
        st.write(input_df)
