import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load the saved model, encoder, and scaler ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first.")
    st.stop()

# --- Function to load data for dropdowns ---
# This helps cache the data so it doesn't reload every time.


@st.cache_data
def load_data():
    # This assumes you have a final CSV of your cleaned data.
    # You should save your 'df_final_cleaned' DataFrame to a CSV in your notebook.
    try:
        df = pd.read_csv('data/After_EDA_crop_data.csv')
        return df
    except FileNotFoundError:
        st.error(
            "Could not find 'final_cleaned_crop_data.csv'. Please save it from your training notebook.")
        return None


df_final_cleaned = load_data()

# --- Streamlit User Interface ---
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")
st.title("🌾 Indian Crop Yield Predictor")
st.write("Enter the details below to predict the crop yield for a given area.")

# --- Get unique values for dropdowns from the original dataset ---
if df_final_cleaned is not None:
    states = sorted(df_final_cleaned['State'].unique())
    districts = sorted(df_final_cleaned['District'].unique())
    seasons = sorted(df_final_cleaned['Season'].unique())
    crops = sorted(df_final_cleaned['Crop'].unique())
else:
    # Fallback to placeholder lists if the file isn't found
    states, districts, seasons, crops = [], [], [], []


# --- User Inputs in Columns ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Location")
    state = st.selectbox("State", states)
    # The selectbox in Streamlit automatically includes a search feature.
    district = st.selectbox("District", districts)

with col2:
    st.header("Crop Details")
    season = st.selectbox("Season", seasons)
    crop = st.selectbox("Crop", crops)
    area = st.number_input(
        "Area (Hectares)", min_value=1.0, value=100.0, step=10.0)

with col3:
    st.header("Environmental Factors")
    rainfall = st.number_input(
        "Annual Rainfall (mm)", min_value=0.0, value=1000.0)
    fertilizer = st.number_input(
        "Fertilizer Usage (kg)", min_value=0.0, value=5000.0)
    pesticide = st.number_input(
        "Pesticide Usage (kg)", min_value=0.0, value=1000.0)
    ph = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)

# --- Prediction Logic ---
if st.button("Predict Crop Yield", use_container_width=True):

    # Create a DataFrame from user inputs
    input_df = pd.DataFrame({
        'State': [state],
        'District': [district],
        'Year': [2024],  # Using a placeholder for the year
        'Season': [season],
        'Crop': [crop],
        'Area_Hectare': [area],
        'Rainfall_mm': [rainfall],
        'Fertilizer_kg': [fertilizer],
        'Pesticide_kg': [pesticide],
        'pH': [ph]
    })

    # --- Preprocessing ---
    # 1. Encode categorical features
    categorical_features = ['State', 'District', 'Season', 'Crop']
    input_df[categorical_features] = encoder.transform(
        input_df[categorical_features])

    # 2. Scale all features
    input_scaled = scaler.transform(input_df)

    # --- Prediction ---
    prediction = model.predict(input_scaled)
    predicted_yield = prediction[0]

    # --- Display Results ---
    st.subheader("Prediction Result")
    st.metric(
        label="Predicted Crop Yield",
        value=f"{predicted_yield:.2f} Tonnes/Hectare"
    )

    st.info("Note: This prediction is based on historical data and serves as an estimate. Actual yield can be affected by various real-world factors.")
