import streamlit as st
import pandas as pd
import joblib

# Load the trained model (saved from Jupyter notebook)
model = joblib.load('gradient_boosting_model.pkl')  # Make sure this file is in the same folder

# Page setup
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 Zameen House Price Predictor (No Location)")
st.markdown("Estimate house prices based on features like area, bedrooms, bathrooms, etc.")

# --- User Input Section ---
with st.form("prediction_form"):
    st.subheader("Enter House Features")

    area = st.number_input("Area (in Marla)", min_value=1.0, max_value=1000.0, value=5.0)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    baths = st.slider("Bathrooms", 1, 10, 2)
    house_age = st.number_input("House Age (in years)", min_value=0, max_value=100, value=5)
    latitude = st.number_input("Latitude", value=33.6844, format="%.4f")
    longitude = st.number_input("Longitude", value=73.0479, format="%.4f")

    submitted = st.form_submit_button("Predict Price")

# --- Prediction ---
if submitted:
    input_data = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'baths': baths,
        'house_age': house_age,
        'latitude': latitude,
        'longitude': longitude
    }])

    prediction = model.predict(input_data)[0]
    price_million = prediction / 1_000_000

    st.success(f"🏷️ Estimated Price: **PKR {price_million:,.2f} Million**")


