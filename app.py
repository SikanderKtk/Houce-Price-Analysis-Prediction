import streamlit as st
import pandas as pd
import joblib

# Load trained model (make sure rf_model.pkl is in the same directory)
model = joblib.load('gradient_boosting_model.pkl')

# Set Streamlit page configuration
st.set_page_config(
    page_title="House Price Predictor",
    layout="centered",
    page_icon="🏡"
)

# --- Main Title & Description ---
st.markdown("""
    <h1 style='text-align: center; color: #004d99;'>🏡 Zameen House Price Predictor</h1>
    <p style='text-align: center; font-size: 18px;'>
        Predict property prices in Pakistan based on features like area, bedrooms, bathrooms, and more.
        Built using Machine Learning & trained on real data from Zameen.com.
    </p>
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("🛠️ Configure House Features")
st.sidebar.markdown("Adjust the inputs below to estimate the house price.")

# --- Location Dropdown (for UI only, not used in model)
st.sidebar.selectbox(
    "📍 Select Location of Property",
    ['LAHORE', 'Bahria Town', 'Islamabad', 'Peshawar', 'Pindi', 'Karachi', 'Gulberg', 'Faisal Town']
)

# --- Input Form ---
area = st.sidebar.slider("📏 Area (Marla)", 1, 1000, 5)
bedrooms = st.sidebar.slider("🛏️ Bedrooms", 1, 10, 3)
baths = st.sidebar.slider("🛁 Bathrooms", 1, 10, 2)
house_age = st.sidebar.slider("🏗️ House Age (Years)", 0, 100, 5)
latitude = st.sidebar.number_input("🌍 Latitude", value=33.6844, format="%.4f")
longitude = st.sidebar.number_input("🌍 Longitude", value=73.0479, format="%.4f")

# --- Predict Button ---
if st.sidebar.button("🔍 Predict Price"):
    input_df = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'baths': baths,
        'house_age': house_age,
        'latitude': latitude,
        'longitude': longitude
    }])

    prediction = model.predict(input_df)[0]
    price_million = prediction / 1_000_000

    st.markdown("""
        <div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px; border: 1px solid #99ccff;'>
            <h2 style='color: #004d99;'>🏷️ Predicted House Price</h2>
            <p style='font-size: 24px; color: #006600;'>PKR {:.2f} Million</p>
        </div>
    """.format(price_million), unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <hr style='border: 1px solid #ccc;'>
    <p style='text-align: center;'>
        Created with ❤️ using <a href='https://streamlit.io' target='_blank'>Streamlit</a> | by <a href='www.linkedin.com/in/m-sikander-bakht' target='_blank'>SikanderKtk</a> | Trained on real data from <strong>Zameen.com</strong>
    </p>
""", unsafe_allow_html=True)


