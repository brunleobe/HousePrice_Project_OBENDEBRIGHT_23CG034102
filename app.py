import streamlit as st
import joblib
import numpy as np
import os

st.title("🏠 House Price Predictor")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model/house_price_model.pkl')

try:
    model = load_model()
    
    # Inputs
    q = st.number_input("Overall Quality", value=5.0)
    liv = st.number_input("Living Area sqft", value=2000.0)
    bsmt = st.number_input("Basement sqft", value=1000.0)
    cars = st.number_input("Garage Cars", value=2.0)
    bath = st.number_input("Full Baths", value=2.0)
    year = st.number_input("Year Built", value=2000.0)

    if st.button("Predict Price"):
        features = np.array([[q, liv, bsmt, cars, bath, year]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Sale Price: ${prediction:,.2f}")

except Exception as e:
    st.error(f"Please run model_building.ipynb first to generate the model file. Error: {e}")