import streamlit as st
import joblib
import numpy as np
import os

# -------------------------------
# Load Model and Preprocessing Objects
# -------------------------------


# Load the model, scaler, and encoder
model = joblib.load("uber_trip.pkl")
scaler = joblib.load("scaler.pkl")
le_dispatching = joblib.load("label_encoder.pkl")

# Available dispatch base numbers for dropdown
dispatching_options = le_dispatching.classes_.tolist()

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Trip Prediction App", layout="centered")

st.title("🚖 Uber Trip Prediction App")
st.markdown("### Predict the number of trips based on daily active vehicles and dispatch base number")

# Input Fields
dispatching_base_number = st.selectbox("Select Dispatching Base Number", dispatching_options)
active_vehicles = st.number_input("Active Vehicles", min_value=0, step=1)
day = st.number_input("Day", min_value=1, max_value=31, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)
year = st.number_input("Year", min_value=2000, max_value=2100, step=1)

# Predict Button
if st.button("🔍 Predict Trips"):
    try:
        # Encode dispatch base number
        encoded_dispatch = le_dispatching.transform([dispatching_base_number])[0]

        # Prepare features
        features = np.array([[encoded_dispatch, active_vehicles, day, month, year]])

        # Scale input
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]

        # Display result
        st.success(f"🚗 **Predicted Number of Trips:** {prediction:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and your trained Gradient Boosting model.")
