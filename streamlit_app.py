from pathlib import Path
import json
import os

import joblib
import pandas as pd
import requests
import streamlit as st

MODEL_PATH = Path("models/hotel_cancellation_model.joblib")
API_URL = os.getenv("API_URL", "").strip()

st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)

st.title("🏨 Hotel Booking Cancellation Predictor")
st.caption(
    "Enter booking information available before arrival to estimate the probability "
    "that the booking will be cancelled."
)

@st.cache_resource
def load_local_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

local_model = load_local_model()

with st.sidebar:
    st.header("About")
    st.write(
        "This application demonstrates an end-to-end machine-learning workflow "
        "for hotel booking cancellation prediction."
    )
    if API_URL:
        st.success("Prediction mode: FastAPI")
    elif local_model is not None:
        st.success("Prediction mode: Local model")
    else:
        st.error("No trained model found.")
    st.info("Run `python train.py` before starting the application.")

st.subheader("Booking Information")

col1, col2, col3 = st.columns(3)

with col1:
    hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=1000, value=60)
    weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=30, value=2)
    week_nights = st.number_input("Week Nights", min_value=0, max_value=60, value=3)
    adults = st.number_input("Adults", min_value=1, max_value=20, value=2)
    children = st.number_input("Children", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    babies = st.number_input("Babies", min_value=0, max_value=10, value=0)

with col2:
    meal = st.selectbox("Meal", ["BB", "HB", "SC", "Undefined", "FB"])
    market_segment = st.selectbox(
        "Market Segment",
        [
            "Online TA",
            "Offline TA/TO",
            "Direct",
            "Groups",
            "Corporate",
            "Complementary",
            "Aviation",
            "Undefined",
        ],
    )
    distribution_channel = st.selectbox(
        "Distribution Channel",
        ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
    )
    repeated_guest = st.selectbox(
        "Repeated Guest",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    previous_cancellations = st.number_input(
        "Previous Cancellations", min_value=0, max_value=100, value=0
    )
    previous_not_cancelled = st.number_input(
        "Previous Non-Cancelled Bookings", min_value=0, max_value=100, value=0
    )

with col3:
    reserved_room_type = st.selectbox(
        "Reserved Room Type",
        ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
    )
    deposit_type = st.selectbox(
        "Deposit Type",
        ["No Deposit", "Non Refund", "Refundable"],
    )
    customer_type = st.selectbox(
        "Customer Type",
        ["Transient", "Transient-Party", "Contract", "Group"],
    )
    adr = st.number_input(
        "Average Daily Rate (ADR)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
    )
    parking = st.number_input(
        "Required Car Parking Spaces", min_value=0, max_value=10, value=0
    )
    special_requests = st.number_input(
        "Total Special Requests", min_value=0, max_value=10, value=1
    )

payload = {
    "hotel": hotel,
    "lead_time": int(lead_time),
    "stays_in_weekend_nights": int(weekend_nights),
    "stays_in_week_nights": int(week_nights),
    "adults": int(adults),
    "children": float(children),
    "babies": int(babies),
    "meal": meal,
    "market_segment": market_segment,
    "distribution_channel": distribution_channel,
    "is_repeated_guest": int(repeated_guest),
    "previous_cancellations": int(previous_cancellations),
    "previous_bookings_not_canceled": int(previous_not_cancelled),
    "reserved_room_type": reserved_room_type,
    "deposit_type": deposit_type,
    "customer_type": customer_type,
    "adr": float(adr),
    "required_car_parking_spaces": int(parking),
    "total_of_special_requests": int(special_requests),
}

st.divider()

if st.button("🔍 Predict Cancellation Risk", type="primary", use_container_width=True):
    try:
        if API_URL:
            response = requests.post(
                f"{API_URL.rstrip('/')}/predict",
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()
            probability = float(result["cancellation_probability"])
            prediction = int(result["prediction"])
            risk = result["risk_level"]
        else:
            if local_model is None:
                st.error("Model not found. Run: python train.py")
                st.stop()

            row = pd.DataFrame([payload])
            probability = float(local_model.predict_proba(row)[0, 1])
            prediction = int(probability >= 0.5)

            if probability >= 0.70:
                risk = "High"
            elif probability >= 0.40:
                risk = "Medium"
            else:
                risk = "Low"

        st.subheader("Prediction Result")
        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Prediction",
            "Likely to Cancel" if prediction == 1 else "Likely to Keep Booking",
        )
        m2.metric("Cancellation Probability", f"{probability * 100:.1f}%")
        m3.metric("Risk Level", risk)

        st.progress(min(max(probability, 0.0), 1.0))

        if risk == "High":
            st.error(
                "High cancellation risk. The hotel may consider proactive confirmation "
                "or appropriate retention actions."
            )
        elif risk == "Medium":
            st.warning(
                "Moderate cancellation risk. Consider monitoring or confirming the booking."
            )
        else:
            st.success("Low predicted cancellation risk.")

        with st.expander("View submitted booking data"):
            st.json(payload)

    except requests.RequestException as exc:
        st.error(f"Could not contact prediction API: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Academic demonstration only. Predictions should not be used as the sole basis "
    "for decisions affecting customers."
)
