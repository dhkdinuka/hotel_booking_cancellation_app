from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = Path("models/hotel_cancellation_model.joblib")
METADATA_PATH = Path("models/metadata.json")

st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ===========================
       MAIN PAGE
    =========================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef6ff 45%,
            #fff7ed 100%
        );
    }

    .main-title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 4px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #1e3a8a;
        margin-top: 10px;
        margin-bottom: 15px;
    }


    /* ===========================
       BUTTON
    =========================== */

    div.stButton > button {
        background: linear-gradient(
            90deg,
            #4f46e5,
            #2563eb,
            #0891b2
        );

        color: white;
        border: none;
        border-radius: 14px;
        height: 56px;
        font-size: 18px;
        font-weight: 700;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        color: white;
        transform: scale(1.01);
        box-shadow: 0 6px 18px rgba(37,99,235,0.25);
    }


    /* ===========================
       SIDEBAR
    =========================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #172554 0%,
            #1e40af 45%,
            #0e7490 100%
        );
    }

    section[data-testid="stSidebar"] > div {
        min-height: 100vh;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }


    /* Sidebar title */

    .sidebar-title {
        text-align: center;
        font-size: 25px;
        font-weight: 900;
        margin-top: 10px;
        margin-bottom: 24px;
    }


    /* Sidebar card */

    .sidebar-card {
        background: rgba(255, 255, 255, 0.12);

        border:
            1px solid
            rgba(255, 255, 255, 0.24);

        border-radius: 15px;

        padding: 18px 18px;

        margin-bottom: 15px;

        box-shadow:
            0 4px 12px
            rgba(0,0,0,0.08);
    }


    /* Card heading */

    .sidebar-heading {
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 11px;
        color: white;
    }


    /* Card text */

    .sidebar-text {
        font-size: 14px;
        line-height: 1.65;
        color: #f8fafc;
    }


    /* Model rows */

    .model-row {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 8px 10px;
        margin-top: 7px;
        font-size: 14px;
        font-weight: 600;
    }


    /* Selected model */

    .selected-model {
        font-size: 18px;
        font-weight: 900;
        color: #fef08a;
        margin-bottom: 7px;
    }


    /* Pipeline */

    .pipeline-flow {
        text-align: center;
        font-size: 14px;
        line-height: 1.85;
        font-weight: 600;
    }


    /* Sidebar footer */

    .sidebar-footer {
        text-align: center;
        font-size: 13px;
        line-height: 1.7;
        padding-top: 14px;
        color: #e0f2fe;
    }


    /* Main footer */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


model = load_model()


# =========================================================
# LOAD MODEL METADATA
# =========================================================

@st.cache_data
def load_metadata():

    if METADATA_PATH.exists():

        try:

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(file)

        except Exception:
            return {}

    return {}


metadata = load_metadata()

best_model = metadata.get(
    "best_model",
    "Best F1 Model",
)

best_model_display = (
    best_model
    .replace("_", " ")
    .title()
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div class='main-title'>"
    "🏨 Hotel Booking Cancellation Predictor"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='subtitle'>"
    "Machine Learning Based Hotel Cancellation Risk Prediction"
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    st.markdown(
        "<div class='sidebar-title'>"
        "🤖 AI Prediction System"
        "</div>",
        unsafe_allow_html=True,
    )


    # -----------------------------------------------------
    # PROJECT PURPOSE
    # -----------------------------------------------------

    st.markdown(
        "<div class='sidebar-card'>"
        "<div class='sidebar-heading'>🎯 Project Purpose</div>"
        "<div class='sidebar-text'>"
        "Predict the probability that a hotel booking "
        "may be cancelled using historical reservation data."
        "<br><br>"
        "The system helps identify bookings with elevated "
        "cancellation risk before the customer's arrival."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


    # -----------------------------------------------------
    # MODELS EVALUATED
    # -----------------------------------------------------

    st.markdown(
        "<div class='sidebar-card'>"
        "<div class='sidebar-heading'>🧠 Models Evaluated</div>"

        "<div class='model-row'>"
        "📘 Logistic Regression"
        "</div>"

        "<div class='model-row'>"
        "🌳 Random Forest"
        "</div>"

        "<div class='model-row'>"
        "🚀 Gradient Boosting"
        "</div>"

        "</div>",
        unsafe_allow_html=True,
    )


    # -----------------------------------------------------
    # SELECTED MODEL
    # -----------------------------------------------------

    st.markdown(
        "<div class='sidebar-card'>"
        "<div class='sidebar-heading'>🏆 Selected Model</div>"

        f"<div class='selected-model'>"
        f"{best_model_display}"
        f"</div>"

        "<div class='sidebar-text'>"
        "Automatically selected using the "
        "<b>highest F1 score</b> during model evaluation."
        "</div>"

        "</div>",
        unsafe_allow_html=True,
    )


    # -----------------------------------------------------
    # ML PIPELINE
    # -----------------------------------------------------

    st.markdown(
        "<div class='sidebar-card'>"
        "<div class='sidebar-heading'>⚙️ Machine Learning Pipeline</div>"

        "<div class='pipeline-flow'>"

        "📂 Booking Data"
        "<br>↓<br>"

        "🧹 Data Cleaning"
        "<br>↓<br>"

        "🔤 Encoding + Scaling"
        "<br>↓<br>"

        "🧠 Classification Model"
        "<br>↓<br>"

        "📈 Cancellation Probability"

        "</div>"

        "</div>",
        unsafe_allow_html=True,
    )


    # -----------------------------------------------------
    # MODEL STATUS
    # -----------------------------------------------------

    if model is not None:

        st.success(
            "✅ Model Ready for Prediction"
        )

    else:

        st.error(
            "❌ Model Not Found"
        )


    # -----------------------------------------------------
    # SIDEBAR FOOTER
    # -----------------------------------------------------

    st.markdown(
        "<div class='sidebar-footer'>"
        "☁️ MSc Cloud Computing Assignment"
        "<br>"
        "Machine Learning • Streamlit • Cloud Deployment"
        "</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# BOOKING INFORMATION
# =========================================================

st.markdown(
    "<div class='section-title'>"
    "📋 Booking Information"
    "</div>",
    unsafe_allow_html=True,
)

st.info(
    "Enter the booking information below and click "
    "**Predict Cancellation Risk**."
)


# =========================================================
# MAIN INPUT COLUMNS
# =========================================================

col1, col2 = st.columns(
    2,
    gap="large",
)


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    st.markdown(
        "### 🏨 Booking Details"
    )

    hotel = st.selectbox(
        "🏢 Hotel Type",
        [
            "City Hotel",
            "Resort Hotel",
        ],
    )

    lead_time = st.slider(
        "📅 Lead Time (Days Before Arrival)",
        min_value=0,
        max_value=365,
        value=60,
    )

    nights = st.slider(
        "🌙 Total Number of Nights",
        min_value=1,
        max_value=30,
        value=5,
    )

    adults = st.number_input(
        "👨‍👩‍👧 Adults",
        min_value=1,
        value=2,
        step=1,
    )

    children = st.number_input(
        "🧒 Children",
        min_value=0,
        value=0,
        step=1,
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    st.markdown(
        "### 💳 Customer & Payment"
    )

    deposit_type = st.selectbox(
        "💳 Deposit Type",
        [
            "No Deposit",
            "Non Refund",
            "Refundable",
        ],
    )

    customer_type = st.selectbox(
        "👤 Customer Type",
        [
            "Transient",
            "Transient-Party",
            "Contract",
            "Group",
        ],
    )

    previous_cancellations = st.number_input(
        "🔁 Previous Cancellations",
        min_value=0,
        value=0,
        step=1,
    )

    adr = st.number_input(
        "💰 Average Daily Rate",
        min_value=0.0,
        value=100.0,
        step=1.0,
        format="%.2f",
        help="Average room price per night.",
    )

    special_requests = st.number_input(
        "⭐ Special Requests",
        min_value=0,
        value=1,
        step=1,
        help="Number of special requests made by the customer.",
    )


# =========================================================
# ADVANCED SETTINGS
# =========================================================

st.markdown("")

with st.expander(
    "⚙️ Advanced Booking Settings",
    expanded=False,
):

    adv1, adv2 = st.columns(2)

    with adv1:

        meal = st.selectbox(
            "🍽 Meal Plan",
            [
                "BB",
                "HB",
                "SC",
                "FB",
                "Undefined",
            ],
        )

        market_segment = st.selectbox(
            "🌍 Market Segment",
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

    with adv2:

        repeated_guest = st.selectbox(
            "🔄 Repeated Guest",
            [
                "No",
                "Yes",
            ],
        )

        parking = st.selectbox(
            "🚗 Car Parking Required",
            [
                "No",
                "Yes",
            ],
        )


# =========================================================
# PREPARE MODEL INPUT
# =========================================================

weekend_nights = min(
    nights,
    2,
)

week_nights = max(
    nights - weekend_nights,
    0,
)


# =========================================================
# DISTRIBUTION CHANNEL
# =========================================================

if market_segment == "Direct":

    distribution_channel = "Direct"

elif market_segment == "Corporate":

    distribution_channel = "Corporate"

else:

    distribution_channel = "TA/TO"


# =========================================================
# MODEL PAYLOAD
# =========================================================

payload = {

    "hotel":
        hotel,

    "lead_time":
        int(lead_time),

    "stays_in_weekend_nights":
        int(weekend_nights),

    "stays_in_week_nights":
        int(week_nights),

    "adults":
        int(adults),

    "children":
        float(children),

    "babies":
        0,

    "meal":
        meal,

    "market_segment":
        market_segment,

    "distribution_channel":
        distribution_channel,

    "is_repeated_guest":
        1 if repeated_guest == "Yes" else 0,

    "previous_cancellations":
        int(previous_cancellations),

    "previous_bookings_not_canceled":
        0,

    "reserved_room_type":
        "A",

    "deposit_type":
        deposit_type,

    "customer_type":
        customer_type,

    "adr":
        float(adr),

    "required_car_parking_spaces":
        1 if parking == "Yes" else 0,

    "total_of_special_requests":
        int(special_requests),
}


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("---")

predict_clicked = st.button(
    "🔍 Predict Cancellation Risk",
    type="primary",
    use_container_width=True,
)


# =========================================================
# RUN PREDICTION
# =========================================================

if predict_clicked:

    if model is None:

        st.error(
            "Model not found. Please run `python train.py` first."
        )

        st.stop()


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    try:

        booking_df = pd.DataFrame(
            [payload]
        )

        probability = float(
            model.predict_proba(
                booking_df
            )[0, 1]
        )

    except Exception as exc:

        st.error(
            f"Model prediction failed: {exc}"
        )

        st.stop()


    # =====================================================
    # RISK CLASSIFICATION
    # =====================================================

    if probability >= 0.70:

        risk = "HIGH RISK"

        risk_icon = "🔴"

        risk_color = "#dc2626"

        background = "#fef2f2"

        border = "#fecaca"

        message = (
            "This booking has a high probability "
            "of cancellation."
        )

        recommendation = (
            "Consider proactively contacting the customer "
            "to confirm the reservation."
        )


    elif probability >= 0.40:

        risk = "MEDIUM RISK"

        risk_icon = "🟠"

        risk_color = "#ea580c"

        background = "#fff7ed"

        border = "#fed7aa"

        message = (
            "This booking has a moderate probability "
            "of cancellation."
        )

        recommendation = (
            "Consider monitoring the booking or confirming "
            "the reservation closer to arrival."
        )


    else:

        risk = "LOW RISK"

        risk_icon = "🟢"

        risk_color = "#16a34a"

        background = "#f0fdf4"

        border = "#bbf7d0"

        message = (
            "This booking has a low probability "
            "of cancellation."
        )

        recommendation = (
            "The booking currently appears relatively stable."
        )


    # =====================================================
    # CANCELLATION RISK TITLE
    # =====================================================

    st.markdown(
        "<h2 style='"
        "text-align:center;"
        "color:#475569;"
        "margin-top:24px;"
        "margin-bottom:15px;"
        "'>"
        "Cancellation Risk"
        "</h2>",
        unsafe_allow_html=True,
    )


    # =====================================================
    # MAIN RISK CARD
    # =====================================================

    risk_html = (
        f"<div style='"
        f"background:{background};"
        f"border:2px solid {border};"
        f"border-radius:24px;"
        f"padding:32px 20px;"
        f"text-align:center;"
        f"box-shadow:0 8px 25px rgba(0,0,0,0.07);"
        f"margin-bottom:24px;"
        f"'>"

        f"<div style='font-size:52px;'>"
        f"{risk_icon}"
        f"</div>"

        f"<div style='"
        f"font-size:50px;"
        f"font-weight:900;"
        f"color:{risk_color};"
        f"'>"
        f"{probability * 100:.1f}%"
        f"</div>"

        f"<div style='"
        f"font-size:27px;"
        f"font-weight:800;"
        f"color:{risk_color};"
        f"margin-top:3px;"
        f"'>"
        f"{risk}"
        f"</div>"

        f"<div style='"
        f"font-size:16px;"
        f"color:#475569;"
        f"margin-top:17px;"
        f"'>"
        f"{message}"
        f"</div>"

        f"</div>"
    )

    st.markdown(
        risk_html,
        unsafe_allow_html=True,
    )


    # =====================================================
    # CANCELLATION PROBABILITY
    # =====================================================

    st.markdown(
        "#### 📈 Cancellation Probability"
    )

    st.progress(
        float(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            )
        )
    )

    st.markdown(
        f"<p style='"
        f"text-align:center;"
        f"font-weight:800;"
        f"color:{risk_color};"
        f"'>"
        f"{probability * 100:.1f}% "
        f"cancellation probability"
        f"</p>",
        unsafe_allow_html=True,
    )


    # =====================================================
    # RISK SCALE
    # =====================================================

    st.markdown(
        "### 🚦 Risk Scale"
    )

    low_col, medium_col, high_col = st.columns(3)


    with low_col:

        st.success(
            """
            🟢 **LOW RISK**

            0% – 39%
            """
        )


    with medium_col:

        st.warning(
            """
            🟠 **MEDIUM RISK**

            40% – 69%
            """
        )


    with high_col:

        st.error(
            """
            🔴 **HIGH RISK**

            70% – 100%
            """
        )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.markdown(
        "### 💡 Recommendation"
    )


    if probability >= 0.70:

        st.error(
            recommendation
        )


    elif probability >= 0.40:

        st.warning(
            recommendation
        )


    else:

        st.success(
            recommendation
        )


    # =====================================================
    # BOOKING SUMMARY
    # =====================================================

    st.markdown(
        "### 📋 Booking Summary"
    )

    summary1, summary2 = st.columns(2)


    with summary1:

        st.write(
            f"**🏨 Hotel Type:** {hotel}"
        )

        st.write(
            f"**📅 Lead Time:** {lead_time} days"
        )

        st.write(
            f"**🌙 Total Nights:** {nights}"
        )

        st.write(
            f"**👨‍👩‍👧 Adults:** {adults}"
        )

        st.write(
            f"**🧒 Children:** {children}"
        )


    with summary2:

        st.write(
            f"**💳 Deposit Type:** {deposit_type}"
        )

        st.write(
            f"**👤 Customer Type:** {customer_type}"
        )

        st.write(
            f"**🔁 Previous Cancellations:** "
            f"{previous_cancellations}"
        )

        st.write(
            f"**💰 Average Daily Rate:** "
            f"{adr:.2f}"
        )

        st.write(
            f"**⭐ Special Requests:** "
            f"{special_requests}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "<div class='footer'>"
    "🤖 AI-Based Hotel Booking Cancellation Predictor"
    "<br>"
    "MSc Cloud Computing Assignment"
    "<br><br>"
    "Predictions are provided for academic demonstration purposes only."
    "</div>",
    unsafe_allow_html=True,
)