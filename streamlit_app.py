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
    page_title="StaySure | Hotel Cancellation Risk",
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

/* =========================================================
   HIDE STREAMLIT BRANDING
========================================================= */

#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"],
.stDeployButton {
    display: none !important;
}

header[data-testid="stHeader"] {
    visibility: hidden !important;
    height: 0 !important;
}


/* =========================================================
   PAGE
========================================================= */

html,
body,
[class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;
}

.stApp {
    background:
        linear-gradient(
            135deg,
            #f8fafc 0%,
            #eff6ff 50%,
            #f8fafc 100%
        );
}

.block-container {
    max-width: 1380px;
    padding-top: 0.9rem;
    padding-bottom: 1.4rem;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}


/* =========================================================
   COMPACT HEADER
========================================================= */

.app-header {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            110deg,
            #0f172a 0%,
            #172554 40%,
            #1e3a8a 72%,
            #2563eb 100%
        );

    border-radius: 18px;

    padding:
        20px 28px;

    margin-bottom: 14px;

    box-shadow:
        0 9px 24px rgba(30,58,138,0.15);
}

.app-header::after {
    content: "";

    position: absolute;

    width: 245px;
    height: 245px;

    right: -60px;
    top: -100px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(255,255,255,0.17),
            rgba(255,255,255,0)
        );
}

.header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;

    position: relative;
    z-index: 2;
}

.header-brand {
    color: #bfdbfe;

    font-size: 10px;
    font-weight: 800;

    text-transform: uppercase;
    letter-spacing: 1px;

    margin-bottom: 4px;
}

.main-title {
    color: white;

    font-size: 27px;
    font-weight: 900;

    line-height: 1.1;

    letter-spacing: -0.5px;
}

.subtitle {
    color: #dbeafe;

    font-size: 12px;

    margin-top: 5px;
}

.header-status {
    white-space: nowrap;

    background:
        rgba(255,255,255,0.10);

    border:
        1px solid rgba(191,219,254,0.25);

    color: #dbeafe;

    padding:
        6px 11px;

    border-radius: 999px;

    font-size: 10.5px;

    font-weight: 700;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0f172a 0%,
            #172554 45%,
            #1e3a8a 100%
        );
}

section[data-testid="stSidebar"] * {
    color: white;
}

.sidebar-logo {
    width: 40px;
    height: 40px;

    margin:
        2px auto 6px auto;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 11px;

    background:
        rgba(255,255,255,0.10);

    border:
        1px solid rgba(255,255,255,0.16);

    font-size: 18px;
    font-weight: 900;
}

.sidebar-title {
    text-align: center;

    font-size: 19px;
    font-weight: 900;
}

.sidebar-subtitle {
    text-align: center;

    color: #bfdbfe !important;

    font-size: 10.5px;

    margin-bottom: 13px;
}

.sidebar-card {
    background:
        rgba(255,255,255,0.075);

    border:
        1px solid rgba(255,255,255,0.12);

    border-radius: 11px;

    padding:
        10px 11px;

    margin-bottom: 8px;
}

.sidebar-heading {
    font-size: 10px;
    font-weight: 850;

    text-transform: uppercase;
    letter-spacing: 0.55px;

    margin-bottom: 6px;
}

.sidebar-text {
    color: #e2e8f0 !important;

    font-size: 10.5px;
    line-height: 1.5;
}

.selected-model {
    color: #bfdbfe !important;

    font-size: 14px;

    font-weight: 850;
}

.model-row {
    background:
        rgba(255,255,255,0.055);

    border-radius: 6px;

    padding:
        4px 6px;

    margin-top: 4px;

    font-size: 10px;
}

.sidebar-status {
    background:
        rgba(96,165,250,0.13);

    border:
        1px solid rgba(147,197,253,0.22);

    border-radius: 8px;

    padding: 7px;

    text-align: center;

    color: #dbeafe !important;

    font-size: 10px;

    font-weight: 700;
}


/* =========================================================
   SECTION TITLES
========================================================= */

.section-title {
    color: #172554;

    font-size: 18px;

    font-weight: 850;

    margin-top: 2px;

    margin-bottom: 1px;
}

.section-subtitle {
    color: #64748b;

    font-size: 11.5px;

    margin-bottom: 7px;
}


/* =========================================================
   INPUTS
========================================================= */

label[data-testid="stWidgetLabel"] p {
    color: #334155 !important;

    font-size: 11.5px !important;

    font-weight: 700 !important;
}

div[data-baseweb="select"] > div {
    min-height: 38px;

    border-radius: 8px !important;
}

input {
    border-radius: 8px !important;

    font-size: 12px !important;
}

[data-testid="stNumberInput"],
[data-testid="stSelectbox"],
[data-testid="stSlider"] {
    margin-bottom: -5px;
}


/* =========================================================
   EXPANDER
========================================================= */

[data-testid="stExpander"] {
    background: white;

    border:
        1px solid #dbeafe !important;

    border-radius: 10px !important;

    box-shadow:
        0 3px 9px rgba(30,58,138,0.025);
}


/* =========================================================
   PREDICT ACTION
========================================================= */

.predict-intro {
    text-align: center;

    color: #64748b;

    font-size: 11px;

    margin-top: 4px;

    margin-bottom: 7px;
}

div.stButton > button {
    height: 50px;

    background:
        linear-gradient(
            90deg,
            #172554 0%,
            #1e40af 45%,
            #2563eb 100%
        );

    color: white;

    border: none;

    border-radius: 12px;

    font-size: 15px;

    font-weight: 800;

    letter-spacing: 0.15px;

    box-shadow:
        0 7px 18px rgba(37,99,235,0.22);

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease;
}

div.stButton > button:hover {
    color: white;

    transform:
        translateY(-1px);

    box-shadow:
        0 10px 24px rgba(37,99,235,0.30);
}

div.stButton > button:active {
    transform:
        translateY(0);
}


/* =========================================================
   PRIMARY RESULT
========================================================= */

.prediction-card {
    margin-top: 8px;

    border-radius: 15px;

    padding:
        15px 20px;

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 20px;

    box-shadow:
        0 6px 16px rgba(30,58,138,0.065);
}

.prediction-left {
    display: flex;

    align-items: center;

    gap: 18px;
}

.prediction-probability {
    font-size: 38px;

    font-weight: 900;

    line-height: 1;

    letter-spacing: -1.1px;
}

.prediction-label {
    color: #64748b;

    font-size: 9px;

    font-weight: 850;

    text-transform: uppercase;

    letter-spacing: 0.6px;

    margin-bottom: 4px;
}

.prediction-risk {
    font-size: 16px;

    font-weight: 850;
}

.prediction-message {
    color: #475569;

    font-size: 11.5px;

    margin-top: 3px;
}

.prediction-action {
    max-width: 420px;

    font-size: 11.5px;

    color: #475569;

    line-height: 1.5;

    text-align: right;
}

.prediction-action-title {
    color: #1e3a8a;

    font-size: 10px;

    font-weight: 850;

    text-transform: uppercase;

    letter-spacing: 0.55px;

    margin-bottom: 3px;
}


/* =========================================================
   BELOW-FOLD DETAILS
========================================================= */

.detail-section {
    margin-top: 24px;

    padding-top: 16px;

    border-top:
        1px solid #dbeafe;
}

.detail-heading {
    color: #172554;

    font-size: 18px;

    font-weight: 850;

    margin-bottom: 11px;
}

.scale-card {
    background: white;

    border:
        1px solid #dbeafe;

    border-radius: 11px;

    padding: 11px;

    text-align: center;
}

.scale-label {
    font-size: 11px;

    font-weight: 850;
}

.scale-range {
    color: #64748b;

    font-size: 10px;

    margin-top: 2px;
}

.recommendation-box {
    background:
        linear-gradient(
            90deg,
            #eff6ff,
            #f8fafc
        );

    border:
        1px solid #bfdbfe;

    border-left:
        4px solid #2563eb;

    border-radius: 11px;

    padding:
        12px 14px;

    margin-top: 13px;
}

.recommendation-heading {
    color: #1e3a8a;

    font-size: 10px;

    font-weight: 850;

    text-transform: uppercase;

    letter-spacing: 0.55px;

    margin-bottom: 4px;
}

.recommendation-text {
    color: #475569;

    font-size: 11.5px;

    line-height: 1.55;
}

.summary-card {
    background: white;

    border:
        1px solid #dbeafe;

    border-radius: 9px;

    padding:
        9px 10px;

    margin-bottom: 6px;
}

.summary-label {
    color: #94a3b8;

    font-size: 8.5px;

    font-weight: 850;

    text-transform: uppercase;

    letter-spacing: 0.45px;
}

.summary-value {
    color: #1e293b;

    font-size: 11.5px;

    font-weight: 750;

    margin-top: 2px;
}


/* =========================================================
   FOOTER
========================================================= */

.app-footer {
    text-align: center;

    color: #94a3b8;

    font-size: 9.5px;

    padding:
        18px 5px 6px;

    margin-top: 22px;
}


/* =========================================================
   MOBILE
========================================================= */

@media (max-width: 768px) {

    .header-row {
        display: block;
    }

    .header-status {
        display: inline-block;

        margin-top: 9px;
    }

    .main-title {
        font-size: 23px;
    }

    .prediction-card {
        display: block;
    }

    .prediction-action {
        text-align: left;

        margin-top: 10px;
    }
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
# LOAD METADATA
# =========================================================

@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():
        return {}

    try:

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return {}


metadata = load_metadata()

best_model = metadata.get(
    "best_model",
    "Production Model",
)

best_model_display = (
    best_model
    .replace("_", " ")
    .title()
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        (
            '<div class="sidebar-logo">H</div>'
            '<div class="sidebar-title">StaySure</div>'
            '<div class="sidebar-subtitle">'
            'Reservation Risk Intelligence'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="sidebar-card">'
            '<div class="sidebar-heading">'
            'Business Purpose'
            '</div>'
            '<div class="sidebar-text">'
            'Identify reservations with elevated cancellation '
            'risk before arrival and support proactive booking '
            'management decisions.'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="sidebar-card">'
            '<div class="sidebar-heading">'
            'Production Model'
            '</div>'
            f'<div class="selected-model">'
            f'{best_model_display}'
            '</div>'
            '<div class="sidebar-text">'
            'Selected automatically using validation F1 performance.'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="sidebar-card">'
            '<div class="sidebar-heading">'
            'Models Evaluated'
            '</div>'
            '<div class="model-row">'
            'Logistic Regression'
            '</div>'
            '<div class="model-row">'
            'Random Forest'
            '</div>'
            '<div class="model-row">'
            'Gradient Boosting'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    if model is not None:

        sidebar_status = (
            '<div class="sidebar-status">'
            '● Prediction Engine Online'
            '</div>'
        )

    else:

        sidebar_status = (
            '<div class="sidebar-status">'
            'Prediction Engine Unavailable'
            '</div>'
        )

    st.markdown(
        sidebar_status,
        unsafe_allow_html=True,
    )


# =========================================================
# HEADER
# =========================================================

status_text = (
    "● Prediction Engine Online"
    if model is not None
    else "Prediction Engine Unavailable"
)

st.markdown(
    (
        '<div class="app-header">'
        '<div class="header-row">'
        '<div>'
        '<div class="header-brand">'
        'StaySure • Reservation Risk Intelligence'
        '</div>'
        '<div class="main-title">'
        'Hotel Booking Cancellation Predictor'
        '</div>'
        '<div class="subtitle">'
        'Real-time cancellation risk assessment '
        'for hotel reservations.'
        '</div>'
        '</div>'
        f'<div class="header-status">'
        f'{status_text}'
        '</div>'
        '</div>'
        '</div>'
    ),
    unsafe_allow_html=True,
)


# =========================================================
# BOOKING FORM HEADER
# =========================================================

st.markdown(
    (
        '<div class="section-title">'
        'Booking Risk Assessment'
        '</div>'
        '<div class="section-subtitle">'
        'Enter reservation details and generate a prediction.'
        '</div>'
    ),
    unsafe_allow_html=True,
)


# =========================================================
# INPUT ROW 1
# =========================================================

c1, c2, c3, c4 = st.columns(
    [1.15, 1, 1, 1]
)

with c1:

    hotel = st.selectbox(
        "Hotel Type",
        [
            "City Hotel",
            "Resort Hotel",
        ],
    )

with c2:

    lead_time = st.number_input(
        "Lead Time (Days)",
        min_value=0,
        max_value=365,
        value=60,
        step=1,
    )

with c3:

    nights = st.number_input(
        "Number of Nights",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
    )

with c4:

    adr = st.number_input(
        "Average Daily Rate",
        min_value=0.0,
        value=100.0,
        step=1.0,
        format="%.2f",
    )


# =========================================================
# INPUT ROW 2
# =========================================================

c5, c6, c7, c8 = st.columns(4)

with c5:

    adults = st.number_input(
        "Adults",
        min_value=1,
        value=2,
        step=1,
    )

with c6:

    children = st.number_input(
        "Children",
        min_value=0,
        value=0,
        step=1,
    )

with c7:

    deposit_type = st.selectbox(
        "Deposit Type",
        [
            "No Deposit",
            "Non Refund",
            "Refundable",
        ],
    )

with c8:

    customer_type = st.selectbox(
        "Customer Type",
        [
            "Transient",
            "Transient-Party",
            "Contract",
            "Group",
        ],
    )


# =========================================================
# INPUT ROW 3
# =========================================================

c9, c10 = st.columns(2)

with c9:

    previous_cancellations = st.number_input(
        "Previous Cancellations",
        min_value=0,
        value=0,
        step=1,
    )

with c10:

    special_requests = st.number_input(
        "Special Requests",
        min_value=0,
        value=1,
        step=1,
    )


# =========================================================
# ADVANCED ATTRIBUTES
# =========================================================

with st.expander(
    "Additional Reservation Attributes",
    expanded=False,
):

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        meal = st.selectbox(
            "Meal Plan",
            [
                "BB",
                "HB",
                "SC",
                "FB",
                "Undefined",
            ],
        )

    with a2:

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

    with a3:

        repeated_guest = st.selectbox(
            "Repeated Guest",
            [
                "No",
                "Yes",
            ],
        )

    with a4:

        parking = st.selectbox(
            "Parking Required",
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


if market_segment == "Direct":

    distribution_channel = "Direct"

elif market_segment == "Corporate":

    distribution_channel = "Corporate"

else:

    distribution_channel = "TA/TO"


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
# PREDICTION ACTION
# =========================================================

st.markdown(
    (
        '<div class="predict-intro">'
        'Ready to assess this reservation? '
        'Generate an AI-assisted cancellation risk score.'
        '</div>'
    ),
    unsafe_allow_html=True,
)

button_left, button_center, button_right = st.columns(
    [1.2, 2.6, 1.2]
)

with button_center:

    predict_clicked = st.button(
        "Run Cancellation Risk Assessment",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# RUN PREDICTION
# =========================================================

if predict_clicked:

    if model is None:

        st.error(
            "Prediction engine is currently unavailable."
        )

        st.stop()


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
            f"Prediction could not be generated: {exc}"
        )

        st.stop()


    # =====================================================
    # RISK CLASSIFICATION
    # =====================================================

    if probability >= 0.70:

        risk = "HIGH RISK"

        risk_color = "#dc2626"

        background = "#fef2f2"

        border = "#fecaca"

        message = (
            "High predicted likelihood of cancellation."
        )

        recommendation = (
            "Prioritise this reservation for confirmation. "
            "Consider contacting the customer and reviewing "
            "the applicable cancellation policy."
        )


    elif probability >= 0.40:

        risk = "MODERATE RISK"

        risk_color = "#d97706"

        background = "#fffbeb"

        border = "#fde68a"

        message = (
            "Moderate predicted likelihood of cancellation."
        )

        recommendation = (
            "Monitor this reservation as the arrival date "
            "approaches and consider confirming the booking."
        )


    else:

        risk = "LOW RISK"

        risk_color = "#2563eb"

        background = "#eff6ff"

        border = "#bfdbfe"

        message = (
            "Low predicted likelihood of cancellation."
        )

        recommendation = (
            "No immediate intervention is indicated. "
            "Continue standard reservation procedures."
        )


    # =====================================================
    # COMPACT PRIMARY RESULT
    # =====================================================

    prediction_html = (
        f'<div class="prediction-card" '
        f'style="'
        f'background:{background};'
        f'border:1px solid {border};'
        f'border-left:5px solid {risk_color};'
        f'">'

        '<div class="prediction-left">'

        '<div>'

        '<div class="prediction-label">'
        'Cancellation Probability'
        '</div>'

        f'<div class="prediction-probability" '
        f'style="color:{risk_color};">'
        f'{probability * 100:.1f}%'
        '</div>'

        '</div>'

        '<div>'

        f'<div class="prediction-risk" '
        f'style="color:{risk_color};">'
        f'{risk}'
        '</div>'

        '<div class="prediction-message">'
        f'{message}'
        '</div>'

        '</div>'

        '</div>'

        '<div class="prediction-action">'

        '<div class="prediction-action-title">'
        'Recommended Action'
        '</div>'

        f'{recommendation}'

        '</div>'

        '</div>'
    )

    st.markdown(
        prediction_html,
        unsafe_allow_html=True,
    )


    # =====================================================
    # RISK CLASSIFICATION DETAILS
    # =====================================================

    st.markdown(
        (
            '<div class="detail-section">'
            '<div class="detail-heading">'
            'Risk Classification'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


    low_col, medium_col, high_col = st.columns(3)


    with low_col:

        st.markdown(
            (
                '<div class="scale-card">'
                '<div class="scale-label" '
                'style="color:#2563eb;">'
                '● LOW RISK'
                '</div>'
                '<div class="scale-range">'
                'Below 40%'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


    with medium_col:

        st.markdown(
            (
                '<div class="scale-card">'
                '<div class="scale-label" '
                'style="color:#d97706;">'
                '● MODERATE RISK'
                '</div>'
                '<div class="scale-range">'
                '40% – 69%'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


    with high_col:

        st.markdown(
            (
                '<div class="scale-card">'
                '<div class="scale-label" '
                'style="color:#dc2626;">'
                '● HIGH RISK'
                '</div>'
                '<div class="scale-range">'
                '70% and above'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.markdown(
        (
            '<div class="recommendation-box">'
            '<div class="recommendation-heading">'
            'Operational Recommendation'
            '</div>'
            '<div class="recommendation-text">'
            f'{recommendation}'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


    # =====================================================
    # BOOKING SUMMARY
    # =====================================================

    st.markdown(
        (
            '<div class="detail-section">'
            '<div class="detail-heading">'
            'Booking Summary'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


    summary_columns = st.columns(5)

    summary_items = [

        (
            "Hotel Type",
            hotel,
        ),

        (
            "Lead Time",
            f"{lead_time} days",
        ),

        (
            "Stay",
            f"{nights} nights",
        ),

        (
            "Guests",
            f"{adults} Adults / {children} Children",
        ),

        (
            "Daily Rate",
            f"{adr:.2f}",
        ),
    ]


    for column, item in zip(
        summary_columns,
        summary_items,
    ):

        label, value = item

        with column:

            st.markdown(
                (
                    '<div class="summary-card">'
                    '<div class="summary-label">'
                    f'{label}'
                    '</div>'
                    '<div class="summary-value">'
                    f'{value}'
                    '</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )


    summary_columns2 = st.columns(5)

    summary_items2 = [

        (
            "Deposit Type",
            deposit_type,
        ),

        (
            "Customer Type",
            customer_type,
        ),

        (
            "Previous Cancellations",
            str(previous_cancellations),
        ),

        (
            "Special Requests",
            str(special_requests),
        ),

        (
            "Market Segment",
            market_segment,
        ),
    ]


    for column, item in zip(
        summary_columns2,
        summary_items2,
    ):

        label, value = item

        with column:

            st.markdown(
                (
                    '<div class="summary-card">'
                    '<div class="summary-label">'
                    f'{label}'
                    '</div>'
                    '<div class="summary-value">'
                    f'{value}'
                    '</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    (
        '<div class="app-footer">'
        '<b>StaySure</b> • Reservation Risk Intelligence'
        '<br>'
        'AI-assisted hotel reservation decision support'
        '</div>'
    ),
    unsafe_allow_html=True,
)