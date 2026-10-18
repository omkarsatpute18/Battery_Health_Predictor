import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Battery Health Predictor",
    page_icon="🔋",
    layout="centered"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
    border: 1px solid #ddd;
}

.result-value {
    font-size: 32px;
    font-weight: 700;
}

.result-label {
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🔋 Battery Health Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter the battery operating parameters to predict its capacity and health.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

@st.cache_data
def load_data():

    data = pd.read_csv("battery_health_data.csv")

    return data


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

@st.cache_resource
def train_model():

    data = load_data()

    # Features used in the project
    features = [
        "cycle",
        "avg_voltage_V",
        "avg_current_A",
        "temperature_C",
        "charge_time_hr",
        "discharge_time_hr",
        "internal_resistance_ohm"
    ]

    target = "capacity_Ah"

    # Select required columns
    X = data[features].copy()
    y = data[target]

    # Fill missing values using mean
    X = X.fillna(X.mean())

    # Train using cycles <= 200
    train_mask = data["cycle"] <= 200

    X_train = X[train_mask]
    y_train = y[train_mask]

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

try:

    model = train_model()

except Exception as e:

    st.error(
        "Unable to load the battery dataset or train the model."
    )

    st.exception(e)

    st.stop()


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.subheader("🔧 Battery Parameters")


# First row
col1, col2 = st.columns(2)

with col1:

    cycle = st.number_input(
        "Cycle Number",
        min_value=1.0,
        max_value=250.0,
        value=150.0,
        step=1.0
    )

with col2:

    avg_voltage = st.number_input(
        "Average Voltage (V)",
        min_value=0.0,
        max_value=10.0,
        value=3.70,
        step=0.01
    )


# Second row
col3, col4 = st.columns(2)

with col3:

    avg_current = st.number_input(
        "Average Current (A)",
        min_value=-100.0,
        max_value=100.0,
        value=2.50,
        step=0.01
    )

with col4:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-50.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )


# Third row
col5, col6 = st.columns(2)

with col5:

    charge_time = st.number_input(
        "Charge Time (hr)",
        min_value=0.0,
        max_value=100.0,
        value=1.20,
        step=0.01
    )

with col6:

    discharge_time = st.number_input(
        "Discharge Time (hr)",
        min_value=0.0,
        max_value=100.0,
        value=1.50,
        step=0.01
    )


# Fourth row
col7, col8 = st.columns(2)

with col7:

    internal_resistance = st.number_input(
        "Internal Resistance (Ω)",
        min_value=0.0,
        max_value=10.0,
        value=0.045,
        step=0.001
    )


# ---------------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------------

st.markdown("")

predict_button = st.button(
    "🔮 Predict Battery Health",
    use_container_width=True
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict_button:

    # Create input DataFrame
    input_data = pd.DataFrame({

        "cycle": [cycle],

        "avg_voltage_V": [avg_voltage],

        "avg_current_A": [avg_current],

        "temperature_C": [temperature],

        "charge_time_hr": [charge_time],

        "discharge_time_hr": [discharge_time],

        "internal_resistance_ohm": [internal_resistance]

    })

    # Predict capacity
    predicted_capacity = model.predict(input_data)[0]

    # -----------------------------------------------------
    # CALCULATE SOH
    # -----------------------------------------------------

    # Maximum capacity observed in the dataset
    data = load_data()

    reference_capacity = data["capacity_Ah"].max()

    soh = (predicted_capacity / reference_capacity) * 100

    # Limit SOH to sensible display range
    soh = max(0, min(100, soh))


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    st.success("Prediction completed successfully!")


    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="result-box">

            <div class="result-label">
            Predicted Capacity
            </div>

            <div class="result-value">
            {predicted_capacity:.3f} Ah
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="result-box">

            <div class="result-label">
            Estimated State of Health
            </div>

            <div class="result-value">
            {soh:.2f} %
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # HEALTH INTERPRETATION
    # -----------------------------------------------------

    st.markdown("### 📊 Battery Condition")

    if soh >= 90:

        st.success(
            "🟢 Battery condition: Excellent"
        )

    elif soh >= 80:

        st.info(
            "🔵 Battery condition: Good"
        )

    elif soh >= 70:

        st.warning(
            "🟡 Battery condition: Moderate"
        )

    else:

        st.error(
            "🔴 Battery condition: Poor"
        )


    # -----------------------------------------------------
    # INPUT SUMMARY
    # -----------------------------------------------------

    with st.expander("View Input Parameters"):

        st.dataframe(
            input_data,
            use_container_width=True
        )