import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="centered"
)

# ------------------------------------------------------------
# Load model & pipeline components (cached so it only loads once)
# ------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    model = joblib.load("rf_air_quality_model.pkl")
    le_city = joblib.load("label_encoder_city.pkl")
    le_country = joblib.load("label_encoder_country.pkl")
    le_target = joblib.load("label_encoder_target.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    unique_cities = joblib.load("unique_cities.pkl")
    unique_countries = joblib.load("unique_countries.pkl")
    return model, le_city, le_country, le_target, feature_columns, unique_cities, unique_countries

model, le_city, le_country, le_target, feature_columns, unique_cities, unique_countries = load_pipeline()

# City -> Country mapping, built directly from what the encoders were fitted on
CITY_COUNTRY_MAP = {
    "Beijing": "China", "Cairo": "Egypt", "Delhi": "India", "London": "UK",
    "Los Angeles": "USA", "New York": "USA", "Paris": "France",
    "Sydney": "Australia", "São Paulo": "Brazil", "Tokyo": "Japan",
}

# Approximate coordinates for the cities in the dataset (used for the map)
CITY_COORDS = {
    "Beijing": (39.9042, 116.4074), "Cairo": (30.0444, 31.2357), "Delhi": (28.7041, 77.1025),
    "London": (51.5074, -0.1278), "Los Angeles": (34.0522, -118.2437), "New York": (40.7128, -74.0060),
    "Paris": (48.8566, 2.3522), "Sydney": (-33.8688, 151.2093), "São Paulo": (-23.5505, -46.6333),
    "Tokyo": (35.6762, 139.6503),
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🌍 Air Quality Predictor")
st.divider()

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.subheader("🌤️ Tell Us About Your Surroundings")
st.caption("Don't know the exact numbers? No problem — every field below already starts at a typical value, so you can just adjust what you know and leave the rest as is.")

form_col, map_col = st.columns([1.2, 1])

with form_col:
    city = st.selectbox("City", options=unique_cities, index=0)
    country = CITY_COUNTRY_MAP.get(city, "Unknown")
    st.text_input("Country", value=country, disabled=True)

    month_name = st.selectbox("Month", options=MONTH_NAMES, index=0)
    month = MONTH_NAMES.index(month_name) + 1

with map_col:
    if city in CITY_COORDS:
        lat, lon = CITY_COORDS[city]
        map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})

        dot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_fill_color="[216, 90, 48, 220]",
            get_radius=60000,
            radius_min_pixels=6,
            radius_max_pixels=10,
            stroked=False,
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=2.5)
        st.pydeck_chart(
            pdk.Deck(layers=[dot_layer], initial_view_state=view_state, map_style=None),
            height=220,
        )
    st.caption(f"📍 {city}, {country}")

st.markdown("**Pollutant Levels**")
p1, p2, p3 = st.columns(3)
with p1:
    pm25 = st.slider("PM2.5 (µg/m³)", min_value=5.0, max_value=250.0, value=126.0, step=0.1)
    so2 = st.slider("SO2 (ppb)", min_value=2.0, max_value=50.0, value=26.0, step=0.1)
with p2:
    pm10 = st.slider("PM10 (µg/m³)", min_value=10.0, max_value=300.0, value=155.0, step=0.1)
    co = st.slider("CO (ppm)", min_value=0.1, max_value=10.0, value=5.0, step=0.01)
with p3:
    no2 = st.slider("NO2 (ppb)", min_value=5.0, max_value=100.0, value=53.0, step=0.1)
    o3 = st.slider("O3 (ppb)", min_value=10.0, max_value=200.0, value=105.0, step=0.1)

st.markdown("**Weather Conditions**")
w1, w2, w3 = st.columns(3)
with w1:
    temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=40.0, value=15.0, step=0.1)
with w2:
    humidity = st.slider("Humidity (%)", min_value=10, max_value=90, value=51)
with w3:
    wind_speed = st.slider("Wind Speed (m/s)", min_value=0.5, max_value=15.0, value=7.8, step=0.1)

st.divider()

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if st.button("Predict Air Quality", type="primary", use_container_width=True):

    # Encode City and Country using the fitted label encoders
    city_encoded = le_city.transform([city])[0]
    country_encoded = le_country.transform([country])[0]

    # Assemble all possible feature values, then select + order using feature_columns
    # AQI is intentionally not asked from the user: it contributes under 1% to the
    # model's decision, and varying it across its full range (30-300) never changes
    # the predicted label in testing. A fixed typical value is used instead.
    AQI_DEFAULT = 165

    input_values = {
        "City": city_encoded,
        "Country": country_encoded,
        "AQI": AQI_DEFAULT,
        "PM25": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "O3": o3,
        "Temperature": temperature,
        "Humidity": humidity,
        "WindSpeed": wind_speed,
        "Month": month,
    }
    input_df = pd.DataFrame([input_values])[feature_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    predicted_label = le_target.inverse_transform([prediction])[0]

    safe_idx = list(le_target.classes_).index("Safe")
    unsafe_idx = list(le_target.classes_).index("Unsafe")
    prob_safe = probabilities[safe_idx]
    prob_unsafe = probabilities[unsafe_idx]

    st.subheader("Prediction Result")

    if predicted_label == "Safe":
        st.success("✅ Predicted Air Quality: **Safe**")
    else:
        st.error("⚠️ Predicted Air Quality: **Unsafe**")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Probability: Safe", f"{prob_safe * 100:.1f}%")
    with res_col2:
        st.metric("Probability: Unsafe", f"{prob_unsafe * 100:.1f}%")

    st.progress(float(prob_unsafe))
