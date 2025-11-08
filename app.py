import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hopsworks
import joblib
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from scipy.interpolate import make_interp_spline
import openmeteo_requests
import requests_cache
from retry_requests import retry
from inference_pipeline import AQIInferencePipeline
 

# ───────────────────────────────
# Load environment variables
# ───────────────────────────────
load_dotenv()
HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
PROJECT_NAME = os.getenv('HOPSWORKS_PROJECT_NAME', 'ubaidrazaaqi')

# ───────────────────────────────
# Setup Open-Meteo client
# ───────────────────────────────
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ───────────────────────────────
# Fetch latest data from Hopsworks
# ───────────────────────────────
@st.cache_data(ttl=3600)
def fetch_latest_from_hopsworks():
    if not HOPSWORKS_API_KEY:
        st.error("Set HOPSWORKS_API_KEY for live data.")
        return pd.DataFrame()
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group('aqi_features', version=1)
    df = fg.read()
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    yesterday = datetime.now() - timedelta(days=2)
    four_days_ago = yesterday - timedelta(days=4)
    # four_days_ago = datetime.now() - timedelta(days=4)
    df = df[df['date'] >= four_days_ago]
    return df

# ───────────────────────────────
# Fetch current pollutants via Open-Meteo
# ───────────────────────────────
@st.cache_data(ttl=900)
def fetch_current_pollutants():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 31.558,     # Lahore
        "longitude": 74.3507,
        "current": ["us_aqi", "pm10", "pm2_5", 
                    "nitrogen_dioxide", "ozone", "sulphur_dioxide"],
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    current = response.Current()
    return {
        "time": current.Time(),
        "us_aqi": current.Variables(0).Value(),
        "pm10": current.Variables(1).Value(),
        "pm2_5": current.Variables(2).Value(),
        "nitrogen_dioxide": current.Variables(3).Value(),
        "ozone": current.Variables(4).Value(),
        "sulphur_dioxide": current.Variables(5).Value(),
    }

# ───────────────────────────────
# AQI Category
# ───────────────────────────────
def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "🟢"
    elif aqi <= 100: return "Moderate", "🟡"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi <= 200: return "Unhealthy", "🔴"
    elif aqi <= 300: return "Very Unhealthy", "🟣"
    else: return "Hazardous", "🟤"

# ───────────────────────────────
# Streamlit UI
# ───────────────────────────────
st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")
st.title("🌫️ Pearls AQI Predictor")
st.markdown("Live AQI for Lahore + 3-Day Forecast | Powered by Hopsworks & Open-Meteo")

st.sidebar.header("Config")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

col1, col2 = st.columns(2)

# ────────────── LEFT COLUMN ──────────────
with col1:
    st.header("📍 Current AQI & Pollutants")

    df = fetch_latest_from_hopsworks()
    current_data = fetch_current_pollutants()

    if current_data:
        curr_aqi = current_data["us_aqi"]
        cat, emoji = get_aqi_category(curr_aqi)

        st.metric("Current AQI", f"{curr_aqi:.0f}", help=cat)
        st.metric(emoji, cat)

        # --- Pollutant Bar Chart ---
        pollutant_labels = {
            "pm2_5": "PM2.5",
            "pm10": "PM10",
            "nitrogen_dioxide": "NO₂",
            "ozone": "O₃",
            "sulphur_dioxide": "SO₂"
        }
        cols = list(pollutant_labels.keys())
        values = [current_data[c] for c in cols]
        labels = [pollutant_labels[c] for c in cols]

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title("Current Pollutant Levels", fontsize=12, weight='bold')
        ax.set_ylabel("Pollutant Level")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        if curr_aqi > 150:
            st.error("⚠️ **Hazard Alert**: High AQI! Limit outdoor time, especially for sensitive groups.")
    else:
        st.warning("Could not fetch current pollutant data.")

# ────────────── RIGHT COLUMN ──────────────
with col2:
    st.header("🔮 3-Day Forecast")
    if not df.empty:
        inference = AQIInferencePipeline()
        result = inference.run_inference(save_log=False)
        if result and result['predictions']:
            for pred in result['predictions']:
                p_aqi = pred['predicted_aqi']
                p_cat, p_emoji = get_aqi_category(p_aqi)
                st.metric(
                    pred['prediction_time'].strftime("%m/%d %H"),
                    f"{p_aqi:.0f}",
                    delta=p_aqi - curr_aqi,
                    help=f"{p_cat} {p_emoji}"
                )
                if p_aqi > 150:
                    st.warning(f"Alert for {pred['horizon']}: {p_cat}")

# ────────────── TABS ──────────────
tab1, tab2, tab3 = st.tabs(["📊 AQI Trends", "🔍 Feature Importance", "ℹ️ About"])

with tab1:
    if not df.empty:
        st.subheader("AQI Trend (Last 4 Days, 6-hour Intervals)")

        df = df.sort_values('date')
        x = df['date']
        y = df['us_aqi']

        # Smooth trend like sine wave
        x_num = np.arange(len(x))
        x_smooth = np.linspace(x_num.min(), x_num.max(), 300)
        spline = make_interp_spline(x_num, y, k=3)
        y_smooth = spline(x_smooth)

        tick_interval = 10
        tick_indices = range(0, len(x), tick_interval)
        tick_labels = [x.iloc[i].strftime("%b %d %H:%M") for i in tick_indices]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_smooth, y_smooth, color='royalblue', linewidth=2.5, label='Smoothed AQI Trend')
        ax.scatter(x_num, y, color='darkorange', s=20, label='Recorded AQI')

        ax.set_xticks([x_num[i] for i in tick_indices])
        ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel("AQI")
        ax.set_title("4-Day AQI Trend", fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

with tab2:
    st.subheader("SHAP: Top Feature Impacts (All Models)")
    image_dir = "images"
    csv_dir = "csv_files"
    if os.path.exists(image_dir):
        model_dirs = ["aqi_model_24h", "aqi_model_48h", "aqi_model_72h"]
        cols = st.columns(3)
        for i, model_dir in enumerate(model_dirs):
            with cols[i]:
                image_path = os.path.join(image_dir, f"{model_dir}_shap_bar.png")
                if os.path.exists(image_path):
                    st.image(image_path, caption=f"{model_dir} SHAP Bar Plot", use_container_width=True)
                csv_path = os.path.join(csv_dir, f"{model_dir}_shap_values.csv")
                if os.path.exists(csv_path):
                    df_shap = pd.read_csv(csv_path)
                    st.dataframe(df_shap.head(10), use_container_width=True)
                    st.download_button(
                        label=f"Download {model_dir} SHAP CSV",
                        data=open(csv_path, 'rb').read(),
                        file_name=f"{model_dir}_shap_values.csv",
                        mime="text/csv"
                    )
        st.info("PM2.5 and lag features dominate across horizons.")
    else:
        st.warning("Run shap_analysis.py to generate SHAP visuals.")

with tab3:
    st.markdown("""
    - **Data Source**: Open-Meteo API → Hopsworks Feature Store  
    - **Models**: Random Forest / Ridge / Neural Network (24h, 48h, 72h)  
    - **Automation**: GitHub Actions (hourly ingestion, daily retraining)  
    - **Built with**: Python, Scikit-learn, TensorFlow, SHAP, Streamlit  
    """)

st.markdown("---")
st.markdown("*Serverless AQI Forecasting *")
