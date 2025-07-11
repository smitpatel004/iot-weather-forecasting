import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from thresholds import THRESHOLDS, gas_status
from datetime import datetime, timedelta

st_autorefresh(interval=10000, key="refresh")

sheet_id = "17LUySWNCgQ7iUsN4cbARkNi1Q6FRPS71Q7N1KbYwWTo"
sheet_name = "IoT_based_AQM"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

@st.cache_data(ttl=5)
def load_data():
    return pd.read_csv(csv_url)

data_raw = load_data()
data_raw["Timestamp"] = pd.to_datetime(data_raw["Date"] + " " + data_raw["Time"], dayfirst=True)

# remove outlier
data_plot = data_raw[(data_raw['Temperature(*C)'] > 15) & (data_raw['Humidity(%)'] > 15)].copy()

st.subheader("🌡️ Temperature Over Time")
fig_temp = px.line(data_plot, x="Timestamp", y="Temperature(*C)", title="Temperature vs Time")
st.plotly_chart(fig_temp, use_container_width=True)

st.subheader("🌡️ Humidity Over Time")
fig_humidity = px.line(data_plot, x="Timestamp", y="Humidity(%)", title="Humidity vs Time")
st.plotly_chart(fig_humidity, use_container_width=True)

st.title("🚨 Real-Time Gas Monitoring Dashboard")
latest = data_raw.iloc[-1]
GASES = ["LPG(ppm)", "Propane(ppm)", "Methane(ppm)", "Smoke(ppm)", "Ammonia(ppm)", "Benzene(ppm)"]

st.subheader("📡 Live Sensor Feed")
cols = st.columns(len(GASES))
for i, gas in enumerate(GASES):
    value = latest[gas]
    status = gas_status(gas, value)
    color = {"Low": "🟢 Low", "Moderate": "🟠 Moderate", "High": "🔴 High"}[status]
    cols[i].metric(label=gas, value=f"{value} ppm", delta=color)

# AQI Computation
PPM_TO_MGM3_CO = 1.145
PPM_TO_MGM3_NO2 = 1.88

PM10_BREAKS = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
               (251, 350, 201, 300), (351, 430, 301, 400), (431, 9999, 401, 500)]
NO2_BREAKS = [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
              (181, 280, 201, 300), (281, 400, 301, 400), (401, 9999, 401, 500)]
CO_BREAKS = [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200),
             (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 9999, 401, 500)]

def _sub_index(value, breaks):
    for BP_LO, BP_HI, I_LO, I_HI in breaks:
        if BP_LO <= value <= BP_HI:
            return ((I_HI - I_LO) / (BP_HI - BP_LO)) * (value - BP_LO) + I_LO
    return np.nan

def compute_india_aqi(row):
    pm10 = row["Dust density(µg/m^3)"]
    no2 = row["NO2(ppm)"] * PPM_TO_MGM3_NO2
    co = row["CO(ppm)"] * PPM_TO_MGM3_CO
    return max(
        _sub_index(pm10, PM10_BREAKS),
        _sub_index(no2, NO2_BREAKS),
        _sub_index(co, CO_BREAKS)
    )

def india_aqi_label(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

data_raw["India_AQI"] = data_raw.apply(compute_india_aqi, axis=1)
data_raw["AQI_Category"] = data_raw["India_AQI"].apply(india_aqi_label)

aqi_value = data_raw["India_AQI"].iloc[-1]
aqi_label = data_raw["AQI_Category"].iloc[-1]
st.subheader("🇮🇳 National AQI (Live)")
st.metric(label="India AQI", value=f"{aqi_value:.1f}", delta=aqi_label)

# 📈 Line Chart - AQI Over Time (full dataset)
st.subheader("📈 AQI Trend Over Time")
fig_line = px.line(data_raw, x="Timestamp", y="India_AQI", color="AQI_Category",
                   title="India AQI Over Time",
                   color_discrete_sequence=px.colors.qualitative.Safe)
st.plotly_chart(fig_line, use_container_width=True)

# 📊 Bar Chart - AQI Category Distribution (full dataset)
st.subheader("📊 AQI Category Distribution")
fig_bar = px.histogram(data_raw, x="AQI_Category",
                       category_orders={"AQI_Category":
                                        ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]},
                       title="AQI Category Frequency",
                       color="AQI_Category",
                       color_discrete_sequence=px.colors.qualitative.Safe)
st.plotly_chart(fig_bar, use_container_width=True)


data_plot["Timestamp_num"] = data_plot["Timestamp"].astype(np.int64) // 10**9  # convert to seconds
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

X_time = data_plot[["Timestamp_num"]].values
y_temp = data_plot["Temperature(*C)"].values

model_temp = LinearRegression()
model_temp.fit(X_time, y_temp)
y_temp_pred = model_temp.predict(X_time)
import plotly.graph_objects as go

fig_temp_pred = go.Figure()
fig_temp_pred.add_trace(go.Scatter(x=data_plot["Timestamp"], y=y_temp, mode='markers', name='Actual Temp'))
fig_temp_pred.add_trace(go.Scatter(x=data_plot["Timestamp"], y=y_temp_pred, mode='lines', name='Predicted Temp'))

fig_temp_pred.update_layout(title="📈 Predicted Temperature Over Time",
                            xaxis_title="Timestamp", yaxis_title="Temperature (°C)")
st.plotly_chart(fig_temp_pred)

#pie chart
latest = data_raw.iloc[-1]

# Compute sub-index for each pollutant
PM10_BREAKS = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
               (251, 350, 201, 300), (351, 430, 301, 400), (431, 9999, 401, 500)]
NO2_BREAKS = [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
              (181, 280, 201, 300), (281, 400, 301, 400), (401, 9999, 401, 500)]
CO_BREAKS = [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200),
             (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 9999, 401, 500)]

PPM_TO_MGM3_CO = 1.145
PPM_TO_MGM3_NO2 = 1.88

def _sub_index(value, breaks):
    for BP_LO, BP_HI, I_LO, I_HI in breaks:
        if BP_LO <= value <= BP_HI:
            return ((I_HI - I_LO) / (BP_HI - BP_LO)) * (value - BP_LO) + I_LO
    return 0

pm10 = latest["Dust density(µg/m^3)"]
no2 = latest["NO2(ppm)"] * PPM_TO_MGM3_NO2
co = latest["CO(ppm)"] * PPM_TO_MGM3_CO

pm10_si = _sub_index(pm10, PM10_BREAKS)
no2_si = _sub_index(no2, NO2_BREAKS)
co_si = _sub_index(co, CO_BREAKS)

total = pm10_si + no2_si + co_si

labels = ["PM10", "NO2", "CO"]
values = [pm10_si / total * 100, no2_si / total * 100, co_si / total * 100]

st.subheader("☁️ Pollution Contribution to AQI")
fig_pie = px.pie(names=labels, values=values, title="Pollutant-wise AQI Contribution")
st.plotly_chart(fig_pie)

st.subheader("🌍 Download Sensor Data")
st.download_button(
    label="🔄 Download CSV",
    data=data_raw.to_csv(index=False),
    file_name="sensor_data.csv",
    mime="text/csv"
)


