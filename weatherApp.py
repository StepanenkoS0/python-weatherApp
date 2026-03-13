import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Weather ML Forecast", layout="centered")
st.title("🌦 Прогноз опадів на основі ML")

# -------------------------------
# Функції
# -------------------------------
def get_city_coordinates(city_name: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city_name, "format": "json", "limit": 1}
    headers = {"User-Agent": "weather-ml-app"}
    r = requests.get(url, params=params, headers=headers)
    data = r.json()
    if not data:
        return None, None
    return float(data[0]["lat"]), float(data[0]["lon"])

def fetch_weather_data(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min,windspeed_10m_max",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto"
    }
    r = requests.get(url, params=params)
    df = pd.DataFrame(r.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df["rain_label"] = df["precipitation_sum"].apply(lambda x: 1 if x > 0 else 0)
    return df

# -------------------------------
# Введення даних користувачем
# -------------------------------
city_name = st.text_input("Введіть місто:", "Kyiv")
start_date_input = st.text_input("Дата початку (YYYY-MM-DD):",
                                 (datetime.now().replace(year=datetime.now().year-1)).strftime("%Y-%m-%d"))
end_date_input = st.text_input("Дата кінця (YYYY-MM-DD):",
                               datetime.now().strftime("%Y-%m-%d"))

# -------------------------------
# Отримання координат
# -------------------------------
latitude, longitude = get_city_coordinates(city_name)
if latitude is None:
    st.error("Місто не знайдено")
    st.stop()

st.write(f"Координати {city_name}: {latitude:.2f}, {longitude:.2f}")

# -------------------------------
# Завантаження історичних даних
# -------------------------------
if st.button("Отримати історичні дані"):
    df_history = fetch_weather_data(latitude, longitude, start_date_input, end_date_input)
    df_history.to_csv("weather_daily.csv", index=False)
    st.session_state["data"] = df_history

    st.success(f"Дані отримані ({len(df_history)} днів)")
    st.subheader("Останні записи")
    st.dataframe(df_history.tail())

    st.subheader("📈 Історичні опади")
    st.line_chart(df_history.set_index("time")["precipitation_sum"])

# -------------------------------
# Навчання моделі
# -------------------------------
if "data" in st.session_state:
    df = st.session_state["data"]
    features = ["rain_sum", "temperature_2m_max", "temperature_2m_min", "windspeed_10m_max"]
    X = df[features]
    y = df["rain_label"]

    if st.button("Навчити модель"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.session_state["model"] = model
        st.session_state["accuracy"] = accuracy
        st.success(f"Модель навчена. Точність: {accuracy*100:.2f}%")

        # Важливість ознак
        st.subheader("Важливість факторів")
        importance_df = pd.DataFrame({
            "Фактор": features,
            "Важливість": model.feature_importances_
        }).sort_values(by="Важливість", ascending=False)
        st.table(importance_df)
        st.bar_chart(importance_df.set_index("Фактор"))

# -------------------------------
# Прогноз на останній день періоду
# -------------------------------
if "model" in st.session_state:
    st.subheader("Прогноз на останній день періоду")
    df_future = st.session_state["data"].iloc[[-1]]  # беремо останній рядок
    X_future = df_future[features]
    model = st.session_state["model"]
    prediction = model.predict(X_future)[0]
    probability = model.predict_proba(X_future)[0][1]

    if prediction == 1:
        st.success(f"Очікуються опади. Ймовірність: {probability*100:.1f}%")
    else:
        st.success(f"Опадів не очікується. Ймовірність: {probability*100:.1f}%")
