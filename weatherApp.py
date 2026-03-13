import streamlit as st
import pandas as pd
import requests
from model import train_model, predict_rain

st.title("Прогноз опадів (Open-Meteo)")

st.write("Координати за замовчуванням – Київ")

latitude = st.number_input("Latitude (широта)", value=50.45)
longitude = st.number_input("Longitude (довгота)", value=30.52)

start_date = st.text_input("Дата початку", "2023-01-01")
end_date = st.text_input("Дата кінця", "2023-04-01")

# =============================
# Отримання даних
# =============================

st.header("1. Отримання метеоданих")

if st.button("Отримати дані з Open-Meteo"):

    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily=precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max&timezone=auto"

    response = requests.get(url)

    data = response.json()

    df = pd.DataFrame(data["daily"])

    df.to_csv("weather_daily.csv", index=False)

    st.success("Дані збережено у weather_daily.csv")

    st.dataframe(df)

# =============================
# Завантаження CSV
# =============================

st.header("2. Завантаження CSV")

uploaded_file = st.file_uploader("Завантаж CSV файл", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Завантажений датасет")

    st.dataframe(df)

# =============================
# Навчання моделі
# =============================

st.header("3. Навчання моделі")

if st.button("Навчити модель"):

    with st.spinner("Навчання моделі..."):

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("weather_daily.csv")

        model, accuracy, report = train_model(df)

        st.session_state["model"] = model
        st.session_state["data"] = df

    st.success("Модель успішно навчена")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Точність моделі", f"{accuracy:.2f}")

    with col2:
        st.metric("Кількість записів", len(df))

    st.subheader("Звіт класифікації")

    report_dict = {}

    for line in report.split("\n")[2:-3]:

        row = line.split()

        if len(row) >= 4:

            report_dict[row[0]] = {
                "precision": row[1],
                "recall": row[2],
                "f1-score": row[3],
            }

    report_df = pd.DataFrame(report_dict).T

    st.table(report_df)

# =============================
# Прогноз
# =============================

st.header("4. Прогноз опадів")

if st.button("Зробити прогноз"):

    if "model" not in st.session_state:

        st.error("Спочатку потрібно навчити модель")

    else:

        model = st.session_state["model"]
        df = st.session_state["data"]

        last_row = df.iloc[-1][[
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max"
        ]].values

        prediction, probability = predict_rain(model, last_row)

        if prediction == 1:
            st.success(f"Очікуються опади. Ймовірність: {probability:.2f}")
        else:
            st.success(f"Опадів не очікується. Ймовірність: {probability:.2f}")