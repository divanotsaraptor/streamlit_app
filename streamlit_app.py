import os
import pandas as pd
import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Оценка стоимости квратиры",
)

# Загрузка данных
city_df=pd.read_csv('city_df.csv')
district_df=pd.read_csv('district_df.csv')

model_path = 'final_gb_model.pkl'

# Входные данные от пользователя
st.subheader("Введите параметры квартиры для предсказания цены")

lat = st.number_input('What is the latitude of your flat?', 37.0, 38.0, step=1e-6, format="%.6f")
lon = st.number_input('What is the longitude of your flat?', 55.0, 56.0, step=1e-6, format="%.6f")
total_square = st.number_input('What is the square of your flat?', 1, 1000)
rooms = st.sidebar.selectbox('How many rooms in your flat?', list(range(1, 17)))
floor = st.number_input('What is the floor of your flat?', 0, 99)
city = st.sidebar.selectbox("Where is your flat? (select city)", city_df['city_cat'])
district = st.sidebar.selectbox("Which district is your flat in?", district_df['district_cat'])

# Кнопка для запуска предсказания
if st.button("Predict Price"):
    # Создание входного DataFrame
    input_df = pd.DataFrame({
        'lat': [lat],
        'lon': [lon],
        'total_square': [total_square],
        'rooms': [rooms],
        'floor': [floor],
        'city_cat': [city],
        'district_cat': [district]
    }, index=[0])

    # Загрузка модели (выполняется только при нажатии кнопки)
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        # Предсказание
        prediction = loaded_model.predict(input_df)
        predicted_price = round(prediction[0])

        # Визуализация результата
        st.success("Оценка завершена")
        st.write("Результаты:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", f"{lat:.6f}")
            st.metric("Longitude", f"{lon:.6f}")
            st.metric("Total square", f"{total_square} m²")
        with col2:
            st.metric("Rooms", rooms)
            st.metric("Floor", floor)
            st.metric("City", city)

        st.markdown("---")
        st.write(f"**Approximate price of your flat is:** {predicted_price:,} roubles")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.warning("Please check the input parameters and try again.")
