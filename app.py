import streamlit as st
import joblib
import numpy as np

model = joblib.load('calories_burnt_model.pkl')  
gender_encoder = joblib.load('gender_encoder.pkl')  

st.set_page_config(page_title='Calories Burnt Predictor', layout='wide')

st.markdown("<h1 style='text-align: center; color: skyblue;'>🔥 Calories Burnt Prediction App 🔥</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>👨‍⚕️ Enter your details below to get an estimate of how many calories you've burned! 💪</p>", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
   
    st.markdown("<h3 style='color: teal;'>Personal Information 🧑‍🤝‍🧑</h3>", unsafe_allow_html=True)
    gender = st.selectbox('Gender', ['male', 'female'])
    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    height = st.number_input('Height (in cm)', min_value=50, max_value=250, value=170)
    weight = st.number_input('Weight (in kg)', min_value=20, max_value=200, value=70)

with col2:
 
    st.markdown("<h3 style='color: teal;'>Activity Details 🏃‍♂️</h3>", unsafe_allow_html=True)
    duration = st.number_input('Duration of activity (in minutes)', min_value=1, max_value=300, value=30)
    heart_rate = st.number_input('Heart rate (beats per minute)', min_value=50, max_value=200, value=100)
    body_temp = st.number_input('Body temperature (in °C)', min_value=35.0, max_value=45.0, value=37.0)

gender_encoded = gender_encoder.transform([gender])[0]

input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])

# Predict button with styling
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button('💥 Predict Calories Burnt 💥'):
 
    calories_burnt = model.predict(input_data)
    st.success(f'Estimated Calories Burnt: {calories_burnt[0]:.2f} kcal 🏋️‍♂️')
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Note: The predictions are based on historical data and are approximate values. 🧑‍⚕️</p>", unsafe_allow_html=True)
