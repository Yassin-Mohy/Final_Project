import streamlit as st
st.title("My Telecom Churn Analysis")
st.write("Hello, this is my Streamlit app!")

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('churn_model.pkl')

# Title and navigation
st.title("Churn Prediction Dashboard")
option = st.sidebar.radio("Select an option:", ["Exploratory Data Analysis (EDA)", "Churn Prediction"])

# Churn Prediction Section
if option == "Churn Prediction":
    st.header("Churn Prediction")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    estimated_salary = st.number_input("Estimated Salary", min_value=0, value=50000)
    calls_made = st.number_input("Calls Made", min_value=0, value=100)
    sms_sent = st.number_input("SMS Sent", min_value=0, value=50)
    data_used = st.number_input("Data Used (GB)", min_value=0, value=5)

    if st.button("Predict Churn"):
        input_data = pd.DataFrame({
            'age': [age],
            'estimated_salary': [estimated_salary],
            'calls_made': [calls_made],
            'sms_sent': [sms_sent],
            'data_used': [data_used]
        })
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("The customer is likely to churn.")
        else:
            st.success("The customer is not likely to churn.")
        st.write(f"Probability of churn: {probability:.2%}")
