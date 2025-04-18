import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset (cached for performance)
@st.cache_data
def load_data():
    return pd.read_csv('telecom_churn.csv')

data = load_data()

# Train the model (inside the app to avoid joblib)
@st.cache_resource
def train_model(df):
    X = df[['age', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used']]
    y = df['churn']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(data)

# Title
st.title("Churn Prediction Dashboard")

# Sidebar
st.sidebar.header("Navigation")
option = st.sidebar.radio("Select an option:", ["Exploratory Data Analysis (EDA)", "Churn Prediction"])

# EDA Section
if option == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Overview")
    st.write(data.head())

    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.subheader("Churn Distribution")
    churn_counts = data['churn'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Prediction Section
elif option == "Churn Prediction":
    st.header("Churn Prediction")

    st.subheader("Enter Customer Details")
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

        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("The customer is likely to churn.")
        else:
            st.success("The customer is not likely to churn.")
        st.write(f"Probability of churn: {probability:.2%}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")

