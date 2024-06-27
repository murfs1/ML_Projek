import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load the trained model
model = joblib.load('gold_stock_model.pkl')

# Title of the app
st.title("Gold Stock Price Prediction")

# Sidebar inputs
st.sidebar.header("Input Features")

def user_input_features():
    Volume = st.sidebar.number_input("Volume", min_value=0.0, value=200000.0)
    Open = st.sidebar.number_input("Open Price", min_value=0.0, value=2000.0)
    High = st.sidebar.number_input("High Price", min_value=0.0, value=2050.0)
    Low = st.sidebar.number_input("Low Price", min_value=0.0, value=2000.0)
    
    data = {'Volume': Volume,
            'Open': Open,
            'High': High,
            'Low': Low}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main panel
st.subheader("Input Features")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)

st.subheader("Predicted Close Price")
st.write(prediction[0])
