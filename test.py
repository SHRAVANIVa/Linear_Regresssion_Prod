import os
import joblib
import streamlit as st
import pandas as pd

# Load the trained model
model = joblib.load("models/linear_regression_model.pkl")


input_data = pd.DataFrame({
    'TV': [100],
    'Radio': [50],
    'Newspaper': [20]
})

# Predict sales
prediction = model.predict(input_data)
print("Predicted Sales:", prediction[0])