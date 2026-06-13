import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("RidgeModel.pkl", "rb"))

st.title("House Price Prediction")

location = st.text_input("Location")
sqft = st.number_input("Total Sqft")
bath = st.number_input("Bath")
bhk = st.number_input("BHK")

if st.button("Predict"):
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location','total_sqft','bath','bhk'])
    result = model.predict(input_df)[0]
    st.success(f"Predicted Price: {round(result,2)} Lakhs")