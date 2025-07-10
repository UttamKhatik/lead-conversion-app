# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessors
model = joblib.load("model/best_random_forest.pkl")
scaler = joblib.load("model/scaler.pkl")
model_columns = joblib.load("model/model_columns.pkl")

st.set_page_config(page_title="🎓 Lead Conversion Predictor", layout="centered")
st.title("🎓 Educational Lead Conversion Predictor")
st.markdown("Predict which leads are likely to convert using metadata and AI.")

# --- PREDICT FUNCTION ---
def preprocess_and_predict(input_df):
    df = input_df.copy()
    df = df.reindex(columns=model_columns, fill_value=0)
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    return preds

# --- SINGLE INPUT FORM ---
with st.expander("📋 Predict One Lead"):
    st.markdown("Fill in lead metadata to predict conversion:")
    user_input = {}

    for col in model_columns:
        if col in ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']:
            user_input[col] = st.number_input(f"{col}", value=0)
        else:
            user_input[col] = st.selectbox(f"{col}", ['0', '1'])

    if st.button("Predict Conversion"):
        user_df = pd.DataFrame([user_input])
        prediction = preprocess_and_predict(user_df)[0]
        st.success(f"🎯 Prediction: {'Converted' if prediction == 1 else 'Not Converted'}")

# --- BATCH CSV UPLOAD ---
st.markdown("---")
st.header("📁 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a leads CSV file", type=["csv"])

if uploaded_file:
    input_csv = pd.read_csv(uploaded_file)
    st.write("📝 Input Preview", input_csv.head())
    try:
        predictions = preprocess_and_predict(input_csv)
        input_csv['Prediction'] = predictions
        st.write("✅ Prediction Results", input_csv)

        # Download result
        csv = input_csv.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "lead_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
