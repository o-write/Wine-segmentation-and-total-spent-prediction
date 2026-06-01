import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models
rf_model = pickle.load(open('model_rf.pkl', 'rb'))
reg_model = pickle.load(open('model_reg.pkl', 'rb'))
scaler = pickle.load(open('reg_scaler.pkl', 'rb'))

st.title("🍷 Wine Customer Segment Predictor")
st.markdown("Aplikasi untuk memprediksi segmen pelanggan dan estimasi pengeluaran.")

# Input Form
with st.sidebar:
    st.header("Input Data Pelanggan")
    income = st.number_input("Pendapatan Tahunan (Rp)", min_value=0, value=500000000)
    age = st.slider("Usia", 18, 100, 30)
    education = st.selectbox("Pendidikan", options=[0, 1, 2], format_func=lambda x: ["Undergraduate", "Graduate", "Postgraduate"][x])
    is_parent = st.radio("Apakah Orang Tua?", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya")
    children = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=0)

if st.button("Analisis Pelanggan"):
    # Prepare data
    input_data = pd.DataFrame([{"Income": income, "Age": age, "Education2": education, "Parent2": is_parent, "Num_Children": children}])
    
    # Classification
    pred_class = rf_model.predict(input_data)[0]
    probs = rf_model.predict_proba(input_data).max()
    
    # Regression
    scaled_input = scaler.transform(input_data)
    pred_amt = max(0, reg_model.predict(scaled_input)[0])
    
    segments = {
        0: 'Uninterested',
        1: 'Premium Wine Enthusiasts',
        2: 'Budget Wine Loyalists',
        3: 'Potential Wine Converts'
    }
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Segmen Prediksi", segments[pred_class])
        st.write(f"Confidence: {probs:.2%}")
    with col2:
        st.metric("Estimasi Pengeluaran", f"Rp {pred_amt:,.2f}")
