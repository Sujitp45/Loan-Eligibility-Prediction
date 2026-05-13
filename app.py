
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# मॉडेल लोड करणे
model = pickle.load(open('loan_model.pkl', 'rb'))

st.title("🏦 Loan Eligibility Predictor")

# इनपुट्स
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
loan_term = st.number_input("Loan Term (Days)", value=360)
app_income = st.number_input("Applicant Income", value=5000)
coapp_income = st.number_input("Co-Applicant Income", value=0)
loan_amt = st.number_input("Loan Amount", value=150)

if st.button("Check Eligibility"):
    total_income = app_income + coapp_income
    t_log = np.log(total_income) if total_income > 0 else 0
    l_log = np.log(loan_amt) if loan_amt > 0 else 0
    
    # ट्रेनिंगमध्ये वापरलेला डेटा फॉरमॅट
    data = [[1 if gender=="Male" else 0, 1 if married=="Yes" else 0, 
             3 if dependents=="3+" else int(dependents), 0 if education=="Graduate" else 1,
             1 if self_employed=="Yes" else 0, loan_term, credit_history,
             2 if property_area=="Urban" else (1 if property_area=="Semiurban" else 0),
             t_log, l_log]]
             
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.success("✅ अभिनंदन! तुमचे कर्ज मंजूर होऊ शकते.")
    else:
        st.error("❌ क्षमस्व! तुम्ही सध्या कर्जासाठी पात्र नाही आहात.")
    