# Loan Eligibility Prediction Project

This repository contains an end-to-end Machine Learning project designed to predict loan approval eligibility based on various applicant criteria.Project OverviewThe objective of this project is to automate the loan eligibility process using a classification model. The project follows a structured data science workflow, including exploratory data analysis, feature engineering, model training, and web deployment.

## 1. Descriptive Statistics : The first step involves understanding the underlying patterns and distribution of the data. Key metrics such as Mean, Median, and Outliers are analyzed to ensure data quality.

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

### Load the dataset

df = pd.read_csv('loan_data.csv')

### Display basic information and statistics

print(df.head())

print(df.info()) 

print(df.describe())

### Visualize target variable distribution

sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Approval Distribution")
plt.show()

## 2. Data Preprocessing (Cleaning & Transformation) : To prepare the data for machine learning, we handle missing values and perform necessary transformations.

Handling Missing Values: Missing data in critical columns like Loan_Amount and Credit_History is filled using median and mode strategies.   
Log Transformation: To reduce skewness and bring skewed features like Applicant_Income into a normal distribution, Log Transformation is applied.
Label Encoding: Categorical text data is converted into numerical values using Label Encoding so the model can process it.

Pythonimport numpy as np
from sklearn.preprocessing import LabelEncoder

### Filling missing values
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

### Applying Log Transformation
df['ApplicantIncome_Log'] = np.log(df['ApplicantIncome'])
df['LoanAmount_Log'] = np.log(df['LoanAmount'])

### Categorical to Numerical conversion
le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

    
## 3. Model Training (Random Forest Classifier) : Once the data is processed, we train a Random Forest Classifier to learn the patterns between applicant features and loan status.

Pythonfrom sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

### Feature Selection
X = df[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome_Log', 'LoanAmount_Log', 'Credit_History', 'Property_Area']]
y = df['Loan_Status']

### Splitting data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Initializing and training the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

### Exporting the model for deployment
pickle.dump(model, open('loan_model.pkl', 'wb'))


## 4. Web Deployment (app.py) : A user-friendly interface was created using Streamlit, allowing users to input their details and receive real-time loan eligibility predictions.

import streamlit as st
import numpy as np
import pickle

### Load the trained model
model = pickle.load(open('loan_model.pkl', 'rb'))

st.title("Loan Eligibility Predictor")

### User input fields

income = st.number_input("Monthly Income")
loan_amt = st.number_input("Loan Amount")

if st.button("Predict"):  79]
    # Input transformation matching the training data
    income_log = np.log(income) if income > 0 else 0
    loan_log = np.log(loan_amt) if loan_amt > 0 else 0
    
    ### Model prediction
    prediction = model.predict([[1, 1, 0, 1, income_log, loan_log, 1.0, 1]])
    
    if prediction[0] == 1:
        st.success("Loan Approved! ✅")
    else:
        st.error("Loan Rejected! ❌")
        

Key Takeaways -

Descriptive Statistics: Provided a fundamental understanding of the data structure.  
Log Transformation: Normalized skewed data to improve model performance. 
Label Encoding: Translated human-readable labels into a machine-readable format.   
Random Forest: Utilized an ensemble learning method to achieve high prediction accuracy. 
