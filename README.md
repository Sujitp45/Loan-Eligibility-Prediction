#Loan Eligibility Prediction Project

This repository contains an end-to-end Machine Learning project designed to predict loan approval eligibility based on various applicant criteria.Project OverviewThe objective of this project is to automate the loan eligibility process using a classification model. The project follows a structured data science workflow, including exploratory data analysis, feature engineering, model training, and web deployment.1. Descriptive StatisticsThe first step involves understanding the underlying patterns and distribution of the data. Key metrics such as Mean, Median, and Outliers are analyzed to ensure data quality.   Pythonimport pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('loan_data.csv') [cite: 9]

# Display basic information and statistics
print(df.head()) [cite: 11]
print(df.info()) [cite: 12]
print(df.describe()) [cite: 13]

# Visualize target variable distribution
sns.countplot(x='Loan_Status', data=df) [cite: 15]
plt.title("Loan Approval Distribution") [cite: 16]
plt.show() [cite: 17]
2. Data Preprocessing (Cleaning & Transformation)To prepare the data for machine learning, we handle missing values and perform necessary transformations.   Handling Missing Values: Missing data in critical columns like LoanAmount and Credit_History is filled using median and mode strategies.   Log Transformation: To reduce skewness and bring skewed features like ApplicantIncome into a normal distribution, Log Transformation is applied.   Label Encoding: Categorical text data is converted into numerical values using Label Encoding so the model can process it.   Pythonimport numpy as np
from sklearn.preprocessing import LabelEncoder

# Filling missing values
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True) [cite: 36]
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True) [cite: 37]

# Applying Log Transformation
df['ApplicantIncome_Log'] = np.log(df['ApplicantIncome']) [cite: 40]
df['LoanAmount_Log'] = np.log(df['LoanAmount']) [cite: 41]

# Categorical to Numerical conversion
le = LabelEncoder() [cite: 44]
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status'] [cite: 45]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col]) [cite: 47]
3. Model Training (Random Forest Classifier)Once the data is processed, we train a Random Forest Classifier to learn the patterns between applicant features and loan status.   Pythonfrom sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Feature Selection
X = df[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome_Log', 'LoanAmount_Log', 'Credit_History', 'Property_Area']] [cite: 57]
y = df['Loan_Status'] [cite: 58]

# Splitting data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) [cite: 60]

# Initializing and training the model
model = RandomForestClassifier(n_estimators=100) [cite: 62]
model.fit(X_train, y_train) [cite: 63]

# Exporting the model for deployment
pickle.dump(model, open('loan_model.pkl', 'wb')) [cite: 65]
4. Web Deployment (app.py)A user-friendly interface was created using Streamlit, allowing users to input their details and receive real-time loan eligibility predictions.   Pythonimport streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('loan_model.pkl', 'rb')) [cite: 74]

st.title("Loan Eligibility Predictor") [cite: 75]

# User input fields
income = st.number_input("Monthly Income") [cite: 77]
loan_amt = st.number_input("Loan Amount") [cite: 78]

if st.button("Predict"): [cite: 79]
    # Input transformation matching the training data
    income_log = np.log(income) if income > 0 else 0 [cite: 81]
    loan_log = np.log(loan_amt) if loan_amt > 0 else 0 [cite: 82]
    
    # Model prediction
    prediction = model.predict([[1, 1, 0, 1, income_log, loan_log, 1.0, 1]]) [cite: 84]
    
    if prediction[0] == 1:
        st.success("Loan Approved! ✅") [cite: 86]
    else:
        st.error("Loan Rejected! ❌") [cite: 88]
Key TakeawaysDescriptive Statistics: Provided a fundamental understanding of the data structure.   Log Transformation: Normalized skewed data to improve model performance.   Label Encoding: Translated human-readable labels into a machine-readable format.   Random Forest: Utilized an ensemble learning method to achieve high prediction accuracy. 
