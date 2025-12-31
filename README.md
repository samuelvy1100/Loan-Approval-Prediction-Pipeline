# Loan Approval Prediction Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to predict loan approval outcomes using applicant financial and demographic data. It features a robust Random Forest training workflow and a professional Streamlit interface for real-time risk assessment.

**Live Demo:** [View Application on Streamlit](https://machine-learning-loan-approval-prediction-pipeline.streamlit.app/)

---

## Project Overview

This project demonstrates the transition from an academic research-oriented ML model to a production-ready application. It automates the pre-qualification process for banking institutions by analyzing applicant creditworthiness through 11 key financial features.

- **Primary Objective:** Create a supervised machine learning model to analyze and predict `loan_status` (Approved/Rejected) based on historical applicant profiles
- **Key Tech Stack:** Python, Scikit-Learn, Pandas, Joblib, Streamlit
- **Architecture:** Modular training script with an automated preprocessing pipeline and a decoupled web interface

---

## Dataset Overview
  - **Features**:  
    1. `no_of_dependents` (integer): Number of dependents of the applicant.  
    2. `education` (categorical): “Graduate” or “Not Graduate”.  
    3. `self_employed` (categorical): “Yes” or “No”.  
    4. `income_annum` (float): Applicant’s annual income (USD).  
    5. `loan_amount` (float): Requested loan amount (USD).  
    6. `loan_term` (float): Requested loan duration in months.  
    7. `cibil_score` (float): Applicant's credit score (CIBIL).  
    8. `residential_assets_value` (float): Value of owned residential assets (USD).  
    9. `commercial_assets_value` (float): Value of owned commercial assets (USD).  
    10. `luxury_assets_value` (float): Value of luxury assets (USD).  
    11. `bank_asset_value` (float): Total bank assets (USD).  
  - **Target Variable**: `loan_status` (binary categorical: “Approved” or “Denied”).  

---

## Model Performance & Metrics

The model was trained on a dataset of 4,269 records using a Random Forest classifier with a 80/20 stratified split and validated through 5-fold cross-validation. 

| Metric | Score |
|--------|-------|
| Accuracy | 98.36% |
| Precision | 98.36% |
| Recall | 98.36% |
| F1-Score | 0.9836 |
| ROC-AUC | 0.9972 |
| CV Mean | 97.60% (+/- 0.0075) |

### Feature Importance (Top 5)

1. **CIBIL Score:** 82.74%
2. **Loan Term:** 5.09%
3. **Loan Amount:** 2.87%
4. **Luxury Assets Value:** 1.71%
5. **Annual Income:** 1.70%

---

## Industry Applications

- **Instant Pre-Qualification:** Reducing initial screening time from days to seconds by providing real-time eligibility feedback
- **Risk Stratification:** Categorizing applicants into risk tiers for differentiated interest rates and loan terms
- **Regulatory Compliance:** Using feature importance and decision factors to provide transparent audit trails for lending decisions

---

## Data Disclosure & Limitations

- **Data Source:** This project utilizes CIBIL scores and data records based on publicly available information from India
- **Demonstration Only:** This system is intended for demonstration and portfolio purposes and is not suitable for actual financial deployment without further rigorous testing
- **Expansion Potential:** While current results are strong, the model's insights would improve significantly with access to larger, proprietary banking datasets containing more diverse features such as transaction history or credit utilization ratios

---

## How it Works

1. **Preprocessing:** Uses a `ColumnTransformer` to handle `StandardScaler` for numeric values and `OneHotEncoder` for categorical data
2. **Imputation:** Automatically handles missing values via mean (numeric) and mode (categorical) strategies to ensure pipeline stability
3. **Training:** Employs a `RandomForestClassifier` with balanced class weights to address the target distribution of 62% Approved vs 38% Rejected
4. **Interface:** A Streamlit-based dark-themed UI that provides dynamic risk metrics (DTI ratio, Asset Coverage) alongside the ML prediction
5. **Optional REST API Mode:** For enterprise integration, mobile apps, or microservices architecture, deploy the included FastAPI server (api_server.py) separately.


