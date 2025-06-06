## Loan Approval Prediction Pipeline  

**🔗 Files**:  
- [`loan_approval_dataset.csv`](./loan_approval_dataset.csv)  
- [`modelo_entrenamiento.py`](./modelo_entrenamiento.py)  
- [`loan_model.joblib`](./loan_model.joblib)  
- [`loan_columns.joblib`](./loan_columns.joblib)  
- [`ml_server.py`](./ml_server.py)  
- [`frontend.py`](./frontend.py)  

#### 📚 Overview
- **Objective**: Design an end-to-end machine learning pipeline to predict whether a loan application is approved or denied, using applicant personal financial features. Furthermore, deploy the model as a REST API and provide a Streamlit-based web interface for end users.
- **Description**: This application evaluates an individual's loan application by analyzing personal and financial details to determine loan approval eligibility. It uses a machine learning model trained on over 4,000 real-world banking records from India, achieving 98% predictive accuracy. One of the most important features is the applicant’s CIBIL score (300 = poor creditworthiness, 900 = excellent creditworthiness). 
- **Dataset**:  
  - **“loan_approval_dataset.csv”** (collected from financial institution records):  
    - **Features**:  
      1. `no_of_dependents` (integer): Number of dependents of the applicant.  
      2. `education` (categorical): “Graduate” or “Not Graduate”.  
      3. `self_employed` (categorical): “Yes” or “No”.  
      4. `income_annum` (float): Applicant’s annual income (USD).  
      5. `loan_amount` (float): Requested loan amount (USD).  
      6. `loan_term` (float): Requested loan duration in months.  
      7. `cibil_score` (float): Creditworthiness score (CIBIL).  
      8. `residential_assets_value` (float): Value of owned residential assets (USD).  
      9. `commercial_assets_value` (float): Value of owned commercial assets (USD).  
      10. `luxury_assets_value` (float): Value of luxury assets (USD).  
      11. `bank_asset_value` (float): Total bank assets (USD).  
    - **Target**: `loan_status` (binary categorical: “Approved” or “Denied”).  

#### 🧠 Methodology
1. **Data Loading & Cleaning**  
   - Loaded raw CSV via `pd.read_csv("loan_approval_dataset.csv")`.  
   - Stripped whitespace from column names to ensure consistent referencing.  

2. **Feature–Target Definition**  
   - **X**: Subset of columns:  
     - Numeric: `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`.  
     - Categorical: `education`, `self_employed`.  
   - **y**: `loan_status`.  

3. **Train/Test Split**  
   - Partitioned data into 80% training and 20% testing with stratification on `loan_status` to preserve class ratios (`random_state=42`).  

4. **Preprocessing Pipeline**  
   - **Numeric Transformer** (`numeric_transformer`):  
     - `SimpleImputer(strategy="mean")` to fill missing values.  
     - `StandardScaler()` to normalize numeric features.  
   - **Categorical Transformer** (`categorical_transformer`):  
     - `SimpleImputer(strategy="most_frequent")` to fill missing strings.  
     - `OneHotEncoder(handle_unknown="ignore")` to convert categories into binary indicator columns.  
   - **ColumnTransformer** (`preprocessor`):  
     - Applies `numeric_transformer` to numeric features.  
     - Applies `categorical_transformer` to `education` and `self_employed`.  

5. **Model Training**  
   - Assembled a scikit-learn `Pipeline`:  
     1. **Step “preprocessor”**: The `ColumnTransformer` defined above.  
     2. **Step “classifier”**: `RandomForestClassifier(n_estimators=100, random_state=42)`.  
   - Fitted the pipeline on `X_train` and `y_train`.  

6. **Evaluation**  
   - Predicted `y_pred = clf.predict(X_test)`.  
   - Generated classification metrics (`classification_report`) including **precision**, **recall**, **F1-score**, and **accuracy** per class.  
   - Observed strong performance with overall F1 score of 0.98 indicating robust generalization on held-out data.  

7. **Artifact Serialization**  
   - **Model**: Saved final fitted pipeline as `loan_model.joblib` using `joblib.dump()`.  
   - **Column List**: Persisted `input_features` list as `loan_columns.joblib` for consistent feature ordering at inference time.  

8. **API Deployment (FastAPI)**  
   - Developed `ml_server.py` to serve prediction endpoints:  
     - **Endpoint**: `POST /predict` accepts JSON payload matching `PredictionRequest` schema (fields identical to input features).  
     - **Server Logic**:  
       - Load `loan_model.joblib` and `loan_columns.joblib` on startup.  
       - Reorder incoming JSON to match training feature order.  
       - Apply pipeline’s `predict()` and return `{"loan_status": "<Approved/Denied>"}`.  
   - **Usage**: Run via `uvicorn ml_server:app --reload` (default `localhost:8000`).  

9. **Web Front-End (Streamlit)**  
   - Created `frontend.py` as a user interface:  
     - Users input values for each predictor via Streamlit widgets (e.g., `st.number_input`, `st.selectbox`).  
     - On clicking “Predict”, sends a `POST` request to `http://127.0.0.1:8000/predict` with JSON payload.  
     - Displays predicted `loan_status` with color-coded success/error feedback.  
   - **Future Work**:  
     - Host Streamlit app on a cloud platform (Heroku, AWS, etc.) to allow remote users to assess loan applications in real time.  

#### 🎯 Results & Takeaways
- **Model Performance**: Overall F1 score of 0.98. Random Forest classifier achieved high precision and recall for both “Approved” and “Denied” classes, indicating minimal bias toward either outcome.  
- **Feature Importance**:  
  - Top predictive features included **CIBIL score**, **annual income**, and **loan amount**, aligning with domain knowledge of credit risk assessment.  
  - Asset values (residential, commercial, bank) provided additional signal for applicant solvency.  
- **Operational Value**:  
  - Financial institutions can embed this pipeline in their loan origination systems to rapidly triage low-risk candidates and flag high-risk applications.  
  - Reduces manual review time and allows credit officers to focus on exceptions and appeals.  

#### 🔧 Technologies Used
- **Python 3.x**  
- **pandas & NumPy** for data ingestion and manipulation  
- **scikit-learn** for preprocessing, model training, and evaluation  
- **Joblib** for artifact serialization  
- **FastAPI** for RESTful model serving  
- **Streamlit** for interactive front-end development  

#### 🌐 Practical Application
- **Credit Underwriting**: Automate preliminary loan decisioning to accelerate turnaround and minimize default risk.  
- **Risk Management**: Provide explainable feature importances to credit officers for auditing and regulatory compliance.  
- **Operational Efficiency**: Integrate with existing ERP/CRM systems (e.g., PeopleSoft, QuickBooks) to streamline data flow.

#### 💻 Interace
<br/>
<img src="https://i.imgur.com/IkjQxFL.png" width="1000" alt="Application Page 1"/>
<br />
<br />
<img src="https://i.imgur.com/3u25rXp.png" width="1000" alt="Application Page 1"/>
<br />
<br />
