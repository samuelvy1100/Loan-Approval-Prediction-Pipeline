import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Data upload and cleaning
df = pd.read_csv('/Users/samuel/Desktop/Docs/Portfolio/Github/Loan_Approval/loan_approval_dataset.csv')
df.columns = df.columns.str.strip()

# Confirming columns look good
print("Columns available:", df.columns.tolist())

# 2. Defining variables
input_features = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
target = "loan_status"

X = df[input_features]
y = df[target]

# 3. Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y
)

# 4. Preprocessing
numeric_features = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])

categorical_features = ["education", "self_employed"]
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# 5. Pipeline
clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

clf.fit(X_train, y_train)

# 6. Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Artefacts saving
joblib.dump(clf, "loan_model.joblib")
joblib.dump(input_features, "loan_columns.joblib")
print("Modelo y columnas guardados correctamente.")
