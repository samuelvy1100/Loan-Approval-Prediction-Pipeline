from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Inicializar app
app = FastAPI(
    title="Loan Status Prediction API",
    description="API to predict if a loan will be approved or not approved",
    version="1.0",
)

# 2. Rutas absolutas a los artefactos (ajusta si cambias de carpeta)
BASE_DIR = "/Users/samuel/Desktop/Docs/Portfolio/Github/Loan_Approval"
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.joblib")
COLS_PATH  = os.path.join(BASE_DIR, "loan_columns.joblib")

try:
    pipeline = joblib.load(MODEL_PATH)
    input_columns = joblib.load(COLS_PATH)
except Exception as e:
    pipeline = None
    print(f"Loading error: {e}")

# 3. Server prediction request
class PredictionRequest(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

# 4. Server prediction response 
class PredictionResponse(BaseModel):
    loan_status: str

# 5. Root
@app.get("/")
def read_root():
    return {"message": "API Loan Status Prediction. Use /docs to explore endpoints."}

# 6. Endpoint of prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not available")

    # Converting petition into DataFrame
    data = pd.DataFrame([req.model_dump()])
    # Reordering columns according to pipeline
    try:
        data = data[input_columns]
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing Column: {ke}")

    # Prediction
    pred = pipeline.predict(data)[0]
    return {"loan_status": str(pred)}
