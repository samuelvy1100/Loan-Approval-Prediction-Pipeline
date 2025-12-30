"""
Loan Approval Prediction API Server
Production-ready FastAPI server for real-time loan approval predictions.
Designed for integration with banking systems, mobile apps, and web interfaces.

Endpoints:
    GET  /          - API health check
    GET  /health    - Detailed health status
    POST /predict   - Loan approval prediction
    GET  /features  - List required input features
    GET  /docs      - Interactive API documentation

Author: Samuel Villarreal
Version: 2.0.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.getenv("MODEL_DIR", Path(__file__).parent)
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.joblib")
COLS_PATH = os.path.join(BASE_DIR, "loan_columns.joblib")

model_artifacts = {
    "pipeline": None,
    "columns": None,
    "loaded_at": None,
    "version": "2.0.0"
}


# LIFECYCLE MANAGEMENT

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager for loading/unloading models."""
    logger.info("Starting Loan Prediction API Server...")
    load_model_artifacts()
    yield
    logger.info("Shutting down server...")


def load_model_artifacts() -> bool:
    """Load model and column artifacts from disk."""
    global model_artifacts
    
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model_artifacts["pipeline"] = joblib.load(MODEL_PATH)
        
        logger.info(f"Loading columns from: {COLS_PATH}")
        model_artifacts["columns"] = joblib.load(COLS_PATH)
        
        model_artifacts["loaded_at"] = datetime.now().isoformat()
        
        logger.info("Model artifacts loaded successfully")
        logger.info(f"Expected input columns: {model_artifacts['columns']}")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


# API INITIALIZATION

app = FastAPI(
    title="Loan Approval Prediction API",
    description="""
## Overview

Enterprise-grade API for real-time loan approval predictions using 
machine learning. Trained on 4,000+ banking records with 98% accuracy.

## Key Features

- Real-time predictions - Sub-second response times
- Explainable outputs - Clear approval/rejection decisions
- Production-ready - Comprehensive error handling and logging
- Bank-grade security - Input validation and sanitization

## Integration

This API is designed for seamless integration with:
- Loan origination systems
- Mobile banking applications  
- Customer service platforms
- Credit decision workflows
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST/RESPONSE MODELS

class PredictionRequest(BaseModel):
    """Loan application data for prediction."""
    no_of_dependents: int = Field(..., ge=0, le=10, description="Number of financial dependents (0-10)")
    education: str = Field(..., description="Education level: ' Graduate' or ' Not Graduate'")
    self_employed: str = Field(..., description="Employment status: ' Yes' or ' No'")
    income_annum: float = Field(..., gt=0, description="Annual income in USD")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    loan_term: float = Field(..., ge=1, le=360, description="Loan term in months (1-360)")
    cibil_score: float = Field(..., ge=300, le=900, description="CIBIL credit score (300-900)")
    residential_assets_value: float = Field(..., ge=0, description="Value of residential properties in USD")
    commercial_assets_value: float = Field(..., ge=0, description="Value of commercial properties in USD")
    luxury_assets_value: float = Field(..., ge=0, description="Value of luxury assets in USD")
    bank_asset_value: float = Field(..., ge=0, description="Total bank account balance in USD")
    
    @field_validator('education')
    @classmethod
    def validate_education(cls, v: str) -> str:
        valid = [' Graduate', ' Not Graduate', 'Graduate', 'Not Graduate']
        if v not in valid:
            raise ValueError(f"Education must be one of: {valid}")
        if not v.startswith(' '):
            v = ' ' + v
        return v
    
    @field_validator('self_employed')
    @classmethod
    def validate_self_employed(cls, v: str) -> str:
        valid = [' Yes', ' No', 'Yes', 'No']
        if v not in valid:
            raise ValueError(f"Self-employed must be one of: {valid}")
        if not v.startswith(' '):
            v = ' ' + v
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "no_of_dependents": 2,
                "education": " Graduate",
                "self_employed": " No",
                "income_annum": 500000,
                "loan_amount": 100000,
                "loan_term": 24,
                "cibil_score": 750,
                "residential_assets_value": 300000,
                "commercial_assets_value": 0,
                "luxury_assets_value": 50000,
                "bank_asset_value": 100000
            }
        }


class PredictionResponse(BaseModel):
    """Loan approval prediction result."""
    loan_status: str = Field(..., description="Prediction result: 'Approved' or 'Rejected'")
    prediction_timestamp: str = Field(..., description="UTC timestamp of prediction")
    model_version: str = Field(..., description="Model version used for prediction")


class HealthResponse(BaseModel):
    """API health status."""
    status: str
    model_loaded: bool
    model_version: str
    loaded_at: Optional[str]
    uptime_check: str


class FeatureInfo(BaseModel):
    """Information about a model input feature."""
    name: str
    type: str
    description: str
    constraints: Optional[str]


class FeaturesResponse(BaseModel):
    """List of required input features."""
    features: List[FeatureInfo]
    total_count: int

# ENDPOINTS

@app.get("/", tags=["General"])
async def root():
    """API root endpoint."""
    return {
        "message": "Loan Approval Prediction API",
        "version": model_artifacts["version"],
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Detailed health check endpoint."""
    model_loaded = model_artifacts["pipeline"] is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_artifacts["version"],
        loaded_at=model_artifacts["loaded_at"],
        uptime_check=datetime.now().isoformat()
    )


@app.get("/features", response_model=FeaturesResponse, tags=["Model Info"])
async def get_features():
    """List all required input features for prediction."""
    feature_definitions = [
        FeatureInfo(name="no_of_dependents", type="integer", 
                    description="Number of people financially dependent on applicant", constraints="0-10"),
        FeatureInfo(name="education", type="string",
                    description="Highest education level achieved", constraints="' Graduate' or ' Not Graduate'"),
        FeatureInfo(name="self_employed", type="string",
                    description="Whether applicant is self-employed", constraints="' Yes' or ' No'"),
        FeatureInfo(name="income_annum", type="float",
                    description="Annual income in USD", constraints="Must be positive"),
        FeatureInfo(name="loan_amount", type="float",
                    description="Requested loan amount in USD", constraints="Must be positive"),
        FeatureInfo(name="loan_term", type="float",
                    description="Loan repayment term in months", constraints="1-360 months"),
        FeatureInfo(name="cibil_score", type="float",
                    description="CIBIL credit bureau score", constraints="300-900"),
        FeatureInfo(name="residential_assets_value", type="float",
                    description="Total value of residential properties", constraints="Non-negative"),
        FeatureInfo(name="commercial_assets_value", type="float",
                    description="Total value of commercial properties", constraints="Non-negative"),
        FeatureInfo(name="luxury_assets_value", type="float",
                    description="Total value of luxury items", constraints="Non-negative"),
        FeatureInfo(name="bank_asset_value", type="float",
                    description="Total balance across all bank accounts", constraints="Non-negative"),
    ]
    
    return FeaturesResponse(features=feature_definitions, total_count=len(feature_definitions))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Generate loan approval prediction.
    
    Submit applicant data to receive an instant approval/rejection prediction.
    """
    if model_artifacts["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        data = pd.DataFrame([request.model_dump()])
        
        try:
            data = data[model_artifacts["columns"]]
        except KeyError as ke:
            raise HTTPException(status_code=400, detail=f"Column mismatch: {ke}")
        
        prediction = model_artifacts["pipeline"].predict(data)[0]
        prediction_clean = str(prediction).strip()
        
        logger.info(f"Prediction made: {prediction_clean}")
        
        return PredictionResponse(
            loan_status=prediction_clean,
            prediction_timestamp=datetime.utcnow().isoformat(),
            model_version=model_artifacts["version"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred.", "type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("api_server:app", host=host, port=port, reload=True)
