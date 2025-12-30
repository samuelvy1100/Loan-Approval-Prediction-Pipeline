"""
Loan Approval Prediction Model Training Pipeline
This module trains a Random Forest classifier to predict loan approval outcomes
based on applicant financial and demographic data. Designed for banking institutions
to adapt with their proprietary datasets.

Model Performance (on test set):
- Accuracy: ~98%
- Suitable for production loan pre-qualification workflows

Author: Samuel Villarreal
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# CONFIGURATION

class ModelConfig:
    """Central configuration for model training parameters."""
    
    # Feature definitions
    NUMERIC_FEATURES: List[str] = [
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
    
    CATEGORICAL_FEATURES: List[str] = [
        "education",
        "self_employed"
    ]
    
    TARGET_COLUMN: str = "loan_status"
    
    # Model hyperparameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    N_ESTIMATORS: int = 100
    CV_FOLDS: int = 5
    
    # Output paths
    MODEL_FILENAME: str = "loan_model.joblib"
    COLUMNS_FILENAME: str = "loan_columns.joblib"
    METRICS_FILENAME: str = "model_metrics.json"

# DATA LOADING & PREPROCESSING

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset and perform initial validation.
    
    Args:
        filepath: Path to the CSV file containing loan data.
        
    Returns:
        Validated pandas DataFrame.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If required columns are missing.
    """
    logger.info(f"Loading data from: {filepath}")
    
    # Check file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Validate required columns
    required_cols = (
        ModelConfig.NUMERIC_FEATURES + 
        ModelConfig.CATEGORICAL_FEATURES + 
        [ModelConfig.TARGET_COLUMN]
    )
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Log data quality metrics
    logger.info(f"Target distribution:\n{df[ModelConfig.TARGET_COLUMN].value_counts()}")
    logger.info(f"Missing values:\n{df[required_cols].isnull().sum()}")
    
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix and target vector.
    
    Args:
        df: Input DataFrame with all columns.
        
    Returns:
        Tuple of (X, y) where X is feature matrix and y is target series.
    """
    input_features = ModelConfig.NUMERIC_FEATURES + ModelConfig.CATEGORICAL_FEATURES
    
    X = df[input_features].copy()
    y = df[ModelConfig.TARGET_COLUMN].copy()
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    return X, y

# MODEL BUILDING

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Construct the feature preprocessing pipeline.
    
    Returns:
        Configured ColumnTransformer for numeric and categorical features.
    """
    # Numeric feature pipeline
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    
    # Categorical feature pipeline
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ModelConfig.NUMERIC_FEATURES),
            ("cat", categorical_transformer, ModelConfig.CATEGORICAL_FEATURES),
        ],
        remainder='drop'
    )
    
    return preprocessor


def build_model_pipeline() -> Pipeline:
    """
    Build the complete model pipeline including preprocessing and classifier.
    
    Returns:
        Configured sklearn Pipeline ready for training.
    """
    preprocessor = build_preprocessing_pipeline()
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=ModelConfig.N_ESTIMATORS,
            random_state=ModelConfig.RANDOM_STATE,
            n_jobs=-1,  # Use all CPU cores
            class_weight='balanced',  # Handle class imbalance
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )),
    ])
    
    logger.info("Model pipeline constructed successfully")
    return pipeline

# TRAINING & EVALUATION

def train_model(
    pipeline: Pipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Pipeline:
    """
    Train the model pipeline on training data.
    
    Args:
        pipeline: Untrained sklearn Pipeline.
        X_train: Training feature matrix.
        y_train: Training target vector.
        
    Returns:
        Trained pipeline.
    """
    logger.info("Starting model training...")
    start_time = datetime.now()
    
    pipeline.fit(X_train, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame = None,
    y_train: pd.Series = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        pipeline: Trained model pipeline.
        X_test: Test feature matrix.
        y_test: Test target vector.
        X_train: Optional training data for cross-validation.
        y_train: Optional training targets for cross-validation.
        
    Returns:
        Dictionary containing all evaluation metrics.
    """
    logger.info("Evaluating model performance...")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # ROC AUC (handle multi-class)
    try:
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except Exception:
        roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    # Cross-validation (if training data provided)
    cv_scores = None
    if X_train is not None and y_train is not None:
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=ModelConfig.CV_FOLDS, 
            scoring='accuracy'
        )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
        'cv_mean': float(cv_scores.mean()) if cv_scores is not None else None,
        'cv_std': float(cv_scores.std()) if cv_scores is not None else None,
    }
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info("MODEL EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    if roc_auc:
        logger.info(f"ROC-AUC:   {roc_auc:.4f}")
    if cv_scores is not None:
        logger.info(f"CV Mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    logger.info(f"\n{report}")
    logger.info(f"{'='*60}\n")
    
    return metrics


def extract_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract and rank feature importances from the trained model.
    
    Args:
        pipeline: Trained model pipeline.
        feature_names: List of input feature names.
        
    Returns:
        DataFrame with feature importances sorted descending.
    """
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names after preprocessing
    try:
        # Get transformed feature names
        cat_features = preprocessor.named_transformers_['cat']\
            .named_steps['onehot'].get_feature_names_out(
                ModelConfig.CATEGORICAL_FEATURES
            ).tolist()
        all_features = ModelConfig.NUMERIC_FEATURES + cat_features
    except Exception:
        all_features = feature_names
    
    # Get importances
    importances = classifier.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': all_features[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    logger.info("\nFeature Importances:")
    logger.info("-" * 40)
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"{row['feature']:35} {row['importance']:.4f}")
    
    return importance_df

# ARTIFACT SAVING

def save_artifacts(
    pipeline: Pipeline,
    input_columns: List[str],
    metrics: Dict[str, Any],
    output_dir: str = "."
) -> None:
    """
    Save trained model and metadata artifacts.
    
    Args:
        pipeline: Trained model pipeline.
        input_columns: List of input feature column names.
        metrics: Evaluation metrics dictionary.
        output_dir: Directory to save artifacts.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / ModelConfig.MODEL_FILENAME
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save column names
    cols_path = output_path / ModelConfig.COLUMNS_FILENAME
    joblib.dump(input_columns, cols_path)
    logger.info(f"Column names saved to: {cols_path}")
    
    # Save metrics as JSON
    import json
    metrics_path = output_path / ModelConfig.METRICS_FILENAME
    
    # Convert non-serializable items
    metrics_serializable = {k: v for k, v in metrics.items() 
                           if k != 'classification_report'}
    metrics_serializable['classification_report'] = metrics.get('classification_report', '')
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

# MAIN EXECUTION

def main(data_path: str, output_dir: str = ".") -> Dict[str, Any]:
    """
    Execute the complete training pipeline.
    
    Args:
        data_path: Path to the training data CSV file.
        output_dir: Directory to save model artifacts.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    logger.info("="*60)
    logger.info("LOAN APPROVAL MODEL TRAINING PIPELINE")
    logger.info("="*60)
    
    # Load and validate data
    df = load_and_validate_data(data_path)
    
    # Prepare features
    X, y = prepare_features(df)
    input_columns = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=ModelConfig.TEST_SIZE,
        random_state=ModelConfig.RANDOM_STATE,
        stratify=y
    )
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set:  {len(X_test)} samples")
    
    # Build pipeline
    pipeline = build_model_pipeline()
    
    # Train model
    pipeline = train_model(pipeline, X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(pipeline, X_test, y_test, X_train, y_train)
    
    # Extract feature importances
    importance_df = extract_feature_importance(pipeline, input_columns)
    
    # Save artifacts
    save_artifacts(pipeline, input_columns, metrics, output_dir)
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    return metrics


if __name__ == "__main__":
    DATA_PATH = "../data/loan_approval_dataset.csv"
    OUTPUT_DIR = "./"
    
    main(DATA_PATH, OUTPUT_DIR)
