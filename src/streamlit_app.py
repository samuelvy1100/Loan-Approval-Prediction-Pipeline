"""
Loan Approval Prediction - Professional Streamlit Interface
Standalone web interface for the loan approval prediction system.
Dark theme design with three-tab navigation.

Author: Samuel Villarreal
Version: 2.1.0 (Standalone for Streamlit Cloud)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Loan Approval Pipeline",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# MODEL LOADING (Cached for performance)
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained model pipeline. Cached to avoid reloading on each interaction."""
    try:
        # Model files are in the same directory as this script (src/)
        base_path = Path(__file__).parent
        model_path = base_path / "loan_model.joblib"
        columns_path = base_path / "loan_columns.joblib"
        
        pipeline = joblib.load(model_path)
        columns = joblib.load(columns_path)
        
        return pipeline, columns, None
    except FileNotFoundError as e:
        return None, None, f"Model files not found: {e}"
    except Exception as e:
        return None, None, f"Error loading model: {e}"

# Load model at startup
MODEL_PIPELINE, MODEL_COLUMNS, MODEL_ERROR = load_model()

# CONFIGURATION

EDUCATION_OPTIONS = {
    "Graduate (Master's or higher)": " Graduate",
    "Not Graduate": " Not Graduate"
}

SELF_EMPLOYED_OPTIONS = {
    "Salaried Employee": " No",
    "Self-Employed": " Yes"
}

# CUSTOM CSS

def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0c1929 0%, #1a365d 50%, #0f2744 100%);
            font-family: 'DM Sans', 'Segoe UI', system-ui, sans-serif;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
            color: #e2e8f0 !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div {
            background: rgba(15, 39, 68, 0.6) !important;
            border: 1px solid rgba(99, 179, 237, 0.25) !important;
            border-radius: 10px !important;
            color: white !important;
        }
        
        .stSelectbox > div > div {
            background: rgba(15, 39, 68, 0.6) !important;
            border-radius: 10px !important;
        }
        
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #ef4444 0%, #f97316 25%, #eab308 50%, #22c55e 75%, #10b981 100%) !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #3182ce 0%, #4299e1 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 16px 32px !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            box-shadow: 0 4px 20px rgba(49, 130, 206, 0.4) !important;
            width: 100% !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center !important;
            background: rgba(26, 54, 93, 0.5) !important;
            border-radius: 12px !important;
            padding: 6px !important;
            gap: 8px !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 8px !important;
            color: #90cdf4 !important;
            font-weight: 600 !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3182ce 0%, #4299e1 100%) !important;
            color: white !important;
        }
        
        .stTabs [data-baseweb="tab-border"], .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        
        [data-testid="stMetricValue"] {
            color: #63b3ed !important;
            font-size: 28px !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
        }
        
        .streamlit-expanderHeader {
            background: rgba(26, 54, 93, 0.4) !important;
            border: 1px solid rgba(99, 179, 237, 0.15) !important;
            border-radius: 12px !important;
        }
        
        code {
            background: rgba(15, 39, 68, 0.6) !important;
            color: #90cdf4 !important;
        }
        
        pre {
            background: rgba(15, 39, 68, 0.6) !important;
            border-radius: 12px !important;
        }
    </style>
    """, unsafe_allow_html=True)


# HELPER FUNCTIONS

def get_cibil_category(score: float) -> Tuple[str, str, str]:
    if score >= 800: return "Excellent", "#10b981", "🌟"
    elif score >= 750: return "Very Good", "#22c55e", "✨"
    elif score >= 700: return "Good", "#84cc16", "👍"
    elif score >= 650: return "Fair", "#eab308", "⚡"
    elif score >= 550: return "Poor", "#f97316", "⚠️"
    else: return "Very Poor", "#ef4444", "🔴"


def calculate_risk_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    total_assets = (data["residential_assets_value"] + data["commercial_assets_value"] + 
                    data["luxury_assets_value"] + data["bank_asset_value"])
    
    loan_to_income = data["loan_amount"] / max(data["income_annum"], 1)
    asset_coverage = total_assets / max(data["loan_amount"], 1)
    monthly_payment = data["loan_amount"] / max(data["loan_term"], 1)
    monthly_income = data["income_annum"] / 12
    dti_ratio = (monthly_payment / max(monthly_income, 1)) * 100
    
    score = 0
    score += min(40, ((data["cibil_score"] - 300) / 600) * 40)
    score += min(25, (1 / max(loan_to_income, 0.1)) * 5)
    score += min(20, asset_coverage * 5)
    score += min(10, data["bank_asset_value"] / max(data["loan_amount"], 1) * 10)
    score += 3 if data.get("education") == "Graduate (Master's or higher)" else 0
    score += 2 if data.get("self_employed") == "Salaried Employee" else 0
    
    return {
        "total_assets": total_assets,
        "loan_to_income": loan_to_income,
        "asset_coverage": asset_coverage,
        "monthly_payment": monthly_payment,
        "dti_ratio": dti_ratio,
        "risk_score": min(100, max(0, score))
    }


def get_decision_factors(data: Dict[str, Any], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    factors = []
    
    if data["cibil_score"] >= 750:
        factors.append({"text": "Excellent credit score", "positive": True})
    elif data["cibil_score"] >= 650:
        factors.append({"text": "Good credit score", "positive": True})
    else:
        factors.append({"text": "Credit score needs improvement", "positive": False})
    
    if metrics["loan_to_income"] < 2:
        factors.append({"text": "Strong income-to-loan ratio", "positive": True})
    elif metrics["loan_to_income"] > 3:
        factors.append({"text": "High loan amount relative to income", "positive": False})
    
    if metrics["asset_coverage"] >= 1.5:
        factors.append({"text": "Excellent asset coverage", "positive": True})
    elif metrics["asset_coverage"] < 0.5:
        factors.append({"text": "Insufficient collateral coverage", "positive": False})
    
    if metrics["dti_ratio"] < 30:
        factors.append({"text": "Manageable debt-to-income ratio", "positive": True})
    elif metrics["dti_ratio"] > 50:
        factors.append({"text": "High monthly payment burden", "positive": False})
    
    return factors


def validate_inputs(data: Dict[str, Any]) -> List[str]:
    errors = []
    if data["income_annum"] <= 0: errors.append("Annual income must be greater than 0")
    if data["loan_amount"] <= 0: errors.append("Loan amount must be greater than 0")
    if data["loan_term"] < 1: errors.append("Loan term must be at least 1 month")
    if not 300 <= data["cibil_score"] <= 900: errors.append("CIBIL score must be between 300 and 900")
    return errors


def predict_loan_approval(data: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Make prediction using the loaded model directly.
    No external API call needed - runs entirely within Streamlit.
    """
    if MODEL_PIPELINE is None:
        return None, MODEL_ERROR or "Model not loaded"
    
    try:
        # Prepare input data
        input_data = {
            "no_of_dependents": data["no_of_dependents"],
            "education": EDUCATION_OPTIONS[data["education"]],
            "self_employed": SELF_EMPLOYED_OPTIONS[data["self_employed"]],
            "income_annum": data["income_annum"],
            "loan_amount": data["loan_amount"],
            "loan_term": data["loan_term"],
            "cibil_score": data["cibil_score"],
            "residential_assets_value": data["residential_assets_value"],
            "commercial_assets_value": data["commercial_assets_value"],
            "luxury_assets_value": data["luxury_assets_value"],
            "bank_asset_value": data["bank_asset_value"]
        }
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([input_data])
        df = df[MODEL_COLUMNS]
        
        # Make prediction
        prediction = MODEL_PIPELINE.predict(df)[0]
        prediction_clean = str(prediction).strip()
        
        return {"loan_status": prediction_clean}, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


# RENDER HEADER

def render_header():
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 16px;">
            <div style="width: 56px; height: 56px; background: linear-gradient(135deg, #3182ce 0%, #63b3ed 100%); 
                        border-radius: 14px; display: flex; align-items: center; justify-content: center; 
                        font-size: 28px; font-weight: bold; color: white; box-shadow: 0 4px 20px rgba(49, 130, 206, 0.4);">
                🏦
            </div>
            <span style="font-size: 32px; font-weight: 700; letter-spacing: -0.5px; 
                         background: linear-gradient(90deg, #fff 0%, #90cdf4 100%); 
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Loan Approval Pipeline
            </span>
        </div>
        <p style="font-size: 18px; color: #94a3b8; max-width: 700px; margin: 0 auto; line-height: 1.6;">
            Production-ready machine learning pipeline trained on real-world banking records, delivering 
            <span style="color: #10b981; font-weight: 600;">98% predictive accuracy</span> for instant loan decisions given applicant's profile. 
            Built and ready for deployment with REST API integration, custom retraining support, and explainable risk scoring.
        </p>
        <div style="display: flex; justify-content: center; gap: 12px; margin-top: 20px; flex-wrap: wrap;">
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); 
                         padding: 8px 20px; border-radius: 100px; font-size: 14px; color: #10b981;">
                ✓ 98% Accuracy
            </span>
            <span style="background: rgba(49, 130, 206, 0.15); border: 1px solid rgba(49, 130, 206, 0.3); 
                         padding: 8px 20px; border-radius: 100px; font-size: 14px; color: #3182ce;">
                ⚡ Real-Time Analysis
            </span>
            <span style="background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.3); 
                         padding: 8px 20px; border-radius: 100px; font-size: 14px; color: #8b5cf6;">
                🔒 Bank-Grade Security
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# PREDICTOR TAB

def render_predictor_tab():
    with st.expander("How to Use This Tool", expanded=False):
        st.markdown("""
        1. **Enter applicant details:** Fill in personal information including education level and employment status.
        2. **Provide financial data:** Input accurate income, loan amount, and CIBIL score (most influential factor).
        3. **Declare assets:** Enter total asset values as collateral indicators.
        4. **Get instant prediction:** Click "Analyze Application" to receive results.
        """)
    
    col_form, col_results = st.columns([3, 2], gap="large")
    
    with col_form:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #3182ce 0%, #4299e1 100%); 
                        border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">📝</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">Loan Application Form</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Personal Information
        st.markdown('<p style="font-size: 13px; font-weight: 600; color: #90cdf4; text-transform: uppercase; letter-spacing: 1px; margin-top: 20px;">Personal Information</p>', unsafe_allow_html=True)
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        with p_col2:
            education = st.selectbox("Education Level", options=list(EDUCATION_OPTIONS.keys()))
        with p_col3:
            self_employed = st.selectbox("Employment Type", options=list(SELF_EMPLOYED_OPTIONS.keys()))
        
        # Financial Information
        st.markdown('<p style="font-size: 13px; font-weight: 600; color: #90cdf4; text-transform: uppercase; letter-spacing: 1px; margin-top: 24px;">Financial Information</p>', unsafe_allow_html=True)
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            income_annum = st.number_input("Annual Income (USD)", min_value=0.0, value=500000.0, step=10000.0, format="%.0f")
            loan_amount = st.number_input("Loan Amount Requested (USD)", min_value=0.0, value=100000.0, step=10000.0, format="%.0f")
        with f_col2:
            loan_term = st.number_input("Loan Term (Months)", min_value=1, max_value=360, value=24)
            cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=700)
            category, color, icon = get_cibil_category(cibil_score)
            st.markdown(f"""
            <div style="background: {color}22; border: 1px solid {color}55; border-radius: 10px; 
                        padding: 12px; text-align: center; margin-top: -10px;">
                <span style="font-size: 28px; font-weight: 700; color: {color};">{cibil_score}</span>
                <span style="font-size: 14px; color: {color}; margin-left: 12px;">{icon} {category}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Asset Declaration
        st.markdown('<p style="font-size: 13px; font-weight: 600; color: #90cdf4; text-transform: uppercase; letter-spacing: 1px; margin-top: 24px;">Asset Declaration (USD)</p>', unsafe_allow_html=True)
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            residential_assets = st.number_input("Residential Property Value", min_value=0.0, value=200000.0, step=10000.0, format="%.0f")
            commercial_assets = st.number_input("Commercial Property Value", min_value=0.0, value=0.0, step=10000.0, format="%.0f")
        with a_col2:
            luxury_assets = st.number_input("Luxury Assets Value", min_value=0.0, value=50000.0, step=5000.0, format="%.0f")
            bank_assets = st.number_input("Bank Account Balance", min_value=0.0, value=100000.0, step=10000.0, format="%.0f")
        
        form_data = {
            "no_of_dependents": no_of_dependents, "education": education, "self_employed": self_employed,
            "income_annum": income_annum, "loan_amount": loan_amount, "loan_term": loan_term,
            "cibil_score": cibil_score, "residential_assets_value": residential_assets,
            "commercial_assets_value": commercial_assets, "luxury_assets_value": luxury_assets,
            "bank_asset_value": bank_assets
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyze Application", use_container_width=True):
            errors = validate_inputs(form_data)
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                with st.spinner("Analyzing application..."):
                    time.sleep(0.3)  # Brief delay for UX
                    result, error = predict_loan_approval(form_data)
                if error:
                    st.error(f"🚫 {error}")
                else:
                    st.session_state["prediction_result"] = result
                    st.session_state["form_data"] = form_data
                    st.rerun()
    
    with col_results:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #10b981 0%, #22c55e 100%); 
                        border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">📊</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">Risk Assessment</span>
        </div>
        """, unsafe_allow_html=True)
        
        metrics = calculate_risk_metrics(form_data)
        category, color, icon = get_cibil_category(cibil_score)
        score_pct = ((cibil_score - 300) / 600) * 100
        
        # CIBIL Gauge
        st.markdown(f"""
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 14px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 13px; color: #94a3b8;">CIBIL Score Position</span>
                <span style="font-size: 18px; font-weight: 700; color: {color};">{cibil_score}</span>
            </div>
            <div style="height: 12px; background: linear-gradient(90deg, #ef4444 0%, #f97316 25%, #eab308 50%, #22c55e 75%, #10b981 100%); 
                        border-radius: 6px; position: relative;">
                <div style="position: absolute; left: {score_pct}%; top: -6px; width: 4px; height: 24px; 
                            background: white; border-radius: 2px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); 
                            transform: translateX(-50%);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 11px; color: #64748b;">
                <span>300 (Poor)</span><span>900 (Excellent)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Total Assets", format_currency(metrics["total_assets"]))
            lti_delta = "Good" if metrics["loan_to_income"] < 2 else "High"
            st.metric("Loan/Income Ratio", f"{metrics['loan_to_income']:.2f}x", delta=lti_delta,
                     delta_color="normal" if metrics["loan_to_income"] < 2 else "inverse")
        with m_col2:
            ac_delta = "Strong" if metrics["asset_coverage"] > 1.5 else "Low"
            st.metric("Asset Coverage", f"{metrics['asset_coverage']:.2f}x", delta=ac_delta,
                     delta_color="normal" if metrics["asset_coverage"] > 1.5 else "inverse")
            dti_delta = "Healthy" if metrics["dti_ratio"] < 30 else "High"
            st.metric("Debt-to-Income", f"{metrics['dti_ratio']:.1f}%", delta=dti_delta,
                     delta_color="normal" if metrics["dti_ratio"] < 30 else "inverse")
        
        # Risk Score
        risk_color = "#10b981" if metrics["risk_score"] >= 70 else ("#eab308" if metrics["risk_score"] >= 50 else "#ef4444")
        st.markdown(f"""
        <div style="margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 13px; color: #94a3b8;">Overall Risk Score</span>
                <span style="font-size: 14px; font-weight: 600; color: #63b3ed;">{int(metrics['risk_score'])}/100</span>
            </div>
            <div style="height: 24px; background: rgba(15, 39, 68, 0.6); border-radius: 12px; overflow: hidden;">
                <div style="height: 100%; width: {metrics['risk_score']}%; background: {risk_color}; border-radius: 12px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction Result
        if "prediction_result" in st.session_state:
            result = st.session_state["prediction_result"]
            status = result.get("loan_status", "Unknown").strip()
            is_approved = status.lower() == "approved"
            factors = get_decision_factors(form_data, metrics)
            
            factors_html = "".join([
                f'<div style="display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; margin-bottom: 8px; font-size: 14px; color: #10b981;"><span>✓</span><span>{f["text"]}</span></div>' if f["positive"] 
                else f'<div style="display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; margin-bottom: 8px; font-size: 14px; color: #ef4444;"><span>!</span><span>{f["text"]}</span></div>'
                for f in factors
            ])
            
            if is_approved:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
                            border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 24px; text-align: center;">
                    <div style="font-size: 32px; font-weight: 700; color: #10b981; margin-bottom: 8px;">✓ APPROVED</div>
                    <div style="font-size: 14px; color: #94a3b8; margin-bottom: 20px;">High confidence prediction</div>
                    <div style="text-align: left;">
                        <div style="font-size: 12px; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase;">Key Decision Factors</div>
                        {factors_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
                            border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 16px; padding: 24px; text-align: center;">
                    <div style="font-size: 32px; font-weight: 700; color: #ef4444; margin-bottom: 8px;">✗ REJECTED</div>
                    <div style="font-size: 14px; color: #94a3b8; margin-bottom: 20px;">Application does not meet criteria</div>
                    <div style="text-align: left;">
                        <div style="font-size: 12px; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase;">Key Decision Factors</div>
                        {factors_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Decision Guidance
        st.markdown("""<div style="background: rgba(26, 54, 93, 0.4); border-radius: 14px; padding: 18px; margin-top: 20px; 
                    border: 1px solid rgba(99, 179, 237, 0.15);">
            <div style="font-size: 14px; font-weight: 600; color: #90cdf4; margin-bottom: 12px;">💡 Decision Guidance</div>""", unsafe_allow_html=True)
        
        if cibil_score < 650:
            st.markdown('<p style="font-size: 13px; color: #94a3b8;">⚠️ <strong style="color: #f97316;">Credit Improvement Needed:</strong> CIBIL scores below 650 significantly reduce approval chances.</p>', unsafe_allow_html=True)
        elif metrics["loan_to_income"] > 3:
            st.markdown('<p style="font-size: 13px; color: #94a3b8;">⚠️ <strong style="color: #eab308;">High Loan-to-Income:</strong> Consider requesting a smaller loan amount.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-size: 13px; color: #94a3b8;">✓ <strong style="color: #10b981;">Strong Application:</strong> This profile shows favorable metrics.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# FEATURES TAB

def render_features_tab():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style="font-size: 36px; font-weight: 700; background: linear-gradient(90deg, #fff 0%, #90cdf4 50%, #63b3ed 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Model Feature Analysis</h1>
        <p style="font-size: 16px; color: #94a3b8; max-width: 600px; margin: 0 auto;">
            Understanding which factors drive loan approval decisions helps banks optimize risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown("""<div style="background: rgba(26, 54, 93, 0.4); border-radius: 16px; border: 1px solid rgba(99, 179, 237, 0.15); 
                padding: 28px; margin-bottom: 32px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%); 
                        border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">📈</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">Feature Importance Rankings</span>
        </div>""", unsafe_allow_html=True)
    
    # Feature importances from actual model training (see training.log)
    features = [
        {"name": "CIBIL Score", "importance": 82.74, "color": "#10b981", "desc": "Credit bureau score (300-900) — dominant predictor"},
        {"name": "Loan Term", "importance": 5.09, "color": "#3182ce", "desc": "Repayment period affecting monthly burden"},
        {"name": "Loan Amount", "importance": 2.87, "color": "#8b5cf6", "desc": "Total principal requested"},
        {"name": "Luxury Assets Value", "importance": 1.71, "color": "#f59e0b", "desc": "Vehicles, jewelry, and other luxury items"},
        {"name": "Annual Income", "importance": 1.70, "color": "#06b6d4", "desc": "Yearly earnings indicating repayment capacity"},
        {"name": "Residential Assets", "importance": 1.59, "color": "#ec4899", "desc": "Value of residential properties owned"},
        {"name": "Commercial Assets", "importance": 1.50, "color": "#f97316", "desc": "Value of commercial properties owned"},
        {"name": "Bank Assets", "importance": 1.41, "color": "#6366f1", "desc": "Liquid savings indicating financial stability"},
        {"name": "Number of Dependents", "importance": 0.74, "color": "#a855f7", "desc": "Financial obligations to family members"},
        {"name": "Employment Status", "importance": 0.65, "color": "#64748b", "desc": "Self-employed vs salaried (combined categories)"}
    ]
    
    # Display dominant feature (CIBIL Score) prominently
    cibil = features[0]
    st.markdown(f"""
    <div style="margin-bottom: 28px; padding: 20px; background: rgba(16, 185, 129, 0.1); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 18px; font-weight: 600; color: #e2e8f0;">🏆 {cibil['name']}</span>
            <span style="font-size: 24px; font-weight: 700; color: #10b981;">{cibil['importance']:.1f}%</span>
        </div>
        <div style="height: 14px; background: rgba(99, 179, 237, 0.2); border-radius: 7px; overflow: hidden;">
            <div style="height: 100%; width: {cibil['importance']}%; background: linear-gradient(90deg, {cibil['color']} 0%, #22c55e 100%); border-radius: 7px;"></div>
        </div>
        <p style="font-size: 13px; color: #94a3b8; margin-top: 10px;">⚡ {cibil['desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display remaining features
    st.markdown('<p style="font-size: 13px; font-weight: 600; color: #90cdf4; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px;">Secondary Features (combined: 17.26%)</p>', unsafe_allow_html=True)
    
    for f in features[1:]:
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 14px; font-weight: 500; color: #e2e8f0;">{f['name']}</span>
                <span style="font-size: 14px; font-weight: 600; color: #63b3ed;">{f['importance']:.2f}%</span>
            </div>
            <div style="height: 8px; background: rgba(99, 179, 237, 0.2); border-radius: 4px; overflow: hidden;">
                <div style="height: 100%; width: {f['importance'] / 5.09 * 100}%; background: {f['color']}; border-radius: 4px;"></div>
            </div>
            <p style="font-size: 11px; color: #64748b; margin-top: 4px;">{f['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add insight callout
    st.markdown("""
    <div style="margin-top: 24px; padding: 16px; background: rgba(251, 191, 36, 0.1); border-left: 3px solid #f59e0b; border-radius: 0 8px 8px 0;">
        <p style="font-size: 13px; color: #fbbf24; font-weight: 600; margin-bottom: 6px;">📊 Key Insight</p>
        <p style="font-size: 13px; color: #94a3b8; margin: 0;">
            CIBIL score alone accounts for <strong style="color: #10b981;">82.74%</strong> of prediction weight. 
            This aligns with banking industry practice where credit history is the primary risk indicator. 
            All other features combined contribute only ~17% to the decision.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Banking Applications
    st.markdown('<h2 style="font-size: 24px; font-weight: 600; margin-bottom: 24px; color: #e2e8f0;">Banking Industry Applications</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    apps = [
        ("⚡", "Instant Pre-Qualification", "#10b981", "Real-time loan eligibility assessment, reducing processing from days to seconds."),
        ("🎯", "Risk Stratification", "#3182ce", "Segment applicants into risk tiers for differentiated pricing and terms."),
        ("📊", "Portfolio Analytics", "#8b5cf6", "Identify at-risk accounts before defaults occur with proactive intervention."),
        ("🔄", "Custom Model Training", "#f59e0b", "Retrain with proprietary data to capture institution-specific patterns."),
        ("⚖️", "Regulatory Compliance", "#ec4899", "Explainable features provide audit trails for lending decisions."),
        ("🔗", "API Integration", "#06b6d4", "RESTful API enables seamless integration with existing systems.")
    ]
    
    for i, (icon, title, color, desc) in enumerate(apps):
        with [col1, col2, col3][i % 3]:
            st.markdown(f"""
            <div style="background: rgba(26, 54, 93, 0.4); border-radius: 16px; border: 1px solid rgba(99, 179, 237, 0.15); 
                        padding: 24px; margin-bottom: 20px; min-height: 200px;">
                <div style="width: 52px; height: 52px; border-radius: 12px; background: linear-gradient(135deg, {color} 0%, {color}88 100%);
                            display: flex; align-items: center; justify-content: center; font-size: 24px; margin-bottom: 16px;">{icon}</div>
                <h3 style="font-size: 18px; font-weight: 600; margin-bottom: 12px; color: #e2e8f0;">{title}</h3>
                <p style="font-size: 14px; color: #94a3b8; line-height: 1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# DOCUMENTATION TAB

def render_documentation_tab():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style="font-size: 36px; font-weight: 700; background: linear-gradient(90deg, #fff 0%, #90cdf4 50%, #63b3ed 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Technical Documentation</h1>
        <p style="font-size: 16px; color: #94a3b8;">Implementation guide for production deployment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture
    st.markdown("""<div style="background: rgba(26, 54, 93, 0.4); border-radius: 16px; border: 1px solid rgba(99, 179, 237, 0.15); 
                padding: 28px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); 
                        border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">🏗️</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">System Architecture</span>
        </div>""", unsafe_allow_html=True)
    
    st.code("""
┌────────────────────────────────────────────────────────────────┐
│                      LOAN APPROVAL SYSTEM                      │
├────────────────────────────────────────────────────────────────┤
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   Frontend   │───▶│   FastAPI    │───▶│   ML Model   │     │
│   │  (Streamlit) │    │   Server     │    │  (Pipeline)  │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                              │                    │            │
│                              ▼                    ▼            │
│                       ┌──────────────┐    ┌──────────────┐     │
│                       │   Request    │    │   Joblib     │     │
│                       │  Validation  │    │   Artifacts  │     │
│                       └──────────────┘    └──────────────┘     │
└────────────────────────────────────────────────────────────────┘
    """, language="text")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Methodology Section
    st.markdown("""
    <div style="background: rgba(26, 54, 93, 0.4); border-radius: 16px; border: 1px solid rgba(99, 179, 237, 0.15); 
                padding: 28px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%); 
                        border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">🔬</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">Methodology</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    meth_col1, meth_col2 = st.columns(2)
    with meth_col1:
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
            <h4 style="color: #90cdf4; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Model Selection</h4>
            <p style="font-size: 14px; color: #e2e8f0; line-height: 1.7; margin: 0;">
                <strong>Random Forest Classifier</strong> selected for its interpretability, robustness to outliers, 
                and native feature importance ranking — critical for regulatory compliance in financial services.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px;">
            <h4 style="color: #90cdf4; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Validation Strategy</h4>
            <p style="font-size: 14px; color: #e2e8f0; line-height: 1.7; margin: 0;">
                <strong>80/20 stratified split</strong> preserving class distribution, supplemented by 
                <strong>5-fold cross-validation</strong> to ensure generalization and mitigate overfitting.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with meth_col2:
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
            <h4 style="color: #90cdf4; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Preprocessing Pipeline</h4>
            <p style="font-size: 14px; color: #e2e8f0; line-height: 1.7; margin: 0;">
                <strong>ColumnTransformer</strong> with dual pathways: StandardScaler + mean imputation for numeric features; 
                OneHotEncoder + mode imputation for categorical variables.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px;">
            <h4 style="color: #90cdf4; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Class Imbalance Handling</h4>
            <p style="font-size: 14px; color: #e2e8f0; line-height: 1.7; margin: 0;">
                <strong>Balanced class weights</strong> automatically adjust for approval/rejection ratio imbalance, 
                preventing bias toward the majority class in predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 16px; margin-top: 16px; margin-bottom: 24px;">
        <h4 style="color: #90cdf4; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Hyperparameters</h4>
        <code style="font-size: 13px; color: #e2e8f0;">
            n_estimators=100 | max_depth=15 | min_samples_split=2 | class_weight='balanced' | random_state=42 | cv_folds=5
        </code>
    </div>
    """, unsafe_allow_html=True)
    
    # Training Data Section
    st.markdown("""
    <div style="background: rgba(26, 54, 93, 0.4); border-radius: 16px; border: 1px solid rgba(99, 179, 237, 0.15); 
                padding: 28px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #06b6d4 0%, #22d3ee 100%); 
                        border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">📊</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">Training Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    data_col1, data_col2 = st.columns([1, 2])
    with data_col1:
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 16px; text-align: center;">
            <div style="font-size: 32px; font-weight: 700; color: #63b3ed;">4,269</div>
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 16px; text-align: center;">
            <div style="font-size: 32px; font-weight: 700; color: #10b981;">62.1%</div>
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Approved</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 32px; font-weight: 700; color: #ef4444;">37.9%</div>
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Rejected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with data_col2:
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px;">
            <h4 style="color: #90cdf4; font-size: 14px; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.5px;">Feature Schema (11 Input Variables)</h4>
            <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.2);">
                        <th style="text-align: left; padding: 8px; color: #94a3b8;">Feature</th>
                        <th style="text-align: left; padding: 8px; color: #94a3b8;">Type</th>
                        <th style="text-align: left; padding: 8px; color: #94a3b8;">Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">no_of_dependents</td>
                        <td style="padding: 8px; color: #8b5cf6;">int</td>
                        <td style="padding: 8px; color: #94a3b8;">Number of financial dependents</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">education</td>
                        <td style="padding: 8px; color: #f59e0b;">categorical</td>
                        <td style="padding: 8px; color: #94a3b8;">Graduate / Not Graduate</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">self_employed</td>
                        <td style="padding: 8px; color: #f59e0b;">categorical</td>
                        <td style="padding: 8px; color: #94a3b8;">Yes / No</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">income_annum</td>
                        <td style="padding: 8px; color: #8b5cf6;">float</td>
                        <td style="padding: 8px; color: #94a3b8;">Annual income (USD)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">loan_amount</td>
                        <td style="padding: 8px; color: #8b5cf6;">float</td>
                        <td style="padding: 8px; color: #94a3b8;">Requested loan principal</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">loan_term</td>
                        <td style="padding: 8px; color: #8b5cf6;">int</td>
                        <td style="padding: 8px; color: #94a3b8;">Repayment period (months)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">cibil_score</td>
                        <td style="padding: 8px; color: #8b5cf6;">int</td>
                        <td style="padding: 8px; color: #94a3b8;">Credit bureau score (300-900)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">residential_assets_value</td>
                        <td style="padding: 8px; color: #8b5cf6;">float</td>
                        <td style="padding: 8px; color: #94a3b8;">Residential property value</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">commercial_assets_value</td>
                        <td style="padding: 8px; color: #8b5cf6;">float</td>
                        <td style="padding: 8px; color: #94a3b8;">Commercial property value</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">luxury_assets_value</td>
                        <td style="padding: 8px; color: #8b5cf6;">float</td>
                        <td style="padding: 8px; color: #94a3b8;">Vehicles, jewelry, etc.</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; color: #e2e8f0; font-family: monospace;">bank_asset_value</td>
                        <td style="padding: 8px; color: #8b5cf6;">float</td>
                        <td style="padding: 8px; color: #94a3b8;">Liquid bank holdings</td>
                    </tr>
                </tbody>
            </table>
            <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(99, 179, 237, 0.2);">
                <span style="font-size: 12px; color: #94a3b8;">Target Variable: </span>
                <code style="color: #10b981; padding: 2px 8px; border-radius: 4px;">loan_status</code>
                <span style="font-size: 12px; color: #94a3b8;"> → Approved / Rejected</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Deployment Architecture - Using simpler HTML that Streamlit renders correctly
    st.markdown("""
    <div style="background: rgba(26, 54, 93, 0.4); border-radius: 16px; border: 1px solid rgba(99, 179, 237, 0.15); padding: 28px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #10b981 0%, #22c55e 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">🚀</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">Deployment Architecture</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Current Mode Box
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 16px; border-radius: 10px; margin-bottom: 16px;">
        <p style="font-size: 13px; color: #10b981; font-weight: 600; margin-bottom: 6px;">✓ CURRENT: Standalone Mode</p>
        <p style="font-size: 13px; color: #94a3b8; margin: 0;">
            This app runs the ML model directly within Streamlit using cached model loading. 
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Optional API Mode Box
    st.markdown("""
    <div style="background: rgba(99, 179, 237, 0.1); border: 1px solid rgba(99, 179, 237, 0.2); padding: 16px; border-radius: 10px; margin-bottom: 24px;">
        <p style="font-size: 13px; color: #63b3ed; font-weight: 600; margin-bottom: 10px;">⚡ OPTIONAL: REST API Mode</p>
        <p style="font-size: 13px; color: #94a3b8; margin-bottom: 12px;">
            For enterprise integration, mobile apps, or microservices architecture, 
            deploy the included FastAPI server (api_server.py) separately.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Endpoints using Streamlit columns for cleaner rendering
    ep_col1, ep_col2, ep_col3 = st.columns(3)
    with ep_col1:
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.6); padding: 16px; border-radius: 8px; text-align: center;">
            <span style="background: #22c55e; padding: 4px 12px; border-radius: 5px; font-size: 11px; font-weight: 700; color: white;">GET</span>
            <p style="color: #90cdf4; font-family: monospace; margin: 10px 0 5px 0;">/</p>
            <p style="color: #64748b; font-size: 11px; margin: 0;">API welcome</p>
        </div>
        """, unsafe_allow_html=True)
    with ep_col2:
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.6); padding: 16px; border-radius: 8px; text-align: center;">
            <span style="background: #22c55e; padding: 4px 12px; border-radius: 5px; font-size: 11px; font-weight: 700; color: white;">GET</span>
            <p style="color: #90cdf4; font-family: monospace; margin: 10px 0 5px 0;">/health</p>
            <p style="color: #64748b; font-size: 11px; margin: 0;">Health check</p>
        </div>
        """, unsafe_allow_html=True)
    with ep_col3:
        st.markdown("""
        <div style="background: rgba(15, 39, 68, 0.6); padding: 16px; border-radius: 8px; text-align: center;">
            <span style="background: #3182ce; padding: 4px 12px; border-radius: 5px; font-size: 11px; font-weight: 700; color: white;">POST</span>
            <p style="color: #90cdf4; font-family: monospace; margin: 10px 0 5px 0;">/predict</p>
            <p style="color: #64748b; font-size: 11px; margin: 0;">Get prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="font-size: 12px; color: #64748b; text-align: center; margin-top: 12px;">
        Deploy API on Railway, Render, or AWS for production use cases requiring HTTP access.
    </p>
    """, unsafe_allow_html=True)
    
    # Model Performance
    st.markdown("""<div style="background: rgba(26, 54, 93, 0.4); border-radius: 16px; border: 1px solid rgba(99, 179, 237, 0.15); 
                padding: 28px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #ec4899 0%, #f472b6 100%); 
                        border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">📋</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">Model Performance Metrics</span>
        </div>""", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    for col, (val, label, clr) in zip([m1, m2, m3, m4], [
        ("98%", "Accuracy", "#10b981"), ("98%", "Precision", "#3182ce"), 
        ("98%", "Recall", "#8b5cf6"), ("0.98", "F1-Score", "#f59e0b")
    ]):
        with col:
            st.markdown(f"""
            <div style="background: {clr}15; border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 32px; font-weight: 700; color: {clr};">{val}</div>
                <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-top: 4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Classification Report Table
    st.markdown("""
    <div style="background: rgba(15, 39, 68, 0.6); border-radius: 12px; padding: 20px; margin-top: 20px;">
        <div style="font-size: 14px; font-weight: 600; color: #90cdf4; margin-bottom: 16px;">Classification Report</div>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <thead>
                <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.2);">
                    <th style="text-align: left; padding: 12px; color: #94a3b8;">Class</th>
                    <th style="text-align: center; padding: 12px; color: #94a3b8;">Precision</th>
                    <th style="text-align: center; padding: 12px; color: #94a3b8;">Recall</th>
                    <th style="text-align: center; padding: 12px; color: #94a3b8;">F1-Score</th>
                    <th style="text-align: center; padding: 12px; color: #94a3b8;">Support</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                    <td style="padding: 12px; color: #10b981; font-weight: 600;">✓ Approved</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.98</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.99</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.99</td>
                    <td style="text-align: center; padding: 12px; color: #94a3b8;">531</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                    <td style="padding: 12px; color: #ef4444; font-weight: 600;">✗ Rejected</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.99</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.97</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.98</td>
                    <td style="text-align: center; padding: 12px; color: #94a3b8;">323</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.2);">
                    <td style="padding: 12px; color: #63b3ed; font-weight: 600;">Accuracy</td>
                    <td style="text-align: center; padding: 12px;" colspan="2"></td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0; font-weight: 700;">0.98</td>
                    <td style="text-align: center; padding: 12px; color: #94a3b8;">854</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(99, 179, 237, 0.1);">
                    <td style="padding: 12px; color: #94a3b8;">Macro Avg</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.98</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.98</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.98</td>
                    <td style="text-align: center; padding: 12px; color: #94a3b8;">854</td>
                </tr>
                <tr>
                    <td style="padding: 12px; color: #94a3b8;">Weighted Avg</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.98</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.98</td>
                    <td style="text-align: center; padding: 12px; color: #e2e8f0;">0.98</td>
                    <td style="text-align: center; padding: 12px; color: #94a3b8;">854</td>
                </tr>
            </tbody>
        </table>
    </div>
    <p style="font-size: 13px; color: #64748b; margin-top: 16px; text-align: center;">
        Evaluated on 20% holdout test set (854 samples) with stratified sampling
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom Training
    st.markdown("""<div style="background: rgba(26, 54, 93, 0.4); border-radius: 16px; border: 1px solid rgba(99, 179, 237, 0.15); 
                padding: 28px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #06b6d4 0%, #22d3ee 100%); 
                        border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">🔧</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">Custom Training Instructions</span>
        </div>
        <p style="font-size: 14px; color: #94a3b8; margin-bottom: 20px;">Banks can train the model on their proprietary data:</p>
        <ol style="padding-left: 20px; color: #e2e8f0; line-height: 2;">
            <li><strong>Prepare Dataset:</strong> Format historical loan data with the 11 required features plus loan_status.</li>
            <li><strong>Data Validation:</strong> Ensure no missing values in critical fields.</li>
            <li><strong>Run Training:</strong> Execute <code>python train_model.py</code> with your dataset.</li>
            <li><strong>Evaluate:</strong> Review classification report metrics.</li>
            <li><strong>Deploy:</strong> Replace <code>loan_model.joblib</code> and <code>loan_columns.joblib</code>.</li>
        </ol>
    </div>""", unsafe_allow_html=True)

    # Data Disclosure & Limitations Section
    st.markdown("""
    <div style="background: rgba(26, 54, 93, 0.4); border-radius: 16px; border: 1px solid rgba(99, 179, 237, 0.15); 
                padding: 28px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); 
                        border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;">⚠️</div>
            <span style="font-size: 20px; font-weight: 600; color: #e2e8f0;">Data Disclosure & Limitations</span>
        </div>
        
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
            <h4 style="color: #90cdf4; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Data Source</h4>
            <p style="font-size: 14px; color: #e2e8f0; line-height: 1.7; margin: 0;">
                This project utilizes <strong>CIBIL scores</strong> and data records based on publicly available 
                information from <strong>India</strong>.
            </p>
        </div>
        
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
            <h4 style="color: #90cdf4; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Demonstration Only</h4>
            <p style="font-size: 14px; color: #e2e8f0; line-height: 1.7; margin: 0;">
                This system is intended for <strong>demonstration and portfolio purposes</strong> and is not suitable 
                for actual financial deployment without further rigorous testing.
            </p>
        </div>
        
        <div style="background: rgba(15, 39, 68, 0.4); border-radius: 12px; padding: 20px;">
            <h4 style="color: #90cdf4; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Expansion Potential</h4>
            <p style="font-size: 14px; color: #e2e8f0; line-height: 1.7; margin: 0;">
                While current results are strong, the model's insights would improve significantly with access to 
                <strong>larger, proprietary banking datasets</strong> containing more diverse features such as 
                <strong>transaction history</strong> or <strong>credit utilization ratios</strong>.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# MAIN

def main():
    apply_custom_css()
    render_header()
    
    tab1, tab2, tab3 = st.tabs(["Analysis", "Features", "Documentation"])
    
    with tab1:
        render_predictor_tab()
    with tab2:
        render_features_tab()
    with tab3:
        render_documentation_tab()
    
    st.markdown("""
    <div style="text-align: center; margin-top: 60px; padding: 30px; border-top: 1px solid rgba(99, 179, 237, 0.2);">
        <p style="font-size: 14px; color: #64748b;">
            © 2025 Samuel Villarreal. All Rights Reserved.<br>
            <a href="https://github.com/samuelvy1100/Loan-Approval-Prediction-Pipeline" target="_blank" style="color: #63b3ed; text-decoration: none;">
                View Project on GitHub
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()