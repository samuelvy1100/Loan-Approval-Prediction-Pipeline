import streamlit as st
import requests

EDUCATION_OPTIONS = {
    "Graduate": " Graduate",
    "Not Graduate": " Not Graduate"
}
SELF_EMPLOYED_OPTIONS = {
    "Yes": " Yes",
    "No": " No"
}

API_URL = "http://127.0.0.1:8000/predict"


def get_user_inputs():
    st.title("🔮 Loan Approval Prediction System")

    st.markdown(
        "This application evaluates an individual's loan application by analyzing personal and financial details "
        "to determine loan approval eligibility. It uses a machine learning model trained on over 4,000 real-world "
        "banking records from India, achieving 98% predictive accuracy. One of the most important features is the "
        "applicant’s CIBIL score (300 = poor creditworthiness, 900 = excellent creditworthiness)."
    )
    st.markdown("Complete the fields below and click **Predict** to see the result.")

    # — Personal Information —
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=5,
            step=1,
            format="%d",
            help="Total count of people financially dependent on the applicant (max 5)."
        )
        education = st.selectbox(
            "Education Level",
            options=list(EDUCATION_OPTIONS.keys()),
            help="Select 'Graduate' if the applicant has a master’s degree or higher."
        )
        self_employed = st.selectbox(
            "Self-Employed?",
            options=list(SELF_EMPLOYED_OPTIONS.keys()),
            help="Select 'Yes' if the applicant is self-employed."
        )

    # — Financial Information —
    st.subheader("Financial Information")
    col3, col4 = st.columns(2)
    with col3:
        income_annum = st.number_input(
            "Annual Income (USD)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            help="Applicant’s total annual income in U.S. dollars."
        )
        loan_amount = st.number_input(
            "Loan Amount Requested (USD)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            help="Total loan amount the applicant is requesting (USD)."
        )
        loan_term = st.number_input(
            "Loan Term (Months)",
            min_value=1.0,
            step=1.0,
            format="%.0f",
            help="Duration of the loan in months."
        )
        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=300.0,
            max_value=900.0,
            step=1.0,
            format="%.0f",
            help="Credit score between 300 (poor) and 900 (excellent)."
        )

    # — Asset Values —
    st.subheader("Asset Values (USD)")
    col5, col6 = st.columns(2)
    with col5:
        residential_assets_value = st.number_input(
            "Residential Assets Value (USD)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            help="Total estimated value of residential real estate owned."
        )
        commercial_assets_value = st.number_input(
            "Commercial Assets Value (USD)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            help="Total estimated value of commercial properties owned."
        )
    with col6:
        luxury_assets_value = st.number_input(
            "Luxury Assets Value (USD)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            help="Total estimated value of luxury items (e.g., vehicles, jewelry)."
        )
        bank_asset_value = st.number_input(
            "Bank Assets (USD)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            help="Current balance across all bank accounts (USD)."
        )

    return {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }


def validate_inputs(inputs):
    """
    Returns a list of error messages if any inputs are invalid.
    """
    errors = []
    if inputs["income_annum"] <= 0:
        errors.append("• Annual income must be greater than 0.")
    if inputs["loan_amount"] <= 0:
        errors.append("• Loan amount requested must be greater than 0.")
    if inputs["loan_term"] < 1:
        errors.append("• Loan term must be at least 1 month.")
    if not (300 <= inputs["cibil_score"] <= 900):
        errors.append("• CIBIL score must be between 300 and 900.")
    return errors


def call_api(inputs):
    """
    Sends a POST request to the prediction API.
    Maps the English dropdown values to the leading-space strings
    the model pipeline expects.
    Returns (parsed_json, error_message). If error_message is not None, display it.
    """
    payload = {
        "no_of_dependents": inputs["no_of_dependents"],
        "education": EDUCATION_OPTIONS[inputs["education"]],
        "self_employed": SELF_EMPLOYED_OPTIONS[inputs["self_employed"]],
        "income_annum": inputs["income_annum"],
        "loan_amount": inputs["loan_amount"],
        "loan_term": inputs["loan_term"],
        "cibil_score": inputs["cibil_score"],
        "residential_assets_value": inputs["residential_assets_value"],
        "commercial_assets_value": inputs["commercial_assets_value"],
        "luxury_assets_value": inputs["luxury_assets_value"],
        "bank_asset_value": inputs["bank_asset_value"]
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again later."
    except requests.exceptions.ConnectionError:
        return None, "Unable to connect to the server. Verify that the API is running."
    except requests.exceptions.HTTPError as http_err:
        return None, f"Server returned an error: {http_err}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"


def main():
    user_inputs = get_user_inputs()

    if st.button("Predict"):
        validation_errors = validate_inputs(user_inputs)
        if validation_errors:
            st.error("Please fix the following before submitting:")
            for err in validation_errors:
                st.write(err)
        else:
            with st.spinner("Performing prediction..."):
                result, error_msg = call_api(user_inputs)

            if error_msg:
                st.error(error_msg)
            else:
                loan_status = result.get("loan_status", "No result returned")
                st.success(f"Loan Status: **{loan_status}**")


if __name__ == "__main__":
    main()
