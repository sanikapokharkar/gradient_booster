import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Gradient Boosting Predictor",
    page_icon="🤖",
    layout="centered"
)

# Load the Trained Model
@st.cache_resource
def load_model():
    return joblib.load("greadientbooster.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'greadientbooster.pkl': {e}")
    st.stop()

# Custom CSS with Shadow Effects & Styling
st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Main Container Card */
    .main-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1), 0 4px 10px rgba(0, 0, 0, 0.06);
        margin-bottom: 25px;
    }

    /* Title Styling */
    .title-text {
        text-align: center;
        color: #1e293b;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle-text {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    /* Input Field Labels */
    label {
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* Custom Prediction Card Effect */
    .result-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.15);
        border: 1px solid #e2e8f0;
        margin-top: 20px;
    }

    /* Custom Glassmorphism Button Shadow */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.39);
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.54);
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    }

    div.stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.39);
    }
    </style>
""", unsafe_allow_html=True)

# Main UI Structure
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title-text">Model Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Enter input features below to get real-time class predictions.</div>', unsafe_allow_html=True)

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

    with col2:
        review = st.selectbox("Review", options=["Poor", "Average", "Good"])
        education = st.selectbox("Education", options=["School", "UG", "PG"])

    submit_button = st.form_submit_button(label="Predict Class")

st.markdown('</div>', unsafe_allow_html=True)

# Processing and Prediction
if submit_button:
    # 1. Structure the input data into a DataFrame matching expected feature names:
    # ['age', 'gender', 'review', 'education']
    raw_data = {
        'age': [age],
        'gender': [gender],
        'review': [review],
        'education': [education]
    }
    input_df = pd.DataFrame(raw_data)

    # 2. Convert categorical columns to 'category' dtype as requested
    categorical_cols = ['gender', 'review', 'education']
    for col in categorical_cols:
        input_df[col] = input_df[col].astype('category')

    try:
        # Generate Prediction
        prediction = model.predict(input_df)[0]
        
        # Display Results in a Styled Shadow Container
        st.markdown(f"""
            <div class="result-card">
                <h3 style="color: #64748b; margin-bottom: 5px; font-weight: 500;">Prediction Output</h3>
                <h1 style="color: #2563eb; margin: 0; font-size: 2.5rem;">{prediction}</h1>
            </div>
        """, unsafe_allow_html=True)

    except Exception as err:
        st.error(f"Prediction failed. Ensure categorical features match the exact training encoding. Error: {err}")
