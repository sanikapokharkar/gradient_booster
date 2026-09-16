from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Gradient Boosting Predictor",
    page_icon="🤖",
    layout="centered"
)

# Resolve model path dynamically relative to app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "greadientbooster.pkl"

# Load Trained Model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at path: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Custom UI Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }
    .title-text {
        text-align: center;
        color: #1e293b;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .result-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.15);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title-text">Model Prediction Dashboard</div>', unsafe_allow_html=True)

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

# Generate Predictions
if submit_button:
    raw_data = {
        'age': [age],
        'gender': [gender],
        'review': [review],
        'education': [education]
    }
    input_df = pd.DataFrame(raw_data)

    # Convert categorical inputs to 'category' dtype
    categorical_cols = ['gender', 'review', 'education']
    for col in categorical_cols:
        input_df[col] = input_df[col].astype('category')

    try:
        prediction = model.predict(input_df)[0]
        
        st.markdown(f"""
            <div class="result-card">
                <h3 style="color: #64748b; margin-bottom: 5px;">Prediction Output</h3>
                <h1 style="color: #2563eb; margin: 0; font-size: 2.5rem;">{prediction}</h1>
            </div>
        """, unsafe_allow_html=True)

    except Exception as err:
        st.error(f"Prediction failed: {err}")
