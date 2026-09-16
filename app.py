import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Gradient Booster Classifier",
    page_icon="🚀",
    layout="centered"
)

# Custom CSS for UI styling & drop-shadow effects
custom_css = """
<style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Main Container Box with Shadow */
    .main-card {
        background-color: #ffffff;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08), 0 4px 10px rgba(0, 0, 0, 0.03);
        margin-bottom: 2rem;
    }
    
    /* Header Styling */
    .title-text {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Style Streamlit Buttons with Soft Shadows */
    div.stButton > button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
        transition: all 0.2s ease-in-out;
    }
    
    div.stButton > button:hover {
        background-color: #4338ca;
        color: white;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45);
        transform: translateY(-1px);
    }

    /* Prediction Result Box with Shadow */
    .result-card {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.3);
        margin-top: 1.5rem;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Function to load model
@st.cache_resource
def load_model():
    with open("GreadientBooster.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: `GreadientBooster.pkl` file not found in the working directory.")
    st.stop()

# Layout Container
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<h1 class="title-text">Classification Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Enter feature details to evaluate model output</p>', unsafe_allow_html=True)

# Form fields based on model features: ['age', 'gender', 'review', 'education']
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    
    with col2:
        review = st.selectbox("Review", options=["Poor", "Average", "Good"])
        education = st.selectbox("Education", options=["School", "UG", "PG"])
        
    submit_button = st.form_submit_button("Run Prediction")

st.markdown('</div>', unsafe_allow_html=True)

# Processing Inputs
if submit_button:
    # Ensure variables match categorical types expected by preprocessing/model pipelines
    input_data = pd.DataFrame([{
        'age': age,
        'gender': pd.Categorical([gender], categories=["Male", "Female", "Other"])[0],
        'review': pd.Categorical([review], categories=["Poor", "Average", "Good"])[0],
        'education': pd.Categorical([education], categories=["School", "UG", "PG"])[0]
    }])
    
    try:
        prediction = model.predict(input_data)[0]
        
        # Display Prediction Result
        st.markdown(
            f'''
            <div class="result-card">
                <h3>Predicted Output: Class {prediction}</h3>
            </div>
            ''', 
            unsafe_allow_html=True
        )
    except Exception as e:
        # Fallback handling for array formatting
        try:
            raw_input = np.array([[age, gender, review, education]], dtype=object)
            prediction = model.predict(raw_input)[0]
            st.markdown(
                f'''
                <div class="result-card">
                    <h3>Predicted Output: Class {prediction}</h3>
                </div>
                ''', 
                unsafe_allow_html=True
            )
        except Exception as err:
            st.error(f"Prediction Error: {str(err)}")
