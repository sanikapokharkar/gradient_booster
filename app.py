import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Model Predictor", page_icon="🤖", layout="centered"
)

# Custom CSS for UI styling, custom cards, and soft shadow effects
st.markdown(
    """
    <style>
    /* Main Background Accent */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Container Card with Elevation/Shadow Effects */
    .css-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }

    /* Input Card Container */
    div[data-testid="stVerticalBlock"] > div:has(div.input-anchor) {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #edf2f7;
    }

    /* Custom Header Styling */
    .main-title {
        color: #1e293b;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        color: #64748b;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Prediction Output Box */
    .result-card {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.3);
        margin-top: 1.5rem;
    }

    /* Styled Buttons */
    .stButton>button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #4338ca;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
        transform: translateY(-1px);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained GradientBoostingClassifier model
@st.cache_resource
def load_model():
    with open("GreadientBooster.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# Header Section
st.markdown(
    "<h1 class='main-title'>Gradient Boosting Classifier</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-title'>Provide feature inputs below to generate predictions</p>",
    unsafe_allow_html=True,
)

# Input Section Inside Styled Container
st.markdown("<div class='input-anchor'></div>", unsafe_allow_html=True)
st.subheader("Input Features")

# Creating form inputs matching model schema: age, gender, review, education
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age", min_value=1, max_value=120, value=30, step=1, help="Enter age"
    )

    # Categorical column as 'category' type/select box
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female", "Other"],
        help="Select gender category",
    )

with col2:
    review = st.selectbox(
        "Review Rating",
        options=["Poor", "Average", "Good"],
        help="Select review level",
    )

    education = st.selectbox(
        "Education Level",
        options=["School", "UG", "PG"],
        help="Select highest education degree",
    )

# Encoding options if raw strings need mapping to numeric formats
# (Adjust target values below to match your original training encoding if applicable)
gender_map = {"Male": 1, "Female": 0, "Other": 2}
review_map = {"Poor": 0, "Average": 1, "Good": 2}
education_map = {"School": 0, "UG": 1, "PG": 2}

# Prepare DataFrame ensuring categorical dtypes are explicit
input_df = pd.DataFrame(
    [
        {
            "age": age,
            "gender": gender_map[gender],
            "review": review_map[review],
            "education": education_map[education],
        }
    ]
)

# Explicitly cast categorical columns to pandas 'category' dtype
categorical_cols = ["gender", "review", "education"]
for col in categorical_cols:
    input_df[col] = input_df[col].astype("category")

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Logic
if st.button("Predict Outcome"):
    try:
        prediction = model.predict(input_df)[0]
        probabilities = (
            model.predict_proba(input_df)[0]
            if hasattr(model, "predict_proba")
            else None
        )

        st.markdown(
            f"""
            <div class='result-card'>
                <h3 style='margin:0;'>Prediction Output</h3>
                <h1 style='margin:0.5rem 0; font-size: 2.5rem;'>Class {prediction}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if probabilities is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.write("**Class Probabilities:**")
            prob_df = pd.DataFrame(
                [probabilities],
                columns=[f"Class {c}" for c in model.classes_],
            )
            st.dataframe(
                prob_df.style.highlight_max(axis=1, color="#e0e7ff"),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Error executing prediction: {e}")
