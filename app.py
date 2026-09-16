import streamlit as st
import pickle
import numpy as np
import os
import time
from sklearn.ensemble import GradientBoostingClassifier

# Page Configuration
st.set_page_config(
    page_title="Gradient Boosting Intelligence Portal", 
    page_icon="⚡", 
    layout="centered"
)

# High-End Custom CSS with Glow and Pulse Effects
st.markdown("""
    <style>
        /* Animated Multi-color Background */
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Glassmorphism Main Container */
        .block-container {
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 40px !important;
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.25), inset 0 0 0 1px rgba(255, 255, 255, 0.5);
            margin-top: 5vh;
            margin-bottom: 5vh;
        }

        /* Typography */
        h1 {
            color: #1a202c;
            text-align: center;
            font-weight: 800;
            letter-spacing: -1px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        p.subtitle {
            text-align: center;
            color: #4a5568;
            font-size: 16px;
            margin-bottom: 30px;
            font-weight: 500;
        }

        /* Glowing Input Fields on Hover/Focus */
        .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            border-radius: 12px !important;
            border: 1px solid #cbd5e0 !important;
            transition: all 0.3s ease !important;
        }
        .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:hover {
            border-color: #667eea !important;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.3) !important;
        }

        /* Pulsing Gradient Button */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 700;
            border-radius: 14px;
            padding: 14px;
            border: none;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #5a6fe0 0%, #684092 100%);
        }

        /* Gorgeous Success Card */
        .success-card {
            background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%);
            border: 2px solid #68d391;
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0 15px 35px rgba(72, 187, 120, 0.2);
            text-align: center;
            margin-top: 25px;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        @keyframes popIn {
            0% { transform: scale(0.9); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        /* Debug Box */
        .debug-box {
            background: #edf2f7;
            border-left: 4px solid #4a5568;
            padding: 12px;
            font-family: monospace;
            font-size: 13px;
            border-radius: 6px;
            margin-top: 20px;
            color: #2d3748;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1>⚡ Gradient Boosting Portal</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Binary Classification Inference Engine</p>", unsafe_allow_html=True)

# Robust Model Loader
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'gradientboosting.pkl')
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f), True  # Real Model Loaded
    else:
        # Fallback binary model if pkl is missing
        X_dummy = np.array([[25, 0, 2, 1], [40, 1, 0, 2]])
        y_dummy = np.array([0, 1])
        model = GradientBoostingClassifier()
        model.fit(X_dummy, y_dummy)
        return model, False  # Fallback Mode

model, is_real_model = load_model()

# Notification Banner
if is_real_model:
    st.success("✨ Connected successfully to **`gradientboosting.pkl`**")
else:
    st.warning("⚠️ Running in **Fallback Mode**. Upload your `gradientboosting.pkl` file to GitHub to use your trained model.")

# Interactive Input Form Layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, help="Enter your age")
        review = st.selectbox("Review", ["Positive", "Neutral", "Negative"])
        
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🔮 Run Prediction Model")

if submit_button:
    with st.spinner("Analyzing data patterns with Gradient Boosting..."):
        time.sleep(0.5)
        
        try:
            # Numeric encoding mappings matching typical training sets
            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            review_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
            education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
            
            g_val = gender_map.get(gender, 0)
            r_val = review_map.get(review, 1)
            e_val = education_map.get(education, 1)
            
            # Construct strict numeric array for prediction
            features = np.array([[float(age), float(g_val), float(r_val), float(e_val)]])
            
            # Predict binary outcome (0 or 1)
            raw_prediction = model.predict(features)[0]
            
            # Format 0/1 or Yes/No nicely for user readability
            if str(raw_prediction) in ["1", "True", "Yes"]:
                formatted_result = "Yes (Class 1)"
            else:
                formatted_result = "No (Class 0)"
            
            # Trigger Celebration Balloons 🎉
            st.balloons()
            
            # Display Styled Result Card
            st.markdown(f"""
                <div class="success-card">
                    <h4 style="color: #276749; margin-bottom: 5px; font-weight: 600;">Prediction Outcome</h4>
                    <h2 style="color: #22543d; font-size: 32px; font-weight: 800;">{formatted_result}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Live Debug Info Viewer
            st.markdown(f"""
                <div class="debug-box">
                    <b>🔍 Numeric Array Sent to Model:</b> {features.tolist()} | <b>Raw Output:</b> {raw_prediction}
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
