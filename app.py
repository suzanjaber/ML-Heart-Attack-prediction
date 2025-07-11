
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Guardian",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main background color */
    .main {
        background-color: #f0f2f6;
    }
    /* Title style */
    .st-emotion-cache-10trblm {
        color: #004d40; /* Dark teal */
        font-family: 'Arial', sans-serif;
        font-size: 2.5em;
        font-weight: bold;
    }
    /* Sidebar style */
    .st-emotion-cache-16txtl3 {
        background-color: #e0f2f1; /* Light teal */
    }
    /* Button style */
    .stButton>button {
        background-color: #00796b; /* Teal */
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        font-size: 1.1em;
        font-weight: bold;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #004d40; /* Darker teal */
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    /* Slider style */
    .stSlider {
        color: #00796b; /* Teal */
    }
    /* Selectbox style */
    .stSelectbox {
        color: #00796b; /* Teal */
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    df = pd.read_csv("heart (1).csv")
    df.columns = ['age', 'gender', 'cp', 'trtbps', 'chol', 'fbs', 'rest_ecg',
                  'thalach', 'exng', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    # Data Cleaning
    df["ca"] = df["ca"].replace(4, 0)
    df["thal"] = df["thal"].replace(0, 2)
    df.drop([272], axis=0, inplace=True)
    df.drop(["chol", "fbs", "rest_ecg"], axis=1, inplace=True)

    # Feature Transformation
    df["oldpeak"] = np.sqrt(df["oldpeak"])
    df = pd.get_dummies(df, columns=["gender", "cp", "exng", "slope", "ca", "thal"], drop_first=True)

    # Scaling
    scaler = RobustScaler()
    df[["age", "thalach", "trtbps", "oldpeak"]] = scaler.fit_transform(df[["age", "thalach", "trtbps", "oldpeak"]])

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y, scaler, list(X.columns)

X, y, scaler, feature_names = load_data()

# --- Model Training ---
@st.cache_resource
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(random_state=42).fit(X, y),
        "Decision Tree": DecisionTreeClassifier(random_state=42).fit(X, y),
        "Random Forest": RandomForestClassifier(random_state=42).fit(X, y),
        "Support Vector Machine": SVC(probability=True, random_state=42).fit(X, y)
    }
    return models

models = train_models()

# --- UI Layout ---
st.title("❤️ Heart Guardian: Risk Predictor")
st.markdown("### *Your personal health companion for predicting heart attack risk.*")
st.markdown("---")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("🩺 Patient Information")
    st.markdown("Please provide the patient's details below.")

    # --- Input Form ---
    age = st.slider("Age", 20, 90, 45, help="Patient's age in years.")
    gender = st.radio("Gender", ["Female", "Male"], help="Patient's gender.")
    trtbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="trtbps")
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150, help="thalach")
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest.")
    
    st.markdown("---")
    st.header("📋 Clinical Details")
    
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: f"Type {x}", help="cp")
    exng = st.radio("Exercise Induced Angina", ["No", "Yes"], help="exng")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], format_func=lambda x: f"Slope {x}", help="slope")
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3], format_func=lambda x: f"{x} vessels", help="ca")
    thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: f"Type {x}", help="thal")

    st.markdown("---")
    model_choice = st.selectbox("🤖 Choose Prediction Model", list(models.keys()))

# --- Main Panel for Prediction ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Prediction")
    if st.button("Analyze Risk"):
        # Create input DataFrame
        input_dict = {
            "age": [age], "thalach": [thalach], "trtbps": [trtbps], "oldpeak": [np.sqrt(oldpeak)],
            "gender_1": [1 if gender == "Male" else 0],
            "cp_1": [1 if cp == 1 else 0], "cp_2": [1 if cp == 2 else 0], "cp_3": [1 if cp == 3 else 0],
            "exng_1": [1 if exng == "Yes" else 0],
            "slope_1": [1 if slope == 1 else 0], "slope_2": [1 if slope == 2 else 0],
            "ca_1": [1 if ca == 1 else 0], "ca_2": [1 if ca == 2 else 0], "ca_3": [1 if ca == 3 else 0],
            "thal_2": [1 if thal == 2 else 0], "thal_3": [1 if thal == 3 else 0]
        }
        input_df = pd.DataFrame(input_dict)

        # Ensure all columns are present and in order
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # Prediction
        model = models[model_choice]
        pred_prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        # Display result
        st.subheader("Risk Analysis Result")
        if pred == 1:
            st.error(f"**High Risk Detected**\n\nProbability of Heart Attack: **{pred_prob:.2%}**")
            st.warning("⚠️ **Disclaimer:** This is a prediction and not a medical diagnosis. Please consult a healthcare professional for an accurate assessment.")
        else:
            st.success(f"**Low Risk Detected**\n\nProbability of Heart Attack: **{pred_prob:.2%}**")
            st.info("✅ **Note:** While the risk is low, maintaining a healthy lifestyle is always recommended.")

with col2:
    st.image("https://www.heart.org/-/media/Images/News/2024/May/0501Heart-attack-stroke-risk_SC.jpg", caption="Stay Heart Healthy")

st.markdown("---")
st.markdown("Created with ❤️ by a Health-conscious Developer")
