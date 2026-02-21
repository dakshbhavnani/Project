import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from utils import get_career_info, skill_gap_analysis

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Career Guidance System",
    layout="wide",
    page_icon="🎓"
)

# ---------------- LOAD FILES SAFELY ----------------
# This section prevents the app from crashing if files are missing
try:
    model = joblib.load("career_model.pkl")
    encoders = joblib.load("encoders.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    career_df = pd.read_csv("career_knowledge_dataset.csv")
    student_df = pd.read_csv("dataSet.csv")
    feature_cols = [c for c in student_df.columns if c != "Suggested Job Role"]
    FILES_LOADED = True
except FileNotFoundError as e:
    st.error(f"Critical file missing: {e}. Please check if .pkl and .csv files are in the folder.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

# ---------------- CSS STYLING ----------------
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .main-title { font-size: 2.8rem; font-weight: 700; color: #ffffff; text-align: center; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.1rem; color: #a0a0a0; text-align: center; margin-bottom: 3rem; font-weight: 300; }

    /* Navigation - Modern Pill Design */
    [data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #21262d; }
    [data-testid="stSidebar"] div[role="radiogroup"] { gap: 12px; padding: 20px 10px; }
    [data-testid="stSidebar"] label { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; color: white !important; padding: 18px 20px !important; margin: 4px 0; transition: all 0.3s ease; font-weight: 500; }
    [data-testid="stSidebar"] label:hover { background: rgba(255, 255, 255, 0.08); transform: translateY(-2px); }
    [data-testid="stSidebar"] label[data-checked="true"] { background: linear-gradient(135deg, #FF512F 0%, #DD2476 100%); color: white !important; font-weight: 700; border: none; box-shadow: 0 8px 20px rgba(221, 36, 118, 0.4); }
    [data-testid="stSidebar"] input[type="radio"] { display: none; }

    /* Cards & Components */
    .info-card { background-color: #232b3e; border-radius: 12px; padding: 25px; border: 1px solid #2d3650; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 20px; color: #e2e8f0; }
    .section-header { color: #ffffff; font-size: 1.1rem; font-weight: 600; margin-bottom: 15px; }
    .stButton > button { width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; padding: 14px; border-radius: 10px; font-weight: 600; font-size: 1.1rem; }
    .result-box { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border: 1px solid #334155; border-radius: 16px; padding: 40px; text-align: center; margin-top: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.4); }
    .result-label { color: #94a3b8; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 1px; }
    .result-career { color: #38bdf8; font-size: 3rem; font-weight: 800; margin: 10px 0; }
    .result-conf { color: #e2e8f0; font-size: 1.5rem; }
    .success-box { background-color: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); padding: 25px; border-radius: 12px; color: white; }
    .error-box { background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); padding: 25px; border-radius: 12px; color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🎓 AI Career Guidance System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning • Career Intelligence • Skill Gap Analysis</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["🏠 Career Prediction", "📘 Career Explorer", "🧠 Skill Gap Analyzer"])

# =====================================================
# 🏠 PAGE 1 : PREDICTION
# =====================================================
if page == "🏠 Career Prediction":
    st.markdown("<div class='section-header'><span>👤</span> Student Profile</div>", unsafe_allow_html=True)
    
    inputs = {}
    cols_per_row = 3
    
    # Robust grid generation
    for i in range(0, len(feature_cols), cols_per_row):
        current_cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(feature_cols):
                col_name = feature_cols[idx]
                col_obj = current_cols[j]
                
                if student_df[col_name].dtype == "object":
                    # Convert to list for safety
                    options = student_df[col_name].dropna().unique().tolist()
                    inputs[col_name] = col_obj.selectbox(col_name, options)
                else:
                    # Handle numeric data safely
                    min_val = int(student_df[col_name].min())
                    max_val = int(student_df[col_name].max())
                    inputs[col_name] = col_obj.slider(col_name, min_val, max_val, min(5, max_val))

    if st.button("🔮 Predict Career"):
        try:
            input_df = pd.DataFrame([inputs])
            for col, enc in encoders.items():
                if col in input_df:
                    input_df[col] = enc.transform(input_df[col].astype(str))
            
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df).max()
            career = label_encoder.inverse_transform([pred])[0]

            st.markdown(f"""
            <div class='result-box'>
                <div class='result-label'>Predicted Career Path</div>
                <div class='result-career'>{career}</div>
                <div class='result-conf'>Confidence Score: {round(proba*100, 2)}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(proba*100))
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =====================================================
# 📘 PAGE 2 : CAREER EXPLORER
# =====================================================
elif page == "📘 Career Explorer":
    st.markdown("<div class='section-header'><span>📘</span> Explore Career Paths</div>", unsafe_allow_html=True)

    career = st.selectbox("Choose Career", career_df.iloc[:,0].unique())
    info = get_career_info(career_df, career)

    if info is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='info-card'><div class='section-header'>📄 Description</div><p>{info.iloc[1]}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-card'><div class='section-header'>🛠 Required Skills</div><p>{info.iloc[2]}</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='info-card'><div class='section-header'>📚 Certifications</div><p>{info.iloc[3]}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-card'><div class='section-header'>🚀 Career Roadmap</div><p>{info.iloc[4]}</p></div>", unsafe_allow_html=True)
    else:
        st.warning("Career information not found.")

# =====================================================
# 🧠 PAGE 3 : SKILL GAP
# =====================================================
else:
    st.markdown("<div class='section-header'><span>🧠</span> Skill Gap Analyzer</div>", unsafe_allow_html=True)

    career = st.selectbox("Select Target Career", career_df.iloc[:,0].unique())
    user_skills = st.text_input("Enter your skills (comma separated)")

    if st.button("Analyze Skill Gap"):
        info = get_career_info(career_df, career)
        if info is not None:
            required = info.iloc[2]
            matched, missing = skill_gap_analysis(user_skills.split(","), required)
            
            col1, col2 = st.columns(2)
            with col1:
                matched_html = "<br>".join([f"✅ {s}" for s in matched]) if matched else "None"
                st.markdown(f"<div class='success-box'><div class='section-header'>✔ Matching Skills</div><p>{matched_html}</p></div>", unsafe_allow_html=True)
            with col2:
                missing_html = "<br>".join([f"❌ {s}" for s in missing]) if missing else "None"
                st.markdown(f"<div class='error-box'><div class='section-header'>❌ Missing Skills</div><p>{missing_html}</p></div>", unsafe_allow_html=True)
