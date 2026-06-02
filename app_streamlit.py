import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Integrated Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# ENHANCED CUSTOM CSS WITH ANIMATIONS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&family=Poppins:wght@300;400;600;700&display=swap');

/* Keyframe Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to   { opacity: 1; transform: translateX(0); }
}

html, body, [class*="css"] {
    font-family: 'Poppins', 'IBM Plex Sans', sans-serif;
    scroll-behavior: smooth;
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1f2d 100%);
    color: #e6edf3;
    animation: fadeInUp 0.8s ease-out;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0f1419 100%);
    border-right: 1px solid rgba(48, 54, 61, 0.5);
    box-shadow: inset -1px 0 0 rgba(88, 166, 255, 0.1);
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(22, 27, 34, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid rgba(48, 54, 61, 0.3);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    border: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #21262d 0%, #1f6feb 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.25);
}

.result-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.8) 0%, rgba(33,38,45,0.6) 100%);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    animation: fadeInUp 0.6s ease-out;
}
.result-card h4 {
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    text-transform: uppercase;
}
.disease-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: #fff;
    border-radius: 24px;
    padding: 6px 18px;
    font-weight: 600;
}
.conf-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(33,38,45,0.9) 0%, rgba(88,166,255,0.15) 100%);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 16px;
    padding: 4px 14px;
    color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# AMHARIC TRANSLATIONS
# ──────────────────────────────────────────────
AMHARIC = {
    "Disease": "በሽታ",
    "Description": "መግለጫ",
    "Dietary Plan": "የአመጋገብ እቅድ",
    "Medications": "መድሃኒቶች",
    "Workout/Activity": "ስፖርት/እንቅስቃሴ",
    "Precautions": "ጥንቃቄዎች",
    "Disease Predictor": "ምልክት መተንበይ",
    "Predict & Recommend": "መተንበይ እና መምከር",
    "Please select symptoms first.": "እባክዎ መጀመሪያ ምልክቶችን ይምረጡ።",
}

def t(text: str, lang: str) -> str:
    if lang.lower() == "amharic":
        return AMHARIC.get(text, text)
    return text

# ──────────────────────────────────────────────
# CORE DATA STRUCT FOR QUICK SELECT
# ──────────────────────────────────────────────
CATEGORIES = {
    "🌡️ General": ["fever", "fatigue", "weakness", "chills", "sweating", "weight gain", "lethargy", "weight loss"],
    "🤕 Pain": ["headache", "back pain", "chest pain", "joint pain", "muscle pain", "abdominal pain", "neck pain", "sore throat"],
    "🫀 Cardio/Resp": ["cough", "shortness of breath", "chest tightness", "palpitations", "wheezing", "sneezing", "runny nose"],
    "🧠 Neuro/Mental": ["dizziness", "confusion", "anxiety", "depression", "insomnia", "memory loss", "fainting", "numbness"],
    "🤢 Gastro": ["nausea", "vomiting", "diarrhea", "constipation", "stomach bloating", "loss of appetite", "heartburn"],
    "🩺 Skin": ["skin rash", "itching", "acne", "skin dryness", "skin swelling", "jaundice"],
}

# Flatten all unique symptoms into a clean catalog list
ALL_SYMPTOMS_LIST = sorted(list(set([sym for sublist in CATEGORIES.values() for sym in sublist])))

# ──────────────────────────────────────────────
# RENDER FIXED QUICK-SELECT COMPONENT
# ──────────────────────────────────────────────
def render_quick_select_symptoms(lang: str) -> list:
    """
    Renders an integrated Python-native multi-selector styled 
    as an interactive category filter layout. 
    Guarantees ML engine uses ONLY selected symptoms.
    """
    is_am = lang.lower() == "amharic"
    quick_label = "ምልክቶችን ፈጥኖ ይምረጡ (ክሊክ ያድርጉ):" if is_am else "Quick-Select Input Symptoms:"
    
    st.markdown(f"<div style='font-size:0.85rem; font-weight:600; color:#58a6ff; margin-bottom:8px;'>{quick_label}</div>", unsafe_allow_html=True)
    
    # Category Filter Tabs using a clean Selectbox Matrix or multi-pill configuration
    cat_choices = list(CATEGORIES.keys())
    selected_cat = st.radio("Filter Symptoms By System Category:", cat_choices, horizontal=True)
    
    # Only show symptoms belonging to the clicked/activated category group
    available_pills = CATEGORIES[selected_cat]
    
    # Save clicked items directly into Python variables, completely isolated from DOM injection hacks
    chosen_symptoms = st.multiselect(
        f"Select active symptoms from {selected_cat}: ", 
        options=available_pills,
        default=[s for s in st.session_state.get("selected_symptoms", []) if s in available_pills],
        key=f"pill_select_{selected_cat}"
    )
    
    # Persistent Session Aggregation Cache
    if "selected_symptoms" not in st.session_state:
        st.session_state["selected_symptoms"] = []
        
    for s in chosen_symptoms:
        if s not in st.session_state["selected_symptoms"]:
            st.session_state["selected_symptoms"].append(s)
            
    # Display current selections cleanly as high-visibility badge elements
    if st.session_state["selected_symptoms"]:
        st.markdown("**Currently Selected Symptoms Sent to ML Engine:**")
        cols = st.columns(len(st.session_state["selected_symptoms"]) if len(st.session_state["selected_symptoms"]) > 0 else 1)
        
        for idx, active_sym in enumerate(st.session_state["selected_symptoms"]):
            st.markdown(f"<span class='conf-pill'>✓ {active_sym.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
            
        if st.button("🧹 Clear All Selections"):
            st.session_state["selected_symptoms"] = []
            st.rerun()
            
    return st.session_state["selected_symptoms"]

# ──────────────────────────────────────────────
# MAIN PIPELINE DEMO IMPLEMENTATION
# ──────────────────────────────────────────────
def main():
    st.markdown("<div class='main-header'><h1 class='main-header-title'>🏥 Diagnostics Module</h1></div>", unsafe_allow_html=True)
    
    # Use selected array directly as inputs 
    selected_symptoms = render_quick_select_symptoms(lang="English")
    
    if st.button("Predict & Recommend"):
        if not selected_symptoms:
            st.warning("Please select symptoms first.")
        else:
            # Join names for compatibility with your existing TF-IDF index pipeline
            input_string = ", ".join(selected_symptoms)
            st.success(f"Processing clinical engine inputs: `{input_string}`")

if __name__ == "__main__":
    main()
