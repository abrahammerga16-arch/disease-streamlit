"""
Integrated Healthcare Dashboard — Streamlit version (Enhanced UI with Animations)
Replicates the Google Colab notebook exactly:
  • Disease Predictor  (symptom input → ML prediction)
  • Health Recommender (disease dropdown → full health plan)
  • Healthcare Chatbot  (natural-language query → semantic answer)
  • Role / Age / ID access control
  • English ↔ Amharic UI

FIX: render_quick_select now uses aria-label CSS selectors and stable string-based
     button keys instead of fragile nth-of-type positional selectors, so symptom
     chips correctly toggle selected/unselected on every click.
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
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

/* ── Keyframe Animations */
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
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.8; }
}
@keyframes pop {
    0%   { transform: scale(0.8); opacity: 0; }
    50%  { transform: scale(1.05); }
    100% { transform: scale(1);   opacity: 1; }
}

/* ── Global Styles */
html, body, [class*="css"] {
    font-family: 'Poppins', 'IBM Plex Sans', sans-serif;
    scroll-behavior: smooth;
}

/* ── Main background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1f2d 100%);
    color: #e6edf3;
    animation: fadeInUp 0.8s ease-out;
}

/* ── Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0f1419 100%);
    border-right: 1px solid rgba(48, 54, 61, 0.5);
    box-shadow: inset -1px 0 0 rgba(88, 166, 255, 0.1);
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── Tabs */
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
    letter-spacing: 0.04em;
    padding: 10px 24px;
    border: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 0.95rem;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(88, 166, 255, 0.1);
    color: #58a6ff;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #21262d 0%, #1f6feb 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.25);
}

/* ── Input fields */
.stTextInput input, .stSelectbox select, .stTextArea textarea,
div[data-baseweb="select"] > div {
    background: rgba(33, 38, 45, 0.8) !important;
    border: 1px solid rgba(48, 54, 61, 0.5) !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    font-size: 0.95rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus,
div[data-baseweb="select"] > div:focus-within {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.2) !important;
    background: rgba(33, 38, 45, 1) !important;
}

/* ── Buttons (default green — applies everywhere EXCEPT chip sections) */
.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 0.95rem;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(46, 160, 67, 0.4);
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
}
.stButton > button:active { transform: translateY(0); }

/* ══════════════════════════════════════════════════
   QUICK-SELECT CHIP OVERRIDES
   ══════════════════════════════════════════════════ */

/* Category row chips */
[data-testid^="cat_btn_"] > button,
div[data-testid^="cat_btn_"] button {
    background: rgba(10, 20, 24, 0.75) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(13, 148, 136, 0.45) !important;
    border-radius: 20px !important;
    padding: 4px 14px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    height: auto !important;
    min-height: 0 !important;
    line-height: 1.6 !important;
    box-shadow: none !important;
    transform: none !important;
}
[data-testid^="cat_btn_"] > button:hover,
div[data-testid^="cat_btn_"] button:hover {
    background: rgba(13, 148, 136, 0.18) !important;
    color: #2dd4bf !important;
    border-color: #0d9488 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Symptom chips */
[data-testid^="sym__"] > button,
div[data-testid^="sym__"] button {
    background: rgba(10, 20, 24, 0.80) !important;
    color: #2dd4bf !important;
    border: 1px solid #0d9488 !important;
    border-radius: 20px !important;
    padding: 4px 14px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    height: auto !important;
    min-height: 0 !important;
    line-height: 1.6 !important;
    box-shadow: none !important;
    transform: none !important;
    letter-spacing: 0.01em !important;
}
[data-testid^="sym__"] > button:hover,
div[data-testid^="sym__"] button:hover {
    background: rgba(13, 148, 136, 0.20) !important;
    color: #5eead4 !important;
    border-color: #14b8a6 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Result cards */
.result-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.8) 0%, rgba(33,38,45,0.6) 100%);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    animation: fadeInUp 0.6s ease-out;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.result-card:hover {
    border-color: rgba(88,166,255,0.4);
    box-shadow: 0 8px 32px rgba(88,166,255,0.15);
    transform: translateY(-4px);
}
.result-card h4 {
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.result-card p, .result-card li {
    color: #c9d1d9;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ── Disease badge / confidence pill */
.disease-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: #fff;
    border-radius: 24px;
    padding: 6px 18px;
    font-weight: 600;
    font-size: 1rem;
    margin: 4px 8px 4px 0;
    animation: slideInLeft 0.4s ease-out;
    box-shadow: 0 4px 12px rgba(31,111,235,0.3);
}
.conf-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(33,38,45,0.9) 0%, rgba(88,166,255,0.15) 100%);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 16px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #58a6ff;
    margin-left: 4px;
    font-weight: 600;
    animation: slideInRight 0.4s ease-out;
}

/* ── Advice / warning banner */
.advice-banner {
    background: linear-gradient(135deg, rgba(45,24,0,0.8) 0%, rgba(240,136,62,0.1) 100%);
    border-left: 4px solid #f0883e;
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    color: #f0883e;
    font-size: 0.9rem;
    margin-top: 16px;
    animation: slideInLeft 0.5s ease-out;
}

/* ── Chat bubble */
.chat-bot {
    background: linear-gradient(135deg, rgba(33,38,45,0.9) 0%, rgba(31,111,235,0.1) 100%);
    border-left: 4px solid #58a6ff;
    padding: 16px 20px;
    border-radius: 0 12px 12px 0;
    color: #c9d1d9;
    font-size: 0.95rem;
    margin-top: 12px;
    animation: slideInLeft 0.5s ease-out;
    line-height: 1.6;
}

/* ── Section header */
.section-header {
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(88,166,255,0.2);
    padding-bottom: 8px;
    margin-bottom: 18px;
    animation: slideInLeft 0.4s ease-out;
    font-weight: 700;
}

/* ── Number input */
.stNumberInput input {
    background: rgba(33,38,45,0.8) !important;
    color: #e6edf3 !important;
    border: 1px solid rgba(48,54,61,0.5) !important;
    border-radius: 8px !important;
}

/* ── COMPACT ULTRA-SMALL SLIDER OVERRIDES ── */
div[data-testid="stSlider"] {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}
div[data-testid="stSlider"] > label {
    margin-bottom: 2px !important;
    font-size: 0.85rem !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}
/* Slider Track */
div[data-testid="stSlider"] [role="slider"] {
    height: 10px !important;
    width: 10px !important;
    top: -3px !important;
    background: #21262d !important;
    border: 2px solid #58a6ff !important;
}
/* Track filled portion background */
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    height: 4px !important;
}
/* Numerical display under the slider container */
div[data-testid="stSlider"] [data-testid="stSliderTickBar"] {
    display: none !important;
}
.stSlider .st-bk { background: linear-gradient(90deg, #1f6feb, #58a6ff) !important; }
.stSlider .st-ao { background: #21262d !important; border: 2px solid #58a6ff !important; }

/* ── Access denied */
.access-denied {
    background: linear-gradient(135deg, rgba(45,14,14,0.8) 0%, rgba(248,81,73,0.1) 100%);
    border-left: 4px solid #f85149;
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    color: #f85149;
    animation: slideInLeft 0.5s ease-out;
}

/* ── Main header */
.main-header { text-align: center; animation: fadeInUp 0.8s ease-out; margin-bottom: 32px; }
.main-header-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #79c0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.main-header-subtitle { color: #8b949e; font-size: 1rem; letter-spacing: 0.05em; margin-top: 8px; }

hr { border-color: rgba(88,166,255,0.1) !important; }
a  { color: #58a6ff; transition: color 0.3s ease; text-decoration: none; }
a:hover { color: #79c0ff; text-decoration: underline; }

@media (max-width: 768px) {
    .disease-badge, .conf-pill { display: block; margin: 6px 0; }
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
    "Hello! How can I help you with health information today?":
        "ሰላም! ዛሬ በጤና መረጃ እንዴት ልረዳዎ እችላለሁ?",
    "Access Denied: Information is not available for users under 18.":
        "የመዳረሻ እገዳ: ከ18 ዓመት በታች ለሆኑ ተጠቃሚዎች መረጃ አይገኝም",
    "Access Denied: Invalid ID for Student role.":
        "የመዳረሻ እገዳ: ለተማሪ ሚና ልክ ያልሆነ መታወቂያ",
    "Access Denied: Invalid ID for Doctor role.":
        "የመዳረሻ እገዳ: ለሐኪም ሚና ልክ ያልሆነ መታወቂያ",
    "Disease Predictor": "ምልክት መተንበይ",
    "Health Recommender": "የጤና ምክር ሰጪ",
    "Healthcare Chatbot": "የጤና እንክብካቤ ቻትቦት",
    "Predict & Recommend": "መተንበይ እና መምከር",
    "Get Plan": "እቅድ ያግኙ",
    "Ask Bot": "ቦት ይጠይቁ",
    "Please enter symptoms.": "እባክዎ ምልክቶችን ያስገቡ።",
    "Please enter a query.": "እባክዎ ጥያቄ ያስገቡ።",
    "medical_advice_disclaimer":
        "ማንኛውንም መድሃኒት ከመውሰድዎ በፊት ወይም ከባድ ምልክቶች ካጋጠሙዎት ሁልጊዜ ሐኪም ያማክሩ።",
    "🔒 Medication details are restricted. Please consult a licensed doctor.":
        "🔒 የመድሃኒት መረጃ የተገደበ ነው። እባክዎ ፈቃድ ያለው ሐኪም ያማክሩ።",
    "ℹ️ Full medication details are only available to Doctors.":
        "ℹ️ ሙሉ የመድሃኒት መረጃ ለሐኪሞች ብቻ ይገኛል።",
}

def t(text: str, lang: str) -> str:
    if lang.lower() == "amharic":
        return AMHARIC.get(text, text)
    return text

# ──────────────────────────────────────────────
# GOOGLE TRANSLATE (optional)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_translator():
    try:
        from googletrans import Translator
        return Translator()
    except Exception:
        return None

def translate_content(text, target_lang="English"):
    if target_lang.lower() != "amharic":
        return text
    if not text or (isinstance(text, float) and pd.isna(text)):
        return text
    translator = get_translator()
    if translator is None:
        return text
    try:
        if isinstance(text, list):
            return [translator.translate(str(i), dest="am").text for i in text]
        return translator.translate(str(text), dest="am").text
    except Exception:
        return text

# ──────────────────────────────────────────────
# LOAD DATA FILES
# ──────────────────────────────────────────────
DATA_FILES = [
    "data/Diseases_and_Symptoms_dataset.csv",
    "data/description.csv", "data/diets.csv",
    "data/medications.csv", "data/precautions.csv", "data/workout.csv",
]
MODEL_FILES = [
    "models/svc_model.pkl",
    "models/decision_tree_model.pkl", "models/label_encoder.pkl",
]

def check_files():
    return [f for f in DATA_FILES + MODEL_FILES if not os.path.exists(f)]

def clean_disease_name(name: str) -> str:
    return str(name).lower().replace("_", " ").strip()

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    main_df        = pd.read_csv("data/Diseases_and_Symptoms_dataset.csv")
    description_df = pd.read_csv("data/description.csv")
    diets_df       = pd.read_csv("data/diets.csv")
    medications_df = pd.read_csv("data/medications.csv")
    precautions_df = pd.read_csv("data/precautions.csv")
    workout_df     = pd.read_csv("data/workout.csv")

    description_map = {clean_disease_name(r["Disease"]): r["Description"]
                       for _, r in description_df.iterrows()}
    diets_map       = {clean_disease_name(r["Disease"]): r["Diet"]
                       for _, r in diets_df.iterrows()}
    medications_map = {clean_disease_name(r["Disease"]): r["Medication"]
                       for _, r in medications_df.iterrows()}

    precautions_map = {}
    for _, r in precautions_df.iterrows():
        d = clean_disease_name(r["Disease"])
        precs = [r.get(f"Precaution_{i}") for i in range(1, 5)]
        precautions_map[d] = [p for p in precs if pd.notna(p)]

    workout_col = workout_df.columns[1]
    workout_map = {clean_disease_name(r["Disease"]): r[workout_col]
                   for _, r in workout_df.iterrows()}

    return (main_df, description_map, diets_map,
            medications_map, precautions_map, workout_map)

@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    svc = joblib.load("models/svc_model.pkl")
    dt  = joblib.load("models/decision_tree_model.pkl")
    le  = joblib.load("models/label_encoder.pkl")
    return svc, dt, le

@st.cache_resource(show_spinner="Building symptom index…")
def build_tfidf_index(symptom_list: tuple, disease_names: tuple):
    """TF-IDF cosine similarity — no torch required."""
    sym_texts = [s.replace("_", " ") for s in symptom_list]
    dis_texts = [d.replace("_", " ") for d in disease_names]
    all_texts = sym_texts + dis_texts

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(all_texts)
    sym_matrix = vec.transform(sym_texts).toarray()
    dis_matrix = vec.transform(dis_texts).toarray()
    return vec, sym_matrix, dis_matrix

# ──────────────────────────────────────────────
# SYMPTOM CATEGORIZATION
# ──────────────────────────────────────────────
def categorize_symptoms(symptom_list):
    categories = {
        "General":        [],
        "Pain":           [],
        "Cardio / Resp":  [],
        "Neuro / Mental": [],
        "Gastro":         [],
        "Urinary":        [],
        "Skin":           [],
        "Reproductive":   [],
        "Other":          [],
    }
    kw = {
        "General":        ["fever","fatigue","tiredness","weakness","chills","sweat","malaise","weight","appetite","feeling"],
        "Pain":           ["pain","ache","cramp","stiff","sore","tender","backache","headache","migraine"],
        "Cardio / Resp":  ["chest","heart","pulse","palpitation","breath","cough","wheeze","asthma","cold","flu","sneeze"],
        "Neuro / Mental": ["dizziness","vertigo","confusion","memory","seizure","tremor","anxiety","depression","mood","sleep","insomnia"],
        "Gastro":         ["vomit","nausea","diarrhea","constipation","stomach","indigestion","bloat","gas","bowel","acidity"],
        "Urinary":        ["urin","bladder","kidney","discharge","burning urination","frequent urination"],
        "Skin":           ["rash","itch","burn","wound","acne","dermatitis","eczema","blister","hives","peeling"],
        "Reproductive":   ["menstrual","period","pregnancy","vaginal","erectile","fertility","libido","ovarian"],
    }
    for symptom in symptom_list:
        sl = symptom.lower().replace("_", " ")
        placed = False
        for cat, words in kw.items():
            if any(w in sl for w in words):
                categories[cat].append(symptom)
                placed = True
                break
        if not placed:
            categories["Other"].append(symptom)
    return {k: v for k, v in categories.items() if v}

# ──────────────────────────────────────────────
# ACCESS CONTROL
# ──────────────────────────────────────────────
def check_access(age: int, role: str, user_id: str, lang: str):
    if age < 18:
        return False, t("Access Denied: Information is not available for users under 18.", lang)
    if role == "Student" and user_id != "1111":
        return False, t("Access Denied: Invalid ID for Student role.", lang)
    if role == "Doctor" and user_id != "0000":
        return False, t("Access Denied: Invalid ID for Doctor role.", lang)
    return True, ""

# ──────────────────────────────────────────────
# ROLE-BASED OUTPUT FILTER
# ──────────────────────────────────────────────
def role_based_recs(role, lang, key,
                    description_map, diets_map, medications_map, precautions_map, workout_map):
    desc    = translate_content(description_map.get(key, "N/A"), lang)
    diet    = translate_content(diets_map.get(key, "N/A"), lang)
    meds    = translate_content(medications_map.get(key, "N/A"), lang)
    precs   = translate_content(precautions_map.get(key, []), lang)
    workout = translate_content(workout_map.get(key, "N/A"), lang)

    if role == "Doctor":
        return {
            t("Description", lang):      desc,
            t("Dietary Plan", lang):     diet,
            t("Medications", lang):      meds,
            t("Precautions", lang):      precs,
            t("Workout/Activity", lang): workout,
        }, ""

    elif role == "Student":
        if isinstance(meds, str) and meds != "N/A":
            meds_limited = ", ".join(
                w.split()[0] for w in meds.split(",") if w.strip()
            ) + " (drug classes only — full details restricted)"
        else:
            meds_limited = "Full medication details restricted to Doctors only."
        return {
            t("Description", lang):      desc,
            t("Dietary Plan", lang):     diet,
            t("Medications", lang):      meds_limited,
            t("Precautions", lang):      precs,
            t("Workout/Activity", lang): workout,
        }, "ℹ️ Full medication details are only available to Doctors."

    else:  # Normal User
        if isinstance(diet, str) and diet != "N/A":
            diet_limited = diet.split(".")[0].strip() + ". (Full dietary plan restricted — consult a nutritionist.)"
        else:
            diet_limited = "Eat balanced meals and stay hydrated. Consult a nutritionist for a personalised plan."
        meds_hidden = "🔒 Medication details are restricted. Please consult a licensed doctor."
        advice = ("⚠️ This information is for general awareness only. "
                  "Always consult a qualified doctor before taking any medication.")
        return {
            t("Description", lang):      desc,
            t("Dietary Plan", lang):     diet_limited,
            t("Medications", lang):      meds_hidden,
            t("Precautions", lang):      precs,
            t("Workout/Activity", lang): workout,
        }, advice

# ──────────────────────────────────────────────
# PREDICTION SYSTEM
# ──────────────────────────────────────────────
def integrated_prediction_system(
    user_input, age, role, user_id, lang,
    main_df, le, svc_model, dt_model,
    description_map, diets_map, medications_map, precautions_map, workout_map,
    tfidf_vec, sym_matrix,
    threshold=0.6,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return None, msg

    symptom_list = list(main_df.drop(columns=["diseases"]).columns)
    raw_symptoms = [s.strip().lower().replace(" ", "_")
                    for s in user_input.split(",") if s.strip()]
    matched_symptoms = set()

    for raw in raw_symptoms:
        if raw in symptom_list:
            matched_symptoms.add(raw)
            continue
        raw_vec = tfidf_vec.transform([raw.replace("_", " ")]).toarray()
        sims    = cosine_similarity(raw_vec, sym_matrix)[0]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            matched_symptoms.add(symptom_list[best_idx])

    if not matched_symptoms:
        return None, "⚠️ No recognised symptoms found. Try terms like: headache, fever, cough, vomiting."

    feature_vector = pd.DataFrame(
        [[1 if s in matched_symptoms else 0 for s in symptom_list]],
        columns=symptom_list,
    )

    preds = []

    for model, name in [(svc_model, "SVC"), (dt_model, "Decision Tree")]:
        try:
            proba    = model.predict_proba(feature_vector)[0]
            top4_idx = np.argsort(proba)[::-1][:4]
            for rank, idx in enumerate(top4_idx):
                disease = le.inverse_transform([idx])[0]
                conf    = f"{proba[idx] * 100:.1f}%"
                preds.append({
                    "model":      name,
                    "disease":    disease.title(),
                    "confidence": conf,
                    "top":        rank == 0,
                })
        except AttributeError:
            pred_idx = model.predict(feature_vector)[0]
            disease  = le.inverse_transform([pred_idx])[0]
            preds.append({
                "model":      name,
                "disease":    disease.title(),
                "confidence": "N/A",
                "top":        True,
            })

    top_disease = next(p["disease"] for p in preds if p["top"] and p["model"] == "SVC")
    top_key     = clean_disease_name(top_disease)
    preds = [p for p in preds if p["model"] == "SVC"]

    recs, advice = role_based_recs(
        role, lang, top_key,
        description_map, diets_map, medications_map, precautions_map, workout_map,
    )

    matched_display = [s.replace("_", " ").title() for s in matched_symptoms]
    return {
        "matched_symptoms":     matched_display,
        "predicted_conditions": preds,
        "top_disease":          top_disease,
        "recommendations":      recs,
        "advice":               advice,
    }, ""

# ──────────────────────────────────────────────
# HEALTH RECOMMENDER
# ──────────────────────────────────────────────
def health_recommender(
    disease_name, age, role, user_id, lang,
    description_map, diets_map, medications_map, precautions_map, workout_map,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return None, msg
    key = clean_disease_name(disease_name)
    return role_based_recs(
        role, lang, key,
        description_map, diets_map, medications_map, precautions_map, workout_map,
    )

# ──────────────────────────────────────────────
# CHATBOT
# ──────────────────────────────────────────────
def chatbot_response(
    query, age, role, user_id, lang,
    description_map, diets_map, medications_map, precautions_map, workout_map,
    disease_names, tfidf_vec, dis_matrix,
    threshold=0.55,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return msg

    q_vec    = tfidf_vec.transform([query]).toarray()
    sims     = cosine_similarity(q_vec, dis_matrix)[0]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    if best_sim < threshold:
        fallback = (
            "I couldn't find specific information for that query. "
            "Try asking about a specific disease (e.g. 'What is Asthma?') "
            "or use the Disease Predictor tab first."
        )
        return translate_content(fallback, lang) if lang == "Amharic" else fallback

    disease_key   = disease_names[best_idx]
    disease_title = disease_key.title()
    q_lower       = query.lower()

    if any(w in q_lower for w in ["diet","food","eat","nutrition","አመጋገብ"]):
        info  = diets_map.get(disease_key, "N/A")
        label = t("Dietary Plan", lang)
    elif any(w in q_lower for w in ["medicine","medication","drug","treat","መድሃኒት"]):
        info  = medications_map.get(disease_key, "N/A")
        label = t("Medications", lang)
    elif any(w in q_lower for w in ["precaution","avoid","prevent","ጥንቃቄ"]):
        info  = precautions_map.get(disease_key, [])
        label = t("Precautions", lang)
    elif any(w in q_lower for w in ["workout","exercise","activity","ስፖርት"]):
        info  = workout_map.get(disease_key, "N/A")
        label = t("Workout/Activity", lang)
    else:
        info  = description_map.get(disease_key, "N/A")
        label = t("Description", lang)

    info = translate_content(info, lang)
    info_str = ("\n• " + "\n• ".join(info)) if isinstance(info, list) else str(info)
    return f"**{disease_title}** — {label}:\n{info_str}"

# ──────────────────────────────────────────────
# RENDER HELPERS
# ──────────────────────────────────────────────
def render_recommendations(recs: dict):
    for label, value in recs.items():
        if isinstance(value, list):
            items   = "".join(f"<li>{i}</li>" for i in value)
            content = f"<ul style='margin:0;padding-left:18px'>{items}</ul>"
        else:
            content = f"<p style='margin:0'>{value}</p>"
        st.markdown(
            f"<div class='result-card'><h4>{label}</h4>{content}</div>",
            unsafe_allow_html=True,
        )

def render_predictions(predictions: list):
    rows_html = (
        "<div style='background:rgba(22,27,34,0.7);border:1px solid rgba(88,166,255,0.2);"
        "border-radius:10px;padding:14px 16px;margin-bottom:8px'>"
        "<div style='color:#79c0ff;font-size:0.75rem;font-weight:700;letter-spacing:0.1em;"
        "text-transform:uppercase;margin-bottom:10px'>🤖 SVC — Top 4 Predictions</div>"
    )
    for rank, p in enumerate(predictions):
        badge_bg  = "linear-gradient(135deg,#1f6feb,#388bfd)" if rank == 0 else "rgba(33,38,45,0.8)"
        badge_col = "#fff" if rank == 0 else "#8b949e"
        bar_w     = p["confidence"].replace("%", "") if p["confidence"] != "N/A" else "0"
        rows_html += (
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
            f"<span style='min-width:18px;color:#8b949e;font-size:0.75rem;font-weight:600'>#{rank+1}</span>"
            f"<span style='flex:1;background:{badge_bg};color:{badge_col};"
            f"padding:5px 12px;border-radius:16px;font-size:0.85rem;font-weight:600'>{p['disease']}</span>"
            f"<span style='min-width:48px;text-align:right;color:#58a6ff;font-size:0.82rem;font-weight:700'>{p['confidence']}</span>"
            f"</div>"
            f"<div style='height:3px;background:rgba(48,54,61,0.5);border-radius:2px;margin-bottom:6px'>"
            f"<div style='height:3px;width:{bar_w}%;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:2px'></div>"
            f"</div>"
        )
    st.markdown(rows_html + "</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# QUICK-SELECT SYMPTOM CHIPS
# ──────────────────────────────────────────────
def render_quick_select(categorized_symptoms: dict):
    """
    Fully interactive chip UI using real st.button calls.
    Chip style matches the provided image:
      - Small dark background pill with teal (#0d9488) border
      - Teal text, compact padding, tight rows
      - Selected state: slightly filled teal background
    """
    cats = list(categorized_symptoms.keys())

    # ── Session state defaults ────────────────────────────────────────────
    if "active_cat" not in st.session_state or st.session_state.active_cat not in cats:
        st.session_state.active_cat = cats[0]
    if "symptoms_selected" not in st.session_state:
        st.session_state.symptoms_selected = []

    # ── Global chip style matching the image ─────────────────────────────
    st.markdown("""
<style>
/* ── Scope all quick-select styling inside .qs-section ── */
.qs-section .stButton > button {
    background: rgba(10, 25, 28, 0.85) !important;
    color: #2dd4bf !important;
    border: 1px solid #0d9488 !important;
    border-radius: 20px !important;
    padding: 3px 12px !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    height: auto !important;
    min-height: 0 !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease !important;
    box-shadow: none !important;
}
.qs-section .stButton > button:hover {
    background: rgba(13, 148, 136, 0.18) !important;
    color: #5eead4 !important;
    border-color: #14b8a6 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Category tab buttons — slightly brighter border to distinguish ── */
.qs-section-cats .stButton > button {
    background: rgba(10, 25, 28, 0.6) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(13, 148, 136, 0.35) !important;
    border-radius: 20px !important;
    padding: 3px 12px !important;
    font-size: 0.73rem !important;
    font-weight: 500 !important;
    height: auto !important;
    min-height: 0 !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    box-shadow: none !important;
}
.qs-section-cats .stButton > button:hover {
    background: rgba(13, 148, 136, 0.12) !important;
    color: #2dd4bf !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Tighten column gaps ── */
.qs-section [data-testid="column"],
.qs-section-cats [data-testid="column"] {
    padding: 2px 3px !important;
}
</style>
""", unsafe_allow_html=True)

    # ── ROW 1: category chips ─────────────────────────────────────────────
    cat_styles = []
    for cat in cats:
        is_active = (cat == st.session_state.active_cat)
        if is_active:
            cat_styles.append(f'div[data-testid="cat_btn_{cat}"] button {{ background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important; color: #ffffff !important; border-color: #14b8a6 !important; font-weight: 600 !important; }}')
    if cat_styles:
        st.markdown(f"<style>{' '.join(cat_styles)}</style>", unsafe_allow_html=True)

    st.markdown("<div class='qs-section-cats'>", unsafe_allow_html=True)
    cols_cat = st.columns(len(cats))
    for i, cat in enumerate(cats):
        with cols_cat[i]:
            if st.button(cat, key=f"cat_btn_{cat}"):
                st.session_state.active_cat = cat
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── ROW 2: items matching active category ──────────────────────────────
    current_symptoms = categorized_symptoms.get(st.session_state.active_cat, [])
    sym_styles = []
    for sym in current_symptoms:
        is_sel = (sym in st.session_state.symptoms_selected)
        if is_sel:
            sym_styles.append(f'div[data-testid="sym__{sym}"] button {{ background: rgba(13, 148, 136, 0.25) !important; border-color: #14b8a6 !important; color: #5eead4 !important; font-weight: 600 !important; }}')
    if sym_styles:
        st.markdown(f"<style>{' '.join(sym_styles)}</style>", unsafe_allow_html=True)

    st.markdown("<div class='qs-section' style='margin-top: 6px;'>", unsafe_allow_html=True)
    if current_symptoms:
        cols_sym = st.columns(min(len(current_symptoms), 5))
        for idx, sym in enumerate(current_symptoms):
            col_idx = idx % 5
            with cols_sym[col_idx]:
                label = sym.replace("_", " ").title()
                if sym in st.session_state.symptoms_selected:
                    label = "✓ " + label
                if st.button(label, key=f"sym__{sym}"):
                    if sym in st.session_state.symptoms_selected:
                        st.session_state.symptoms_selected.remove(sym)
                    else:
                        st.session_state.symptoms_selected.append(sym)
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────
def main():
    missing = check_files()
    if missing:
        st.error("### Missing required system files:")
        for m in missing:
            st.code(f" ❌ {m}")
        st.info("Ensure training outputs match path configuration criteria.")
        return

    main_df, desc_map, diet_map, meds_map, precs_map, work_map = load_data()
    svc, dt, le = load_models()

    symptom_list = tuple(main_df.drop(columns=["diseases"]).columns)
    disease_names = tuple(clean_disease_name(d) for d in le.classes_)
    tfidf_vec, sym_matrix, dis_matrix = build_tfidf_index(symptom_list, disease_names)
    cat_symptoms_map = categorize_symptoms(symptom_list)

    # ── SIDEBAR ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div class='section-header'>Configuration Engine</div>", unsafe_allow_html=True)
        lang = st.selectbox("Language / ቋንቋ", ["English", "Amharic"])
        role = st.selectbox("Access Role", ["User", "Student", "Doctor"])
        
        user_id = ""
        if role in ["Student", "Doctor"]:
            user_id = st.text_input(f"Enter {role} Security ID Key", type="password")

        st.markdown("<br><hr>", unsafe_allow_html=True)
        age = st.slider("User Patient Age", min_value=1, max_value=100, value=25)

    # ── MAIN PANEL HEADER ─────────────────────────────────────────────────
    st.markdown(
        f"<div class='main-header'>"
        f"<div class='main-header-title'>🏥 Integrated Healthcare Dashboard</div>"
        f"<div class='main-header-subtitle'>Streamlit Deep Diagnostics Protocol Environment</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    t1, t2, t3 = st.tabs([t("Disease Predictor", lang), t("Health Recommender", lang), t("Healthcare Chatbot", lang)])

    # ── TAB 1: PREDICTOR ──────────────────────────────────────────────────
    with t1:
        st.markdown(f"<div class='section-header'>{t('Disease Predictor', lang)}</div>", unsafe_allow_html=True)
        
        st.write("**Quick-Select Core Symptoms Inventory:**")
        render_quick_select(cat_symptoms_map)
        st.markdown("<br>", unsafe_allow_html=True)

        sel_str = ", ".join(st.session_state.symptoms_selected)
        user_input = st.text_input(
            "Symptoms Input Box (Selected inventory auto-updates below — or type manually):",
            value=sel_str,
            key="predict_input_field"
        )

        if st.button(t("Predict & Recommend", lang), key="btn_predict"):
            if not user_input.strip():
                st.warning(t("Please enter symptoms.", lang))
            else:
                res, err = integrated_prediction_system(
                    user_input, age, role, user_id, lang,
                    main_df, le, svc, dt,
                    desc_map, diet_map, meds_map, precs_map, work_map,
                    tfidf_vec, sym_matrix
                )
                if err:
                    st.markdown(f"<div class='access-denied'>{err}</div>", unsafe_allow_html=True)
                elif res:
                    col_left, col_right = st.columns([1, 1.4])
                    with col_left:
                        st.markdown("**Matched Input Tokens:**")
                        st.code(", ".join(res["matched_symptoms"]))
                        render_predictions(res["predicted_conditions"])
                    with col_right:
                        st.markdown(
                            f"Primary Condition Identified: <span class='disease-badge'>{res['top_disease']}</span>",
                            unsafe_allow_html=True
                        )
                        render_recommendations(res["recommendations"])
                        if res["advice"]:
                            st.markdown(f"<div class='advice-banner'>{res['advice']}</div>", unsafe_allow_html=True)

    # ── TAB 2: RECOMMENDER ────────────────────────────────────────────────
    with t2:
        st.markdown(f"<div class='section-header'>{t('Health Recommender', lang)}</div>", unsafe_allow_html=True)
        chosen_disease = st.selectbox("Select Target Condition Profile:", ["Select Disease"] + sorted([d.title() for d in le.classes_]))
        
        if st.button(t("Get Plan", lang), key="btn_recommender"):
            if chosen_disease == "Select Disease":
                st.warning("Please select a valid disease profile.")
            else:
                recs, advice = health_recommender(chosen_disease, age, role, user_id, lang, desc_map, diet_map, meds_map, precs_map, work_map)
                if recs is None:
                    st.markdown(f"<div class='access-denied'>{advice}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"Management Stratum for: <span class='disease-badge'>{chosen_disease}</span>", unsafe_allow_html=True)
                    render_recommendations(recs)
                    if advice:
                        st.markdown(f"<div class='advice-banner'>{advice}</div>", unsafe_allow_html=True)

    # ── TAB 3: CHATBOT ────────────────────────────────────────────────────
    with t3:
        st.markdown(f"<div class='section-header'>{t('Healthcare Chatbot', lang)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bot'>🤖 {t('Hello! How can I help you with health information today?', lang)}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        query = st.text_input("Enter natural language question regarding conditions, metrics, rules, or regimes:")
        if st.button(t("Ask Bot", lang), key="btn_chat"):
            if not query.strip():
                st.warning(t("Please enter a query.", lang))
            else:
                answer = chatbot_response(query, age, role, user_id, lang, desc_map, diet_map, meds_map, precs_map, work_map, disease_names, tfidf_vec, dis_matrix)
                if "Access Denied" in answer:
                    st.markdown(f"<div class='access-denied'>{answer}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-bot'>🤖 {answer}</div>", unsafe_allow_html=True)

    st.markdown("<br><br><hr><center style='color:#8b949e; font-size:0.8rem;'>Medical Disclaimer: For testing and validation environments only.</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
