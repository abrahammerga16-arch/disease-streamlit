"""
Integrated Healthcare Dashboard — Streamlit version (Enhanced UI with Animations)
Replicates the Google Colab notebook exactly:
  • Disease Predictor  (symptom input → ML prediction)
  • Health Recommender (disease dropdown → full health plan)
  • Healthcare Chatbot  (natural-language query → semantic answer)
  • Role / Age / ID access control
  • English ↔ Amharic UI
"""

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
# ENHANCED CUSTOM CSS WITH ANIMATIONS & BLINKING CURSOR
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght=400;600&family=IBM+Plex+Sans:wght=300;400;600;700&family=Poppins:wght=300;400;600;700&display=swap');

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

/* Artificial Cursor Pipe Blinking Sequence */
@keyframes blink-cursor {
    0%, 100% { border-left-color: #58a6ff; }
    50% { border-left-color: transparent; }
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

/* Artificial Blinking Entry Pipe when focused and empty */
.stTextArea textarea:focus:placeholder-shown {
    animation: blink-cursor 1s step-end infinite;
    border-left: 3px solid #58a6ff !important;
    padding-left: 10px !important;
}

/* ── Buttons */
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
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(46, 160, 67, 0.4);
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
}
.stButton > button:active { transform: translateY(0); }

/* Custom Secondary Clear Button Styling */
div.clear-btn-container > div > button {
    background: linear-gradient(135deg, #21262d 0%, #30363d 100%) !important;
    color: #f85149 !important;
    border: 1px solid rgba(248, 81, 73, 0.4) !important;
}
div.clear-btn-container > div > button:hover {
    background: linear-gradient(135deg, rgba(248, 81, 73, 0.1) 0%, rgba(248, 81, 73, 0.2) 100%) !important;
    box-shadow: 0 8px 24px rgba(248, 81, 73, 0.2) !important;
    border-color: #f85149 !important;
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

/* ── Slider */
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
    "Medications": "مድሃኒቶች",
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
    "Clear Symptoms": "ምልክቶችን አጽዳ",
    "Clear Diagnosis": "ውጤቶችን አጽዳ",
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
# GOOGLE TRANSLATE CONFIG
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
    sym_texts = [s.replace("_", " ") for s in symptom_list]
    dis_texts = [d.replace("_", " ") for d in disease_names]
    all_texts = sym_texts + dis_texts

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(all_texts)
    sym_matrix = vec.transform(sym_texts).toarray()
    dis_matrix = vec.transform(dis_texts).toarray()
    return vec, sym_matrix, dis_matrix


# ──────────────────────────────────────────────
# QUICK-SELECT SYMPTOM WIDGET (STREAMLIT IFRAME SYNC BRIDGE)
# ──────────────────────────────────────────────
def render_quick_select_symptoms(lang: str):
    CATEGORIES_EN = {
        "🌡️ General":       ["fever", "fatigue", "weakness", "chills", "sweating",
                              "weight gain", "malaise", "lethargy", "weight loss",
                              "night sweats"],
        "🤕 Pain":            ["headache", "back pain", "chest pain", "joint pain",
                              "muscle pain", "abdominal pain", "neck pain", "knee pain",
                              "shoulder pain", "sore throat", "ear pain", "eye pain"],
        "🫀 Cardio/Resp":    ["cough", "shortness of breath", "chest tightness",
                              "palpitations", "irregular heartbeat",
                              "difficulty breathing", "wheezing", "sneezing",
                              "runny nose", "nasal congestion"],
        "🧠 Neuro/Mental":   ["dizziness", "confusion", "anxiety", "depression",
                              "insomnia", "memory loss", "seizures", "tremors",
                              "fainting", "numbness", "blurred vision"],
        "🤢 Gastro":         ["nausea", "vomiting", "diarrhea", "constipation",
                              "stomach bloating", "loss of appetite", "heartburn",
                              "indigestion", "blood in stool"],
        "🩺 Skin":           ["skin rash", "itching", "acne", "skin dryness",
                              "skin swelling", "jaundice", "skin lesion",
                              "hives", "peeling skin"],
        "👁️ Eye/Ear/Nose":  ["eye redness", "ear pain", "blurred vision",
                              "runny nose", "nasal congestion", "hearing loss",
                              "watery eyes"],
        "🦴 Musculo":        ["joint stiffness", "muscle cramps", "swelling",
                              "leg weakness", "arm weakness", "back stiffness",
                              "peripheral edema"],
        "🚻 Urinary":        ["frequent urination", "painful urination",
                              "blood in urine", "urinary retention", "dark urine"],
        "🔬 Other":          ["jaundice", "hair loss", "swollen lymph nodes",
                              "high blood sugar", "low blood pressure"],
    }

    CAT_AM = {
        "🌡️ General":      "አጠቃላይ",
        "🤕 Pain":          "ህመም",
        "🫀 Cardio/Resp":   "ልብ/መተንፈሻ",
        "🧠 Neuro/Mental":  "ነርቭ/አዕምሮ",
        "🤢 Gastro":        "የምግብ መፈጨት",
        "🩺 Skin":          "ቆዳ",
        "👁️ Eye/Ear/Nose": "ዓይን/ጆሮ/አፍንጫ",
        "🦴 Musculo":       "ጡንቻ",
        "🚻 Urinary":       "የሽንት",
        "🔬 Other":         "ሌላ",
    }

    SYMPTOM_AM = {
        "fever": "ትኩሳት", "fatigue": "ድካም", "weakness": "ድክመት",
        "chills": "ብርድ", "sweating": "ላብ", "weight gain": "ክብደት መጨመር",
        "malaise": "ስሜት መጥፎ", "lethargy": "ዝላይ", "weight loss": "ክብደት መቀነስ",
        "night sweats": "ሌሊት ላብ",
        "headache": "ራስ ምታት", "back pain": "የጀርባ ህመም",
        "chest pain": "የደረት ህመም", "joint pain": "የመገጣጠሚያ ህመም",
        "muscle pain": "የጡንቻ ህመም", "abdominal pain": "የሆድ ህመም",
        "neck pain": "የአንገት ህመም", "knee pain": "የጉልበት ህመም",
        "shoulder pain": "የትከሻ ህመም", "sore throat": "ጉሮሮ ህመም",
        "ear pain": "የጆሮ ህመም", "eye pain": "የዓይን ህመም",
        "cough": "ሳል", "shortness of breath": "መተንፈስ ማጠር",
        "chest tightness": "ደረት መጠበቅ", "palpitations": "ልብ ምት ስሜት",
        "irregular heartbeat": "ያልተስተካከለ የልብ ምት",
        "difficulty breathing": "ለመተንፈስ ችግር",
        "wheezing": "ድምፅ ሲተነፍሱ", "sneezing": "ማስነጠስ",
        "runny nose": "አፍንጫ ፍሳሽ", "nasal congestion": "አፍንጫ መዘጋት",
        "dizziness": "ራስ ዞር", "confusion": "ግራ መጋባት",
        "anxiety": "ጭንቀት", "depression": "ድብርት",
        "insomnia": "እንቅልፍ ማጣት", "memory loss": "ትውስታ ማጣት",
        "seizures": "ቅብጠት", "tremors": "መርበድበድ",
        "fainting": "ዋዛ ማጣት", "numbness": "ደንዘዝ ስሜት",
        "blurred vision": "ደበዘዘ ዕይታ",
        "nausea": "ማቅለሽለሽ", "vomiting": "ማስታወክ", "diarrhea": "ተቅማጥ",
        "constipation": "ሆድ መጠፍጠፍ", "stomach bloating": "ሆድ ማበጥ",
        "loss of appetite": "የምግብ ፍቅር ማጣት", "heartburn": "ሆድ ማቃጠል",
        "indigestion": "ምግብ አለመፈጨት", "blood in stool": "ሰገራ ውስጥ ደም",
        "skin rash": "ቆዳ ሽፍታ", "itching": "ማሳከክ", "acne": "ሽፍታ",
        "skin dryness": "ቆዳ ደረቅ", "skin swelling": "ቆዳ ማበጥ",
        "jaundice": "ቢጫ በሽታ", "skin lesion": "ቆዳ ቁስለት",
        "hives": "ድርቀት", "peeling skin": "ቆዳ መላጥ",
        "eye redness": "ቀይ ዓይን", "hearing loss": "የመስሚያ ችግር",
        "watery eyes": "እንባ ዓይን",
        "joint stiffness": "መገጣጠሚያ ጥበቃ", "muscle cramps": "ጡንቻ ቁርጠት",
        "swelling": "ማበጥ", "leg weakness": "የእግር ድክመት",
        "arm weakness": "የእጅ ድክመት", "back stiffness": "ጀርባ ጥበቃ",
        "peripheral edema": "ዳርቻ ማበጥ",
        "frequent urination": "ተደጋጋሚ ሽንት",
        "painful urination": "ሽንት ሲሸኑ ህመም",
        "blood in urine": "ሽንት ውስጥ ደም",
        "urinary retention": "ሽንት ማቆር", "dark urine": "ጨለማ ሽንት",
        "hair loss": "ፀጉር መርገፍ", "swollen lymph nodes": "ሊምፍ ኖድ ማበጥ",
        "high blood sugar": "ከጨማሪ የደም ስኳር",
        "low blood pressure": "ዝቅተኛ የደም ግፊት",
    }

    is_am = lang.lower() == "amharic"

    js_cats = {}
    for cat_en, symptoms in CATEGORIES_EN.items():
        label = CAT_AM.get(cat_en, cat_en) if is_am else cat_en
        js_cats[label] = [
            {
                "en":      s,
                "display": SYMPTOM_AM.get(s, s) if is_am else s.title(),
            }
            for s in symptoms
        ]

    current_val = st.session_state.get("symptoms_text", "")
    
    # SAFEGUARD CRASH FIX: Intercept any DeltaGenerator layout proxy objects passed by Streamlit initial state
    if not isinstance(current_val, str):
        current_val = ""
        
    current_list = [s.strip().lower() for s in current_val.split(",") if s.strip()]

    quick_label = "ምልክቶችን ፈጥኖ ይምረጡ:" if is_am else "Quick-select symptoms:"
    or_text     = "ወይም ምልክቶችን ይተይቡ"   if is_am else "or type symptoms manually:"

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: transparent;
    font-family: 'Poppins', -apple-system, sans-serif;
    padding: 4px 2px 0;
  }}
  .qs-label {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.38);
    font-weight: 600;
    margin-bottom: 10px;
  }}
  .cat-tabs {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
  }}
  .cat-tab {{
    padding: 5px 12px;
    border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.5);
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.18s ease;
  }}
  .cat-tab:hover, .cat-tab.active {{
    border-color: #14b8a6;
    color: #14b8a6;
    background: rgba(13,148,136,0.1);
  }}
  .pills-wrap {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    max-height: 118px;
    overflow-y: auto;
    padding-bottom: 6px;
  }}
  .pill {{
    padding: 5px 13px;
    border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.75);
    font-size: 0.77rem;
    cursor: pointer;
    transition: all 0.16s ease;
  }}
  .pill:hover {{
    border-color: #14b8a6;
    color: #fff;
    background: rgba(13,148,136,0.12);
  }}
  .pill.selected {{
    background: rgba(13, 148, 136, 0.25) !important;
    border-color: #14b8a6 !important;
    color: #14b8a6 !important;
    font-weight: 600;
  }}
  .or-div {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 14px 0 2px;
    color: rgba(255,255,255,0.22);
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
  .or-div::before, .or-div::after {{ content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.08); }}
</style>
</head>
<body>

<div class="qs-label">{quick_label}</div>
<div class="cat-tabs" id="catTabs"></div>
<div class="pills-wrap" id="pillsWrap"></div>
<div class="or-div">{or_text}</div>

<script>
  var CATS = {json.dumps(js_cats, ensure_ascii=False)};
  var KEYS = Object.keys(CATS);
  var active = KEYS[0];
  var selected = {json.dumps(current_list)};

  var tabsEl  = document.getElementById('catTabs');
  var pillsEl = document.getElementById('pillsWrap');

  function renderTabs() {{
    tabsEl.innerHTML = '';
    KEYS.forEach(function(k) {{
      var b = document.createElement('button');
      b.className = 'cat-tab' + (k === active ? ' active' : '');
      b.textContent = k;
      b.onclick = function() {{ active = k; renderTabs(); renderPills(); }};
      tabsEl.appendChild(b);
    }});
  }}

  function renderPills() {{
    pillsEl.innerHTML = '';
    (CATS[active] || []).forEach(function(item) {{
      var en = item.en;
      var isSel = selected.indexOf(en.toLowerCase()) !== -1;
      var p = document.createElement('button');
      p.className = 'pill' + (isSel ? ' selected' : '');
      p.textContent = isSel ? '✓ ' + item.display : item.display;
      p.onclick = function() {{ toggleSym(en); }};
      pillsEl.appendChild(p);
    }});
  }}

  function toggleSym(sym) {{
    var lo = sym.toLowerCase();
    var idx = selected.indexOf(lo);
    if (idx === -1) {{ selected.push(lo); }} 
    else {{ selected.splice(idx, 1); }}
    renderPills();
    
    // Dispatch text value straight into parent layout document securely
    var ta = window.parent.document.querySelector('textarea[aria-label="Symptoms Input Field Area:"]');
    if (ta) {{
      ta.value = selected.join(', ');
      ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
    }}
  }}

  renderTabs();
  renderPills();
</script>
</body>
</html>
"""
    components.html(html_code, height=210, scrolling=False)


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
            proba = model.predict_proba(feature_vector)[0]
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
    preds       = [p for p in preds if p["model"] == "SVC"]

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
    description_map, diets_map, medications_map, precautions_map, workout_map
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return None, msg

    top_key = clean_disease_name(disease_name)
    recs, advice = role_based_recs(
        role, lang, top_key,
        description_map, diets_map, medications_map, precautions_map, workout_map,
    )

    return {
        "top_disease":     disease_name.title(),
        "recommendations": recs,
        "advice":          advice
    }, ""


# ──────────────────────────────────────────────
# PARAMETERS RESETS
# ──────────────────────────────────────────────
def clear_symptoms_callback():
    st.session_state["symptoms_text"] = ""
    st.session_state["prediction_result"] = None
    st.session_state["prediction_error"] = None


def clear_diagnosis_callback():
    st.session_state["prediction_result"] = None
    st.session_state["prediction_error"] = None


# ──────────────────────────────────────────────
# APP MAIN ENGINE
# ──────────────────────────────────────────────
def main():
    missing = check_files()
    if missing:
        st.error(f"Missing required execution dependencies: {missing}")
        return

    main_df, desc_map, diets_map, meds_map, precs_map, workout_map = load_data()
    svc, dt, le = load_models()
    
    symptom_list = tuple(main_df.drop(columns=["diseases"]).columns)
    disease_names = tuple(main_df["diseases"].unique())
    vec, sym_matrix, _ = build_tfidf_index(symptom_list, disease_names)

    # ── Sidebar Layout Config
    st.sidebar.markdown(f'<div class="section-header">User Profile Context</div>', unsafe_allow_html=True)
    lang = st.sidebar.selectbox("Interface Language / ቋንቋ", ["English", "Amharic"])
    role = st.sidebar.selectbox("Security Context Role", ["Normal User", "Student", "Doctor"])
    
    user_id = ""
    if role in ["Student", "Doctor"]:
        user_id = st.sidebar.text_input("Role Authorization Token ID", type="password")
        
    age = st.sidebar.number_input("Biological Evaluation Age", min_value=0, max_value=120, value=25)

    # Initialize Tracking States Safely
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None
    if "prediction_error" not in st.session_state:
        st.session_state["prediction_error"] = None
    if "symptoms_text" not in st.session_state:
        st.session_state["symptoms_text"] = ""

    # ── Main Header
    st.markdown(f"""
    <div class="main-header">
        <h1 class="main-header-title">Integrated Healthcare Dashboard</h1>
        <p class="main-header-subtitle">Intelligent Core ML Engine Diagnosis Hub</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([t("Disease Predictor", lang), t("Health Recommender", lang), t("Healthcare Chatbot", lang)])

    # TAB 1: DISEASE PREDICTOR INTERACTION
    with tab1:
        st.markdown(f'<div class="section-header">{t("Disease Predictor", lang)}</div>', unsafe_allow_html=True)
        
        # Render custom selection component directly
        render_quick_select_symptoms(lang)

        # Render Text Area directly bound to state object layout
        user_input = st.text_area(
            "Symptoms Input Field Area:",
            key="symptoms_text",
            placeholder="e.g., headache, fever, chills"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(t("Predict & Recommend", lang)):
                raw_text_value = st.session_state.get("symptoms_text", "")
                if not isinstance(raw_text_value, str):
                    raw_text_value = ""
                raw_text_value = raw_text_value.strip()
                
                if not raw_text_value:
                    st.warning(t("Please enter symptoms.", lang))
                    st.session_state["prediction_result"] = None
                    st.session_state["prediction_error"] = None
                else:
                    res, err_msg = integrated_prediction_system(
                        raw_text_value, age, role, user_id, lang,
                        main_df, le, svc, dt,
                        desc_map, diets_map, meds_map, precs_map, workout_map,
                        vec, sym_matrix
                    )
                    st.session_state["prediction_result"] = res
                    st.session_state["prediction_error"] = err_msg

        with col2:
            st.markdown('<div class="clear-btn-container">', unsafe_allow_html=True)
            st.button(
                t("Clear Symptoms", lang), 
                key="clear_symptoms_btn", 
                on_click=clear_symptoms_callback
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Diagnosis Display Workspace
        if st.session_state["prediction_error"]:
            st.markdown(f'<div class="access-denied">{st.session_state["prediction_error"]}</div>', unsafe_allow_html=True)
            
        elif st.session_state["prediction_result"]:
            res = st.session_state["prediction_result"]
            
            st.markdown("---")
            st.markdown("### System Diagnosis Results")
            
            st.markdown(f'**Matched Symptoms Context:** {", ".join(res["matched_symptoms"])}')
            
            badge_html = ""
            for p in res["predicted_conditions"]:
                badge_html += f'<div class="disease-badge">{p["disease"]} <span class="conf-pill">{p["confidence"]}</span></div>'
            st.markdown(badge_html, unsafe_allow_html=True)
            
            st.markdown("---")
            for title, content in res["recommendations"].items():
                st.markdown(f"""
                <div class="result-card">
                    <h4>{title}</h4>
                    <p>{content if not isinstance(content, list) else ", ".join(content)}</p>
                </div>
                """, unsafe_allow_html=True)
                
            if res["advice"]:
                st.markdown(f'<div class="advice-banner">{res["advice"]}</div>', unsafe_allow_html=True)
                st.caption(f'_{t("medical_advice_disclaimer", lang)}_')
                
            st.markdown('<div class="clear-btn-container" style="margin-top: 20px; max-width: 200px;">', unsafe_allow_html=True)
            st.button(
                t("Clear Diagnosis", lang), 
                key="clear_diagnosis_btn", 
                on_click=clear_diagnosis_callback
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: EXPLICIT ADVICE INTERACTION ROUTER
    with tab2:
        st.markdown(f'<div class="section-header">{t("Health Recommender", lang)}</div>', unsafe_allow_html=True)
        sorted_diseases = sorted([d.title() for d in disease_names])
        selected_disease = st.selectbox("Select Target Condition Profile Database:", sorted_diseases)
        
        if st.button(t("Get Plan", lang)):
            res, err_msg = health_recommender(
                selected_disease, age, role, user_id, lang,
                desc_map, diets_map, meds_map, precs_map, workout_map
            )
            if err_msg:
                st.markdown(f'<div class="access-denied">{err_msg}</div>', unsafe_allow_html=True)
            elif res:
                st.markdown(f"### Treatment Outline Matrix: {res['top_disease']}")
                for title, content in res["recommendations"].items():
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>{title}</h4>
                        <p>{content if not isinstance(content, list) else ", ".join(content)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                if res["advice"]:
                    st.markdown(f'<div class="advice-banner">{res["advice"]}</div>', unsafe_allow_html=True)

    # TAB 3: NLP KNOWLEDGE ENHANCED SIMULATOR 
    with tab3:
        st.markdown(f'<div class="section-header">{t("Healthcare Chatbot", lang)}</div>', unsafe_allow_html=True)
        disease_descriptions = [desc_map.get(clean_disease_name(d), "") for d in disease_names]
        
        bot_query = st.text_input(
            "Inquire regarding specific biological conditions, treatments, or symptoms:"
        )
        
        if st.button(t("Ask Bot", lang)):
            if not bot_query.strip():
                st.info(t("Please enter a query.", lang))
            else:
                ok, err_msg = check_access(age, role, user_id, lang)
                if not ok:
                    st.markdown(f'<div class="access-denied">{err_msg}</div>', unsafe_allow_html=True)
                else:
                    q_vec = vec.transform([bot_query.lower()]).toarray()
                    s_texts = [s.replace("_", " ") for s in symptom_list]
                    d_texts = [d.replace("_", " ") for d in disease_names]
                    
                    s_matrix = vec.transform(s_texts).toarray()
                    d_matrix = vec.transform(d_texts).toarray()
                    desc_matrix = vec.transform(disease_descriptions).toarray()
                    
                    sim_s = cosine_similarity(q_vec, s_matrix)[0]
                    sim_d = cosine_similarity(q_vec, d_matrix)[0]
                    sim_desc = cosine_similarity(q_vec, desc_matrix)[0]
                    
                    best_s = np.argmax(sim_s)
                    best_d = np.argmax(sim_d)
                    best_desc = np.argmax(sim_desc)
                    
                    max_val = max(sim_s[best_s], sim_d[best_d], sim_desc[best_desc])
                    
                    if max_val < 0.2:
                        ans = "I apologize, but I am unable to confidently resolve your query relative to our verified parameters. Could you elaborate on specific indicators?"
                    else:
                        if max_val == sim_s[best_s]:
                            target_sym = symptom_list[best_s].replace("_", " ")
                            ans = f"Your query closely correlates with the cataloged symptom framework for **{target_sym}**. This observation signature can point to multi-tier profile types inside the core disease matrix diagnostics tool."
                        elif max_val == sim_d[best_d]:
                            target_dis = disease_names[best_d]
                            ans = f"Your inquiry targets the diagnostic profile for **{target_dis}**. {desc_map.get(clean_disease_name(target_dis), 'No comprehensive summary available for this profile entry.')}"
                        else:
                            target_dis = disease_names[best_desc]
                            ans = f"In reviewing treatment knowledge spaces for structural context: {disease_descriptions[best_desc]}"
                            
                    translated_ans = translate_content(ans, lang)
                    st.markdown(f'<div class="chat-bot">🤖 <b>Bot Response:</b><br/>{translated_ans}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
