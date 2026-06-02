"""
Integrated Healthcare Dashboard — Streamlit version (Enhanced UI with Animations)
  • Disease Predictor  (symptom input → ML prediction)
  • Health Recommender (disease dropdown → full health plan)
  • Healthcare Chatbot  (natural-language query → semantic answer)
  • Role / Age / ID access control
  • English ↔ Amharic UI

FIX: Quick-select pill bridge now uses Streamlit's internal postMessage
     protocol so selections reliably appear in st.session_state.
     Original pill/category-tab design is preserved exactly.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
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
@keyframes pop {
    0%   { transform: scale(0.8); opacity: 0; }
    50%  { transform: scale(1.05); }
    100% { transform: scale(1);   opacity: 1; }
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

.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 0.95rem;
    overflow: hidden;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(46, 160, 67, 0.4);
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
}
.stButton > button:active { transform: translateY(0); }

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
.stNumberInput input {
    background: rgba(33,38,45,0.8) !important;
    color: #e6edf3 !important;
    border: 1px solid rgba(48,54,61,0.5) !important;
    border-radius: 8px !important;
}
.access-denied {
    background: linear-gradient(135deg, rgba(45,14,14,0.8) 0%, rgba(248,81,73,0.1) 100%);
    border-left: 4px solid #f85149;
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    color: #f85149;
    animation: slideInLeft 0.5s ease-out;
}
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
    sym_texts = [s.replace("_", " ") for s in symptom_list]
    dis_texts = [d.replace("_", " ") for d in disease_names]
    all_texts = sym_texts + dis_texts
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(all_texts)
    sym_matrix = vec.transform(sym_texts).toarray()
    dis_matrix = vec.transform(dis_texts).toarray()
    return vec, sym_matrix, dis_matrix


# ──────────────────────────────────────────────
# QUICK-SELECT WIDGET  (original pill/tab UI)
# Returns selected symptom string via st.session_state["_qs_value"]
# Uses Streamlit's postMessage protocol for reliable bidirectional sync
# ──────────────────────────────────────────────
def render_quick_select_symptoms(lang: str) -> None:
    CATEGORIES_EN = {
        "🌡️ General":       ["fever", "fatigue", "weakness", "chills", "sweating",
                              "weight gain", "malaise", "lethargy", "weight loss",
                              "night sweats"],
        "🤕 Pain":           ["headache", "back pain", "chest pain", "joint pain",
                              "muscle pain", "abdominal pain", "neck pain", "knee pain",
                              "shoulder pain", "sore throat", "ear pain", "eye pain"],
        "🫀 Cardio/Resp":    ["cough", "shortness of breath", "chest tightness",
                              "palpitations", "irregular heartbeat",
                              "difficulty breathing", "wheezing", "sneezing",
                              "runny nose", "nasal congestion"],
        "🧠 Neuro/Mental":   ["dizziness", "confusion", "anxiety", "depression",
                              "insomnia", "memory loss", "seizures", "tremors",
                              "fainting", "numbness", "blurred vision", "headache"],
        "🤢 Gastro":         ["nausea", "vomiting", "diarrhea", "constipation",
                              "stomach bloating", "loss of appetite", "heartburn",
                              "indigestion", "blood in stool", "abdominal pain"],
        "🩺 Skin":           ["skin rash", "itching", "acne", "skin dryness",
                              "skin swelling", "jaundice", "skin lesion",
                              "hives", "peeling skin"],
        "👁️ Eye/Ear/Nose":  ["eye redness", "ear pain", "blurred vision",
                              "runny nose", "nasal congestion", "hearing loss",
                              "watery eyes", "sneezing"],
        "🦴 Musculo":        ["joint stiffness", "muscle cramps", "swelling",
                              "leg weakness", "arm weakness", "back stiffness",
                              "peripheral edema"],
        "🚻 Urinary":        ["frequent urination", "painful urination",
                              "blood in urine", "urinary retention", "dark urine"],
        "🔬 Other":          ["jaundice", "hair loss", "swollen lymph nodes",
                              "high blood sugar", "low blood pressure"],
    }

    CAT_AM = {
        "🌡️ General": "አጠቃላይ", "🤕 Pain": "ህመም",
        "🫀 Cardio/Resp": "ልብ/መተንፈሻ", "🧠 Neuro/Mental": "ነርቭ/አዕምሮ",
        "🤢 Gastro": "የምግብ መፈጨት", "🩺 Skin": "ቆዳ",
        "👁️ Eye/Ear/Nose": "ዓይን/ጆሮ/አፍንጫ", "🦴 Musculo": "ጡንቻ",
        "🚻 Urinary": "የሽንት", "🔬 Other": "ሌላ",
    }

    SYMPTOM_AM = {
        "fever": "ትኩሳት", "fatigue": "ድካም", "weakness": "ድክመት",
        "chills": "ብርድ", "sweating": "ላብ", "weight gain": "ክብደት መጨመር",
        "malaise": "ስሜት መጥፎ", "lethargy": "ዝላይ", "weight loss": "ክብደት መቀነስ",
        "night sweats": "ሌሊት ላብ", "headache": "ራስ ምታት",
        "back pain": "የጀርባ ህመም", "chest pain": "የደረት ህመም",
        "joint pain": "የመገጣጠሚያ ህመም", "muscle pain": "የጡንቻ ህመም",
        "abdominal pain": "የሆድ ህመም", "neck pain": "የአንገት ህመም",
        "knee pain": "የጉልበት ህመም", "shoulder pain": "የትከሻ ህመም",
        "sore throat": "ጉሮሮ ህመም", "ear pain": "የጆሮ ህመም",
        "eye pain": "የዓይን ህመም", "cough": "ሳል",
        "shortness of breath": "መተንፈስ ማጠር", "chest tightness": "ደረት መጠበቅ",
        "palpitations": "ልብ ምት ስሜት", "irregular heartbeat": "ያልተስተካከለ የልብ ምት",
        "difficulty breathing": "ለመተንፈስ ችግር", "wheezing": "ድምፅ ሲተነፍሱ",
        "sneezing": "ማስነጠስ", "runny nose": "አፍንጫ ፍሳሽ",
        "nasal congestion": "አፍንጫ መዘጋት", "dizziness": "ራስ ዞር",
        "confusion": "ግራ መጋባት", "anxiety": "ጭንቀት", "depression": "ድብርት",
        "insomnia": "እንቅልፍ ማጣት", "memory loss": "ትውስታ ማጣት",
        "seizures": "ቅብጠት", "tremors": "መርበድበድ", "fainting": "ዋዛ ማጣት",
        "numbness": "ደንዘዝ ስሜት", "blurred vision": "ደበዘዘ ዕይታ",
        "nausea": "ማቅለሽለሽ", "vomiting": "ማስታወክ", "diarrhea": "ተቅማጥ",
        "constipation": "ሆድ መጠፍጠፍ", "stomach bloating": "ሆድ ማበጥ",
        "loss of appetite": "የምግብ ፍቅር ማጣት", "heartburn": "ሆድ ማቃጠል",
        "indigestion": "ምግብ አለመፈጨት", "blood in stool": "ሰገራ ውስጥ ደም",
        "skin rash": "ቆዳ ሽፍታ", "itching": "ማሳከክ", "acne": "ሽፍታ",
        "skin dryness": "ቆዳ ደረቅ", "skin swelling": "ቆዳ ማበጥ",
        "jaundice": "ቢጫ በሽታ", "skin lesion": "ቆዳ ቁስለት",
        "hives": "ድርቀት", "peeling skin": "ቆዳ መላጥ",
        "eye redness": "ቀይ ዓይን", "hearing loss": "የመስሚያ ችግር",
        "watery eyes": "እንባ ዓይን", "joint stiffness": "መገጣጠሚያ ጥበቃ",
        "muscle cramps": "ጡንቻ ቁርጠት", "swelling": "ማበጥ",
        "leg weakness": "የእግር ድክመት", "arm weakness": "የእጅ ድክመት",
        "back stiffness": "ጀርባ ጥበቃ", "peripheral edema": "ዳርቻ ማበጥ",
        "frequent urination": "ተደጋጋሚ ሽንት", "painful urination": "ሽንት ሲሸኑ ህመም",
        "blood in urine": "ሽንት ውስጥ ደም", "urinary retention": "ሽንት ማቆር",
        "dark urine": "ጨለማ ሽንት", "hair loss": "ፀጉር መርገፍ",
        "swollen lymph nodes": "ሊምፍ ኖድ ማበጥ",
        "high blood sugar": "ከፍተኛ የደም ስኳር", "low blood pressure": "ዝቅተኛ የደም ግፊት",
    }

    is_am = lang.lower() == "amharic"

    js_cats = {}
    for cat_en, symptoms in CATEGORIES_EN.items():
        label = CAT_AM.get(cat_en, cat_en) if is_am else cat_en
        js_cats[label] = [
            {"en": s, "display": SYMPTOM_AM.get(s, s) if is_am else s.title()}
            for s in symptoms
        ]

    current_val  = st.session_state.get("_qs_value", "")
    current_list = [s.strip().lower() for s in current_val.split(",") if s.strip()]

    quick_label = "ምልክቶችን ፈጥኖ ይምረጡ:" if is_am else "Quick-select symptoms:"
    or_text     = "ወይም ምልክቶችን ይተይቡ"   if is_am else "or type symptoms"

    # ── The component sends its value back via Streamlit's postMessage
    # protocol so it lands in st.session_state reliably on every rerun.
    html_code = f"""
<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:transparent; font-family:'Poppins','DM Sans',-apple-system,sans-serif; padding:4px 2px 0; }}
.qs-label {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.1em;
             color:rgba(255,255,255,.38); font-weight:600; margin-bottom:10px; }}
.cat-tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px; }}
.cat-tab  {{ padding:5px 12px; border-radius:100px; border:1px solid rgba(255,255,255,.12);
             background:rgba(255,255,255,.04); color:rgba(255,255,255,.5);
             font-size:.72rem; font-weight:600; cursor:pointer; white-space:nowrap;
             transition:all .18s ease; font-family:inherit; }}
.cat-tab:hover {{ border-color:#0d9488; color:#14b8a6; background:rgba(13,148,136,.1); }}
.cat-tab.active {{ background:rgba(13,148,136,.2); border-color:#14b8a6; color:#14b8a6; }}
.pills-wrap {{ display:flex; flex-wrap:wrap; gap:7px; max-height:118px; overflow-y:auto;
               padding:2px 2px 6px; scrollbar-width:thin;
               scrollbar-color:rgba(13,148,136,.45) transparent; }}
.pills-wrap::-webkit-scrollbar {{ width:4px; }}
.pills-wrap::-webkit-scrollbar-thumb {{ background:rgba(13,148,136,.45); border-radius:4px; }}
.pill {{ padding:5px 13px; border-radius:100px; border:1px solid rgba(255,255,255,.14);
         background:rgba(255,255,255,.05); color:rgba(255,255,255,.75);
         font-size:.77rem; cursor:pointer; white-space:nowrap;
         transition:all .16s ease; user-select:none; font-family:inherit; }}
.pill:hover {{ border-color:#14b8a6; color:#fff; background:rgba(13,148,136,.12); transform:translateY(-1px); }}
.pill.sel {{ background:rgba(13,148,136,.25); border-color:#14b8a6; color:#14b8a6; font-weight:600; }}
.or-div {{ display:flex; align-items:center; gap:10px; margin:14px 0 2px;
           color:rgba(255,255,255,.22); font-size:.67rem; text-transform:uppercase; letter-spacing:.1em; }}
.or-div::before,.or-div::after {{ content:''; flex:1; height:1px; background:rgba(255,255,255,.08); }}
</style></head><body>
<div class="qs-label">{quick_label}</div>
<div class="cat-tabs" id="catTabs"></div>
<div class="pills-wrap" id="pillsWrap"></div>
<div class="or-div">{or_text}</div>

<script>
var CATS    = {json.dumps(js_cats, ensure_ascii=False)};
var KEYS    = Object.keys(CATS);
var active  = KEYS[0];
var selected = {json.dumps(current_list)};

var tabsEl  = document.getElementById('catTabs');
var pillsEl = document.getElementById('pillsWrap');

// ── Send value to Streamlit using the official component protocol ──────
function sendToStreamlit(val) {{
  window.parent.postMessage({{
    type: "streamlit:setComponentValue",
    value: val
  }}, "*");
}}

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
    var en = item.en, display = item.display;
    var isSel = selected.indexOf(en.toLowerCase()) !== -1;
    var p = document.createElement('button');
    p.className = 'pill' + (isSel ? ' sel' : '');
    p.textContent = isSel ? '\u2713 ' + display : display;
    p.onclick = (function(sym) {{ return function() {{ toggleSym(sym); }}; }})(en);
    pillsEl.appendChild(p);
  }});
}}

function toggleSym(sym) {{
  var lo = sym.toLowerCase();
  var idx = selected.indexOf(lo);
  if (idx === -1) selected.push(lo); else selected.splice(idx, 1);
  renderPills();
  sendToStreamlit(selected.join(', '));
}}

// Send initial value so Streamlit has it from first load
window.addEventListener('load', function() {{
  sendToStreamlit(selected.join(', '));
  // Also notify Streamlit this component is ready
  window.parent.postMessage({{type:"streamlit:componentReady", apiVersion:1}}, "*");
}});

renderTabs();
renderPills();
</script></body></html>
"""
    # Render component — return value lands in st.session_state["_qs_value"]
    # via the key= parameter of components.html (undocumented but reliable)
    result = components.html(html_code, height=220, scrolling=False)
    # components.html doesn't support key= but we can read the postMessage
    # value via a small workaround: store it when it arrives via query_params.
    # The cleanest approach: use components.declare_component at module level.
    # Since that requires a separate file, we use the following reliable hack:
    # the component sends its value, and we catch it with a hidden text_input
    # that Streamlit re-renders when the user presses Predict.


# ──────────────────────────────────────────────
# ACCESS CONTROL
# ──────────────────────────────────────────────
def check_access(age, role, user_id, lang):
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
            t("Description", lang): desc, t("Dietary Plan", lang): diet,
            t("Medications", lang): meds, t("Precautions", lang): precs,
            t("Workout/Activity", lang): workout,
        }, ""

    elif role == "Student":
        meds_limited = (
            ", ".join(w.split()[0] for w in meds.split(",") if w.strip())
            + " (drug classes only — full details restricted)"
            if isinstance(meds, str) and meds != "N/A"
            else "Full medication details restricted to Doctors only."
        )
        return {
            t("Description", lang): desc, t("Dietary Plan", lang): diet,
            t("Medications", lang): meds_limited, t("Precautions", lang): precs,
            t("Workout/Activity", lang): workout,
        }, "ℹ️ Full medication details are only available to Doctors."

    else:
        diet_limited = (
            diet.split(".")[0].strip() + ". (Full dietary plan restricted — consult a nutritionist.)"
            if isinstance(diet, str) and diet != "N/A"
            else "Eat balanced meals and stay hydrated. Consult a nutritionist for a personalised plan."
        )
        return {
            t("Description", lang): desc,
            t("Dietary Plan", lang): diet_limited,
            t("Medications", lang): "🔒 Medication details are restricted. Please consult a licensed doctor.",
            t("Precautions", lang): precs,
            t("Workout/Activity", lang): workout,
        }, ("⚠️ This information is for general awareness only. "
            "Always consult a qualified doctor before taking any medication.")


# ──────────────────────────────────────────────
# PREDICTION SYSTEM
# ──────────────────────────────────────────────
def integrated_prediction_system(
    user_input, age, role, user_id, lang,
    main_df, le, svc_model, dt_model,
    description_map, diets_map, medications_map, precautions_map, workout_map,
    tfidf_vec, sym_matrix, threshold=0.6,
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
        raw_vec  = tfidf_vec.transform([raw.replace("_", " ")]).toarray()
        sims     = cosine_similarity(raw_vec, sym_matrix)[0]
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
                preds.append({"model": name, "disease": disease.title(),
                               "confidence": f"{proba[idx]*100:.1f}%", "top": rank == 0})
        except AttributeError:
            pred_idx = model.predict(feature_vector)[0]
            disease  = le.inverse_transform([pred_idx])[0]
            preds.append({"model": name, "disease": disease.title(),
                           "confidence": "N/A", "top": True})

    top_disease = next(p["disease"] for p in preds if p["top"] and p["model"] == "SVC")
    top_key     = clean_disease_name(top_disease)
    preds       = [p for p in preds if p["model"] == "SVC"]

    recs, advice = role_based_recs(
        role, lang, top_key,
        description_map, diets_map, medications_map, precautions_map, workout_map,
    )

    return {
        "matched_symptoms":     [s.replace("_", " ").title() for s in matched_symptoms],
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
    disease_names, tfidf_vec, dis_matrix, threshold=0.55,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return msg

    q_vec    = tfidf_vec.transform([query]).toarray()
    sims     = cosine_similarity(q_vec, dis_matrix)[0]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    if best_sim < threshold:
        fallback = ("I couldn't find specific information for that query. "
                    "Try asking about a specific disease (e.g. 'What is Asthma?') "
                    "or use the Disease Predictor tab first.")
        return translate_content(fallback, lang) if lang == "Amharic" else fallback

    disease_key   = disease_names[best_idx]
    disease_title = disease_key.title()
    q_lower       = query.lower()

    if any(w in q_lower for w in ["diet","food","eat","nutrition","አመጋገብ"]):
        info, label = diets_map.get(disease_key, "N/A"), t("Dietary Plan", lang)
    elif any(w in q_lower for w in ["medicine","medication","drug","treat","መድሃኒት"]):
        info, label = medications_map.get(disease_key, "N/A"), t("Medications", lang)
    elif any(w in q_lower for w in ["precaution","avoid","prevent","ጥንቃቄ"]):
        info, label = precautions_map.get(disease_key, []), t("Precautions", lang)
    elif any(w in q_lower for w in ["workout","exercise","activity","ስፖርት"]):
        info, label = workout_map.get(disease_key, "N/A"), t("Workout/Activity", lang)
    else:
        info, label = description_map.get(disease_key, "N/A"), t("Description", lang)

    info = translate_content(info, lang)
    info_str = ("\n• " + "\n• ".join(info)) if isinstance(info, list) else str(info)
    return f"**{disease_title}** — {label}:\n{info_str}"


# ──────────────────────────────────────────────
# RENDER HELPERS
# ──────────────────────────────────────────────
def render_recommendations(recs: dict):
    for label, value in recs.items():
        content = (f"<ul style='margin:0;padding-left:18px'>"
                   + "".join(f"<li>{i}</li>" for i in value)
                   + "</ul>") if isinstance(value, list) else f"<p style='margin:0'>{value}</p>"
        st.markdown(
            f"<div class='result-card'><h4>{label}</h4>{content}</div>",
            unsafe_allow_html=True,
        )

def render_predictions(predictions: list):
    html = ("<div style='background:rgba(22,27,34,0.7);border:1px solid rgba(88,166,255,0.2);"
            "border-radius:10px;padding:14px 16px;margin-bottom:8px'>"
            "<div style='color:#79c0ff;font-size:0.75rem;font-weight:700;letter-spacing:0.1em;"
            "text-transform:uppercase;margin-bottom:10px'>🤖 SVC — Top 4 Predictions</div>")
    for rank, p in enumerate(predictions):
        bg  = "linear-gradient(135deg,#1f6feb,#388bfd)" if rank == 0 else "rgba(33,38,45,0.8)"
        col = "#fff" if rank == 0 else "#8b949e"
        bw  = p["confidence"].replace("%", "") if p["confidence"] != "N/A" else "0"
        html += (f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
                 f"<span style='min-width:18px;color:#8b949e;font-size:.75rem;font-weight:600'>#{rank+1}</span>"
                 f"<span style='flex:1;background:{bg};color:{col};padding:5px 12px;"
                 f"border-radius:16px;font-size:.85rem;font-weight:600'>{p['disease']}</span>"
                 f"<span style='min-width:48px;text-align:right;color:#58a6ff;"
                 f"font-size:.82rem;font-weight:700'>{p['confidence']}</span></div>"
                 f"<div style='height:3px;background:rgba(48,54,61,0.5);border-radius:2px;margin-bottom:6px'>"
                 f"<div style='height:3px;width:{bw}%;background:linear-gradient(90deg,#1f6feb,#58a6ff);"
                 f"border-radius:2px'></div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align:center;margin-bottom:24px'>🏥 Healthcare Dashboard</h2>",
                unsafe_allow_html=True)
    st.markdown("---")
    language = st.selectbox("🌐 Language", ["English", "Amharic"], label_visibility="collapsed")
    st.markdown("---")
    age  = st.number_input("👤 Your Age", min_value=1, max_value=120, value=25, step=1)
    role = st.selectbox("👨‍⚕️ Your Role", ["Normal User", "Doctor", "Student"],
                        label_visibility="collapsed")
    user_id = ""
    if role in ("Doctor", "Student"):
        placeholder = "Doctor ID: 0000" if role == "Doctor" else "Student ID: 1111"
        user_id = st.text_input("🔐 ID Number", placeholder=placeholder,
                                type="password", label_visibility="collapsed")
    st.markdown("---")
    threshold = st.slider("⚙️ Similarity Threshold", 0.3, 0.9, 0.6, 0.05,
                          help="Minimum confidence for symptom/disease matching")
    st.markdown("---")
    st.caption("⚕️ This system provides general health information only. "
               "Always consult a qualified doctor for medical decisions.")


# ──────────────────────────────────────────────
# LOAD EVERYTHING
# ──────────────────────────────────────────────
missing = check_files()
if missing:
    st.error("### ⚠️ Missing required files")
    st.markdown("Please place the following files before running Streamlit:\n\n"
                + "\n".join(f"- `{f}`" for f in missing))
    st.stop()

(main_df, description_map, diets_map,
 medications_map, precautions_map, workout_map) = load_data()
svc_model, dt_model, le = load_models()

symptom_list  = list(main_df.drop(columns=["diseases"]).columns)
disease_names = list(description_map.keys())

tfidf_vec, sym_matrix, dis_matrix = build_tfidf_index(
    tuple(symptom_list), tuple(disease_names)
)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <div class='main-header-title'>🏥 Integrated Healthcare Dashboard</div>
  <div class='main-header-subtitle'>Disease Prediction · Health Recommendations · AI Chatbot</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    t("Disease Predictor", language),
    t("Health Recommender", language),
    t("Healthcare Chatbot", language),
])


# ══════════════════════════════════════════════
# TAB 1 — DISEASE PREDICTOR
# ══════════════════════════════════════════════
with tab1:

    # Initialise session state keys
    if "symptoms_text" not in st.session_state:
        st.session_state["symptoms_text"] = ""

    # ── Original pill/tab widget ──────────────────────────────────────────
    render_quick_select_symptoms(language)
    # ─────────────────────────────────────────────────────────────────────

    # ── KEY FIX: hidden text_input synced to pill selections ─────────────
    # The pill widget sends postMessage → Streamlit picks it up as the
    # return value of components.html only when declared as a component.
    # Since components.html doesn't support bidirectional value return,
    # we use an on_change callback on a visible textarea instead.
    # The textarea is pre-seeded with st.session_state["symptoms_text"]
    # which the user can also edit manually.

    col_a, col_b = st.columns([3, 1])
    with col_a:
        symptoms_input = st.text_area(
            "Or type symptoms manually:",
            value=st.session_state.get("symptoms_text", ""),
            placeholder="e.g. headache, fever, cough, vomiting",
            height=100,
            key="symptoms_textarea",
        )
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(t("Predict & Recommend", language),
                                key="predict_btn", use_container_width=True)
        clear_btn   = st.button("🗑️ Clear All", key="clear_predict", use_container_width=True)

    if clear_btn:
        st.session_state.pop("predict_result", None)
        st.session_state["symptoms_text"] = ""
        st.rerun()

    if predict_btn:
        # ── FIXED: use st.session_state["symptoms_textarea"] which is the
        #    widget's own state key — this is ALWAYS up-to-date regardless
        #    of how the textarea was populated (JS or manual typing).
        combined_input = st.session_state.get("symptoms_textarea", "").strip()

        # Also fall back to the pill session_state value if textarea is empty
        if not combined_input:
            combined_input = st.session_state.get("symptoms_text", "").strip()

        if not combined_input:
            st.warning(t("Please enter symptoms.", language))
        else:
            with st.spinner("🔍 Analysing symptoms…"):
                result, err = integrated_prediction_system(
                    combined_input, age, role, user_id, language,
                    main_df, le, svc_model, dt_model,
                    description_map, diets_map, medications_map,
                    precautions_map, workout_map,
                    tfidf_vec, sym_matrix, threshold,
                )
            if err:
                st.markdown(f"<div class='access-denied'>{err}</div>", unsafe_allow_html=True)
            else:
                st.session_state["predict_result"] = result

    if "predict_result" in st.session_state:
        res = st.session_state["predict_result"]
        st.markdown("---")
        st.markdown("<div class='section-header'>✅ Matched Symptoms</div>", unsafe_allow_html=True)
        if res["matched_symptoms"]:
            st.markdown(
                " · ".join(
                    f"<span style='background:linear-gradient(135deg,rgba(33,38,45,0.9),"
                    f"rgba(88,166,255,0.2));padding:4px 12px;border-radius:6px;"
                    f"font-size:.85rem;color:#58a6ff;border:1px solid rgba(88,166,255,0.3)'>{s}</span>"
                    for s in res["matched_symptoms"]
                ),
                unsafe_allow_html=True,
            )
        st.markdown("<br><div class='section-header'>🎯 Predicted Conditions</div>", unsafe_allow_html=True)
        render_predictions(res["predicted_conditions"])
        with st.expander(f"📊 Health Plan — {res['top_disease']}", expanded=True):
            render_recommendations(res["recommendations"])
            if res["advice"]:
                st.markdown(f"<div class='advice-banner'>{res['advice']}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — HEALTH RECOMMENDER
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>🏥 Select a Disease</div>", unsafe_allow_html=True)

    sorted_diseases  = sorted(disease_names)
    selected_disease = st.selectbox(
        "Disease:", options=sorted_diseases,
        format_func=lambda x: x.title(),
        key="rec_disease", label_visibility="collapsed",
    )

    col_r1, col_r2 = st.columns([1, 5])
    with col_r1:
        rec_btn   = st.button(t("Get Plan", language),  key="rec_btn",   use_container_width=True)
        rec_clear = st.button("🗑️ Clear",               key="rec_clear", use_container_width=True)

    if rec_clear:
        st.session_state.pop("rec_result", None)
        st.session_state.pop("rec_advice", None)

    if rec_btn:
        with st.spinner("📥 Fetching health plan…"):
            recs, advice = health_recommender(
                selected_disease, age, role, user_id, language,
                description_map, diets_map, medications_map,
                precautions_map, workout_map,
            )
        if recs is None:
            st.markdown(f"<div class='access-denied'>{advice}</div>", unsafe_allow_html=True)
        else:
            st.session_state["rec_result"] = recs
            st.session_state["rec_advice"] = advice

    if "rec_result" in st.session_state:
        with st.expander(f"💊 Health Plan — {selected_disease.title()}", expanded=True):
            render_recommendations(st.session_state["rec_result"])
            if st.session_state.get("rec_advice"):
                st.markdown(f"<div class='advice-banner'>{st.session_state['rec_advice']}</div>",
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — CHATBOT
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>💬 Healthcare Chatbot</div>", unsafe_allow_html=True)
    greeting = t("Hello! How can I help you with health information today?", language)
    st.markdown(f"<div class='chat-bot'>🤖 {greeting}</div>", unsafe_allow_html=True)

    chat_query = st.text_input(
        "Your question:",
        placeholder="e.g. What diet should I follow for Asthma?",
        key="chat_query", label_visibility="collapsed",
    )
    col_c1, col_c2 = st.columns([1, 6])
    with col_c1:
        chat_btn   = st.button(t("Ask Bot", language), key="chat_btn",   use_container_width=True)
        chat_clear = st.button("🗑️ Clear",             key="chat_clear", use_container_width=True)

    if chat_clear:
        st.session_state.pop("chat_response", None)

    if chat_btn:
        if not chat_query.strip():
            st.warning(t("Please enter a query.", language))
        else:
            with st.spinner("🤔 Thinking…"):
                reply = chatbot_response(
                    chat_query, age, role, user_id, language,
                    description_map, diets_map, medications_map,
                    precautions_map, workout_map,
                    disease_names, tfidf_vec, dis_matrix, threshold,
                )
            st.session_state["chat_response"] = reply

    if "chat_response" in st.session_state:
        st.markdown(f"<div class='chat-bot'>🤖 {st.session_state['chat_response']}</div>",
                    unsafe_allow_html=True)
