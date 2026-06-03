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
   These must live in the global block so they win
   over the green .stButton rule above.
   Target: any button whose key starts with
   "sym__" or "cat_btn_" (set via the key= arg,
   which Streamlit exposes as data-testid on the
   wrapping div and as aria-label on the button).
   We use [data-testid] on the wrapper + descendant
   button as the scope — fully DOM-position-independent.
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
            t("Dietary Plan", lang):      diet,
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
            t("Dietary Plan", lang):      diet,
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
            t("Dietary Plan", lang):      diet_limited,
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
# ──────────────────────────────────────────────
# QUICK-SELECT SYMPTOM CHIPS
# ──────────────────────────────────────────────
def render_quick_select(categorized_symptoms: dict):
    cats = list(categorized_symptoms.keys())

    if 'active_cat' not in st.session_state or st.session_state.active_cat not in cats:
        st.session_state.active_cat = cats[0]
    if 'symptoms_selected' not in st.session_state:
        st.session_state.symptoms_selected = []

    active_cat = st.session_state.active_cat
    symptoms   = sorted(categorized_symptoms.get(active_cat, []))

    # Build per-render override CSS (active tab + selected chips)
    css_rules = []
    for i, cat in enumerate(cats):
        if cat == active_cat:
            sel = f'[data-testid="cat_btn_{i}"] button, div[data-testid="cat_btn_{i}"] button'
            css_rules.append(
                f'{sel} {{'
                ' background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;'
                ' color: #ffffff !important;'
                ' border-color: #14b8a6 !important;'
                ' font-weight: 600 !important;'
                ' box-shadow: 0 2px 8px rgba(13, 148, 136, 0.3) !important;}'
            )
    for sym in symptoms:
        if sym in st.session_state.symptoms_selected:
            key = f'sym__{active_cat}__{sym}'
            sel = f'[data-testid="{key}"] button, div[data-testid="{key}"] button'
            css_rules.append(
                f'{sel} {{'
                ' background: rgba(20, 184, 166, 0.18) !important;'
                ' color: #2dd4bf !important;'
                ' border-color: #2dd4bf !important;'
                ' font-weight: 600 !important;'
                ' box-shadow: 0 0 0 1px #2dd4bf !important;}'
            )
    if css_rules:
        st.markdown('<style>' + ' '.join(css_rules) + '</style>', unsafe_allow_html=True)

    # Fluid flexbox container styling for modern chip layout wrapping
    st.markdown("""
<style>
.flex-chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 4px 0;
}
.flex-chip-container > div {
    flex: 0 1 auto !important;
    width: auto !important;
}
.flex-chip-container button {
    padding: 6px 16px !important;
    border-radius: 100px !important;
    font-size: 0.82rem !important;
    width: auto !important;
}
.quick-select-panel {
    background: rgba(15, 23, 42, 0.35);
    border: 1px solid rgba(48, 54, 61, 0.4);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 15px;
}
</style>""", unsafe_allow_html=True)

    # Main Visual Panel Wrapper
    st.markdown("<div class='quick-select-panel'>", unsafe_allow_html=True)

    # Category Tab Row (Fluid Wrap Layout)
    st.markdown("<div class='flex-chip-container'>", unsafe_allow_html=True)
    for i, cat in enumerate(cats):
        st.markdown("<div>", unsafe_allow_html=True)
        if st.button(cat, key=f'cat_btn_{i}'):
            st.session_state.active_cat = cat
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Visual Separator Divider Line
    st.markdown(
        "<div style='border-top:1px solid rgba(48, 54, 61, 0.3); margin:12px 0 10px 0;'></div>",
        unsafe_allow_html=True,
    )

    # Symptom Chip Row (Fluid Wrap Layout)
    st.markdown("<div class='flex-chip-container'>", unsafe_allow_html=True)
    for sym in symptoms:
        label   = sym.replace('_', ' ').title()
        is_sel  = sym in st.session_state.symptoms_selected
        lbl     = f'✓ {label}' if is_sel else label
        btn_key = f'sym__{active_cat}__{sym}'
        
        st.markdown("<div>", unsafe_allow_html=True)
        if st.button(lbl, key=btn_key):
            if sym in st.session_state.symptoms_selected:
                st.session_state.symptoms_selected.remove(sym)
            else:
                st.session_state.symptoms_selected.append(sym)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Close Visual Panel Wrapper
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN APP ENTRYWAY
# ──────────────────────────────────────────────
def main():
    missing = check_files()
    if missing:
        st.error(f"Missing required execution dependencies: {missing}")
        st.info("Ensure files exist in data/ and models/ matching your training structure.")
        return

    # Pipeline initialization
    main_df, desc_map, diets_map, meds_map, precs_map, workout_map = load_data()
    svc_model, dt_model, le_encoder = load_models()

    symptom_list = list(main_df.drop(columns=["diseases"]).columns)
    disease_list = sorted(list(main_df["diseases"].unique()))
    disease_keys = tuple(clean_disease_name(d) for d in disease_list)

    tfidf_vec, sym_matrix, dis_matrix = build_tfidf_index(tuple(symptom_list), disease_keys)
    categorized = categorize_symptoms(symptom_list)

    # Sidebar parameters
    st.sidebar.markdown(f"<div class='section-header'>🔒 Verification Engine</div>", unsafe_allow_html=True)
    role = st.sidebar.selectbox("Access Profile Tier", ["Normal User", "Student", "Doctor"])
    user_id = st.sidebar.text_input("Profile Verification Key ID", value="", type="password")
    age = st.sidebar.number_input("Target User Age Group", min_value=0, max_value=120, value=25)

    st.sidebar.markdown("<br><hr>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='section-header'>🌐 Localization Pack</div>", unsafe_allow_html=True)
    lang = st.sidebar.radio("Interface Localization Variant", ["English", "Amharic"])

    # Application identity layout
    st.markdown(
        f"<div class='main-header'>"
        f"<div class='main-header-title'>🏥 Integrated Healthcare Dashboard</div>"
        f"<div class='main-header-subtitle'>AI Diagnostic Inference • Semantic Extraction Recommender</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs([t("Disease Predictor", lang), t("Health Recommender", lang), t("Healthcare Chatbot", lang)])

    # TAB 1: PREDICTOR
    with tab1:
        st.markdown(f"<div class='section-header'>🔬 Cognitive Symptom Diagnostic Engine</div>", unsafe_allow_html=True)
        
        # Interactive chip layout module
        render_quick_select(categorized)

        # Dynamic query box state matching loop
        current_input = st.text_input(
            "Target Manifestation Manifest (Comma Delimited Evaluator)",
            value=", ".join(st.session_state.symptoms_selected).replace("_", " "),
            placeholder="Type symptoms (e.g. fever, headache) or toggle using the quick selection helper panel above.",
        )

        if st.button(t("Predict & Recommend", lang), key="btn_predict"):
            if not current_input.strip():
                st.warning(t("Please enter symptoms.", lang))
            else:
                res, err = integrated_prediction_system(
                    current_input, age, role, user_id, lang,
                    main_df, le_encoder, svc_model, dt_model,
                    desc_map, diets_map, meds_map, precs_map, workout_map,
                    tfidf_vec, sym_matrix
                )
                if err:
                    st.markdown(f"<div class='access-denied'>{err}</div>", unsafe_allow_html=True)
                elif res:
                    st.success(f"Extracted Analytical Matrix: {', '.join(res['matched_symptoms'])}")
                    
                    c1, c2 = st.columns([1, 1.2])
                    with c1:
                        render_predictions(res["predicted_conditions"])
                    with c2:
                        st.markdown(
                            f"<div>"
                            f"<span class='disease-badge'>{res['top_disease']}</span>"
                            f"<span class='conf-pill'>Selected Primary Core Route</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        render_recommendations(res["recommendations"])
                        if res["advice"]:
                            st.markdown(f"<div class='advice-banner'>{res['advice']}</div>", unsafe_allow_html=True)

    # TAB 2: RECOMMENDER
    with tab2:
        st.markdown(f"<div class='section-header'>📑 Disease Regimen Rule Database lookup</div>", unsafe_allow_html=True)
        selected_disease = st.selectbox("Select Target Condition Profile", [d.title() for d in disease_list])
        
        if st.button(t("Get Plan", lang), key="btn_get_plan"):
            recs, err = health_recommender(
                selected_disease, age, role, user_id, lang,
                desc_map, diets_map, meds_map, precs_map, workout_map
            )
            if err:
                st.markdown(f"<div class='access-denied'>{err}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3>Regimen Strategy Map: {selected_disease}</h3>", unsafe_allow_html=True)
                render_recommendations(recs)

    # TAB 3: CHATBOT
    with tab3:
        st.markdown(f"<div class='section-header'>💬 Semantic Virtual Assistant Agent Interface</div>", unsafe_allow_html=True)
        st.info(t("Hello! How can I help you with health information today?", lang))
        
        chat_query = st.text_input("Enter natural-language diagnostic query parameter", placeholder="e.g. What structural foods are ideal for treating Malaria?")
        
        if st.button(t("Ask Bot", lang), key="btn_ask_bot"):
            if not chat_query.strip():
                st.warning(t("Please enter a query.", lang))
            else:
                reply = chatbot_response(
                    chat_query, age, role, user_id, lang,
                    desc_map, diets_map, meds_map, precs_map, workout_map,
                    disease_keys, tfidf_vec, dis_matrix
                )
                st.markdown(f"<div class='chat-bot'>{reply}</div>", unsafe_allow_html=True)

    st.markdown("<br><br><hr><div style='text-align:center;color:#8b949e;font-size:0.8rem;'>"
                "Core Processing Node Operational Tier Matrix. For emergency workflows, please execute standard protocols.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
