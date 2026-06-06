"""
Integrated Healthcare Dashboard — Streamlit (No-Server Auth Edition)
  • Disease Predictor  (symptom input → ML prediction)
  • Health Recommender (disease dropdown → full health plan)
  • Healthcare Chatbot  (natural-language query → semantic answer)
  • Role / Age access control  ← NO server needed, IDs validated locally
  • English ↔ Amharic UI
  • NLP paragraph symptom extraction
  • Auth managed via companion auth.html — paste valid IDs into VALID_DOCTORS / VALID_STUDENTS below
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# LOCAL AUTH CONFIG
# Paste the IDs you created in auth.html here.
# Format: a Python set of uppercase ID strings.
# ══════════════════════════════════════════════════════════════════════════════
VALID_DOCTORS: set = {
    "DR001",
    # "DR002", "DR003",  ← add more as needed
}
VALID_STUDENTS: set = {
    "ST001",
    # "ST002", "ST003",
}

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Integrated Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Poppins:wght@300;400;600;700&display=swap');

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to   { opacity: 1; transform: translateX(0); }
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    scroll-behavior: smooth;
}
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1f2d 100%);
    color: #e6edf3;
    animation: fadeInUp 0.8s ease-out;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0f1419 100%);
    border-right: 1px solid rgba(48,54,61,0.5);
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(22,27,34,0.6);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid rgba(48,54,61,0.3);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 10px 24px;
    border: none !important;
    transition: all 0.3s ease;
    font-size: 0.95rem;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(88,166,255,0.1); color: #58a6ff; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #21262d 0%, #1f6feb 100%) !important;
    color: #fff !important;
    box-shadow: 0 4px 12px rgba(88,166,255,0.25);
}

.stTextInput input, .stSelectbox select, .stTextArea textarea,
div[data-baseweb="select"] > div {
    background: rgba(33,38,45,0.8) !important;
    border: 1px solid rgba(48,54,61,0.5) !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    font-size: 0.95rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus,
div[data-baseweb="select"] > div:focus-within {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.3s ease;
    font-size: 0.95rem;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(46,160,67,0.4);
}

div.clear-btn-container > div > button {
    background: linear-gradient(135deg, #21262d 0%, #30363d 100%) !important;
    color: #f85149 !important;
    border: 1px solid rgba(248,81,73,0.4) !important;
}
div.clear-btn-container > div > button:hover {
    background: rgba(248,81,73,0.1) !important;
    box-shadow: 0 8px 24px rgba(248,81,73,0.2) !important;
}

.result-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.8) 0%, rgba(33,38,45,0.6) 100%);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    animation: fadeInUp 0.6s ease-out;
    transition: all 0.3s ease;
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
.result-card p, .result-card li { color: #c9d1d9; font-size: 0.95rem; line-height: 1.6; }

.result-card-locked {
    background: linear-gradient(135deg, rgba(45,14,14,0.5) 0%, rgba(33,38,45,0.4) 100%);
    border: 1px solid rgba(248,81,73,0.25);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.result-card-locked h4 {
    color: #f85149;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.result-card-locked p { color: #8b949e; font-size: 0.95rem; line-height: 1.6; font-style: italic; }

.result-card-limited {
    background: linear-gradient(135deg, rgba(22,27,34,0.8) 0%, rgba(33,38,45,0.6) 100%);
    border: 1px solid rgba(240,136,62,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    animation: fadeInUp 0.6s ease-out;
}
.result-card-limited h4 {
    color: #f0883e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.result-card-limited p { color: #c9d1d9; font-size: 0.95rem; line-height: 1.6; }
.result-card-limited .restriction-note {
    color: #f0883e;
    font-size: 0.8rem;
    margin-top: 8px;
    font-style: italic;
}

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
.role-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
hr { border-color: rgba(88,166,255,0.1) !important; }

.stNumberInput input {
    background: rgba(22, 30, 40, 1) !important;
    color: #e6edf3 !important;
    border: 1px solid rgba(88, 166, 255, 0.35) !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}
.stNumberInput input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.2) !important;
}
.stNumberInput button {
    background: rgba(33, 38, 45, 1) !important;
    color: #e6edf3 !important;
    border-color: rgba(88, 166, 255, 0.25) !important;
}

/* ── Auth panel ── */
.auth-panel {
    background: rgba(22,27,34,0.7);
    border: 1px solid rgba(48,54,61,0.6);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.auth-success {
    background: rgba(35,134,54,0.15);
    border-left: 3px solid #2ea043;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    color: #3fb950;
    font-size: 0.82rem;
    margin-top: 6px;
}
.auth-error {
    background: rgba(248,81,73,0.1);
    border-left: 3px solid #f85149;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    color: #f85149;
    font-size: 0.82rem;
    margin-top: 6px;
}
.auth-info {
    color: #8b949e;
    font-size: 0.72rem;
    margin-top: 6px;
    font-style: italic;
}
.logged-in-badge {
    background: rgba(35,134,54,0.2);
    border: 1px solid rgba(46,160,67,0.4);
    border-radius: 8px;
    padding: 8px 12px;
    color: #3fb950;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.nlp-info-banner {
    background: linear-gradient(135deg, rgba(31,111,235,0.08) 0%, rgba(33,38,45,0.6) 100%);
    border-left: 3px solid rgba(88,166,255,0.5);
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    color: #8b949e;
    font-size: 0.82rem;
    margin-bottom: 10px;
    line-height: 1.5;
}
.nlp-info-banner span { color: #58a6ff; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════════
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
    "Clear Symptoms": "ምልክቶችን አጽዳ",
    "Clear Diagnosis": "ውጤቶችን አጽዳ",
    "Please enter symptoms.": "እባክዎ ምልክቶችን ያስገቡ።",
    "Please enter a query.": "እባክዎ ጥያቄ ያስገቡ።",
    "medical_advice_disclaimer":
        "ማንኛውንም መድሃኒት ከመውሰድዎ በፊት ወይም ከባድ ምልክቶች ካጋጠሙዎት ሁልጊዜ ሐኪም ያማክሩ።",
}

def t(text: str, lang: str) -> str:
    return AMHARIC.get(text, text) if lang.lower() == "amharic" else text


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE TRANSLATE (optional)
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# FILE CHECKS
# ══════════════════════════════════════════════════════════════════════════════
DATA_FILES = [
    "data/Diseases_and_Symptoms_dataset.csv",
    "data/description.csv", "data/diets.csv",
    "data/medications.csv", "data/precautions.csv", "data/workout.csv",
]
MODEL_FILES = ["models/svc_model.pkl", "models/decision_tree_model.pkl", "models/label_encoder.pkl"]

def check_files():
    return [f for f in DATA_FILES + MODEL_FILES if not os.path.exists(f)]

def clean_disease_name(name: str) -> str:
    return str(name).lower().replace("_", " ").strip()


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
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

    return main_df, description_map, diets_map, medications_map, precautions_map, workout_map


@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    svc = joblib.load("models/svc_model.pkl")
    dt  = joblib.load("models/decision_tree_model.pkl")
    le  = joblib.load("models/label_encoder.pkl")
    return svc, dt, le


@st.cache_resource(show_spinner="Building search index…")
def build_tfidf_index(symptom_list: tuple, disease_names: tuple):
    sym_texts = [s.replace("_", " ") for s in symptom_list]
    dis_texts = [d.replace("_", " ") for d in disease_names]
    all_texts = sym_texts + dis_texts
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(all_texts)
    sym_matrix = vec.transform(sym_texts).toarray()
    dis_matrix = vec.transform(dis_texts).toarray()
    return vec, sym_matrix, dis_matrix


# ══════════════════════════════════════════════════════════════════════════════
# NLP SYMPTOM EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
NEGATION_TRIGGERS = [
    "no ", "not ", "without ", "don't have", "doesn't have",
    "do not have", "i have no", "haven't", "havent",
    "denies", "denied", "no history of", "no sign of",
    "no complaint of", "absence of",
]

FILLER_RE = re.compile(
    r"\b(i|i've|i have|i am|i'm|been|having|some|really|very|quite|"
    r"a\s+bit|little|a\s+lot|feel|feeling|seems|seem|also|and|but|for|"
    r"the|a|an|my|me|with|since|past|two|three|one|few|several|many|"
    r"days|day|weeks|week|hours|hour|terrible|awful|bad|mild|severe|"
    r"chronic|sudden|slight|slightly|intense|occasionally|sometimes|"
    r"today|yesterday|lately|recently|currently|experiencing|suffer|"
    r"suffering|complaining|complaint|though|however|about|around|"
    r"kind\s+of|sort\s+of|type\s+of|bit\s+of|lot\s+of|still|keep|"
    r"keeps|kept|started|starting|getting|got|have|has|had)\b",
    re.IGNORECASE,
)


def _is_negated(sentence: str) -> bool:
    sl = sentence.lower()
    return any(trigger in sl for trigger in NEGATION_TRIGGERS)


def _fuzzy_match(phrase: str, symptom_list: list, tfidf_vec, sym_matrix, threshold: float):
    phrase_norm = phrase.replace("_", " ").strip()
    if not phrase_norm or len(phrase_norm) < 3:
        return None
    phrase_key = phrase_norm.replace(" ", "_").lower()
    if phrase_key in symptom_list:
        return phrase_key
    vec_query = tfidf_vec.transform([phrase_norm]).toarray()
    sims = cosine_similarity(vec_query, sym_matrix)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= threshold:
        return symptom_list[best_idx]
    return None


def extract_symptoms_from_text(
    user_input: str,
    symptom_list: list,
    tfidf_vec,
    sym_matrix,
    threshold: float = 0.35,
) -> tuple:
    sentences = re.split(r"[.!?;]+", user_input)
    matched: set = set()
    negated: set = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        is_neg = _is_negated(sentence)
        clauses = re.split(
            r"[,\n]|\band\b|\bor\b|\bbut\b|\balso\b|\bwith\b|\bplus\b",
            sentence, flags=re.IGNORECASE,
        )
        for clause in clauses:
            cleaned = FILLER_RE.sub(" ", clause)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            cleaned_key = cleaned.replace(" ", "_")
            sym = _fuzzy_match(cleaned_key, symptom_list, tfidf_vec, sym_matrix, threshold)
            if sym:
                if is_neg:
                    negated.add(sym)
                else:
                    matched.add(sym)

    confirmed = matched - negated
    return confirmed, negated


# ══════════════════════════════════════════════════════════════════════════════
# ACCESS CONTROL  ←  100% LOCAL, no server needed
# ══════════════════════════════════════════════════════════════════════════════
def check_access(age: int, role: str, user_id: str, lang: str) -> tuple:
    """
    Returns (allowed: bool, error_message: str).
    Validates purely against local sets — no HTTP calls.
    """
    if age < 18:
        return False, t("Access Denied: Information is not available for users under 18.", lang)

    uid = user_id.strip().upper()

    if role == "Doctor":
        # Already verified this session
        if st.session_state.get("verified_id") == uid and st.session_state.get("verified_role") == "Doctor":
            return True, ""
        if uid in VALID_DOCTORS:
            st.session_state["verified_id"]   = uid
            st.session_state["verified_role"] = "Doctor"
            return True, ""
        return False, t("Access Denied: Invalid ID for Doctor role.", lang)

    if role == "Student":
        if st.session_state.get("verified_id") == uid and st.session_state.get("verified_role") == "Student":
            return True, ""
        if uid in VALID_STUDENTS:
            st.session_state["verified_id"]   = uid
            st.session_state["verified_role"] = "Student"
            return True, ""
        return False, t("Access Denied: Invalid ID for Student role.", lang)

    # Normal User — no ID check needed
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR AUTH PANEL  ←  SAFE STATE RETRIEVAL PATCH (Line 672 Unpack Fix)
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar_auth(role: str, lang: str) -> str:
    """
    Shows ID entry for Student / Doctor roles.
    Verifies against local sets instantly. Preserves exact layout structure.
    """
    if "verified_id"   not in st.session_state: st.session_state["verified_id"]   = ""
    if "verified_role" not in st.session_state: st.session_state["verified_role"] = ""
    if "auth_msg"      not in st.session_state: st.session_state["auth_msg"]      = ("", False)

    vid  = st.session_state.get("verified_id", "")
    vrol = st.session_state.get("verified_role", "")

    # Already verified for this role
    if vid and vrol == role:
        st.sidebar.markdown(
            f"<div class='logged-in-badge'>✅ Verified as {role}<br>"
            f"<span style='font-size:0.75rem;opacity:0.8'>ID: {vid}</span></div>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🔒 Clear verification", key="clear_verify", use_container_width=True):
            st.session_state["verified_id"]   = ""
            st.session_state["verified_role"] = ""
            st.session_state["auth_msg"]      = ("", False)
            st.rerun()
        return vid

    prefix      = "ST" if role == "Student" else "DR"
    placeholder = f"e.g. {prefix}001"
    hint        = f"ID must start with <b>{prefix}</b> — e.g. {prefix}001"

    st.sidebar.markdown(
        f"<div class='auth-info' style='margin-bottom:6px'>{hint}</div>",
        unsafe_allow_html=True,
    )

    entered_id = st.sidebar.text_input(
        "Enter your ID", placeholder=placeholder,
        type="password", key="sidebar_id_field",
    )

    if st.sidebar.button("✅ Verify ID", key="verify_btn", use_container_width=True):
        uid = entered_id.strip().upper()
        if not uid:
            st.session_state["auth_msg"] = ("Please enter your ID.", False)
        elif role == "Doctor" and uid not in VALID_DOCTORS:
            st.session_state["auth_msg"] = ("❌ ID not recognised for Doctor role.", False)
        elif role == "Student" and uid not in VALID_STUDENTS:
            st.session_state["auth_msg"] = ("❌ ID not recognised for Student role.", False)
        else:
            st.session_state["verified_id"]   = uid
            st.session_state["verified_role"] = role
            st.session_state["auth_msg"]      = (f"✅ Verified as {role}.", True)
            st.rerun()

    # 🛠️ INTEGRATED FIX FOR CRASHING LINE 672: Safe unpack handling falling back to default states
    auth_data = st.session_state.get("auth_msg", ("", False))
    if isinstance(auth_data, tuple) and len(auth_data) == 2:
        msg_text, msg_ok = auth_data
    else:
        msg_text, msg_ok = "", False

    if msg_text:
        css = "auth-success" if msg_ok else "auth-error"
        st.sidebar.markdown(
            f"<div class='{css}'>{msg_text}</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(
        "<div class='auth-info'>Don't have an ID? Open <b>auth.html</b> in your browser "
        "to create one, then add it to <code>VALID_DOCTORS</code> / <code>VALID_STUDENTS</code> "
        "at the top of <code>app.py</code>.</div>",
        unsafe_allow_html=True,
    )

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# ROLE-BASED RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
def role_based_recs(role, lang, key,
                    description_map, diets_map, medications_map,
                    precautions_map, workout_map):
    desc    = translate_content(description_map.get(key, "N/A"), lang)
    diet    = translate_content(diets_map.get(key, "N/A"), lang)
    meds    = translate_content(medications_map.get(key, "N/A"), lang)
    precs   = translate_content(precautions_map.get(key, []), lang)
    workout = translate_content(workout_map.get(key, "N/A"), lang)

    if role == "Doctor":
        cards = [
            {"label": t("Description",     lang), "content": desc,    "card_type": "full"},
            {"label": t("Dietary Plan",     lang), "content": diet,    "card_type": "full"},
            {"label": t("Medications",      lang), "content": meds,    "card_type": "full"},
            {"label": t("Precautions",      lang), "content": precs,   "card_type": "full"},
            {"label": t("Workout/Activity", lang), "content": workout, "card_type": "full"},
        ]
        advice = ""

    elif role == "Student":
        if isinstance(meds, str) and meds not in ("N/A", ""):
            drug_classes = []
            for item in meds.split(","):
                item = item.strip()
                if item:
                    drug_classes.append(item.split()[0].rstrip(".,;"))
            meds_display = ", ".join(dict.fromkeys(drug_classes))
            meds_note    = "Drug class names shown only. Full details restricted to licensed Doctors."
        else:
            meds_display = "N/A"
            meds_note    = "Full medication details restricted to Doctors only."

        cards = [
            {"label": t("Description",     lang), "content": desc,         "card_type": "full"},
            {"label": t("Dietary Plan",     lang), "content": diet,         "card_type": "full"},
            {"label": t("Medications",      lang), "content": meds_display,
             "note": meds_note,                                              "card_type": "limited"},
            {"label": t("Precautions",      lang), "content": precs,        "card_type": "full"},
            {"label": t("Workout/Activity", lang), "content": workout,       "card_type": "full"},
        ]
        advice = "ℹ️ Full medication details are only available to licensed Doctors."

    else:
        cards = [
            {"label": t("Description",     lang), "content": desc,   "card_type": "full"},
            {"label": t("Dietary Plan",     lang),
             "content": "🔒 Dietary plan details are restricted. Please consult a licensed nutritionist or doctor.",
             "card_type": "locked"},
            {"label": t("Medications",     lang),
             "content": "🔒 Medication details are restricted to licensed Doctors only. Please consult a healthcare professional.",
             "card_type": "locked"},
            {"label": t("Precautions",     lang), "content": precs,   "card_type": "full"},
            {"label": t("Workout/Activity",lang), "content": workout, "card_type": "full"},
        ]
        advice = ("⚠️ This information is for general awareness only. "
                  "Always consult a qualified doctor before taking any medication.")

    return cards, advice


# ══════════════════════════════════════════════════════════════════════════════
# RENDER RECOMMENDATION CARDS
# ══════════════════════════════════════════════════════════════════════════════
def render_rec_cards(cards: list):
    for card in cards:
        label     = card["label"]
        content   = card["content"]
        card_type = card.get("card_type", "full")
        note      = card.get("note", "")

        if isinstance(content, list):
            body = "<ul style='margin:0;padding-left:18px'>" + \
                   "".join(f"<li>{i}</li>" for i in content) + "</ul>"
        else:
            body = f"<p style='margin:0'>{content}</p>"

        if note:
            body += (f"<p class='restriction-note' style='margin-top:8px;color:#f0883e;"
                     f"font-size:0.8rem;font-style:italic'>⚠️ {note}</p>")

        css_class = {
            "full":    "result-card",
            "limited": "result-card-limited",
            "locked":  "result-card-locked",
        }.get(card_type, "result-card")

        st.markdown(
            f"<div class='{css_class}'><h4>{label}</h4>{body}</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
def integrated_prediction_system(
    user_input, age, role, user_id, lang,
    main_df, le, svc_model, dt_model,
    description_map, diets_map, medications_map, precautions_map, workout_map,
    tfidf_vec, sym_matrix, threshold=0.35,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return None, msg

    symptom_list = list(main_df.drop(columns=["diseases"]).columns)
    matched_symptoms, negated_symptoms = extract_symptoms_from_text(
        user_input, symptom_list, tfidf_vec, sym_matrix, threshold
    )

    if not matched_symptoms:
        hint = ""
        if negated_symptoms:
            hint = (f" (Negated symptoms detected: "
                    f"{', '.join(s.replace('_',' ') for s in negated_symptoms)})")
        return None, (
            "⚠️ No recognised symptoms found. Try describing your symptoms naturally, e.g.:\n"
            "• *I have a headache and sore throat*\n"
            "• *I've been feeling tired with fever and chills*\n"
            "• *My joints ache and I'm short of breath*" + hint
        )

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
                preds.append({"model": name, "disease": disease.title(),
                               "confidence": conf, "top": rank == 0})
        except AttributeError:
            pred_idx = model.predict(feature_vector)[0]
            disease  = le.inverse_transform([pred_idx])[0]
            preds.append({"model": name, "disease": disease.title(),
                          "confidence": "N/A", "top": True})

    top_disease = next(p["disease"] for p in preds if p["top"] and p["model"] == "SVC")
    top_key     = clean_disease_name(top_disease)
    svc_preds   = [p for p in preds if p["model"] == "SVC"]

    cards, advice = role_based_recs(
        role, lang, top_key,
        description_map, diets_map, medications_map, precautions_map, workout_map,
    )

    return {
        "matched_symptoms":     [s.replace("_", " ").title() for s in matched_symptoms],
        "negated_symptoms":     [s.replace("_", " ").title() for s in negated_symptoms],
        "predicted_conditions": svc_preds,
        "top_disease":          top_disease,
        "rec_cards":            cards,
        "advice":               advice,
    }, ""


# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
def chatbot_response(
    query, age, role, user_id, lang,
    description_map, diets_map, medications_map, precautions_map, workout_map,
    tfidf_vec, symptom_list=None, sym_matrix=None, threshold=0.3,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return msg

    q = query.lower().strip()

    INTENT_KEYWORDS = {
        "diet":        ["diet", "food", "eat", "nutrition", "meal", "drink", "አመጋገብ"],
        "medication":  ["medicine", "medication", "drug", "pill", "treat", "tablet", "prescription", "dose", "መድሃኒት"],
        "precaution":  ["precaution", "avoid", "prevent", "careful", "warning", "ጥንቃቄ"],
        "workout":     ["workout", "exercise", "activity", "fitness", "sport", "physical", "ስፖርት"],
        "description": ["what is", "describe", "meaning", "define", "definition", "መግለጫ"]
    }

    intent = "general"
    for k, words in INTENT_KEYWORDS.items():
        if any(w in q for w in words):
            intent = k
            break

    matched_disease = None
    all_diseases = list(description_map.keys())

    for d in all_diseases:
        if d in q or d.replace(" ", "_") in q:
            matched_disease = d
            break

    if not matched_disease and tfidf_vec is not None and sym_matrix is not None:
        cleaned_query = re.sub(r"\b(what|is|the|of|for|in|a|an|treatment|plan|info|information)\b", "", q).strip()
        if len(cleaned_query) >= 3:
            vec_query = tfidf_vec.transform([cleaned_query]).toarray()
            sims = cosine_similarity(vec_query, sym_matrix)[0]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= threshold:
                matched_disease = all_diseases[best_idx % len(all_diseases)]

    if not matched_disease:
        return (t("Hello! How can I help you with health information today?", lang) + "\n\n" + 
                "💡 *Try asking about specific conditions, e.g., 'What is Malaria?', 'Diet plan for Diabetes', or 'Precautions for Hypertension'*")

    display_name = matched_disease.title()
    cards, advice = role_based_recs(role, lang, matched_disease, description_map, diets_map, medications_map, precautions_map, workout_map)

    if intent == "general":
        res = f"### 🏥 Information Summary for **{display_name}**\n\n"
        for card in cards:
            lbl, content = card["label"], card["content"]
            if isinstance(content, list):
                content = "\n".join(f"- {i}" for i in content)
            res += f"**{lbl}**:\n{content}\n\n"
        if advice:
            res += f"\n*{advice}*"
        return res

    target_card = next((c for c in cards if c["label"] == t(intent.replace("_", " ").title(), lang) or intent in c["label"].lower()), None)
    if target_card:
        content = target_card["content"]
        if isinstance(content, list):
            content = "\n".join(f"- {i}" for i in content)
        return f"### {target_card['label']} for **{display_name}**\n\n{content}\n\n*{advice}*"

    return f"I found records matching **{display_name}**, but no explicit specific data regarding your intent target. Please try searching generally."


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP ENTRY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown(
        "<div class='main-header'>"
        "<h1 class='main-header-title'>🏥 Integrated Healthcare Dashboard</h1>"
        "<p class='main-header-subtitle'>Advanced AI Symptom Mapping & Healthcare Management Framework</p>"
        "</div>",
        unsafe_allow_html=True
    )

    missing = check_files()
    if missing:
        st.error("### ⚠️ Missing Required System Resource Files")
        st.markdown("The pipeline cannot initialize because the following files are missing:")
        for m in missing:
            st.markdown(f"- ` {m} `")
        st.info("Ensure datasets are placed inside `data/` and trained artifacts inside `models/` directories.")
        return

    # Load resources
    main_df, description_map, diets_map, medications_map, precautions_map, workout_map = load_data()
    svc_model, dt_model, le = load_models()

    symptom_list = list(main_df.drop(columns=["diseases"]).columns)
    disease_names = list(description_map.keys())
    tfidf_vec, sym_matrix, dis_matrix = build_tfidf_index(tuple(symptom_list), tuple(disease_names))

    # Sidebar parameters Setup
    st.sidebar.markdown("<div class='section-header'>🌐 Global Framework Options</div>", unsafe_allow_html=True)
    lang = st.sidebar.selectbox("Language / ቋንቋ", ["English", "Amharic"])
    
    st.sidebar.markdown("<div class='section-header'>👤 Access & Roles Configuration</div>", unsafe_allow_html=True)
    role = st.sidebar.selectbox("Your Role", ["Normal User", "Student", "Doctor"])
    age = st.sidebar.number_input("Your Age", min_value=1, max_value=120, value=25, step=1)

    # Invoke patched auth processor safely returning state string
    user_id = render_sidebar_auth(role, lang)

    # System layout Tabs definition context 
    tabs = st.tabs([f"🔍 {t('Disease Predictor', lang)}", f"📋 {t('Health Recommender', lang)}", f"💬 {t('Healthcare Chatbot', lang)}"])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: DISEASE PREDICTOR
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(f"<div class='section-header'>{t('Disease Predictor', lang)}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='nlp-info-banner'>💡 <b>Natural Language Processing Enabled:</b> "
            "Describe your context smoothly. The extractor filters out fillers like <i>'terrible'</i>, "
            "handles negations such as <i>'no fever'</i> automatically, and isolates confirmed matching clinical symptoms.</div>",
            unsafe_allow_html=True
        )

        user_input = st.text_area(
            "Describe your health symptoms naturally:",
            placeholder="e.g., I've been feeling a lot of shivering and severe joint pain since yesterday, but I have no cough or runny nose.",
            key="symptom_input_area"
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            predict_triggered = st.button(t("Predict & Recommend", lang), use_container_width=True)
        with col2:
            if st.button(t("Clear Symptoms", lang), key="clear_sym_btn"):
                st.session_state["symptom_input_area"] = ""
                st.rerun()

        if predict_triggered:
            if not user_input.strip():
                st.warning(t("Please enter symptoms.", lang))
            else:
                with st.spinner("Analyzing text profile indicators..."):
                    res, err = integrated_prediction_system(
                        user_input, age, role, user_id, lang,
                        main_df, le, svc_model, dt_model,
                        description_map, diets_map, medications_map, precautions_map, workout_map,
                        tfidf_vec, sym_matrix
                    )
                if err:
                    st.markdown(f"<div class='access-denied'>{err}</div>", unsafe_allow_html=True)
                elif res:
                    st.success("Analysis Complete!")
                    
                    # Display extracted configurations tokens mappings metrics metrics
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### Extracted Symptoms")
                        for s in res["matched_symptoms"]:
                            st.markdown(f"<span class='disease-badge'>✓ {s}</span>", unsafe_allow_html=True)
                    with c2:
                        if res["negated_symptoms"]:
                            st.markdown("#### Negated (Excluded) Symptoms")
                            for s in res["negated_symptoms"]:
                                st.markdown(f"<span class='disease-badge' style='background:linear-gradient(135deg, #21262d 0%, #30363d 100%);color:#8b949e;border:1px solid rgba(248,81,73,0.2)'>✕ {s}</span>", unsafe_allow_html=True)

                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"### ML Model Diagnostic Conditions for Top Target: **{res['top_disease']}**")
                    
                    for pred in res["predicted_conditions"]:
                        st.markdown(f"📍 **Condition Matched**: {pred['disease']} <span class='conf-pill'>Confidence Score: {pred['confidence']}</span>", unsafe_allow_html=True)
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    render_rec_cards(res["rec_cards"])
                    
                    if res["advice"]:
                        st.markdown(f"<div class='advice-banner'>{res['advice']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:0.78rem;opacity:0.6;margin-top:12px'>⚠️ {t('medical_advice_disclaimer', lang)}</p>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: HEALTH RECOMMENDER
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown(f"<div class='section-header'>{t('Health Recommender', lang)}</div>", unsafe_allow_html=True)
        sorted_diseases = sorted([d.title() for d in description_map.keys()])
        selected_disease = st.selectbox("Select Condition to Retrieve Management Protocol:", sorted_diseases)

        if st.button(t("Get Plan", lang), key="get_plan_btn"):
            with st.spinner("Compiling strategic health plan vectors..."):
                ok, msg = check_access(age, role, user_id, lang)
                if not ok:
                    st.markdown(f"<div class='access-denied'>{msg}</div>", unsafe_allow_html=True)
                else:
                    dk = clean_disease_name(selected_disease)
                    cards, advice = role_based_recs(role, lang, dk, description_map, diets_map, medications_map, precautions_map, workout_map)
                    
                    st.markdown(f"### 📋 Strategic Optimization Summary for: **{selected_disease}**")
                    render_rec_cards(cards)
                    if advice:
                        st.markdown(f"<div class='advice-banner'>{advice}</div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:0.78rem;opacity:0.6;margin-top:12px'>⚠️ {t('medical_advice_disclaimer', lang)}</p>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: HEALTHCARE CHATBOT
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"<div class='section-header'>{t('Healthcare Chatbot', lang)}</div>", unsafe_allow_html=True)
        bot_query = st.text_input("Ask any health or diagnostic plan question:", placeholder="e.g., What is the best diet profile for Diabetes?", key="bot_query_field")

        if st.button(t("Ask Bot", lang), key="ask_bot_btn"):
            if not bot_query.strip():
                st.warning(t("Please enter a query.", lang))
            else:
                with st.spinner("Processing semantic classification parameters..."):
                    response_text = chatbot_response(
                        bot_query, age, role, user_id, lang,
                        description_map, diets_map, medications_map, precautions_map, workout_map,
                        tfidf_vec, symptom_list, dis_matrix
                    )
                st.markdown(f"<div class='chat-bot'>{response_text}</div>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:0.78rem;opacity:0.6;margin-top:12px'>⚠️ {t('medical_advice_disclaimer', lang)}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
