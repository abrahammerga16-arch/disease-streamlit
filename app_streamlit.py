"""
Integrated Healthcare Dashboard — Streamlit version (Enhanced UI with Animations)
  • Disease Predictor  (symptom input → ML prediction)
  • Health Recommender (disease dropdown → full health plan)
  • Healthcare Chatbot  (natural-language query → semantic answer)
  • Role / Age / ID access control  ← signup/login via Colab Flask API
  • English ↔ Amharic UI
"""

import os, json, warnings, requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ── Hardcoded Colab API URL (no UI input needed) ──────────────────────────────
COLAB_API_URL = "https://porcupine-universal-timing.ngrok-free.dev"

st.set_page_config(
    page_title="Integrated Healthcare Dashboard",
    page_icon="",
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

/* ── Auth panel styles ── */
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
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TRANSLATIONS
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
    "Clear Symptoms": "ምልክቶችን አጽዳ",
    "Clear Diagnosis": "ውጤቶችን አጽዳ",
    "Please enter symptoms.": "እባክዎ ምልክቶችን ያስገቡ።",
    "Please enter a query.": "እባክዎ ጥያቄ ያስገቡ።",
    "medical_advice_disclaimer":
        "ማንኛውንም መድሃኒት ከመውሰድዎ በፊት ወይም ከባድ ምልክቶች ካጋጠሙዎት ሁልጊዜ ሐኪም ያማክሩ።",
}

def t(text: str, lang: str) -> str:
    return AMHARIC.get(text, text) if lang.lower() == "amharic" else text


# ──────────────────────────────────────────────
# GOOGLE TRANSLATE
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
# FILE CHECKS
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
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
# ACCESS CONTROL  ←  Uses hardcoded COLAB_API_URL constant
# ══════════════════════════════════════════════════════════════════════════════

def _api_login(role: str, user_id: str) -> tuple[bool, str]:
    """
    Call POST {COLAB_API_URL}/login on the Colab Flask server.
    Returns (success, message).
    """
    base = COLAB_API_URL.rstrip("/")
    try:
        resp = requests.post(
            f"{base}/login",
            json={"role": role, "user_id": user_id},
            timeout=8,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("success"):
            return True, data.get("message", "Login successful.")
        return False, data.get("message", "Login failed.")
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot reach server. Please try again later."
    except requests.exceptions.Timeout:
        return False, "❌ Request timed out. Please try again."
    except Exception as e:
        return False, f"❌ API error: {e}"


def _api_signup(role: str, user_id: str, name: str = "") -> tuple[bool, str]:
    """
    Call POST {COLAB_API_URL}/signup on the Colab Flask server.
    Returns (success, message).
    """
    base = COLAB_API_URL.rstrip("/")
    try:
        resp = requests.post(
            f"{base}/signup",
            json={"role": role, "user_id": user_id, "name": name},
            timeout=8,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("success"):
            return True, data.get("message", "Sign-up successful.")
        return False, data.get("message", "Sign-up failed.")
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot reach server. Please try again later."
    except requests.exceptions.Timeout:
        return False, "❌ Request timed out. Please try again."
    except Exception as e:
        return False, f"❌ API error: {e}"


def check_access(age: int, role: str, user_id: str, lang: str) -> tuple[bool, str]:
    """
    Central access-control gate used by all three tabs.
    • Age check is local (no API needed).
    • Student / Doctor identity is verified via the Colab /login endpoint.
    • Normal User always passes (no ID required).
    """
    if age < 18:
        return False, t("Access Denied: Information is not available for users under 18.", lang)

    if role in ("Student", "Doctor"):
        # Use the cached login state from session so we don't hit the API on every render
        logged_in_as = st.session_state.get("logged_in_as")
        if (
            logged_in_as
            and logged_in_as.get("role") == role
            and logged_in_as.get("user_id", "").upper() == user_id.strip().upper()
        ):
            return True, ""
        # Not cached — try a live login check
        ok, msg = _api_login(role, user_id)
        if ok:
            st.session_state["logged_in_as"] = {"role": role, "user_id": user_id.strip().upper()}
            return True, ""
        return False, msg

    return True, ""   # Normal User — no check needed


# ──────────────────────────────────────────────
# ROLE-BASED RECOMMENDATIONS
# ──────────────────────────────────────────────
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
            {"label": t("Description",      lang), "content": desc,    "card_type": "full"},
            {"label": t("Dietary Plan",      lang), "content": diet,    "card_type": "full"},
            {"label": t("Medications",       lang), "content": meds,    "card_type": "full"},
            {"label": t("Precautions",       lang), "content": precs,   "card_type": "full"},
            {"label": t("Workout/Activity",  lang), "content": workout, "card_type": "full"},
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
            {"label": t("Description",      lang), "content": desc,         "card_type": "full"},
            {"label": t("Dietary Plan",      lang), "content": diet,         "card_type": "full"},
            {"label": t("Medications",       lang), "content": meds_display,
             "note": meds_note,                                               "card_type": "limited"},
            {"label": t("Precautions",       lang), "content": precs,        "card_type": "full"},
            {"label": t("Workout/Activity",  lang), "content": workout,      "card_type": "full"},
        ]
        advice = "ℹ️ Full medication details are only available to licensed Doctors."

    else:
        cards = [
            {"label": t("Description",      lang), "content": desc,   "card_type": "full"},
            {"label": t("Dietary Plan",      lang),
             "content": "🔒 Dietary plan details are restricted. Please consult a licensed nutritionist or doctor for a personalised plan.",
                                                                        "card_type": "locked"},
            {"label": t("Medications",       lang),
             "content": "🔒 Medication details are restricted to licensed Doctors only. Please consult a healthcare professional.",
                                                                        "card_type": "locked"},
            {"label": t("Precautions",       lang), "content": precs, "card_type": "full"},
            {"label": t("Workout/Activity",  lang), "content": workout, "card_type": "full"},
        ]
        advice = ("⚠️ This information is for general awareness only. "
                  "Always consult a qualified doctor before taking any medication or following a treatment plan.")

    return cards, advice


# ──────────────────────────────────────────────
# RENDER RECOMMENDATION CARDS
# ──────────────────────────────────────────────
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
            body += f"<p class='restriction-note' style='margin-top:8px;color:#f0883e;font-size:0.8rem;font-style:italic'>⚠️ {note}</p>"

        css_class = {
            "full":    "result-card",
            "limited": "result-card-limited",
            "locked":  "result-card-locked",
        }.get(card_type, "result-card")

        st.markdown(
            f"<div class='{css_class}'><h4>{label}</h4>{body}</div>",
            unsafe_allow_html=True,
        )


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

    matched_display = [s.replace("_", " ").title() for s in matched_symptoms]
    return {
        "matched_symptoms":     matched_display,
        "predicted_conditions": svc_preds,
        "top_disease":          top_disease,
        "rec_cards":            cards,
        "advice":               advice,
    }, ""


# ──────────────────────────────────────────────
# CHATBOT
# ──────────────────────────────────────────────
def chatbot_response(
    query, age, role, user_id, lang,
    description_map, diets_map, medications_map, precautions_map, workout_map,
    tfidf_vec, threshold=0.3,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return msg

    q = query.lower().strip()

    INTENT_KEYWORDS = {
        "diet":       ["diet", "food", "eat", "nutrition", "meal", "drink", "አመጋገብ"],
        "medication": ["medicine", "medication", "drug", "pill", "treat", "tablet",
                       "prescription", "dose", "መድሃኒት"],
        "precaution": ["precaution", "avoid", "prevent", "careful", "warning", "ጥንቃቄ"],
        "workout":    ["workout", "exercise", "activity", "fitness", "sport", "physical", "ስፖርት"],
        "description":["what is", "describe", "about", "explain", "overview",
                       "definition", "meaning", "symptom", "cause", "sign"],
    }

    intent = "description"
    for key, keywords in INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            intent = key
            break

    disease_keys   = list(description_map.keys())
    rich_texts     = []
    for dk in disease_keys:
        desc = description_map.get(dk, "")
        rich_texts.append(f"{dk} {desc}")

    if not rich_texts:
        return "Sorry, the knowledge base is empty."

    chat_vec    = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
    ).fit(rich_texts)

    rich_matrix = chat_vec.transform(rich_texts).toarray()
    q_vec       = chat_vec.transform([q]).toarray()
    sims        = cosine_similarity(q_vec, rich_matrix)[0]
    best_idx    = int(np.argmax(sims))
    best_sim    = float(sims[best_idx])

    if best_sim < threshold:
        fallback = (
            "I couldn't find specific information for that query. "
            "Try asking about a specific disease, e.g.:\n"
            "• *What is diabetes?*\n"
            "• *What diet should I follow for asthma?*\n"
            "• *What medications are used for hypertension?*\n"
            "• *What precautions should I take for migraine?*"
        )
        return translate_content(fallback, lang) if lang.lower() == "amharic" else fallback

    disease_key   = disease_keys[best_idx]
    disease_title = disease_key.title()

    if intent == "diet":
        raw   = diets_map.get(disease_key, "N/A")
        label = t("Dietary Plan", lang)
        if role == "Normal User":
            info        = "🔒 Dietary plan details are restricted. Please consult a licensed nutritionist or doctor for a personalised plan."
            restriction = ""
        else:
            info        = raw
            restriction = ""

    elif intent == "medication":
        raw   = medications_map.get(disease_key, "N/A")
        label = t("Medications", lang)
        if role == "Normal User":
            info        = "🔒 Medication details are restricted. Please consult a licensed doctor."
            restriction = ""
        elif role == "Student":
            if isinstance(raw, str) and raw not in ("N/A", ""):
                drug_classes = [item.strip().split()[0].rstrip(".,;")
                                for item in raw.split(",") if item.strip()]
                info = ", ".join(dict.fromkeys(drug_classes))
            else:
                info = "N/A"
            restriction = "\n\n⚠️ *Drug class names only. Full details restricted to Doctors.*"
        else:
            info        = raw
            restriction = ""

    elif intent == "precaution":
        raw   = precautions_map.get(disease_key, [])
        label = t("Precautions", lang)
        if isinstance(raw, list):
            info = "\n" + "\n".join(f"• {p}" for p in raw) if raw else "N/A"
        else:
            info = str(raw)
        restriction = ""

    elif intent == "workout":
        info        = workout_map.get(disease_key, "N/A")
        label       = t("Workout/Activity", lang)
        restriction = ""

    else:
        info        = description_map.get(disease_key, "N/A")
        label       = t("Description", lang)
        restriction = ""

    info = translate_content(info, lang)
    if isinstance(info, list):
        info_str = "\n" + "\n".join(f"• {i}" for i in info)
    else:
        info_str = str(info)

    return f"**{disease_title}** — {label}:\n\n{info_str}{restriction}"


# ──────────────────────────────────────────────
# QUICK-SELECT WIDGET
# ──────────────────────────────────────────────
def render_quick_select_symptoms(lang: str) -> None:
    CATEGORIES_EN = {
        "🌡️ General": [
            "fever", "chills", "sweating", "fatigue", "weakness", "feeling ill",
            "weight gain", "ache all over", "flu-like syndrome", "restlessness",
            "sleepiness", "decreased appetite", "fluid retention",
        ],
        "🧠 Neuro/Mental": [
            "anxiety and nervousness", "depression", "insomnia", "dizziness",
            "abnormal involuntary movements", "depressive or psychotic symptoms",
            "disturbance of memory", "paresthesia", "loss of sensation", "focal weakness",
            "seizures", "delusions or hallucinations", "fainting", "temper problems",
            "fears and phobias", "obsessions and compulsions", "antisocial behavior",
            "hysterical behavior", "low self-esteem", "excessive anger", "hostile behavior",
            "drug abuse", "abusing alcohol",
        ],
        "🫀 Cardio/Resp": [
            "shortness of breath", "sharp chest pain", "chest tightness", "palpitations",
            "irregular heartbeat", "breathing fast", "cough", "wheezing", "difficulty breathing",
            "congestion in chest", "abnormal breathing sounds", "hurts to breath",
            "apnea", "hoarse voice", "hemoptysis", "coughing up sputum",
            "increased heart rate", "decreased heart rate", "burning chest pain",
        ],
        "👃 ENT": [
            "sore throat", "difficulty speaking", "nasal congestion", "throat swelling",
            "diminished hearing", "difficulty in swallowing", "pus draining from ear",
            "ear pain", "ringing in ear", "plugged feeling in ear", "itchy ear(s)",
            "fluid in ear", "redness in ear", "bleeding from ear",
            "swollen or red tonsils", "sneezing", "coryza", "sinus congestion",
            "painful sinuses", "nosebleed",
        ],
        "🤢 Gastro": [
            "sharp abdominal pain", "upper abdominal pain", "burning abdominal pain",
            "lower abdominal pain", "nausea", "vomiting", "vomiting blood", "diarrhea",
            "constipation", "stomach bloating", "heartburn", "regurgitation",
            "blood in stool", "melena", "rectal bleeding", "changes in stool appearance",
            "pain of the anus", "mass or swelling around the anus", "itching of the anus",
        ],
        "🤕 Pain": [
            "headache", "frontal headache", "back pain", "low back pain",
            "neck pain", "shoulder pain", "hip pain", "knee pain", "leg pain",
            "foot or toe pain", "ankle pain", "elbow pain", "arm pain",
            "wrist pain", "hand or finger pain", "joint pain", "rib pain",
            "groin pain", "suprapubic pain", "side pain", "facial pain",
            "bones are painful", "back cramps or spasms", "cramps and spasms",
        ],
        "🦴 Musculo/Limbs": [
            "arm stiffness or tightness", "arm swelling", "arm weakness", "arm lump or mass",
            "wrist swelling", "hand or finger swelling", "hand or finger stiffness or tightness",
            "hand or finger weakness", "hand or finger lump or mass",
            "knee swelling", "knee stiffness or tightness",
            "leg swelling", "leg weakness", "ankle swelling",
            "foot or toe swelling", "elbow swelling",
            "shoulder stiffness or tightness", "hip stiffness or tightness",
            "back stiffness or tightness", "neck mass", "neck swelling",
            "peripheral edema", "problems with movement",
        ],
        "🩺 Skin": [
            "abnormal appearing skin", "skin lesion", "acne or pimples", "skin growth",
            "skin moles", "warts", "skin rash", "itching of skin",
            "skin dryness, peeling, scaliness, or roughness", "skin irritation",
            "itchy scalp", "irregular appearing scalp", "jaundice",
            "diaper rash", "eyelid lesion or rash", "irregular appearing nails",
        ],
        "👁️ Eye": [
            "diminished vision", "double vision", "symptoms of eye", "pain in eye",
            "abnormal movement of eyelid", "foreign body sensation in eye",
            "eye redness", "lacrimation", "itchiness of eye", "blindness",
            "eye burns or stings", "spots or clouds in vision",
            "bleeding from eye", "mass on eyelid", "swollen eye", "eyelid swelling",
            "white discharge from eye",
        ],
        "🚻 Urinary/Repro": [
            "painful urination", "frequent urination", "involuntary urination",
            "blood in urine", "retention of urine", "unusual color or odor to urine",
            "excessive urination at night", "low urine output", "hesitancy",
            "symptoms of bladder", "symptoms of the kidneys", "kidney mass",
            "symptoms of prostate",
            "vaginal itching", "vaginal discharge", "vaginal pain", "vaginal redness",
            "pain during intercourse", "impotence",
            "symptoms of the scrotum and testes", "swelling of scrotum", "pain in testicles",
        ],
        "🤰 Women's Health": [
            "hot flashes", "intermenstrual bleeding", "pain during pregnancy",
            "problems during pregnancy", "spotting or bleeding during pregnancy",
            "uterine contractions", "recent pregnancy", "pelvic pain",
            "long menstrual periods", "heavy menstrual flow", "unpredictable menstruation",
            "painful menstruation", "infertility", "frequent menstruation",
            "blood clots during menstrual periods",
        ],
        "👶 Pediatric": [
            "lack of growth", "irritable infant", "infant feeding problem",
            "pulling at ears", "diaper rash",
        ],
        "🔬 Other": [
            "jaundice", "back mass or lump", "neck mass", "jaw swelling",
            "lip swelling", "toothache", "mouth ulcer", "gum pain",
            "mouth dryness", "mouth pain", "bleeding gums", "pain in gums",
            "allergic reaction", "symptoms of the face", "lower body pain",
        ],
    }

    SYMPTOM_AM = {
        # General
        "fever": "ትኩሳት", "chills": "ብርድ", "sweating": "ላብ",
        "fatigue": "ድካም", "weakness": "ድክመት", "feeling ill": "ታሞ ስሜት",
        "weight gain": "ክብደት መጨመር", "ache all over": "ሙሉ አካል ህመም",
        "flu-like syndrome": "ጉንፋን መሰል ምልክቶች", "restlessness": "ጸጥ አለማለት",
        "sleepiness": "ድካም/ናፍቆት", "decreased appetite": "የምግብ ፍቅር መቀነስ",
        "fluid retention": "ፈሳሽ ማቆር",
        # Neuro/Mental
        "anxiety and nervousness": "ጭንቀት እና ነርቭ", "depression": "ድብርት",
        "insomnia": "እንቅልፍ ማጣት", "dizziness": "ራስ ዞር",
        "abnormal involuntary movements": "ያልተፈለጉ እንቅስቃሴዎች",
        "depressive or psychotic symptoms": "ድብርት ወይም ሳይኮቲክ ምልክቶች",
        "disturbance of memory": "የትውስታ ችግር", "paresthesia": "መቆጥቆጥ ስሜት",
        "loss of sensation": "ስሜት ማጣት", "focal weakness": "ጠቃላይ ድክመት",
        "seizures": "ቅብጠት", "delusions or hallucinations": "ቅዠት",
        "fainting": "ዋዛ ማጣት", "temper problems": "ቁጣ ችግር",
        "fears and phobias": "ፍርሃቶች", "obsessions and compulsions": "ቋሚ ሃሳቦች",
        "antisocial behavior": "ፀረ-ማህበራዊ ባህሪ", "hysterical behavior": "ሂስቴሪካዊ ባህሪ",
        "low self-esteem": "ዝቅተኛ ራስ-ግምት", "excessive anger": "ከፍተኛ ቁጣ",
        "hostile behavior": "ጠበኛ ባህሪ", "drug abuse": "የዕፅ አጠቃቀም",
        "abusing alcohol": "የአልኮል ሱሰኝነት",
        # Cardio/Resp
        "shortness of breath": "መተንፈስ ማጠር", "sharp chest pain": "ሹል የደረት ህመም",
        "chest tightness": "ደረት መጠበቅ", "palpitations": "ልብ ምት ስሜት",
        "irregular heartbeat": "ያልተስተካከለ ልብ ምት", "breathing fast": "ፈጣን መተንፈስ",
        "cough": "ሳል", "wheezing": "ሲፏፏ ድምፅ", "difficulty breathing": "ለመተንፈስ ችግር",
        "congestion in chest": "ደረት ውስጥ ብናኝ", "abnormal breathing sounds": "ያልተለመዱ የትንፋሽ ድምፆች",
        "hurts to breath": "ሲተነፍሱ ህመም", "apnea": "መተንፈስ መቆም",
        "hoarse voice": "ሻካራ ድምፅ", "hemoptysis": "ደም አሳልፎ ማስወጣት",
        "coughing up sputum": "ምራቅ ማሳል", "increased heart rate": "ልብ ምት መጨመር",
        "decreased heart rate": "ልብ ምት መቀነስ", "burning chest pain": "የሚቃጠል የደረት ህመም",
        # ENT
        "sore throat": "ጉሮሮ ህመም", "difficulty speaking": "ለመናገር ችግር",
        "nasal congestion": "አፍንጫ መዘጋት", "throat swelling": "ጉሮሮ ማበጥ",
        "diminished hearing": "ሰሚ ማቀዝቀዝ", "difficulty in swallowing": "ለመዋጥ ችግር",
        "pus draining from ear": "ጆሮ ፕስ", "ear pain": "የጆሮ ህመም",
        "ringing in ear": "ጆሮ ድምፅ", "plugged feeling in ear": "ጆሮ መዘጋት",
        "itchy ear(s)": "ጆሮ ማሳከክ", "fluid in ear": "ጆሮ ውስጥ ፈሳሽ",
        "redness in ear": "ጆሮ ቀያ", "bleeding from ear": "ጆሮ ደም",
        "swollen or red tonsils": "ቲንሲሎች ማበጥ", "sneezing": "ማስነጠስ",
        "coryza": "አፍንጫ ፍሳሽ", "sinus congestion": "ሳይነስ መዘጋት",
        "painful sinuses": "ሳይነስ ህመም", "nosebleed": "አፍንጫ ደም",
        # Gastro
        "sharp abdominal pain": "ሹል የሆድ ህመም", "upper abdominal pain": "የላይ ሆድ ህመም",
        "burning abdominal pain": "የሚቃጠል የሆድ ህመም", "lower abdominal pain": "የታች ሆድ ህመም",
        "nausea": "ማቅለሽለሽ", "vomiting": "ማስታወክ", "vomiting blood": "ደም ማስታወክ",
        "diarrhea": "ተቅማጥ", "constipation": "ሆድ መጠፍጠፍ",
        "stomach bloating": "ሆድ ማበጥ", "heartburn": "ሆድ ማቃጠል",
        "regurgitation": "ምግብ መመለስ", "blood in stool": "ሰገራ ውስጥ ደም",
        "melena": "ጥቁር ሰገራ", "rectal bleeding": "የፊንጥ ደም",
        "changes in stool appearance": "ሰገራ ቀለም ለውጥ",
        "pain of the anus": "የፊንጥ ህመም",
        "mass or swelling around the anus": "የፊንጥ አካባቢ ማበጥ",
        "itching of the anus": "የፊንጥ ማሳከክ",
        # Pain
        "headache": "ራስ ምታት", "frontal headache": "ፊት ራስ ምታት",
        "back pain": "የጀርባ ህመም", "low back pain": "የታች ጀርባ ህመም",
        "neck pain": "የአንገት ህመም", "shoulder pain": "የትከሻ ህመም",
        "hip pain": "የጭን ህመም", "knee pain": "የጉልበት ህመም",
        "leg pain": "የእግር ህመም", "foot or toe pain": "የጣት/እግር ህመም",
        "ankle pain": "የቁርጭምጭሚት ህመም", "elbow pain": "የክርን ህመም",
        "arm pain": "የክንድ ህመም", "wrist pain": "የሚዳቃ ህመም",
        "hand or finger pain": "የእጅ/ጣት ህመም", "joint pain": "የመገጣጠሚያ ህመም",
        "rib pain": "የጎን ሳምባ ህመም", "groin pain": "የጉሮሮ ህመም",
        "suprapubic pain": "የታች ሆድ ህመም", "side pain": "የጎን ህመም",
        "facial pain": "የፊት ህመም", "bones are painful": "አጥንቶች ህማም",
        "back cramps or spasms": "የጀርባ ቁርጠት", "cramps and spasms": "ቁርጠት",
        # Musculo
        "arm stiffness or tightness": "ክንድ ድርቀት", "arm swelling": "ክንድ ማበጥ",
        "arm weakness": "ክንድ ድክመት", "arm lump or mass": "ክንድ እብጠት",
        "wrist swelling": "ሚዳቃ ማበጥ",
        "hand or finger swelling": "እጅ/ጣት ማበጥ",
        "hand or finger stiffness or tightness": "እጅ/ጣት ድርቀት",
        "hand or finger weakness": "እጅ/ጣት ድክመት",
        "hand or finger lump or mass": "እጅ/ጣት እብጠት",
        "knee swelling": "ጉልበት ማበጥ", "knee stiffness or tightness": "ጉልበት ድርቀት",
        "leg swelling": "እግር ማበጥ", "leg weakness": "እግር ድክመት",
        "ankle swelling": "ቁርጭምጭሚት ማበጥ", "foot or toe swelling": "ጣት/እግር ማበጥ",
        "elbow swelling": "ክርን ማበጥ",
        "shoulder stiffness or tightness": "ትከሻ ድርቀት",
        "hip stiffness or tightness": "ጭን ድርቀት",
        "back stiffness or tightness": "ጀርባ ድርቀት",
        "neck mass": "አንገት እብጠት", "neck swelling": "አንገት ማበጥ",
        "peripheral edema": "ዳርቻ ማበጥ", "problems with movement": "እንቅስቃሴ ችግር",
        # Skin
        "abnormal appearing skin": "ቆዳ ለውጥ", "skin lesion": "ቆዳ ቁስለት",
        "acne or pimples": "ሽፍታ/ፊጥ", "skin growth": "ቆዳ ዕድገት",
        "skin moles": "የቆዳ ምልክቶች", "warts": "ዕጢ",
        "skin rash": "ቆዳ ሽፍታ", "itching of skin": "ቆዳ ማሳከክ",
        "skin dryness, peeling, scaliness, or roughness": "ቆዳ ደረቅ/መላጥ",
        "skin irritation": "ቆዳ ቅርጫ", "itchy scalp": "ጭንቅላት ማሳከክ",
        "irregular appearing scalp": "ጭንቅላት ለውጥ", "jaundice": "ቢጫ በሽታ",
        "diaper rash": "ዳይፐር ሽፍታ", "eyelid lesion or rash": "ዐይን ሽፍታ",
        "irregular appearing nails": "ጥፍር ለውጥ",
        # Eye
        "diminished vision": "ዕይታ መቀነስ", "double vision": "ድርብ ዕይታ",
        "symptoms of eye": "የዓይን ምልክቶች", "pain in eye": "የዓይን ህመም",
        "abnormal movement of eyelid": "የዐይን ሽፋን እንቅስቃሴ",
        "foreign body sensation in eye": "ዓይን ውስጥ ነገር ስሜት",
        "eye redness": "ቀይ ዓይን", "lacrimation": "ዕንባ",
        "itchiness of eye": "ዓይን ማሳከክ", "blindness": "ዓይነ ስውርነት",
        "eye burns or stings": "ዓይን ማቃጠል",
        "spots or clouds in vision": "ዕይታ ደበዘዘ", "bleeding from eye": "ዓይን ደም",
        "mass on eyelid": "የዐይን ሽፋን እብጠት", "swollen eye": "ዓይን ማበጥ",
        "eyelid swelling": "የዐይን ሽፋን ማበጥ", "white discharge from eye": "ዓይን ፈሳሽ",
        # Urinary/Repro
        "painful urination": "ሽንት ሲሸኑ ህመም", "frequent urination": "ተደጋጋሚ ሽንት",
        "involuntary urination": "ያለፈቃድ ሽንት", "blood in urine": "ሽንት ውስጥ ደም",
        "retention of urine": "ሽንት ማቆር", "unusual color or odor to urine": "ሽንት ቀለም/ሽታ ለውጥ",
        "excessive urination at night": "ሌሊት ሽንት", "low urine output": "ሽንት መቀነስ",
        "hesitancy": "ሽንት ማቅማማት",
        "symptoms of bladder": "የፊኛ ምልክቶች",
        "symptoms of the kidneys": "የኩላሊት ምልክቶች", "kidney mass": "ኩላሊት እብጠት",
        "symptoms of prostate": "የፕሮስቴት ምልክቶች",
        "vaginal itching": "ብልት ማሳከክ", "vaginal discharge": "ብልት ፍሳሽ",
        "vaginal pain": "ብልት ህመም", "vaginal redness": "ብልት ቀያ",
        "pain during intercourse": "ወሲብ ጊዜ ህመም", "impotence": "ወሲብ ድካም",
        "symptoms of the scrotum and testes": "የፍልፈል ምልክቶች",
        "swelling of scrotum": "ፍልፈል ማበጥ", "pain in testicles": "ፍልፈል ህመም",
        # Women's Health
        "hot flashes": "ሙቀት ፍላጻ", "intermenstrual bleeding": "ወር አበባ መካከል ደም",
        "pain during pregnancy": "እርግዝና ጊዜ ህመም",
        "problems during pregnancy": "እርግዝና ጊዜ ችግሮች",
        "spotting or bleeding during pregnancy": "እርግዝና ጊዜ ደም",
        "uterine contractions": "የማህፀን መኮማተር", "recent pregnancy": "ቅርብ ጊዜ እርግዝና",
        "pelvic pain": "የዳሌ ህመም",
        "long menstrual periods": "ረዥም ወር አበባ",
        "heavy menstrual flow": "ከባድ ወር አበባ", "unpredictable menstruation": "ያልተስተካከለ ወር አበባ",
        "painful menstruation": "ወር አበባ ህመም", "infertility": "መካንነት",
        "frequent menstruation": "ተደጋጋሚ ወር አበባ",
        "blood clots during menstrual periods": "ወር አበባ ደም ስብ",
        # Pediatric
        "lack of growth": "እድገት ማጣት", "irritable infant": "ቅሬታ ሕፃን",
        "infant feeding problem": "ሕፃን ምግብ ችግር", "pulling at ears": "ጆሮ መጎተት",
        # Other
        "back mass or lump": "ጀርባ እብጠት", "jaw swelling": "መንጋጋ ማበጥ",
        "lip swelling": "ከንፈር ማበጥ", "toothache": "ጥርስ ህመም",
        "mouth ulcer": "አፍ ቁስለት", "gum pain": "ድድ ህመም",
        "mouth dryness": "አፍ ደረቅ", "mouth pain": "አፍ ህመም",
        "bleeding gums": "ድድ ደም", "pain in gums": "ድድ ህመም",
        "allergic reaction": "አለርጂ", "symptoms of the face": "የፊት ምልክቶች",
        "lower body pain": "የታች አካል ህመም",
    }

    CAT_AM = {
        "🌡️ General": "አጠቃላይ",
        "🧠 Neuro/Mental": "ነርቭ/አዕምሮ",
        "🫀 Cardio/Resp": "ልብ/መተንፈሻ",
        "👃 ENT": "ጆሮ/አፍንጫ/ጉሮሮ",
        "🤢 Gastro": "የምግብ መፈጨት",
        "🤕 Pain": "ህመም",
        "🦴 Musculo/Limbs": "ጡንቻ/አካላት",
        "🩺 Skin": "ቆዳ",
        "👁️ Eye": "ዓይን",
        "🚻 Urinary/Repro": "ሽንት/ስሜት",
        "🤰 Women's Health": "የሴቶች ጤና",
        "👶 Pediatric": "ሕፃናት",
        "🔬 Other": "ሌላ",
    }

    is_am   = lang.lower() == "amharic"
    js_cats = {}
    for cat_en, symptoms in CATEGORIES_EN.items():
        label = CAT_AM.get(cat_en, cat_en) if is_am else cat_en
        js_cats[label] = [
            {"en": s, "display": SYMPTOM_AM.get(s, s) if is_am else s.title()}
            for s in symptoms
        ]

    current_val  = st.session_state.get("symptoms_text", "")
    current_list = [s.strip().lower() for s in current_val.split(",") if s.strip()]
    quick_label  = "ምልክቶችን ፈጥኖ ይምረጡ:" if is_am else "Quick-select symptoms:"
    or_text      = "ወይም ምልክቶችን ይተይቡ" if is_am else "or type symptoms below"

    html_code = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: transparent; font-family: 'Poppins','DM Sans',-apple-system,sans-serif; padding: 4px 2px 0; }}
.qs-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.38); font-weight: 600; margin-bottom: 10px; }}
.cat-tabs {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
.cat-tab {{ padding: 5px 12px; border-radius: 100px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.5); font-size: 0.72rem; font-weight: 600; cursor: pointer; white-space: nowrap; transition: all 0.18s ease; font-family: inherit; }}
.cat-tab:hover {{ border-color: #0d9488; color: #14b8a6; background: rgba(13,148,136,0.1); }}
.cat-tab.active {{ background: rgba(13,148,136,0.2); border-color: #14b8a6; color: #14b8a6; }}
.pills-wrap {{ display: flex; flex-wrap: wrap; gap: 7px; max-height: 118px; overflow-y: auto; padding: 2px 2px 6px; scrollbar-width: thin; scrollbar-color: rgba(13,148,136,0.45) transparent; }}
.pills-wrap::-webkit-scrollbar {{ width: 4px; }}
.pills-wrap::-webkit-scrollbar-thumb {{ background: rgba(13,148,136,0.45); border-radius: 4px; }}
.pill {{ padding: 5px 13px; border-radius: 100px; border: 1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.75); font-size: 0.77rem; cursor: pointer; white-space: nowrap; transition: all 0.16s ease; user-select: none; font-family: inherit; }}
.pill:hover {{ border-color: #14b8a6; color: #fff; background: rgba(13,148,136,0.12); transform: translateY(-1px); }}
.pill.sel {{ background: rgba(13,148,136,0.25) !important; border-color: #14b8a6 !important; color: #14b8a6 !important; font-weight: 600 !important; }}
.or-div {{ display: flex; align-items: center; gap: 10px; margin: 14px 0 2px; color: rgba(255,255,255,0.22); font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.1em; }}
.or-div::before, .or-div::after {{ content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.08); }}
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

function renderTabs() {{
  var el = document.getElementById('catTabs'); el.innerHTML = '';
  KEYS.forEach(function(k) {{
    var b = document.createElement('button');
    b.className = 'cat-tab' + (k === active ? ' active' : '');
    b.textContent = k;
    b.onclick = function() {{ active = k; renderTabs(); renderPills(); }};
    el.appendChild(b);
  }});
}}
function renderPills() {{
  var el = document.getElementById('pillsWrap'); el.innerHTML = '';
  (CATS[active] || []).forEach(function(item) {{
    var isSel = selected.indexOf(item.en.toLowerCase()) !== -1;
    var p = document.createElement('button');
    p.className = 'pill' + (isSel ? ' sel' : '');
    p.textContent = (isSel ? '✓ ' : '') + item.display;
    p.onclick = (function(sym) {{ return function() {{ toggleSym(sym); }}; }})(item.en);
    el.appendChild(p);
  }});
}}
function syncToStreamlit() {{
  var val = selected.join(', ');
  try {{
    var doc = window.parent.document;
    var areas = doc.querySelectorAll('textarea');
    for (var i = 0; i < areas.length; i++) {{
      var ta = areas[i];
      if (ta.placeholder && ta.placeholder.indexOf('headache') !== -1) {{
        var setter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
        setter.call(ta, val);
        ta.dispatchEvent(new window.parent.Event('input', {{ bubbles: true }}));
        break;
      }}
    }}
  }} catch(e) {{}}
}}
function toggleSym(sym) {{
  var lo = sym.toLowerCase();
  var idx = selected.indexOf(lo);
  if (idx === -1) {{ selected.push(lo); }} else {{ selected.splice(idx, 1); }}
  renderPills(); syncToStreamlit();
}}
renderTabs(); renderPills();
</script></body></html>"""

    components.html(html_code, height=220, scrolling=False)


# ──────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────
def clear_symptoms_callback():
    st.session_state["symptoms_text"]    = ""
    st.session_state["prediction_result"] = None
    st.session_state["prediction_error"]  = None

def clear_diagnosis_callback():
    st.session_state["prediction_result"] = None
    st.session_state["prediction_error"]  = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Sign Up / Login panel (no ngrok URL field; URL is hardcoded)
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar_auth(role: str, lang: str) -> str:
    """
    Renders the Sign Up / Login section in the sidebar for Student / Doctor roles.
    Returns the active user_id string (empty string if not logged in).
    The Colab API URL is hardcoded as COLAB_API_URL — no UI input required.
    """

    # ── Session state keys ───────────────────────────────────────────────────
    if "logged_in_as"   not in st.session_state: st.session_state["logged_in_as"]   = None
    if "auth_msg"       not in st.session_state: st.session_state["auth_msg"]        = ("", False)
    if "auth_mode"      not in st.session_state: st.session_state["auth_mode"]       = "login"

    # ── If already logged in as this role, show badge + logout ───────────────
    li = st.session_state.get("logged_in_as")
    if li and li.get("role") == role:
        uid = li["user_id"]
        st.sidebar.markdown(
            f"<div class='logged-in-badge'>✅ Logged in as {role}<br>"
            f"<span style='font-size:0.75rem;opacity:0.8'>ID: {uid}</span></div>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🚪 Log out", key="logout_btn", use_container_width=True):
            st.session_state["logged_in_as"] = None
            st.session_state["auth_msg"]     = ("", False)
            st.rerun()
        return uid

    # ── Mode toggle: Login / Sign Up ─────────────────────────────────────────
    mode_col1, mode_col2 = st.sidebar.columns(2)
    with mode_col1:
        if st.button("🔑 Login",   key="mode_login",  use_container_width=True):
            st.session_state["auth_mode"] = "login"
            st.session_state["auth_msg"]  = ("", False)
    with mode_col2:
        if st.button("📝 Sign Up", key="mode_signup", use_container_width=True):
            st.session_state["auth_mode"] = "signup"
            st.session_state["auth_msg"]  = ("", False)

    mode = st.session_state["auth_mode"]

    # ── ID format hint ────────────────────────────────────────────────────────
    prefix      = "ST" if role == "Student" else "DR"
    hint        = f"ID must start with <b>{prefix}</b> — e.g. {prefix}001"
    placeholder = f"e.g. {prefix}001"

    st.sidebar.markdown(
        f"<div class='auth-info' style='margin-bottom:6px'>{hint}</div>",
        unsafe_allow_html=True,
    )

    if mode == "signup":
        # ── Sign Up form ──────────────────────────────────────────────────────
        su_id   = st.sidebar.text_input("Choose an ID",  placeholder=placeholder, key="su_id_field")
        su_name = st.sidebar.text_input("Your name (optional)", placeholder="Full name", key="su_name_field")
        if st.sidebar.button("✅ Create Account", key="signup_submit_btn", use_container_width=True):
            if not su_id.strip():
                st.session_state["auth_msg"] = ("Please enter an ID.", False)
            else:
                ok, msg = _api_signup(role, su_id.strip(), su_name.strip())
                st.session_state["auth_msg"] = (msg, ok)
                if ok:
                    # Auto-login after successful signup
                    st.session_state["logged_in_as"] = {
                        "role": role, "user_id": su_id.strip().upper()
                    }
                    st.rerun()
    else:
        # ── Login form ────────────────────────────────────────────────────────
        li_id = st.sidebar.text_input("Your ID", placeholder=placeholder,
                                      type="password", key="li_id_field")
        if st.sidebar.button("🔓 Login", key="login_submit_btn", use_container_width=True):
            if not li_id.strip():
                st.session_state["auth_msg"] = ("Please enter your ID.", False)
            else:
                ok, msg = _api_login(role, li_id.strip())
                st.session_state["auth_msg"] = (msg, ok)
                if ok:
                    st.session_state["logged_in_as"] = {
                        "role": role, "user_id": li_id.strip().upper()
                    }
                    st.rerun()

    # ── Auth feedback message ─────────────────────────────────────────────────
    msg_text, msg_ok = st.session_state["auth_msg"]
    if msg_text:
        css = "auth-success" if msg_ok else "auth-error"
        st.sidebar.markdown(
            f"<div class='{css}'>{msg_text}</div>",
            unsafe_allow_html=True,
        )

    return ""   # not logged in yet


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    missing = check_files()
    if missing:
        st.error(f"Missing required files: {missing}")
        return

    main_df, desc_map, diets_map, meds_map, precs_map, workout_map = load_data()
    svc, dt, le = load_models()

    symptom_list  = tuple(main_df.drop(columns=["diseases"]).columns)
    disease_names = tuple(main_df["diseases"].unique())
    vec, sym_matrix, _ = build_tfidf_index(symptom_list, disease_names)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown('<div class="section-header">User Profile</div>', unsafe_allow_html=True)
    lang = st.sidebar.selectbox("🌐 Language", ["English", "Amharic"])
    role = st.sidebar.selectbox("👤 Role", ["Normal User", "Student", "Doctor"])
    age  = st.sidebar.number_input("🎂 Age", min_value=0, max_value=120, value=25)

    # ── Auth panel — only for Student / Doctor ───────────────────────────────
    user_id = ""
    if role in ("Student", "Doctor"):
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f'<div class="section-header">{"Student" if role=="Student" else "Doctor"} Access</div>',
            unsafe_allow_html=True,
        )
        user_id = render_sidebar_auth(role, lang)

    # ── Role badge ────────────────────────────────────────────────────────────
    role_colors = {"Doctor": "#1f6feb", "Student": "#2ea043", "Normal User": "#6e7681"}
    rc = role_colors.get(role, "#6e7681")
    st.sidebar.markdown(
        f"<div style='margin-top:8px'>"
        f"<span class='role-badge' style='background:rgba({int(rc[1:3],16)},{int(rc[3:5],16)},{int(rc[5:7],16)},0.2);"
        f"color:{rc};border:1px solid {rc}40'>● {role}</span></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("⚕️ General health information only. Always consult a qualified doctor.")

    # ── Session state init ────────────────────────────────────────────────────
    for key in ("prediction_result", "prediction_error", "chat_response"):
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='main-header'>
      <div class='main-header-title'>🏥 Integrated Healthcare Dashboard</div>
      <div class='main-header-subtitle'>Disease Prediction · Health Recommendations · AI Chatbot</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        t("Disease Predictor",  lang),
        t("Health Recommender", lang),
        t("Healthcare Chatbot", lang),
    ])

    # ══════════════════════════════════════════
    # TAB 1 — DISEASE PREDICTOR
    # ══════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">🩺 Symptom-Based Disease Predictor</div>',
                    unsafe_allow_html=True)
        render_quick_select_symptoms(lang)

        user_input = st.text_area(
            "Symptoms",
            key="symptoms_text",
            placeholder="e.g., headache, fever, chills",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"✦ {t('Predict & Recommend', lang)}", use_container_width=True):
                if not user_input.strip():
                    st.warning(t("Please enter symptoms.", lang))
                else:
                    with st.spinner("🔍 Analysing symptoms…"):
                        res, err = integrated_prediction_system(
                            user_input, age, role, user_id, lang,
                            main_df, le, svc, dt,
                            desc_map, diets_map, meds_map, precs_map, workout_map,
                            vec, sym_matrix,
                        )
                    st.session_state["prediction_result"] = res
                    st.session_state["prediction_error"]  = err

        with col2:
            st.markdown('<div class="clear-btn-container">', unsafe_allow_html=True)
            st.button(t("Clear Symptoms", lang), key="clear_sym",
                      on_click=clear_symptoms_callback, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state["prediction_error"]:
            st.markdown(
                f'<div class="access-denied">{st.session_state["prediction_error"]}</div>',
                unsafe_allow_html=True,
            )

        elif st.session_state["prediction_result"]:
            res = st.session_state["prediction_result"]
            st.markdown("---")

            st.markdown('<div class="section-header">✅ Matched Symptoms</div>',
                        unsafe_allow_html=True)
            chips = " · ".join(
                f"<span style='background:rgba(88,166,255,0.15);padding:4px 12px;"
                f"border-radius:6px;font-size:0.85rem;color:#58a6ff;"
                f"border:1px solid rgba(88,166,255,0.3)'>{s}</span>"
                for s in res["matched_symptoms"]
            )
            st.markdown(chips, unsafe_allow_html=True)

            st.markdown('<br><div class="section-header">🎯 Predicted Conditions</div>',
                        unsafe_allow_html=True)
            rows = (
                "<div style='background:rgba(22,27,34,0.7);border:1px solid rgba(88,166,255,0.2);"
                "border-radius:10px;padding:14px 16px;margin-bottom:8px'>"
                "<div style='color:#79c0ff;font-size:0.75rem;font-weight:700;"
                "letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px'>"
                "🤖 SVC — Top Predictions</div>"
            )
            for rank, p in enumerate(res["predicted_conditions"]):
                bg  = "linear-gradient(135deg,#1f6feb,#388bfd)" if rank == 0 else "rgba(33,38,45,0.8)"
                col = "#fff" if rank == 0 else "#8b949e"
                bar = p["confidence"].replace("%", "") if p["confidence"] != "N/A" else "0"
                rows += (
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
                    f"<span style='min-width:18px;color:#8b949e;font-size:0.75rem'>#{rank+1}</span>"
                    f"<span style='flex:1;background:{bg};color:{col};padding:5px 12px;"
                    f"border-radius:16px;font-size:0.85rem;font-weight:600'>{p['disease']}</span>"
                    f"<span style='min-width:48px;text-align:right;color:#58a6ff;"
                    f"font-size:0.82rem;font-weight:700'>{p['confidence']}</span></div>"
                    f"<div style='height:3px;background:rgba(48,54,61,0.5);border-radius:2px;margin-bottom:6px'>"
                    f"<div style='height:3px;width:{bar}%;background:linear-gradient(90deg,#1f6feb,#58a6ff);"
                    f"border-radius:2px'></div></div>"
                )
            st.markdown(rows + "</div>", unsafe_allow_html=True)

            with st.expander(f"📊 Health Plan — {res['top_disease']}", expanded=True):
                render_rec_cards(res["rec_cards"])
                if res["advice"]:
                    st.markdown(
                        f'<div class="advice-banner">{res["advice"]}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="clear-btn-container" style="margin-top:16px;max-width:200px">',
                        unsafe_allow_html=True)
            st.button(t("Clear Diagnosis", lang), key="clear_diag",
                      on_click=clear_diagnosis_callback, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # TAB 2 — HEALTH RECOMMENDER
    # ══════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">🏥 Select a Disease</div>',
                    unsafe_allow_html=True)
        sorted_diseases  = sorted(list(desc_map.keys()))
        selected_disease = st.selectbox(
            "Disease", options=sorted_diseases,
            format_func=lambda x: x.title(),
            label_visibility="collapsed",
        )

        col_r1, col_r2 = st.columns([1, 5])
        with col_r1:
            get_plan = st.button(t("Get Plan", lang), use_container_width=True)

        if get_plan:
            ok, err_msg = check_access(age, role, user_id, lang)
            if not ok:
                st.markdown(f'<div class="access-denied">{err_msg}</div>',
                            unsafe_allow_html=True)
            else:
                key = clean_disease_name(selected_disease)
                cards, advice = role_based_recs(
                    role, lang, key,
                    desc_map, diets_map, meds_map, precs_map, workout_map,
                )
                with st.expander(f"💊 Health Plan — {selected_disease.title()}", expanded=True):
                    render_rec_cards(cards)
                    if advice:
                        st.markdown(
                            f'<div class="advice-banner">{advice}</div>',
                            unsafe_allow_html=True,
                        )

    # ══════════════════════════════════════════
    # TAB 3 — CHATBOT
    # ══════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">💬 Healthcare Chatbot</div>',
                    unsafe_allow_html=True)

        greeting = t("Hello! How can I help you with health information today?", lang)
        st.markdown(f'<div class="chat-bot">🤖 {greeting}</div>', unsafe_allow_html=True)

        st.markdown(
            "<div style='margin:10px 0 4px;font-size:0.72rem;color:#475569;"
            "text-transform:uppercase;letter-spacing:0.1em;font-weight:600'>"
            "Example queries:</div>"
            "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:12px'>"
            "• <em>What is allergy?</em> &nbsp;|&nbsp; "
            "• <em>What diet should I follow for asthma?</em> &nbsp;|&nbsp; "
            "• <em>What medications are used for allergy?</em> &nbsp;|&nbsp; "
            "• <em>What precautions for Common Cold?</em></div>",
            unsafe_allow_html=True,
        )

        chat_query = st.text_input(
            "Your question:",
            placeholder="e.g. What diet should I follow for Asthma?",
            key="chat_query",
            label_visibility="collapsed",
        )

        col_c1, col_c2 = st.columns([1, 6])
        with col_c1:
            ask_btn = st.button(t("Ask Bot", lang), use_container_width=True)

        with col_c2:
            st.markdown('<div class="clear-btn-container">', unsafe_allow_html=True)
            if st.button("🗑️ Clear", key="chat_clear", use_container_width=True):
                st.session_state["chat_response"] = None
            st.markdown('</div>', unsafe_allow_html=True)

        if ask_btn:
            if not chat_query.strip():
                st.warning(t("Please enter a query.", lang))
            else:
                with st.spinner("🤔 Thinking…"):
                    reply = chatbot_response(
                        chat_query, age, role, user_id, lang,
                        desc_map, diets_map, meds_map, precs_map, workout_map,
                        vec,
                    )
                st.session_state["chat_response"] = reply

        if st.session_state.get("chat_response"):
            st.markdown(
                f'<div class="chat-bot">🤖 {st.session_state["chat_response"]}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
