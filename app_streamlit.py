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

# ── Dark and Light Mode Configuration ──────────────────────────────────────────
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"

with st.sidebar:
    st.session_state["theme_mode"] = st.selectbox(
        "🌓 Choose Theme Theme / ጭብጥ ይምረጡ",
        ["Dark", "Light"],
        index=0 if st.session_state["theme_mode"] == "Dark" else 1
    )

if st.session_state["theme_mode"] == "Dark":
    # Dark Mode Styles
    theme_bg = "linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1f2d 100%)"
    theme_text = "#e6edf3"
    sidebar_bg = "linear-gradient(180deg, #161b22 0%, #0f1419 100%)"
    sidebar_border = "rgba(48,54,61,0.5)"
    sidebar_text = "#e6edf3 !important"
    input_bg = "rgba(33,38,45,0.8) !important"
    input_border = "rgba(48,54,61,0.5) !important"
    card_bg = "linear-gradient(135deg, rgba(22,27,34,0.8) 0%, rgba(33,38,45,0.6) 100%)"
    card_text = "#c9d1d9"
else:
    # Light Mode Styles
    theme_bg = "linear-gradient(135deg, #f0f2f5 0%, #ffffff 50%, #eef2f7 100%)"
    theme_text = "#1f2937"
    sidebar_bg = "linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%)"
    sidebar_border = "rgba(209,213,219,0.8)"
    sidebar_text = "#1f2937 !important"
    input_bg = "#ffffff !important"
    input_border = "rgba(156,163,175,0.8) !important"
    card_bg = "linear-gradient(135deg, #ffffff 0%, #f9fafb 100%)"
    card_text = "#374151"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Poppins:wght@300;400;600;700&display=swap');

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes slideInLeft {{
    from {{ opacity: 0; transform: translateX(-30px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif;
    scroll-behavior: smooth;
}}
.stApp {{
    background: {theme_bg};
    color: {theme_text};
    animation: fadeInUp 0.8s ease-out;
}}
section[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    border-right: 1px solid {sidebar_border};
}}
section[data-testid="stSidebar"] * {{ color: {sidebar_text}; }}

.stTabs [data-baseweb="tab-list"] {{
    background: rgba(22,27,34,0.6);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid rgba(48,54,61,0.3);
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: #8b949e;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 10px 24px;
    border: none !important;
    transition: all 0.3s ease;
    font-size: 0.95rem;
}}
.stTabs [data-baseweb="tab"]:hover {{ background: rgba(88,166,255,0.1); color: #58a6ff; }}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #21262d 0%, #1f6feb 100%) !important;
    color: #fff !important;
    box-shadow: 0 4px 12px rgba(88,166,255,0.25);
}}

.stTextInput input, .stSelectbox select, .stTextArea textarea,
div[data-baseweb="select"] > div {{
    background: {input_bg};
    border: {input_border};
    color: {theme_text} !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    font-size: 0.95rem !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus,
div[data-baseweb="select"] > div:focus-within {{
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.2) !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.3s ease;
    font-size: 0.95rem;
    width: 100%;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(46,160,67,0.4);
}}

div.clear-btn-container > div > button {{
    background: linear-gradient(135deg, #21262d 0%, #30363d 100%) !important;
    color: #f85149 !important;
    border: 1px solid rgba(248,81,73,0.4) !important;
}}
div.clear-btn-container > div > button:hover {{
    background: rgba(248,81,73,0.1) !important;
    box-shadow: 0 8px 24px rgba(248,81,73,0.2) !important;
}}

.result-card {{
    background: {card_bg};
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    animation: fadeInUp 0.6s ease-out;
    transition: all 0.3s ease;
}}
.result-card:hover {{
    border-color: rgba(88,166,255,0.4);
    box-shadow: 0 8px 32px rgba(88,166,255,0.15);
    transform: translateY(-4px);
}}
.result-card h4 {{
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}}
.result-card p, .result-card li {{ color: {card_text}; font-size: 0.95rem; line-height: 1.6; }}

.result-card-locked {{
    background: linear-gradient(135deg, rgba(45,14,14,0.5) 0%, rgba(33,38,45,0.4) 100%);
    border: 1px solid rgba(248,81,73,0.25);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}}
.result-card-locked h4 {{
    color: #f85149;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}}
.result-card-locked p {{ color: #8b949e; font-size: 0.95rem; line-height: 1.6; font-style: italic; }}

.result-card-limited {{
    background: {card_bg};
    border: 1px solid rgba(240,136,62,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    animation: fadeInUp 0.6s ease-out;
}}
.result-card-limited h4 {{
    color: #f0883e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}}
.result-card-limited p {{ color: {card_text}; font-size: 0.95rem; line-height: 1.6; }}
.result-card-limited .restriction-note {{
    color: #f0883e;
    font-size: 0.8rem;
    margin-top: 8px;
    font-style: italic;
}}

.disease-badge {{
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
}}
.conf-pill {{
    display: inline-block;
    background: linear-gradient(135deg, rgba(33,38,45,0.9) 0%, rgba(88,166,255,0.15) 100%);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 16px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #58a6ff;
    margin-left: 4px;
    font-weight: 600;
}}

.advice-banner {{
    background: linear-gradient(135deg, rgba(45,24,0,0.8) 0%, rgba(240,136,62,0.1) 100%);
    border-left: 4px solid #f0883e;
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    color: #f0883e;
    font-size: 0.9rem;
    margin-top: 16px;
    animation: slideInLeft 0.5s ease-out;
}}

.chat-bot {{
    background: linear-gradient(135deg, rgba(33,38,45,0.9) 0%, rgba(31,111,235,0.1) 100%);
    border-left: 4px solid #58a6ff;
    padding: 16px 20px;
    border-radius: 0 12px 12px 0;
    color: {card_text};
    font-size: 0.95rem;
    margin-top: 12px;
    animation: slideInLeft 0.5s ease-out;
    line-height: 1.6;
}}

.section-header {{
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
}}
.access-denied {{
    background: linear-gradient(135deg, rgba(45,14,14,0.8) 0%, rgba(248,81,73,0.1) 100%);
    border-left: 4px solid #f85149;
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    color: #f85149;
    animation: slideInLeft 0.5s ease-out;
}}
.main-header {{ text-align: center; animation: fadeInUp 0.8s ease-out; margin-bottom: 32px; }}
.main-header-title {{
    font-family: 'Poppins', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #79c0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.main-header-subtitle {{ color: #8b949e; font-size: 1rem; letter-spacing: 0.05em; margin-top: 8px; }}
.role-badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}}
hr {{ border-color: rgba(88,166,255,0.1) !important; }}

.stNumberInput input {{
    background: rgba(22, 30, 40, 1) !important;
    color: #e6edf3 !important;
    border: 1px solid rgba(88, 166, 255, 0.35) !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}}
.stNumberInput input:focus {{
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.2) !important;
}}
.stNumberInput button {{
    background: rgba(33, 38, 45, 1) !important;
    color: #e6edf3 !important;
    border-color: rgba(88, 166, 255, 0.25) !important;
}}

/* ── Auth panel styles ── */
.auth-panel {{
    background: rgba(22,27,34,0.7);
    border: 1px solid rgba(48,54,61,0.6);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}}
.auth-success {{
    background: rgba(35,134,54,0.15);
    border-left: 3px solid #2ea043;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    color: #3fb950;
    font-size: 0.82rem;
    margin-top: 6px;
}}
.auth-error {{
    background: rgba(248,81,73,0.1);
    border-left: 3px solid #f85149;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    color: #f85149;
    font-size: 0.82rem;
    margin-top: 6px;
}}
.auth-info {{
    color: #8b949e;
    font-size: 0.72rem;
    margin-top: 6px;
    font-style: italic;
}}
.logged-in-badge {{
    background: rgba(35,134,54,0.2);
    border: 1px solid rgba(46,160,67,0.4);
    border-radius: 8px;
    padding: 8px 12px;
    color: #3fb950;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 8px;
}}
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
             "note": meds_note,                                                "card_type": "limited"},
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
        ]
    }
