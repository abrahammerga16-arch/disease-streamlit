"""
Integrated Healthcare Dashboard — Streamlit version
  • Disease Predictor  (symptom input → ML prediction)
  • Health Recommender (disease dropdown → full health plan)
  • Healthcare Chatbot  (natural-language query → semantic answer)
  • Role / Age / ID access control  ← signup/login via Supabase 2.x
  • English ↔ Amharic UI
  • NLP paragraph symptom extraction
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE  —  v2.x API  (create_client returns a single Client object)
# ══════════════════════════════════════════════════════════════════════════════
from supabase import create_client, Client

SUPABASE_URL: str = st.secrets["https://wpzueewhzimmtrtjdmoe.supabase.co"]
SUPABASE_KEY: str = st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndwenVlZXdoemltbXRydGpkbW9lIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA3Mjk1MDUsImV4cCI6MjA5NjMwNTUwNX0.XZVyzqrvGkO2xJg3jw1IBcSHIpf-m3Uua40Sz2UE7w8"]
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _supabase_signup(role: str, user_id: str, name: str = "") -> tuple[bool, str]:
    uid = user_id.strip().upper()
    if not uid:
        return False, "User ID cannot be empty."
    try:
        sb = get_supabase()
        existing = sb.table("app_users").select("user_id").eq("user_id", uid).execute()
        if existing.data:
            return False, "⚠️ User ID already exists. Please log in instead."
        sb.table("app_users").insert({
            "user_id": uid,
            "role":    role,
            "name":    name.strip(),
        }).execute()
        return True, f"✅ Account created! Your ID: **{uid}**"
    except Exception as e:
        return False, f"❌ Signup error: {e}"


def _supabase_login(role: str, user_id: str) -> tuple[bool, str]:
    uid = user_id.strip().upper()
    if not uid:
        return False, "User ID cannot be empty."
    try:
        sb = get_supabase()
        result = (
            sb.table("app_users")
            .select("user_id, name")
            .eq("user_id", uid)
            .eq("role", role)
            .execute()
        )
        if result.data:
            name = result.data[0].get("name", "")
            greeting = f" ({name})" if name else ""
            return True, f"✅ Welcome back{greeting}!"
        return False, "❌ Invalid ID or role mismatch."
    except Exception as e:
        return False, f"❌ Login error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Integrated Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Poppins:wght@300;400;600;700&display=swap');
@keyframes fadeInUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
@keyframes slideInLeft { from{opacity:0;transform:translateX(-30px)} to{opacity:1;transform:translateX(0)} }
html,body,[class*="css"]{ font-family:'Poppins',sans-serif; scroll-behavior:smooth; }
.stApp{ background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1f2d 100%); color:#e6edf3; animation:fadeInUp 0.8s ease-out; }
section[data-testid="stSidebar"]{ background:linear-gradient(180deg,#161b22 0%,#0f1419 100%); border-right:1px solid rgba(48,54,61,0.5); }
section[data-testid="stSidebar"] *{ color:#e6edf3 !important; }
.stTabs [data-baseweb="tab-list"]{ background:rgba(22,27,34,0.6); backdrop-filter:blur(10px); border-radius:12px; padding:6px; gap:4px; border:1px solid rgba(48,54,61,0.3); }
.stTabs [data-baseweb="tab"]{ background:transparent; color:#8b949e; border-radius:8px; font-weight:600; letter-spacing:0.04em; padding:10px 24px; border:none !important; transition:all 0.3s ease; font-size:0.95rem; }
.stTabs [data-baseweb="tab"]:hover{ background:rgba(88,166,255,0.1); color:#58a6ff; }
.stTabs [aria-selected="true"]{ background:linear-gradient(135deg,#21262d 0%,#1f6feb 100%) !important; color:#fff !important; box-shadow:0 4px 12px rgba(88,166,255,0.25); }
.stTextInput input,.stSelectbox select,.stTextArea textarea,div[data-baseweb="select"]>div{ background:rgba(33,38,45,0.8) !important; border:1px solid rgba(48,54,61,0.5) !important; color:#e6edf3 !important; border-radius:8px !important; transition:all 0.3s ease !important; font-size:0.95rem !important; }
.stTextInput input:focus,.stTextArea textarea:focus,div[data-baseweb="select"]>div:focus-within{ border-color:#58a6ff !important; box-shadow:0 0 0 3px rgba(88,166,255,0.2) !important; }
.stButton>button{ background:linear-gradient(135deg,#238636 0%,#2ea043 100%); color:#fff; border:none; border-radius:8px; font-weight:600; padding:10px 24px; transition:all 0.3s ease; font-size:0.95rem; width:100%; }
.stButton>button:hover{ transform:translateY(-2px); box-shadow:0 8px 24px rgba(46,160,67,0.4); }
div.clear-btn-container>div>button{ background:linear-gradient(135deg,#21262d 0%,#30363d 100%) !important; color:#f85149 !important; border:1px solid rgba(248,81,73,0.4) !important; }
div.clear-btn-container>div>button:hover{ background:rgba(248,81,73,0.1) !important; box-shadow:0 8px 24px rgba(248,81,73,0.2) !important; }
.result-card{ background:linear-gradient(135deg,rgba(22,27,34,0.8) 0%,rgba(33,38,45,0.6) 100%); border:1px solid rgba(88,166,255,0.2); border-radius:12px; padding:20px 24px; margin-bottom:16px; animation:fadeInUp 0.6s ease-out; transition:all 0.3s ease; }
.result-card:hover{ border-color:rgba(88,166,255,0.4); box-shadow:0 8px 32px rgba(88,166,255,0.15); transform:translateY(-4px); }
.result-card h4{ color:#58a6ff; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:12px; }
.result-card p,.result-card li{ color:#c9d1d9; font-size:0.95rem; line-height:1.6; }
.result-card-locked{ background:linear-gradient(135deg,rgba(45,14,14,0.5) 0%,rgba(33,38,45,0.4) 100%); border:1px solid rgba(248,81,73,0.25); border-radius:12px; padding:20px 24px; margin-bottom:16px; }
.result-card-locked h4{ color:#f85149; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:12px; }
.result-card-locked p{ color:#8b949e; font-size:0.95rem; line-height:1.6; font-style:italic; }
.result-card-limited{ background:linear-gradient(135deg,rgba(22,27,34,0.8) 0%,rgba(33,38,45,0.6) 100%); border:1px solid rgba(240,136,62,0.3); border-radius:12px; padding:20px 24px; margin-bottom:16px; animation:fadeInUp 0.6s ease-out; }
.result-card-limited h4{ color:#f0883e; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:12px; }
.result-card-limited p{ color:#c9d1d9; font-size:0.95rem; line-height:1.6; }
.disease-badge{ display:inline-block; background:linear-gradient(135deg,#1f6feb 0%,#388bfd 100%); color:#fff; border-radius:24px; padding:6px 18px; font-weight:600; font-size:1rem; margin:4px 8px 4px 0; animation:slideInLeft 0.4s ease-out; box-shadow:0 4px 12px rgba(31,111,235,0.3); }
.advice-banner{ background:linear-gradient(135deg,rgba(45,24,0,0.8) 0%,rgba(240,136,62,0.1) 100%); border-left:4px solid #f0883e; padding:12px 18px; border-radius:0 8px 8px 0; color:#f0883e; font-size:0.9rem; margin-top:16px; animation:slideInLeft 0.5s ease-out; }
.chat-bot{ background:linear-gradient(135deg,rgba(33,38,45,0.9) 0%,rgba(31,111,235,0.1) 100%); border-left:4px solid #58a6ff; padding:16px 20px; border-radius:0 12px 12px 0; color:#c9d1d9; font-size:0.95rem; margin-top:12px; animation:slideInLeft 0.5s ease-out; line-height:1.6; }
.section-header{ color:#58a6ff; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; letter-spacing:0.15em; text-transform:uppercase; border-bottom:1px solid rgba(88,166,255,0.2); padding-bottom:8px; margin-bottom:18px; animation:slideInLeft 0.4s ease-out; font-weight:700; }
.access-denied{ background:linear-gradient(135deg,rgba(45,14,14,0.8) 0%,rgba(248,81,73,0.1) 100%); border-left:4px solid #f85149; padding:12px 18px; border-radius:0 8px 8px 0; color:#f85149; animation:slideInLeft 0.5s ease-out; }
.main-header{ text-align:center; animation:fadeInUp 0.8s ease-out; margin-bottom:32px; }
.main-header-title{ font-family:'Poppins',sans-serif; font-size:2.8rem; font-weight:700; background:linear-gradient(135deg,#58a6ff,#79c0ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.main-header-subtitle{ color:#8b949e; font-size:1rem; letter-spacing:0.05em; margin-top:8px; }
.role-badge{ display:inline-block; padding:3px 12px; border-radius:12px; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; }
hr{ border-color:rgba(88,166,255,0.1) !important; }
.stNumberInput input{ background:rgba(22,30,40,1) !important; color:#e6edf3 !important; border:1px solid rgba(88,166,255,0.35) !important; border-radius:8px !important; font-size:0.95rem !important; }
.auth-success{ background:rgba(35,134,54,0.15); border-left:3px solid #2ea043; border-radius:0 6px 6px 0; padding:8px 12px; color:#3fb950; font-size:0.82rem; margin-top:6px; }
.auth-error{ background:rgba(248,81,73,0.1); border-left:3px solid #f85149; border-radius:0 6px 6px 0; padding:8px 12px; color:#f85149; font-size:0.82rem; margin-top:6px; }
.logged-in-badge{ background:rgba(35,134,54,0.2); border:1px solid rgba(46,160,67,0.4); border-radius:8px; padding:8px 12px; color:#3fb950; font-size:0.8rem; font-weight:600; margin-bottom:8px; }
.nlp-info-banner{ background:linear-gradient(135deg,rgba(31,111,235,0.08) 0%,rgba(33,38,45,0.6) 100%); border-left:3px solid rgba(88,166,255,0.5); padding:10px 14px; border-radius:0 8px 8px 0; color:#8b949e; font-size:0.82rem; margin-bottom:10px; line-height:1.5; }
.nlp-info-banner span{ color:#58a6ff; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════════
AMHARIC = {
    "Disease":"በሽታ","Description":"መግለጫ","Dietary Plan":"የአመጋገብ እቅድ",
    "Medications":"መድሃኒቶች","Workout/Activity":"ስፖርት/እንቅስቃሴ","Precautions":"ጥንቃቄዎች",
    "Hello! How can I help you with health information today?":"ሰላም! ዛሬ በጤና መረጃ እንዴት ልረዳዎ እችላለሁ?",
    "Access Denied: Information is not available for users under 18.":"የመዳረሻ እገዳ: ከ18 ዓመት በታች ለሆኑ ተጠቃሚዎች መረጃ አይገኝም",
    "Disease Predictor":"ምልክት መተንበይ","Health Recommender":"የጤና ምክር ሰጪ",
    "Healthcare Chatbot":"የጤና እንክብካቤ ቻትቦት","Predict & Recommend":"መተንበይ እና መምከር",
    "Get Plan":"እቅድ ያግኙ","Ask Bot":"ቦት ይጠይቁ","Clear Symptoms":"ምልክቶችን አጽዳ",
    "Clear Diagnosis":"ውጤቶችን አጽዳ","Please enter symptoms.":"እባክዎ ምልክቶችን ያስገቡ።",
    "Please enter a query.":"እባክዎ ጥያቄ ያስገቡ።",
}

def t(text, lang):
    return AMHARIC.get(text, text) if lang.lower() == "amharic" else text


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE TRANSLATE
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
# FILE CHECKS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
DATA_FILES = [
    "data/Diseases_and_Symptoms_dataset.csv","data/description.csv","data/diets.csv",
    "data/medications.csv","data/precautions.csv","data/workout.csv",
]
MODEL_FILES = ["models/svc_model.pkl","models/decision_tree_model.pkl","models/label_encoder.pkl"]

def check_files():
    return [f for f in DATA_FILES + MODEL_FILES if not os.path.exists(f)]

def clean_disease_name(name):
    return str(name).lower().replace("_"," ").strip()


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    main_df        = pd.read_csv("data/Diseases_and_Symptoms_dataset.csv")
    description_df = pd.read_csv("data/description.csv")
    diets_df       = pd.read_csv("data/diets.csv")
    medications_df = pd.read_csv("data/medications.csv")
    precautions_df = pd.read_csv("data/precautions.csv")
    workout_df     = pd.read_csv("data/workout.csv")

    description_map = {clean_disease_name(r["Disease"]): r["Description"] for _,r in description_df.iterrows()}
    diets_map       = {clean_disease_name(r["Disease"]): r["Diet"]        for _,r in diets_df.iterrows()}
    medications_map = {clean_disease_name(r["Disease"]): r["Medication"]  for _,r in medications_df.iterrows()}

    precautions_map = {}
    for _,r in precautions_df.iterrows():
        d = clean_disease_name(r["Disease"])
        precautions_map[d] = [r.get(f"Precaution_{i}") for i in range(1,5) if pd.notna(r.get(f"Precaution_{i}"))]

    workout_col = workout_df.columns[1]
    workout_map = {clean_disease_name(r["Disease"]): r[workout_col] for _,r in workout_df.iterrows()}

    return main_df, description_map, diets_map, medications_map, precautions_map, workout_map


@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    svc = joblib.load("models/svc_model.pkl")
    dt  = joblib.load("models/decision_tree_model.pkl")
    le  = joblib.load("models/label_encoder.pkl")
    return svc, dt, le


@st.cache_resource(show_spinner="Building search index…")
def build_tfidf_index(symptom_list, disease_names):
    sym_texts  = [s.replace("_"," ") for s in symptom_list]
    dis_texts  = [d.replace("_"," ") for d in disease_names]
    vec        = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4)).fit(sym_texts+dis_texts)
    sym_matrix = vec.transform(sym_texts).toarray()
    dis_matrix = vec.transform(dis_texts).toarray()
    return vec, sym_matrix, dis_matrix


# ══════════════════════════════════════════════════════════════════════════════
# NLP SYMPTOM EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
NEGATION_TRIGGERS = [
    "no ","not ","without ","don't have","doesn't have","do not have",
    "i have no","haven't","havent","denies","denied","no history of",
    "no sign of","no complaint of","absence of",
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

def _is_negated(sentence):
    sl = sentence.lower()
    return any(trigger in sl for trigger in NEGATION_TRIGGERS)

def _fuzzy_match(phrase, symptom_list, tfidf_vec, sym_matrix, threshold):
    phrase_norm = phrase.replace("_"," ").strip()
    if not phrase_norm or len(phrase_norm) < 3:
        return None
    phrase_key = phrase_norm.replace(" ","_").lower()
    if phrase_key in symptom_list:
        return phrase_key
    vec_query = tfidf_vec.transform([phrase_norm]).toarray()
    sims      = cosine_similarity(vec_query, sym_matrix)[0]
    best_idx  = int(np.argmax(sims))
    if sims[best_idx] >= threshold:
        return symptom_list[best_idx]
    return None

def extract_symptoms_from_text(user_input, symptom_list, tfidf_vec, sym_matrix, threshold=0.35):
    sentences = re.split(r"[.!?;]+", user_input)
    matched, negated = set(), set()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        is_neg  = _is_negated(sentence)
        clauses = re.split(r"[,\n]|\band\b|\bor\b|\bbut\b|\balso\b|\bwith\b|\bplus\b", sentence, flags=re.IGNORECASE)
        for clause in clauses:
            cleaned = FILLER_RE.sub(" ", clause)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            sym = _fuzzy_match(cleaned.replace(" ","_"), symptom_list, tfidf_vec, sym_matrix, threshold)
            if sym:
                (negated if is_neg else matched).add(sym)
    return matched - negated, negated


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR AUTH PANEL
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar_auth(role: str, lang: str) -> str:
    """Renders Sign Up / Log In panel. Returns authenticated user_id or ''."""

    # Already logged in?
    logged = st.session_state.get("logged_in_as")
    if logged and logged.get("role") == role and logged.get("ok"):
        uid = logged["user_id"]
        st.sidebar.markdown(
            f"<div class='logged-in-badge'>✔ Logged in as {role}<br>"
            f"<span style='font-family:monospace;font-size:0.75rem'>{uid}</span></div>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🚪 Log Out", key="logout_btn"):
            st.session_state["logged_in_as"] = None
            st.session_state["auth_message"]  = ""
            st.rerun()
        return uid

    # Mode toggle
    mode = st.sidebar.radio("", options=["Log In", "Sign Up"], horizontal=True, key="auth_mode_radio")

    user_id_input = st.sidebar.text_input("User ID", placeholder=f"Enter your {role} ID", key="auth_uid_input")

    name_input = ""
    if mode == "Sign Up":
        name_input = st.sidebar.text_input("Full Name (optional)", key="auth_name_input")

    if st.sidebar.button("🔑 Log In" if mode == "Log In" else "📝 Sign Up", key="auth_submit_btn", use_container_width=True):
        if not user_id_input.strip():
            st.session_state["auth_message"] = "⚠️ Please enter your User ID."
            st.session_state["auth_ok"]      = False
        else:
            if mode == "Log In":
                ok, msg = _supabase_login(role, user_id_input)
            else:
                ok, msg = _supabase_signup(role, user_id_input, name_input)

            st.session_state["auth_message"] = msg
            st.session_state["auth_ok"]      = ok

            if ok:
                st.session_state["logged_in_as"] = {
                    "role":    role,
                    "user_id": user_id_input.strip().upper(),
                    "ok":      True,
                }
                st.rerun()

    msg = st.session_state.get("auth_message", "")
    if msg:
        css = "auth-success" if st.session_state.get("auth_ok") else "auth-error"
        st.sidebar.markdown(f"<div class='{css}'>{msg}</div>", unsafe_allow_html=True)

    st.sidebar.markdown(
        "<div style='color:#8b949e;font-size:0.72rem;margin-top:6px;font-style:italic'>"
        "Your ID is stored securely. Use it to log in next time.</div>",
        unsafe_allow_html=True,
    )

    logged = st.session_state.get("logged_in_as")
    if logged and logged.get("role") == role and logged.get("ok"):
        return logged["user_id"]
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# ACCESS CONTROL
# ══════════════════════════════════════════════════════════════════════════════
def check_access(age, role, user_id, lang):
    if age < 18:
        return False, t("Access Denied: Information is not available for users under 18.", lang)
    if role in ("Student", "Doctor"):
        logged = st.session_state.get("logged_in_as")
        if (logged and logged.get("role") == role and logged.get("ok")
                and logged.get("user_id","").upper() == (user_id or "").strip().upper()):
            return True, ""
        return False, f"🔒 Please log in with your {role} ID in the sidebar to access this content."
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# ROLE-BASED RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
def role_based_recs(role, lang, key, description_map, diets_map, medications_map, precautions_map, workout_map):
    desc    = translate_content(description_map.get(key,"N/A"), lang)
    diet    = translate_content(diets_map.get(key,"N/A"), lang)
    meds    = translate_content(medications_map.get(key,"N/A"), lang)
    precs   = translate_content(precautions_map.get(key,[]), lang)
    workout = translate_content(workout_map.get(key,"N/A"), lang)

    if role == "Doctor":
        cards = [
            {"label":t("Description",lang),     "content":desc,    "card_type":"full"},
            {"label":t("Dietary Plan",lang),     "content":diet,    "card_type":"full"},
            {"label":t("Medications",lang),      "content":meds,    "card_type":"full"},
            {"label":t("Precautions",lang),      "content":precs,   "card_type":"full"},
            {"label":t("Workout/Activity",lang), "content":workout, "card_type":"full"},
        ]
        advice = ""

    elif role == "Student":
        if isinstance(meds, str) and meds not in ("N/A",""):
            drug_classes = list(dict.fromkeys([i.strip().split()[0].rstrip(".,;") for i in meds.split(",") if i.strip()]))
            meds_display = ", ".join(drug_classes)
            meds_note    = "Drug class names shown only. Full details restricted to licensed Doctors."
        else:
            meds_display, meds_note = "N/A", "Full medication details restricted to Doctors only."
        cards = [
            {"label":t("Description",lang),     "content":desc,         "card_type":"full"},
            {"label":t("Dietary Plan",lang),     "content":diet,         "card_type":"full"},
            {"label":t("Medications",lang),      "content":meds_display, "note":meds_note, "card_type":"limited"},
            {"label":t("Precautions",lang),      "content":precs,        "card_type":"full"},
            {"label":t("Workout/Activity",lang), "content":workout,      "card_type":"full"},
        ]
        advice = "ℹ️ Full medication details are only available to licensed Doctors."

    else:
        cards = [
            {"label":t("Description",lang),     "content":desc,    "card_type":"full"},
            {"label":t("Dietary Plan",lang),     "content":"🔒 Dietary plan details are restricted. Please consult a licensed nutritionist or doctor.", "card_type":"locked"},
            {"label":t("Medications",lang),      "content":"🔒 Medication details are restricted to licensed Doctors only.", "card_type":"locked"},
            {"label":t("Precautions",lang),      "content":precs,   "card_type":"full"},
            {"label":t("Workout/Activity",lang), "content":workout, "card_type":"full"},
        ]
        advice = "⚠️ This information is for general awareness only. Always consult a qualified doctor."

    return cards, advice


def render_rec_cards(cards):
    for card in cards:
        label, content = card["label"], card["content"]
        card_type, note = card.get("card_type","full"), card.get("note","")
        body = ("<ul style='margin:0;padding-left:18px'>" + "".join(f"<li>{i}</li>" for i in content) + "</ul>"
                if isinstance(content, list) else f"<p style='margin:0'>{content}</p>")
        if note:
            body += f"<p style='margin-top:8px;color:#f0883e;font-size:0.8rem;font-style:italic'>⚠️ {note}</p>"
        css = {"full":"result-card","limited":"result-card-limited","locked":"result-card-locked"}.get(card_type,"result-card")
        st.markdown(f"<div class='{css}'><h4>{label}</h4>{body}</div>", unsafe_allow_html=True)


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
    matched_symptoms, negated_symptoms = extract_symptoms_from_text(user_input, symptom_list, tfidf_vec, sym_matrix, threshold)

    if not matched_symptoms:
        hint = (f" (Negated: {', '.join(s.replace('_',' ') for s in negated_symptoms)})" if negated_symptoms else "")
        return None, ("⚠️ No recognised symptoms found. Try describing naturally, e.g.:\n"
                      "• *I have a headache and sore throat*\n"
                      "• *I've been feeling tired with fever and chills*" + hint)

    feature_vector = pd.DataFrame(
        [[1 if s in matched_symptoms else 0 for s in symptom_list]], columns=symptom_list
    )

    preds = []
    for model, name in [(svc_model,"SVC"),(dt_model,"Decision Tree")]:
        try:
            proba    = model.predict_proba(feature_vector)[0]
            top4_idx = np.argsort(proba)[::-1][:4]
            for rank, idx in enumerate(top4_idx):
                preds.append({"model":name,"disease":le.inverse_transform([idx])[0].title(),
                               "confidence":f"{proba[idx]*100:.1f}%","top":rank==0})
        except AttributeError:
            pred_idx = model.predict(feature_vector)[0]
            preds.append({"model":name,"disease":le.inverse_transform([pred_idx])[0].title(),
                          "confidence":"N/A","top":True})

    top_disease = next(p["disease"] for p in preds if p["top"] and p["model"]=="SVC")
    top_key     = clean_disease_name(top_disease)
    cards, advice = role_based_recs(role, lang, top_key, description_map, diets_map, medications_map, precautions_map, workout_map)

    return {
        "matched_symptoms":     [s.replace("_"," ").title() for s in matched_symptoms],
        "negated_symptoms":     [s.replace("_"," ").title() for s in negated_symptoms],
        "predicted_conditions": [p for p in preds if p["model"]=="SVC"],
        "top_disease":          top_disease,
        "rec_cards":            cards,
        "advice":               advice,
    }, ""


# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
def chatbot_response(query, age, role, user_id, lang,
                     description_map, diets_map, medications_map, precautions_map, workout_map,
                     tfidf_vec, symptom_list=None, sym_matrix=None, threshold=0.3):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return msg

    q = query.lower().strip()
    INTENT = {
        "diet":       ["diet","food","eat","nutrition","meal","drink","አመጋገብ"],
        "medication": ["medicine","medication","drug","pill","treat","tablet","prescription","dose","መድሃኒት"],
        "precaution": ["precaution","avoid","prevent","careful","warning","ጥንቃቄ"],
        "workout":    ["workout","exercise","activity","fitness","sport","physical","ስፖርት"],
    }
    intent = "description"
    for key, kws in INTENT.items():
        if any(kw in q for kw in kws):
            intent = key; break

    disease_keys = list(description_map.keys())
    rich_texts   = [f"{dk} {description_map.get(dk,'')}" for dk in disease_keys]
    if not rich_texts:
        return "Sorry, the knowledge base is empty."

    chat_vec    = TfidfVectorizer(analyzer="word", ngram_range=(1,2), stop_words="english", min_df=1).fit(rich_texts)
    rich_matrix = chat_vec.transform(rich_texts).toarray()
    q_cleaned   = re.sub(r"\s+", " ", FILLER_RE.sub(" ", q)).strip()
    sims        = cosine_similarity(chat_vec.transform([q_cleaned]).toarray(), rich_matrix)[0]
    best_idx    = int(np.argmax(sims))

    if float(sims[best_idx]) < threshold:
        if symptom_list and sym_matrix is not None:
            confirmed, _ = extract_symptoms_from_text(query, symptom_list, tfidf_vec, sym_matrix, threshold=0.35)
            if confirmed:
                sym_names = ", ".join(s.replace("_"," ") for s in list(confirmed)[:5])
                return (f"I found possible symptoms: *{sym_names}*.\n\n"
                        "Try the **Disease Predictor** tab for a full diagnosis.")
        return ("I couldn't find specific information. Try asking:\n"
                "• *What is diabetes?*\n• *What diet for asthma?*\n• *Medications for hypertension?*")

    disease_key   = disease_keys[best_idx]
    disease_title = disease_key.title()

    if intent == "diet":
        raw   = diets_map.get(disease_key,"N/A")
        label = t("Dietary Plan", lang)
        info  = "🔒 Restricted. Consult a nutritionist." if role=="Normal User" else raw
        restriction = ""
    elif intent == "medication":
        raw   = medications_map.get(disease_key,"N/A")
        label = t("Medications", lang)
        if role == "Normal User":
            info, restriction = "🔒 Restricted. Consult a doctor.", ""
        elif role == "Student":
            info = ", ".join(dict.fromkeys([i.strip().split()[0].rstrip(".,;") for i in raw.split(",") if i.strip()])) if isinstance(raw,str) and raw not in ("N/A","") else "N/A"
            restriction = "\n\n⚠️ *Drug class names only. Full details for Doctors.*"
        else:
            info, restriction = raw, ""
    elif intent == "precaution":
        raw   = precautions_map.get(disease_key,[])
        label = t("Precautions", lang)
        info  = "\n"+"\n".join(f"• {p}" for p in raw) if isinstance(raw,list) and raw else str(raw)
        restriction = ""
    elif intent == "workout":
        info, label, restriction = workout_map.get(disease_key,"N/A"), t("Workout/Activity",lang), ""
    else:
        info, label, restriction = description_map.get(disease_key,"N/A"), t("Description",lang), ""

    info = translate_content(info, lang)
    info_str = "\n"+"\n".join(f"• {i}" for i in info) if isinstance(info,list) else str(info)
    return f"**{disease_title}** — {label}:\n\n{info_str}{restriction}"


# ══════════════════════════════════════════════════════════════════════════════
# QUICK-SELECT SYMPTOM WIDGET
# ══════════════════════════════════════════════════════════════════════════════
def render_quick_select_symptoms(lang):
    CATEGORIES_EN = {
        "🌡️ General":["fever","chills","sweating","fatigue","weakness","feeling ill","weight gain","ache all over","flu-like syndrome","restlessness","sleepiness","decreased appetite","fluid retention"],
        "🧠 Neuro/Mental":["anxiety and nervousness","depression","insomnia","dizziness","abnormal involuntary movements","disturbance of memory","paresthesia","loss of sensation","focal weakness","seizures","delusions or hallucinations","fainting","temper problems","fears and phobias","low self-esteem","excessive anger","drug abuse","abusing alcohol"],
        "🫀 Cardio/Resp":["shortness of breath","sharp chest pain","chest tightness","palpitations","irregular heartbeat","breathing fast","cough","wheezing","difficulty breathing","congestion in chest","hoarse voice","hemoptysis","coughing up sputum","increased heart rate","decreased heart rate","burning chest pain"],
        "👃 ENT":["sore throat","difficulty speaking","nasal congestion","throat swelling","diminished hearing","difficulty in swallowing","ear pain","ringing in ear","itchy ear(s)","swollen or red tonsils","sneezing","coryza","sinus congestion","painful sinuses","nosebleed"],
        "🤢 Gastro":["sharp abdominal pain","upper abdominal pain","burning abdominal pain","lower abdominal pain","nausea","vomiting","vomiting blood","diarrhea","constipation","stomach bloating","heartburn","regurgitation","blood in stool","melena","rectal bleeding"],
        "🤕 Pain":["headache","frontal headache","back pain","low back pain","neck pain","shoulder pain","hip pain","knee pain","leg pain","foot or toe pain","ankle pain","elbow pain","arm pain","wrist pain","hand or finger pain","joint pain","rib pain","groin pain","facial pain","bones are painful","cramps and spasms"],
        "🦴 Musculo":["arm stiffness or tightness","arm swelling","arm weakness","wrist swelling","hand or finger swelling","knee swelling","leg swelling","ankle swelling","elbow swelling","shoulder stiffness or tightness","back stiffness or tightness","neck swelling","peripheral edema","problems with movement"],
        "🩺 Skin":["abnormal appearing skin","skin lesion","acne or pimples","skin rash","itching of skin","skin dryness, peeling, scaliness, or roughness","skin irritation","itchy scalp","jaundice","eyelid lesion or rash","irregular appearing nails"],
        "👁️ Eye":["diminished vision","double vision","pain in eye","eye redness","lacrimation","itchiness of eye","blindness","eye burns or stings","spots or clouds in vision","swollen eye","eyelid swelling","white discharge from eye"],
        "🚻 Urinary":["painful urination","frequent urination","involuntary urination","blood in urine","retention of urine","unusual color or odor to urine","excessive urination at night","low urine output","vaginal itching","vaginal discharge","vaginal pain","pain during intercourse","swelling of scrotum","pain in testicles"],
        "🤰 Women":["hot flashes","intermenstrual bleeding","pain during pregnancy","problems during pregnancy","uterine contractions","pelvic pain","long menstrual periods","heavy menstrual flow","painful menstruation","infertility"],
        "👶 Pediatric":["lack of growth","irritable infant","infant feeding problem","pulling at ears","diaper rash"],
        "🔬 Other":["jaundice","back mass or lump","neck mass","jaw swelling","lip swelling","toothache","mouth ulcer","gum pain","mouth dryness","allergic reaction","lower body pain"],
    }
    is_am   = lang.lower() == "amharic"
    js_cats = {cat: [{"en":s,"display":s.title()} for s in syms] for cat,syms in CATEGORIES_EN.items()}
    current_list = [s.strip().lower() for s in st.session_state.get("symptoms_text","").split(",") if s.strip()]
    quick_label  = "ምልክቶችን ፈጥኖ ይምረጡ:" if is_am else "Quick-select symptoms:"
    or_text      = "ወይም ምልክቶችን ይተይቡ" if is_am else "or type symptoms below — comma list OR full sentence"

    html_code = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:'Poppins',sans-serif;padding:4px 2px 0}}
.qs-label{{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:rgba(255,255,255,0.38);font-weight:600;margin-bottom:10px}}
.cat-tabs{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}}
.cat-tab{{padding:5px 12px;border-radius:100px;border:1px solid rgba(255,255,255,0.12);background:rgba(255,255,255,0.04);color:rgba(255,255,255,0.5);font-size:0.72rem;font-weight:600;cursor:pointer;white-space:nowrap;transition:all 0.18s ease;font-family:inherit}}
.cat-tab:hover,.cat-tab.active{{border-color:#14b8a6;color:#14b8a6;background:rgba(13,148,136,0.2)}}
.pills-wrap{{display:flex;flex-wrap:wrap;gap:7px;max-height:118px;overflow-y:auto;padding:2px 2px 6px;scrollbar-width:thin;scrollbar-color:rgba(13,148,136,0.45) transparent}}
.pill{{padding:5px 13px;border-radius:100px;border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.05);color:rgba(255,255,255,0.75);font-size:0.77rem;cursor:pointer;white-space:nowrap;transition:all 0.16s ease;user-select:none;font-family:inherit}}
.pill:hover{{border-color:#14b8a6;color:#fff;background:rgba(13,148,136,0.12);transform:translateY(-1px)}}
.pill.sel{{background:rgba(13,148,136,0.25)!important;border-color:#14b8a6!important;color:#14b8a6!important;font-weight:600!important}}
.or-div{{display:flex;align-items:center;gap:10px;margin:14px 0 2px;color:rgba(255,255,255,0.22);font-size:0.67rem;text-transform:uppercase;letter-spacing:0.1em}}
.or-div::before,.or-div::after{{content:'';flex:1;height:1px;background:rgba(255,255,255,0.08)}}
</style></head><body>
<div class="qs-label">{quick_label}</div>
<div class="cat-tabs" id="catTabs"></div>
<div class="pills-wrap" id="pillsWrap"></div>
<div class="or-div">{or_text}</div>
<script>
var CATS={json.dumps(js_cats,ensure_ascii=False)};
var KEYS=Object.keys(CATS);
var active=KEYS[0];
var selected={json.dumps(current_list)};
function renderTabs(){{var el=document.getElementById('catTabs');el.innerHTML='';KEYS.forEach(function(k){{var b=document.createElement('button');b.className='cat-tab'+(k===active?' active':'');b.textContent=k;b.onclick=function(){{active=k;renderTabs();renderPills()}};el.appendChild(b)}})}}
function renderPills(){{var el=document.getElementById('pillsWrap');el.innerHTML='';(CATS[active]||[]).forEach(function(item){{var isSel=selected.indexOf(item.en.toLowerCase())!==-1;var p=document.createElement('button');p.className='pill'+(isSel?' sel':'');p.textContent=(isSel?'✓ ':'')+item.display;p.onclick=(function(sym){{return function(){{toggleSym(sym)}}}})(item.en);el.appendChild(p)}})}}
function syncToStreamlit(){{var val=selected.join(', ');try{{var doc=window.parent.document;var areas=doc.querySelectorAll('textarea');for(var i=0;i<areas.length;i++){{var ta=areas[i];if(ta.placeholder&&ta.placeholder.indexOf('headache')!==-1){{var setter=Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype,'value').set;setter.call(ta,val);ta.dispatchEvent(new window.parent.Event('input',{{bubbles:true}}));break}}}}}}catch(e){{}}}}
function toggleSym(sym){{var lo=sym.toLowerCase();var idx=selected.indexOf(lo);if(idx===-1){{selected.push(lo)}}else{{selected.splice(idx,1)}};renderPills();syncToStreamlit()}}
renderTabs();renderPills();
</script></body></html>"""
    components.html(html_code, height=220, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
def clear_symptoms_callback():
    st.session_state["symptoms_text"]     = ""
    st.session_state["prediction_result"] = None
    st.session_state["prediction_error"]  = None

def clear_diagnosis_callback():
    st.session_state["prediction_result"] = None
    st.session_state["prediction_error"]  = None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
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

    # ── Init session state ────────────────────────────────────────────────────
    for key in ("prediction_result","prediction_error","chat_response","auth_message","auth_ok","logged_in_as"):
        if key not in st.session_state:
            st.session_state[key] = None if key not in ("auth_message",) else ""

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown('<div class="section-header">User Profile</div>', unsafe_allow_html=True)
    lang = st.sidebar.selectbox("🌐 Language", ["English","Amharic"])
    role = st.sidebar.selectbox("👤 Role",     ["Normal User","Student","Doctor"])
    age  = st.sidebar.number_input("🎂 Age", min_value=0, max_value=120, value=25)

    user_id = ""
    if role in ("Student","Doctor"):
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f'<div class="section-header">{"Student" if role=="Student" else "Doctor"} Access</div>',
            unsafe_allow_html=True,
        )
        user_id = render_sidebar_auth(role, lang)

    role_colors = {"Doctor":"#1f6feb","Student":"#2ea043","Normal User":"#6e7681"}
    rc = role_colors.get(role,"#6e7681")
    st.sidebar.markdown(
        f"<div style='margin-top:8px'><span class='role-badge' style='"
        f"background:rgba({int(rc[1:3],16)},{int(rc[3:5],16)},{int(rc[5:7],16)},0.2);"
        f"color:{rc};border:1px solid {rc}40'>● {role}</span></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("⚕️ General health information only. Always consult a qualified doctor.")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='main-header'>
      <div class='main-header-title'>🏥 Integrated Healthcare Dashboard</div>
      <div class='main-header-subtitle'>Disease Prediction · Health Recommendations · AI Chatbot</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        t("Disease Predictor",lang), t("Health Recommender",lang), t("Healthcare Chatbot",lang)
    ])

    # ════════════════════════════ TAB 1 — PREDICTOR ═══════════════════════════
    with tab1:
        st.markdown('<div class="section-header">🩺 Symptom-Based Disease Predictor</div>', unsafe_allow_html=True)
        st.markdown(
            "<div class='nlp-info-banner'><span>✦ Smart symptom extraction</span> — type naturally: "
            "<em>\"I've been having terrible headaches and my throat is sore.\"</em> "
            "Negations like <em>\"no vomiting\"</em> are automatically excluded.</div>",
            unsafe_allow_html=True,
        )

        render_quick_select_symptoms(lang)

        user_input = st.text_area(
            "Symptoms", key="symptoms_text",
            placeholder="e.g., headache, fever, chills  — or: I have a headache and sore throat",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"✦ {t('Predict & Recommend',lang)}", use_container_width=True):
                if not user_input.strip():
                    st.warning(t("Please enter symptoms.",lang))
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
            st.button(t("Clear Symptoms",lang), key="clear_sym", on_click=clear_symptoms_callback, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state["prediction_error"]:
            st.markdown(f'<div class="access-denied">{st.session_state["prediction_error"]}</div>', unsafe_allow_html=True)

        elif st.session_state["prediction_result"]:
            res = st.session_state["prediction_result"]
            st.markdown("---")
            st.markdown('<div class="section-header">✅ Matched Symptoms</div>', unsafe_allow_html=True)
            chips = " · ".join(
                f"<span style='background:rgba(88,166,255,0.15);padding:4px 12px;border-radius:6px;"
                f"font-size:0.85rem;color:#58a6ff;border:1px solid rgba(88,166,255,0.3)'>{s}</span>"
                for s in res["matched_symptoms"]
            )
            st.markdown(chips, unsafe_allow_html=True)

            if res.get("negated_symptoms"):
                neg_chips = " · ".join(
                    f"<span style='background:rgba(248,81,73,0.08);padding:4px 12px;border-radius:6px;"
                    f"font-size:0.82rem;color:#f85149;border:1px solid rgba(248,81,73,0.25);"
                    f"text-decoration:line-through'>{s}</span>"
                    for s in res["negated_symptoms"]
                )
                st.markdown(f"<div style='margin-top:8px;font-size:0.78rem;color:#8b949e'>Excluded (negated): {neg_chips}</div>", unsafe_allow_html=True)

            st.markdown('<br><div class="section-header">🎯 Predicted Conditions</div>', unsafe_allow_html=True)
            rows = ("<div style='background:rgba(22,27,34,0.7);border:1px solid rgba(88,166,255,0.2);"
                    "border-radius:10px;padding:14px 16px;margin-bottom:8px'>"
                    "<div style='color:#79c0ff;font-size:0.75rem;font-weight:700;letter-spacing:0.1em;"
                    "text-transform:uppercase;margin-bottom:10px'>🤖 SVC — Top Predictions</div>")
            for rank, p in enumerate(res["predicted_conditions"]):
                bg  = "linear-gradient(135deg,#1f6feb,#388bfd)" if rank==0 else "rgba(33,38,45,0.8)"
                col = "#fff" if rank==0 else "#8b949e"
                bar = p["confidence"].replace("%","") if p["confidence"]!="N/A" else "0"
                rows += (f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
                         f"<span style='min-width:18px;color:#8b949e;font-size:0.75rem'>#{rank+1}</span>"
                         f"<span style='flex:1;background:{bg};color:{col};padding:5px 12px;border-radius:16px;font-size:0.85rem;font-weight:600'>{p['disease']}</span>"
                         f"<span style='min-width:48px;text-align:right;color:#58a6ff;font-size:0.82rem;font-weight:700'>{p['confidence']}</span></div>"
                         f"<div style='height:3px;background:rgba(48,54,61,0.5);border-radius:2px;margin-bottom:6px'>"
                         f"<div style='height:3px;width:{bar}%;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:2px'></div></div>")
            st.markdown(rows+"</div>", unsafe_allow_html=True)

            with st.expander(f"📊 Health Plan — {res['top_disease']}", expanded=True):
                render_rec_cards(res["rec_cards"])
                if res["advice"]:
                    st.markdown(f'<div class="advice-banner">{res["advice"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="clear-btn-container" style="margin-top:16px;max-width:200px">', unsafe_allow_html=True)
            st.button(t("Clear Diagnosis",lang), key="clear_diag", on_click=clear_diagnosis_callback, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════ TAB 2 — RECOMMENDER ═════════════════════════
    with tab2:
        st.markdown('<div class="section-header">🏥 Select a Disease</div>', unsafe_allow_html=True)
        selected_disease = st.selectbox(
            "Disease", options=sorted(desc_map.keys()),
            format_func=lambda x: x.title(), label_visibility="collapsed",
        )
        col_r1, _ = st.columns([1,5])
        with col_r1:
            if st.button(t("Get Plan",lang), use_container_width=True):
                ok, err_msg = check_access(age, role, user_id, lang)
                if not ok:
                    st.markdown(f'<div class="access-denied">{err_msg}</div>', unsafe_allow_html=True)
                else:
                    key    = clean_disease_name(selected_disease)
                    cards, advice = role_based_recs(role, lang, key, desc_map, diets_map, meds_map, precs_map, workout_map)
                    with st.expander(f"💊 Health Plan — {selected_disease.title()}", expanded=True):
                        render_rec_cards(cards)
                        if advice:
                            st.markdown(f'<div class="advice-banner">{advice}</div>', unsafe_allow_html=True)

    # ════════════════════════════ TAB 3 — CHATBOT ═════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">💬 Healthcare Chatbot</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="chat-bot">🤖 {t("Hello! How can I help you with health information today?",lang)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin:10px 0 4px;font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;font-weight:600'>Example queries:</div>"
            "<div style='font-size:0.82rem;color:#8b949e;margin-bottom:12px'>"
            "• <em>What is allergy?</em> &nbsp;|&nbsp; • <em>What diet for asthma?</em> &nbsp;|&nbsp; "
            "• <em>Medications for hypertension?</em> &nbsp;|&nbsp; • <em>Precautions for migraine?</em></div>",
            unsafe_allow_html=True,
        )

        chat_query = st.text_input(
            "Your question:", placeholder="e.g. What diet should I follow for Asthma?",
            key="chat_query", label_visibility="collapsed",
        )

        col_c1, col_c2 = st.columns([1,6])
        with col_c1:
            ask_btn = st.button(t("Ask Bot",lang), use_container_width=True)
        with col_c2:
            st.markdown('<div class="clear-btn-container">', unsafe_allow_html=True)
            if st.button("🗑️ Clear", key="chat_clear", use_container_width=True):
                st.session_state["chat_response"] = None
            st.markdown('</div>', unsafe_allow_html=True)

        if ask_btn:
            if not chat_query.strip():
                st.warning(t("Please enter a query.",lang))
            else:
                with st.spinner("🤔 Thinking…"):
                    reply = chatbot_response(
                        chat_query, age, role, user_id, lang,
                        desc_map, diets_map, meds_map, precs_map, workout_map,
                        vec, symptom_list=list(symptom_list), sym_matrix=sym_matrix,
                    )
                st.session_state["chat_response"] = reply

        if st.session_state.get("chat_response"):
            st.markdown(
                f'<div class="chat-bot">🤖 {st.session_state["chat_response"]}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
