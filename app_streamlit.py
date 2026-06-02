"""
Integrated Healthcare Dashboard — Streamlit version (Enhanced UI + 4 fixes)
Changes vs previous version:
  1. Quick-select symptom pills with check-toggle on click
  2. Dietary Plan hidden for Normal User role
  3. All UI text switches to Amharic when language = Amharic
  4. Language / Role / Age controls moved to main page (sidebar removed)
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

st.set_page_config(
    page_title="Integrated Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# AMHARIC TRANSLATIONS  (full UI strings)
# ──────────────────────────────────────────────
AMHARIC = {
    "Disease": "በሽታ",
    "Description": "መግለጫ",
    "Dietary Plan": "የአመጋገብ እቅድ",
    "Medications": "መድሃኒቶች",
    "Workout/Activity": "ስፖርት/እንቅስቃሴ",
    "Precautions": "ጥንቃቄዎች",
    "Disease Predictor": "ምልክት መተንበይ",
    "Health Recommender": "የጤና ምክር ሰጪ",
    "Healthcare Chatbot": "የጤና ቻትቦት",
    "Predict & Recommend": "ተንበይ እና ምከር",
    "Get Plan": "እቅድ ያግኙ",
    "Ask Bot": "ቦት ይጠይቁ",
    "Clear All": "ሁሉን አጽዳ",
    "Clear": "አጽዳ",
    "Browse symptoms by category": "ምልክቶችን በምድብ ፈልግ",
    "Or type symptoms manually:": "ወይም ምልክቶችን በእጅ ይጻፉ:",
    "Your question:": "ጥያቄዎ:",
    "Selected:": "የተመረጡ:",
    "Quick-select symptoms:": "ምልክቶችን በፍጥነት ምረጥ:",
    "All": "ሁሉም",
    "General": "ጠቅላላ",
    "Pain": "ሕመም",
    "Cardio / Resp": "ልብ / ትንፋሽ",
    "Neuro / Mental": "ነርቭ / አዕምሮ",
    "Gastro": "ሆድ",
    "Urinary": "ሽንት",
    "Skin": "ቆዳ",
    "Reproductive": "ስርዓተ-ዘር",
    "Other": "ሌሎች",
    "Access Denied: Information is not available for users under 18.":
        "የመዳረሻ እገዳ: ከ18 ዓመት በታች ለሆኑ ተጠቃሚዎች መረጃ አይገኝም",
    "Access Denied: Invalid ID for Student role.":
        "የመዳረሻ እገዳ: ለተማሪ ሚና ልክ ያልሆነ መታወቂያ",
    "Access Denied: Invalid ID for Doctor role.":
        "የመዳረሻ እገዳ: ለሐኪም ሚና ልክ ያልሆነ መታወቂያ",
    "Please enter symptoms.": "እባክዎ ምልክቶችን ያስገቡ።",
    "Please enter a query.": "እባክዎ ጥያቄ ያስገቡ።",
    "Hello! How can I help you with health information today?":
        "ሰላም! ዛሬ በጤና መረጃ እንዴት ልረዳዎ እችላለሁ?",
    "🔒 Medication details are restricted. Please consult a licensed doctor.":
        "🔒 የመድሃኒት መረጃ የተገደበ ነው። እባክዎ ፈቃድ ያለው ሐኪም ያማክሩ።",
    "ℹ️ Full medication details are only available to Doctors.":
        "ℹ️ ሙሉ የመድሃኒት መረጃ ለሐኪሞች ብቻ ይገኛል።",
    "medical_advice_disclaimer":
        "ማንኛውንም መድሃኒት ከመውሰድዎ በፊት ወይም ከባድ ምልክቶች ካጋጠሙዎት ሁልጊዜ ሐኪም ያማክሩ።",
    "Settings": "ቅንብሮች",
    "🌐 Language": "🌐 ቋንቋ",
    "👤 Your Age": "👤 ዕድሜዎ",
    "👨‍⚕️ Your Role": "👨‍⚕️ ሚናዎ",
    "🔐 ID Number": "🔐 መታወቂያ ቁጥር",
    "Normal User": "መደበኛ ተጠቃሚ",
    "Doctor": "ሐኪም",
    "Student": "ተማሪ",
    "⚙️ Similarity Threshold": "⚙️ የቅርበት ደረጃ",
    "English": "እንግሊዝኛ",
    "Amharic": "አማርኛ",
    "Analysing symptoms…": "ምልክቶችን በመተንተን ላይ…",
    "Fetching health plan…": "የጤና እቅድ በማምጣት ላይ…",
    "Thinking…": "በማሰብ ላይ…",
    "✅ Matched Symptoms": "✅ የተዛመዱ ምልክቶች",
    "🎯 Predicted Conditions": "🎯 የተተነበዩ ሁኔታዎች",
    "Health Plan": "የጤና እቅድ",
    "🏥 Select a Disease": "🏥 በሽታ ምረጥ",
    "💬 Healthcare Chatbot": "💬 የጤና ቻትቦት",
    "SVC — Top 4 Predictions": "SVC — ዋና 4 ትንበያዎች",
    "⚠️ This information is for general awareness only. Always consult a qualified doctor before taking any medication.":
        "⚠️ ይህ መረጃ ለአጠቃላይ ግንዛቤ ብቻ ነው። ማንኛውንም መድሃኒት ከመውሰድዎ ሁልጊዜ ሐኪም ያማክሩ።",
}


def t(text: str, lang: str) -> str:
    if lang.lower() == "amharic":
        return AMHARIC.get(text, text)
    return text


# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Poppins:wght@300;400;600;700&display=swap');

@keyframes fadeInUp    { from{opacity:0;transform:translateY(20px)}  to{opacity:1;transform:translateY(0)} }
@keyframes slideInLeft { from{opacity:0;transform:translateX(-30px)} to{opacity:1;transform:translateX(0)} }
@keyframes pillPop     { 0%{transform:scale(.9)} 60%{transform:scale(1.08)} 100%{transform:scale(1)} }

html,body,[class*="css"]{ font-family:'Poppins',sans-serif; scroll-behavior:smooth; }
.stApp{
  background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1f2d 100%);
  color:#e6edf3; animation:fadeInUp .8s ease-out;
}

/* hide sidebar */
section[data-testid="stSidebar"]{ display:none !important; }
[data-testid="collapsedControl"]{ display:none !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  background:rgba(22,27,34,.6); backdrop-filter:blur(10px);
  border-radius:12px; padding:6px; gap:4px;
  border:1px solid rgba(48,54,61,.3);
}
.stTabs [data-baseweb="tab"]{
  background:transparent; color:#8b949e; border-radius:8px;
  font-weight:600; letter-spacing:.04em; padding:10px 24px;
  border:none !important; transition:all .3s cubic-bezier(.4,0,.2,1); font-size:.95rem;
}
.stTabs [data-baseweb="tab"]:hover{ background:rgba(88,166,255,.1); color:#58a6ff; }
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#21262d,#1f6feb) !important;
  color:#fff !important; box-shadow:0 4px 12px rgba(88,166,255,.25);
}

/* Inputs */
.stTextInput input,.stSelectbox select,.stTextArea textarea,
div[data-baseweb="select"]>div{
  background:rgba(33,38,45,.8) !important;
  border:1px solid rgba(48,54,61,.5) !important;
  color:#e6edf3 !important; border-radius:8px !important;
  transition:all .3s ease !important; font-size:.95rem !important;
}
.stTextInput input:focus,.stTextArea textarea:focus,
div[data-baseweb="select"]>div:focus-within{
  border-color:#58a6ff !important;
  box-shadow:0 0 0 3px rgba(88,166,255,.2) !important;
}

/* Symptoms textarea — compact size + teal-tinted background */
[data-testid="stTextArea"] textarea {
  background: linear-gradient(135deg, rgba(13,30,35,.95) 0%, rgba(10,40,45,.9) 100%) !important;
  border: 1.5px solid rgba(20,184,166,.35) !important;
  border-radius: 10px !important;
  color: #e6edf3 !important;
  font-size: .88rem !important;
  min-height: 52px !important;
  max-height: 52px !important;
  resize: none !important;
  padding: 8px 12px !important;
  line-height: 1.5 !important;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: #14b8a6 !important;
  box-shadow: 0 0 0 3px rgba(20,184,166,.2) !important;
  background: linear-gradient(135deg, rgba(13,35,40,1) 0%, rgba(10,48,52,1) 100%) !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: rgba(20,184,166,.45) !important; }
.stNumberInput input{
  background:rgba(33,38,45,.8) !important; color:#e6edf3 !important;
  border:1px solid rgba(48,54,61,.5) !important; border-radius:8px !important;
}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,#238636,#2ea043);
  color:#fff; border:none; border-radius:8px;
  font-weight:600; padding:10px 24px;
  transition:all .3s cubic-bezier(.4,0,.2,1); font-size:.95rem;
}
.stButton>button:hover{
  transform:translateY(-2px);
  box-shadow:0 8px 24px rgba(46,160,67,.4);
  background:linear-gradient(135deg,#2ea043,#3fb950);
}
.stButton>button:active{ transform:translateY(0); }

/* ── Category filter buttons */
.qs-cat-row div[data-testid="stButton"] button {
  background: rgba(22,27,34,.9) !important;
  border: 1px solid rgba(88,166,255,.2) !important;
  color: #8b949e !important;
  border-radius: 20px !important;
  font-size: .78rem !important;
  font-weight: 600 !important;
  padding: 5px 14px !important;
  height: auto !important; min-height: 30px !important;
  transition: all .2s ease !important;
  box-shadow: none !important; transform: none !important;
}
.qs-cat-row div[data-testid="stButton"] button:hover {
  background: rgba(88,166,255,.12) !important;
  border-color: #58a6ff !important;
  color: #58a6ff !important;
  transform: none !important; box-shadow: none !important;
}
.qs-cat-row .cat-active div[data-testid="stButton"] button {
  background: rgba(88,166,255,.18) !important;
  border-color: #58a6ff !important; color: #58a6ff !important;
  box-shadow: 0 0 0 1px rgba(88,166,255,.3) !important;
}

/* ── Symptom pill buttons */
.qs-pill-row div[data-testid="stButton"] button {
  background: rgba(22,27,34,.85) !important;
  border: 1px solid rgba(88,166,255,.18) !important;
  color: #8b949e !important;
  border-radius: 18px !important;
  font-size: .78rem !important; font-weight: 500 !important;
  padding: 4px 10px !important;
  height: auto !important; min-height: 28px !important;
  transition: all .2s ease !important;
  box-shadow: none !important; transform: none !important;
}
.qs-pill-row div[data-testid="stButton"] button:hover {
  background: rgba(88,166,255,.1) !important;
  border-color: #58a6ff !important; color: #79c0ff !important;
  transform: none !important; box-shadow: none !important;
}
.qs-pill-row .pill-active div[data-testid="stButton"] button {
  background: linear-gradient(135deg,rgba(31,111,235,.75),rgba(56,139,253,.65)) !important;
  border-color: #388bfd !important; color: #fff !important;
  box-shadow: 0 2px 8px rgba(31,111,235,.35) !important;
  font-weight: 600 !important;
}

/* Result cards */
.result-card{
  background:linear-gradient(135deg,rgba(22,27,34,.8),rgba(33,38,45,.6));
  border:1px solid rgba(88,166,255,.2); border-radius:12px;
  padding:20px 24px; margin-bottom:16px;
  animation:fadeInUp .6s ease-out; transition:all .3s cubic-bezier(.4,0,.2,1);
}
.result-card:hover{
  border-color:rgba(88,166,255,.4);
  box-shadow:0 8px 32px rgba(88,166,255,.15); transform:translateY(-4px);
}
.result-card h4{
  color:#58a6ff; font-family:'IBM Plex Mono',monospace;
  font-size:.82rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:12px;
}
.result-card p,.result-card li{ color:#c9d1d9; font-size:.95rem; line-height:1.6; }

.advice-banner{
  background:linear-gradient(135deg,rgba(45,24,0,.8),rgba(240,136,62,.1));
  border-left:4px solid #f0883e; padding:12px 18px;
  border-radius:0 8px 8px 0; color:#f0883e;
  font-size:.9rem; margin-top:16px; animation:slideInLeft .5s ease-out;
}
.chat-bot{
  background:linear-gradient(135deg,rgba(33,38,45,.9),rgba(31,111,235,.1));
  border-left:4px solid #58a6ff; padding:16px 20px;
  border-radius:0 12px 12px 0; color:#c9d1d9;
  font-size:.95rem; margin-top:12px; animation:slideInLeft .5s ease-out; line-height:1.6;
}
.section-header{
  color:#58a6ff; font-family:'IBM Plex Mono',monospace;
  font-size:.75rem; letter-spacing:.15em; text-transform:uppercase;
  border-bottom:1px solid rgba(88,166,255,.2);
  padding-bottom:8px; margin-bottom:18px;
  animation:slideInLeft .4s ease-out; font-weight:700;
}
.access-denied{
  background:linear-gradient(135deg,rgba(45,14,14,.8),rgba(248,81,73,.1));
  border-left:4px solid #f85149; padding:12px 18px;
  border-radius:0 8px 8px 0; color:#f85149; animation:slideInLeft .5s ease-out;
}
.main-header{ text-align:center; animation:fadeInUp .8s ease-out; margin-bottom:32px; }
.main-header-title{
  font-family:'Poppins',sans-serif; font-size:2.8rem; font-weight:700;
  background:linear-gradient(135deg,#58a6ff,#79c0ff);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.main-header-subtitle{ color:#8b949e; font-size:1rem; letter-spacing:.05em; margin-top:8px; }
hr{ border-color:rgba(88,166,255,.1) !important; }
label,.stSelectbox label,.stNumberInput label,.stSlider label{ color:#8b949e !important; }
.stSlider .st-bk{ background:linear-gradient(90deg,#1f6feb,#58a6ff) !important; }
.stSlider .st-ao{ background:#21262d !important; border:2px solid #58a6ff !important; }
</style>
""", unsafe_allow_html=True)


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
# SYMPTOM CATEGORIZATION
# ──────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "General":        ["fever", "fatigue", "tiredness", "weakness", "chills", "sweat", "malaise", "ache", "weight"],
    "Pain":           ["pain", "cramp", "stiff", "sore", "tender"],
    "Cardio / Resp":  ["chest", "heart", "pulse", "breath", "cough", "wheeze", "asthma", "sneeze", "cold", "flu"],
    "Neuro / Mental": ["headache", "migraine", "dizziness", "vertigo", "confusion", "memory", "seizure", "tremor", "anxiety"],
    "Gastro":         ["vomit", "nausea", "diarrhea", "constipation", "stomach", "appetite", "indigestion", "bloat"],
    "Urinary":        ["urine", "urinary", "bladder", "kidney", "frequency"],
    "Skin":           ["rash", "itch", "burn", "wound", "acne", "dermatitis", "eczema", "blister", "swell"],
    "Reproductive":   ["menstrual", "vaginal", "penile", "discharge", "pregnancy", "fertility"],
    "Other":          [],
}


def categorize_symptoms(symptom_list):
    cats = {k: [] for k in CATEGORY_KEYWORDS}
    for symptom in symptom_list:
        sl = symptom.lower().replace("_", " ")
        placed = False
        for cat, words in CATEGORY_KEYWORDS.items():
            if cat == "Other":
                continue
            if any(w in sl for w in words):
                cats[cat].append(symptom)
                placed = True
                break
        if not placed:
            cats["Other"].append(symptom)
    return {k: v for k, v in cats.items() if v}


# ──────────────────────────────────────────────
# ACCESS CONTROL
# ──────────────────────────────────────────────
def check_access(age: int, role: str, user_id: str, lang: str):
    if age < 18:
        return False, t("Access Denied: Information is not available for users under 18.", lang)
    if role == "Student" and user_id != "1111":
        return False, t("Access Denied: Invalid ID for Student role.", lang)
    if role == "Doctor"  and user_id != "0000":
        return False, t("Access Denied: Invalid ID for Doctor role.", lang)
    return True, ""


# ──────────────────────────────────────────────
# ROLE-BASED OUTPUT  — Change 2: Dietary Plan hidden for Normal User
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
        }, t("ℹ️ Full medication details are only available to Doctors.", lang)

    else:  # Normal User — Dietary Plan intentionally omitted (Change 2)
        meds_hidden = t("🔒 Medication details are restricted. Please consult a licensed doctor.", lang)
        advice = t(
            "⚠️ This information is for general awareness only. "
            "Always consult a qualified doctor before taking any medication.",
            lang
        )
        return {
            t("Description", lang):      desc,
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
    tfidf_vec, sym_matrix, threshold=0.6,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return None, msg

    symptom_list = list(main_df.drop(columns=["diseases"]).columns)
    raw_symptoms = [s.strip().lower().replace(" ", "_") for s in user_input.split(",") if s.strip()]
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
        fallback = (
            "I couldn't find specific information for that query. "
            "Try asking about a specific disease (e.g. 'What is Asthma?') "
            "or use the Disease Predictor tab first."
        )
        return translate_content(fallback, lang) if lang == "Amharic" else fallback

    disease_key   = disease_names[best_idx]
    disease_title = disease_key.title()
    q_lower       = query.lower()

    if any(w in q_lower for w in ["diet", "food", "eat", "nutrition", "አመጋገብ"]):
        info = diets_map.get(disease_key, "N/A")
        label = t("Dietary Plan", lang)
    elif any(w in q_lower for w in ["medicine", "medication", "drug", "treat", "መድሃኒት"]):
        info = medications_map.get(disease_key, "N/A")
        label = t("Medications", lang)
    elif any(w in q_lower for w in ["precaution", "avoid", "prevent", "ጥንቃቄ"]):
        info = precautions_map.get(disease_key, [])
        label = t("Precautions", lang)
    elif any(w in q_lower for w in ["workout", "exercise", "activity", "ስፖርት"]):
        info = workout_map.get(disease_key, "N/A")
        label = t("Workout/Activity", lang)
    else:
        info = description_map.get(disease_key, "N/A")
        label = t("Description", lang)

    info     = translate_content(info, lang)
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


def render_predictions(predictions: list, lang: str):
    label = t("SVC — Top 4 Predictions", lang)
    rows_html = (
        "<div style='background:rgba(22,27,34,.7);border:1px solid rgba(88,166,255,.2);"
        "border-radius:10px;padding:14px 16px;margin-bottom:8px'>"
        f"<div style='color:#79c0ff;font-size:.75rem;font-weight:700;letter-spacing:.1em;"
        f"text-transform:uppercase;margin-bottom:10px'>🤖 {label}</div>"
    )
    for rank, p in enumerate(predictions):
        badge_bg  = "linear-gradient(135deg,#1f6feb,#388bfd)" if rank == 0 else "rgba(33,38,45,.8)"
        badge_col = "#fff" if rank == 0 else "#8b949e"
        bar_w     = p["confidence"].replace("%", "") if p["confidence"] != "N/A" else "0"
        rows_html += (
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
            f"<span style='min-width:18px;color:#8b949e;font-size:.75rem;font-weight:600'>#{rank+1}</span>"
            f"<span style='flex:1;background:{badge_bg};color:{badge_col};"
            f"padding:5px 12px;border-radius:16px;font-size:.85rem;font-weight:600'>{p['disease']}</span>"
            f"<span style='min-width:48px;text-align:right;color:#58a6ff;font-size:.82rem;font-weight:700'>{p['confidence']}</span>"
            f"</div>"
            f"<div style='height:3px;background:rgba(48,54,61,.5);border-radius:2px;margin-bottom:6px'>"
            f"<div style='height:3px;width:{bar_w}%;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:2px'></div>"
            f"</div>"
        )
    st.markdown(rows_html + "</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# QUICK-SELECT PILLS  — Change 1
# Symptom pills with ✓ checkmark when selected
# ──────────────────────────────────────────────
def render_quick_select(categorized_symptoms: dict, lang: str):
    """
    Category filter row + toggleable symptom pill grid.
    Selected pills get a ✓ prefix and blue gradient styling.
    State persists in st.session_state.symptoms_selected.
    """
    if "qs_cat" not in st.session_state:
        st.session_state.qs_cat = "All"

    all_cats = ["All"] + list(categorized_symptoms.keys())

    # ── Category filter row
    cat_cols = st.columns(len(all_cats))
    for i, cat in enumerate(all_cats):
        display = t(cat, lang)
        is_active = st.session_state.qs_cat == cat
        btn_style = (
            "background:rgba(88,166,255,.2);border:1px solid rgba(88,166,255,.5);"
            "color:#58a6ff;border-radius:16px;font-size:.78rem;font-weight:700;"
        ) if is_active else (
            "background:rgba(33,38,45,.6);border:1px solid rgba(48,54,61,.5);"
            "color:#8b949e;border-radius:16px;font-size:.78rem;"
        )
        # Inject override style for this specific button
        st.markdown(
            f"<style>div[data-testid='stButton'] button[title='cat_{cat}']"
            f"{{{btn_style}}}</style>",
            unsafe_allow_html=True,
        )
        with cat_cols[i]:
            if st.button(display, key=f"catbtn_{cat}", use_container_width=True,
                         help=f"cat_{cat}"):
                st.session_state.qs_cat = cat
                st.rerun()

    # ── Which symptoms to display
    if st.session_state.qs_cat == "All":
        visible = sorted(set(s for syms in categorized_symptoms.values() for s in syms))
    else:
        visible = sorted(categorized_symptoms.get(st.session_state.qs_cat, []))

    # ── Render pills in a 5-column grid
    NUM_COLS = 5
    cols = st.columns(NUM_COLS)
    for idx, sym in enumerate(visible):
        raw_label = sym.replace("_", " ").title()
        is_sel    = sym in st.session_state.symptoms_selected
        pill_label = ("✓ " if is_sel else "") + raw_label

        # Inject per-pill CSS override for selected state
        if is_sel:
            st.markdown(
                f"""<style>
                div[data-testid="stButton"]:has(button[title="pill_{sym}"]) button {{
                    background: linear-gradient(135deg,#1f6feb,#388bfd) !important;
                    border-color: transparent !important;
                    color: #fff !important;
                    box-shadow: 0 4px 12px rgba(31,111,235,.35) !important;
                }}
                </style>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<style>
                div[data-testid="stButton"]:has(button[title="pill_{sym}"]) button {{
                    background: rgba(33,38,45,.8) !important;
                    border: 1.5px solid rgba(88,166,255,.25) !important;
                    color: #8b949e !important;
                    border-radius: 20px !important;
                    font-size: .82rem !important;
                    font-weight: 600 !important;
                }}
                div[data-testid="stButton"]:has(button[title="pill_{sym}"]) button:hover {{
                    border-color: #58a6ff !important;
                    color: #58a6ff !important;
                    background: rgba(88,166,255,.1) !important;
                }}
                </style>""",
                unsafe_allow_html=True,
            )

        with cols[idx % NUM_COLS]:
            if st.button(pill_label, key=f"pill_{sym}", use_container_width=True,
                         help=f"pill_{sym}"):
                if is_sel:
                    st.session_state.symptoms_selected.remove(sym)
                else:
                    st.session_state.symptoms_selected.append(sym)
                st.rerun()


# ──────────────────────────────────────────────
# LOAD CHECK
# ──────────────────────────────────────────────
missing = check_files()
if missing:
    st.error("### ⚠️ Missing required files")
    st.markdown(
        "Please place the following files before running Streamlit:\n\n"
        + "\n".join(f"- `{f}`" for f in missing)
    )
    st.stop()

(main_df, description_map, diets_map,
 medications_map, precautions_map, workout_map) = load_data()
svc_model, dt_model, le = load_models()

symptom_list  = list(main_df.drop(columns=["diseases"]).columns)
disease_names = list(description_map.keys())

tfidf_vec, sym_matrix, dis_matrix = build_tfidf_index(
    tuple(symptom_list), tuple(disease_names)
)
categorized_symptoms = categorize_symptoms(symptom_list)


# ──────────────────────────────────────────────
# SETTINGS PANEL on main page — Change 4
# ──────────────────────────────────────────────
if "language"  not in st.session_state: st.session_state.language  = "English"
if "age"       not in st.session_state: st.session_state.age       = 25
if "role"      not in st.session_state: st.session_state.role      = "Normal User"
if "user_id"   not in st.session_state: st.session_state.user_id   = ""
if "threshold" not in st.session_state: st.session_state.threshold = 0.6

lang = st.session_state.language

with st.expander(f"⚙️  {t('Settings', lang)}", expanded=False):
    sc1, sc2, sc3, sc4, sc5 = st.columns([1.2, 1, 1.5, 1.5, 1.8])
    with sc1:
        lang_choice = st.selectbox(
            t("🌐 Language", lang),
            ["English", "Amharic"],
            index=["English", "Amharic"].index(st.session_state.language),
            key="lang_select",
        )
        if lang_choice != st.session_state.language:
            st.session_state.language = lang_choice
            st.rerun()
    with sc2:
        st.session_state.age = st.number_input(
            t("👤 Your Age", lang),
            min_value=1, max_value=120,
            value=st.session_state.age, step=1,
            key="age_input",
        )
    with sc3:
        role_options = ["Normal User", "Doctor", "Student"]
        role_sel = st.selectbox(
            t("👨‍⚕️ Your Role", lang),
            options=role_options,
            format_func=lambda x: t(x, lang),
            index=role_options.index(st.session_state.role),
            key="role_select",
        )
        st.session_state.role = role_sel
    with sc4:
        if st.session_state.role in ("Doctor", "Student"):
            placeholder = "Doctor ID: 0000" if st.session_state.role == "Doctor" else "Student ID: 1111"
            st.session_state.user_id = st.text_input(
                t("🔐 ID Number", lang),
                placeholder=placeholder,
                type="password",
                value=st.session_state.user_id,
                key="uid_input",
            )
        else:
            st.session_state.user_id = ""
    with sc5:
        st.session_state.threshold = st.slider(
            t("⚙️ Similarity Threshold", lang),
            0.3, 0.9, st.session_state.threshold, 0.05,
            key="thresh_slider",
        )

language  = st.session_state.language
age       = st.session_state.age
role      = st.session_state.role
user_id   = st.session_state.user_id
threshold = st.session_state.threshold


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
    t("Disease Predictor",  language),
    t("Health Recommender", language),
    t("Healthcare Chatbot", language),
])


# ══════════════════════════════════════════════
# TAB 1 — DISEASE PREDICTOR
# ══════════════════════════════════════════════
with tab1:
    if "symptoms_selected" not in st.session_state:
        st.session_state.symptoms_selected = []

    st.markdown(
        f"<div class='section-header'>{t('Quick-select symptoms:', language)}</div>",
        unsafe_allow_html=True,
    )
    render_quick_select(categorized_symptoms, language)

    # Selected summary pills
    if st.session_state.symptoms_selected:
        pills = " ".join(
            f"<span style='background:rgba(31,111,235,.25);color:#79c0ff;"
            f"padding:2px 10px;border-radius:12px;font-size:.8rem;margin:2px;"
            f"display:inline-block'>{s.replace('_',' ').title()}</span>"
            for s in st.session_state.symptoms_selected
        )
        st.markdown(f"**{t('Selected:', language)}** {pills}", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        if "symptoms_text" not in st.session_state:
            st.session_state.symptoms_text = ""
        symptoms_input = st.text_area(
            t("Or type symptoms manually:", language),
            value=st.session_state.symptoms_text,
            placeholder="e.g. headache, fever, cough, vomiting",
            height=56,
            key="symptoms_textarea",
        )
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(t("Predict & Recommend", language),
                                key="predict_btn", use_container_width=True)
        clear_btn   = st.button(t("Clear All", language),
                                key="clear_predict", use_container_width=True)

    if clear_btn:
        st.session_state.pop("predict_result", None)
        st.session_state.symptoms_selected = []
        st.session_state.symptoms_text = ""
        st.rerun()

    if predict_btn:
        combined_input = symptoms_input.strip()
        if st.session_state.symptoms_selected:
            selected_str   = ", ".join(st.session_state.symptoms_selected)
            combined_input = (selected_str + ", " + combined_input
                              if combined_input else selected_str)

        if not combined_input:
            st.warning(t("Please enter symptoms.", language))
        else:
            with st.spinner(t("Analysing symptoms…", language)):
                result, err = integrated_prediction_system(
                    combined_input, age, role, user_id, language,
                    main_df, le, svc_model, dt_model,
                    description_map, diets_map, medications_map,
                    precautions_map, workout_map,
                    tfidf_vec, sym_matrix, threshold,
                )
            if err:
                st.markdown(f"<div class='access-denied'>{err}</div>",
                            unsafe_allow_html=True)
            else:
                st.session_state["predict_result"] = result

    if "predict_result" in st.session_state:
        res = st.session_state["predict_result"]
        st.markdown("---")
        st.markdown(
            f"<div class='section-header'>{t('✅ Matched Symptoms', language)}</div>",
            unsafe_allow_html=True,
        )
        if res["matched_symptoms"]:
            st.markdown(
                " · ".join(
                    f"<span style='background:linear-gradient(135deg,rgba(33,38,45,.9),"
                    f"rgba(88,166,255,.2));padding:4px 12px;border-radius:6px;"
                    f"font-size:.85rem;color:#58a6ff;border:1px solid rgba(88,166,255,.3)'>{s}</span>"
                    for s in res["matched_symptoms"]
                ),
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<br><div class='section-header'>{t('🎯 Predicted Conditions', language)}</div>",
            unsafe_allow_html=True,
        )
        render_predictions(res["predicted_conditions"], language)

        with st.expander(
            f"📊 {t('Health Plan', language)} — {res['top_disease']}", expanded=True
        ):
            render_recommendations(res["recommendations"])
            if res["advice"]:
                st.markdown(
                    f"<div class='advice-banner'>{res['advice']}</div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════
# TAB 2 — HEALTH RECOMMENDER
# ══════════════════════════════════════════════
with tab2:
    st.markdown(
        f"<div class='section-header'>{t('🏥 Select a Disease', language)}</div>",
        unsafe_allow_html=True,
    )
    sorted_diseases  = sorted(disease_names)
    selected_disease = st.selectbox(
        "Disease:", options=sorted_diseases,
        format_func=lambda x: x.title(),
        key="rec_disease", label_visibility="collapsed",
    )
    col_r1, col_r2 = st.columns([1, 5])
    with col_r1:
        rec_btn   = st.button(t("Get Plan", language), key="rec_btn",   use_container_width=True)
        rec_clear = st.button(t("Clear", language),    key="rec_clear", use_container_width=True)

    if rec_clear:
        st.session_state.pop("rec_result", None)
        st.session_state.pop("rec_advice", None)

    if rec_btn:
        with st.spinner(t("Fetching health plan…", language)):
            recs, advice = health_recommender(
                selected_disease, age, role, user_id, language,
                description_map, diets_map, medications_map,
                precautions_map, workout_map,
            )
        if recs is None:
            st.markdown(f"<div class='access-denied'>{advice}</div>",
                        unsafe_allow_html=True)
        else:
            st.session_state["rec_result"] = recs
            st.session_state["rec_advice"] = advice

    if "rec_result" in st.session_state:
        with st.expander(
            f"💊 {t('Health Plan', language)} — {selected_disease.title()}", expanded=True
        ):
            render_recommendations(st.session_state["rec_result"])
            if st.session_state.get("rec_advice"):
                st.markdown(
                    f"<div class='advice-banner'>{st.session_state['rec_advice']}</div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════
# TAB 3 — CHATBOT
# ══════════════════════════════════════════════
with tab3:
    st.markdown(
        f"<div class='section-header'>{t('💬 Healthcare Chatbot', language)}</div>",
        unsafe_allow_html=True,
    )
    greeting = t("Hello! How can I help you with health information today?", language)
    st.markdown(f"<div class='chat-bot'>🤖 {greeting}</div>", unsafe_allow_html=True)

    chat_query = st.text_input(
        t("Your question:", language),
        placeholder="e.g. What diet should I follow for Asthma?",
        key="chat_query", label_visibility="collapsed",
    )
    col_c1, col_c2 = st.columns([1, 6])
    with col_c1:
        chat_btn   = st.button(t("Ask Bot", language), key="chat_btn",   use_container_width=True)
        chat_clear = st.button(t("Clear", language),   key="chat_clear", use_container_width=True)

    if chat_clear:
        st.session_state.pop("chat_response", None)

    if chat_btn:
        if not chat_query.strip():
            st.warning(t("Please enter a query.", language))
        else:
            with st.spinner(t("Thinking…", language)):
                reply = chatbot_response(
                    chat_query, age, role, user_id, language,
                    description_map, diets_map, medications_map,
                    precautions_map, workout_map,
                    disease_names, tfidf_vec, dis_matrix, threshold,
                )
            st.session_state["chat_response"] = reply

    if "chat_response" in st.session_state:
        st.markdown(
            f"<div class='chat-bot'>🤖 {st.session_state['chat_response']}</div>",
            unsafe_allow_html=True,
        )
