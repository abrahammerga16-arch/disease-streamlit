"""
Integrated Healthcare Dashboard — Streamlit version
Replicates the Google Colab notebook exactly:
  • Disease Predictor  (symptom input → ML prediction)
  • Health Recommender (disease dropdown → full health plan)
  • Healthcare Chatbot  (natural-language query → semantic answer)
  • Role / Age / ID access control
  • English ↔ Amharic UI
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
# CUSTOM CSS  (clean medical aesthetic)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Main background */
.stApp { background: #0d1117; color: #e6edf3; }

/* ── Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 8px 20px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #58a6ff !important;
}

/* ── Inputs */
.stTextInput input, .stSelectbox select, .stTextArea textarea,
div[data-baseweb="select"] > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
}

/* ── Buttons */
.stButton > button {
    background: #238636;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 20px;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2ea043; }

/* ── Result cards */
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.result-card h4 {
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.result-card p, .result-card li { color: #c9d1d9; font-size: 0.95rem; }

/* ── Disease badge */
.disease-badge {
    display: inline-block;
    background: #1f6feb;
    color: #fff;
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 6px;
}

/* ── Confidence pill */
.conf-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.8rem;
    color: #8b949e;
    margin-left: 8px;
}

/* ── Warning / advice banner */
.advice-banner {
    background: #2d1800;
    border-left: 4px solid #f0883e;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    color: #f0883e;
    font-size: 0.9rem;
    margin-top: 12px;
}

/* ── Chat bubble */
.chat-bot {
    background: #21262d;
    border-left: 3px solid #58a6ff;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    color: #c9d1d9;
    font-size: 0.95rem;
    margin-top: 10px;
}

/* ── Section header */
.section-header {
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin-bottom: 16px;
}

/* ── Number widget */
.stNumberInput input { background: #21262d !important; color: #e6edf3 !important; }

/* ── Slider */
.stSlider .st-bk { background: #58a6ff; }

/* ── Access denied */
.access-denied {
    background: #2d0e0e;
    border-left: 4px solid #f85149;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    color: #f85149;
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
    """Translate text to Amharic if needed; fallback to original."""
    if lang.lower() == "amharic":
        return AMHARIC.get(text, text)
    return text


# ──────────────────────────────────────────────
# GOOGLE TRANSLATE (optional, for Amharic content)
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
    missing = [f for f in DATA_FILES + MODEL_FILES if not os.path.exists(f)]
    return missing


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
    dt = joblib.load("models/decision_tree_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return svc, dt, le


@st.cache_resource(show_spinner="Building symptom index…")
def build_tfidf_index(symptom_list: tuple, disease_names: tuple):
    """TF-IDF based similarity — no torch required."""
    sym_texts = [s.replace("_", " ") for s in symptom_list]
    dis_texts = [d.replace("_", " ") for d in disease_names]
    all_texts = sym_texts + dis_texts

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(all_texts)
    sym_matrix = vec.transform(sym_texts).toarray()
    dis_matrix = vec.transform(dis_texts).toarray()
    return vec, sym_matrix, dis_matrix


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
def role_based_recs(
    role: str, lang: str, key: str,
    description_map, diets_map, medications_map, precautions_map, workout_map
):
    """
    Returns a dict of health info filtered by role:

    Normal User — Description + Precautions + Workout only.
                  Dietary plan is shown as a general tip (first sentence only).
                  Medications are hidden with a redirect to see a doctor.

    Student     — Description + Precautions + Workout + full Diet.
                  Medications shown as drug class names only (first word of each).

    Doctor      — Full unrestricted output: all fields complete.
    """
    desc     = translate_content(description_map.get(key, "N/A"), lang)
    diet     = translate_content(diets_map.get(key, "N/A"), lang)
    meds     = translate_content(medications_map.get(key, "N/A"), lang)
    precs    = translate_content(precautions_map.get(key, []), lang)
    workout  = translate_content(workout_map.get(key, "N/A"), lang)

    if role == "Doctor":
        return {
            t("Description", lang):      desc,
            t("Dietary Plan", lang):     diet,
            t("Medications", lang):      meds,
            t("Precautions", lang):      precs,
            t("Workout/Activity", lang): workout,
        }, ""

    elif role == "Student":
        # Show drug class hint only — first word of each medication
        if isinstance(meds, str) and meds != "N/A":
            meds_limited = ", ".join(
                word.split()[0] for word in meds.split(",") if word.strip()
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
        # Diet: general tip — first sentence only
        if isinstance(diet, str) and diet != "N/A":
            diet_limited = diet.split(".")[0].strip() + ". (Full dietary plan restricted — consult a nutritionist.)"
        else:
            diet_limited = "Eat balanced meals and stay hydrated. Consult a nutritionist for a personalised plan."

        # Medications: completely hidden
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
    user_input: str,
    age: int,
    role: str,
    user_id: str,
    lang: str,
    main_df: pd.DataFrame,
    le: LabelEncoder,
    svc_model,
    dt_model,
    description_map: dict,
    diets_map: dict,
    medications_map: dict,
    precautions_map: dict,
    workout_map: dict,
    tfidf_vec,
    sym_matrix,
    threshold: float = 0.6,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return None, msg

    symptom_list = list(main_df.drop(columns=["diseases"]).columns)

    # Parse user symptoms → match to dataset columns via TF-IDF
    raw_symptoms = [s.strip().lower().replace(" ", "_") for s in user_input.split(",") if s.strip()]
    matched_symptoms = set()

    for raw in raw_symptoms:
        # exact match first
        if raw in symptom_list:
            matched_symptoms.add(raw)
            continue
        # TF-IDF fallback
        raw_vec = tfidf_vec.transform([raw.replace("_", " ")]).toarray()
        sims = cosine_similarity(raw_vec, sym_matrix)[0]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            matched_symptoms.add(symptom_list[best_idx])

    if not matched_symptoms:
        return None, "⚠️ No recognised symptoms found. Try terms like: headache, fever, cough, vomiting."

    # Build feature vector
    feature_vector = pd.DataFrame([[1 if s in matched_symptoms else 0
                                    for s in symptom_list]],
                                  columns=symptom_list)

    # Predict with both models
    preds = []
    for model, name in [(svc_model, "SVC"), (dt_model, "Decision Tree")]:
        pred_idx = model.predict(feature_vector)[0]
        disease = le.inverse_transform([pred_idx])[0]
        try:
            proba = model.predict_proba(feature_vector)[0]
            conf = f"{proba[pred_idx] * 100:.1f}%"
        except AttributeError:
            conf = "N/A"
        preds.append({"model": name, "disease": disease.title(), "confidence": conf})

    # Top disease (SVC)
    top_disease = preds[0]["disease"]
    top_key = clean_disease_name(top_disease)

    recs, advice = role_based_recs(
        role, lang, top_key,
        description_map, diets_map, medications_map, precautions_map, workout_map
    )

    matched_display = [s.replace("_", " ").title() for s in matched_symptoms]
    return {
        "matched_symptoms": matched_display,
        "predicted_conditions": preds,
        "top_disease": top_disease,
        "recommendations": recs,
        "advice": advice,
    }, ""


# ──────────────────────────────────────────────
# HEALTH RECOMMENDER
# ──────────────────────────────────────────────
def health_recommender(
    disease_name: str, age: int, role: str, user_id: str, lang: str,
    description_map, diets_map, medications_map, precautions_map, workout_map,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return None, msg

    key = clean_disease_name(disease_name)
    recs, advice = role_based_recs(
        role, lang, key,
        description_map, diets_map, medications_map, precautions_map, workout_map
    )
    return recs, advice


# ──────────────────────────────────────────────
# CHATBOT
# ──────────────────────────────────────────────
def chatbot_response(
    query: str, age: int, role: str, user_id: str, lang: str,
    description_map, diets_map, medications_map, precautions_map, workout_map,
    disease_names, tfidf_vec, dis_matrix,
    threshold: float = 0.55,
):
    ok, msg = check_access(age, role, user_id, lang)
    if not ok:
        return msg

    q_vec = tfidf_vec.transform([query]).toarray()
    sims = cosine_similarity(q_vec, dis_matrix)[0]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    if best_sim < threshold:
        fallback = ("I couldn't find specific information for that query. "
                    "Try asking about a specific disease (e.g. 'What is Asthma?') "
                    "or use the Disease Predictor tab first.")
        return translate_content(fallback, lang) if lang == "Amharic" else fallback

    disease_key = disease_names[best_idx]
    disease_title = disease_key.title()
    q_lower = query.lower()

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

    info = translate_content(info, lang)
    if isinstance(info, list):
        info_str = "\n• " + "\n• ".join(info)
    else:
        info_str = str(info)

    return f"**{disease_title}** — {label}:\n{info_str}"


# ──────────────────────────────────────────────
# RENDER HELPERS
# ──────────────────────────────────────────────
def render_recommendations(recs: dict):
    for label, value in recs.items():
        if isinstance(value, list):
            items = "".join(f"<li>{i}</li>" for i in value)
            content = f"<ul style='margin:0;padding-left:18px'>{items}</ul>"
        else:
            content = f"<p style='margin:0'>{value}</p>"
        st.markdown(
            f"<div class='result-card'><h4>{label}</h4>{content}</div>",
            unsafe_allow_html=True,
        )


def render_predictions(predictions: list):
    for p in predictions:
        st.markdown(
            f"<span class='disease-badge'>{p['disease']}</span>"
            f"<span class='conf-pill'>{p['model']} · {p['confidence']}</span>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# SIDEBAR — GLOBAL CONTROLS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Healthcare Dashboard")
    st.markdown("---")
    language = st.selectbox("🌐 Language", ["English", "Amharic"])
    st.markdown("---")
    age = st.number_input("Your Age", min_value=1, max_value=120, value=25, step=1)
    role = st.selectbox("Your Role", ["Normal User", "Doctor", "Student"])
    user_id = ""
    if role in ("Doctor", "Student"):
        placeholder = "Doctor ID: 0000" if role == "Doctor" else "Student ID: 1111"
        user_id = st.text_input("ID Number", placeholder=placeholder, type="password")
    st.markdown("---")
    threshold = st.slider("Similarity Threshold", 0.3, 0.9, 0.6, 0.05,
                          help="Minimum confidence for symptom/disease matching")
    st.markdown("---")
    st.caption("⚕️ This system provides general health information only. "
               "Always consult a qualified doctor for medical decisions.")


# ──────────────────────────────────────────────
# LOAD EVERYTHING — or show upload instructions
# ──────────────────────────────────────────────
missing = check_files()

if missing:
    st.error("### ⚠️ Missing required files")
    st.markdown(
        "Please place the following files **in the same directory as `app.py`** "
        "before running Streamlit:\n\n"
        + "\n".join(f"- `{f}`" for f in missing)
    )
    st.markdown("""
**How to fix:**
Your repo must have this folder structure:
```
disease-streamlit/
  app_streamlit.py
  requirements.txt
  models/
    svc_model.pkl
    decision_tree_model.pkl
    label_encoder.pkl
  data/
    Diseases_and_Symptoms_dataset.csv
    description.csv
    diets.csv
    medications.csv
    precautions.csv
    workout.csv
```
""")
    st.stop()

# Load everything
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
<div style='text-align:center;padding:24px 0 8px'>
  <span style='font-family:"IBM Plex Mono",monospace;font-size:2.2rem;
               font-weight:700;color:#58a6ff;letter-spacing:0.04em'>
    🏥 Integrated Healthcare Dashboard
  </span><br>
  <span style='color:#8b949e;font-size:0.9rem'>
    Disease Prediction · Health Recommendations · AI Chatbot
  </span>
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
    st.markdown("<div class='section-header'>Symptom Input</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        symptoms_input = st.text_area(
            t("Symptoms:", language),
            placeholder="e.g. headache, fever, vomiting, fatigue",
            height=100,
            key="symptoms_input",
        )
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(t("Predict & Recommend", language), key="predict_btn",
                                use_container_width=True)
        clear_btn   = st.button("🗑️ Clear", key="clear_predict", use_container_width=True)

    if clear_btn:
        st.session_state.pop("predict_result", None)

    if predict_btn:
        if not symptoms_input.strip():
            st.warning(t("Please enter symptoms.", language))
        else:
            with st.spinner("Analysing symptoms…"):
                result, err = integrated_prediction_system(
                    symptoms_input, age, role, user_id, language,
                    main_df, le, svc_model, dt_model,
                    description_map, diets_map, medications_map,
                    precautions_map, workout_map,
                    tfidf_vec, sym_matrix, threshold,
                )
            if err:
                st.markdown(
                    f"<div class='access-denied'>{err}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.session_state["predict_result"] = result

    if "predict_result" in st.session_state:
        res = st.session_state["predict_result"]
        st.markdown("---")
        st.markdown("<div class='section-header'>Matched Symptoms</div>", unsafe_allow_html=True)
        if res["matched_symptoms"]:
            st.markdown(" · ".join(
                f"<span style='background:#21262d;padding:2px 8px;border-radius:4px;"
                f"font-size:0.85rem;color:#58a6ff'>{s}</span>"
                for s in res["matched_symptoms"]
            ), unsafe_allow_html=True)

        st.markdown("<br><div class='section-header'>Predicted Conditions</div>",
                    unsafe_allow_html=True)
        render_predictions(res["predicted_conditions"])

        st.markdown("<br><div class='section-header'>Health Plan for Top Prediction</div>",
                    unsafe_allow_html=True)
        render_recommendations(res["recommendations"])

        if res["advice"]:
            st.markdown(f"<div class='advice-banner'>{res['advice']}</div>",
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — HEALTH RECOMMENDER
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Select a Disease</div>", unsafe_allow_html=True)

    sorted_diseases = sorted(disease_names)
    selected_disease = st.selectbox(
        t("Disease:", language),
        options=sorted_diseases,
        format_func=lambda x: x.title(),
        key="rec_disease",
    )

    col_r1, col_r2 = st.columns([1, 5])
    with col_r1:
        rec_btn   = st.button(t("Get Plan", language), key="rec_btn", use_container_width=True)
        rec_clear = st.button("🗑️ Clear", key="rec_clear", use_container_width=True)

    if rec_clear:
        st.session_state.pop("rec_result", None)
        st.session_state.pop("rec_advice", None)

    if rec_btn:
        with st.spinner("Fetching health plan…"):
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
        st.markdown("---")
        st.markdown(
            f"<div class='section-header'>Health Plan — {selected_disease.title()}</div>",
            unsafe_allow_html=True,
        )
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
    st.markdown("<div class='section-header'>Healthcare Chatbot</div>",
                unsafe_allow_html=True)
    greeting = t("Hello! How can I help you with health information today?", language)
    st.markdown(f"<div class='chat-bot'>🤖 {greeting}</div>", unsafe_allow_html=True)

    chat_query = st.text_input(
        t("Query:", language),
        placeholder="e.g. What diet should I follow for Asthma?",
        key="chat_query",
    )
    col_c1, col_c2 = st.columns([1, 6])
    with col_c1:
        chat_btn   = st.button(t("Ask Bot", language), key="chat_btn", use_container_width=True)
        chat_clear = st.button("🗑️ Clear", key="chat_clear", use_container_width=True)

    if chat_clear:
        st.session_state.pop("chat_response", None)

    if chat_btn:
        if not chat_query.strip():
            st.warning(t("Please enter a query.", language))
        else:
            with st.spinner("Thinking…"):
                reply = chatbot_response(
                    chat_query, age, role, user_id, language,
                    description_map, diets_map, medications_map,
                    precautions_map, workout_map,
                    disease_names, tfidf_vec, dis_matrix,
                    threshold,
                )
            st.session_state["chat_response"] = reply

    if "chat_response" in st.session_state:
        st.markdown(
            f"<div class='chat-bot'>🤖 {st.session_state['chat_response']}</div>",
            unsafe_allow_html=True,
        )
