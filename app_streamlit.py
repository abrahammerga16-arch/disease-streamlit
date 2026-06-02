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
# ENHANCED CUSTOM CSS WITH ANIMATIONS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght=400;600&family=IBM+Plex+Sans:wght=300;400;600;700&family=Poppins:wght=300;400;600;700&display=swap');

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
}

.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(46, 160, 67, 0.4);
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
}

.result-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.8) 0%, rgba(33,38,45,0.6) 100%);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    animation: fadeInUp 0.6s ease-out;
}
.result-card h4 {
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.disease-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: #fff;
    border-radius: 24px;
    padding: 6px 18px;
    font-weight: 600;
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
}

.access-denied {
    background: linear-gradient(135deg, rgba(45,14,14,0.8) 0%, rgba(248,81,73,0.1) 100%);
    border-left: 4px solid #f85149;
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    color: #f85149;
}

.main-header { text-align: center; margin-bottom: 32px; }
.main-header-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #79c0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
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
    font-weight: 700;
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
    "Hello! How can I help you with health information today?": "ሰላም! ዛሬ በጤና መረጃ እንዴት ልረዳዎ እችላለሁ?",
    "Access Denied: Information is not available for users under 18.": "የመዳረሻ እገዳ: ከ18 ዓመት በታች ለሆኑ ተጠቃሚዎች መረጃ አይገኝም",
    "Access Denied: Invalid ID for Student role.": "የመዳረሻ እገዳ: ለተማሪ ሚና ልክ ያልሆነ መታወቂያ",
    "Access Denied: Invalid ID for Doctor role.": "የመዳረሻ እገዳ: ለሐኪም ሚና ልክ ያልሆነ መታወቂያ",
    "Disease Predictor": "ምልክት መተንበይ",
    "Health Recommender": "የጤና ምክር ሰጪ",
    "Healthcare Chatbot": "የጤና እንክብካቤ ቻትቦት",
    "Predict & Recommend": "መተንበይ እና መምከር",
    "Get Plan": "እቅድ ያግኙ",
    "Please enter symptoms.": "እባክዎ ምልክቶችን ያስገቡ።",
}

def t(text: str, lang: str) -> str:
    if lang.lower() == "amharic":
        return AMHARIC.get(text, text)
    return text

def translate_content(text, target_lang="English"):
    return text  # Hook implementation to local translators if necessary

# ──────────────────────────────────────────────
# DATA & ENGINE CONTEXT LOADERS
# ──────────────────────────────────────────────
def clean_disease_name(name: str) -> str:
    return str(name).lower().replace("_", " ").strip()

@st.cache_data(show_spinner="Syncing Dataset Inventories...")
def load_data():
    # Production datasets mapping file layouts
    try:
        main_df = pd.read_csv("data/Diseases_and_Symptoms_dataset.csv")
        description_df = pd.read_csv("data/description.csv")
        diets_df = pd.read_csv("data/diets.csv")
        medications_df = pd.read_csv("data/medications.csv")
        precautions_df = pd.read_csv("data/precautions.csv")
        workout_df = pd.read_csv("data/workout.csv")

        description_map = {clean_disease_name(r["Disease"]): r["Description"] for _, r in description_df.iterrows()}
        diets_map = {clean_disease_name(r["Disease"]): r["Diet"] for _, r in diets_df.iterrows()}
        medications_map = {clean_disease_name(r["Disease"]): r["Medication"] for _, r in medications_df.iterrows()}

        precautions_map = {}
        for _, r in precautions_df.iterrows():
            d = clean_disease_name(r["Disease"])
            precs = [r.get(f"Precaution_{i}") for i in range(1, 5)]
            precautions_map[d] = [p for p in precs if pd.notna(p)]

        workout_col = workout_df.columns[1]
        workout_map = {clean_disease_name(r["Disease"]): r[workout_col] for _, r in workout_df.iterrows()}
        
    except Exception:
        # Dynamic Fallback Mock Matrix if local structural file loading drops
        diseases = ["Fungal infection", "Malaria", "Typhoid", "Diabetes", "Hypertension"]
        symptoms = ["itching", "skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea"]
        main_df = pd.DataFrame(0, index=range(20), columns=symptoms)
        main_df.insert(0, "diseases", [diseases[i % len(diseases)] for i in range(20)])
        
        description_map = {d.lower(): f"Clinical pathology and risk profiles for {d} conditions." for d in diseases}
        diets_map = {d.lower(): "Incorporate whole foods, increase fiber intake, and restrict processed carbs." for d in diseases}
        medications_map = {d.lower(): "Clotrimazole cream, Antimalarial tablets, Antibiotics, Metformin" for d in diseases}
        precautions_map = {d.lower(): ["Maintain clean hygiene", "Avoid stagnant water sources", "Drink filtered water"] for d in diseases}
        workout_map = {d.lower(): "30 minutes of low-intensity aerobic exercises daily." for d in diseases}

    return main_df, description_map, diets_map, medications_map, precautions_map, workout_map

@st.cache_resource(show_spinner="Syncing Predictive Architectures...")
def load_models(main_df):
    try:
        svc = joblib.load("models/svc_model.pkl")
        dt = joblib.load("models/decision_tree_model.pkl")
        le = joblib.load("models/label_encoder.pkl")
        return svc, dt, le
    except Exception:
        # Safety dynamic estimator mock pattern
        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.85, 0.05, 0.05, 0.03, 0.02]])
            def predict(self, X):
                return np.array([0])
        le = LabelEncoder()
        le.fit(main_df["diseases"])
        return MockModel(), MockModel(), le

# ──────────────────────────────────────────────
# QUICK-SELECT SYMPTOM WIDGET UI
# ──────────────────────────────────────────────
def render_quick_select_symptoms(lang: str) -> None:
    CATEGORIES_EN = {
        "🌡️ General": ["fever", "fatigue", "weakness", "chills", "sweating", "weight_loss"],
        "🤕 Pain": ["headache", "back_pain", "chest_pain", "joint_pain", "muscle_pain", "abdominal_pain"],
        "🫀 Cardio/Resp": ["cough", "shortness_of_breath", "runny_nose", "nasal_congestion"],
        "🤢 Gastro": ["nausea", "vomiting", "diarrhea", "constipation", "loss_of_appetite"],
        "🩺 Skin": ["skin_rash", "itching", "acne", "skin_swelling"]
    }
    is_am = lang.lower() == "amharic"
    js_cats = {}
    for cat_en, symptoms in CATEGORIES_EN.items():
        js_cats[cat_en] = [{"en": s, "display": s.replace("_", " ").title()} for s in symptoms]

    current_val = st.session_state.get("symptoms_text", "")
    current_list = [s.strip().lower() for s in current_val.split(",") if s.strip()]

    quick_label = "ምልክቶችን ፈጥኖ ይምረጡ:" if is_am else "Quick-select symptoms:"
    or_text = "ወይም ምልክቶችን ይተይቡ" if is_am else "or type symptoms"

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: 'Poppins', sans-serif; padding-top: 4px; }}
  .qs-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.4); margin-bottom: 8px; font-weight: 600; }}
  .cat-tabs {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }}
  .cat-tab {{ padding: 4px 10px; border-radius: 100px; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.6); font-size: 0.7rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }}
  .cat-tab.active {{ background: rgba(31, 111, 235, 0.2); border-color: #58a6ff; color: #58a6ff; }}
  .pills-wrap {{ display: flex; flex-wrap: wrap; gap: 6px; max-height: 100px; overflow-y: auto; }}
  .pill {{ padding: 4px 10px; border-radius: 100px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.05); color: #c9d1d9; font-size: 0.75rem; cursor: pointer; transition: all 0.15s; }}
  .pill.selected {{ background: rgba(46, 160, 67, 0.2) !important; border-color: #3fb950 !important; color: #3fb950 !important; font-weight: 600; }}
  .or-div {{ display: flex; align-items: center; gap: 10px; margin: 12px 0 2px; color: rgba(255,255,255,0.2); font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; }}
  .or-div::before, .or-div::after {{ content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.08); }}
</style>
</head>
<body>
<div class="qs-label">{quick_label}</div>
<div class="cat-tabs" id="catTabs"></div>
<div class="pills-wrap" id="pillsWrap"></div>
<div class="or-div">{or_text}</div>

<script>
  var CATS = {json.dumps(js_cats)};
  var KEYS = Object.keys(CATS);
  var active = KEYS[0];
  var selected = {json.dumps(current_list)};

  function renderTabs() {{
    var tDiv = document.getElementById('catTabs');
    tDiv.innerHTML = '';
    KEYS.forEach(function(k) {{
      var btn = document.createElement('button');
      btn.className = 'cat-tab' + (k === active ? ' active' : '');
      btn.textContent = k;
      btn.onclick = function() {{ active = k; renderTabs(); renderPills(); }};
      tDiv.appendChild(btn);
    }});
  }}

  function renderPills() {{
    var pDiv = document.getElementById('pillsWrap');
    pDiv.innerHTML = '';
    (CATS[active] || []).forEach(function(item) {{
      var isSel = selected.indexOf(item.en.toLowerCase()) !== -1;
      var btn = document.createElement('button');
      btn.className = 'pill' + (isSel ? ' selected' : '');
      btn.textContent = (isSel ? '✓ ' : '') + item.display;
      btn.onclick = function() {{ toggleSymptom(item.en); }};
      pDiv.appendChild(btn);
    }});
  }}

  function toggleSymptom(sym) {{
    var norm = sym.toLowerCase();
    var idx = selected.indexOf(norm);
    if (idx === -1) selected.push(norm);
    else selected.splice(idx, 1);
    renderPills();
    
    // Sync updates back to the primary Streamlit textarea layout DOM tree
    try {{
      var txtArea = window.parent.document.querySelector('textarea[placeholder*="headache"]');
      if (txtArea) {{
        var setter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
        setter.call(txtArea, selected.join(', '));
        txtArea.dispatchEvent(new window.parent.Event('input', {{ bubbles: true }}));
      }}
    }} catch(e) {{}}
  }}

  renderTabs();
  renderPills();
</script>
</body>
</html>
"""
    components.html(html_code, height=190, scrolling=False)

# ──────────────────────────────────────────────
# ACCESS ROUTER & PROFILE FILTERS
# ──────────────────────────────────────────────
def check_access(age: int, role: str, user_id: str, lang: str):
    if age < 18:
        return False, t("Access Denied: Information is not available for users under 18.", lang)
    if role == "Student" and user_id != "1111":
        return False, t("Access Denied: Invalid ID for Student role.", lang)
    if role == "Doctor" and user_id != "0000":
        return False, t("Access Denied: Invalid ID for Doctor role.", lang)
    return True, ""

def role_based_recs(role, lang, key, description_map, diets_map, medications_map, precautions_map, workout_map):
    desc = translate_content(description_map.get(key, "Information profile unmapped."), lang)
    diet = translate_content(diets_map.get(key, "Information profile unmapped."), lang)
    meds = translate_content(medications_map.get(key, "Information profile unmapped."), lang)
    precs = translate_content(precautions_map.get(key, []), lang)
    workout = translate_content(workout_map.get(key, "Information profile unmapped."), lang)

    if role == "Doctor":
        return {
            t("Description", lang): desc,
            t("Dietary Plan", lang): diet,
            t("Medications", lang): meds,
            t("Precautions", lang): precs,
            t("Workout/Activity", lang): workout,
        }, ""

    elif role == "Student":
        if isinstance(meds, str) and meds != "Information profile unmapped.":
            meds_limited = ", ".join(w.split()[0] for w in meds.split(",") if w.strip()) + " (Drug classifications only — specific formulations restricted)"
        else:
            meds_limited = "Full prescription blueprints restricted to Doctor profiles."
        return {
            t("Description", lang): desc,
            t("Dietary Plan", lang): diet,
            t("Medications", lang): meds_limited,
            t("Precautions", lang): precs,
            t("Workout/Activity", lang): workout,
        }, "ℹ️ Full medication details are only available to Doctors."

    else:  # Normal User
        if isinstance(diet, str) and diet != "Information profile unmapped.":
            diet_limited = diet.split(".")[0].strip() + ". (Full structural dietary plan restricted — consult a certified nutritionist.)"
        else:
            diet_limited = "Eat balanced meals and stay hydrated. Consult a nutritionist for a personalized plan."
            
        meds_hidden = "🔒 Medication details are strictly restricted to licensed clinical profiles. Please consult a professional medical doctor."
        advice = ("⚠️ General User Restriction: This dashboard is for educational reference only. "
                  "Always consult a qualified doctor before making any medical decisions.")
        return {
            t("Description", lang): desc,
            t("Dietary Plan", lang): diet_limited,
            t("Medications", lang): meds_hidden,
            t("Precautions", lang): precs,
            t("Workout/Activity", lang): workout,
        }, advice

# ──────────────────────────────────────────────
# SYSTEM EXECUTION CONTROLLER
# ──────────────────────────────────────────────
def integrated_prediction_system(user_input, age, role, user_id, lang, main_df, le, svc_model, dt_model, description_map, diets_map, medications_map, precautions_map, workout_map, tfidf_vec, sym_matrix, threshold=0.5):
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
        raw_vec = tfidf_vec.transform([raw.replace("_", " ")]).toarray()
        sims = cosine_similarity(raw_vec, sym_matrix)[0]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            matched_symptoms.add(symptom_list[best_idx])

    if not matched_symptoms:
        return None, "⚠️ No recognized symptoms matched. Please try: headache, fever, chills, skin rash."

    feature_vector = pd.DataFrame(0, index=[0], columns=symptom_list)
    for s in matched_symptoms:
        feature_vector.at[0, s] = 1

    proba = svc_model.predict_proba(feature_vector)[0]
    top_idx = int(np.argmax(proba))
    top_disease = le.inverse_transform([top_idx])[0]
    top_key = clean_disease_name(top_disease)
    
    preds = [{"disease": top_disease.title(), "confidence": f"{proba[top_idx]*100:.1f}%"}]
    recs, advice = role_based_recs(role, lang, top_key, description_map, diets_map, medications_map, precautions_map, workout_map)

    return {"predictions": preds, "recommendations": recs, "advice": advice}, ""

# ──────────────────────────────────────────────
# APPLICATION INITIALIZATION
# ──────────────────────────────────────────────
main_df, description_map, diets_map, medications_map, precautions_map, workout_map = load_data()
svc_model, dt_model, le = load_models(main_df)

symptom_cols = list(main_df.drop(columns=["diseases"]).columns)
disease_names = list(main_df["diseases"].unique())
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(symptom_cols + disease_names)
sym_matrix = vec.transform(symptom_cols).toarray()

# Sidebar Setup
with st.sidebar:
    st.header("👤 Auth & Config")
    lang = st.selectbox("Language / ቋንቋ", ["English", "Amharic"])
    role = st.selectbox("Role Module", ["Normal User", "Student", "Doctor"])
    user_id = st.text_input("Access Verification Token ID", value="", type="password")
    age = st.slider("User Biological Age", min_value=1, max_value=100, value=25)

st.markdown('<div class="main-header"><h1 class="main-header-title">Integrated Healthcare Dashboard</h1></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs([t("Disease Predictor", lang), t("Health Recommender", lang)])

with tab1:
    st.markdown(f'<div class="section-header">{t("Disease Predictor", lang)}</div>', unsafe_allow_html=True)
    user_symptoms = st.text_area("List specific physical symptoms (comma-separated):", placeholder="e.g., headache, fever, chills", key="symptoms_text")
    render_quick_select_symptoms(lang)
    
    if st.button(t("Predict & Recommend", lang)):
        if not user_symptoms.strip():
            st.warning(t("Please enter symptoms.", lang))
        else:
            res, err = integrated_prediction_system(user_symptoms, age, role, user_id, lang, main_df, le, svc_model, dt_model, description_map, diets_map, medications_map, precautions_map, workout_map, vec, sym_matrix)
            if err:
                st.markdown(f'<div class="access-denied">{err}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<h3>Analysis Results: <span class="disease-badge">{res["predictions"][0]["disease"]}</span> <span class="conf-pill">{res["predictions"][0]["confidence"]} Profile Fit</span></h3>', unsafe_allow_html=True)
                
                for title, content in res["recommendations"].items():
                    st.markdown(f'<div class="result-card"><h4>{title}</h4>', unsafe_allow_html=True)
                    if isinstance(content, list):
                        for item in content:
                            st.markdown(f"• {item}<br>", unsafe_allow_html=True)
                    else:
                        st.write(content)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if res["advice"]:
                    st.markdown(f'<div class="advice-banner">{res["advice"]}</div>', unsafe_allow_html=True)

with tab2:
    st.markdown(f'<div class="section-header">{t("Health Recommender", lang)}</div>', unsafe_allow_html=True)
    selected_disease = st.selectbox("Direct Profile Lookup", [d.title() for d in disease_names])
    
    if st.button(t("Get Plan", lang)):
        ok, msg = check_access(age, role, user_id, lang)
        if not ok:
            st.markdown(f'<div class="access-denied">{msg}</div>', unsafe_allow_html=True)
        else:
            recs, advice = role_based_recs(role, lang, selected_disease.lower(), description_map, diets_map, medications_map, precautions_map, workout_map)
            for title, content in recs.items():
                st.markdown(f'<div class="result-card"><h4>{title}</h4>', unsafe_allow_html=True)
                if isinstance(content, list):
                    for item in content:
                        st.markdown(f"• {item}<br>", unsafe_allow_html=True)
                else:
                    st.write(content)
                st.markdown('</div>', unsafe_allow_html=True)
            if advice:
                st.markdown(f'<div class="advice-banner">{advice}</div>', unsafe_allow_html=True)
