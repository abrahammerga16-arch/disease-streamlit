import os
import ast
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from googletrans import Translator

warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Data ────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_data
def load_data():
    main_df        = pd.read_csv(os.path.join(DATA_DIR, "Diseases_and_Symptoms_dataset.csv"))
    description_df = pd.read_csv(os.path.join(DATA_DIR, "description.csv"))
    diets_df       = pd.read_csv(os.path.join(DATA_DIR, "diets.csv"))
    medications_df = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
    precautions_df = pd.read_csv(os.path.join(DATA_DIR, "precautions.csv"))
    workout_df     = pd.read_csv(os.path.join(DATA_DIR, "workout.csv"))
    return main_df, description_df, diets_df, medications_df, precautions_df, workout_df

@st.cache_resource
def load_models():
    svc_model = joblib.load(os.path.join(MODELS_DIR, "svc_model.pkl"))
    dt_model  = joblib.load(os.path.join(MODELS_DIR, "decision_tree_model.pkl"))
    le        = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    return svc_model, dt_model, le

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

main_df, description_df, diets_df, medications_df, precautions_df, workout_df = load_data()
svc_model, dt_model, le = load_models()
semantic_model = load_semantic_model()

# ─── Build lookups ────────────────────────────────────────────────────────────
def clean(name):
    return str(name).lower().replace("_", " ").strip()

X            = main_df.drop(columns=["diseases"])
symptom_list = X.columns.tolist()

description_map = {clean(r["Disease"]): r["Description"]  for _, r in description_df.iterrows()}
diets_map       = {clean(r["Disease"]): r["Diet"]          for _, r in diets_df.iterrows()}
medications_map = {clean(r["Disease"]): r["Medication"]    for _, r in medications_df.iterrows()}
workout_col     = workout_df.columns[1]
workout_map     = {clean(r["Disease"]): r[workout_col]     for _, r in workout_df.iterrows()}

precautions_map = {}
for _, row in precautions_df.iterrows():
    d = clean(row["Disease"])
    precautions_map[d] = [p for p in [row["Precaution_1"], row["Precaution_2"], row["Precaution_3"], row["Precaution_4"]] if pd.notna(p)]

disease_names = list(description_map.keys())

disease_symptoms = {}
for disease in main_df["diseases"].unique():
    d_df = main_df[main_df["diseases"] == disease]
    cols = d_df.loc[:, d_df.columns != "diseases"]
    disease_symptoms[disease] = sorted(cols.columns[(cols == 1).any()].tolist())

@st.cache_data
def get_embeddings():
    s_emb = semantic_model.encode(symptom_list, show_progress_bar=False)
    d_emb = semantic_model.encode(disease_names, show_progress_bar=False)
    return s_emb, d_emb

symptom_embeddings, disease_embeddings = get_embeddings()

translator = Translator()

amharic_labels = {
    "Disease": "በሽታ", "Description": "መግለጫ", "Dietary Plan": "የአመጋገብ እቅድ",
    "Medications": "መድሃኒቶች", "Workout/Activity": "ስፖርት/እንቅስቃሴ",
    "Precautions": "ጥንቃቄዎች", "Symptoms": "ምልክቶች",
}

def lbl(key, lang):
    return amharic_labels.get(key, key) if lang == "Amharic" else key

def to_amharic(text):
    if not text: return text
    try:
        return translator.translate(str(text), dest="am").text
    except:
        return text

def to_english(text):
    if not text: return text
    try:
        return translator.translate(str(text), dest="en").text
    except:
        return text

# ─── Prediction logic ─────────────────────────────────────────────────────────
def predict(symptoms_input, lang, threshold=0.6):
    processed = to_english(symptoms_input) if lang == "Amharic" else symptoms_input
    phrases   = [s.strip().lower() for s in processed.split(",") if s.strip()]
    found     = []
    for phrase in phrases:
        emb  = semantic_model.encode(phrase)
        sims = cosine_similarity([emb], symptom_embeddings)[0]
        idx  = int(np.argmax(sims))
        if sims[idx] > threshold:
            found.append(symptom_list[idx])
    found = list(set(found))
    if not found:
        return None, None, "❌ No symptoms recognized. Try different keywords."

    symptom_df = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    for sym in found:
        if sym in symptom_df.columns:
            symptom_df.iloc[0, symptom_df.columns.get_loc(sym)] = 1

    probs    = svc_model.predict_proba(symptom_df)[0]
    top_idx  = np.argsort(probs)[::-1][:4]
    top_dis  = le.inverse_transform(top_idx)
    top_conf = probs[top_idx]
    return top_dis, top_conf, found

def get_recommendations(disease_key, role, lang):
    if disease_key not in description_map:
        return None
    desc  = description_map.get(disease_key, "")
    diet  = diets_map.get(disease_key, "")
    meds  = medications_map.get(disease_key, "")
    wo    = workout_map.get(disease_key, "")
    precs = precautions_map.get(disease_key, [])
    syms  = [s.replace("_", " ").title() for s in disease_symptoms.get(disease_key, [])]

    if lang == "Amharic":
        desc  = to_amharic(desc)
        diet  = to_amharic(diet)
        meds  = to_amharic(meds)
        wo    = to_amharic(wo)
        precs = [to_amharic(p) for p in precs]
        syms  = [to_amharic(s) for s in syms]

    rec = {
        lbl("Description", lang):     desc,
        lbl("Symptoms", lang):        syms,
        lbl("Precautions", lang):     precs,
        lbl("Workout/Activity", lang): wo,
    }
    if role in ["Student", "Doctor"]:
        rec[lbl("Dietary Plan", lang)] = diet
        rec[lbl("Medications", lang)]  = meds
    return rec

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚕️ Settings")
    lang = st.selectbox("🌐 Language", ["English", "Amharic"])
    age  = st.number_input("🎂 Age", min_value=1, max_value=120, value=25)
    role = st.selectbox("👤 Role", ["Normal User", "Student", "Doctor"])
    if age < 18:
        st.error("⚠️ Access restricted for users under 18.")
    st.markdown("---")
    st.markdown("**Roles & Access**")
    st.markdown("- **Normal User** — Basic info + precautions")
    st.markdown("- **Student** — Full info including medications")
    st.markdown("- **Doctor** — Complete access")

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("🏥 Disease Prediction & Health Recommendation")
st.markdown("Enter your symptoms to get a prediction and personalized health recommendations.")

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "💊 Recommend", "💬 Chat"])

# ── Tab 1: Predict ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🔍 Disease Prediction")
    sym_input = st.text_area(
        "Enter symptoms (comma separated)" if lang == "English" else "ምልክቶችን ያስገቡ (በኮማ ይለዩ)",
        placeholder="e.g. headache, fever, vomiting" if lang == "English" else "ለምሳሌ፡ ራስ ምታት, ትኩሳት",
        height=100
    )
    if st.button("🔮 Predict Disease", use_container_width=True):
        if age < 18:
            st.error("Access restricted for users under 18.")
        elif not sym_input.strip():
            st.warning("Please enter at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms..."):
                top_dis, top_conf, found = predict(sym_input, lang)
            if top_dis is None:
                st.error(found)
            else:
                st.success(f"✅ Found {len(found)} matching symptom(s): {', '.join(found)}")
                st.subheader("🎯 Predicted Conditions")
                cols = st.columns(len(top_dis))
                for i, (col, dis, conf) in enumerate(zip(cols, top_dis, top_conf)):
                    with col:
                        st.metric(f"#{i+1} {'Most Likely' if i == 0 else ''}", dis.title(), f"{conf:.1%}")

                st.subheader(f"📋 Recommendations for {top_dis[0].title()}")
                rec = get_recommendations(clean(top_dis[0]), role, lang)
                if rec:
                    for key, val in rec.items():
                        with st.expander(f"**{key}**", expanded=(key in [lbl("Description", lang), lbl("Precautions", lang)])):
                            if isinstance(val, list):
                                for item in val:
                                    st.markdown(f"• {item}")
                            else:
                                st.markdown(val)

# ── Tab 2: Recommend ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("💊 Health Recommendations by Disease")
    disease_input = st.selectbox(
        "Select a disease" if lang == "English" else "በሽታ ይምረጡ",
        options=sorted(disease_names),
        format_func=lambda x: x.title()
    )
    if st.button("📋 Get Recommendations", use_container_width=True):
        if age < 18:
            st.error("Access restricted for users under 18.")
        else:
            with st.spinner("Loading recommendations..."):
                rec = get_recommendations(disease_input, role, lang)
            if rec:
                st.subheader(f"📋 {disease_input.title()}")
                for key, val in rec.items():
                    with st.expander(f"**{key}**", expanded=True):
                        if isinstance(val, list):
                            for item in val:
                                st.markdown(f"• {item}")
                        else:
                            st.markdown(val)

# ── Tab 3: Chat ───────────────────────────────────────────────────────────────
with tab3:
    st.subheader("💬 Health Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about a disease, symptoms, diet, medications..." if lang == "English" else "ስለ በሽታ ይጠይቁ...")
    if query:
        if age < 18:
            st.error("Access restricted for users under 18.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Thinking..."):
                processed = to_english(query) if lang == "Amharic" else query.lower()
                found_disease = None
                q_emb = semantic_model.encode(processed)
                sims  = cosine_similarity([q_emb], disease_embeddings)[0]
                best  = int(np.argmax(sims))
                if sims[best] > 0.5:
                    found_disease = disease_names[best]
                if not found_disease:
                    for key in description_map:
                        if key in processed.lower():
                            found_disease = key
                            break

                if found_disease:
                    parts = []
                    if any(w in processed.lower() for w in ["what is", "tell me", "description", "about"]):
                        desc = description_map.get(found_disease, "")
                        if lang == "Amharic": desc = to_amharic(desc)
                        parts.append(f"**{lbl('Description', lang)} of {found_disease.title()}:**\n{desc}")
                    if role in ["Student", "Doctor"]:
                        if any(w in processed.lower() for w in ["diet", "eat", "food"]):
                            diet = diets_map.get(found_disease, "")
                            if lang == "Amharic": diet = to_amharic(diet)
                            parts.append(f"**{lbl('Dietary Plan', lang)} for {found_disease.title()}:**\n{diet}")
                        if any(w in processed.lower() for w in ["medication", "medicine", "drug", "treatment"]):
                            med = medications_map.get(found_disease, "")
                            if lang == "Amharic": med = to_amharic(med)
                            parts.append(f"**{lbl('Medications', lang)} for {found_disease.title()}:**\n{med}")
                    if any(w in processed.lower() for w in ["precaution", "prevent", "care"]):
                        precs = ", ".join(precautions_map.get(found_disease, ["No specific precautions."]))
                        if lang == "Amharic": precs = to_amharic(precs)
                        parts.append(f"**{lbl('Precautions', lang)} for {found_disease.title()}:**\n{precs}")
                    if any(w in processed.lower() for w in ["workout", "exercise", "activity"]):
                        wo = workout_map.get(found_disease, "")
                        if lang == "Amharic": wo = to_amharic(wo)
                        parts.append(f"**{lbl('Workout/Activity', lang)} for {found_disease.title()}:**\n{wo}")

                    response = "\n\n".join(parts) if parts else f"I found info on **{found_disease.title()}**. Ask about its description, diet, medications, precautions, or workout."
                else:
                    response = "I couldn't find specific information. Try asking about a specific disease (e.g. 'What is Diabetes?')."
                    if lang == "Amharic":
                        response = to_amharic(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
