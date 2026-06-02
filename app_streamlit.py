"""
Integrated Healthcare Dashboard — Streamlit version (Enhanced UI with Animations)
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
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# ENHANCED CUSTOM CSS WITH ANIMATIONS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&family=Poppins:wght@300;400;600;700&display=swap');

@keyframes fadeInUp   { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideInLeft{ from { opacity: 0; transform: translateX(-30px); } to { opacity: 1; transform: translateX(0); } }
@keyframes slideInRight{from { opacity: 0; transform: translateX(30px); }  to { opacity: 1; transform: translateX(0); } }

html, body, [class*="css"] {
    font-family: 'Poppins', 'IBM Plex Sans', sans-serif;
    scroll-behavior: smooth;
}
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1f2d 100%);
    color: #e6edf3;
    animation: fadeInUp 0.8s ease-out;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(22,27,34,0.6); backdrop-filter: blur(10px);
    border-radius: 12px; padding: 6px; gap: 4px;
    border: 1px solid rgba(48,54,61,0.3);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #8b949e;
    border-radius: 8px; font-weight: 600; letter-spacing: 0.04em;
    padding: 10px 24px; border: none !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    font-size: 0.95rem;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(88,166,255,0.1); color: #58a6ff; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #21262d 0%, #1f6feb 100%) !important;
    color: #fff !important;
    box-shadow: 0 4px 12px rgba(88,166,255,0.25);
}

/* Inputs */
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
    background: rgba(33,38,45,1) !important;
}

/* Generic buttons */
.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    color: #fff; border: none; border-radius: 8px;
    font-weight: 600; padding: 10px 24px;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    font-size: 0.95rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(46,160,67,0.4);
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
}
.stButton > button:active { transform: translateY(0); }

/* ── PILL BUTTONS — small + light teal */
.pill-cat button {
    background: rgba(255,255,255,0.04) !important;
    color: #2dd4bf !important;
    border: 1px solid rgba(45,212,191,0.35) !important;
    border-radius: 18px !important;
    padding: 4px 14px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    height: auto !important; min-height: 28px !important;
    line-height: 1.2 !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important; transform: none !important;
    white-space: nowrap !important;
}
.pill-cat button:hover {
    background: rgba(20,184,166,0.15) !important;
    color: #14b8a6 !important;
    border-color: #14b8a6 !important;
    transform: translateY(-1px) !important;
}
.pill-cat-active button {
    background: linear-gradient(135deg, #14b8a6 0%, #2dd4bf 100%) !important;
    color: #ffffff !important;
    border: 1px solid #14b8a6 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(20,184,166,0.4) !important;
}
.pill-cat-active button:hover { transform: translateY(-1px) !important; }

.pill-sym button {
    background: rgba(255,255,255,0.04) !important;
    color: #2dd4bf !important;
    border: 1px solid rgba(45,212,191,0.35) !important;
    border-radius: 18px !important;
    padding: 4px 12px !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    height: auto !important; min-height: 26px !important;
    line-height: 1.2 !important;
    transition: all 0.
