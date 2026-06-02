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

/* ── Keyframe Animations */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideInLeft { from { opacity: 0; transform: translateX(-30px); } to { opacity: 1; transform: translateX(0); } }
@keyframes slideInRight { from { opacity: 0; transform: translateX(30px); } to { opacity: 1; transform: translateX(0); } }

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
    background: rgba(22, 27, 34, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 6px; gap: 4px;
    border: 1px solid rgba(48, 54, 61, 0.3);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #8b949e;
    border-radius: 8px; font-weight: 600;
    letter-spacing: 0.04em; padding: 10px 24px;
    border: none !important; font-size: 0.95rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
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
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 0.95rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(46,160,67,0.4);
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
}
.stButton > button:active { transform: translateY(0); }

/* PILL BUTTONS */
.pill-cat button {
    background: transparent !important; color: #8b949e !important;
    border: 1px solid rgba(139,148,158,0.4) !important;
    border-radius: 22px !important; padding: 8px 20px !important;
    font-size: 0.9rem !important; font-weight: 600 !important;
    height: auto !important; min-height: 36px !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: none !important; transform: none !important;
}
.pill-cat button:hover {
    background: rgba(20,184,166,0.1) !important; color: #2dd4bf !important;
    border-color: #14b8a6 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(20,184,166,0.2) !important;
}
.pill-cat-active button {
    background: linear-gradient(135deg, rgba(20,184,166,0.25), rgba(20,184,166,0.15)) !important;
    color: #2dd4bf !important; border: 1px solid #14b8a6 !important;
    box-shadow: 0 0 0 1px #14b8a6, 0 4px 12px rgba(20,184,166,0.3) !important;
}

.pill-sym button {
    background: rgba(33,38,45,0.6) !important; color: #c9d1d9 !important;
    border: 1px solid rgba(48,54,61,0.6) !important;
    border-radius: 20px !important; padding: 6px 16px !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    height: auto !important; min-height: 32px !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important; transform: none !important;
    text-align: center !important; white-space: nowrap !important;
}
.pill-sym button:hover {
    background: rgba(20,184,166,0.1) !important; color: #2dd4bf !important;
    border-color: #14b8a6 !important;
    transform: translateY(-1px) !important;
}
.pill-sym-active button {
    background: linear-gradient(135deg, rgba(20,184,166,0.25), rgba(20,184,166,0.15)) !important;
    color: #2dd4bf !important; border: 1px solid #14b8a6 !important;
    box-shadow: 0 0 0 1px #14b8a6, 0 2px 8px rgba(20,184,166,0.3) !important;
    font-weight: 600 !important;
}

/* Predictor card */
.predictor-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.7) 0%, rgba(33,38,45,0.5) 100%);
    border: 1px solid rgba(48,54,61,0.4); border-radius: 16px;
    padding: 28px 32px; margin-bottom: 20px;
    animation: fadeInUp 0.6s ease-out;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.predictor-card-title { color: #e6edf3; font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
.predictor-card-sub   { color: #8b949e; font-size: 0.9rem;  margin-bottom: 20px; }
.quick-select-label   { color: #8b949e; font-size: 0.7rem; font-weight: 700;
                        letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px; }
.divider-or { text-align: center; color: #6b7280; font-size: 0.7rem; font-weight: 700;
              letter-spacing: 0.12em; text-transform: uppercase; margin: 18px 0; }
.divider-or::before, .divider-or::after {
    content: ""; display: inline-block; width: 40%; height: 1px;
    background: rgba(48,54,61,0.5); vertical-align: middle; margin: 0 12px;
}
.section-label { color: #8b949e; font-size: 0.7rem; font-weight: 700;
                 letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px; }

/* Top control bar */
.top-control-bar {
    background: linear-gradient(135deg, rgba(22,27,34,0.7) 0%, rgba(33,38,45,0.5) 100%);
    border: 1px solid rgba(48,54,61,0.4); border-radius: 14px;
    padding: 16px 20px; margin-bottom: 24px;
    animation: fadeInUp 0.5s ease-out;
}
.top-control-bar .ctrl-label {
    color: #79c0ff; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px;
}

/* Result cards */
.result-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.8) 0%, rgba(33,38,45,0.6) 100%);
    border: 1px solid rgba(88,166,255,0.2); border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px;
    animation: fadeInUp 0.6s ease-out;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.result-card:hover {
    border-color: rgba(88,166,255,0.4);
    box-shadow: 0 8px 32px rgba(88,166,255,0.15);
    transform: translateY(-4px);
}
.result-card h4 { color: #58a6ff; font-family: 'IBM Plex Mono', monospace;
                  font-size: 0.82rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 12px; }
.result-card p, .result-card li { color: #c9d1d9; font-size: 0.95rem; line-height: 1.6; }

.advice-banner {
    background: linear-gradient(135deg, rgba(45,24,0,0.8) 0%, rgba(240,136,62,0.1) 100%);
    border-left: 4px solid #f0883e; padding: 12px 18px;
    border-radius: 0 8px 8px 0; color: #f0883e;
    font-size: 0.9rem; margin-top: 16px;
    animation: slideInLeft 0.5s ease-out;
}
.chat-bot {
    background: linear-gradient(135deg, rgba(33,38,45,0.9) 0%, rgba(31,111,235,0.1) 100%);
    border-left: 4px solid #58a6ff; padding: 16px 20px;
    border-radius: 0 12px 12px 0; color: #c9d1d9;
    font-size: 0.95rem; margin-top: 12px; line-height: 1.6;
    animation: slideInLeft 0.5s ease-out;
}
.section-header {
    color: #58a6ff; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase;
    border-bottom: 1px solid rgba(88,166,255,0.2);
    padding-bottom: 8px; margin-bottom: 18px;
    animation: slideInLeft 0.4s ease-out; font-weight: 700;
}
.stNumberInput input {
    background: rgba(33,38,45,0.8) !important; color: #e6edf3 !important;
    border: 1px solid rgba(48,54,61,0.5) !important; border-radius: 8px !important;
}
.access-denied {
    background: linear-grad
