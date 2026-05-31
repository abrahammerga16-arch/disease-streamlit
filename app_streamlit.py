import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(
    page_title="MediCare – Integrated Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit chrome so the HTML fills the viewport
st.markdown(
    """
    <style>
        #MainMenu, header, footer { visibility: hidden; height: 0; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        section[data-testid="stSidebar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

HTML_PATH = os.path.join(os.path.dirname(__file__), "healthcare_dashboard_v2 (1)")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(
    html_content,
    height=3800,   # tall enough for the full page
    scrolling=True,
)
