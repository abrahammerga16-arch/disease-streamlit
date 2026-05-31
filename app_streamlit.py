import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(
    page_title="MediCare – Integrated Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, header, footer { visibility: hidden; height: 0; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Get Groq API key from Streamlit secrets (NO DEFAULT VALUE - SECURITY FIX)
groq_key = st.secrets.get("GROQ_API_KEY")
if not groq_key:
    st.error(
        "⚠️ **GROQ_API_KEY not found in Streamlit secrets.**\n\n"
        "Go to your app's **Settings → Secrets** and add:\n\n"
        "```toml\nGROQ_API_KEY = \"gsk_your_key_here\"\n```\n\n"
        "Get a free key at [console.groq.com](https://console.groq.com)."
    )
    st.stop()

# Load HTML file (must be committed to the repo as healthcare_dashboard.html)
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthcare_dashboard.html")

try:
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
except FileNotFoundError:
    st.error(
        "⚠️ **healthcare_dashboard.html not found.**\n\n"
        "Make sure the file is committed to your GitHub repo at the same level as `app_streamlit.py`."
    )
    st.stop()

# Inject Groq API key via JavaScript window variable (SECURE APPROACH)
html_with_key = f"""
<script>
    window.GROQ_API_KEY = '{groq_key}';
</script>
{html_content}
"""

components.html(html_with_key, height=3800, scrolling=True)
