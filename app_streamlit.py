import streamlit as st
import os

st.title("🔍 Diagnostic Check")

st.subheader("Files in current directory:")
st.write(os.listdir("."))

st.subheader("Working directory:")
st.write(os.getcwd())

st.subheader("Python packages:")
import importlib
for pkg in ["joblib", "sklearn", "sentence_transformers", "pandas", "numpy", "googletrans"]:
    try:
        importlib.import_module(pkg)
        st.success(f"✅ {pkg}")
    except ImportError as e:
        st.error(f"❌ {pkg}: {e}")
