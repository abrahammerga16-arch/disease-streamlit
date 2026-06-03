"""
COMPLETE DROP-IN REPLACEMENT for render_quick_select() in your app.py

Two strategies are used:
  1. If Streamlit >= 1.37: uses st.pills() — the cleanest native solution
  2. Fallback: uses st.multiselect() per category — always works

HOW TO APPLY:
  - Replace the entire render_quick_select() function in app.py with the one below.
  - No CSS changes needed (removes the broken JS + hidden input entirely).
  - Also DELETE the hidden st.text_input("_qs", key="_qs_action", ...) block in Tab 1
    and the action-handling block at the top of render_quick_select.
"""

import streamlit as st
from packaging import version


def _streamlit_version() -> str:
    try:
        import streamlit
        return streamlit.__version__
    except Exception:
        return "0.0.0"


def render_quick_select(categorized_symptoms: dict):
    cats = list(categorized_symptoms.keys())

    # ── Init session state
    if "active_cat" not in st.session_state or st.session_state.active_cat not in cats:
        st.session_state.active_cat = cats[0]
    if "symptoms_selected" not in st.session_state:
        st.session_state.symptoms_selected = []

    use_pills = version.parse(_streamlit_version()) >= version.parse("1.37.0")

    # ────────────────────────────────────────────
    # STRATEGY 1 — st.pills  (Streamlit >= 1.37)
    # ────────────────────────────────────────────
    if use_pills:
        # Category selector
        active_cat = st.pills(
            "Category",
            options=cats,
            default=st.session_state.active_cat,
            key="cat_pills_selector",
            label_visibility="collapsed",
        )
        if active_cat and active_cat != st.session_state.active_cat:
            st.session_state.active_cat = active_cat
            st.rerun()

        active_cat = st.session_state.active_cat or cats[0]

        st.markdown(
            "<div style='height:1px;background:linear-gradient(90deg,"
            "transparent,rgba(20,184,166,0.2),transparent);margin:4px 0 8px'></div>",
            unsafe_allow_html=True,
        )

        symptoms = sorted(categorized_symptoms.get(active_cat, []))
        sym_labels = {s: s.replace("_", " ").title() for s in symptoms}

        # Show currently-selected that belong to this category so the widget
        # reflects the correct state even after switching categories.
        current_in_cat = [s for s in st.session_state.symptoms_selected if s in symptoms]

        selected_now = st.pills(
            "Symptoms",
            options=symptoms,
            format_func=lambda s: sym_labels[s],
            selection_mode="multi",
            default=current_in_cat,
            key=f"sym_pills_{active_cat}",
            label_visibility="collapsed",
        )

        # Merge: keep selections from OTHER categories, replace this category's selections
        other_cats_selected = [
            s for s in st.session_state.symptoms_selected if s not in symptoms
        ]
        merged = other_cats_selected + (selected_now or [])
        if set(merged) != set(st.session_state.symptoms_selected):
            st.session_state.symptoms_selected = merged
            st.rerun()

    # ────────────────────────────────────────────
    # STRATEGY 2 — multiselect fallback (always works)
    # ────────────────────────────────────────────
    else:
        # Category tabs via st.selectbox (compact)
        active_cat = st.selectbox(
            "Category",
            options=cats,
            index=cats.index(st.session_state.active_cat),
            key="cat_selectbox",
            label_visibility="collapsed",
        )
        if active_cat != st.session_state.active_cat:
            st.session_state.active_cat = active_cat

        st.markdown(
            "<div style='height:1px;background:linear-gradient(90deg,"
            "transparent,rgba(20,184,166,0.2),transparent);margin:4px 0 8px'></div>",
            unsafe_allow_html=True,
        )

        symptoms = sorted(categorized_symptoms.get(active_cat, []))
        current_in_cat = [s for s in st.session_state.symptoms_selected if s in symptoms]

        selected_now = st.multiselect(
            f"Select symptoms — {active_cat}",
            options=symptoms,
            default=current_in_cat,
            format_func=lambda s: s.replace("_", " ").title(),
            key=f"sym_multi_{active_cat}",
            label_visibility="collapsed",
        )

        other_cats_selected = [
            s for s in st.session_state.symptoms_selected if s not in symptoms
        ]
        merged = other_cats_selected + selected_now
        if set(merged) != set(st.session_state.symptoms_selected):
            st.session_state.symptoms_selected = merged
