"""
DROP-IN REPLACEMENT for render_quick_select() in your app.py
Replace the entire render_quick_select function (and the CSS for chip buttons) with this.

Also add this to your ENHANCED CUSTOM CSS block (anywhere inside the <style> tag):

    .chip-btn > button {
        background: rgba(30,41,59,0.5) !important;
        color: #64748b !important;
        border: 1px solid rgba(71,85,105,0.35) !important;
        border-radius: 5px !important;
        padding: 2px 10px !important;
        font-size: 0.71rem !important;
        font-weight: 400 !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease !important;
    }
    .chip-btn > button:hover {
        background: rgba(20,184,166,0.1) !important;
        color: #94a3b8 !important;
        border-color: rgba(20,184,166,0.4) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    .chip-btn-active > button {
        background: rgba(20,184,166,0.18) !important;
        color: #5eead4 !important;
        border: 1px solid rgba(20,184,166,0.6) !important;
        border-radius: 5px !important;
        padding: 2px 10px !important;
        font-size: 0.71rem !important;
        font-weight: 600 !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease !important;
    }
    .cat-btn > button {
        background: transparent !important;
        color: #475569 !important;
        border: 1px solid rgba(71,85,105,0.3) !important;
        border-radius: 20px !important;
        padding: 2px 10px !important;
        font-size: 0.68rem !important;
        font-weight: 400 !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease !important;
    }
    .cat-btn > button:hover {
        color: #94a3b8 !important;
        border-color: rgba(71,85,105,0.6) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    .cat-btn-active > button {
        background: rgba(20,184,166,0.18) !important;
        color: #2dd4bf !important;
        border: 1px solid rgba(20,184,166,0.5) !important;
        border-radius: 20px !important;
        padding: 2px 10px !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.4 !important;
    }
"""

import streamlit as st


def render_quick_select(categorized_symptoms: dict):
    cats = list(categorized_symptoms.keys())

    # ── Init session state
    if "active_cat" not in st.session_state or st.session_state.active_cat not in cats:
        st.session_state.active_cat = cats[0]
    if "symptoms_selected" not in st.session_state:
        st.session_state.symptoms_selected = []

    active_cat = st.session_state.active_cat

    # ── Category pill row
    cat_cols = st.columns(len(cats))
    for i, cat in enumerate(cats):
        is_active = (cat == active_cat)
        css_class = "cat-btn-active" if is_active else "cat-btn"
        with cat_cols[i]:
            st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
            if st.button(cat, key=f"cat_pill_{i}"):
                st.session_state.active_cat = cat
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,"
        "transparent,rgba(20,184,166,0.2),transparent);margin:4px 0 8px'></div>",
        unsafe_allow_html=True,
    )

    # ── Symptom chips — 7 per row
    symptoms       = sorted(categorized_symptoms.get(active_cat, []))
    selected       = st.session_state.symptoms_selected
    CHIPS_PER_ROW  = 7

    for row_start in range(0, len(symptoms), CHIPS_PER_ROW):
        row_syms  = symptoms[row_start: row_start + CHIPS_PER_ROW]
        chip_cols = st.columns(len(row_syms))
        for j, sym in enumerate(row_syms):
            label     = sym.replace("_", " ").title()
            is_sel    = sym in selected
            css_class = "chip-btn-active" if is_sel else "chip-btn"
            btn_label = f"✓ {label}" if is_sel else label
            with chip_cols[j]:
                st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
                if st.button(btn_label, key=f"chip_{sym}"):
                    if sym in st.session_state.symptoms_selected:
                        st.session_state.symptoms_selected.remove(sym)
                    else:
                        st.session_state.symptoms_selected.append(sym)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
