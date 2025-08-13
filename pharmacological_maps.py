
import streamlit as st
try:
    from .pages._pharma_maps_proxy import run as run_proxy
    run_proxy()
except Exception:
    try:
        from .pages import _pharma_maps_original
    except Exception as e:
        st.error("Pharmacological Maps module failed to load.")
        st.exception(e)
