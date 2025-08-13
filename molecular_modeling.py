
import streamlit as st
try:
    from .pages._molecular_proxy import run as run_proxy
    run_proxy()
except Exception:
    try:
        from .pages import _molecular_original
        # If original module exposes Streamlit layout directly
    except Exception as e:
        st.error("Molecular Docking module failed to load.")
        st.exception(e)
