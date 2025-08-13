
import streamlit as st
def run():
    try:
        from . import _molecular_original as original
        # Some original pages execute on import; no explicit main
    except Exception as e:
        st.error("Could not run original Molecular Docking page.")
        st.exception(e)
