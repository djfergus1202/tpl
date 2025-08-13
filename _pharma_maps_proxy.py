
import streamlit as st
def run():
    try:
        from . import _pharma_maps_original as original
        # Some original pages execute on import; no explicit main
    except Exception as e:
        st.error("Could not run original Pharmacological Maps page.")
        st.exception(e)
