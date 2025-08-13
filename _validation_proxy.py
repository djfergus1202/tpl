
import streamlit as st
def run():
    try:
        from . import _validation_original as original
        original.main()
    except Exception as e:
        st.error("Could not run original Validation Results page.")
        st.exception(e)
