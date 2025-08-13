
import streamlit as st
def run():
    try:
        # Try user's Data Import page
        from . import  _data_import_original as original
        original.main()
    except Exception as e:
        st.error("Could not run original Data Import page.")
        st.exception(e)
