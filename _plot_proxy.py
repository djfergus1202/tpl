
import streamlit as st
def run():
    try:
        from . import _plot_original as original
        original.main()
    except Exception as e:
        st.error("Could not run original Plot Implications page.")
        st.exception(e)
