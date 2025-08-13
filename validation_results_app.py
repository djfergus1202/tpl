
import streamlit as st
try:
    from .pages._validation_proxy import run as run_proxy
    run_proxy()
except Exception:
    try:
        from .pages import _validation_original
        _validation_original.main()
    except Exception as e:
        st.error("Validation Results module failed to load.")
        st.exception(e)
