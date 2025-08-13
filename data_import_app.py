
import streamlit as st
try:
    # Use user's original Streamlit page if available
    from .pages._data_import_proxy import run as run_proxy
    run_proxy()
except Exception:
    # Fallback: light-weight import of original page module
    try:
        from .pages import _data_import_original
        _data_import_original.main()
    except Exception as e:
        st.error("Data Import module failed to load.")
        st.exception(e)
