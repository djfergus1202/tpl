
import streamlit as st
try:
    from .pages._plot_proxy import run as run_proxy
    run_proxy()
except Exception:
    try:
        from .pages import _plot_original
        _plot_original.main()
    except Exception as e:
        st.error("Plot Implications module failed to load.")
        st.exception(e)
