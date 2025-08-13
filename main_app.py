
import streamlit as st
from importlib import import_module

st.set_page_config(page_title="Pharma Lab Suite", page_icon="🧪", layout="wide")

st.title("🧪 Pharma Lab Suite")
st.caption("Data analysis • Molecular modeling • Pharmacological topological maps • Paper generation")

pages = {
    "📁 Data Import & Processing": "pharma_lab_suite.data_import_app",
    "🧬 Molecular Docking & Simulation": "pharma_lab_suite.molecular_modeling",
    "🗺️ Pharmacological Topological Maps": "pharma_lab_suite.pharmacological_maps",
    "📊 Validation Results Dashboard": "pharma_lab_suite.validation_results_app",
    "📈 Plot Implications & Clinical Insights": "pharma_lab_suite.plot_implications_app",
    "📄 Paper Generator": "pharma_lab_suite.doc_generator_integration",
}

choice = st.sidebar.selectbox("Navigate", list(pages.keys()))

module_name = pages[choice]
try:
    mod = import_module(module_name)
    if hasattr(mod, "main"):
        mod.main()
    else:
        st.error(f"Selected module '{module_name}' does not expose a main() function.")
except Exception as e:
    st.exception(e)
