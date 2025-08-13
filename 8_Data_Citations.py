import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_citations import get_data_citations, get_computational_methods_citations, get_pharmacological_models_citations

st.set_page_config(page_title="Data Citations & Sources", page_icon="📚", layout="wide")

def main():
    st.title("📚 Data Citations & Sources")
    st.markdown("Complete attribution and citations for all data sources used in the Academic Research Platform")
    
    # Get all citation data
    data_citations = get_data_citations()
    methods_citations = get_computational_methods_citations()
    models_citations = get_pharmacological_models_citations()
    
    # Main sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Sources", "Computational Methods", "Pharmacological Models", "Complete Bibliography"
    ])
    
    with tab1:
        st.header("🗄️ Primary Data Sources")
        st.markdown("All external databases and APIs used for data retrieval:")
        
        for source_key, citation_info in data_citations.items():
            with st.expander(f"📊 {citation_info['description']}", expanded=False):
                st.markdown(f"**Citation:** {citation_info['citation']}")
                if 'doi' in citation_info:
                    st.markdown(f"**DOI:** [{citation_info['doi']}](https://doi.org/{citation_info['doi']})")
                if 'url' in citation_info:
                    st.markdown(f"**URL:** [{citation_info['url']}]({citation_info['url']})")
                
                # Usage context
                if source_key == 'uniprot':
                    st.markdown("**Usage:** Protein sequences, functional annotations, enzyme classifications, protein-protein interactions")
                elif source_key == 'pubchem':
                    st.markdown("**Usage:** Chemical structures, molecular properties, SMILES notation, compound identifiers")
                elif source_key == 'drugbank':
                    st.markdown("**Usage:** Drug targets, pharmacokinetic parameters, drug classifications, therapeutic indications")
                elif source_key == 'chembl':
                    st.markdown("**Usage:** Bioactivity data, IC50 values, Ki constants, structure-activity relationships")
                elif source_key == 'pdb':
                    st.markdown("**Usage:** 3D protein structures, binding sites, molecular conformations")
    
    with tab2:
        st.header("🔧 Computational Methods & Tools")
        st.markdown("Software libraries and algorithms used for data processing and analysis:")
        
        for method_key, citation_info in methods_citations.items():
            with st.expander(f"⚙️ {citation_info['description']}", expanded=False):
                st.markdown(f"**Citation:** {citation_info['citation']}")
                if 'doi' in citation_info:
                    st.markdown(f"**DOI:** [{citation_info['doi']}](https://doi.org/{citation_info['doi']})")
                if 'url' in citation_info:
                    st.markdown(f"**URL:** [{citation_info['url']}]({citation_info['url']})")
                
                # Implementation context
                if method_key == 'sympy':
                    st.markdown("**Usage:** Symbolic mathematics, differential equations, pharmacokinetic modeling")
                elif method_key == 'scipy':
                    st.markdown("**Usage:** Statistical analysis, optimization, numerical integration")
                elif method_key == 'rdkit':
                    st.markdown("**Usage:** Molecular descriptors, drug-likeness calculations, chemical fingerprints")
                elif method_key == 'py3dmol':
                    st.markdown("**Usage:** 3D molecular visualization, protein structure display")
                elif method_key == 'networkx':
                    st.markdown("**Usage:** Protein-drug interaction networks, graph analysis")
    
    with tab3:
        st.header("🧮 Pharmacological Models & Equations")
        st.markdown("Mathematical models and theoretical frameworks used for pharmacological analysis:")
        
        for model_key, citation_info in models_citations.items():
            with st.expander(f"📐 {citation_info['description']}", expanded=False):
                st.markdown(f"**Citation:** {citation_info['citation']}")
                if 'doi' in citation_info:
                    st.markdown(f"**DOI:** [{citation_info['doi']}](https://doi.org/{citation_info['doi']})")
                
                # Mathematical context
                if model_key == 'hill_equation':
                    st.latex(r'''
                    E = \frac{E_{max} \cdot C^n}{EC_{50}^n + C^n}
                    ''')
                    st.markdown("**Usage:** Dose-response relationships, receptor binding curves")
                elif model_key == 'pk_modeling':
                    st.latex(r'''
                    C(t) = \frac{D \cdot F \cdot k_a}{V_d(k_a - k_e)} \cdot (e^{-k_e \cdot t} - e^{-k_a \cdot t})
                    ''')
                    st.markdown("**Usage:** One-compartment pharmacokinetic modeling")
                elif model_key == 'competitive_inhibition':
                    st.latex(r'''
                    v = \frac{V_{max} \cdot [S]}{K_m(1 + \frac{[I]}{K_i}) + [S]}
                    ''')
                    st.markdown("**Usage:** Drug-drug interactions, enzyme inhibition")
    
    with tab4:
        st.header("📖 Complete Bibliography")
        st.markdown("Full bibliographic reference list for academic use:")
        
        all_citations = {}
        all_citations.update(data_citations)
        all_citations.update(methods_citations)
        all_citations.update(models_citations)
        
        # Format for academic use
        st.markdown("### References")
        
        # Sort citations alphabetically by first author
        sorted_citations = sorted(all_citations.items(), key=lambda x: x[1]['citation'].split('.')[0])
        
        for i, (key, citation_info) in enumerate(sorted_citations, 1):
            citation_text = f"{i}. {citation_info['citation']}"
            if 'doi' in citation_info:
                citation_text += f" DOI: {citation_info['doi']}"
            st.markdown(citation_text)
        
        # Export options
        st.markdown("### Export Options")
        
        # Generate BibTeX
        bibtex_content = generate_bibtex(all_citations)
        st.download_button(
            label="📥 Download BibTeX",
            data=bibtex_content,
            file_name="academic_research_platform_references.bib",
            mime="text/plain"
        )
        
        # Generate formatted text
        formatted_refs = generate_formatted_references(sorted_citations)
        st.download_button(
            label="📥 Download Text References",
            data=formatted_refs,
            file_name="academic_research_platform_references.txt",
            mime="text/plain"
        )
    
    # Academic use guidelines
    st.markdown("---")
    st.markdown("## 📋 Academic Use Guidelines")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Citation Requirements
        - Cite original data sources when using results
        - Include DOI links when available
        - Acknowledge computational methods used
        - Reference pharmacological models applied
        """)
    
    with col2:
        st.markdown("""
        ### Ethical Use
        - Validate computational results experimentally
        - Respect database terms of service
        - Use for academic/research purposes only
        - Ensure reproducibility by citing versions
        """)
    
    # Platform attribution
    st.info("""
    **Platform Citation:** When referencing this Academic Research Platform in publications, please cite:
    "Ferguson, D.J., BS, MS, PharmD Candidate, RSci MRSB MRSC. Academic Research Platform for Systematic Review Validation and Pharmacological Analysis. 
    Developed using Streamlit, integrating data from UniProt, PubChem, DrugBank, and ChEMBL databases. 
    Available at: [Your Replit URL]. 2025."
    """)

def generate_bibtex(citations_dict):
    """Generate BibTeX format citations"""
    bibtex_content = "% Academic Research Platform - Complete Bibliography\n"
    bibtex_content += "% Developed by: Ferguson, D.J., BS, MS, PharmD Candidate, RSci MRSB MRSC\n\n"
    
    # Add platform citation first
    bibtex_content += "@misc{ferguson2025platform,\n"
    bibtex_content += "  author = {Ferguson, David Joshua},\n"
    bibtex_content += "  title = {Academic Research Platform for Systematic Review Validation and Pharmacological Analysis},\n"
    bibtex_content += "  year = {2025},\n"
    bibtex_content += "  note = {Developed using Streamlit, integrating data from UniProt, PubChem, DrugBank, and ChEMBL databases}\n"
    bibtex_content += "}\n\n"
    
    for key, citation_info in citations_dict.items():
        # Extract basic info from citation string
        citation = citation_info['citation']
        
        # Generate BibTeX entry (simplified)
        bibtex_content += f"@article{{{key},\n"
        bibtex_content += f"  title = {{{citation.split('.')[0]}}},\n"
        bibtex_content += f"  note = {{{citation}}},\n"
        
        if 'doi' in citation_info:
            bibtex_content += f"  doi = {{{citation_info['doi']}}},\n"
        if 'url' in citation_info:
            bibtex_content += f"  url = {{{citation_info['url']}}},\n"
        
        bibtex_content += "}\n\n"
    
    return bibtex_content

def generate_formatted_references(sorted_citations):
    """Generate formatted text references"""
    content = "Academic Research Platform - Complete References\n"
    content += "Developed by: Ferguson, D.J., BS, MS, PharmD Candidate, RSci MRSB MRSC\n"
    content += "=" * 50 + "\n\n"
    
    for i, (key, citation_info) in enumerate(sorted_citations, 1):
        content += f"{i}. {citation_info['citation']}"
        if 'doi' in citation_info:
            content += f" DOI: {citation_info['doi']}"
        content += "\n\n"
    
    content += "\nPlatform Citation:\n"
    content += "Ferguson, D.J., BS, MS, PharmD Candidate, RSci MRSB MRSC. Academic Research Platform for Systematic Review Validation and Pharmacological Analysis. Developed using Streamlit, integrating data from UniProt, PubChem, DrugBank, and ChEMBL databases. 2025.\n\n"
    from datetime import datetime
    content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    return content

if __name__ == "__main__":
    main()