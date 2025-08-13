import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Plot Implications & Clinical Insights", page_icon="📈", layout="wide")

def main():
    st.title("📈 Plot Implications & Clinical Insights")
    st.markdown("Understanding the clinical and research implications of pharmacological topological maps")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "3D Topological Maps", "Network Analysis", "Therapeutic Windows", "Statistical Distributions", "Clinical Applications"
    ])
    
    with tab1:
        st.header("🧬 3D Pharmacological Topological Maps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("What the 3D Maps Show")
            st.markdown("""
            **Axes Interpretation:**
            - **Drug Response (Z-axis)**: Measured biological effect or therapeutic efficacy
            - **Log(Agonist Concentration) (Y-axis)**: Drug dose concentration on logarithmic scale
            - **Time (X-axis)**: Duration of drug exposure showing onset, peak, and decline phases
            
            **Color Coding:**
            - **Intensity**: Represents therapeutic window or drug-likeness
            - **Gradients**: Show transitions between effective and toxic doses
            - **Clusters**: Indicate drugs with similar mechanisms of action
            """)
            
        with col2:
            st.subheader("Clinical Implications")
            st.markdown("""
            **Drug Development Insights:**
            - **Peak Heights**: Maximum therapeutic effect achievable at optimal dosing
            - **Peak Locations**: Ideal drug concentration and timing for best therapeutic outcome
            - **Surface Smoothness**: Consistent and predictable patient response patterns
            - **Multiple Peaks**: Active metabolites, sustained-release formulations, or multiple mechanism pathways
            
            **Safety Considerations:**
            - **Steep Gradients**: Narrow therapeutic windows requiring careful dosing
            - **Plateau Regions**: Safe operating zones with consistent efficacy
            - **Valley Areas**: Subtherapeutic or potentially dangerous regions
            """)
        
        st.markdown("---")
        st.subheader("🎯 Therapeutic Optimization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Dosing Strategy:**
            - Identify optimal concentration ranges
            - Determine dosing frequency from time profiles
            - Assess need for loading doses
            - Evaluate steady-state requirements
            """)
        
        with col2:
            st.markdown("""
            **Patient Stratification:**
            - Population-specific response patterns
            - Genetic variant considerations
            - Age and organ function impacts
            - Drug-drug interaction predictions
            """)
            
        with col3:
            st.markdown("""
            **Formulation Design:**
            - Sustained vs immediate release needs
            - Bioavailability optimization
            - Route of administration selection
            - Combination therapy potential
            """)
    
    with tab2:
        st.header("🕸️ Protein-Drug Interaction Networks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Network Structure Analysis")
            st.markdown("""
            **Node Properties:**
            - **Size**: Represents binding affinity or importance
            - **Color**: Drug class or protein family classification
            - **Position**: Clustering by functional similarity
            
            **Edge Properties:**
            - **Thickness**: Strength of interaction
            - **Color**: Type of interaction (agonist, antagonist, allosteric)
            - **Pattern**: Confidence level or experimental validation
            """)
        
        with col2:
            st.subheader("Clinical Network Implications")
            st.markdown("""
            **Target Selectivity:**
            - Hub proteins: High-value therapeutic targets
            - Isolated nodes: Highly selective drugs
            - Dense clusters: Pathway-specific interventions
            
            **Polypharmacology:**
            - Multiple connections: Multi-target drugs
            - Pathway coverage: Comprehensive therapeutic effects
            - Side effect prediction: Off-target interactions
            """)
        
        st.markdown("---")
        st.subheader("🔬 Drug Discovery Insights")
        
        # Create example network visualization
        fig_network_example = create_example_network()
        st.plotly_chart(fig_network_example, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Lead Optimization:**
            - Identify scaffolds connecting multiple targets
            - Optimize selectivity profiles
            - Predict metabolic pathways through CYP450 connections
            - Assess druggability of target combinations
            """)
        
        with col2:
            st.markdown("""
            **Resistance Mechanisms:**
            - Backup pathway identification
            - Compensation mechanism prediction
            - Combination therapy design
            - Biomarker development for patient selection
            """)
    
    with tab3:
        st.header("📊 Therapeutic Window Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Therapeutic Index Interpretation")
            
            # Create example therapeutic window plot
            fig_tw = create_therapeutic_window_example()
            st.plotly_chart(fig_tw, use_container_width=True)
            
        with col2:
            st.subheader("Safety Margin Assessment")
            st.markdown("""
            **Window Width:**
            - **Wide windows**: Safer drugs, easier dosing
            - **Narrow windows**: Require therapeutic drug monitoring
            - **Variable windows**: Population-dependent safety
            
            **Window Position:**
            - **Low threshold**: High potency, low therapeutic doses
            - **High threshold**: May require high doses, absorption issues
            - **Shifted windows**: Population-specific differences
            """)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Clinical Monitoring:**
            - Blood level monitoring frequency
            - Biomarker selection for efficacy
            - Safety parameter monitoring
            - Dose adjustment protocols
            """)
        
        with col2:
            st.markdown("""
            **Personalized Medicine:**
            - Genetic testing requirements
            - Pharmacokinetic modeling needs
            - Population-specific dosing
            - Precision medicine opportunities
            """)
            
        with col3:
            st.markdown("""
            **Regulatory Considerations:**
            - Risk-benefit assessment
            - Special population studies
            - Post-market surveillance
            - Label optimization
            """)
    
    with tab4:
        st.header("📈 Statistical Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution Shape Implications")
            
            # Create example distribution plots
            fig_dist = create_distribution_examples()
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with col2:
            st.subheader("Population Pharmacokinetics")
            st.markdown("""
            **Normal Distributions:**
            - Predictable population responses
            - Standard dosing regimens applicable
            - Classical pharmacokinetic models valid
            
            **Skewed Distributions:**
            - Population subgroups with different responses
            - Need for stratified dosing approaches
            - Genetic or demographic factors influential
            
            **Bimodal Distributions:**
            - Distinct population subgroups
            - Metabolizer phenotypes (poor vs extensive)
            - Require different therapeutic strategies
            """)
        
        st.markdown("---")
        st.subheader("🎯 Precision Dosing Implications")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Dose Optimization:**
            - Population mean as starting point
            - Individual adjustment based on response
            - Bayesian feedback for refinement
            - Covariate-based initial dosing
            """)
        
        with col2:
            st.markdown("""
            **Biomarker Stratification:**
            - Identify response predictors
            - Develop diagnostic companions
            - Enable precision medicine
            - Optimize clinical trial design
            """)
            
        with col3:
            st.markdown("""
            **Risk Assessment:**
            - Identify outlier populations
            - Assess extreme response risks
            - Develop safety monitoring
            - Plan mitigation strategies
            """)
    
    with tab5:
        st.header("🏥 Clinical Applications & Decision Support")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clinical Decision Making")
            st.markdown("""
            **Drug Selection:**
            - Compare efficacy profiles across time
            - Assess safety margins for patient populations  
            - Evaluate drug-drug interaction potential
            - Consider patient-specific factors
            
            **Dosing Decisions:**
            - Initial dose selection based on population data
            - Titration strategies from time-response curves
            - Monitoring frequency from therapeutic windows
            - Combination therapy optimization
            """)
        
        with col2:
            st.subheader("Clinical Trial Design")
            st.markdown("""
            **Study Population:**
            - Identify responder vs non-responder populations
            - Design enrichment strategies
            - Plan adaptive trial designs
            - Optimize sample sizes
            
            **Endpoint Selection:**
            - Choose optimal measurement timepoints
            - Select appropriate biomarkers
            - Design composite endpoints
            - Plan interim analyses
            """)
        
        st.markdown("---")
        st.subheader("🔬 Translational Research Applications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Mechanism of Action Studies:**
            - Identify key pathways and targets
            - Understand temporal dynamics of drug action
            - Map drug-protein interaction networks
            - Predict synergistic combinations
            
            **Biomarker Development:**
            - Correlate molecular signatures with response
            - Develop predictive algorithms
            - Validate companion diagnostics
            - Enable precision medicine approaches
            """)
        
        with col2:
            st.markdown("""
            **Drug Repurposing:**
            - Identify new therapeutic applications
            - Understand off-target effects
            - Map drug similarity networks
            - Predict therapeutic potential in new indications
            
            **Safety Assessment:**
            - Predict adverse event profiles
            - Identify vulnerable populations
            - Design risk mitigation strategies
            - Optimize benefit-risk profiles
            """)
        
        st.markdown("---")
        st.subheader("📋 Regulatory and Commercial Implications")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Regulatory Strategy:**
            - Support dose selection rationale
            - Justify special population studies
            - Design post-market studies
            - Optimize product labeling
            """)
        
        with col2:
            st.markdown("""
            **Market Access:**
            - Demonstrate clinical value
            - Support health economic models
            - Enable precision medicine pricing
            - Justify therapeutic positioning
            """)
            
        with col3:
            st.markdown("""
            **Clinical Implementation:**
            - Develop dosing guidelines
            - Create clinical decision tools
            - Train healthcare providers
            - Monitor real-world outcomes
            """)
        
        # Key insights summary
        st.markdown("---")
        st.info("""
        **Key Takeaway:** Pharmacological topological maps provide a multi-dimensional view of drug action that enables:
        - **Precision dosing** based on individual patient characteristics
        - **Optimal therapeutic windows** identification for maximum efficacy and safety  
        - **Drug combination strategies** based on network interactions
        - **Population stratification** for personalized medicine approaches
        - **Clinical decision support** for healthcare providers
        """)
        
        # Platform citation
        st.markdown("---")
        st.markdown("### 📚 Platform Citation")
        st.info("""
        **When citing this platform:** Ferguson, D.J., BS, MS, PharmD Candidate, RSci MRSB MRSC. Academic Research Platform for Systematic Review Validation and Pharmacological Analysis. 
        Developed using Streamlit, integrating data from UniProt, PubChem, DrugBank, and ChEMBL databases. 2025.
        """)

def create_example_network():
    """Create example protein-drug interaction network"""
    fig = go.Figure()
    
    # Example nodes and edges
    proteins = ['CYP3A4', 'CYP2D6', 'ABCB1', 'OATP1B1']
    drugs = ['Drug A', 'Drug B', 'Drug C']
    
    # Add protein nodes
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 1],
        y=[1, 0, 1, 2],
        mode='markers+text',
        marker=dict(size=30, color='lightblue', line=dict(width=2)),
        text=proteins,
        textposition="middle center",
        name='Proteins'
    ))
    
    # Add drug nodes
    fig.add_trace(go.Scatter(
        x=[0.5, 1.5, 1],
        y=[0.5, 0.5, 1.5],
        mode='markers+text',
        marker=dict(size=25, color='lightcoral', line=dict(width=2)),
        text=drugs,
        textposition="middle center",
        name='Drugs'
    ))
    
    # Add edges
    edges_x = [0, 0.5, None, 1, 1.5, None, 2, 1, None]
    edges_y = [1, 0.5, None, 0, 0.5, None, 1, 1.5, None]
    
    fig.add_trace(go.Scatter(
        x=edges_x, y=edges_y,
        mode='lines',
        line=dict(color='gray', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Example Protein-Drug Interaction Network",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )
    
    return fig

def create_therapeutic_window_example():
    """Create example therapeutic window plot"""
    concentration = np.linspace(0, 100, 100)
    efficacy = 100 * concentration / (concentration + 20)
    toxicity = 0.1 * concentration ** 1.5
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=concentration, y=efficacy,
        mode='lines',
        name='Efficacy',
        line=dict(color='green', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=concentration, y=toxicity,
        mode='lines',
        name='Toxicity',
        line=dict(color='red', width=3)
    ))
    
    # Therapeutic window
    fig.add_vrect(
        x0=15, x1=40,
        fillcolor="lightgreen", opacity=0.3,
        layer="below", line_width=0,
    )
    
    fig.add_annotation(
        x=27.5, y=50,
        text="Therapeutic<br>Window",
        showarrow=True,
        arrowhead=2,
        arrowcolor="black"
    )
    
    fig.update_layout(
        title="Therapeutic Window Analysis",
        xaxis_title="Drug Concentration (ng/mL)",
        yaxis_title="Response (%)",
        height=400
    )
    
    return fig

def create_distribution_examples():
    """Create example distribution plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Normal Distribution', 'Skewed Distribution', 
                       'Bimodal Distribution', 'Uniform Distribution')
    )
    
    x = np.linspace(-4, 4, 100)
    
    # Normal distribution
    y1 = np.exp(-x**2/2) / np.sqrt(2*np.pi)
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Normal'), row=1, col=1)
    
    # Skewed distribution
    y2 = np.exp(-x) * (x > 0)
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Skewed'), row=1, col=2)
    
    # Bimodal distribution
    y3 = 0.5 * (np.exp(-(x+1.5)**2/0.5) + np.exp(-(x-1.5)**2/0.5))
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='Bimodal'), row=2, col=1)
    
    # Uniform distribution
    y4 = np.ones_like(x) * 0.25 * ((x > -2) & (x < 2))
    fig.add_trace(go.Scatter(x=x, y=y4, mode='lines', name='Uniform'), row=2, col=2)
    
    fig.update_layout(
        title="Population Response Distributions",
        height=500,
        showlegend=False
    )
    
    return fig

if __name__ == "__main__":
    main()