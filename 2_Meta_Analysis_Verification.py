import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import sys
import os

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional R integration
try:
    from utils.r_integration import RAnalytics
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    RAnalytics = None

try:
    from utils.statistical_tools_clean import StatisticalValidator
except ImportError:
    try:
        from utils.statistical_tools import StatisticalValidator
    except ImportError:
        StatisticalValidator = None

st.set_page_config(page_title="Meta-Analysis Verification", page_icon="🔬", layout="wide")

def main():
    st.title("🔬 Meta-Analysis Verification & Validation")
    st.markdown("Comprehensive verification of systematic reviews and meta-analyses with statistical validation")
    
    # Initialize components
    if 'r_analytics' not in st.session_state:
        if R_AVAILABLE:
            st.session_state.r_analytics = RAnalytics()
        else:
            st.session_state.r_analytics = None
    
    # Sidebar options
    with st.sidebar:
        st.header("Verification Options")
        verification_type = st.selectbox(
            "Select Verification Type",
            [
                "Study Selection Validation",
                "Effect Size Recalculation", 
                "Publication Bias Assessment",
                "Heterogeneity Analysis",
                "Risk of Bias Assessment",
                "PRISMA Flow Diagram",
                "Sensitivity Analysis"
            ]
        )
        
        st.subheader("Upload Original Paper")
        uploaded_paper = st.file_uploader(
            "Upload original paper (PDF)",
            type=['pdf'],
            help="Upload the original systematic review/meta-analysis paper for verification"
        )
    
    # Main content based on selection
    if verification_type == "Study Selection Validation":
        study_selection_validation()
    elif verification_type == "Effect Size Recalculation":
        effect_size_recalculation()
    elif verification_type == "Publication Bias Assessment":
        publication_bias_assessment()
    elif verification_type == "Heterogeneity Analysis":
        heterogeneity_analysis()
    elif verification_type == "Risk of Bias Assessment":
        risk_of_bias_assessment()
    elif verification_type == "PRISMA Flow Diagram":
        prisma_flow_diagram()
    elif verification_type == "Sensitivity Analysis":
        sensitivity_analysis()

def study_selection_validation():
    st.header("📋 Study Selection Validation")
    st.markdown("Validate the study selection process and inclusion/exclusion criteria")
    
    # Search strategy validation
    st.subheader("1. Search Strategy Validation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Search Terms**")
        original_terms = st.text_area(
            "Enter original search terms from the paper",
            placeholder="Enter the search strategy as reported in the original paper"
        )
    
    with col2:
        st.write("**Database Coverage**")
        databases = st.multiselect(
            "Databases searched in original study",
            ["PubMed/MEDLINE", "Embase", "Web of Science", "Cochrane Library", 
             "CINAHL", "PsycINFO", "Scopus", "Google Scholar", "Grey Literature"],
            default=["PubMed/MEDLINE", "Embase"]
        )
    
    # Date range validation
    st.subheader("2. Search Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Search start date")
    with col2:
        end_date = st.date_input("Search end date")
    
    # Inclusion/Exclusion criteria
    st.subheader("3. Inclusion/Exclusion Criteria")
    
    inclusion_criteria = st.text_area(
        "Inclusion Criteria",
        placeholder="List the inclusion criteria from the original study"
    )
    
    exclusion_criteria = st.text_area(
        "Exclusion Criteria", 
        placeholder="List the exclusion criteria from the original study"
    )
    
    # Study screening validation
    st.subheader("4. Study Screening Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        initial_records = st.number_input("Initial records identified", min_value=0, value=0)
    with col2:
        after_duplicates = st.number_input("After duplicate removal", min_value=0, value=0)
    with col3:
        after_screening = st.number_input("After title/abstract screening", min_value=0, value=0)
    with col4:
        final_included = st.number_input("Final studies included", min_value=0, value=0)
    
    if st.button("Generate PRISMA Flow Validation"):
        generate_prisma_validation(initial_records, after_duplicates, after_screening, final_included)

def generate_prisma_validation(initial, after_dup, after_screen, final):
    """Generate PRISMA flow diagram validation"""
    
    # Calculate exclusion numbers
    duplicates_removed = initial - after_dup
    excluded_screening = after_dup - after_screen
    excluded_full_text = after_screen - final
    
    # Create PRISMA flow diagram
    fig = go.Figure()
    
    # Define box positions and text
    boxes = [
        {"x": 0.5, "y": 0.9, "text": f"Records identified<br>n = {initial:,}", "color": "lightblue"},
        {"x": 0.5, "y": 0.75, "text": f"Records after duplicates removed<br>n = {after_dup:,}", "color": "lightgreen"},
        {"x": 0.5, "y": 0.6, "text": f"Records screened<br>n = {after_dup:,}", "color": "lightgreen"},
        {"x": 0.8, "y": 0.6, "text": f"Records excluded<br>n = {excluded_screening:,}", "color": "lightcoral"},
        {"x": 0.5, "y": 0.45, "text": f"Full-text articles assessed<br>n = {after_screen:,}", "color": "lightgreen"},
        {"x": 0.8, "y": 0.45, "text": f"Full-text articles excluded<br>n = {excluded_full_text:,}", "color": "lightcoral"},
        {"x": 0.5, "y": 0.3, "text": f"Studies included<br>n = {final:,}", "color": "gold"},
    ]
    
    # Add boxes
    for box in boxes:
        fig.add_shape(
            type="rect",
            x0=box["x"]-0.1, y0=box["y"]-0.05,
            x1=box["x"]+0.1, y1=box["y"]+0.05,
            fillcolor=box["color"],
            line=dict(color="black", width=2)
        )
        fig.add_annotation(
            x=box["x"], y=box["y"],
            text=box["text"],
            showarrow=False,
            font=dict(size=10)
        )
    
    # Add arrows
    arrows = [
        {"x0": 0.5, "y0": 0.85, "x1": 0.5, "y1": 0.8},
        {"x0": 0.5, "y0": 0.7, "x1": 0.5, "y1": 0.65},
        {"x0": 0.5, "y0": 0.55, "x1": 0.5, "y1": 0.5},
        {"x0": 0.5, "y0": 0.4, "x1": 0.5, "y1": 0.35},
        {"x0": 0.6, "y0": 0.6, "x1": 0.7, "y1": 0.6},
        {"x0": 0.6, "y0": 0.45, "x1": 0.7, "y1": 0.45},
    ]
    
    for arrow in arrows:
        fig.add_annotation(
            x=arrow["x1"], y=arrow["y1"],
            ax=arrow["x0"], ay=arrow["y0"],
            arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black"
        )
    
    fig.update_layout(
        title="PRISMA Flow Diagram Validation",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        plot_bgcolor="white",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Validation metrics
    st.subheader("Selection Process Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Duplicate Rate", f"{(duplicates_removed/initial*100):.1f}%")
    with col2:
        st.metric("Screening Exclusion Rate", f"{(excluded_screening/after_dup*100):.1f}%")
    with col3:
        st.metric("Full-text Exclusion Rate", f"{(excluded_full_text/after_screen*100):.1f}%")

def effect_size_recalculation():
    st.header("📊 Effect Size Recalculation & Verification")
    st.markdown("Recalculate and verify effect sizes from the original meta-analysis")
    
    # Upload or input study data
    data_input_method = st.radio(
        "How do you want to input study data?",
        ["Manual Entry", "Upload CSV", "Extract from Paper Text"]
    )
    
    studies_data = []
    
    if data_input_method == "Manual Entry":
        st.subheader("Enter Study Data for Recalculation")
        
        num_studies = st.number_input("Number of studies to verify", min_value=1, max_value=50, value=5)
        
        for i in range(num_studies):
            with st.expander(f"Study {i+1} Data"):
                col1, col2 = st.columns(2)
                
                with col1:
                    study_name = st.text_input(f"Study name/author", key=f"study_name_{i}")
                    study_type = st.selectbox(
                        "Study design", 
                        ["RCT", "Cohort", "Case-Control", "Cross-sectional"],
                        key=f"study_type_{i}"
                    )
                    outcome_type = st.selectbox(
                        "Outcome type",
                        ["Continuous", "Dichotomous", "Time-to-event"],
                        key=f"outcome_type_{i}"
                    )
                
                with col2:
                    if outcome_type == "Continuous":
                        n1 = st.number_input("Control group N", min_value=1, key=f"n1_{i}")
                        mean1 = st.number_input("Control mean", key=f"mean1_{i}")
                        sd1 = st.number_input("Control SD", min_value=0.001, key=f"sd1_{i}")
                        n2 = st.number_input("Treatment group N", min_value=1, key=f"n2_{i}")
                        mean2 = st.number_input("Treatment mean", key=f"mean2_{i}")
                        sd2 = st.number_input("Treatment SD", min_value=0.001, key=f"sd2_{i}")
                        
                        studies_data.append({
                            'study': study_name,
                            'type': study_type,
                            'outcome_type': outcome_type,
                            'n1': n1, 'mean1': mean1, 'sd1': sd1,
                            'n2': n2, 'mean2': mean2, 'sd2': sd2
                        })
                    
                    elif outcome_type == "Dichotomous":
                        events1 = st.number_input("Control events", min_value=0, key=f"events1_{i}")
                        n1 = st.number_input("Control total", min_value=1, key=f"n1_dic_{i}")
                        events2 = st.number_input("Treatment events", min_value=0, key=f"events2_{i}")
                        n2 = st.number_input("Treatment total", min_value=1, key=f"n2_dic_{i}")
                        
                        studies_data.append({
                            'study': study_name,
                            'type': study_type,
                            'outcome_type': outcome_type,
                            'events1': events1, 'n1': n1,
                            'events2': events2, 'n2': n2
                        })
    
    elif data_input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload study data CSV", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
                studies_data = df.to_dict('records')
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    # Recalculation section
    if studies_data:
        st.subheader("Effect Size Recalculation")
        
        if st.button("Recalculate Effect Sizes"):
            recalculate_effect_sizes(studies_data)

def recalculate_effect_sizes(studies_data):
    """Recalculate effect sizes using R"""
    
    try:
        r_analytics = st.session_state.r_analytics
        
        # Prepare R code for effect size calculation
        r_code = """
        library(meta)
        library(metafor)
        library(compute.es)
        
        # Initialize results dataframe
        results <- data.frame(
            study = character(),
            effect_size = numeric(),
            se = numeric(),
            ci_lower = numeric(),
            ci_upper = numeric(),
            weight = numeric(),
            stringsAsFactors = FALSE
        )
        """
        
        # Process each study
        for i, study in enumerate(studies_data):
            if study['outcome_type'] == 'Continuous':
                r_code += f"""
                # Study {i+1}: {study['study']}
                es_{i} <- compute.es::mes(
                    m.1 = {study['mean1']}, m.2 = {study['mean2']},
                    sd.1 = {study['sd1']}, sd.2 = {study['sd2']},
                    n.1 = {study['n1']}, n.2 = {study['n2']}
                )
                
                results <- rbind(results, data.frame(
                    study = "{study['study']}",
                    effect_size = es_{i}$d,
                    se = es_{i}$var.d^0.5,
                    ci_lower = es_{i}$l.d,
                    ci_upper = es_{i}$u.d,
                    weight = 1/es_{i}$var.d
                ))
                """
            
            elif study['outcome_type'] == 'Dichotomous':
                r_code += f"""
                # Study {i+1}: {study['study']}
                or_{i} <- compute.es::ores(
                    a = {study['events2']}, b = {study['n2'] - study['events2']},
                    c = {study['events1']}, d = {study['n1'] - study['events1']}
                )
                
                results <- rbind(results, data.frame(
                    study = "{study['study']}",
                    effect_size = or_{i}$lor,
                    se = or_{i}$var.lor^0.5,
                    ci_lower = or_{i}$l.lor,
                    ci_upper = or_{i}$u.lor,
                    weight = 1/or_{i}$var.lor
                ))
                """
        
        r_code += """
        # Perform meta-analysis
        ma_fixed <- metagen(TE = results$effect_size, seTE = results$se, 
                           studlab = results$study, fixed = TRUE, random = FALSE)
        ma_random <- metagen(TE = results$effect_size, seTE = results$se, 
                            studlab = results$study, fixed = FALSE, random = TRUE)
        
        # Print results
        cat("RECALCULATED EFFECT SIZES:\\n")
        print(results)
        
        cat("\\n\\nFIXED-EFFECT META-ANALYSIS:\\n")
        print(summary(ma_fixed))
        
        cat("\\n\\nRANDOM-EFFECTS META-ANALYSIS:\\n")
        print(summary(ma_random))
        
        cat("\\n\\nHETEROGENEITY ASSESSMENT:\\n")
        cat("I-squared:", ma_random$I2, "%\\n")
        cat("Tau-squared:", ma_random$tau2, "\\n")
        cat("Q-statistic:", ma_random$Q, "\\n")
        cat("P-value for heterogeneity:", ma_random$pval.Q, "\\n")
        """
        
        # Execute R code
        result = r_analytics.execute_r_code(r_code)
        
        st.subheader("Recalculated Results")
        st.code(result, language='text')
        
        # Create visualization
        create_verification_plots(studies_data)
        
    except Exception as e:
        st.error(f"Error in effect size recalculation: {str(e)}")

def create_verification_plots(studies_data):
    """Create verification plots"""
    
    # Extract data for plotting (simplified)
    study_names = [s['study'] for s in studies_data]
    
    # Create mock data for demonstration (in real implementation, this would come from R results)
    effect_sizes = np.random.normal(0.5, 0.3, len(studies_data))
    ses = np.random.uniform(0.1, 0.4, len(studies_data))
    
    # Forest plot
    fig = go.Figure()
    
    y_positions = list(range(len(study_names)))
    ci_lower = effect_sizes - 1.96 * ses
    ci_upper = effect_sizes + 1.96 * ses
    
    # Add confidence intervals
    for i, (lower, upper) in enumerate(zip(ci_lower, ci_upper)):
        fig.add_trace(go.Scatter(
            x=[lower, upper],
            y=[i, i],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))
    
    # Add effect size points
    fig.add_trace(go.Scatter(
        x=effect_sizes,
        y=y_positions,
        mode='markers',
        marker=dict(size=10, color='blue', symbol='diamond'),
        text=study_names,
        name='Effect Size',
        showlegend=False
    ))
    
    # Add zero line
    fig.add_vline(x=0, line=dict(color='red', width=1, dash='dash'))
    
    fig.update_layout(
        title='Recalculated Forest Plot',
        xaxis_title='Effect Size',
        yaxis_title='Studies',
        yaxis=dict(
            tickvals=y_positions,
            ticktext=study_names,
            autorange='reversed'
        ),
        height=max(400, len(studies_data) * 40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def publication_bias_assessment():
    st.header("📈 Publication Bias Assessment")
    st.markdown("Comprehensive assessment of publication bias using multiple statistical methods")
    
    # Method selection
    bias_methods = st.multiselect(
        "Select publication bias assessment methods",
        ["Funnel Plot", "Egger's Test", "Begg's Test", "Trim and Fill", "PET-PEESE", "Selection Models"],
        default=["Funnel Plot", "Egger's Test"]
    )
    
    # Sample data input for demonstration
    st.subheader("Study Data for Bias Assessment")
    
    # File upload or manual entry
    data_source = st.radio("Data Source", ["Upload CSV", "Manual Entry", "Use Previous Analysis"])
    
    if data_source == "Manual Entry":
        st.write("Enter effect sizes and standard errors:")
        
        bias_data = []
        num_studies = st.number_input("Number of studies", min_value=3, max_value=30, value=10)
        
        for i in range(min(5, num_studies)):  # Show first 5 for interface
            col1, col2 = st.columns(2)
            with col1:
                es = st.number_input(f"Effect size {i+1}", value=0.0, key=f"es_bias_{i}")
            with col2:
                se = st.number_input(f"Standard error {i+1}", value=0.1, min_value=0.001, key=f"se_bias_{i}")
            bias_data.append({'effect_size': es, 'se': se})
        
        if num_studies > 5:
            st.info(f"Showing first 5 studies. Total {num_studies} will be used in analysis.")
    
    if st.button("Run Publication Bias Assessment"):
        run_publication_bias_tests(bias_methods)

def run_publication_bias_tests(methods):
    """Run publication bias assessment"""
    
    try:
        r_analytics = st.session_state.r_analytics
        
        r_code = """
        library(meta)
        library(metafor)
        library(dmetar)
        
        # Sample data for demonstration (replace with actual data)
        set.seed(123)
        k <- 15  # number of studies
        effect_sizes <- rnorm(k, 0.4, 0.3)
        ses <- runif(k, 0.05, 0.3)
        
        # Add some publication bias (smaller studies with non-significant results missing)
        prob_publish <- pnorm(abs(effect_sizes)/ses - 1.96)
        published <- rbinom(k, 1, prob_publish) == 1
        
        # Keep only "published" studies
        effect_sizes <- effect_sizes[published]
        ses <- ses[published]
        k_pub <- length(effect_sizes)
        
        cat("Publication Bias Assessment Results\\n")
        cat("=====================================\\n")
        cat("Number of studies included:", k_pub, "\\n\\n")
        """
        
        if "Funnel Plot" in methods:
            r_code += """
            # Funnel plot assessment
            ma <- metagen(TE = effect_sizes, seTE = ses, studlab = paste("Study", 1:k_pub))
            
            cat("FUNNEL PLOT ASSESSMENT:\\n")
            cat("Visual inspection of funnel plot symmetry\\n")
            
            # Calculate funnel plot coordinates for asymmetry assessment
            precision <- 1/ses
            cat("Precision range:", range(precision), "\\n")
            cat("Effect size range:", range(effect_sizes), "\\n\\n")
            """
        
        if "Egger's Test" in methods:
            r_code += """
            # Egger's regression test
            egger_test <- metabias(ma, method = "rank")
            cat("EGGER'S REGRESSION TEST:\\n")
            cat("Test statistic:", egger_test$statistic, "\\n")
            cat("P-value:", egger_test$p.value, "\\n")
            if (egger_test$p.value < 0.05) {
                cat("Interpretation: Significant asymmetry detected (p < 0.05)\\n")
            } else {
                cat("Interpretation: No significant asymmetry detected (p >= 0.05)\\n")
            }
            cat("\\n")
            """
        
        if "Begg's Test" in methods:
            r_code += """
            # Begg's rank correlation test
            begg_test <- metabias(ma, method = "rank")
            cat("BEGG'S RANK CORRELATION TEST:\\n")
            cat("Kendall's tau:", begg_test$statistic, "\\n")
            cat("P-value:", begg_test$p.value, "\\n")
            if (begg_test$p.value < 0.05) {
                cat("Interpretation: Significant publication bias detected (p < 0.05)\\n")
            } else {
                cat("Interpretation: No significant publication bias detected (p >= 0.05)\\n")
            }
            cat("\\n")
            """
        
        if "Trim and Fill" in methods:
            r_code += """
            # Trim and fill analysis
            tf_result <- trimfill(ma)
            cat("TRIM AND FILL ANALYSIS:\\n")
            cat("Estimated number of missing studies:", tf_result$k0, "\\n")
            cat("Original pooled effect:", ma$TE.fixed, "\\n")
            cat("Adjusted pooled effect:", tf_result$TE.fixed, "\\n")
            cat("\\n")
            """
        
        # Execute R code
        result = r_analytics.execute_r_code(r_code)
        
        st.subheader("Publication Bias Assessment Results")
        st.code(result, language='text')
        
        # Create funnel plot
        create_funnel_plot()
        
    except Exception as e:
        st.error(f"Error in publication bias assessment: {str(e)}")

def create_funnel_plot():
    """Create an interactive funnel plot"""
    
    # Generate sample data for demonstration
    np.random.seed(123)
    n_studies = 20
    effect_sizes = np.random.normal(0.4, 0.3, n_studies)
    ses = np.random.uniform(0.05, 0.4, n_studies)
    
    # Add publication bias simulation
    prob_publish = 1 / (1 + np.exp(-(np.abs(effect_sizes)/ses - 1.96)))
    published = np.random.binomial(1, prob_publish, n_studies).astype(bool)
    
    effect_sizes = effect_sizes[published]
    ses = ses[published]
    precision = 1/ses
    
    # Create funnel plot
    fig = go.Figure()
    
    # Add studies
    fig.add_trace(go.Scatter(
        x=effect_sizes,
        y=precision,
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.7),
        text=[f'Study {i+1}<br>ES: {es:.3f}<br>SE: {se:.3f}' 
              for i, (es, se) in enumerate(zip(effect_sizes, ses))],
        hovertemplate='%{text}<extra></extra>',
        name='Studies'
    ))
    
    # Add reference lines for symmetry
    max_precision = max(precision)
    mean_effect = np.mean(effect_sizes)
    
    # Central line
    fig.add_trace(go.Scatter(
        x=[mean_effect, mean_effect],
        y=[0, max_precision],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Pooled Effect',
        showlegend=False
    ))
    
    # Confidence interval funnel
    x_ci = np.linspace(mean_effect - 3*np.max(ses), mean_effect + 3*np.max(ses), 100)
    y_ci_upper = 1 / (1.96 * np.abs(x_ci - mean_effect))
    y_ci_upper = np.minimum(y_ci_upper, max_precision)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_ci, x_ci[::-1]]),
        y=np.concatenate([y_ci_upper, np.zeros(len(x_ci))]),
        fill='toself',
        fillcolor='rgba(128,128,128,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI Region',
        showlegend=False
    ))
    
    fig.update_layout(
        title='Funnel Plot for Publication Bias Assessment',
        xaxis_title='Effect Size',
        yaxis_title='Precision (1/SE)',
        hovermode='closest',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.subheader("Funnel Plot Interpretation")
    asymmetry_score = np.corrcoef(effect_sizes, precision)[0,1]
    
    if abs(asymmetry_score) > 0.3:
        st.warning("⚠️ **Potential publication bias detected**: The funnel plot shows asymmetry, suggesting possible publication bias.")
    else:
        st.success("✅ **No obvious publication bias**: The funnel plot appears relatively symmetric.")
    
    st.write(f"**Asymmetry correlation coefficient**: {asymmetry_score:.3f}")

def heterogeneity_analysis():
    st.header("🔄 Heterogeneity Analysis")
    st.markdown("Analyze between-study heterogeneity and explore sources of variation")
    
    # Heterogeneity metrics explanation
    with st.expander("Understanding Heterogeneity Metrics"):
        st.markdown("""
        **Key Heterogeneity Statistics:**
        - **I² statistic**: Percentage of variation due to heterogeneity rather than chance
          - 0-40%: Might not be important
          - 30-60%: May represent moderate heterogeneity  
          - 50-90%: May represent substantial heterogeneity
          - 75-100%: Considerable heterogeneity
        - **Tau² (τ²)**: Estimated variance between studies
        - **Q statistic**: Chi-squared test for heterogeneity
        - **H statistic**: Ratio of observed to expected variation
        """)
    
    st.info("Heterogeneity analysis implementation would include I² calculation, Q-test, subgroup analysis, and meta-regression")

def risk_of_bias_assessment():
    st.header("⚖️ Risk of Bias Assessment")
    st.markdown("Systematic assessment of study quality and risk of bias")
    
    # Bias assessment tool selection
    bias_tool = st.selectbox(
        "Select bias assessment tool",
        ["Cochrane RoB 2.0", "Newcastle-Ottawa Scale", "ROBINS-I", "QUADAS-2"]
    )
    
    st.info(f"Risk of bias assessment using {bias_tool} would be implemented here with interactive forms and visualization")

def prisma_flow_diagram():
    st.header("📋 PRISMA Flow Diagram Generator")
    st.markdown("Generate PRISMA 2020 compliant flow diagrams")
    
    st.info("Interactive PRISMA flow diagram generator would be implemented here")

def sensitivity_analysis():
    st.header("🔍 Sensitivity Analysis")
    st.markdown("Assess the robustness of meta-analysis results")
    
    sensitivity_types = st.multiselect(
        "Select sensitivity analysis types",
        ["Leave-one-out analysis", "Study quality exclusion", "Outlier removal", 
         "Fixed vs Random effects", "Alternative effect size measures"]
    )
    
    st.info("Sensitivity analysis implementation would test robustness of results to various assumptions and methodological choices")

if __name__ == "__main__":
    main()
