import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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

st.set_page_config(page_title="Statistical Analysis", page_icon="📈", layout="wide")

def main():
    st.title("📈 R Statistical Analysis Environment")
    st.markdown("Comprehensive statistical validation and analysis using integrated R computing")
    
    # Initialize R analytics
    if 'r_analytics' not in st.session_state:
        if R_AVAILABLE:
            st.session_state.r_analytics = RAnalytics()
        else:
            st.session_state.r_analytics = None
    
    r_analytics = st.session_state.r_analytics
    
    # Sidebar for analysis options
    with st.sidebar:
        st.header("Analysis Options")
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Descriptive Statistics", "Meta-Analysis", "Effect Size Calculation", 
             "Publication Bias Assessment", "Heterogeneity Analysis", "Custom R Code"]
        )
        
        st.subheader("R Environment Status")
        if st.button("Check R Installation"):
            if R_AVAILABLE and r_analytics:
                status = r_analytics.check_r_status()
                if status:
                    st.success("R environment is ready")
                else:
                    st.error("R environment not available")
            else:
                st.warning("R integration not available - using Python-based statistical analysis")
    
    # Main content area
    if analysis_type == "Descriptive Statistics":
        descriptive_statistics_analysis()
    elif analysis_type == "Meta-Analysis":
        meta_analysis_section()
    elif analysis_type == "Effect Size Calculation":
        effect_size_calculation()
    elif analysis_type == "Publication Bias Assessment":
        publication_bias_assessment()
    elif analysis_type == "Heterogeneity Analysis":
        heterogeneity_analysis()
    elif analysis_type == "Custom R Code":
        custom_r_code_section()

def descriptive_statistics_analysis():
    st.header("Descriptive Statistics Analysis")
    
    # Data input options
    data_source = st.radio("Data Source", ["Upload File", "Manual Entry", "Use Session Data"])
    
    df = None
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    elif data_source == "Manual Entry":
        st.subheader("Enter Data Manually")
        col1, col2 = st.columns(2)
        with col1:
            data_text = st.text_area("Enter comma-separated values (one row per line)", 
                                   placeholder="1,2,3\n4,5,6\n7,8,9")
        with col2:
            headers = st.text_input("Column headers (comma-separated)", placeholder="A,B,C")
        
        if st.button("Process Manual Data"):
            try:
                rows = [row.split(',') for row in data_text.strip().split('\n') if row.strip()]
                if headers and rows:
                    header_list = [h.strip() for h in headers.split(',')]
                    df = pd.DataFrame(rows, columns=header_list)
                elif rows:
                    df = pd.DataFrame(rows)
                if df is not None:
                    df = df.apply(pd.to_numeric, errors='ignore')
            except Exception as e:
                st.error(f"Error processing manual data: {str(e)}")
    
    elif data_source == "Use Session Data" and st.session_state.uploaded_data:
        try:
            df = pd.read_csv(st.session_state.uploaded_data) if st.session_state.uploaded_data.name.endswith('.csv') else pd.read_excel(st.session_state.uploaded_data)
        except:
            st.warning("No valid session data available")
    
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Select numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect("Select columns for analysis", numeric_cols, default=numeric_cols[:3])
            
            if selected_cols:
                # R-based descriptive statistics
                st.subheader("R Statistical Summary")
                r_code = f"""
                library(dplyr)
                data <- data.frame({', '.join([f'{col}=c({",".join(map(str, df[col].dropna().tolist()))})' for col in selected_cols])})
                summary_stats <- data %>% 
                  summarise_all(list(
                    mean = ~mean(., na.rm=TRUE),
                    median = ~median(., na.rm=TRUE),
                    sd = ~sd(., na.rm=TRUE),
                    min = ~min(., na.rm=TRUE),
                    max = ~max(., na.rm=TRUE),
                    n = ~sum(!is.na(.))
                  ))
                print(summary_stats)
                """
                
                if st.button("Run R Analysis"):
                    try:
                        r_analytics = st.session_state.r_analytics
                        result = r_analytics.execute_r_code(r_code)
                        st.code(result, language='r')
                    except Exception as e:
                        st.error(f"R execution error: {str(e)}")
                
                # Python-based visualizations
                st.subheader("Data Visualizations")
                viz_type = st.selectbox("Visualization Type", 
                                      ["Histograms", "Box Plots", "Correlation Matrix", "Scatter Plot"])
                
                if viz_type == "Histograms":
                    fig, axes = plt.subplots(len(selected_cols), 1, figsize=(10, 4*len(selected_cols)))
                    if len(selected_cols) == 1:
                        axes = [axes]
                    for i, col in enumerate(selected_cols):
                        axes[i].hist(df[col].dropna(), bins=30, alpha=0.7)
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                elif viz_type == "Box Plots":
                    fig = px.box(df[selected_cols])
                    fig.update_layout(title="Box Plots of Selected Variables")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Correlation Matrix":
                    corr_matrix = df[selected_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                  title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Scatter Plot" and len(selected_cols) >= 2:
                    x_col = st.selectbox("X-axis", selected_cols)
                    y_col = st.selectbox("Y-axis", [col for col in selected_cols if col != x_col])
                    fig = px.scatter(df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
                    st.plotly_chart(fig, use_container_width=True)

def meta_analysis_section():
    st.header("Meta-Analysis Validation")
    st.markdown("Verify and recalculate meta-analysis results using R's meta and metafor packages")
    
    # Input method selection
    input_method = st.radio("Data Input Method", 
                           ["Manual Entry", "Upload Effect Size Data", "Extract from Text"])
    
    if input_method == "Manual Entry":
        st.subheader("Enter Study Data")
        
        # Study data entry
        num_studies = st.number_input("Number of studies", min_value=2, max_value=50, value=5)
        
        study_data = []
        for i in range(num_studies):
            with st.expander(f"Study {i+1}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    study_name = st.text_input(f"Study name", value=f"Study_{i+1}", key=f"name_{i}")
                with col2:
                    effect_size = st.number_input(f"Effect size", value=0.0, key=f"es_{i}")
                with col3:
                    se = st.number_input(f"Standard error", value=0.1, min_value=0.001, key=f"se_{i}")
                with col4:
                    sample_size = st.number_input(f"Sample size", value=100, min_value=1, key=f"n_{i}")
                
                study_data.append({
                    'study': study_name,
                    'effect_size': effect_size,
                    'se': se,
                    'n': sample_size
                })
        
        if st.button("Run Meta-Analysis"):
            run_meta_analysis(study_data)

def run_meta_analysis(study_data):
    """Execute meta-analysis using R"""
    try:
        r_analytics = st.session_state.r_analytics
        
        # Prepare data for R
        studies = [s['study'] for s in study_data]
        effect_sizes = [s['effect_size'] for s in study_data]
        ses = [s['se'] for s in study_data]
        sample_sizes = [s['n'] for s in study_data]
        
        r_code = f"""
        library(meta)
        library(metafor)
        
        # Create meta-analysis data
        studies <- c({', '.join([f'"{s}"' for s in studies])})
        effect_sizes <- c({', '.join(map(str, effect_sizes))})
        ses <- c({', '.join(map(str, ses))})
        sample_sizes <- c({', '.join(map(str, sample_sizes))})
        
        # Fixed-effect meta-analysis
        ma_fixed <- metagen(TE = effect_sizes, seTE = ses, studlab = studies, 
                           fixed = TRUE, random = FALSE)
        
        # Random-effects meta-analysis
        ma_random <- metagen(TE = effect_sizes, seTE = ses, studlab = studies, 
                            fixed = FALSE, random = TRUE)
        
        # Results summary
        cat("FIXED-EFFECT META-ANALYSIS:\\n")
        print(summary(ma_fixed))
        cat("\\n\\nRANDOM-EFFECTS META-ANALYSIS:\\n")
        print(summary(ma_random))
        
        # Heterogeneity statistics
        cat("\\n\\nHETEROGENEITY STATISTICS:\\n")
        cat("I-squared:", ma_random$I2, "%\\n")
        cat("Tau-squared:", ma_random$tau2, "\\n")
        cat("Q-statistic:", ma_random$Q, "\\n")
        cat("P-value for heterogeneity:", ma_random$pval.Q, "\\n")
        """
        
        result = r_analytics.execute_r_code(r_code)
        
        st.subheader("Meta-Analysis Results")
        st.code(result, language='text')
        
        # Create forest plot data for Python visualization
        df_studies = pd.DataFrame(study_data)
        
        # Calculate confidence intervals
        df_studies['ci_lower'] = df_studies['effect_size'] - 1.96 * df_studies['se']
        df_studies['ci_upper'] = df_studies['effect_size'] + 1.96 * df_studies['se']
        
        # Create forest plot
        fig = create_forest_plot(df_studies)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Meta-analysis error: {str(e)}")

def create_forest_plot(df):
    """Create a forest plot using Plotly"""
    fig = go.Figure()
    
    y_pos = list(range(len(df)))
    
    # Add confidence intervals
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['ci_lower'], row['ci_upper']],
            y=[i, i],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))
    
    # Add effect size points
    fig.add_trace(go.Scatter(
        x=df['effect_size'],
        y=y_pos,
        mode='markers',
        marker=dict(
            size=df['n']/10,  # Size proportional to sample size
            color='blue',
            symbol='diamond'
        ),
        text=df['study'],
        textposition='middle right',
        name='Effect Size',
        showlegend=False
    ))
    
    # Add vertical line at zero
    fig.add_vline(x=0, line=dict(color='red', width=1, dash='dash'))
    
    fig.update_layout(
        title='Forest Plot - Effect Sizes with 95% Confidence Intervals',
        xaxis_title='Effect Size',
        yaxis_title='Studies',
        yaxis=dict(
            tickvals=y_pos,
            ticktext=df['study'],
            autorange='reversed'
        ),
        height=max(400, len(df) * 50)
    )
    
    return fig

def effect_size_calculation():
    st.header("Effect Size Calculation")
    st.markdown("Calculate various effect size measures using R")
    
    effect_type = st.selectbox(
        "Effect Size Type",
        ["Cohen's d", "Hedges' g", "Odds Ratio", "Risk Ratio", "Correlation"]
    )
    
    if effect_type in ["Cohen's d", "Hedges' g"]:
        st.subheader(f"Calculate {effect_type}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Group 1 (Control)**")
            n1 = st.number_input("Sample size", value=30, key="n1")
            mean1 = st.number_input("Mean", value=10.0, key="mean1")
            sd1 = st.number_input("Standard deviation", value=2.0, key="sd1")
        
        with col2:
            st.write("**Group 2 (Treatment)**")
            n2 = st.number_input("Sample size", value=30, key="n2")
            mean2 = st.number_input("Mean", value=12.0, key="mean2")
            sd2 = st.number_input("Standard deviation", value=2.5, key="sd2")
        
        if st.button(f"Calculate {effect_type}"):
            try:
                r_analytics = st.session_state.r_analytics
                
                r_code = f"""
                library(effsize)
                library(compute.es)
                
                # Calculate effect size
                pooled_sd <- sqrt(((n1-1)*sd1^2 + (n2-1)*sd2^2) / (n1+n2-2))
                cohens_d <- (mean2 - mean1) / pooled_sd
                
                # Hedges' g (bias-corrected)
                hedges_g <- cohens_d * (1 - 3/(4*(n1+n2)-9))
                
                # Standard error
                se_d <- sqrt((n1+n2)/(n1*n2) + cohens_d^2/(2*(n1+n2)))
                
                # Confidence intervals
                ci_lower <- cohens_d - 1.96 * se_d
                ci_upper <- cohens_d + 1.96 * se_d
                
                cat("Effect Size Calculations:\\n")
                cat("Cohen's d:", cohens_d, "\\n")
                cat("Hedges' g:", hedges_g, "\\n")
                cat("Standard Error:", se_d, "\\n")
                cat("95% CI: [", ci_lower, ",", ci_upper, "]\\n")
                
                # Effect size interpretation
                if (abs(cohens_d) < 0.2) {{
                  interpretation <- "Small effect"
                }} else if (abs(cohens_d) < 0.5) {{
                  interpretation <- "Small to medium effect"
                }} else if (abs(cohens_d) < 0.8) {{
                  interpretation <- "Medium to large effect"
                }} else {{
                  interpretation <- "Large effect"
                }}
                
                cat("Interpretation:", interpretation, "\\n")
                """
                
                r_code = r_code.replace('n1', str(n1)).replace('n2', str(n2))
                r_code = r_code.replace('mean1', str(mean1)).replace('mean2', str(mean2))
                r_code = r_code.replace('sd1', str(sd1)).replace('sd2', str(sd2))
                
                result = r_analytics.execute_r_code(r_code)
                st.code(result, language='text')
                
            except Exception as e:
                st.error(f"Effect size calculation error: {str(e)}")

def publication_bias_assessment():
    st.header("Publication Bias Assessment")
    st.markdown("Assess publication bias using funnel plots and statistical tests")
    
    # This would implement publication bias tests
    st.info("This section would implement Egger's test, Begg's test, and funnel plot analysis using R")

def heterogeneity_analysis():
    st.header("Heterogeneity Analysis")
    st.markdown("Analyze between-study heterogeneity in meta-analyses")
    
    # This would implement heterogeneity analysis
    st.info("This section would implement I² statistics, Q-test, and subgroup analysis using R")

def custom_r_code_section():
    st.header("Custom R Code Execution")
    st.markdown("Execute custom R statistical code")
    
    # R code editor
    r_code = st.text_area(
        "Enter R code:",
        value="""# Example: Basic statistical analysis
data <- c(1, 2, 3, 4, 5)
mean(data)
sd(data)
summary(data)""",
        height=200
    )
    
    if st.button("Execute R Code"):
        try:
            r_analytics = st.session_state.r_analytics
            result = r_analytics.execute_r_code(r_code)
            st.subheader("R Output:")
            st.code(result, language='text')
        except Exception as e:
            st.error(f"R execution error: {str(e)}")
    
    # R help section
    with st.expander("R Help & Examples"):
        st.markdown("""
        **Common R packages for meta-analysis:**
        - `meta`: General meta-analysis functions
        - `metafor`: Advanced meta-analysis modeling
        - `dmetar`: Diagnostic meta-analysis
        - `robvis`: Risk of bias visualization
        
        **Example codes:**
        ```r
        # Load required libraries
        library(meta)
        library(metafor)
        
        # Basic meta-analysis
        ma <- metagen(TE=effect, seTE=se, studlab=study)
        summary(ma)
        forest(ma)
        ```
        """)