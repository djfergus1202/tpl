import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from utils.auto_validator import display_validation_results

st.set_page_config(page_title="Validation Results", page_icon="📊", layout="wide")

def main():
    st.title("📊 Validation Results Dashboard")
    st.markdown("View and analyze automatic validation results from uploaded research papers")
    
    # Check if there are any validation results in session
    validation_keys = [key for key in st.session_state.keys() if key.startswith('pdf_validation_')]
    
    if not validation_keys:
        st.info("🔍 No validation results found. Upload and process PDF papers in the Data Import module to see validation results here.")
        
        # Quick access to Data Import
        if st.button("📁 Go to Data Import", type="primary"):
            st.switch_page("pages/4_Data_Import.py")
        return
    
    # Sidebar for result selection
    with st.sidebar:
        st.header("📋 Validation Sessions")
        
        # Get all available validation results
        available_results = {}
        for key in validation_keys:
            file_index = key.split('_')[-1]
            pdf_text_key = f'pdf_text_{file_index}'
            
            # Try to get the original filename
            filename = f"Paper {file_index}"
            if pdf_text_key in st.session_state:
                # Try to extract title from text
                text_content = st.session_state[pdf_text_key]
                lines = text_content.split('\n')[:10]
                for line in lines:
                    if len(line.strip()) > 10 and line.strip().istitle():
                        filename = line.strip()[:50] + "..." if len(line.strip()) > 50 else line.strip()
                        break
            
            available_results[key] = {
                'filename': filename,
                'timestamp': st.session_state[key].get('timestamp', 'Unknown'),
                'file_index': file_index
            }
        
        # Result selection
        selected_result_key = st.selectbox(
            "Select validation result",
            options=list(available_results.keys()),
            format_func=lambda x: f"{available_results[x]['filename']}"
        )
        
        # Result info
        if selected_result_key:
            result_info = available_results[selected_result_key]
            st.write(f"**File:** {result_info['filename']}")
            st.write(f"**Validated:** {result_info['timestamp'][:19] if result_info['timestamp'] != 'Unknown' else 'Unknown'}")
        
        # Action buttons
        st.subheader("🛠️ Actions")
        if st.button("🔄 Refresh Results"):
            st.rerun()
        
        if st.button("📤 Export Results"):
            export_validation_results(selected_result_key)
        
        if st.button("🗑️ Clear All Results"):
            clear_all_validation_results()
    
    # Main content
    if selected_result_key and selected_result_key in st.session_state:
        validation_result = st.session_state[selected_result_key]
        
        # Display the validation results
        display_validation_results(validation_result)
        
        # Additional analysis tabs
        st.header("📈 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Score Breakdown", "📋 Test Summary", "💡 Recommendations", "📄 Raw Data"])
        
        with tab1:
            display_score_breakdown(validation_result)
        
        with tab2:
            display_test_summary(validation_result)
        
        with tab3:
            display_recommendations_analysis(validation_result)
        
        with tab4:
            display_raw_data(validation_result)

def display_score_breakdown(validation_result):
    """Display detailed score breakdown with visualizations"""
    
    st.subheader("📊 Validation Score Breakdown")
    
    # Extract scores from validation results
    scores = {}
    test_results = validation_result.get('validation_results', {})
    
    for test_name, result in test_results.items():
        if result.get('status') == 'completed':
            # Extract different types of scores
            if 'overall_quality_score' in result:
                scores[f"{test_name}_quality"] = result['overall_quality_score']
            if 'reproducibility_score' in result:
                scores[f"{test_name}_reproducibility"] = result['reproducibility_score']
            if 'methodology_score' in result:
                scores[f"{test_name}_methodology"] = result['methodology_score']
            if 'readability_score' in result:
                scores[f"{test_name}_readability"] = result['readability_score']
            if 'academic_tone_score' in result:
                scores[f"{test_name}_academic_tone"] = result['academic_tone_score']
            if 'structure_score' in result:
                scores[f"{test_name}_structure"] = result['structure_score']
            if 'completeness_score' in result:
                scores[f"{test_name}_completeness"] = result['completeness_score']
    
    if scores:
        # Create radar chart for scores
        fig = go.Figure()
        
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Validation Scores',
            line_color='rgb(34, 139, 34)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Validation Score Radar Chart",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Score table
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Score Details")
            score_df = pd.DataFrame(list(scores.items()), columns=['Metric', 'Score'])
            score_df['Score'] = score_df['Score'].round(1)
            score_df = score_df.sort_values('Score', ascending=False)
            st.dataframe(score_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Score Distribution")
            fig_hist = px.histogram(
                x=list(scores.values()),
                nbins=10,
                title="Score Distribution",
                labels={'x': 'Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    else:
        st.info("No quantitative scores found in validation results")

def display_test_summary(validation_result):
    """Display summary of all validation tests"""
    
    st.subheader("📋 Validation Test Summary")
    
    test_results = validation_result.get('validation_results', {})
    summary_data = []
    
    for test_name, result in test_results.items():
        status = result.get('status', 'unknown')
        test_display_name = test_name.replace('_', ' ').title()
        
        # Count key findings
        key_findings = 0
        if status == 'completed':
            for key, value in result.items():
                if isinstance(value, (int, float)) and value > 0:
                    key_findings += 1
                elif isinstance(value, list) and value:
                    key_findings += len(value)
                elif isinstance(value, bool) and value:
                    key_findings += 1
        
        summary_data.append({
            'Test': test_display_name,
            'Status': '✅ Passed' if status == 'completed' else '❌ Failed',
            'Key Findings': key_findings,
            'Recommendations': len(result.get('recommendations', []))
        })
    
    # Display summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            passed_tests = len([d for d in summary_data if 'Passed' in d['Status']])
            st.metric("Tests Passed", f"{passed_tests}/{len(summary_data)}")
        
        with col2:
            total_findings = sum(d['Key Findings'] for d in summary_data)
            st.metric("Total Findings", total_findings)
        
        with col3:
            total_recommendations = sum(d['Recommendations'] for d in summary_data)
            st.metric("Total Recommendations", total_recommendations)
        
        with col4:
            success_rate = (passed_tests / len(summary_data)) * 100 if summary_data else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

def display_recommendations_analysis(validation_result):
    """Display and analyze all recommendations"""
    
    st.subheader("💡 Recommendations Analysis")
    
    all_recommendations = []
    test_results = validation_result.get('validation_results', {})
    
    # Collect all recommendations
    for test_name, result in test_results.items():
        recommendations = result.get('recommendations', [])
        for rec in recommendations:
            all_recommendations.append({
                'Test': test_name.replace('_', ' ').title(),
                'Recommendation': rec,
                'Priority': categorize_recommendation_priority(rec)
            })
    
    if all_recommendations:
        rec_df = pd.DataFrame(all_recommendations)
        
        # Priority breakdown
        col1, col2 = st.columns(2)
        with col1:
            priority_counts = rec_df['Priority'].value_counts()
            fig_pie = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Recommendations by Priority",
                color_discrete_map={
                    'High': 'red',
                    'Medium': 'orange',
                    'Low': 'green'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            test_counts = rec_df['Test'].value_counts()
            fig_bar = px.bar(
                x=test_counts.index,
                y=test_counts.values,
                title="Recommendations by Test",
                labels={'x': 'Test', 'y': 'Count'}
            )
            fig_bar.update_xaxis(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed recommendations
        st.subheader("📝 Detailed Recommendations")
        
        # Filter by priority
        priority_filter = st.selectbox("Filter by Priority", ["All", "High", "Medium", "Low"])
        
        filtered_df = rec_df if priority_filter == "All" else rec_df[rec_df['Priority'] == priority_filter]
        
        for _, row in filtered_df.iterrows():
            priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[row['Priority']]
            st.write(f"{priority_color} **{row['Test']}**: {row['Recommendation']}")
    
    else:
        st.info("No recommendations generated from validation tests")

def display_raw_data(validation_result):
    """Display raw validation data in JSON format"""
    
    st.subheader("📄 Raw Validation Data")
    st.markdown("Complete validation results in JSON format for detailed analysis")
    
    # Option to download results
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy to Clipboard"):
            st.code(json.dumps(validation_result, indent=2, default=str))
    
    with col2:
        json_data = json.dumps(validation_result, indent=2, default=str)
        st.download_button(
            label="💾 Download JSON",
            data=json_data,
            file_name=f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Expandable JSON viewer
    with st.expander("🔍 View Raw Data", expanded=False):
        st.json(validation_result)

def categorize_recommendation_priority(recommendation):
    """Categorize recommendation priority based on keywords"""
    
    rec_lower = recommendation.lower()
    
    # High priority keywords
    high_priority_keywords = [
        'critical', 'invalid', 'error', 'missing', 'fix', 'incorrect',
        'failed', 'broken', 'remove', 'delete', 'urgent'
    ]
    
    # Medium priority keywords  
    medium_priority_keywords = [
        'consider', 'improve', 'enhance', 'add', 'include', 'update',
        'modify', 'adjust', 'review'
    ]
    
    if any(keyword in rec_lower for keyword in high_priority_keywords):
        return "High"
    elif any(keyword in rec_lower for keyword in medium_priority_keywords):
        return "Medium"
    else:
        return "Low"

def export_validation_results(result_key):
    """Export validation results to different formats"""
    
    if result_key in st.session_state:
        validation_result = st.session_state[result_key]
        
        # Create exportable data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'validation_data': validation_result
        }
        
        # JSON export
        json_data = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="📄 Download JSON Report",
            data=json_data,
            file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Export ready! Click the download button above.")

def clear_all_validation_results():
    """Clear all validation results from session state"""
    
    validation_keys = [key for key in st.session_state.keys() if key.startswith('pdf_validation_')]
    
    if validation_keys:
        for key in validation_keys:
            del st.session_state[key]
        
        st.success(f"✅ Cleared {len(validation_keys)} validation results")
        st.rerun()
    else:
        st.info("No validation results to clear")

if __name__ == "__main__":
    main()