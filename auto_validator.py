"""
Automatic Validation System for Research Papers
Runs comprehensive validation tests when papers are uploaded
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

from .pdf_processor import PDFProcessor
from .citation_validator import CitationValidator
from .r_integration import RAnalytics
from .statistical_tools import StatisticalAnalyzer
from .nlp_processor import NLPProcessor

class AutoValidator:
    """Automatic validation system for uploaded research papers"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.citation_validator = CitationValidator()
        self.r_analytics = RAnalytics()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.nlp_processor = NLPProcessor()
        
        # Validation test registry
        self.validation_tests = {
            'citation_analysis': self.validate_citations,
            'statistical_analysis': self.validate_statistics,
            'meta_analysis_check': self.validate_meta_analysis,
            'data_extraction': self.extract_research_data,
            'quality_assessment': self.assess_paper_quality,
            'reproducibility_check': self.check_reproducibility,
            'methodology_validation': self.validate_methodology
        }
        
    def run_automatic_validation(self, uploaded_file, validation_options: Dict[str, bool] = None) -> Dict[str, Any]:
        """
        Run comprehensive automatic validation on uploaded paper
        
        Args:
            uploaded_file: Streamlit uploaded file object
            validation_options: Dict of validation tests to run
        
        Returns:
            Dict containing all validation results
        """
        
        if validation_options is None:
            validation_options = {test: True for test in self.validation_tests.keys()}
        
        results = {
            'paper_info': {},
            'validation_results': {},
            'extracted_data': {},
            'recommendations': [],
            'validation_summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Extract text from PDF
        st.info("🔍 Extracting text from PDF...")
        try:
            text_content = self.pdf_processor.extract_text(uploaded_file)
            results['paper_info']['text_extracted'] = True
            results['paper_info']['text_length'] = len(text_content)
        except Exception as e:
            st.error(f"Failed to extract text: {str(e)}")
            results['paper_info']['text_extracted'] = False
            return results
        
        # Step 2: Extract basic paper information
        st.info("📋 Analyzing paper structure...")
        paper_info = self._extract_paper_info(text_content)
        results['paper_info'].update(paper_info)
        
        # Step 3: Run validation tests
        total_tests = sum(validation_options.values())
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        completed_tests = 0
        
        for test_name, should_run in validation_options.items():
            if should_run and test_name in self.validation_tests:
                status_text.text(f"Running {test_name.replace('_', ' ').title()}...")
                
                try:
                    test_result = self.validation_tests[test_name](text_content, results['paper_info'])
                    results['validation_results'][test_name] = test_result
                    
                    # Add test-specific recommendations
                    if 'recommendations' in test_result:
                        results['recommendations'].extend(test_result['recommendations'])
                        
                except Exception as e:
                    st.warning(f"Test {test_name} failed: {str(e)}")
                    results['validation_results'][test_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                
                completed_tests += 1
                progress_bar.progress(completed_tests / total_tests)
        
        # Step 4: Generate validation summary
        st.info("📊 Generating validation summary...")
        results['validation_summary'] = self._generate_validation_summary(results['validation_results'])
        
        status_text.text("✅ Automatic validation complete!")
        
        return results
    
    def _extract_paper_info(self, text_content: str) -> Dict[str, Any]:
        """Extract basic information about the paper"""
        
        info = {
            'title': '',
            'authors': [],
            'abstract': '',
            'doi': '',
            'publication_year': None,
            'journal': '',
            'sections': [],
            'has_methods': False,
            'has_results': False,
            'has_discussion': False,
            'has_references': False
        }
        
        # Extract title (usually first line or in title case)
        lines = text_content.split('\n')
        for line in lines[:10]:
            if len(line.strip()) > 10 and line.strip().istitle():
                info['title'] = line.strip()
                break
        
        # Extract DOI
        doi_pattern = r'doi[:\s]*([0-9]+\.[0-9]+/[^\s]+)'
        doi_match = re.search(doi_pattern, text_content, re.IGNORECASE)
        if doi_match:
            info['doi'] = doi_match.group(1)
        
        # Extract year
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text_content)
        if years:
            info['publication_year'] = int(years[0])
        
        # Check for standard sections
        text_lower = text_content.lower()
        info['has_methods'] = any(keyword in text_lower for keyword in ['methods', 'methodology', 'materials and methods'])
        info['has_results'] = 'results' in text_lower
        info['has_discussion'] = 'discussion' in text_lower
        info['has_references'] = any(keyword in text_lower for keyword in ['references', 'bibliography'])
        
        return info
    
    def validate_citations(self, text_content: str, paper_info: Dict) -> Dict[str, Any]:
        """Validate citations in the paper"""
        
        result = {
            'status': 'completed',
            'total_citations': 0,
            'valid_citations': 0,
            'invalid_citations': 0,
            'citation_formats': [],
            'duplicate_citations': 0,
            'recommendations': []
        }
        
        try:
            # Extract citations using citation validator
            citations = self.citation_validator.extract_citations(text_content)
            result['total_citations'] = len(citations)
            
            # Validate each citation
            valid_count = 0
            for citation in citations:
                if self.citation_validator.validate_citation_format(citation):
                    valid_count += 1
            
            result['valid_citations'] = valid_count
            result['invalid_citations'] = len(citations) - valid_count
            
            # Check for duplicates
            duplicates = self.citation_validator.find_duplicate_citations(citations)
            result['duplicate_citations'] = len(duplicates)
            
            # Generate recommendations
            if result['invalid_citations'] > 0:
                result['recommendations'].append(f"Fix {result['invalid_citations']} invalid citation formats")
            
            if result['duplicate_citations'] > 0:
                result['recommendations'].append(f"Remove {result['duplicate_citations']} duplicate citations")
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def validate_statistics(self, text_content: str, paper_info: Dict) -> Dict[str, Any]:
        """Validate statistical analyses in the paper"""
        
        result = {
            'status': 'completed',
            'statistical_tests_found': [],
            'p_values_found': [],
            'effect_sizes_found': [],
            'confidence_intervals_found': [],
            'sample_sizes_found': [],
            'recommendations': []
        }
        
        try:
            # Extract statistical information
            p_values = re.findall(r'p\s*[<=>\s]*\s*0?\.\d+', text_content, re.IGNORECASE)
            result['p_values_found'] = p_values[:10]  # Limit output
            
            # Extract sample sizes
            n_values = re.findall(r'n\s*=\s*(\d+)', text_content, re.IGNORECASE)
            result['sample_sizes_found'] = [int(n) for n in n_values if n.isdigit()]
            
            # Extract confidence intervals
            ci_pattern = r'(\d+%?\s*(?:confidence\s*interval|CI)|\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\])'
            confidence_intervals = re.findall(ci_pattern, text_content, re.IGNORECASE)
            result['confidence_intervals_found'] = confidence_intervals[:5]
            
            # Check for common statistical tests
            statistical_tests = [
                't-test', 'anova', 'chi-square', 'regression', 'correlation',
                'mann-whitney', 'wilcoxon', 'kruskal-wallis', 'fisher'
            ]
            
            found_tests = []
            for test in statistical_tests:
                if test.replace('-', '[ -]') in text_content.lower():
                    found_tests.append(test)
            
            result['statistical_tests_found'] = found_tests
            
            # Generate recommendations
            if not result['p_values_found']:
                result['recommendations'].append("No p-values found - consider adding statistical significance tests")
            
            if not result['confidence_intervals_found']:
                result['recommendations'].append("No confidence intervals found - consider adding for effect estimates")
            
            if not result['sample_sizes_found']:
                result['recommendations'].append("No sample sizes clearly reported - ensure power calculations are adequate")
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def validate_meta_analysis(self, text_content: str, paper_info: Dict) -> Dict[str, Any]:
        """Check if paper contains meta-analysis and validate it"""
        
        result = {
            'status': 'completed',
            'is_meta_analysis': False,
            'search_strategy_described': False,
            'inclusion_criteria_described': False,
            'exclusion_criteria_described': False,
            'data_extraction_described': False,
            'quality_assessment_described': False,
            'heterogeneity_assessed': False,
            'funnel_plot_mentioned': False,
            'prisma_compliance': False,
            'recommendations': []
        }
        
        try:
            text_lower = text_content.lower()
            
            # Check if it's a meta-analysis
            meta_keywords = ['meta-analysis', 'meta analysis', 'systematic review', 'pooled analysis']
            result['is_meta_analysis'] = any(keyword in text_lower for keyword in meta_keywords)
            
            if result['is_meta_analysis']:
                # Check for key meta-analysis components
                result['search_strategy_described'] = any(keyword in text_lower for keyword in 
                    ['search strategy', 'database search', 'pubmed', 'embase', 'cochrane'])
                
                result['inclusion_criteria_described'] = 'inclusion criteria' in text_lower
                result['exclusion_criteria_described'] = 'exclusion criteria' in text_lower
                result['data_extraction_described'] = 'data extraction' in text_lower
                result['quality_assessment_described'] = any(keyword in text_lower for keyword in 
                    ['quality assessment', 'risk of bias', 'cochrane risk', 'newcastle-ottawa'])
                
                result['heterogeneity_assessed'] = any(keyword in text_lower for keyword in 
                    ['heterogeneity', 'i-squared', 'i2', 'tau-squared', 'cochran'])
                
                result['funnel_plot_mentioned'] = 'funnel plot' in text_lower
                result['prisma_compliance'] = 'prisma' in text_lower
                
                # Generate recommendations
                missing_components = []
                if not result['search_strategy_described']:
                    missing_components.append("search strategy")
                if not result['inclusion_criteria_described']:
                    missing_components.append("inclusion criteria")
                if not result['exclusion_criteria_described']:
                    missing_components.append("exclusion criteria")
                if not result['data_extraction_described']:
                    missing_components.append("data extraction methodology")
                if not result['quality_assessment_described']:
                    missing_components.append("quality assessment")
                if not result['heterogeneity_assessed']:
                    missing_components.append("heterogeneity assessment")
                
                if missing_components:
                    result['recommendations'].append(f"Consider adding: {', '.join(missing_components)}")
                
                if not result['prisma_compliance']:
                    result['recommendations'].append("Consider following PRISMA guidelines for reporting")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def extract_research_data(self, text_content: str, paper_info: Dict) -> Dict[str, Any]:
        """Extract research data from the paper"""
        
        result = {
            'status': 'completed',
            'tables_detected': 0,
            'figures_detected': 0,
            'extracted_tables': [],
            'extracted_numbers': [],
            'recommendations': []
        }
        
        try:
            # Count table and figure references
            table_refs = re.findall(r'table\s+\d+', text_content, re.IGNORECASE)
            figure_refs = re.findall(r'figure\s+\d+', text_content, re.IGNORECASE)
            
            result['tables_detected'] = len(set(table_refs))
            result['figures_detected'] = len(set(figure_refs))
            
            # Extract numerical data
            numbers = re.findall(r'\b\d+\.?\d*\b', text_content)
            result['extracted_numbers'] = [float(n) for n in numbers[:50] if n.replace('.', '').isdigit()]
            
            # Generate recommendations
            if result['tables_detected'] == 0:
                result['recommendations'].append("No tables detected - consider adding summary tables")
            
            if result['figures_detected'] == 0:
                result['recommendations'].append("No figures detected - consider adding visual representations")
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def assess_paper_quality(self, text_content: str, paper_info: Dict) -> Dict[str, Any]:
        """Assess overall quality of the paper"""
        
        result = {
            'status': 'completed',
            'readability_score': 0,
            'academic_tone_score': 0,
            'structure_score': 0,
            'completeness_score': 0,
            'overall_quality_score': 0,
            'recommendations': []
        }
        
        try:
            # Use NLP processor for quality assessment
            readability = self.nlp_processor.calculate_readability(text_content)
            result['readability_score'] = readability.get('flesch_reading_ease', 0)
            
            # Check academic tone
            academic_words = self.nlp_processor.count_academic_vocabulary(text_content)
            total_words = len(text_content.split())
            result['academic_tone_score'] = min(100, (academic_words / total_words) * 1000) if total_words > 0 else 0
            
            # Assess structure
            structure_components = [
                paper_info.get('has_methods', False),
                paper_info.get('has_results', False),
                paper_info.get('has_discussion', False),
                paper_info.get('has_references', False)
            ]
            result['structure_score'] = (sum(structure_components) / len(structure_components)) * 100
            
            # Assess completeness
            completeness_components = [
                bool(paper_info.get('title')),
                bool(paper_info.get('abstract')),
                bool(paper_info.get('doi')),
                paper_info.get('publication_year') is not None
            ]
            result['completeness_score'] = (sum(completeness_components) / len(completeness_components)) * 100
            
            # Calculate overall score
            scores = [
                result['readability_score'],
                result['academic_tone_score'],
                result['structure_score'],
                result['completeness_score']
            ]
            result['overall_quality_score'] = sum(scores) / len(scores)
            
            # Generate recommendations
            if result['readability_score'] < 30:
                result['recommendations'].append("Text may be too complex - consider simplifying language")
            elif result['readability_score'] > 70:
                result['recommendations'].append("Text may be too simple for academic paper")
            
            if result['academic_tone_score'] < 20:
                result['recommendations'].append("Consider using more academic vocabulary")
            
            if result['structure_score'] < 75:
                result['recommendations'].append("Improve paper structure - ensure all standard sections are present")
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def check_reproducibility(self, text_content: str, paper_info: Dict) -> Dict[str, Any]:
        """Check reproducibility elements"""
        
        result = {
            'status': 'completed',
            'data_availability_mentioned': False,
            'code_availability_mentioned': False,
            'methodology_detailed': False,
            'statistical_software_mentioned': False,
            'funding_disclosed': False,
            'conflicts_disclosed': False,
            'reproducibility_score': 0,
            'recommendations': []
        }
        
        try:
            text_lower = text_content.lower()
            
            # Check reproducibility elements
            result['data_availability_mentioned'] = any(keyword in text_lower for keyword in 
                ['data availability', 'available upon request', 'supplementary data', 'repository'])
            
            result['code_availability_mentioned'] = any(keyword in text_lower for keyword in 
                ['code availability', 'github', 'software available', 'statistical code'])
            
            result['methodology_detailed'] = len(re.findall(r'(step|procedure|protocol|method)', text_lower)) > 5
            
            result['statistical_software_mentioned'] = any(keyword in text_lower for keyword in 
                ['spss', 'r statistical', 'stata', 'sas', 'python', 'matlab'])
            
            result['funding_disclosed'] = any(keyword in text_lower for keyword in 
                ['funding', 'grant', 'supported by', 'financial support'])
            
            result['conflicts_disclosed'] = any(keyword in text_lower for keyword in 
                ['conflict of interest', 'competing interests', 'no conflicts'])
            
            # Calculate reproducibility score
            components = [
                result['data_availability_mentioned'],
                result['code_availability_mentioned'],
                result['methodology_detailed'],
                result['statistical_software_mentioned'],
                result['funding_disclosed'],
                result['conflicts_disclosed']
            ]
            result['reproducibility_score'] = (sum(components) / len(components)) * 100
            
            # Generate recommendations
            missing_elements = []
            if not result['data_availability_mentioned']:
                missing_elements.append("data availability statement")
            if not result['code_availability_mentioned']:
                missing_elements.append("code/software availability")
            if not result['statistical_software_mentioned']:
                missing_elements.append("statistical software details")
            if not result['funding_disclosed']:
                missing_elements.append("funding disclosure")
            if not result['conflicts_disclosed']:
                missing_elements.append("conflict of interest statement")
            
            if missing_elements:
                result['recommendations'].append(f"Add: {', '.join(missing_elements)}")
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def validate_methodology(self, text_content: str, paper_info: Dict) -> Dict[str, Any]:
        """Validate research methodology"""
        
        result = {
            'status': 'completed',
            'study_design_mentioned': False,
            'sample_size_justified': False,
            'statistical_power_mentioned': False,
            'randomization_mentioned': False,
            'blinding_mentioned': False,
            'primary_outcome_defined': False,
            'methodology_score': 0,
            'recommendations': []
        }
        
        try:
            text_lower = text_content.lower()
            
            # Check methodology elements
            result['study_design_mentioned'] = any(keyword in text_lower for keyword in 
                ['study design', 'randomized', 'observational', 'cross-sectional', 'cohort', 'case-control'])
            
            result['sample_size_justified'] = any(keyword in text_lower for keyword in 
                ['sample size', 'power calculation', 'power analysis', 'effect size'])
            
            result['statistical_power_mentioned'] = any(keyword in text_lower for keyword in 
                ['statistical power', 'power = ', 'power >'])
            
            result['randomization_mentioned'] = any(keyword in text_lower for keyword in 
                ['randomization', 'randomized', 'random allocation'])
            
            result['blinding_mentioned'] = any(keyword in text_lower for keyword in 
                ['blinding', 'blinded', 'double-blind', 'single-blind'])
            
            result['primary_outcome_defined'] = any(keyword in text_lower for keyword in 
                ['primary outcome', 'main outcome', 'primary endpoint'])
            
            # Calculate methodology score
            components = [
                result['study_design_mentioned'],
                result['sample_size_justified'],
                result['statistical_power_mentioned'],
                result['randomization_mentioned'],
                result['blinding_mentioned'],
                result['primary_outcome_defined']
            ]
            result['methodology_score'] = (sum(components) / len(components)) * 100
            
            # Generate recommendations
            missing_elements = []
            if not result['study_design_mentioned']:
                missing_elements.append("clear study design description")
            if not result['sample_size_justified']:
                missing_elements.append("sample size justification")
            if not result['primary_outcome_defined']:
                missing_elements.append("primary outcome definition")
            
            if missing_elements:
                result['recommendations'].append(f"Consider adding: {', '.join(missing_elements)}")
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def _generate_validation_summary(self, validation_results: Dict) -> Dict[str, Any]:
        """Generate overall validation summary"""
        
        summary = {
            'total_tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'critical_issues': [],
            'warnings': [],
            'overall_score': 0,
            'quality_grade': 'F'
        }
        
        scores = []
        
        for test_name, result in validation_results.items():
            summary['total_tests_run'] += 1
            
            if result.get('status') == 'completed':
                summary['tests_passed'] += 1
                
                # Extract scores where available
                if 'overall_quality_score' in result:
                    scores.append(result['overall_quality_score'])
                elif 'reproducibility_score' in result:
                    scores.append(result['reproducibility_score'])
                elif 'methodology_score' in result:
                    scores.append(result['methodology_score'])
                
                # Check for critical issues
                if test_name == 'citation_analysis' and result.get('invalid_citations', 0) > 5:
                    summary['critical_issues'].append("High number of invalid citations")
                
                if test_name == 'reproducibility_check' and result.get('reproducibility_score', 0) < 50:
                    summary['critical_issues'].append("Low reproducibility score")
                
            else:
                summary['tests_failed'] += 1
                summary['warnings'].append(f"Test {test_name} failed to complete")
        
        # Calculate overall score
        if scores:
            summary['overall_score'] = sum(scores) / len(scores)
        
        # Assign quality grade
        if summary['overall_score'] >= 90:
            summary['quality_grade'] = 'A'
        elif summary['overall_score'] >= 80:
            summary['quality_grade'] = 'B'
        elif summary['overall_score'] >= 70:
            summary['quality_grade'] = 'C'
        elif summary['overall_score'] >= 60:
            summary['quality_grade'] = 'D'
        else:
            summary['quality_grade'] = 'F'
        
        return summary

def display_validation_results(results: Dict[str, Any]):
    """Display validation results in Streamlit interface"""
    
    st.header("🔍 Automatic Validation Results")
    
    # Display summary
    summary = results.get('validation_summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{summary.get('overall_score', 0):.1f}")
    with col2:
        st.metric("Quality Grade", summary.get('quality_grade', 'N/A'))
    with col3:
        st.metric("Tests Passed", f"{summary.get('tests_passed', 0)}/{summary.get('total_tests_run', 0)}")
    with col4:
        st.metric("Critical Issues", len(summary.get('critical_issues', [])))
    
    # Critical issues
    if summary.get('critical_issues'):
        st.error("⚠️ Critical Issues Found:")
        for issue in summary['critical_issues']:
            st.write(f"• {issue}")
    
    # Validation test results
    if results.get('validation_results'):
        st.subheader("📋 Detailed Test Results")
        
        for test_name, test_result in results['validation_results'].items():
            with st.expander(f"{test_name.replace('_', ' ').title()}", expanded=False):
                
                if test_result.get('status') == 'completed':
                    st.success("✅ Test completed successfully")
                    
                    # Display test-specific results
                    for key, value in test_result.items():
                        if key not in ['status', 'recommendations']:
                            if isinstance(value, (int, float)):
                                st.metric(key.replace('_', ' ').title(), value)
                            elif isinstance(value, bool):
                                st.write(f"**{key.replace('_', ' ').title()}:** {'Yes' if value else 'No'}")
                            elif isinstance(value, list) and value:
                                st.write(f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value[:5]))}")
                    
                    # Display recommendations
                    if test_result.get('recommendations'):
                        st.write("**Recommendations:**")
                        for rec in test_result['recommendations']:
                            st.write(f"• {rec}")
                
                else:
                    st.error(f"❌ Test failed: {test_result.get('error', 'Unknown error')}")
    
    # Overall recommendations
    all_recommendations = []
    for test_result in results.get('validation_results', {}).values():
        if test_result.get('recommendations'):
            all_recommendations.extend(test_result['recommendations'])
    
    if all_recommendations:
        st.subheader("💡 Key Recommendations")
        unique_recommendations = list(set(all_recommendations))[:10]  # Top 10 unique recommendations
        for i, rec in enumerate(unique_recommendations, 1):
            st.write(f"{i}. {rec}")