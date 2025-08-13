import streamlit as st
import re
import pandas as pd
from utils.nlp_processor import NLPProcessor
from utils.citation_validator import CitationValidator
from utils.document_generator import DocumentGenerator
from utils.pdf_processor import PDFProcessor
import nltk
from textstat import flesch_kincaid_grade, flesch_reading_ease
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Academic Paper Rewriter", page_icon="✍️", layout="wide")

def main():
    st.title("✍️ Academic Paper Rewriter & Enhancement")
    st.markdown("AI-powered academic writing enhancement with tone optimization and structure improvement")
    
    # Initialize components
    if 'nlp_processor' not in st.session_state:
        st.session_state.nlp_processor = NLPProcessor()
    if 'citation_validator' not in st.session_state:
        st.session_state.citation_validator = CitationValidator()
    if 'doc_generator' not in st.session_state:
        st.session_state.doc_generator = DocumentGenerator()
    
    # Sidebar options
    with st.sidebar:
        st.header("Rewriting Options")
        
        writing_style = st.selectbox(
            "Target Writing Style",
            ["Academic Research", "Clinical Medicine", "Technical Writing", 
             "Grant Proposal", "Thesis/Dissertation", "Journal Article"]
        )
        
        citation_style = st.selectbox(
            "Citation Style",
            ["APA 7th", "MLA 9th", "Chicago", "Vancouver", "Harvard"]
        )
        
        enhancement_level = st.slider(
            "Enhancement Level",
            min_value=1, max_value=5, value=3,
            help="1=Light editing, 5=Complete rewrite"
        )
        
        target_audience = st.selectbox(
            "Target Audience",
            ["Academic Researchers", "Clinicians", "Graduate Students", 
             "Undergraduate Students", "General Scientific Audience"]
        )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Text Input & Analysis", 
        "🔧 Enhancement Tools", 
        "📊 Quality Metrics", 
        "📄 Export Options"
    ])
    
    with tab1:
        text_input_analysis()
    
    with tab2:
        enhancement_tools()
    
    with tab3:
        quality_metrics()
    
    with tab4:
        export_options()

def text_input_analysis():
    st.header("📝 Text Input & Initial Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Direct Text Entry", "Upload PDF", "Upload Word Document", "Paste from Clipboard"]
    )
    
    original_text = ""
    
    if input_method == "Direct Text Entry":
        original_text = st.text_area(
            "Paste your academic text here:",
            height=400,
            placeholder="Enter the text you want to enhance and rewrite..."
        )
    
    elif input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])
        if uploaded_file:
            try:
                pdf_processor = PDFProcessor()
                original_text = pdf_processor.extract_text(uploaded_file)
                st.success(f"Extracted {len(original_text)} characters from PDF")
                
                with st.expander("Preview extracted text"):
                    st.text_area("Extracted text", original_text[:2000], height=200)
            except Exception as e:
                st.error(f"Error extracting text from PDF: {str(e)}")
    
    elif input_method == "Upload Word Document":
        uploaded_file = st.file_uploader("Upload Word document", type=['docx'])
        if uploaded_file:
            try:
                # Implementation would use python-docx to extract text
                st.info("Word document processing would be implemented here")
            except Exception as e:
                st.error(f"Error reading Word document: {str(e)}")
    
    # Store text in session state
    if original_text:
        st.session_state.original_text = original_text
        
        # Initial analysis
        st.subheader("📊 Initial Text Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            word_count = len(original_text.split())
            st.metric("Word Count", f"{word_count:,}")
        
        with col2:
            char_count = len(original_text)
            st.metric("Character Count", f"{char_count:,}")
        
        with col3:
            sentence_count = len(re.findall(r'[.!?]+', original_text))
            st.metric("Sentences", sentence_count)
        
        with col4:
            paragraph_count = len([p for p in original_text.split('\n\n') if p.strip()])
            st.metric("Paragraphs", paragraph_count)
        
        # Readability analysis
        st.subheader("📈 Readability Analysis")
        
        try:
            flesch_score = flesch_reading_ease(original_text)
            fk_grade = flesch_kincaid_grade(original_text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Flesch Reading Ease", f"{flesch_score:.1f}")
                if flesch_score >= 60:
                    st.success("Easy to read")
                elif flesch_score >= 30:
                    st.warning("Moderately difficult")
                else:
                    st.error("Very difficult to read")
            
            with col2:
                st.metric("Flesch-Kincaid Grade", f"{fk_grade:.1f}")
                st.info(f"Requires {fk_grade:.1f} years of education")
                
        except Exception as e:
            st.warning("Could not calculate readability metrics")
        
        # Structure analysis
        analyze_document_structure(original_text)

def analyze_document_structure(text):
    st.subheader("🏗️ Document Structure Analysis")
    
    # Identify sections
    sections = identify_sections(text)
    
    if sections:
        st.write("**Identified Sections:**")
        for i, section in enumerate(sections, 1):
            st.write(f"{i}. {section['title']} ({section['word_count']} words)")
    
    # Citation analysis
    citations = find_citations(text)
    if citations:
        st.write(f"**Citations Found:** {len(citations)} references")
        
        # Show first few citations
        with st.expander("Preview citations"):
            for i, citation in enumerate(citations[:5], 1):
                st.write(f"{i}. {citation}")
            if len(citations) > 5:
                st.write(f"... and {len(citations) - 5} more")

def identify_sections(text):
    """Identify document sections based on headers and structure"""
    sections = []
    
    # Common academic paper sections
    section_patterns = [
        r'\b(abstract|summary)\b',
        r'\b(introduction|background)\b',
        r'\b(methods?|methodology)\b',
        r'\b(results?|findings)\b',
        r'\b(discussion|analysis)\b',
        r'\b(conclusion|summary)\b',
        r'\b(references?|bibliography)\b'
    ]
    
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if len(paragraph.split()) < 20:  # Likely a header
            for pattern in section_patterns:
                if re.search(pattern, paragraph.lower()):
                    sections.append({
                        'title': paragraph.strip()[:50],
                        'word_count': len(paragraph.split())
                    })
                    break
    
    return sections

def find_citations(text):
    """Find citations in various formats"""
    citation_patterns = [
        r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023) style
        r'\[[^\]]*\d+[^\]]*\]',  # [1], [Author, 2023] style
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\(\d{4}\)',  # Author (2023) style
    ]
    
    citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
    
    return list(set(citations))  # Remove duplicates

def enhancement_tools():
    st.header("🔧 Text Enhancement Tools")
    
    if 'original_text' not in st.session_state:
        st.warning("Please input text in the first tab before using enhancement tools.")
        return
    
    text = st.session_state.original_text
    
    # Enhancement options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Grammar & Style")
        if st.button("Fix Grammar & Punctuation"):
            enhanced_text = fix_grammar_style(text)
            display_enhancement_result("Grammar & Style", text, enhanced_text)
        
        if st.button("Improve Sentence Structure"):
            enhanced_text = improve_sentence_structure(text)
            display_enhancement_result("Sentence Structure", text, enhanced_text)
        
        if st.button("Enhance Vocabulary"):
            enhanced_text = enhance_vocabulary(text)
            display_enhancement_result("Vocabulary", text, enhanced_text)
    
    with col2:
        st.subheader("Academic Tone")
        if st.button("Make More Formal"):
            enhanced_text = make_more_formal(text)
            display_enhancement_result("Formal Tone", text, enhanced_text)
        
        if st.button("Improve Clarity"):
            enhanced_text = improve_clarity(text)
            display_enhancement_result("Clarity", text, enhanced_text)
        
        if st.button("Strengthen Arguments"):
            enhanced_text = strengthen_arguments(text)
            display_enhancement_result("Arguments", text, enhanced_text)
    
    # Section-by-section enhancement
    st.subheader("📋 Section-by-Section Enhancement")
    
    sections_to_enhance = st.multiselect(
        "Select sections to enhance:",
        ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]
    )
    
    if sections_to_enhance and st.button("Enhance Selected Sections"):
        for section in sections_to_enhance:
            st.write(f"**Enhancing {section}...**")
            enhanced_section = enhance_section(text, section)
            st.text_area(f"Enhanced {section}", enhanced_section, height=200)

def fix_grammar_style(text):
    """Fix grammar and style issues"""
    # This would integrate with grammar checking APIs or libraries
    # For demonstration, showing basic improvements
    
    improvements = {
        r'\bthat\s+that\b': 'that',  # Remove duplicate "that"
        r'\bvery\s+very\b': 'extremely',  # Replace double "very"
        r'\bin\s+order\s+to\b': 'to',  # Simplify "in order to"
        r'\bdue\s+to\s+the\s+fact\s+that\b': 'because',  # Simplify wordy phrases
        r'\bit\s+is\s+important\s+to\s+note\s+that\b': 'notably,',
    }
    
    enhanced_text = text
    for pattern, replacement in improvements.items():
        enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
    
    return enhanced_text

def improve_sentence_structure(text):
    """Improve sentence structure and flow"""
    # Split into sentences and analyze structure
    sentences = re.split(r'[.!?]+', text)
    improved_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Basic improvements (real implementation would be more sophisticated)
        if len(sentence.split()) > 30:  # Very long sentences
            # Suggest breaking into shorter sentences
            improved_sentence = sentence  # Placeholder
        else:
            improved_sentence = sentence
            
        improved_sentences.append(improved_sentence)
    
    return '. '.join(improved_sentences) + '.'

def enhance_vocabulary(text):
    """Enhance vocabulary with more academic terms"""
    
    academic_replacements = {
        r'\bshow\b': 'demonstrate',
        r'\bget\b': 'obtain',
        r'\bfind\b': 'identify',
        r'\buse\b': 'utilize',
        r'\bhelp\b': 'facilitate',
        r'\bbig\b': 'substantial',
        r'\bsmall\b': 'minimal',
        r'\bstart\b': 'initiate',
        r'\bend\b': 'conclude',
    }
    
    enhanced_text = text
    for pattern, replacement in academic_replacements.items():
        enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
    
    return enhanced_text

def make_more_formal(text):
    """Make text more formal and academic"""
    
    formal_replacements = {
        r'\bdon\'t\b': 'do not',
        r'\bcan\'t\b': 'cannot',
        r'\bwon\'t\b': 'will not',
        r'\bisn\'t\b': 'is not',
        r'\baren\'t\b': 'are not',
        r'\bI\s+think\b': 'It is suggested that',
        r'\bWe\s+believe\b': 'It is believed that',
        r'\bpretty\s+good\b': 'satisfactory',
        r'\ba\s+lot\s+of\b': 'numerous',
    }
    
    enhanced_text = text
    for pattern, replacement in formal_replacements.items():
        enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
    
    return enhanced_text

def improve_clarity(text):
    """Improve text clarity and readability"""
    
    clarity_improvements = {
        r'\bfacilitate\s+the\s+process\s+of\b': 'enable',
        r'\bin\s+the\s+vicinity\s+of\b': 'near',
        r'\bwith\s+regard\s+to\b': 'regarding',
        r'\bin\s+the\s+event\s+that\b': 'if',
        r'\bprior\s+to\b': 'before',
        r'\bsubsequent\s+to\b': 'after',
    }
    
    enhanced_text = text
    for pattern, replacement in clarity_improvements.items():
        enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
    
    return enhanced_text

def strengthen_arguments(text):
    """Strengthen argumentative language"""
    
    stronger_language = {
        r'\bmight\s+be\b': 'is likely to be',
        r'\bcould\s+be\b': 'may be',
        r'\bseems\s+to\b': 'appears to',
        r'\bprobably\b': 'likely',
        r'\bmaybe\b': 'potentially',
    }
    
    enhanced_text = text
    for pattern, replacement in stronger_language.items():
        enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
    
    return enhanced_text

def enhance_section(text, section_name):
    """Enhance specific sections with targeted improvements"""
    # This would implement section-specific enhancement logic
    return f"[Enhanced {section_name} section would appear here with targeted improvements for academic writing in this specific section type]"

def display_enhancement_result(enhancement_type, original, enhanced):
    """Display before/after comparison"""
    st.subheader(f"✨ {enhancement_type} Enhancement Result")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before:**")
        st.text_area("Original", original[:500], height=150, key=f"before_{enhancement_type}")
    
    with col2:
        st.write("**After:**")
        st.text_area("Enhanced", enhanced[:500], height=150, key=f"after_{enhancement_type}")
    
    # Option to apply changes
    if st.button(f"Apply {enhancement_type} Changes", key=f"apply_{enhancement_type}"):
        st.session_state.original_text = enhanced
        st.success(f"{enhancement_type} enhancement applied!")
        st.rerun()

def quality_metrics():
    st.header("📊 Writing Quality Metrics & Analysis")
    
    if 'original_text' not in st.session_state:
        st.warning("Please input text in the first tab before analyzing quality metrics.")
        return
    
    text = st.session_state.original_text
    
    # Overall quality score
    quality_score = calculate_quality_score(text)
    
    st.subheader("🎯 Overall Quality Score")
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = quality_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Writing Quality Score"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.subheader("📈 Detailed Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Readability Metrics:**")
        try:
            flesch_score = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
            
            st.metric("Flesch Reading Ease", f"{flesch_score:.1f}")
            st.metric("Flesch-Kincaid Grade", f"{fk_grade:.1f}")
            
            # Average sentence length
            sentences = re.split(r'[.!?]+', text)
            avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
            st.metric("Avg Sentence Length", f"{avg_sentence_length:.1f} words")
            
        except Exception as e:
            st.error(f"Error calculating readability: {str(e)}")
    
    with col2:
        st.write("**Academic Style Metrics:**")
        
        # Passive voice detection
        passive_sentences = count_passive_voice(text)
        st.metric("Passive Voice Usage", f"{passive_sentences}%")
        
        # Academic vocabulary
        academic_words = count_academic_vocabulary(text)
        st.metric("Academic Vocabulary", f"{academic_words}%")
        
        # Citation density
        citation_count = len(find_citations(text))
        word_count = len(text.split())
        citation_density = (citation_count / word_count) * 1000 if word_count > 0 else 0
        st.metric("Citations per 1000 words", f"{citation_density:.1f}")
    
    # Improvement suggestions
    st.subheader("💡 Improvement Suggestions")
    suggestions = generate_improvement_suggestions(text)
    
    for i, suggestion in enumerate(suggestions, 1):
        st.write(f"{i}. {suggestion}")

def calculate_quality_score(text):
    """Calculate overall writing quality score"""
    
    scores = []
    
    # Readability score (0-100)
    try:
        flesch_score = flesch_reading_ease(text)
        # Convert to 0-100 scale where 60-100 is good for academic writing
        readability_score = min(100, max(0, (flesch_score - 30) * 2))
        scores.append(readability_score)
    except:
        scores.append(50)  # Default score
    
    # Sentence length score
    sentences = re.split(r'[.!?]+', text)
    avg_length = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
    # Optimal sentence length for academic writing: 15-25 words
    if 15 <= avg_length <= 25:
        length_score = 100
    else:
        length_score = max(0, 100 - abs(avg_length - 20) * 5)
    scores.append(length_score)
    
    # Academic vocabulary score
    academic_score = count_academic_vocabulary(text)
    scores.append(academic_score)
    
    # Citation score (basic check)
    citation_count = len(find_citations(text))
    word_count = len(text.split())
    if word_count > 500:  # Only for longer texts
        citation_score = min(100, citation_count * 10)
    else:
        citation_score = 80  # Default for shorter texts
    scores.append(citation_score)
    
    return sum(scores) / len(scores)

def count_passive_voice(text):
    """Count percentage of passive voice usage"""
    
    passive_patterns = [
        r'\b(is|are|was|were|been|being)\s+\w+ed\b',
        r'\b(is|are|was|were|been|being)\s+\w+en\b',
    ]
    
    sentences = re.split(r'[.!?]+', text)
    passive_count = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        for pattern in passive_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                passive_count += 1
                break
    
    total_sentences = len([s for s in sentences if s.strip()])
    return (passive_count / max(1, total_sentences)) * 100

def count_academic_vocabulary(text):
    """Count percentage of academic vocabulary"""
    
    academic_words = {
        'analyze', 'analysis', 'approach', 'area', 'assessment', 'assume', 'authority',
        'concept', 'conclude', 'consistent', 'constitute', 'context', 'contract', 'create',
        'data', 'definition', 'derive', 'distribution', 'economic', 'environment', 'establish',
        'estimate', 'evaluate', 'evidence', 'export', 'factor', 'financial', 'formula',
        'function', 'identify', 'income', 'indicate', 'individual', 'interpretation', 'involve',
        'issue', 'labor', 'legal', 'legislation', 'major', 'method', 'occur', 'percent',
        'period', 'policy', 'principle', 'procedure', 'process', 'require', 'research',
        'response', 'role', 'section', 'sector', 'significant', 'similar', 'source',
        'specific', 'structure', 'theory', 'variable'
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    academic_count = sum(1 for word in words if word in academic_words)
    
    return (academic_count / max(1, len(words))) * 100

def generate_improvement_suggestions(text):
    """Generate specific improvement suggestions"""
    
    suggestions = []
    
    # Check readability
    try:
        flesch_score = flesch_reading_ease(text)
        if flesch_score < 30:
            suggestions.append("Consider breaking down complex sentences to improve readability.")
    except:
        pass
    
    # Check sentence length
    sentences = re.split(r'[.!?]+', text)
    long_sentences = [s for s in sentences if len(s.split()) > 30]
    if len(long_sentences) > len(sentences) * 0.2:
        suggestions.append("Consider shortening some sentences. Many sentences exceed 30 words.")
    
    # Check passive voice
    passive_pct = count_passive_voice(text)
    if passive_pct > 30:
        suggestions.append("Consider reducing passive voice usage for more direct writing.")
    
    # Check paragraph length
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    long_paragraphs = [p for p in paragraphs if len(p.split()) > 150]
    if len(long_paragraphs) > len(paragraphs) * 0.3:
        suggestions.append("Consider breaking long paragraphs into shorter ones for better readability.")
    
    # Check transitions
    transition_words = ['however', 'furthermore', 'moreover', 'therefore', 'consequently']
    transition_count = sum(text.lower().count(word) for word in transition_words)
    if transition_count < len(paragraphs):
        suggestions.append("Consider adding more transition words to improve flow between ideas.")
    
    if not suggestions:
        suggestions.append("Your text demonstrates good writing quality! Consider minor refinements based on the specific metrics above.")
    
    return suggestions

def export_options():
    st.header("📄 Export & Document Generation")
    
    if 'original_text' not in st.session_state:
        st.warning("Please input and enhance text before exporting.")
        return
    
    text = st.session_state.original_text
    
    # Export format options
    st.subheader("📋 Export Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Document Formats:**")
        
        if st.button("📄 Export as Word Document"):
            doc_generator = st.session_state.doc_generator
            doc_content = doc_generator.create_word_document(text)
            st.download_button(
                label="Download Word Document",
                data=doc_content,
                file_name="enhanced_paper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        
        if st.button("📄 Export as PDF"):
            st.info("PDF export functionality would be implemented here")
        
        if st.button("📄 Export as LaTeX"):
            latex_content = convert_to_latex(text)
            st.download_button(
                label="Download LaTeX File",
                data=latex_content,
                file_name="enhanced_paper.tex",
                mime="text/plain"
            )
    
    with col2:
        st.write("**Citation Formats:**")
        
        citations = find_citations(text)
        if citations:
            citation_style = st.selectbox(
                "Select citation style for export:",
                ["APA 7th", "MLA 9th", "Chicago", "Vancouver"]
            )
            
            if st.button("📚 Generate Bibliography"):
                bibliography = generate_bibliography(citations, citation_style)
                st.text_area("Bibliography", bibliography, height=300)
        else:
            st.info("No citations found in the text")
    
    # Template options
    st.subheader("📐 Academic Templates")
    
    template_type = st.selectbox(
        "Select template type:",
        ["Research Article", "Thesis Chapter", "Grant Proposal", "Conference Abstract", "Review Article"]
    )
    
    if st.button(f"Apply {template_type} Template"):
        templated_content = apply_academic_template(text, template_type)
        st.text_area("Templated Content", templated_content, height=400)
        
        # Download option
        st.download_button(
            label=f"Download {template_type}",
            data=templated_content,
            file_name=f"{template_type.lower().replace(' ', '_')}.txt",
            mime="text/plain"
        )
    
    # Quality report
    st.subheader("📊 Quality Report")
    
    if st.button("Generate Quality Report"):
        quality_report = generate_quality_report(text)
        st.markdown(quality_report)

def convert_to_latex(text):
    """Convert text to LaTeX format"""
    
    latex_template = r"""
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Enhanced Academic Paper}
\author{Author Name}
\date{\today}

\begin{document}

\maketitle

""" + text.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$') + r"""

\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""
    
    return latex_template

def generate_bibliography(citations, style):
    """Generate bibliography in specified style"""
    
    bibliography = f"Bibliography ({style} Style)\n" + "="*50 + "\n\n"
    
    for i, citation in enumerate(citations, 1):
        # This would implement proper citation formatting
        # For demonstration, showing basic format
        bibliography += f"{i}. {citation}\n\n"
    
    return bibliography

def apply_academic_template(text, template_type):
    """Apply academic template structure"""
    
    templates = {
        "Research Article": {
            "sections": ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "References"],
            "structure": "Standard IMRAD format for research articles"
        },
        "Thesis Chapter": {
            "sections": ["Introduction", "Literature Review", "Methodology", "Analysis", "Conclusion"],
            "structure": "Chapter structure for thesis/dissertation"
        },
        "Grant Proposal": {
            "sections": ["Executive Summary", "Problem Statement", "Objectives", "Methodology", "Budget", "Timeline"],
            "structure": "Grant proposal format"
        },
        "Conference Abstract": {
            "sections": ["Background", "Methods", "Results", "Conclusions"],
            "structure": "Conference abstract format (250-300 words)"
        },
        "Review Article": {
            "sections": ["Abstract", "Introduction", "Literature Search", "Analysis", "Discussion", "Conclusion"],
            "structure": "Systematic review article format"
        }
    }
    
    template = templates.get(template_type, templates["Research Article"])
    
    output = f"{template_type} Template\n" + "="*50 + "\n\n"
    output += f"Structure: {template['structure']}\n\n"
    
    for section in template["sections"]:
        output += f"{section.upper()}\n" + "-"*len(section) + "\n"
        output += "[Content for this section would be extracted and organized here]\n\n"
    
    return output

def generate_quality_report(text):
    """Generate comprehensive quality report"""
    
    word_count = len(text.split())
    char_count = len(text)
    
    try:
        flesch_score = flesch_reading_ease(text)
        fk_grade = flesch_kincaid_grade(text)
    except:
        flesch_score = 0
        fk_grade = 0
    
    quality_score = calculate_quality_score(text)
    
    report = f"""
# Writing Quality Report

## Summary Statistics
- **Word Count**: {word_count:,}
- **Character Count**: {char_count:,}
- **Overall Quality Score**: {quality_score:.1f}/100

## Readability Analysis
- **Flesch Reading Ease**: {flesch_score:.1f}
- **Flesch-Kincaid Grade Level**: {fk_grade:.1f}

## Style Analysis
- **Passive Voice Usage**: {count_passive_voice(text):.1f}%
- **Academic Vocabulary**: {count_academic_vocabulary(text):.1f}%

## Recommendations
{chr(10).join(f"- {suggestion}" for suggestion in generate_improvement_suggestions(text))}

## Export Information
- **Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Platform**: Academic Research Platform
"""
    
    return report

if __name__ == "__main__":
    main()
