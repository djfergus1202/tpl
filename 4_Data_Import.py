import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from utils.pdf_processor import PDFProcessor
from utils.r_integration import RAnalytics
from utils.auto_validator import AutoValidator, display_validation_results
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

st.set_page_config(page_title="Data Import & Processing", page_icon="📁", layout="wide")

def main():
    st.title("📁 Data Import & Processing")
    st.markdown("Import, validate, and prepare research data for systematic review and meta-analysis")
    
    # Initialize components
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    if 'r_analytics' not in st.session_state:
        st.session_state.r_analytics = RAnalytics()
    if 'auto_validator' not in st.session_state:
        st.session_state.auto_validator = AutoValidator()
    
    # Sidebar options
    with st.sidebar:
        st.header("Import Options")
        import_type = st.selectbox(
            "Select Import Type",
            [
                "Research Data Files",
                "PDF Papers Extraction", 
                "Database Export",
                "Citation Manager Export",
                "Manual Data Entry",
                "Web Scraping Results"
            ]
        )
        
        st.subheader("Data Validation")
        validate_on_import = st.checkbox("Validate data on import", value=True)
        check_duplicates = st.checkbox("Check for duplicates", value=True)
        
        st.subheader("Processing Options")
        auto_clean = st.checkbox("Auto-clean data", value=False)
        standardize_format = st.checkbox("Standardize formats", value=True)
    
    # Main content based on import type
    if import_type == "Research Data Files":
        research_data_import()
    elif import_type == "PDF Papers Extraction":
        pdf_extraction_import()
    elif import_type == "Database Export":
        database_export_import()
    elif import_type == "Citation Manager Export":
        citation_manager_import()
    elif import_type == "Manual Data Entry":
        manual_data_entry()
    elif import_type == "Web Scraping Results":
        web_scraping_import()

def research_data_import():
    st.header("📊 Research Data Files Import")
    st.markdown("Import CSV, Excel, or other structured data files containing study information")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload research data files",
        type=['csv', 'xlsx', 'xls', 'tsv', 'json'],
        accept_multiple_files=True,
        help="Upload CSV, Excel, TSV, or JSON files containing study data"
    )
    
    if uploaded_files:
        st.subheader("📁 Uploaded Files")
        
        # Display file information
        file_info = []
        for file in uploaded_files:
            file_info.append({
                'Filename': file.name,
                'Size (KB)': round(file.size / 1024, 2),
                'Type': file.type
            })
        
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)
        
        # Process each file
        for i, file in enumerate(uploaded_files):
            with st.expander(f"📄 Process: {file.name}"):
                process_research_data_file(file, i)

def process_research_data_file(file, file_index):
    """Process individual research data file"""
    
    try:
        # Read file based on type
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # CSV reading options
            col1, col2, col3 = st.columns(3)
            with col1:
                separator = st.selectbox("Separator", [',', ';', '\t', '|'], key=f"sep_{file_index}")
            with col2:
                encoding = st.selectbox("Encoding", ['utf-8', 'latin-1', 'cp1252'], key=f"enc_{file_index}")
            with col3:
                header_row = st.number_input("Header row", min_value=0, value=0, key=f"header_{file_index}")
            
            df = pd.read_csv(file, sep=separator, encoding=encoding, header=header_row)
            
        elif file_extension in ['xlsx', 'xls']:
            # Excel reading options
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names
            
            col1, col2 = st.columns(2)
            with col1:
                selected_sheet = st.selectbox("Select sheet", sheet_names, key=f"sheet_{file_index}")
            with col2:
                header_row = st.number_input("Header row", min_value=0, value=0, key=f"excel_header_{file_index}")
            
            df = pd.read_excel(file, sheet_name=selected_sheet, header=header_row)
            
        elif file_extension == 'json':
            json_data = json.load(file)
            df = pd.json_normalize(json_data)
            
        elif file_extension == 'tsv':
            df = pd.read_csv(file, sep='\t')
        
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return
        
        # Display data preview
        st.subheader("📋 Data Preview")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data validation
        validate_research_data(df, file.name)
        
        # Column mapping for meta-analysis
        map_columns_for_analysis(df, file_index)
        
        # Store in session state
        st.session_state[f'imported_data_{file_index}'] = df
        st.session_state[f'imported_filename_{file_index}'] = file.name
        
        st.success(f"✅ Successfully imported {file.name}")
        
    except Exception as e:
        st.error(f"❌ Error processing {file.name}: {str(e)}")

def validate_research_data(df, filename):
    """Validate imported research data"""
    
    st.subheader("🔍 Data Validation Results")
    
    # Basic validation metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
        if missing_values > 0:
            st.warning(f"Found {missing_values} missing values")
    
    with col2:
        duplicate_rows = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_rows)
        if duplicate_rows > 0:
            st.warning(f"Found {duplicate_rows} duplicate rows")
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)
    
    with col4:
        text_cols = len(df.select_dtypes(include=['object']).columns)
        st.metric("Text Columns", text_cols)
    
    # Detailed validation
    validation_issues = []
    
    # Check for common meta-analysis columns
    expected_columns = ['study', 'author', 'year', 'effect_size', 'standard_error', 'sample_size', 'n1', 'n2']
    missing_expected = [col for col in expected_columns if col not in df.columns.str.lower()]
    
    if missing_expected:
        validation_issues.append(f"Missing common meta-analysis columns: {', '.join(missing_expected)}")
    
    # Check for invalid effect sizes
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if 'effect' in col.lower() or 'es' in col.lower():
            invalid_effects = df[col].abs() > 10  # Very large effect sizes might be errors
            if invalid_effects.any():
                validation_issues.append(f"Potentially invalid effect sizes in column '{col}': {invalid_effects.sum()} values > 10")
    
    # Check for negative standard errors or sample sizes
    for col in numeric_columns:
        if any(term in col.lower() for term in ['se', 'standard_error', 'std_err']):
            negative_se = df[col] < 0
            if negative_se.any():
                validation_issues.append(f"Negative standard errors in column '{col}': {negative_se.sum()} values")
        
        if any(term in col.lower() for term in ['n', 'sample_size', 'participants']):
            invalid_n = (df[col] < 1) | (df[col] != df[col].astype(int))
            if invalid_n.any():
                validation_issues.append(f"Invalid sample sizes in column '{col}': {invalid_n.sum()} values")
    
    # Display validation results
    if validation_issues:
        st.subheader("⚠️ Validation Issues Found")
        for issue in validation_issues:
            st.warning(f"• {issue}")
    else:
        st.success("✅ No validation issues found")
    
    # Missing data visualization
    if df.isnull().sum().sum() > 0:
        st.subheader("📊 Missing Data Pattern")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column",
            labels={'x': 'Column', 'y': 'Missing Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

def map_columns_for_analysis(df, file_index):
    """Map columns to standard meta-analysis variables"""
    
    st.subheader("🎯 Column Mapping for Analysis")
    st.markdown("Map your data columns to standard meta-analysis variables")
    
    # Standard meta-analysis variables
    standard_vars = {
        'study_id': 'Study identifier',
        'author': 'Author/first author',
        'year': 'Publication year',
        'effect_size': 'Effect size (ES)',
        'standard_error': 'Standard error (SE)',
        'variance': 'Variance',
        'confidence_interval_lower': 'CI Lower bound',
        'confidence_interval_upper': 'CI Upper bound',
        'sample_size_total': 'Total sample size',
        'sample_size_treatment': 'Treatment group N',
        'sample_size_control': 'Control group N',
        'mean_treatment': 'Treatment group mean',
        'mean_control': 'Control group mean',
        'sd_treatment': 'Treatment group SD',
        'sd_control': 'Control group SD',
        'events_treatment': 'Treatment group events',
        'events_control': 'Control group events'
    }
    
    column_mapping = {}
    available_columns = ['None'] + list(df.columns)
    
    # Create mapping interface
    col1, col2 = st.columns(2)
    
    for i, (var_name, var_description) in enumerate(standard_vars.items()):
        with col1 if i % 2 == 0 else col2:
            mapped_column = st.selectbox(
                f"{var_description}",
                available_columns,
                key=f"map_{var_name}_{file_index}",
                help=f"Map to variable: {var_name}"
            )
            
            if mapped_column != 'None':
                column_mapping[var_name] = mapped_column
    
    # Store mapping
    if column_mapping:
        st.session_state[f'column_mapping_{file_index}'] = column_mapping
        
        # Preview mapped data
        with st.expander("Preview Mapped Data"):
            mapped_df = df[list(column_mapping.values())].copy()
            mapped_df.columns = list(column_mapping.keys())
            st.dataframe(mapped_df.head(), use_container_width=True)
        
        st.success(f"✅ Mapped {len(column_mapping)} columns")

def pdf_extraction_import():
    st.header("📄 PDF Papers Extraction")
    st.markdown("Extract data and metadata from PDF research papers")
    
    # PDF upload
    uploaded_pdfs = st.file_uploader(
        "Upload PDF papers",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF files to extract text, data, and metadata"
    )
    
    if uploaded_pdfs:
        st.subheader("📚 PDF Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            extract_tables = st.checkbox("Extract tables", value=True)
            extract_references = st.checkbox("Extract references", value=True)
        with col2:
            extract_metadata = st.checkbox("Extract metadata", value=True)
            ocr_mode = st.checkbox("Use OCR for scanned PDFs", value=False)
        
        # Automatic validation options
        st.subheader("🔍 Automatic Validation Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            run_auto_validation = st.checkbox("Run automatic validation tests", value=True, 
                                            help="Automatically validate research methodology, statistics, and citations")
        with col2:
            validation_level = st.selectbox("Validation Level", 
                                          ["Quick", "Standard", "Comprehensive"], 
                                          index=1,
                                          help="Choose validation depth")
        with col3:
            save_results = st.checkbox("Save validation results", value=True,
                                     help="Save validation results to session")
        
        # Configure validation tests
        validation_options = {}
        if run_auto_validation:
            with st.expander("🛠️ Configure Validation Tests", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    validation_options['citation_analysis'] = st.checkbox("Citation Analysis", value=True)
                    validation_options['statistical_analysis'] = st.checkbox("Statistical Analysis", value=True)
                    validation_options['meta_analysis_check'] = st.checkbox("Meta-Analysis Check", value=True)
                    validation_options['quality_assessment'] = st.checkbox("Quality Assessment", value=True)
                with col2:
                    validation_options['data_extraction'] = st.checkbox("Data Extraction", value=True)
                    validation_options['reproducibility_check'] = st.checkbox("Reproducibility Check", value=True)
                    validation_options['methodology_validation'] = st.checkbox("Methodology Validation", value=True)
        
        # Process PDFs
        for i, pdf_file in enumerate(uploaded_pdfs):
            with st.expander(f"📄 Process: {pdf_file.name}", expanded=True):
                process_pdf_file(pdf_file, i, extract_tables, extract_references, extract_metadata, 
                               ocr_mode, run_auto_validation, validation_options, save_results)

def process_pdf_file(pdf_file, file_index, extract_tables, extract_references, extract_metadata, 
                   ocr_mode, run_auto_validation=False, validation_options=None, save_results=True):
    """Process individual PDF file with automatic validation"""
    
    try:
        pdf_processor = st.session_state.pdf_processor
        
        # Extract text
        st.write("🔍 Extracting text...")
        text_content = pdf_processor.extract_text(pdf_file)
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", len(text_content))
        with col2:
            st.metric("Words", len(text_content.split()))
        with col3:
            pages_count = pdf_processor.get_page_count(pdf_file) if hasattr(pdf_processor, 'get_page_count') else "N/A"
            st.metric("Pages", pages_count)
        
        # Text preview
        with st.expander("📖 Text Preview"):
            st.text_area("Extracted Text", text_content[:2000], height=300, key=f"text_preview_{file_index}")
        
        # Extract tables if requested
        tables = None
        if extract_tables:
            st.write("📊 Extracting tables...")
            tables = pdf_processor.extract_tables(pdf_file)
            
            if tables:
                st.write(f"Found {len(tables)} tables")
                for i, table in enumerate(tables):
                    with st.expander(f"Table {i+1}"):
                        st.dataframe(table, use_container_width=True)
            else:
                st.info("No tables found")
        
        # Extract references if requested
        references = None
        if extract_references:
            st.write("📚 Extracting references...")
            references = extract_references_from_text(text_content)
            
            if references:
                st.write(f"Found {len(references)} references")
                with st.expander("References"):
                    for i, ref in enumerate(references[:10], 1):
                        st.write(f"{i}. {ref}")
                    if len(references) > 10:
                        st.write(f"... and {len(references) - 10} more")
            else:
                st.info("No references found")
        
        # Extract metadata if requested
        metadata = None
        if extract_metadata:
            st.write("ℹ️ Extracting metadata...")
            metadata = pdf_processor.extract_metadata(pdf_file)
            
            if metadata:
                with st.expander("PDF Metadata"):
                    for key, value in metadata.items():
                        st.write(f"**{key}**: {value}")
        
        # Run automatic validation if requested
        validation_results = None
        if run_auto_validation and validation_options:
            st.write("🔍 Running automatic validation tests...")
            
            try:
                auto_validator = st.session_state.auto_validator
                
                # Create a temporary file-like object for validation
                validation_file = io.BytesIO()
                validation_file.write(pdf_file.getvalue())
                validation_file.seek(0)
                validation_file.name = pdf_file.name
                
                # Run validation
                validation_results = auto_validator.run_automatic_validation(
                    validation_file, validation_options
                )
                
                # Display validation results
                display_validation_results(validation_results)
                
            except Exception as e:
                st.error(f"Validation failed: {str(e)}")
                st.info("PDF processing will continue without validation")
        
        # Store extracted data
        st.session_state[f'pdf_text_{file_index}'] = text_content
        if tables:
            st.session_state[f'pdf_tables_{file_index}'] = tables
        if references:
            st.session_state[f'pdf_references_{file_index}'] = references
        if metadata:
            st.session_state[f'pdf_metadata_{file_index}'] = metadata
        if validation_results and save_results:
            st.session_state[f'pdf_validation_{file_index}'] = validation_results
        
        st.success(f"✅ Successfully processed {pdf_file.name}")
        
        # Provide next steps if validation was run
        if validation_results:
            st.info("💡 Next steps: Review validation results above, then use other modules to verify specific analyses or rewrite sections as needed.")
        
    except Exception as e:
        st.error(f"❌ Error processing {pdf_file.name}: {str(e)}")

def extract_references_from_text(text):
    """Extract references from text using pattern matching"""
    
    # Common reference patterns
    patterns = [
        r'\n\d+\.\s+[A-Z][^.\n]+\.\s+[^.\n]+\.\s+\d{4}[^.\n]*\.',  # Numbered references
        r'\n[A-Z][a-z]+,\s+[A-Z]\.\s*[A-Z]?\.\s*\(\d{4}\)[^.\n]+\.',  # Author (Year) format
        r'\n[A-Z][^.\n]+\(\d{4}\)[^.\n]+\.',  # Author (Year) format variations
    ]
    
    references = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        references.extend([match.strip() for match in matches])
    
    # Remove duplicates and filter valid references
    unique_refs = []
    for ref in references:
        if len(ref) > 50 and ref not in unique_refs:  # Minimum length filter
            unique_refs.append(ref)
    
    return unique_refs

def database_export_import():
    st.header("🗄️ Database Export Import")
    st.markdown("Import data from research databases and systematic review platforms")
    
    # Database source selection
    database_source = st.selectbox(
        "Select database source",
        [
            "PubMed/MEDLINE",
            "Web of Science", 
            "Scopus",
            "Cochrane Library",
            "Embase",
            "Google Scholar",
            "Custom Database",
            "Other"
        ]
    )
    
    st.subheader(f"📊 {database_source} Data Import")
    
    # File upload for database exports
    exported_file = st.file_uploader(
        f"Upload {database_source} export file",
        type=['csv', 'txt', 'ris', 'bib', 'xml', 'xlsx'],
        help="Upload the exported file from your database search"
    )
    
    if exported_file:
        file_extension = exported_file.name.split('.')[-1].lower()
        
        try:
            if file_extension in ['csv', 'txt']:
                # Try to detect delimiter
                sample = str(exported_file.read(1000), 'utf-8')
                exported_file.seek(0)
                
                delimiter = detect_delimiter(sample)
                df = pd.read_csv(exported_file, delimiter=delimiter)
                
                st.subheader("📋 Imported Database Records")
                st.write(f"Records imported: {len(df)}")
                st.dataframe(df.head(), use_container_width=True)
                
                # Database-specific processing
                process_database_records(df, database_source)
                
            elif file_extension == 'ris':
                st.info("RIS file processing would be implemented here")
                # Would implement RIS (Research Information Systems) format parsing
                
            elif file_extension == 'bib':
                st.info("BibTeX file processing would be implemented here")
                # Would implement BibTeX format parsing
                
            elif file_extension == 'xml':
                st.info("XML file processing would be implemented here")
                # Would implement XML parsing for database exports
                
        except Exception as e:
            st.error(f"Error processing {database_source} export: {str(e)}")

def detect_delimiter(sample):
    """Detect the delimiter used in a CSV file"""
    common_delimiters = [',', '\t', ';', '|']
    delimiter_counts = {}
    
    for delimiter in common_delimiters:
        delimiter_counts[delimiter] = sample.count(delimiter)
    
    return max(delimiter_counts, key=delimiter_counts.get)

def process_database_records(df, database_source):
    """Process records from specific databases"""
    
    st.subheader(f"🔧 {database_source} Record Processing")
    
    # Common database fields mapping
    field_mappings = {
        "PubMed/MEDLINE": {
            'PMID': 'pmid',
            'Title': 'title', 
            'Authors': 'authors',
            'Journal': 'journal',
            'Publication Year': 'year',
            'Abstract': 'abstract',
            'DOI': 'doi'
        },
        "Web of Science": {
            'UT': 'ut_id',
            'TI': 'title',
            'AU': 'authors', 
            'SO': 'journal',
            'PY': 'year',
            'AB': 'abstract',
            'DI': 'doi'
        }
    }
    
    # Apply database-specific processing
    if database_source in field_mappings:
        mapping = field_mappings[database_source]
        
        # Show field mapping
        st.write("**Field Mapping:**")
        for db_field, standard_field in mapping.items():
            if db_field in df.columns:
                st.write(f"• {db_field} → {standard_field}")
        
        # Standardize column names
        df_standardized = df.rename(columns=mapping)
        
        # Data quality checks
        check_database_quality(df_standardized, database_source)
        
        # Store processed data
        st.session_state[f'database_records_{database_source}'] = df_standardized
        
        st.success(f"✅ Processed {len(df_standardized)} records from {database_source}")

def check_database_quality(df, database_source):
    """Check quality of database records"""
    
    st.subheader("🔍 Data Quality Assessment")
    
    quality_metrics = {}
    
    # Check for essential fields
    essential_fields = ['title', 'authors', 'year']
    for field in essential_fields:
        if field in df.columns:
            missing_count = df[field].isnull().sum()
            quality_metrics[f"Missing {field}"] = missing_count
    
    # Check for duplicates
    if 'title' in df.columns:
        duplicate_titles = df['title'].duplicated().sum()
        quality_metrics["Duplicate titles"] = duplicate_titles
    
    # Year range validation
    if 'year' in df.columns:
        current_year = datetime.now().year
        invalid_years = ((df['year'] < 1900) | (df['year'] > current_year)).sum()
        quality_metrics["Invalid years"] = invalid_years
    
    # Display quality metrics
    col1, col2, col3 = st.columns(3)
    metrics_items = list(quality_metrics.items())
    
    for i, (metric, value) in enumerate(metrics_items):
        with [col1, col2, col3][i % 3]:
            st.metric(metric, value)
            if value > 0:
                st.warning(f"Issues found: {value}")

def citation_manager_import():
    st.header("📚 Citation Manager Export")
    st.markdown("Import references from citation management software")
    
    # Citation manager selection
    citation_manager = st.selectbox(
        "Select citation manager",
        ["Zotero", "Mendeley", "EndNote", "RefWorks", "Citavi", "Other"]
    )
    
    st.subheader(f"📖 {citation_manager} Import")
    
    # File format options
    export_format = st.selectbox(
        "Export format",
        ["BibTeX (.bib)", "RIS (.ris)", "CSV (.csv)", "XML (.xml)", "JSON (.json)"]
    )
    
    uploaded_citation_file = st.file_uploader(
        f"Upload {citation_manager} export file",
        type=['bib', 'ris', 'csv', 'xml', 'json'],
        help=f"Export your references from {citation_manager} and upload here"
    )
    
    if uploaded_citation_file:
        try:
            process_citation_file(uploaded_citation_file, citation_manager, export_format)
        except Exception as e:
            st.error(f"Error processing citation file: {str(e)}")

def process_citation_file(file, citation_manager, export_format):
    """Process citation manager export file"""
    
    file_extension = file.name.split('.')[-1].lower()
    
    st.write(f"📖 Processing {citation_manager} {export_format} file...")
    
    if file_extension == 'bib':
        # BibTeX processing
        content = str(file.read(), 'utf-8')
        citations = parse_bibtex(content)
        
    elif file_extension == 'ris':
        # RIS processing
        content = str(file.read(), 'utf-8')
        citations = parse_ris(content)
        
    elif file_extension == 'csv':
        # CSV processing
        df = pd.read_csv(file)
        citations = df.to_dict('records')
        
    elif file_extension == 'json':
        # JSON processing
        citations = json.load(file)
        
    else:
        st.error(f"Unsupported format: {file_extension}")
        return
    
    # Display results
    st.write(f"📊 Imported {len(citations)} citations")
    
    # Show preview
    if citations:
        st.subheader("📋 Citation Preview")
        
        # Convert to DataFrame for display
        if isinstance(citations[0], dict):
            preview_df = pd.DataFrame(citations[:10])
            st.dataframe(preview_df, use_container_width=True)
        else:
            for i, citation in enumerate(citations[:5], 1):
                st.write(f"{i}. {citation}")
    
    # Store citations
    st.session_state[f'citations_{citation_manager}'] = citations
    st.success(f"✅ Successfully imported citations from {citation_manager}")

def parse_bibtex(content):
    """Parse BibTeX content"""
    # Basic BibTeX parsing (would use a proper library in production)
    citations = []
    entries = re.findall(r'@\w+\{[^@]+\}', content)
    
    for entry in entries:
        citation = {}
        # Extract title
        title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry)
        if title_match:
            citation['title'] = title_match.group(1)
        
        # Extract author
        author_match = re.search(r'author\s*=\s*\{([^}]+)\}', entry)
        if author_match:
            citation['author'] = author_match.group(1)
        
        # Extract year
        year_match = re.search(r'year\s*=\s*\{([^}]+)\}', entry)
        if year_match:
            citation['year'] = year_match.group(1)
        
        citations.append(citation)
    
    return citations

def parse_ris(content):
    """Parse RIS content"""
    # Basic RIS parsing
    citations = []
    records = content.split('ER  -')
    
    for record in records:
        if not record.strip():
            continue
            
        citation = {}
        lines = record.split('\n')
        
        for line in lines:
            if line.startswith('TI  -'):
                citation['title'] = line[5:].strip()
            elif line.startswith('AU  -'):
                if 'authors' not in citation:
                    citation['authors'] = []
                citation['authors'].append(line[5:].strip())
            elif line.startswith('PY  -'):
                citation['year'] = line[5:].strip()
            elif line.startswith('JF  -'):
                citation['journal'] = line[5:].strip()
        
        if citation:
            citations.append(citation)
    
    return citations

def manual_data_entry():
    st.header("✏️ Manual Data Entry")
    st.markdown("Manually enter study data for meta-analysis")
    
    # Study entry form
    st.subheader("📝 Enter Study Information")
    
    # Initialize studies list in session state
    if 'manual_studies' not in st.session_state:
        st.session_state.manual_studies = []
    
    # Study entry form
    with st.form("study_entry_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            study_id = st.text_input("Study ID/Name")
            first_author = st.text_input("First Author")
            publication_year = st.number_input("Publication Year", min_value=1900, max_value=2025, value=2023)
            journal = st.text_input("Journal")
            
        with col2:
            study_design = st.selectbox("Study Design", 
                                      ["RCT", "Cohort", "Case-Control", "Cross-sectional", "Other"])
            country = st.text_input("Country")
            sample_size_total = st.number_input("Total Sample Size", min_value=1, value=100)
            outcome_measure = st.text_input("Outcome Measure")
        
        # Effect size data
        st.subheader("📊 Effect Size Data")
        
        effect_size_type = st.selectbox("Effect Size Type",
                                      ["Mean Difference", "Standardized Mean Difference", 
                                       "Odds Ratio", "Risk Ratio", "Correlation"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            effect_size = st.number_input("Effect Size", value=0.0)
        with col2:
            standard_error = st.number_input("Standard Error", min_value=0.001, value=0.1)
        with col3:
            confidence_level = st.number_input("Confidence Level (%)", min_value=80, max_value=99, value=95)
        
        # Raw data (optional)
        with st.expander("Raw Data (Optional)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Treatment Group**")
                n_treatment = st.number_input("Sample Size", min_value=1, value=50, key="n_treat")
                mean_treatment = st.number_input("Mean", value=0.0, key="mean_treat")
                sd_treatment = st.number_input("Standard Deviation", min_value=0.001, value=1.0, key="sd_treat")
                
            with col2:
                st.write("**Control Group**")
                n_control = st.number_input("Sample Size", min_value=1, value=50, key="n_ctrl")
                mean_control = st.number_input("Mean", value=0.0, key="mean_ctrl")
                sd_control = st.number_input("Standard Deviation", min_value=0.001, value=1.0, key="sd_ctrl")
        
        # Submit button
        submitted = st.form_submit_button("Add Study")
        
        if submitted and study_id:
            # Create study record
            study_record = {
                'study_id': study_id,
                'first_author': first_author,
                'publication_year': publication_year,
                'journal': journal,
                'study_design': study_design,
                'country': country,
                'sample_size_total': sample_size_total,
                'outcome_measure': outcome_measure,
                'effect_size_type': effect_size_type,
                'effect_size': effect_size,
                'standard_error': standard_error,
                'confidence_level': confidence_level,
                'n_treatment': n_treatment,
                'mean_treatment': mean_treatment,
                'sd_treatment': sd_treatment,
                'n_control': n_control,
                'mean_control': mean_control,
                'sd_control': sd_control,
                'date_entered': datetime.now().isoformat()
            }
            
            st.session_state.manual_studies.append(study_record)
            st.success(f"✅ Added study: {study_id}")
            st.rerun()
    
    # Display entered studies
    if st.session_state.manual_studies:
        st.subheader("📋 Entered Studies")
        
        studies_df = pd.DataFrame(st.session_state.manual_studies)
        st.dataframe(studies_df, use_container_width=True)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Analyze Studies"):
                analyze_manual_studies(studies_df)
        
        with col2:
            csv = studies_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="manual_studies.csv",
                mime="text/csv"
            )
        
        with col3:
            if st.button("🗑️ Clear All Studies"):
                st.session_state.manual_studies = []
                st.rerun()

def analyze_manual_studies(studies_df):
    """Analyze manually entered studies"""
    
    st.subheader("📊 Manual Studies Analysis")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Studies", len(studies_df))
    with col2:
        mean_es = studies_df['effect_size'].mean()
        st.metric("Mean Effect Size", f"{mean_es:.3f}")
    with col3:
        total_n = studies_df['sample_size_total'].sum()
        st.metric("Total Participants", f"{total_n:,}")
    with col4:
        year_range = f"{studies_df['publication_year'].min()}-{studies_df['publication_year'].max()}"
        st.metric("Year Range", year_range)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Effect sizes plot
        fig = px.scatter(
            studies_df, 
            x='publication_year', 
            y='effect_size',
            size='sample_size_total',
            hover_data=['study_id', 'first_author'],
            title="Effect Sizes by Publication Year"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Study design distribution
        design_counts = studies_df['study_design'].value_counts()
        fig = px.pie(
            values=design_counts.values,
            names=design_counts.index,
            title="Study Design Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def web_scraping_import():
    st.header("🕷️ Web Scraping Results")
    st.markdown("Import data from web scraping of research databases and journals")
    
    st.info("Web scraping functionality would be implemented here with appropriate rate limiting and respect for robots.txt")
    
    # Placeholder for web scraping interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Search Parameters")
        search_query = st.text_input("Search Query")
        databases = st.multiselect("Target Databases", 
                                 ["PubMed", "arXiv", "bioRxiv", "Google Scholar"])
        max_results = st.number_input("Maximum Results", min_value=10, max_value=1000, value=100)
    
    with col2:
        st.subheader("⚙️ Scraping Options")
        delay_between_requests = st.slider("Delay Between Requests (seconds)", 1, 10, 2)
        respect_robots_txt = st.checkbox("Respect robots.txt", value=True)
        use_proxies = st.checkbox("Use proxy rotation", value=False)
    
    if st.button("🚀 Start Scraping"):
        st.warning("Web scraping would be implemented here with proper rate limiting and ethical considerations")

if __name__ == "__main__":
    main()
