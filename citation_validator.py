import re
import requests
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import urllib.parse
import time
from dataclasses import dataclass
import difflib

@dataclass
class CitationInfo:
    """Container for citation information"""
    original_text: str
    authors: List[str]
    title: str
    journal: str
    year: Optional[int]
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    doi: Optional[str]
    pmid: Optional[str]
    citation_style: str
    is_valid: bool
    issues: List[str]

@dataclass
class ValidationResult:
    """Container for citation validation results"""
    total_citations: int
    valid_citations: int
    invalid_citations: int
    missing_info: int
    duplicate_citations: int
    style_inconsistencies: int
    issues: List[str]
    suggestions: List[str]

class CitationValidator:
    """Citation validation and checking utilities"""
    
    def __init__(self):
        self.api_delays = {
            'crossref': 1.0,  # Delay between CrossRef API calls
            'pubmed': 0.5,    # Delay between PubMed API calls
        }
        self.last_api_call = {}
        
        # Common journal abbreviations
        self.journal_abbreviations = self._load_journal_abbreviations()
        
        # Citation style patterns
        self.citation_patterns = self._load_citation_patterns()
        
        # Common citation issues
        self.common_issues = self._load_common_issues()
    
    def _load_journal_abbreviations(self) -> Dict[str, str]:
        """Load common journal abbreviations"""
        return {
            'Nature': 'Nat.',
            'Science': 'Science',
            'Cell': 'Cell',
            'New England Journal of Medicine': 'N. Engl. J. Med.',
            'The Lancet': 'Lancet',
            'Journal of the American Medical Association': 'JAMA',
            'Proceedings of the National Academy of Sciences': 'Proc. Natl. Acad. Sci.',
            'Journal of Biological Chemistry': 'J. Biol. Chem.',
            'Nature Medicine': 'Nat. Med.',
            'Nature Genetics': 'Nat. Genet.',
            'Nature Biotechnology': 'Nat. Biotechnol.',
            'Cancer Research': 'Cancer Res.',
            'Journal of Clinical Oncology': 'J. Clin. Oncol.',
            'Blood': 'Blood',
            'Circulation': 'Circulation',
            'Journal of the National Cancer Institute': 'J. Natl. Cancer Inst.',
            'Clinical Cancer Research': 'Clin. Cancer Res.',
            'Oncogene': 'Oncogene',
            'Cancer Cell': 'Cancer Cell',
            'Molecular Cell': 'Mol. Cell'
        }
    
    def _load_citation_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load citation style patterns"""
        return {
            'APA': {
                'in_text': r'\([^)]*\d{4}[^)]*\)',
                'reference': r'^[A-Z][^.]*\.\s*\(\d{4}\)\.\s*[^.]*\.\s*[^.]*\.',
                'format': 'Author, A. A. (Year). Title. Journal Name, Volume(Issue), pages.'
            },
            'MLA': {
                'in_text': r'\([^)]*\d+[^)]*\)',
                'reference': r'^[A-Z][^.]*\.\s*"[^"]*"\s*[^,]*,\s*\d+',
                'format': 'Author, First. "Title." Journal Name, vol. #, no. #, Year, pages.'
            },
            'Chicago': {
                'in_text': r'\([^)]*\d{4}[^)]*\)',
                'reference': r'^[A-Z][^.]*\.\s*"[^"]*"\s*[^.]*\.\s*[^.]*\.',
                'format': 'Author, First Last. "Title." Journal Name vol, no. # (Year): pages.'
            },
            'Vancouver': {
                'in_text': r'\[\d+\]',
                'reference': r'^\d+\.\s*[A-Z][^.]*\.\s*[^.]*\.\s*[^.]*\.\s*\d{4}',
                'format': '1. Author AA. Title. Journal Name. Year;Vol(Issue):pages.'
            },
            'Harvard': {
                'in_text': r'\([^)]*\d{4}[^)]*\)',
                'reference': r'^[A-Z][^,]*,\s*[A-Z]\.\s*\(\d{4}\)',
                'format': 'Author, A. (Year) Title. Journal Name, Volume(Issue), pages.'
            },
            'Nature': {
                'in_text': r'\b\d+\b',
                'reference': r'^\d+\.\s*[A-Z][^.]*\.\s*[^.]*\.\s*[^.]*\.\s*\d{4}',
                'format': '1. Author, A. A. Title. Journal Name vol, pages (year).'
            }
        }
    
    def _load_common_issues(self) -> List[Dict[str, str]]:
        """Load common citation issues and fixes"""
        return [
            {
                'pattern': r'\bet\s+al\b',
                'replacement': 'et al.',
                'description': 'Missing period after "et al"'
            },
            {
                'pattern': r'\b(\d+)\s*-\s*(\d+)\b',
                'replacement': r'\1–\2',
                'description': 'Use en dash for page ranges'
            },
            {
                'pattern': r'\bvol\.\s*(\d+)\s*,\s*no\.\s*(\d+)\b',
                'replacement': r'vol. \1, no. \2',
                'description': 'Standardize volume and issue format'
            },
            {
                'pattern': r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
                'replacement': lambda m: {
                    'Jan': 'January', 'Feb': 'February', 'Mar': 'March',
                    'Apr': 'April', 'May': 'May', 'Jun': 'June',
                    'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
                    'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
                }[m.group(1)],
                'description': 'Expand abbreviated month names'
            }
        ]
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citations from text"""
        citations = []
        
        # Pattern for reference list
        ref_patterns = [
            r'(?:References?|Bibliography)\s*\n(.*?)(?=\n\s*(?:Appendix|Figure|Table|\Z))',
            r'\n\d+\.\s+([^\n]+(?:\n(?!\d+\.)[^\n]+)*)',  # Numbered references
            r'\n([A-Z][^.\n]+\.\s*\(\d{4}\)[^\n]+)',  # Author (Year) format
        ]
        
        for pattern in ref_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                if isinstance(match, str) and len(match.strip()) > 20:
                    # Split multiple references if found together
                    if pattern == ref_patterns[0]:  # Full reference section
                        sub_refs = re.split(r'\n(?=\d+\.|\n[A-Z])', match)
                        citations.extend([ref.strip() for ref in sub_refs if len(ref.strip()) > 20])
                    else:
                        citations.append(match.strip())
        
        # Clean and deduplicate
        cleaned_citations = []
        for citation in citations:
            citation = re.sub(r'\s+', ' ', citation.strip())
            if len(citation) > 20 and citation not in cleaned_citations:
                cleaned_citations.append(citation)
        
        return cleaned_citations
    
    def parse_citation(self, citation_text: str) -> CitationInfo:
        """Parse a citation and extract components"""
        
        issues = []
        
        # Initialize citation info
        citation_info = CitationInfo(
            original_text=citation_text,
            authors=[],
            title="",
            journal="",
            year=None,
            volume=None,
            issue=None,
            pages=None,
            doi=None,
            pmid=None,
            citation_style="Unknown",
            is_valid=False,
            issues=[]
        )
        
        # Detect citation style
        citation_style = self._detect_citation_style(citation_text)
        citation_info.citation_style = citation_style
        
        # Extract components based on style
        if citation_style in ['APA', 'Harvard']:
            citation_info = self._parse_apa_harvard(citation_text, citation_info)
        elif citation_style == 'MLA':
            citation_info = self._parse_mla(citation_text, citation_info)
        elif citation_style == 'Chicago':
            citation_info = self._parse_chicago(citation_text, citation_info)
        elif citation_style == 'Vancouver':
            citation_info = self._parse_vancouver(citation_text, citation_info)
        else:
            citation_info = self._parse_generic(citation_text, citation_info)
        
        # Extract DOI and PMID
        citation_info.doi = self._extract_doi(citation_text)
        citation_info.pmid = self._extract_pmid(citation_text)
        
        # Validate citation
        citation_info.is_valid = self._validate_citation_info(citation_info)
        
        return citation_info
    
    def _detect_citation_style(self, citation: str) -> str:
        """Detect the citation style"""
        
        for style, patterns in self.citation_patterns.items():
            if re.search(patterns['reference'], citation):
                return style
        
        # Additional heuristics
        if re.search(r'^\d+\.', citation.strip()):
            return 'Vancouver'
        elif re.search(r'\(\d{4}\)', citation):
            if '"' in citation:
                return 'Chicago'
            else:
                return 'APA'
        elif '"' in citation and re.search(r'vol\.\s*\d+', citation):
            return 'MLA'
        
        return 'Unknown'
    
    def _parse_apa_harvard(self, citation: str, citation_info: CitationInfo) -> CitationInfo:
        """Parse APA or Harvard style citation"""
        
        # Author(s) (Year). Title. Journal, Volume(Issue), pages.
        
        # Extract year
        year_match = re.search(r'\((\d{4})\)', citation)
        if year_match:
            citation_info.year = int(year_match.group(1))
        
        # Extract authors (before year)
        if year_match:
            author_part = citation[:year_match.start()].strip()
            authors = self._parse_authors(author_part)
            citation_info.authors = authors
        
        # Extract title (after year, before journal)
        if year_match:
            after_year = citation[year_match.end():].strip()
            if after_year.startswith('.'):
                after_year = after_year[1:].strip()
            
            # Title usually ends with period before journal
            title_match = re.search(r'^([^.]+)\.', after_year)
            if title_match:
                citation_info.title = title_match.group(1).strip()
                
                # Extract journal and volume info
                remainder = after_year[title_match.end():].strip()
                journal_info = self._parse_journal_info(remainder)
                citation_info.journal = journal_info.get('journal', '')
                citation_info.volume = journal_info.get('volume')
                citation_info.issue = journal_info.get('issue')
                citation_info.pages = journal_info.get('pages')
        
        return citation_info
    
    def _parse_mla(self, citation: str, citation_info: CitationInfo) -> CitationInfo:
        """Parse MLA style citation"""
        
        # Author, First. "Title." Journal Name, vol. #, no. #, Year, pages.
        
        # Extract author (before first period)
        author_match = re.search(r'^([^.]+)\.', citation)
        if author_match:
            citation_info.authors = [author_match.group(1).strip()]
        
        # Extract title (in quotes)
        title_match = re.search(r'"([^"]+)"', citation)
        if title_match:
            citation_info.title = title_match.group(1).strip()
        
        # Extract year
        year_match = re.search(r'\b(\d{4})\b', citation)
        if year_match:
            citation_info.year = int(year_match.group(1))
        
        # Extract journal and other info
        if title_match:
            after_title = citation[title_match.end():].strip()
            journal_info = self._parse_journal_info(after_title)
            citation_info.journal = journal_info.get('journal', '')
            citation_info.volume = journal_info.get('volume')
            citation_info.issue = journal_info.get('issue')
            citation_info.pages = journal_info.get('pages')
        
        return citation_info
    
    def _parse_chicago(self, citation: str, citation_info: CitationInfo) -> CitationInfo:
        """Parse Chicago style citation"""
        
        # Similar to APA but with different punctuation
        return self._parse_apa_harvard(citation, citation_info)
    
    def _parse_vancouver(self, citation: str, citation_info: CitationInfo) -> CitationInfo:
        """Parse Vancouver style citation"""
        
        # 1. Author AA. Title. Journal Name. Year;Vol(Issue):pages.
        
        # Remove number at beginning
        citation_clean = re.sub(r'^\d+\.\s*', '', citation)
        
        # Extract authors (before title, usually ends with period)
        author_match = re.search(r'^([^.]+)\.', citation_clean)
        if author_match:
            authors = self._parse_authors(author_match.group(1))
            citation_info.authors = authors
            
            remainder = citation_clean[author_match.end():].strip()
            
            # Extract title (next part before period)
            title_match = re.search(r'^([^.]+)\.', remainder)
            if title_match:
                citation_info.title = title_match.group(1).strip()
                
                remainder = remainder[title_match.end():].strip()
                
                # Extract journal and other info
                journal_info = self._parse_journal_info(remainder)
                citation_info.journal = journal_info.get('journal', '')
                citation_info.year = journal_info.get('year')
                citation_info.volume = journal_info.get('volume')
                citation_info.issue = journal_info.get('issue')
                citation_info.pages = journal_info.get('pages')
        
        return citation_info
    
    def _parse_generic(self, citation: str, citation_info: CitationInfo) -> CitationInfo:
        """Parse citation with unknown style"""
        
        # Extract year
        year_match = re.search(r'\b(\d{4})\b', citation)
        if year_match:
            citation_info.year = int(year_match.group(1))
        
        # Extract potential authors (at beginning)
        author_part = citation.split('.')[0] if '.' in citation else citation[:50]
        authors = self._parse_authors(author_part)
        if authors:
            citation_info.authors = authors
        
        # Extract potential title (in quotes or between periods)
        title_match = re.search(r'"([^"]+)"', citation)
        if title_match:
            citation_info.title = title_match.group(1)
        else:
            # Look for title-like content
            parts = citation.split('.')
            for part in parts[1:3]:  # Check second and third parts
                if len(part.strip()) > 10 and not re.search(r'\d{4}', part):
                    citation_info.title = part.strip()
                    break
        
        return citation_info
    
    def _parse_authors(self, author_text: str) -> List[str]:
        """Parse author names from text"""
        
        authors = []
        
        # Handle different author separators
        separators = [' and ', ', and ', '; ', ', ']
        
        for sep in separators:
            if sep in author_text:
                authors = [author.strip() for author in author_text.split(sep)]
                break
        
        if not authors:
            authors = [author_text.strip()]
        
        # Clean author names
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if author and len(author) > 1:
                # Remove common prefixes
                author = re.sub(r'^(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s*', '', author)
                cleaned_authors.append(author)
        
        return cleaned_authors[:10]  # Limit to 10 authors
    
    def _parse_journal_info(self, journal_text: str) -> Dict[str, Optional[str]]:
        """Parse journal, volume, issue, and page information"""
        
        info = {
            'journal': None,
            'volume': None,
            'issue': None,
            'pages': None,
            'year': None
        }
        
        # Remove leading/trailing periods
        journal_text = journal_text.strip(' .')
        
        # Extract year if present
        year_match = re.search(r'\b(\d{4})\b', journal_text)
        if year_match:
            info['year'] = int(year_match.group(1))
        
        # Extract volume and issue patterns
        vol_issue_patterns = [
            r'(\d+)\((\d+)\)',  # 123(4)
            r'vol\.\s*(\d+),?\s*no\.\s*(\d+)',  # vol. 123, no. 4
            r'Volume\s*(\d+),?\s*Issue\s*(\d+)',  # Volume 123, Issue 4
            r'(\d+):(\d+)',  # 123:4
        ]
        
        for pattern in vol_issue_patterns:
            match = re.search(pattern, journal_text, re.IGNORECASE)
            if match:
                info['volume'] = match.group(1)
                info['issue'] = match.group(2)
                break
        
        # Extract volume only patterns
        if not info['volume']:
            vol_patterns = [r'vol\.\s*(\d+)', r'Volume\s*(\d+)', r'\b(\d+)\b']
            for pattern in vol_patterns:
                match = re.search(pattern, journal_text, re.IGNORECASE)
                if match:
                    info['volume'] = match.group(1)
                    break
        
        # Extract pages
        page_patterns = [
            r'pp?\.\s*(\d+[-–]\d+)',  # pp. 123-456
            r'pages?\s*(\d+[-–]\d+)',  # pages 123-456
            r'(\d+[-–]\d+)',  # 123-456
            r':(\d+[-–]\d+)',  # :123-456
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, journal_text)
            if match:
                info['pages'] = match.group(1)
                break
        
        # Extract journal name (remaining text after removing other components)
        journal_name = journal_text
        
        # Remove volume, issue, pages, year
        for key, value in info.items():
            if value and key != 'journal':
                # Remove the matched patterns
                journal_name = re.sub(rf'\b{re.escape(str(value))}\b', '', journal_name)
        
        # Remove common patterns
        journal_name = re.sub(r'\d+\(\d+\)', '', journal_name)
        journal_name = re.sub(r'vol\.\s*\d+', '', journal_name, flags=re.IGNORECASE)
        journal_name = re.sub(r'no\.\s*\d+', '', journal_name, flags=re.IGNORECASE)
        journal_name = re.sub(r'pp?\.\s*\d+[-–]\d+', '', journal_name, flags=re.IGNORECASE)
        journal_name = re.sub(r':\d+[-–]\d+', '', journal_name)
        journal_name = re.sub(r'\b\d{4}\b', '', journal_name)
        
        # Clean up journal name
        journal_name = re.sub(r'[,;:.]+', '', journal_name)
        journal_name = re.sub(r'\s+', ' ', journal_name).strip()
        
        if journal_name:
            info['journal'] = journal_name
        
        return info
    
    def _extract_doi(self, citation: str) -> Optional[str]:
        """Extract DOI from citation"""
        
        doi_patterns = [
            r'doi:\s*(10\.\d+/[^\s]+)',
            r'DOI:\s*(10\.\d+/[^\s]+)',
            r'https?://doi\.org/(10\.\d+/[^\s]+)',
            r'https?://dx\.doi\.org/(10\.\d+/[^\s]+)',
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, citation, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_pmid(self, citation: str) -> Optional[str]:
        """Extract PMID from citation"""
        
        pmid_patterns = [
            r'PMID:\s*(\d+)',
            r'PubMed:\s*(\d+)',
            r'pmid\s*(\d+)',
        ]
        
        for pattern in pmid_patterns:
            match = re.search(pattern, citation, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _validate_citation_info(self, citation_info: CitationInfo) -> bool:
        """Validate citation information"""
        
        issues = []
        
        # Check required fields
        if not citation_info.authors:
            issues.append("Missing author information")
        
        if not citation_info.title:
            issues.append("Missing title")
        
        if not citation_info.journal:
            issues.append("Missing journal name")
        
        if not citation_info.year:
            issues.append("Missing publication year")
        elif citation_info.year < 1800 or citation_info.year > datetime.now().year + 1:
            issues.append(f"Invalid publication year: {citation_info.year}")
        
        # Check author format
        for author in citation_info.authors:
            if len(author) < 2:
                issues.append(f"Author name too short: {author}")
            elif not re.search(r'[A-Za-z]', author):
                issues.append(f"Author name contains no letters: {author}")
        
        # Check title format
        if citation_info.title and len(citation_info.title) < 5:
            issues.append("Title too short")
        
        citation_info.issues = issues
        
        return len(issues) == 0
    
    def validate_citations(self, citations: List[str]) -> ValidationResult:
        """Validate a list of citations"""
        
        validation_result = ValidationResult(
            total_citations=len(citations),
            valid_citations=0,
            invalid_citations=0,
            missing_info=0,
            duplicate_citations=0,
            style_inconsistencies=0,
            issues=[],
            suggestions=[]
        )
        
        if not citations:
            validation_result.issues.append("No citations found")
            return validation_result
        
        citation_styles = []
        parsed_citations = []
        citation_texts = []
        
        for i, citation in enumerate(citations):
            try:
                parsed = self.parse_citation(citation)
                parsed_citations.append(parsed)
                citation_styles.append(parsed.citation_style)
                citation_texts.append(citation.lower().strip())
                
                if parsed.is_valid:
                    validation_result.valid_citations += 1
                else:
                    validation_result.invalid_citations += 1
                    validation_result.issues.extend([f"Citation {i+1}: {issue}" for issue in parsed.issues])
                
                # Check for missing information
                missing_fields = []
                if not parsed.authors:
                    missing_fields.append("authors")
                if not parsed.title:
                    missing_fields.append("title")
                if not parsed.journal:
                    missing_fields.append("journal")
                if not parsed.year:
                    missing_fields.append("year")
                
                if missing_fields:
                    validation_result.missing_info += 1
                    validation_result.issues.append(f"Citation {i+1}: Missing {', '.join(missing_fields)}")
                
            except Exception as e:
                validation_result.invalid_citations += 1
                validation_result.issues.append(f"Citation {i+1}: Error parsing citation - {str(e)}")
        
        # Check for duplicates
        for i in range(len(citation_texts)):
            for j in range(i+1, len(citation_texts)):
                similarity = difflib.SequenceMatcher(None, citation_texts[i], citation_texts[j]).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    validation_result.duplicate_citations += 1
                    validation_result.issues.append(f"Potential duplicate: Citations {i+1} and {j+1}")
        
        # Check style consistency
        unique_styles = set(style for style in citation_styles if style != 'Unknown')
        if len(unique_styles) > 1:
            validation_result.style_inconsistencies = len(citations) - citation_styles.count(max(unique_styles, key=citation_styles.count))
            validation_result.issues.append(f"Multiple citation styles detected: {', '.join(unique_styles)}")
        
        # Generate suggestions
        if validation_result.style_inconsistencies > 0:
            most_common_style = max(unique_styles, key=citation_styles.count) if unique_styles else "APA"
            validation_result.suggestions.append(f"Consider standardizing all citations to {most_common_style} style")
        
        if validation_result.missing_info > 0:
            validation_result.suggestions.append("Complete missing citation information (authors, titles, journals, years)")
        
        if validation_result.duplicate_citations > 0:
            validation_result.suggestions.append("Remove or consolidate duplicate citations")
        
        return validation_result
    
    def check_citation_accuracy(self, citation_info: CitationInfo) -> Dict[str, Any]:
        """Check citation accuracy against external databases"""
        
        accuracy_result = {
            'verified': False,
            'source': None,
            'matches': [],
            'discrepancies': [],
            'confidence': 0.0
        }
        
        # Check DOI first if available
        if citation_info.doi:
            crossref_result = self._check_crossref(citation_info.doi)
            if crossref_result:
                accuracy_result.update(crossref_result)
                accuracy_result['source'] = 'CrossRef'
                return accuracy_result
        
        # Check PubMed if PMID available
        if citation_info.pmid:
            pubmed_result = self._check_pubmed(citation_info.pmid)
            if pubmed_result:
                accuracy_result.update(pubmed_result)
                accuracy_result['source'] = 'PubMed'
                return accuracy_result
        
        # Search by title and authors
        if citation_info.title and citation_info.authors:
            search_result = self._search_citation(citation_info)
            if search_result:
                accuracy_result.update(search_result)
        
        return accuracy_result
    
    def _check_crossref(self, doi: str) -> Optional[Dict[str, Any]]:
        """Check citation against CrossRef database"""
        
        try:
            # Rate limiting
            self._rate_limit('crossref')
            
            url = f"https://api.crossref.org/works/{doi}"
            headers = {'User-Agent': 'Academic Research Platform (mailto:contact@example.com)'}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                work = data.get('message', {})
                
                return {
                    'verified': True,
                    'title': work.get('title', [''])[0],
                    'authors': [f"{author.get('given', '')} {author.get('family', '')}" 
                              for author in work.get('author', [])],
                    'journal': work.get('container-title', [''])[0],
                    'year': work.get('published-print', {}).get('date-parts', [[None]])[0][0],
                    'volume': work.get('volume'),
                    'issue': work.get('issue'),
                    'pages': work.get('page'),
                    'confidence': 0.95
                }
            
        except Exception as e:
            pass  # Silently handle API errors
        
        return None
    
    def _check_pubmed(self, pmid: str) -> Optional[Dict[str, Any]]:
        """Check citation against PubMed database"""
        
        try:
            # Rate limiting
            self._rate_limit('pubmed')
            
            # Use PubMed E-utilities API
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {}).get(pmid, {})
                
                if result:
                    return {
                        'verified': True,
                        'title': result.get('title', ''),
                        'authors': [author.get('name', '') for author in result.get('authors', [])],
                        'journal': result.get('source', ''),
                        'year': int(result.get('pubdate', '').split()[0]) if result.get('pubdate') else None,
                        'confidence': 0.9
                    }
            
        except Exception as e:
            pass  # Silently handle API errors
        
        return None
    
    def _search_citation(self, citation_info: CitationInfo) -> Optional[Dict[str, Any]]:
        """Search for citation by title and authors"""
        
        # This would implement a search across multiple databases
        # For now, return a placeholder result
        
        confidence = 0.0
        
        # Calculate confidence based on available information
        if citation_info.title:
            confidence += 0.3
        if citation_info.authors:
            confidence += 0.3
        if citation_info.journal:
            confidence += 0.2
        if citation_info.year:
            confidence += 0.2
        
        return {
            'verified': False,
            'confidence': confidence,
            'matches': [],
            'note': 'Automated verification not available without DOI/PMID'
        }
    
    def _rate_limit(self, api_name: str):
        """Implement rate limiting for API calls"""
        
        current_time = time.time()
        last_call = self.last_api_call.get(api_name, 0)
        min_interval = self.api_delays.get(api_name, 1.0)
        
        if current_time - last_call < min_interval:
            time.sleep(min_interval - (current_time - last_call))
        
        self.last_api_call[api_name] = time.time()
    
    def format_citation(self, citation_info: CitationInfo, target_style: str) -> str:
        """Format citation according to specified style"""
        
        if not citation_info.is_valid:
            return citation_info.original_text
        
        authors_str = self._format_authors(citation_info.authors, target_style)
        title_str = citation_info.title
        journal_str = citation_info.journal
        year_str = str(citation_info.year) if citation_info.year else ""
        
        if target_style == 'APA':
            citation = f"{authors_str} ({year_str}). {title_str}. {journal_str}"
            if citation_info.volume:
                citation += f", {citation_info.volume}"
                if citation_info.issue:
                    citation += f"({citation_info.issue})"
            if citation_info.pages:
                citation += f", {citation_info.pages}"
            citation += "."
            
        elif target_style == 'MLA':
            citation = f"{authors_str} \"{title_str}.\" {journal_str}"
            if citation_info.volume:
                citation += f", vol. {citation_info.volume}"
                if citation_info.issue:
                    citation += f", no. {citation_info.issue}"
            citation += f", {year_str}"
            if citation_info.pages:
                citation += f", {citation_info.pages}"
            citation += "."
            
        elif target_style == 'Vancouver':
            citation = f"{authors_str} {title_str}. {journal_str}. {year_str}"
            if citation_info.volume:
                citation += f";{citation_info.volume}"
                if citation_info.issue:
                    citation += f"({citation_info.issue})"
            if citation_info.pages:
                citation += f":{citation_info.pages}"
            citation += "."
            
        else:  # Default to original
            citation = citation_info.original_text
        
        return citation
    
    def _format_authors(self, authors: List[str], style: str) -> str:
        """Format authors according to citation style"""
        
        if not authors:
            return ""
        
        if style == 'APA':
            if len(authors) == 1:
                return authors[0]
            elif len(authors) <= 7:
                return ", ".join(authors[:-1]) + ", & " + authors[-1]
            else:
                return ", ".join(authors[:6]) + ", ... " + authors[-1]
                
        elif style == 'MLA':
            if len(authors) == 1:
                return authors[0]
            else:
                return authors[0] + ", et al."
                
        elif style == 'Vancouver':
            if len(authors) <= 6:
                return ", ".join(authors) + "."
            else:
                return ", ".join(authors[:3]) + ", et al."
        
        else:
            return ", ".join(authors)
    
    def generate_bibliography(self, citations: List[CitationInfo], style: str = 'APA') -> str:
        """Generate formatted bibliography"""
        
        bibliography = f"References ({style} Style)\n"
        bibliography += "=" * 50 + "\n\n"
        
        # Sort citations (usually by author last name or chronologically)
        sorted_citations = sorted(citations, key=lambda x: (x.authors[0] if x.authors else "", x.year or 0))
        
        for i, citation in enumerate(sorted_citations, 1):
            formatted = self.format_citation(citation, style)
            
            if style == 'Vancouver':
                bibliography += f"{i}. {formatted}\n\n"
            else:
                bibliography += f"{formatted}\n\n"
        
        return bibliography
    
    def check_citation_completeness(self, text: str) -> Dict[str, Any]:
        """Check completeness of citations in text"""
        
        # Find in-text citations
        in_text_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\[[^\]]*\d+[^\]]*\]',  # [1], [Author, 2023]
        ]
        
        in_text_citations = []
        for pattern in in_text_patterns:
            matches = re.findall(pattern, text)
            in_text_citations.extend(matches)
        
        # Find reference list
        references = self.extract_citations(text)
        
        # Analyze completeness
        result = {
            'in_text_citations': len(in_text_citations),
            'reference_list_entries': len(references),
            'potential_missing_references': [],
            'potential_orphaned_references': [],
            'completeness_score': 0.0
        }
        
        # Basic completeness check
        if result['in_text_citations'] > 0 and result['reference_list_entries'] > 0:
            ratio = min(result['in_text_citations'], result['reference_list_entries']) / max(result['in_text_citations'], result['reference_list_entries'])
            result['completeness_score'] = ratio
        
        return result
