"""
Data Citations and Attribution Module
Provides proper citations for all data sources used in the Academic Research Platform
"""

def get_data_citations():
    """Return comprehensive citations for all data sources"""
    citations = {
        "uniprot": {
            "citation": "The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2023. Nucleic Acids Res. 2023;51(D1):D523-D531.",
            "doi": "10.1093/nar/gkac1052",
            "url": "https://www.uniprot.org/",
            "description": "Protein sequence and functional information"
        },
        
        "pubchem": {
            "citation": "Kim S, Chen J, Cheng T, et al. PubChem 2023 update. Nucleic Acids Res. 2023;51(D1):D1373-D1380.",
            "doi": "10.1093/nar/gkac956", 
            "url": "https://pubchem.ncbi.nlm.nih.gov/",
            "description": "Chemical compound properties and structures"
        },
        
        "drugbank": {
            "citation": "Wishart DS, Feunang YD, Guo AC, et al. DrugBank 5.0: a major update to the DrugBank database for 2018. Nucleic Acids Res. 2018;46(D1):D1074-D1082.",
            "doi": "10.1093/nar/gkx1037",
            "url": "https://go.drugbank.com/",
            "description": "Drug and drug target information"
        },
        
        "chembl": {
            "citation": "Mendez D, Gaulton A, Bento AP, et al. ChEMBL: towards direct deposition of bioassay data. Nucleic Acids Res. 2019;47(D1):D930-D940.",
            "doi": "10.1093/nar/gky1075",
            "url": "https://www.ebi.ac.uk/chembl/",
            "description": "Bioactivity data for drug discovery"
        },
        
        "pdb": {
            "citation": "Burley SK, Bhikadiya C, Bi C, et al. RCSB Protein Data Bank: powerful new tools for exploring 3D structures of biological macromolecules for basic and applied research and education. Nucleic Acids Res. 2021;49(D1):D437-D451.",
            "doi": "10.1093/nar/gkaa1038",
            "url": "https://www.rcsb.org/",
            "description": "3D protein structure data"
        },
        
        "pubmed": {
            "citation": "NCBI Resource Coordinators. Database Resources of the National Center for Biotechnology Information. Nucleic Acids Res. 2018;46(D1):D8-D13.",
            "doi": "10.1093/nar/gkx1095",
            "url": "https://pubmed.ncbi.nlm.nih.gov/",
            "description": "Biomedical literature database"
        },
        
        "crossref": {
            "citation": "Hendricks G, Tkaczyk D, Lin J, Feeney P. Crossref: The sustainable source of community-owned scholarly metadata. Quantitative Science Studies. 2020;1(1):414-427.",
            "doi": "10.1162/qss_a_00022",
            "url": "https://www.crossref.org/",
            "description": "Citation linking and metadata"
        },
        
        "cyp450_data": {
            "citation": "Zanger UM, Schwab M. Cytochrome P450 enzymes in drug metabolism: regulation of gene expression, enzyme activities, and impact of genetic variation. Pharmacol Ther. 2013;138(1):103-141.",
            "doi": "10.1016/j.pharmthera.2012.12.007",
            "url": "https://www.pharmvar.org/",
            "description": "Cytochrome P450 enzyme data and pharmacogenomics"
        },
        
        "admet_data": {
            "citation": "Daina A, Michielin O, Zoete V. SwissADME: a free web tool to evaluate pharmacokinetics, drug-likeness and medicinal chemistry friendliness of small molecules. Sci Rep. 2017;7:42717.",
            "doi": "10.1038/srep42717",
            "url": "http://www.swissadme.ch/",
            "description": "ADMET prediction and drug-likeness assessment"
        },
        
        "lipinski_rules": {
            "citation": "Lipinski CA, Lombardo F, Dominy BW, Feeney PJ. Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. Adv Drug Deliv Rev. 2001;46(1-3):3-26.",
            "doi": "10.1016/s0169-409x(00)00129-0",
            "url": "https://doi.org/10.1016/s0169-409x(00)00129-0",
            "description": "Rule of Five for drug-likeness"
        }
    }
    
    return citations

def get_computational_methods_citations():
    """Citations for computational methods and algorithms used"""
    methods_citations = {
        "sympy": {
            "citation": "Meurer A, Smith CP, Paprocki M, et al. SymPy: symbolic computing in Python. PeerJ Computer Science. 2017;3:e103.",
            "doi": "10.7717/peerj-cs.103",
            "url": "https://www.sympy.org/",
            "description": "Symbolic mathematics computations"
        },
        
        "scipy": {
            "citation": "Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods. 2020;17(3):261-272.",
            "doi": "10.1038/s41592-019-0686-2",
            "url": "https://scipy.org/",
            "description": "Scientific computing algorithms"
        },
        
        "rdkit": {
            "citation": "Landrum G. RDKit: Open-source cheminformatics. http://www.rdkit.org",
            "url": "http://www.rdkit.org",
            "description": "Chemical informatics and molecular property calculations"
        },
        
        "py3dmol": {
            "citation": "Rego N, Koes D. 3Dmol.js: molecular visualization with WebGL. Bioinformatics. 2015;31(8):1322-1324.",
            "doi": "10.1093/bioinformatics/btu829",
            "url": "https://3dmol.csb.pitt.edu/",
            "description": "3D molecular visualization"
        },
        
        "networkx": {
            "citation": "Hagberg AA, Schult DA, Swart PJ. Exploring Network Structure, Dynamics, and Function using NetworkX. In: Proceedings of the 7th Python in Science Conference. 2008:11-15.",
            "url": "https://networkx.org/",
            "description": "Network analysis and graph algorithms"
        },
        
        "plotly": {
            "citation": "Plotly Technologies Inc. Collaborative data science. Montreal, QC: Plotly Technologies Inc.; 2015.",
            "url": "https://plot.ly",
            "description": "Interactive data visualization"
        }
    }
    
    return methods_citations

def get_pharmacological_models_citations():
    """Citations for pharmacological models and equations used"""
    model_citations = {
        "hill_equation": {
            "citation": "Hill AV. The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves. J Physiol. 1910;40(Suppl):iv-vii.",
            "description": "Hill equation for dose-response relationships"
        },
        
        "pk_modeling": {
            "citation": "Gabrielsson J, Weiner D. Pharmacokinetic and Pharmacodynamic Data Analysis: Concepts and Applications. 5th ed. Swedish Academy of Pharmaceutical Sciences; 2016.",
            "description": "Pharmacokinetic modeling principles"
        },
        
        "pbpk_modeling": {
            "citation": "Jones HM, Rowland-Yeo K. Basic concepts in physiologically based pharmacokinetic modeling in drug discovery and development. CPT Pharmacometrics Syst Pharmacol. 2013;2(8):e63.",
            "doi": "10.1038/psp.2013.41",
            "description": "Physiologically-based pharmacokinetic modeling"
        },
        
        "competitive_inhibition": {
            "citation": "Michaelis L, Menten ML. Die kinetik der invertinwirkung. Biochem Z. 1913;49:333-369.",
            "description": "Michaelis-Menten kinetics and competitive inhibition"
        },
        
        "allometric_scaling": {
            "citation": "Mahmood I. Application of allometric principles for the prediction of pharmacokinetics in human and veterinary drug development. Adv Drug Deliv Rev. 2007;59(11):1177-1192.",
            "doi": "10.1016/j.addr.2007.05.001",
            "description": "Allometric scaling in pharmacokinetics"
        }
    }
    
    return model_citations

def format_citation_text(citations_dict):
    """Format citations as text for display"""
    citation_text = "## Data Sources and Citations\n\n"
    
    for key, citation_info in citations_dict.items():
        citation_text += f"**{citation_info.get('description', key)}:**\n"
        citation_text += f"{citation_info['citation']}\n"
        
        if 'doi' in citation_info:
            citation_text += f"DOI: {citation_info['doi']}\n"
        if 'url' in citation_info:
            citation_text += f"URL: {citation_info['url']}\n"
        citation_text += "\n"
    
    return citation_text

def get_complete_bibliography():
    """Get complete bibliography for the platform"""
    all_citations = {}
    all_citations.update(get_data_citations())
    all_citations.update(get_computational_methods_citations()) 
    all_citations.update(get_pharmacological_models_citations())
    
    return all_citations

def get_citation_footer():
    """Get standardized citation footer for all pages"""
    footer = """
---
**Data Attribution Notice:** This platform integrates data from multiple authoritative sources including UniProt, PubChem, DrugBank, ChEMBL, and PDB. All data sources are properly cited and used in accordance with their respective terms of service. Computational methods utilize established algorithms from peer-reviewed literature. For complete citations, see the Data Sources section.

**Academic Use:** This platform is designed for academic and research purposes. Users should cite original data sources when publishing results derived from this platform.
"""
    return footer