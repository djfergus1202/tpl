import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import mathematical engine and drug database
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mathematical_engine import MathematicalEngine
from utils.drug_database import DrugDatabase
from utils.data_citations import get_data_citations, get_computational_methods_citations, get_pharmacological_models_citations, get_citation_footer
from utils.teratrend_analyzer import TeratrendAnalyzer, create_teratrend_visualizations
from utils.literature_analyzer import LiteratureAnalyzer

st.set_page_config(page_title="Pharmacological Topological Maps", page_icon="🧬", layout="wide")

class PharmacologicalMapper:
    """Generate 3D pharmacological topological maps from multiomic data"""
    
    def __init__(self):
        self.uniprot_base_url = "https://rest.uniprot.org"
        self.chembl_base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()
        
    def fetch_uniprot_data(self, protein_ids: List[str], include_features: bool = True) -> Dict[str, Any]:
        """Fetch protein data from UniProt"""
        
        protein_data = {}
        
        for protein_id in protein_ids:
            try:
                # Fetch basic protein information
                url = f"{self.uniprot_base_url}/uniprotkb/{protein_id}.json"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    protein_info = {
                        'id': protein_id,
                        'name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown'),
                        'gene_name': data.get('genes', [{}])[0].get('geneName', {}).get('value', 'Unknown') if data.get('genes') else 'Unknown',
                        'organism': data.get('organism', {}).get('scientificName', 'Unknown'),
                        'length': data.get('sequence', {}).get('length', 0),
                        'mass': data.get('sequence', {}).get('molWeight', 0),
                        'function': self._extract_function(data),
                        'pathways': self._extract_pathways(data),
                        'domains': self._extract_domains(data),
                        'modifications': self._extract_modifications(data),
                        'interactions': self._extract_interactions(data),
                        'expression_data': self._generate_expression_profile(),
                        'binding_sites': self._extract_binding_sites(data)
                    }
                    
                    protein_data[protein_id] = protein_info
                    
                else:
                    st.warning(f"Failed to fetch data for protein {protein_id}")
                    
            except Exception as e:
                st.error(f"Error fetching UniProt data for {protein_id}: {str(e)}")
        
        return protein_data
    
    def _extract_function(self, data: Dict) -> str:
        """Extract protein function from UniProt data"""
        comments = data.get('comments', [])
        for comment in comments:
            if comment.get('commentType') == 'FUNCTION':
                return comment.get('texts', [{}])[0].get('value', '')
        return 'Function not specified'
    
    def _extract_pathways(self, data: Dict) -> List[str]:
        """Extract pathway information"""
        pathways = []
        db_refs = data.get('dbReferences', [])
        for ref in db_refs:
            if ref.get('type') in ['KEGG', 'Reactome', 'BioCyc']:
                pathways.append(ref.get('id', ''))
        return pathways[:5]  # Limit to top 5
    
    def _extract_domains(self, data: Dict) -> List[Dict]:
        """Extract protein domains"""
        domains = []
        features = data.get('features', [])
        for feature in features:
            if feature.get('type') == 'Domain':
                domains.append({
                    'name': feature.get('description', ''),
                    'start': feature.get('location', {}).get('start', {}).get('value', 0),
                    'end': feature.get('location', {}).get('end', {}).get('value', 0)
                })
        return domains[:10]  # Limit to top 10
    
    def _extract_modifications(self, data: Dict) -> List[Dict]:
        """Extract post-translational modifications"""
        modifications = []
        features = data.get('features', [])
        for feature in features:
            if feature.get('type') in ['Modified residue', 'Glycosylation', 'Phosphorylation']:
                modifications.append({
                    'type': feature.get('type', ''),
                    'position': feature.get('location', {}).get('start', {}).get('value', 0),
                    'description': feature.get('description', '')
                })
        return modifications[:15]  # Limit to top 15
    
    def _extract_interactions(self, data: Dict) -> int:
        """Extract number of protein interactions"""
        db_refs = data.get('dbReferences', [])
        interactions = 0
        for ref in db_refs:
            if ref.get('type') in ['STRING', 'IntAct']:
                interactions += 1
        return interactions
    
    def _extract_binding_sites(self, data: Dict) -> List[Dict]:
        """Extract binding sites information"""
        binding_sites = []
        features = data.get('features', [])
        for feature in features:
            if feature.get('type') == 'Binding site':
                binding_sites.append({
                    'description': feature.get('description', ''),
                    'position': feature.get('location', {}).get('start', {}).get('value', 0),
                    'ligand': feature.get('ligand', {}).get('name', '') if feature.get('ligand') else ''
                })
        return binding_sites[:10]  # Limit to top 10
    
    def _generate_expression_profile(self) -> Dict[str, float]:
        """Generate tissue expression profile (simulated from typical ranges)"""
        tissues = ['brain', 'liver', 'kidney', 'heart', 'lung', 'muscle', 'skin', 'blood']
        # Generate realistic expression values (FPKM-like)
        expression = {}
        for tissue in tissues:
            expression[tissue] = np.random.lognormal(2, 1.5)  # Log-normal distribution typical of expression
        return expression
    
    def generate_drug_response_data(self, proteins: Dict[str, Any], n_compounds: int = 300, 
                                  drug_names: List[str] = None, drug_class: str = None) -> pd.DataFrame:
        """Generate drug response data based on protein characteristics - optimized for large datasets"""
        
        drug_data = []
        total_compounds = len(proteins) * n_compounds
        
        # Progress tracking for large datasets
        if total_compounds > 1000:
            st.info(f"Generating data for {total_compounds} compound-protein combinations...")
            progress_bar = st.progress(0)
            progress_counter = 0
        
        for protein_idx, (protein_id, protein_info) in enumerate(proteins.items()):
            # Pre-calculate protein-specific factors for efficiency
            protein_factor = self._calculate_protein_binding_factor(protein_info)
            
            # Generate compound batches for better performance
            batch_size = min(50, n_compounds)
            for batch_start in range(0, n_compounds, batch_size):
                batch_end = min(batch_start + batch_size, n_compounds)
                batch_data = self._generate_compound_batch(
                    protein_id, protein_info, protein_factor, 
                    batch_start, batch_end
                )
                drug_data.extend(batch_data)
                
                # Update progress for large datasets
                if total_compounds > 1000:
                    progress_counter += (batch_end - batch_start)
                    progress_bar.progress(progress_counter / total_compounds)
        
        if total_compounds > 1000:
            progress_bar.empty()
        
        return pd.DataFrame(drug_data)
    
    def generate_drug_response_data_with_params(self, protein_data: List[Dict], n_compounds: int, 
                                             drug_names: List[str] = None, drug_class: str = None,
                                             interactive_params: Dict = None) -> pd.DataFrame:
        """
        Generate drug response data using interactive parameters for full customization
        """
        if interactive_params is None:
            return self.generate_drug_response_data(protein_data, n_compounds, drug_names, drug_class)
            
        # Extract interactive parameters
        time_start = interactive_params.get('time_start', 0.0)
        time_end = interactive_params.get('time_end', 24.0)
        time_points_count = int(interactive_params.get('time_points', 50))
        genomic_hits = interactive_params.get('genomic_hits', 100)
        mutation_rate = interactive_params.get('mutation_rate', 15.0) / 100.0
        dose_response_curve = interactive_params.get('dose_response_curve', 'Sigmoid')
        binding_affinity_range = interactive_params.get('binding_affinity_range', (1.0, 100.0))
        protein_expression = interactive_params.get('protein_expression', 1.0)
        tissue_specificity = interactive_params.get('tissue_specificity', 'Generic')
        
        np.random.seed(42)
        data_points = []
        
        # Custom time array
        time_array = np.linspace(time_start, time_end, time_points_count)
        
        for protein_id, protein in protein_data.items():
            protein_name = protein.get('name', protein_id)
            
            # Apply tissue-specific protein expression modulation
            tissue_modulation = {
                'Liver': {'CYP': 1.5, 'UGT': 1.3, 'default': 0.9},
                'Brain': {'GABA': 1.4, 'DA': 1.2, 'default': 0.8},
                'Heart': {'ADRB': 1.3, 'SCN': 1.2, 'default': 0.9},
                'Kidney': {'ACE': 1.4, 'AT1': 1.2, 'default': 0.9},
                'Lung': {'ADRB': 1.2, 'H1': 1.1, 'default': 0.9},
                'Generic': {'default': 1.0}
            }.get(tissue_specificity, {'default': 1.0})
            
            expression_factor = protein_expression
            for key in tissue_modulation:
                if key != 'default' and key in protein_name:
                    expression_factor *= tissue_modulation[key]
                    break
            else:
                expression_factor *= tissue_modulation.get('default', 1.0)
            
            for compound_idx in range(n_compounds):
                # Use drug names if provided
                if drug_names and compound_idx < len(drug_names):
                    compound_name = drug_names[compound_idx]
                    compound_id = f"DRUG_{compound_idx:03d}"
                else:
                    compound_name = f"Compound_{compound_idx:03d}"
                    compound_id = f"CMPD_{compound_idx:03d}"
                
                # Generate molecular properties with binding affinity constraints
                ki_nm = np.random.uniform(binding_affinity_range[0], binding_affinity_range[1])
                
                # Implement different dose-response curves
                for time_idx, time_hours in enumerate(time_array):
                    # Base drug response calculation
                    log_agonist = np.log10(ki_nm + np.random.uniform(0.1, 10))
                    
                    # Apply dose-response curve
                    if dose_response_curve == 'Sigmoid':
                        base_response = 1 / (1 + np.exp(-(log_agonist - 1.5)))
                    elif dose_response_curve == 'Linear':
                        base_response = np.clip(0.1 * log_agonist, 0, 1)
                    elif dose_response_curve == 'Exponential':
                        base_response = 1 - np.exp(-log_agonist / 2)
                    elif dose_response_curve == 'Biphasic':
                        base_response = 0.7 * (1 / (1 + np.exp(-(log_agonist - 1)))) + 0.3 * (1 / (1 + np.exp(-(log_agonist - 3))))
                    
                    # Time-dependent response
                    if time_hours == 0:
                        time_factor = 0.1
                    else:
                        time_factor = 1 - np.exp(-time_hours / 8)  # 8-hour half-life
                    
                    # Apply protein expression factor
                    drug_response = base_response * time_factor * expression_factor
                    drug_response = np.clip(drug_response + np.random.normal(0, 0.05), 0, 1)
                    
                    # Enhanced genomic simulation
                    genomic_score = 0
                    for _ in range(int(genomic_hits)):
                        if np.random.random() < mutation_rate:
                            # Mutation present - affects drug response
                            mutation_effect = np.random.uniform(-0.3, 0.3)
                            genomic_score += mutation_effect
                    
                    genomic_score = genomic_score / genomic_hits  # Normalize
                    genomic_modified_response = np.clip(drug_response + genomic_score, 0, 1)
                    
                    # Calculate enhanced therapeutic window
                    efficacy = genomic_modified_response
                    safety = 1 - abs(genomic_score) * 2  # Genetic variants affect safety
                    selectivity = np.random.beta(2, 2)
                    
                    therapeutic_window = (efficacy * 0.4 + safety * 0.4 + selectivity * 0.2)
                    
                    data_point = {
                        'protein_id': protein_id,
                        'protein_name': protein_name,
                        'compound_id': compound_id,
                        'compound_name': compound_name,
                        'time_hours': time_hours,
                        'log_agonist': log_agonist,
                        'drug_response': genomic_modified_response,
                        'ki_nm': ki_nm,
                        'therapeutic_window': therapeutic_window,
                        'genomic_hits': genomic_hits,
                        'mutation_rate': mutation_rate,
                        'tissue_expression': expression_factor,
                        'dose_response_model': dose_response_curve,
                        'efficacy': efficacy,
                        'safety': safety,
                        'selectivity': selectivity,
                        'genomic_effect': genomic_score,
                        'molecular_weight': np.random.normal(350, 100),
                        'logp': np.random.normal(2.5, 1.5),
                        'drug_likeness': np.random.beta(3, 2)
                    }
                    
                    data_points.append(data_point)
        
        return pd.DataFrame(data_points)
    
    def _calculate_protein_binding_factor(self, protein_info: Dict[str, Any]) -> float:
        """Pre-calculate protein-specific binding factors for efficiency"""
        protein_factor = 1.0
        if len(protein_info['binding_sites']) > 5:
            protein_factor *= 0.7  # Better binding if more sites
        if protein_info['mass'] > 50000:
            protein_factor *= 1.3  # Larger proteins harder to bind
        if len(protein_info['domains']) > 3:
            protein_factor *= 0.9  # More domains may facilitate binding
        return protein_factor
    
    def _generate_compound_batch(self, protein_id: str, protein_info: Dict[str, Any], 
                               protein_factor: float, batch_start: int, batch_end: int) -> List[Dict]:
        """Generate a batch of compounds for better memory efficiency"""
        
        batch_data = []
        
        # Pre-generate molecular properties arrays for efficiency
        n_batch = batch_end - batch_start
        mw_array = np.random.normal(350, 100, n_batch)
        logp_array = np.random.normal(2.5, 1.5, n_batch)
        hbd_array = np.random.poisson(2, n_batch)
        hba_array = np.random.poisson(4, n_batch)
        tpsa_array = np.random.normal(80, 30, n_batch)
        
        # Pre-calculate time points
        time_points = np.array([0.5, 1, 2, 4, 8, 12, 24, 48])  # hours
        
        for i in range(n_batch):
            compound_idx = batch_start + i
            # Use drug names if provided, otherwise generate compound IDs
            if drug_names and compound_idx < len(drug_names):
                compound_id = drug_names[compound_idx]
                compound_name = drug_names[compound_idx]
            else:
                compound_id = f"CMPD_{protein_id}_{compound_idx:04d}"
                compound_name = f"Compound {compound_idx+1}"
            
            # Use pre-generated molecular properties
            mw = mw_array[i]
            logp = logp_array[i]
            hbd = hbd_array[i]
            hba = hba_array[i]
            tpsa = tpsa_array[i]
            
            # Calculate drug-likeness score
            drug_likeness = self._calculate_drug_likeness(mw, logp, hbd, hba, tpsa)
            
            # Generate binding affinity (Ki, nM)
            base_affinity = np.random.lognormal(5, 2)
            ki_nm = base_affinity * protein_factor
            
            # Generate pharmacokinetic parameters
            kabs = np.random.uniform(0.1, 2.0)
            kel = np.random.uniform(0.05, 0.5)
            hill_coeff = np.random.uniform(0.8, 2.0)
            ic50 = ki_nm * np.random.uniform(0.5, 2.0)
            max_response = np.random.uniform(0.7, 1.0)
            
            # Generate time-dependent response data
            for t in time_points:
                # Calculate drug concentration over time (one-compartment model)
                conc = (kabs / (kabs - kel)) * (np.exp(-kel * t) - np.exp(-kabs * t))
                
                # Calculate response based on Hill equation
                response = max_response * (conc ** hill_coeff) / (ic50 ** hill_coeff + conc ** hill_coeff)
                
                # Add noise
                response += np.random.normal(0, 0.05)
                response = max(0, min(1, response))  # Clamp to [0, 1]
                
                # Calculate log(agonist) concentration
                log_agonist = np.log10(max(conc, 1e-9))
                
                batch_data.append({
                    'protein_id': protein_id,
                    'protein_name': protein_info['name'],
                    'gene_name': protein_info['gene_name'],
                    'compound_id': compound_id,
                    'compound_name': compound_name,
                    'drug_class': drug_class if drug_class else 'Unknown',
                    'molecular_weight': mw,
                    'logp': logp,
                    'hbd': hbd,
                    'hba': hba,
                    'tpsa': tpsa,
                    'drug_likeness': drug_likeness,
                    'ki_nm': ki_nm,
                    'ic50_nm': ic50,
                    'hill_coefficient': hill_coeff,
                    'time_hours': t,
                    'concentration_um': conc,
                    'log_agonist': log_agonist,
                    'drug_response': response,
                    'therapeutic_window': self._calculate_therapeutic_window(response, conc),
                    'selectivity_index': np.random.uniform(1, 100),
                    'toxicity_score': np.random.uniform(0, 1),
                    'protein_mass': protein_info['mass'],
                    'protein_length': protein_info['length'],
                    'binding_sites_count': len(protein_info['binding_sites']),
                    'domain_count': len(protein_info['domains']),
                    'pathway_count': len(protein_info['pathways'])
                })
        
        return batch_data
    
    def _calculate_drug_likeness(self, mw: float, logp: float, hbd: int, hba: int, tpsa: float) -> float:
        """Calculate drug-likeness score based on Lipinski's rule of five and other factors"""
        
        score = 1.0
        
        # Lipinski's Rule of Five violations
        if mw > 500:
            score -= 0.2
        if logp > 5:
            score -= 0.2
        if hbd > 5:
            score -= 0.2
        if hba > 10:
            score -= 0.2
        
        # TPSA constraint
        if tpsa > 140:
            score -= 0.1
        
        # Additional favorable characteristics
        if 200 < mw < 400:
            score += 0.1
        if 1 < logp < 3:
            score += 0.1
        if 60 < tpsa < 90:
            score += 0.1
        
        return max(0, min(1, score))
    
    def _calculate_therapeutic_window(self, response: float, concentration: float) -> float:
        """Calculate therapeutic window score"""
        
        # Optimal response range is typically 0.5-0.8
        if 0.5 <= response <= 0.8:
            window_score = 1.0
        elif response < 0.5:
            window_score = response / 0.5
        else:
            window_score = max(0, 1 - (response - 0.8) / 0.2)
        
        # Adjust for concentration (favor lower concentrations)
        conc_factor = 1 / (1 + concentration / 10)  # Prefer concentrations < 10 µM
        
        return window_score * conc_factor
    
    def create_3d_topological_map(self, df: pd.DataFrame, protein_filter: str = None, 
                                 sample_size: int = 3000) -> go.Figure:
        """Create 3D topological map of drug response vs log(agonist) vs time - optimized for large datasets"""
        
        # Filter data if protein specified
        if protein_filter:
            df_filtered = df[df['protein_id'] == protein_filter].copy()
        else:
            df_filtered = df.copy()
        
        # Sample data for visualization if dataset is large
        if len(df_filtered) > sample_size:
            # Stratified sampling to maintain distribution
            df_sample = df_filtered.sample(n=sample_size, random_state=42)
            st.info(f"Displaying {sample_size:,} sampled points from {len(df_filtered):,} total points for performance")
        else:
            df_sample = df_filtered.copy()
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Color by therapeutic window with optimized hover text
        fig.add_trace(go.Scatter3d(
            x=df_sample['log_agonist'],
            y=df_sample['drug_response'],
            z=df_sample['time_hours'],
            mode='markers',
            marker=dict(
                size=6,  # Slightly smaller for better performance with many points
                color=df_sample['therapeutic_window'],
                colorscale='Viridis',
                colorbar=dict(
                    title="Therapeutic<br>Window Score"
                ),
                opacity=0.7,  # Slightly more transparent for overlapping points
                line=dict(width=0.3, color='black')
            ),
            text=[f"ID: {cid[-8:]}<br>{pname}<br>Resp: {resp:.3f}<br>Time: {t}h<br>TW: {tw:.3f}" 
                  for cid, pname, resp, t, tw in zip(
                      df_sample['compound_id'], 
                      df_sample['protein_name'],
                      df_sample['drug_response'], 
                      df_sample['time_hours'],
                      df_sample['therapeutic_window']
                  )],
            hovertemplate='%{text}<extra></extra>',
            name=f'Drug Compounds (n={len(df_sample):,})'
        ))
        
        # Add optimal therapeutic zone surface (reduced resolution for performance)
        log_agonist_range = np.linspace(df_sample['log_agonist'].min(), df_sample['log_agonist'].max(), 15)
        time_range = np.linspace(df_sample['time_hours'].min(), df_sample['time_hours'].max(), 15)
        
        # Create meshgrid for optimal response surface
        X, Z = np.meshgrid(log_agonist_range, time_range)
        
        # Define optimal response surface (parabolic with time decay)
        optimal_response = 0.7 * np.exp(-0.02 * Z) * (1 - 0.1 * (X + 2)**2)
        optimal_response = np.clip(optimal_response, 0, 1)
        
        fig.add_trace(go.Surface(
            x=X,
            y=optimal_response,
            z=Z,
            colorscale='Reds',
            opacity=0.3,
            name='Optimal Therapeutic Zone',
            showscale=False
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"3D Pharmacological Topological Map{' - ' + protein_filter if protein_filter else ''}",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="Log(Agonist Concentration)",
                yaxis_title="Drug Response (0-1)",
                zaxis_title="Time (hours)",
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="lightgray"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="lightgray"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="lightgray")
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
    
    def create_therapeutic_window_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create heatmap showing therapeutic windows across time and concentration"""
        
        # Aggregate data by time and log_agonist bins
        df_binned = df.copy()
        df_binned['time_bin'] = pd.cut(df_binned['time_hours'], bins=10, precision=1)
        df_binned['log_agonist_bin'] = pd.cut(df_binned['log_agonist'], bins=15, precision=1)
        
        # Calculate mean therapeutic window for each bin
        heatmap_data = df_binned.groupby(['time_bin', 'log_agonist_bin'])['therapeutic_window'].mean().reset_index()
        
        # Pivot for heatmap
        pivot_data = heatmap_data.pivot(index='time_bin', columns='log_agonist_bin', values='therapeutic_window')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=[f"{interval.left:.1f}-{interval.right:.1f}" for interval in pivot_data.columns],
            y=[f"{interval.left:.1f}-{interval.right:.1f}" for interval in pivot_data.index],
            colorscale='RdYlGn',
            colorbar=dict(title="Therapeutic Window Score"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Therapeutic Window Heatmap",
            xaxis_title="Log(Agonist Concentration) Bins",
            yaxis_title="Time (hours) Bins",
            height=500
        )
        
        return fig
    
    def create_protein_network_map(self, df: pd.DataFrame, max_compounds: int = 100) -> go.Figure:
        """Create network visualization of protein-drug interactions - optimized for large datasets"""
        
        # For large datasets, sample compounds to avoid overcrowding
        if df['compound_id'].nunique() > max_compounds:
            # Sample top compounds by therapeutic window
            top_compounds = (df.groupby('compound_id')['therapeutic_window']
                           .max().sort_values(ascending=False).head(max_compounds))
            df_sampled = df[df['compound_id'].isin(top_compounds.index)]
            st.info(f"Showing network for top {max_compounds} compounds (by therapeutic window) out of {df['compound_id'].nunique():,} total")
        else:
            df_sampled = df
        
        # Create network graph
        G = nx.Graph()
        
        # Add protein nodes
        proteins = df_sampled['protein_id'].unique()
        for protein in proteins:
            protein_data = df_sampled[df_sampled['protein_id'] == protein].iloc[0]
            G.add_node(protein, 
                      node_type='protein',
                      name=protein_data.get('protein_name', protein),
                      mass=protein_data.get('protein_mass', 50.0))
        
        # Add drug compound nodes and edges
        for _, row in df_sampled.iterrows():
            compound = row['compound_id']
            protein = row['protein_id']
            
            if not G.has_node(compound):
                G.add_node(compound, 
                          node_type='compound',
                          drug_likeness=row['drug_likeness'],
                          molecular_weight=row['molecular_weight'])
            
            # Add edge with therapeutic window as weight
            if G.has_edge(protein, compound):
                # Update edge weight with maximum therapeutic window
                G[protein][compound]['weight'] = max(G[protein][compound]['weight'], row['therapeutic_window'])
            else:
                G.add_edge(protein, compound, weight=row['therapeutic_window'])
        
        # Create layout using spring algorithm
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Separate protein and compound nodes
        protein_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'protein']
        compound_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'compound']
        
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*3, color='gray'),
                opacity=0.5,
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Add protein nodes
        protein_x = [pos[node][0] for node in protein_nodes]
        protein_y = [pos[node][1] for node in protein_nodes]
        protein_text = [G.nodes[node]['name'] for node in protein_nodes]
        
        fig.add_trace(go.Scatter(
            x=protein_x,
            y=protein_y,
            mode='markers+text',
            marker=dict(size=20, color='red', symbol='diamond'),
            text=protein_text,
            textposition="middle center",
            name='Proteins',
            hovertemplate='<b>%{text}</b><br>Type: Protein<extra></extra>'
        ))
        
        # Add compound nodes (adaptive sampling based on dataset size)
        sample_step = max(1, len(compound_nodes) // 50)  # Show max 50 compound nodes
        sample_compounds = compound_nodes[::sample_step]
        compound_x = [pos[node][0] for node in sample_compounds]
        compound_y = [pos[node][1] for node in sample_compounds]
        compound_colors = [G.nodes[node]['drug_likeness'] for node in sample_compounds]
        
        fig.add_trace(go.Scatter(
            x=compound_x,
            y=compound_y,
            mode='markers',
            marker=dict(
                size=12, 
                color=compound_colors,
                colorscale='Blues',
                colorbar=dict(title="Drug-likeness Score")
            ),
            name='Drug Compounds',
            hovertemplate='<b>%{text}</b><br>Type: Compound<br>Drug-likeness: %{marker.color:.3f}<extra></extra>',
            text=sample_compounds
        ))
        
        fig.update_layout(
            title="Protein-Drug Interaction Network",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Node size represents binding strength, edge thickness represents therapeutic window",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    font=dict(color="gray", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig

def main():
    st.title("🧬 Pharmacological Topological Maps")
    st.markdown("Generate 3D topological maps of drug responses using multiomic data from UniProt")
    
    # Initialize mapper, mathematical engine, and analyzers
    if 'pharma_mapper' not in st.session_state:
        st.session_state.pharma_mapper = PharmacologicalMapper()
    
    if 'math_engine' not in st.session_state:
        st.session_state.math_engine = MathematicalEngine()
    
    if 'drug_db' not in st.session_state:
        st.session_state.drug_db = DrugDatabase()
    
    if 'teratrend_analyzer' not in st.session_state:
        st.session_state.teratrend_analyzer = TeratrendAnalyzer()
    
    if 'literature_analyzer' not in st.session_state:
        st.session_state.literature_analyzer = LiteratureAnalyzer()
    
    mapper = st.session_state.pharma_mapper
    math_engine = st.session_state.math_engine
    drug_db = st.session_state.drug_db
    teratrend_analyzer = st.session_state.teratrend_analyzer
    literature_analyzer = st.session_state.literature_analyzer
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Map Configuration")
        
        # Protein input
        st.subheader("Target Proteins")
        protein_input_method = st.radio(
            "Input Method",
            ["Manual Entry", "Predefined Set", "Upload File"]
        )
        
        if protein_input_method == "Manual Entry":
            protein_ids_text = st.text_area(
                "UniProt Protein IDs",
                value="P04637,P53350,P42574,P35354,P08183",
                help="Enter UniProt IDs separated by commas"
            )
            protein_ids = [pid.strip() for pid in protein_ids_text.split(',') if pid.strip()]
        
        elif protein_input_method == "Predefined Set":
            predefined_set = st.selectbox(
                "Select Protein Set",
                [
                    "300-Drug Demo (5 proteins x 300 compounds)",
                    "Kinase Targets (5 proteins)",
                    "GPCR Targets (5 proteins)", 
                    "Ion Channels (5 proteins)",
                    "Nuclear Receptors (5 proteins)",
                    "Cancer Targets (7 proteins)"
                ]
            )
            
            protein_sets = {
                "Cytochrome P450 Complex (6 major enzymes)": ["P08684", "P05181", "P11712", "P05177", "P33261", "P20815"],
                "300-Drug Demo (5 proteins x 300 compounds)": ["P04637", "P42574", "P35354", "P08183", "P42345"],
                "Kinase Targets (5 proteins)": ["P04637", "P42574", "P35354", "P08183", "P42345"],
                "GPCR Targets (5 proteins)": ["P08913", "P35367", "P21728", "P25101", "P08170"],
                "Ion Channels (5 proteins)": ["P35499", "Q14524", "P78537", "P25021", "P22460"],
                "Nuclear Receptors (5 proteins)": ["P03372", "P10828", "P04150", "P06401", "P11473"],
                "Cancer Targets (7 proteins)": ["P04637", "P53350", "P42574", "P35354", "P08183", "P04629", "P42345"]
            }
            
            # Auto-set compounds to 300 for the demo
            if predefined_set == "300-Drug Demo (5 proteins x 300 compounds)":
                n_compounds = 300
                drug_class = "Multi-class demo"
            elif predefined_set == "Cytochrome P450 Complex (6 major enzymes)":
                # Set up for CYP450 analysis
                drug_class = "CYP450 Substrates (Major)"
                drug_names = ["Warfarin", "Codeine", "Tamoxifen", "Clopidogrel", "Omeprazole", "Diazepam", "Caffeine", "Acetaminophen"]
                n_compounds = len(drug_names)
            
            protein_ids = protein_sets[predefined_set]
        
        else:  # Upload File
            uploaded_file = st.file_uploader("Upload protein IDs file", type=['txt', 'csv'])
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                if uploaded_file.name.endswith('.csv'):
                    df_uploaded = pd.read_csv(uploaded_file)
                    protein_ids = df_uploaded.iloc[:, 0].tolist()[:10]  # First column, max 10
                else:
                    protein_ids = [pid.strip() for pid in content.split('\n') if pid.strip()][:10]
            else:
                protein_ids = []
        
        # Drug-specific input
        st.subheader("💊 Drug-Specific Analysis")
        
        drug_input_method = st.radio(
            "Drug Input Method",
            ["Interactive Parameter Control", "Generate Virtual Compounds", "Specify Drug Names", "Drug Class Analysis"]
        )
        
        drug_names = None
        drug_class = None
        
        if drug_input_method == "Interactive Parameter Control":
            st.markdown("### 🎛️ Real-time Parameter Control")
            
            # Drug selection with real-time updates
            available_drugs = [
                "Warfarin", "Codeine", "Tamoxifen", "Clopidogrel", "Omeprazole", "Diazepam", "Caffeine", "Acetaminophen",
                "Ketoconazole", "Fluconazole", "Clarithromycin", "Ritonavir", "Phenytoin", "Carbamazepine", "Rifampin",
                "Aspirin", "Ibuprofen", "Metoprolol", "Atorvastatin", "Lisinopril", "Metformin", "Digoxin"
            ]
            
            selected_drugs = st.multiselect(
                "Select Drugs for Analysis",
                available_drugs,
                default=["Warfarin", "Codeine", "Omeprazole"],
                help="Choose specific drugs to analyze"
            )
            drug_names = selected_drugs if selected_drugs else ["Warfarin", "Codeine", "Omeprazole"]
            
            # Time interval controls
            st.markdown("#### ⏱️ Time Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                time_start = st.number_input("Start Time (hours)", min_value=0.0, max_value=72.0, value=0.0, step=0.5)
            with col2:
                time_end = st.number_input("End Time (hours)", min_value=0.5, max_value=168.0, value=24.0, step=1.0)
            with col3:
                time_points = st.number_input("Time Points", min_value=10, max_value=100, value=50, step=10)
            
            # Genomic hits controls
            st.markdown("#### 🧬 Genomic Parameters")
            col1, col2 = st.columns(2)
            with col1:
                genomic_hits = st.slider("Genomic Hits Count", min_value=10, max_value=1000, value=100, step=10,
                                       help="Number of genomic targets/variants to simulate")
            with col2:
                mutation_rate = st.slider("Mutation Frequency (%)", min_value=1.0, max_value=50.0, value=15.0, step=1.0,
                                        help="Percentage of population with genetic variants")
            
            # Protein customization
            st.markdown("#### 🎯 Protein Target Control")
            if protein_input_method == "Predefined Set" and "Cytochrome P450" in predefined_set:
                # Special CYP450 controls
                cyp450_enzymes = {
                    "CYP1A2": "P05177", "CYP2C9": "P11712", "CYP2C19": "P33261", 
                    "CYP2D6": "P10635", "CYP2E1": "P05181", "CYP3A4": "P08684"
                }
                selected_cyps = st.multiselect(
                    "Select CYP450 Enzymes",
                    list(cyp450_enzymes.keys()),
                    default=list(cyp450_enzymes.keys())[:3]
                )
                protein_ids = [cyp450_enzymes[cyp] for cyp in selected_cyps]
            
            # Advanced parameters
            st.markdown("#### ⚙️ Advanced Controls")
            col1, col2 = st.columns(2)
            with col1:
                dose_response_curve = st.selectbox("Dose-Response Model", 
                                                 ["Sigmoid", "Linear", "Exponential", "Biphasic"])
                binding_affinity_range = st.slider("Binding Affinity Range (nM)", 
                                                  min_value=0.1, max_value=10000.0, 
                                                  value=(1.0, 100.0), step=0.1)
            with col2:
                protein_expression = st.slider("Protein Expression Level", 
                                              min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                              help="Relative protein expression (1.0 = normal)")
                tissue_specificity = st.selectbox("Tissue Type", 
                                                 ["Liver", "Brain", "Heart", "Kidney", "Lung", "Generic"])
            
            # Store parameters in session state for analysis
            st.session_state.interactive_params = {
                'time_start': time_start,
                'time_end': time_end,
                'time_points': time_points,
                'genomic_hits': genomic_hits,
                'mutation_rate': mutation_rate,
                'dose_response_curve': dose_response_curve,
                'binding_affinity_range': binding_affinity_range,
                'protein_expression': protein_expression,
                'tissue_specificity': tissue_specificity
            }
            
            n_compounds = len(drug_names)
            st.success(f"Interactive mode: {n_compounds} drugs, {len(protein_ids)} proteins, {time_points} time points")
            
        elif drug_input_method == "Specify Drug Names":
            drug_names_text = st.text_area(
                "Drug Names (one per line or comma-separated)",
                value="Aspirin\nIbuprofen\nWarfarin\nDigoxin\nInsulin\nMetformin\nLisinopril\nAmlodipine\nSimvastatin\nLevothyroxine",
                help="Enter specific drug names to analyze"
            )
            drug_names = [drug.strip() for drug in re.split('[,\n]', drug_names_text) if drug.strip()]
            n_compounds = len(drug_names) if drug_names else 10
            st.info(f"Analyzing {len(drug_names)} specific drugs")
            
        elif drug_input_method == "Drug Class Analysis":
            drug_class = st.selectbox(
                "Select Drug Class",
                [
                    "CYP450 Substrates (Major)", "CYP450 Inhibitors", "CYP450 Inducers",
                    "Beta-blockers", "ACE inhibitors", "Statins", "NSAIDs", 
                    "Benzodiazepines", "Antibiotics", "Antidepressants", 
                    "Chemotherapy agents", "Antihistamines", "Proton pump inhibitors"
                ]
            )
            
            # Generate representative drugs for the class
            drug_class_examples = {
                "CYP450 Substrates (Major)": ["Warfarin", "Codeine", "Tamoxifen", "Clopidogrel", "Omeprazole", "Diazepam", "Caffeine", "Acetaminophen"],
                "CYP450 Inhibitors": ["Ketoconazole", "Fluconazole", "Clarithromycin", "Grapefruit", "Fluvoxamine", "Quinidine", "Ritonavir", "Cimetidine"],
                "CYP450 Inducers": ["Phenytoin", "Carbamazepine", "Rifampin", "St John's Wort", "Phenobarbital", "Smoking", "Ethanol", "Dexamethasone"],
                "Beta-blockers": ["Metoprolol", "Propranolol", "Atenolol", "Carvedilol", "Bisoprolol"],
                "ACE inhibitors": ["Lisinopril", "Enalapril", "Captopril", "Ramipril", "Quinapril"],
                "Statins": ["Atorvastatin", "Simvastatin", "Rosuvastatin", "Pravastatin", "Lovastatin"],
                "NSAIDs": ["Ibuprofen", "Naproxen", "Diclofenac", "Celecoxib", "Indomethacin"],
                "Benzodiazepines": ["Lorazepam", "Diazepam", "Alprazolam", "Clonazepam", "Temazepam"],
                "Antibiotics": ["Amoxicillin", "Ciprofloxacin", "Doxycycline", "Azithromycin", "Ceftriaxone"],
                "Antidepressants": ["Sertraline", "Fluoxetine", "Citalopram", "Venlafaxine", "Bupropion"],
                "Chemotherapy agents": ["Doxorubicin", "Cisplatin", "Methotrexate", "Paclitaxel", "Cyclophosphamide"],
                "Antihistamines": ["Loratadine", "Cetirizine", "Diphenhydramine", "Fexofenadine", "Chlorpheniramine"],
                "Proton pump inhibitors": ["Omeprazole", "Lansoprazole", "Pantoprazole", "Esomeprazole", "Rabeprazole"]
            }
            
            drug_names = drug_class_examples.get(drug_class, [])
            n_compounds = len(drug_names)
            st.info(f"Analyzing {n_compounds} drugs from {drug_class} class")
        
        else:  # Generate Virtual Compounds
            n_compounds = st.slider("Virtual compounds per protein", 50, 300, 300)
        
        # Map parameters
        st.subheader("⚙️ Map Parameters")
        time_resolution = st.selectbox("Time resolution", ["Standard (8 points)", "High (16 points)", "Ultra-High (32 points)"])
        
        # Performance settings for large datasets
        performance_mode = st.selectbox("Performance Mode", 
                                      ["Standard", "Optimized (300+ compounds)", "Ultra-Fast (1000+ points)"])
        
        # Data sampling for visualization
        if n_compounds > 100:
            st.info(f"Large dataset detected ({n_compounds} compounds). Visualizations will use intelligent sampling for performance.")
            viz_sample_size = st.slider("Visualization sample size", 1000, 10000, 5000,
                                      help="Number of points to show in 3D visualizations (full data used for analysis)")
            st.session_state.viz_sample_size = viz_sample_size
        
        # Visualization options
        st.subheader("📊 Visualization")
        show_3d_map = st.checkbox("3D Topological Map", value=True)
        show_heatmap = st.checkbox("Therapeutic Window Heatmap", value=True)
        show_network = st.checkbox("Protein-Drug Network", value=True)
        show_statistics = st.checkbox("Statistical Analysis", value=True)
        show_mathematical = st.checkbox("Mathematical Analysis (Symbolic + Wolfram Alpha)", value=False)
        
        # Wolfram Alpha configuration
        if show_mathematical:
            st.subheader("🔬 Mathematical Engine Setup")
            use_wolfram = st.checkbox("Enable Wolfram Alpha Integration")
            
            if use_wolfram:
                wolfram_api_key = st.text_input(
                    "Wolfram Alpha API Key",
                    type="password",
                    help="Get your API key from developer.wolframalpha.com"
                )
                
                if wolfram_api_key:
                    if math_engine.setup_wolfram_alpha(wolfram_api_key):
                        st.success("Wolfram Alpha connected successfully!")
                    else:
                        st.error("Failed to connect to Wolfram Alpha")
                else:
                    st.info("MATLAB-style numerical computing is enabled by default for enhanced mathematical analysis")
    
    # Main interface
    if not protein_ids:
        st.info("Please specify protein IDs in the sidebar to generate pharmacological maps.")
        return
    
    st.subheader(f"📋 Analyzing {len(protein_ids)} Target Proteins")
    
    # Display protein list
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Target Proteins:**")
        for i, pid in enumerate(protein_ids, 1):
            st.write(f"{i}. {pid}")
    
    with col2:
        if st.button("🔄 Generate Pharmacological Maps", type="primary"):
            
            # Clear any existing data
            if 'protein_data' in st.session_state:
                del st.session_state.protein_data
            if 'drug_response_data' in st.session_state:
                del st.session_state.drug_response_data
            
            # Fetch protein data
            with st.spinner("Fetching protein data from UniProt..."):
                protein_data = mapper.fetch_uniprot_data(protein_ids)
                st.session_state.protein_data = protein_data
            
            if protein_data:
                st.success(f"Successfully fetched data for {len(protein_data)} proteins")
                
                # Generate drug response data with interactive parameters
                with st.spinner("Generating drug response profiles..."):
                    # Use interactive parameters if available
                    interactive_params = st.session_state.get('interactive_params', {})
                    
                    if interactive_params:
                        drug_df = mapper.generate_drug_response_data_with_params(
                            protein_data, n_compounds, drug_names, drug_class, interactive_params
                        )
                    else:
                        drug_df = mapper.generate_drug_response_data(protein_data, n_compounds, drug_names, drug_class)
                    
                    # Enhance with real drug properties if drug names are provided
                    if drug_names:
                        with st.spinner("Enhancing with real drug properties..."):
                            drug_df = drug_db.enhance_drug_data_with_real_properties(drug_df)
                    
                    st.session_state.drug_response_data = drug_df
                
                st.success(f"Generated {len(drug_df):,} drug-protein-time data points")
                
                # Show memory usage for large datasets
                if len(drug_df) > 10000:
                    memory_mb = drug_df.memory_usage(deep=True).sum() / 1024**2
                    st.info(f"Dataset memory usage: {memory_mb:.1f} MB")
            else:
                st.error("Failed to fetch protein data. Please check protein IDs and try again.")
    
    # Display results if data is available
    if 'drug_response_data' in st.session_state:
        
        drug_df = st.session_state.drug_response_data
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data Points", f"{len(drug_df):,}")
        with col2:
            st.metric("Proteins Analyzed", drug_df['protein_id'].nunique())
        with col3:
            st.metric("Compounds Tested", f"{drug_df['compound_id'].nunique():,}")
        with col4:
            avg_tw = drug_df['therapeutic_window'].mean()
            st.metric("Avg Therapeutic Window", f"{avg_tw:.3f}")
        
        # Additional metrics for large datasets
        if len(drug_df) > 10000:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                high_tw_count = len(drug_df[drug_df['therapeutic_window'] > 0.7])
                st.metric("High TW Compounds", f"{high_tw_count:,}")
            with col2:
                druglike_count = len(drug_df[drug_df['drug_likeness'] > 0.8])
                st.metric("Drug-like Compounds", f"{druglike_count:,}")
            with col3:
                potent_count = len(drug_df[drug_df['ki_nm'] < 100])
                st.metric("Potent Compounds (Ki<100nM)", f"{potent_count:,}")
            with col4:
                optimal_count = len(drug_df[(drug_df['therapeutic_window'] > 0.7) & 
                                          (drug_df['drug_likeness'] > 0.8)])
                st.metric("Optimal Candidates", f"{optimal_count:,}")
        
        # Protein selector for focused analysis
        selected_protein = st.selectbox(
            "Focus on specific protein (optional)",
            ["All Proteins"] + list(drug_df['protein_id'].unique())
        )
        
        protein_filter = None if selected_protein == "All Proteins" else selected_protein
        
        # Create visualizations
        if show_3d_map:
            st.subheader("🗺️ 3D Pharmacological Topological Map")
            st.markdown("Interactive 3D visualization showing drug response vs log(agonist concentration) vs time. Points colored by therapeutic window score.")
            
            # Use sampling size if defined for large datasets
            sample_size = st.session_state.get('viz_sample_size', 5000)
            fig_3d = mapper.create_3d_topological_map(drug_df, protein_filter, sample_size)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Clinical interpretation
            with st.expander("🏥 Clinical Interpretation Guide"):
                st.markdown("""
                **How to interpret the 3D topological map:**
                
                - **X-axis (Log Agonist)**: Logarithm of drug concentration - lower values indicate higher potency
                - **Y-axis (Drug Response)**: Normalized response (0-1) - higher values indicate stronger therapeutic effect
                - **Z-axis (Time)**: Time course in hours - shows how response changes over time
                - **Color (Therapeutic Window)**: Green points indicate optimal therapeutic windows
                
                **Optimal therapeutic regions:**
                - Look for green clusters in the 0.5-0.8 response range
                - Prefer compounds with sustained response over time
                - Lower log(agonist) values indicate more potent drugs
                - The red surface shows the ideal therapeutic zone
                
                **For 300-drug analysis:**
                - Use protein filters to focus on specific targets
                - High-scoring compounds (TW > 0.7) represent promising candidates
                - Time dimension shows pharmacokinetic profiles
                - Clustering indicates structure-activity relationships
                """)
        
        if show_heatmap:
            st.subheader("🔥 Therapeutic Window Heatmap")
            st.markdown("Heatmap showing therapeutic window scores across concentration and time bins.")
            
            fig_heatmap = mapper.create_therapeutic_window_heatmap(drug_df)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        if show_network:
            st.subheader("🕸️ Protein-Drug Interaction Network")
            st.markdown("Network visualization showing relationships between proteins and drug compounds.")
            
            # Allow user to set max compounds for network visualization
            max_network_compounds = st.slider("Max compounds in network", 50, 200, 100,
                                             help="Limit compounds shown in network for better performance")
            
            fig_network = mapper.create_protein_network_map(drug_df, max_network_compounds)
            st.plotly_chart(fig_network, use_container_width=True)
        
        if show_statistics:
            st.subheader("📈 Statistical Analysis")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Compound Statistics", "Protein Analysis", "Temporal Patterns", "🔮 Teratrend Analysis", "📚 Literature Review"])
            
            with tab1:
                # Compound property distributions
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_mw = px.histogram(drug_df, x='molecular_weight', 
                                        title="Molecular Weight Distribution",
                                        labels={'molecular_weight': 'Molecular Weight (Da)'})
                    st.plotly_chart(fig_mw, use_container_width=True)
                
                with col2:
                    fig_druglike = px.histogram(drug_df, x='drug_likeness',
                                              title="Drug-likeness Score Distribution",
                                              labels={'drug_likeness': 'Drug-likeness Score'})
                    st.plotly_chart(fig_druglike, use_container_width=True)
                
                # Therapeutic window analysis
                st.subheader("Therapeutic Window Analysis")
                tw_stats = drug_df.groupby('compound_id')['therapeutic_window'].agg(['mean', 'max', 'std']).reset_index()
                tw_stats = tw_stats.sort_values('mean', ascending=False)
                
                fig_tw_scatter = px.scatter(tw_stats, x='mean', y='max', size='std',
                                          title="Therapeutic Window: Mean vs Max (size = std)",
                                          labels={'mean': 'Mean TW Score', 'max': 'Max TW Score'})
                st.plotly_chart(fig_tw_scatter, use_container_width=True)
                
                # Top compounds (show more for large datasets)
                st.subheader("🏆 Top Therapeutic Compounds")
                n_top = 20 if len(drug_df) > 10000 else 10
                top_compounds = tw_stats.head(n_top)[['compound_id', 'mean', 'max']]
                top_compounds.columns = ['Compound ID', 'Mean TW Score', 'Max TW Score']
                st.dataframe(top_compounds, use_container_width=True)
                
                # Download top compounds for large datasets
                if len(drug_df) > 10000:
                    top_100 = tw_stats.head(100)
                    csv_top = top_100.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Top 100 Compounds",
                        data=csv_top,
                        file_name=f"top_100_compounds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with tab2:
                # Protein-specific analysis
                protein_stats = drug_df.groupby(['protein_id', 'protein_name']).agg({
                    'drug_response': ['mean', 'max'],
                    'therapeutic_window': ['mean', 'max'],
                    'ki_nm': 'mean'
                }).round(3)
                
                protein_stats.columns = ['Mean Response', 'Max Response', 'Mean TW', 'Max TW', 'Mean Ki (nM)']
                protein_stats = protein_stats.reset_index()
                
                st.dataframe(protein_stats, use_container_width=True)
                
                # Protein comparison
                fig_protein_comparison = px.box(drug_df, x='protein_name', y='therapeutic_window',
                                              title="Therapeutic Window Distribution by Protein")
                fig_protein_comparison.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_protein_comparison, use_container_width=True)
            
            with tab3:
                # Temporal analysis
                time_analysis = drug_df.groupby('time_hours').agg({
                    'drug_response': 'mean',
                    'therapeutic_window': 'mean',
                    'concentration_um': 'mean'
                }).reset_index()
                
                fig_temporal = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Mean Drug Response', 'Mean Therapeutic Window', 
                                  'Mean Concentration', 'Response vs TW'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Mean response over time
                fig_temporal.add_trace(
                    go.Scatter(x=time_analysis['time_hours'], y=time_analysis['drug_response'],
                             name='Drug Response', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Mean therapeutic window over time
                fig_temporal.add_trace(
                    go.Scatter(x=time_analysis['time_hours'], y=time_analysis['therapeutic_window'],
                             name='Therapeutic Window', line=dict(color='green')),
                    row=1, col=2
                )
                
                # Mean concentration over time
                fig_temporal.add_trace(
                    go.Scatter(x=time_analysis['time_hours'], y=time_analysis['concentration_um'],
                             name='Concentration', line=dict(color='red')),
                    row=2, col=1
                )
                
                # Response vs therapeutic window correlation
                fig_temporal.add_trace(
                    go.Scatter(x=time_analysis['drug_response'], y=time_analysis['therapeutic_window'],
                             mode='markers+lines', name='Response-TW', 
                             marker=dict(size=8, color='purple')),
                    row=2, col=2
                )
                
                fig_temporal.update_layout(height=600, showlegend=False, 
                                         title_text="Temporal Analysis of Drug Response")
                st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Mathematical Analysis Section
        if show_mathematical:
            st.subheader("🧮 Advanced Mathematical Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Symbolic Math", "Wolfram Alpha", "MATLAB-Style Analysis", "PK-PD Modeling"
            ])
            
            with tab1:
                st.markdown("### Symbolic Mathematics")
                
                # MATLAB-style analysis
                matlab_results = math_engine.matlab_style_analysis(drug_df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Statistical Summary**")
                    if 'therapeutic_window' in matlab_results:
                        tw_stats = matlab_results['therapeutic_window']
                        st.write(f"Mean: {tw_stats['mean']:.4f}")
                        st.write(f"Std Dev: {tw_stats['std']:.4f}")
                        st.write(f"Skewness: {tw_stats['skewness']:.4f}")
                        st.write(f"Kurtosis: {tw_stats['kurtosis']:.4f}")
                
                with col2:
                    st.markdown("**Matrix Properties**")
                    matrix_analysis = math_engine.matrix_pharmacology_analysis(drug_df)
                    if 'matrix_shape' in matrix_analysis:
                        st.write(f"Matrix Shape: {matrix_analysis['matrix_shape']}")
                        st.write(f"Condition Number: {matrix_analysis['condition_number']:.2f}")
                        st.write(f"Matrix Rank: {matrix_analysis['matrix_rank']}")
                
                # Mathematical dashboard
                math_figures = math_engine.create_mathematical_dashboard(drug_df)
                
                if 'correlation_heatmap' in math_figures:
                    st.plotly_chart(math_figures['correlation_heatmap'], use_container_width=True)
                
                if 'pca_plot' in math_figures:
                    st.plotly_chart(math_figures['pca_plot'], use_container_width=True)
                
                if 'distribution_analysis' in math_figures:
                    st.plotly_chart(math_figures['distribution_analysis'], use_container_width=True)
            
            with tab2:
                st.markdown("### MATLAB-style Numerical Computing")
                
                # MATLAB-style numerical analysis
                if math_engine.matlab_available:
                    st.success("✅ MATLAB-style numerical computing enabled")
                    
                    # Enhanced pharmacological analysis using MATLAB-style methods
                    st.markdown("#### Advanced Pharmacokinetic Analysis (MATLAB-style)")
                    
                    sample_drug = drug_df.iloc[0].to_dict()
                    matlab_results = math_engine.analyze_drug_pharmacokinetics_matlab(sample_drug)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**MATLAB-style PK Analysis:**")
                        for key, value in matlab_results.items():
                            if not key.startswith('_') and key != 'error':
                                st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                    
                    with col2:
                        st.markdown("**Numerical Equations:**")
                        equation_results = math_engine.generate_pharmacological_equations_matlab(drug_df)
                        for key, value in equation_results.items():
                            if 'matlab' in key.lower() or 'numerical' in key.lower():
                                st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                
                # Advanced MATLAB-style calculations
                st.markdown("#### Custom Pharmacokinetic Calculations")
                
                col1, col2 = st.columns(2)
                with col1:
                    dose_input = st.number_input("Dose (mg)", value=100.0, min_value=0.1, step=10.0)
                    ka_input = st.number_input("Absorption rate (ka, 1/h)", value=1.0, min_value=0.01, step=0.1)
                    ke_input = st.number_input("Elimination rate (ke, 1/h)", value=0.1, min_value=0.001, step=0.01)
                
                with col2:
                    vd_input = st.number_input("Volume of distribution (L)", value=70.0, min_value=1.0, step=5.0)
                    bioavail_input = st.slider("Bioavailability", 0.1, 1.0, 0.8, step=0.05)
                    
                if st.button("Calculate PK Parameters"):
                    custom_drug = {
                        'compound_name': 'Custom Drug',
                        'dose': dose_input,
                        'kabs': ka_input,
                        'kel': ke_input,
                        'vd': vd_input,
                        'bioavailability': bioavail_input
                    }
                    
                    pk_results = math_engine.analyze_drug_pharmacokinetics_matlab(custom_drug)
                    
                    st.markdown("**Calculated PK Parameters:**")
                    for key, value in pk_results.items():
                        if not key.startswith('_') and key != 'error':
                            st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                
                # Predefined pharmacological calculations
                st.markdown("**Predefined PK/PD Models:**")
                
                model_buttons = [
                    ("One-compartment PK Model", "Calculate concentration vs time profile"),
                    ("Two-compartment PK Model", "Distribution and elimination phases"),
                    ("Hill Equation Analysis", "Dose-response relationship modeling"),
                    ("Bioequivalence Analysis", "Compare test vs reference formulations"),
                    ("Population PK Simulation", "Inter-individual variability modeling")
                ]
                
                for model_name, description in model_buttons:
                    if st.button(f"Run: {model_name}", key=f"model_{model_name}"):
                        st.info(f"Running {model_name}: {description}")
                        
                        # Example calculations for demonstration
                        if "One-compartment" in model_name:
                            time_range = np.linspace(0, 24, 100)
                            conc_profile = 100 * np.exp(-0.1 * time_range)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=time_range, y=conc_profile, 
                                                   mode='lines', name='Concentration'))
                            fig.update_layout(title="One-compartment PK Model",
                                            xaxis_title="Time (hours)",
                                            yaxis_title="Concentration (ng/mL)")
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("### MATLAB-Style Matrix Analysis")
                
                matrix_results = math_engine.matrix_pharmacology_analysis(drug_df)
                
                if 'error' in matrix_results:
                    st.error(matrix_results['error'])
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Covariance Matrix**")
                        if matrix_results['covariance_matrix'] is not None:
                            cov_df = pd.DataFrame(
                                matrix_results['covariance_matrix'],
                                index=matrix_results['feature_names'],
                                columns=matrix_results['feature_names']
                            )
                            st.dataframe(cov_df)
                    
                    with col2:
                        st.markdown("**Eigenvalue Analysis**")
                        if matrix_results['eigenvalues'] is not None:
                            eigenval_df = pd.DataFrame({
                                'Eigenvalue': matrix_results['eigenvalues'],
                                'Index': range(len(matrix_results['eigenvalues']))
                            })
                            fig_eigen = px.bar(eigenval_df, x='Index', y='Eigenvalue',
                                             title="Eigenvalue Spectrum")
                            st.plotly_chart(fig_eigen, use_container_width=True)
                    
                    # PCA Analysis
                    if 'pca_explained_variance' in matrix_results:
                        st.markdown("**Principal Component Analysis**")
                        pca_var = matrix_results['pca_explained_variance']
                        cumsum_var = np.cumsum(pca_var) / np.sum(pca_var)
                        
                        fig_pca = go.Figure()
                        fig_pca.add_trace(go.Bar(
                            x=list(range(1, len(pca_var)+1)),
                            y=pca_var / np.sum(pca_var),
                            name='Individual Variance'
                        ))
                        fig_pca.add_trace(go.Scatter(
                            x=list(range(1, len(cumsum_var)+1)),
                            y=cumsum_var,
                            mode='lines+markers',
                            name='Cumulative Variance',
                            yaxis='y2'
                        ))
                        
                        fig_pca.update_layout(
                            title="PCA Variance Explained",
                            xaxis_title="Principal Component",
                            yaxis_title="Variance Explained",
                            yaxis2=dict(overlaying='y', side='right', title="Cumulative Variance"),
                            height=400
                        )
                        
                        st.plotly_chart(fig_pca, use_container_width=True)
            
            with tab4:
                st.markdown("### Pharmacokinetic-Pharmacodynamic Modeling")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**PK Parameters**")
                    dose = st.number_input("Dose (mg)", value=100.0, min_value=1.0)
                    ka = st.number_input("Absorption rate (1/h)", value=1.5, min_value=0.1)
                    ke = st.number_input("Elimination rate (1/h)", value=0.2, min_value=0.01)
                    vd = st.number_input("Volume of distribution (L)", value=50.0, min_value=1.0)
                    bioavail = st.number_input("Bioavailability", value=1.0, min_value=0.1, max_value=1.0)
                
                with col2:
                    st.markdown("**PD Parameters**")
                    emax = st.number_input("Maximum effect", value=1.0, min_value=0.1)
                    ec50 = st.number_input("EC50 (μM)", value=10.0, min_value=0.1)
                    hill_coeff = st.number_input("Hill coefficient", value=1.0, min_value=0.1)
                
                if st.button("Generate PK-PD Model"):
                    pk_params = {
                        'dose': dose, 'ka': ka, 'ke': ke, 'vd': vd, 'F': bioavail
                    }
                    pd_params = {
                        'emax': emax, 'ec50': ec50, 'n': hill_coeff
                    }
                    
                    # Create symbolic models
                    pk_model, combined_model = math_engine.create_pk_pd_model(pk_params, pd_params)
                    
                    # Display symbolic expressions
                    st.markdown("**Symbolic Models:**")
                    st.latex(f"PK: C(t) = {pk_model}")
                    st.latex(f"Combined: E(t) = {combined_model}")
                    
                    # Analyze therapeutic window
                    tw_analysis = math_engine.analyze_therapeutic_window(combined_model, 0.2, 0.8)
                    
                    # Create mathematical plots
                    time_range = (0, 24)  # 24 hours
                    
                    pk_plot = math_engine.create_mathematical_plots(
                        pk_model, math_engine.t, time_range, "line"
                    )
                    pk_plot.update_layout(title="Pharmacokinetic Profile")
                    st.plotly_chart(pk_plot, use_container_width=True)
                    
                    pd_plot = math_engine.create_mathematical_plots(
                        combined_model, math_engine.t, time_range, "line"
                    )
                    pd_plot.update_layout(title="Pharmacodynamic Response")
                    st.plotly_chart(pd_plot, use_container_width=True)
                    
                    # Create dose-response surface
                    dose_range = np.linspace(50, 200, 10)
                    time_range_array = np.linspace(0, 24, 50)
                    
                    surface_plot = math_engine.create_dose_response_surface(
                        dose_range, time_range_array, pk_params, pd_params
                    )
                    st.plotly_chart(surface_plot, use_container_width=True)
                    
                    # Symbolic calculus operations
                    st.markdown("**Symbolic Calculus Operations:**")
                    calculus_ops = math_engine.symbolic_calculus_operations(combined_model)
                    
                    with st.expander("View Calculus Operations"):
                        for op_name, op_result in calculus_ops.items():
                            st.latex(f"{op_name}: {op_result}")
            
            with tab4:
                st.markdown("### 🔮 Teratrend Analysis: Large-Scale Drug Class Pattern Recognition")
                st.markdown("Analyze massive-scale trends beyond megatrends for comprehensive drug class motif exploration")
                
                # Drug input for teratrend analysis
                drug_for_analysis = st.text_input(
                    "Enter drug name for teratrend analysis:",
                    value="Atorvastatin",
                    help="Enter a specific drug name to analyze its class patterns and motifs"
                )
                
                if st.button("🚀 Run Teratrend Analysis", type="primary"):
                    with st.spinner("Performing comprehensive teratrend analysis..."):
                        # Run teratrend analysis
                        teratrend_result = teratrend_analyzer.analyze_drug_teratrends(drug_for_analysis)
                        
                        # Store result in session state
                        st.session_state.teratrend_result = teratrend_result
                        
                        st.success(f"Teratrend analysis completed for {drug_for_analysis}!")
                
                # Display teratrend results if available
                if 'teratrend_result' in st.session_state:
                    result = st.session_state.teratrend_result
                    
                    # Overview metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Drug Class", result.drug_class)
                    with col2:
                        st.metric("Structural Motifs", len(result.structural_motifs))
                    with col3:
                        st.metric("Prediction Confidence", f"{result.prediction_confidence:.1%}")
                    with col4:
                        st.metric("Innovation Potential", "High" if result.prediction_confidence > 0.8 else "Medium")
                    
                    # Structural motifs analysis
                    st.subheader("🧪 Structural Motifs & Innovation Patterns")
                    motifs_df = pd.DataFrame(result.structural_motifs)
                    if not motifs_df.empty:
                        st.dataframe(motifs_df[['motif_type', 'frequency', 'therapeutic_impact', 'innovation_potential']], 
                                   use_container_width=True)
                    
                    # Mechanism trends
                    st.subheader("⚙️ Mechanism of Action Evolution")
                    with st.expander("Historical Evolution Timeline"):
                        for period, description in result.mechanism_trends.get('historical_evolution', {}).items():
                            st.markdown(f"**{period}**: {description}")
                    
                    # Innovation velocity metrics
                    if 'innovation_velocity' in result.mechanism_trends:
                        iv = result.mechanism_trends['innovation_velocity']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Patents/Year", f"{iv.get('patents_per_year', 0):.0f}")
                            st.metric("Clinical Trials/Year", f"{iv.get('clinical_trials_per_year', 0):.0f}")
                        with col2:
                            st.metric("Approvals/Year", f"{iv.get('regulatory_approvals_per_year', 0):.1f}")
                            st.metric("Innovation Index", f"{iv.get('innovation_index', 0):.2f}")
                    
                    # Therapeutic evolution
                    st.subheader("🎯 Therapeutic Application Evolution")
                    ind_exp = result.therapeutic_evolution.get('indication_expansion', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Primary Indications:**")
                        for indication in ind_exp.get('primary_indications', []):
                            st.markdown(f"• {indication}")
                    
                    with col2:
                        st.markdown("**Emerging Indications:**")
                        for indication in ind_exp.get('emerging_indications', []):
                            st.markdown(f"• {indication}")
                    
                    # Combination potential
                    st.subheader("🔗 Combination Therapy Potential")
                    combo_df = pd.DataFrame(result.combination_potential)
                    if not combo_df.empty:
                        st.dataframe(combo_df[['combination_type', 'mechanism', 'clinical_potential', 'development_timeline']], 
                                   use_container_width=True)
                    
                    # Market dynamics
                    st.subheader("💰 Market Dynamics & Innovation Patterns")
                    market = result.market_dynamics
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Market Size Evolution:**")
                        st.markdown(f"Current: {market.get('market_size_evolution', {}).get('current_market_size', 'N/A')}")
                        st.markdown(f"Growth Rate: {market.get('market_size_evolution', {}).get('projected_growth_rate', 'N/A')}")
                    
                    with col2:
                        st.markdown("**Competitive Landscape:**")
                        st.markdown(f"Intensity: {market.get('competitive_landscape', {}).get('competitive_intensity', 'N/A')}")
                    
                    # Create teratrend visualizations
                    with st.spinner("Generating teratrend visualizations..."):
                        try:
                            teratrend_figs = create_teratrend_visualizations(result)
                            
                            for fig_name, fig in teratrend_figs.items():
                                if fig_name == 'structural_motifs':
                                    st.subheader("📊 Structural Motif Frequency Analysis")
                                elif fig_name == 'therapeutic_timeline':
                                    st.subheader("⏱️ Therapeutic Evolution Timeline")
                                elif fig_name == 'market_radar':
                                    st.subheader("🎯 Market Dynamics Radar")
                                elif fig_name == 'combination_network':
                                    st.subheader("🕸️ Combination Therapy Network")
                                
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Visualization generation skipped: {str(e)}")
            
            with tab5:
                st.markdown("### 📚 Comprehensive Literature Review Generator")
                st.markdown("Generate systematic reviews, meta-analyses, and scoping reviews with 50+ articles from regular and gray literature")
                
                # Literature review parameters
                col1, col2 = st.columns(2)
                with col1:
                    drug_for_literature = st.text_input(
                        "Drug name for literature review:",
                        value=drug_for_analysis if 'drug_for_analysis' in locals() else "Atorvastatin",
                        help="Enter drug name for comprehensive literature analysis"
                    )
                    
                    target_articles = st.slider(
                        "Target number of articles:",
                        min_value=25, max_value=100, value=50,
                        help="Number of articles to include in comprehensive review"
                    )
                
                with col2:
                    review_types = st.multiselect(
                        "Review types to generate:",
                        ["Systematic Review", "Meta-Analysis", "Scoping Review", "Narrative Review"],
                        default=["Systematic Review", "Meta-Analysis"],
                        help="Select types of reviews to include"
                    )
                    
                    include_gray_literature = st.checkbox(
                        "Include gray literature",
                        value=True,
                        help="Include regulatory documents, conference abstracts, and other gray literature"
                    )
                
                if st.button("📖 Generate Comprehensive Literature Review", type="primary"):
                    with st.spinner(f"Generating comprehensive {target_articles}-article literature review..."):
                        # Generate comprehensive review
                        comprehensive_review = literature_analyzer.generate_comprehensive_review(
                            drug_for_literature, target_articles
                        )
                        
                        # Store in session state
                        st.session_state.comprehensive_review = comprehensive_review
                        
                        st.success(f"Comprehensive literature review generated with {comprehensive_review.article_count} articles!")
                
                # Display literature review results
                if 'comprehensive_review' in st.session_state:
                    review = st.session_state.comprehensive_review
                    
                    # Overview metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Articles", review.article_count)
                    with col2:
                        sr_count = len(review.systematic_review_summary.get('results', {}).get('studies_included', []))
                        st.metric("Systematic Reviews", sr_count)
                    with col3:
                        ma_count = review.meta_analysis_results.get('pooled_estimates', {}).get('effect_size', 0)
                        st.metric("Meta-Analysis Effect", f"{ma_count:.3f}" if isinstance(ma_count, (int, float)) else "N/A")
                    with col4:
                        eq_score = review.evidence_quality.get('overall_quality_score', 0)
                        st.metric("Evidence Quality", f"{eq_score:.2f}/1.0" if isinstance(eq_score, (int, float)) else "N/A")
                    
                    # Literature review tabs
                    lit_tab1, lit_tab2, lit_tab3, lit_tab4 = st.tabs([
                        "📋 Systematic Review", "📊 Meta-Analysis", "🔍 Scoping Review", "📖 Clinical Trials"
                    ])
                    
                    with lit_tab1:
                        st.subheader("Systematic Review Summary")
                        
                        # Methodology
                        st.markdown("**Methodology:**")
                        methodology = review.systematic_review_summary.get('methodology', {})
                        st.markdown(f"Search Strategy: {methodology.get('search_strategy', 'N/A')}")
                        
                        # Inclusion criteria
                        with st.expander("Inclusion Criteria"):
                            for criterion in methodology.get('inclusion_criteria', []):
                                st.markdown(f"• {criterion}")
                        
                        # Results summary
                        st.markdown("**Results:**")
                        results = review.systematic_review_summary.get('results', {})
                        st.markdown(f"Total identified: {results.get('total_identified', 0)}")
                        st.markdown(f"Studies included: {results.get('studies_included', 0)}")
                        
                        # Quality assessment
                        qa = review.systematic_review_summary.get('quality_assessment', {})
                        if qa:
                            st.markdown("**Quality Assessment:**")
                            quality_dist = qa.get('quality_distribution', {})
                            if quality_dist:
                                quality_df = pd.DataFrame([quality_dist]).T
                                quality_df.columns = ['Count']
                                st.bar_chart(quality_df)
                    
                    with lit_tab2:
                        st.subheader("Meta-Analysis Results")
                        
                        # Pooled estimates
                        pooled = review.meta_analysis_results.get('pooled_estimates', {})
                        if pooled:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Pooled Effect Size", f"{pooled.get('effect_size', 0):.3f}")
                                st.metric("P-value", f"{pooled.get('p_value', 1):.3f}")
                            with col2:
                                ci = pooled.get('confidence_interval', [0, 0])
                                st.metric("95% CI Lower", f"{ci[0]:.3f}")
                                st.metric("95% CI Upper", f"{ci[1]:.3f}")
                        
                        # Heterogeneity
                        heterogeneity = review.meta_analysis_results.get('heterogeneity', {})
                        if heterogeneity:
                            st.markdown("**Heterogeneity Assessment:**")
                            st.markdown(f"I² = {heterogeneity.get('i_squared', 0):.1f}%")
                            st.markdown(f"Interpretation: {heterogeneity.get('interpretation', 'N/A')}")
                        
                        # Publication bias
                        pub_bias = review.meta_analysis_results.get('publication_bias', {})
                        if pub_bias:
                            st.markdown("**Publication Bias Assessment:**")
                            st.markdown(f"Egger test p-value: {pub_bias.get('egger_test_p', 1):.3f}")
                            st.markdown(f"Funnel plot asymmetry: {pub_bias.get('funnel_plot_asymmetry', 'N/A')}")
                    
                    with lit_tab3:
                        st.subheader("Scoping Review")
                        
                        # Research landscape
                        landscape = review.scoping_review.get('research_landscape', {})
                        if landscape:
                            st.markdown("**Key Research Areas:**")
                            for area in landscape.get('key_research_areas', []):
                                st.markdown(f"• {area}")
                            
                            # Geographic distribution
                            geo_dist = landscape.get('geographic_distribution', {})
                            if geo_dist:
                                st.markdown("**Geographic Distribution:**")
                                geo_df = pd.DataFrame([geo_dist]).T
                                geo_df.columns = ['Studies']
                                st.bar_chart(geo_df)
                        
                        # Knowledge gaps
                        gaps = review.scoping_review.get('knowledge_gaps', {})
                        if gaps:
                            st.markdown("**Knowledge Gaps Identified:**")
                            for gap_type, gap_list in gaps.items():
                                if isinstance(gap_list, list):
                                    st.markdown(f"**{gap_type.replace('_', ' ').title()}:**")
                                    for gap in gap_list:
                                        st.markdown(f"• {gap}")
                    
                    with lit_tab4:
                        st.subheader("Clinical Trials Summary")
                        
                        trials = review.clinical_trial_summary
                        
                        # Trial characteristics
                        characteristics = trials.get('trial_characteristics', {})
                        if characteristics:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Trials", characteristics.get('total_trials', 0))
                                st.markdown("**Phase Distribution:**")
                                phase_dist = characteristics.get('phase_distribution', {})
                                if phase_dist:
                                    phase_df = pd.DataFrame([phase_dist]).T
                                    phase_df.columns = ['Count']
                                    st.bar_chart(phase_df)
                            
                            with col2:
                                st.markdown("**Status Distribution:**")
                                status_dist = characteristics.get('status_distribution', {})
                                if status_dist:
                                    status_df = pd.DataFrame([status_dist]).T
                                    status_df.columns = ['Count']
                                    st.bar_chart(status_df)
                    
                    # Recommendations and future directions
                    st.subheader("📋 Clinical Recommendations")
                    recommendations = review.recommendations
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                    
                    st.subheader("🔬 Future Research Directions")
                    future_directions = review.future_research_directions
                    for i, direction in enumerate(future_directions, 1):
                        st.markdown(f"{i}. {direction}")
                    
                    # Evidence summary
                    st.subheader("📊 Evidence Quality Summary")
                    evidence = review.evidence_quality
                    grade_assessment = evidence.get('grade_assessment', {})
                    
                    if grade_assessment:
                        col1, col2 = st.columns(2)
                        with col1:
                            efficacy = grade_assessment.get('efficacy_outcomes', {})
                            st.markdown(f"**Efficacy Evidence Quality:** {efficacy.get('quality', 'N/A')}")
                        with col2:
                            safety = grade_assessment.get('safety_outcomes', {})
                            st.markdown(f"**Safety Evidence Quality:** {safety.get('quality', 'N/A')}")
                    
                    # Download options
                    st.subheader("💾 Export Options")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📄 Generate PDF Report"):
                            st.info("PDF generation would create comprehensive review document")
                    with col2:
                        if st.button("📊 Export Data"):
                            st.info("Data export would provide structured review data")
                            if op_name != 'error':
                                st.write(f"**{op_name.replace('_', ' ').title()}:**")
                                st.latex(str(op_result))
                            else:
                                st.error(op_result)
        
        # Clinical significance analysis
        st.subheader("🏥 Clinical Significance & Analysis")
        
        st.markdown("""
        ### Why Pharmacological Topological Maps Matter
        
        These 3D visualizations provide critical insights for drug development and clinical practice:
        
        **🎯 Therapeutic Window Optimization**
        - The 3D maps reveal the narrow band where drugs are both effective and safe
        - Green regions (high therapeutic window scores) indicate optimal dosing zones
        - Time dimension shows how therapeutic windows change with drug metabolism
        
        **📊 Drug Discovery Applications**
        - Structure-activity relationships become visible as clustering patterns
        - Lead compound optimization guided by topological proximity to successful drugs  
        - Virtual screening enhanced by 3D pharmacological space navigation
        
        **⚕️ Clinical Decision Support**
        - Personalized dosing regimens based on individual pharmacokinetic profiles
        - Drug interaction predictions through overlapping topological regions
        - Therapeutic monitoring guided by expected response trajectories
        
        **🔬 Mechanistic Insights**
        - Protein-drug selectivity patterns revealed through network analysis
        - Temporal pharmacodynamics show onset, peak, and duration of action
        - Multi-target drug effects visualized simultaneously
        """)
        
        # Drug-specific insights
        if drug_names:
            # Special analysis for CYP450 drugs
            if drug_class and 'CYP450' in drug_class:
                st.markdown(f"""
                ### CYP450 Complex Analysis for {drug_class}
                
                **Cytochrome P450 Enzyme Analysis:**
                - Analyzed {len(drug_names)} CYP450-related drugs across {drug_df['protein_id'].nunique()} major CYP enzymes
                - Average therapeutic window score: {drug_df['therapeutic_window'].mean():.3f}
                - Most metabolically active enzyme: {drug_df.groupby('protein_name')['therapeutic_window'].mean().idxmax()}
                - Highest risk drug: {drug_df.loc[drug_df['therapeutic_window'].idxmin(), 'compound_name']}
                
                **CYP450-Specific Clinical Insights:**
                - **Drug-Drug Interactions**: Inhibitors can cause dangerous accumulation of substrate drugs
                - **Genetic Polymorphisms**: CYP2D6, CYP2C19 variants affect 10-15% of populations differently
                - **Metabolic Phenotypes**: Poor metabolizers at higher risk for toxicity, ultra-rapid metabolizers may need higher doses
                - **Time-Dependent Effects**: Inducers take days-weeks to show full effect, inhibitors act immediately
                - **Clinical Monitoring**: Therapeutic drug monitoring essential for narrow therapeutic index drugs
                
                **Enzyme-Specific Patterns:**
                - **CYP3A4** (40-50% of drugs): Highly inducible, major interaction site
                - **CYP2D6** (20-25% of drugs): High genetic variability, psychiatric/cardiac drugs
                - **CYP2C19** (10-15% of drugs): Proton pump inhibitors, antiplatelet drugs
                - **CYP2C9** (10-15% of drugs): Warfarin metabolism, bleeding risk
                - **CYP1A2** (5-10% of drugs): Caffeine, smoking interactions
                - **CYP2E1** (2-5% of drugs): Alcohol-drug interactions, hepatotoxicity
                """)
            else:
                st.markdown(f"""
                ### Analysis Summary for {drug_class if drug_class else 'Specified Drugs'}
                
                **Key Findings:**
                - Analyzed {len(drug_names)} specific drugs across {drug_df['protein_id'].nunique()} protein targets
                - Average therapeutic window score: {drug_df['therapeutic_window'].mean():.3f}
                - Top performing drug: {drug_df.loc[drug_df['therapeutic_window'].idxmax(), 'compound_name']}
                - Most challenging target: {drug_df.groupby('protein_name')['therapeutic_window'].mean().idxmin()}
                
                **Clinical Implications:**
                - Higher therapeutic window scores indicate safer, more predictable drug responses
                - Temporal patterns guide optimal dosing intervals and administration timing
                - Protein selectivity data informs combination therapy decisions
                """)
        
        # Interactive analysis
        with st.expander("🔍 Interactive Drug Analysis"):
            if drug_names:
                selected_drug = st.selectbox(
                    "Select drug for detailed analysis",
                    drug_names
                )
                
                drug_data = drug_df[drug_df['compound_name'] == selected_drug]
                
                if not drug_data.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Peak Response", f"{drug_data['drug_response'].max():.3f}")
                        st.metric("Therapeutic Window", f"{drug_data['therapeutic_window'].mean():.3f}")
                    
                    with col2:
                        st.metric("Optimal Time Point", f"{drug_data.loc[drug_data['therapeutic_window'].idxmax(), 'time_hours']:.1f}h")
                        st.metric("Drug-likeness", f"{drug_data['drug_likeness'].iloc[0]:.3f}")
                    
                    with col3:
                        st.metric("Potency (Ki)", f"{drug_data['ki_nm'].iloc[0]:.1f} nM")
                        st.metric("Selectivity Index", f"{drug_data['selectivity_index'].iloc[0]:.1f}")
                    
                    # Drug-specific temporal profile
                    drug_temporal = drug_data.groupby('time_hours').agg({
                        'drug_response': 'mean',
                        'therapeutic_window': 'mean'
                    }).reset_index()
                    
                    fig_drug_profile = go.Figure()
                    fig_drug_profile.add_trace(go.Scatter(
                        x=drug_temporal['time_hours'],
                        y=drug_temporal['drug_response'],
                        mode='lines+markers',
                        name='Drug Response',
                        line=dict(color='blue')
                    ))
                    
                    fig_drug_profile.add_trace(go.Scatter(
                        x=drug_temporal['time_hours'],
                        y=drug_temporal['therapeutic_window'],
                        mode='lines+markers',
                        name='Therapeutic Window',
                        yaxis='y2',
                        line=dict(color='green')
                    ))
                    
                    fig_drug_profile.update_layout(
                        title=f"Temporal Profile: {selected_drug}",
                        xaxis_title="Time (hours)",
                        yaxis_title="Drug Response",
                        yaxis2=dict(overlaying='y', side='right', title='Therapeutic Window'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_drug_profile, use_container_width=True)
        
        # Data export
        st.subheader("💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = drug_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV Data",
                data=csv_data,
                file_name=f"pharma_topological_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Prepare summary report
            summary_data = {
                'proteins_analyzed': drug_df['protein_id'].nunique(),
                'compounds_tested': drug_df['compound_id'].nunique(),
                'total_data_points': len(drug_df),
                'avg_therapeutic_window': drug_df['therapeutic_window'].mean(),
                'top_compounds': tw_stats.head(5).to_dict('records') if 'tw_stats' in locals() else []
            }
            
            summary_json = json.dumps(summary_data, indent=2, default=str)
            st.download_button(
                label="📊 Download Summary JSON",
                data=summary_json,
                file_name=f"pharma_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            st.write("**Analysis Complete!**")
            st.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Quick citation notice
            st.markdown("---")
            st.info("📚 **Data Attribution:** This analysis uses data from UniProt, PubChem, DrugBank, and ChEMBL. For complete citations, see the Data Citations page.")
            st.info("📝 **Platform Citation:** Ferguson, D.J., BS, MS, PharmD Candidate, RSci MRSB MRSC. Academic Research Platform for Systematic Review Validation and Pharmacological Analysis. 2025.")

if __name__ == "__main__":
    main()