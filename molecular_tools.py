"""
Molecular Tools and Calculations for Docking and Biophysical Chemistry
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import math
import tempfile
import os
import requests
import json
from Bio import PDB
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.SeqUtils import ProtParam
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import warnings
warnings.filterwarnings('ignore')

class MolecularCalculator:
    """Advanced molecular property calculations"""
    
    def __init__(self):
        self.amino_acid_properties = self._load_aa_properties()
        self.atomic_weights = self._load_atomic_weights()
    
    def _load_aa_properties(self) -> Dict:
        """Load amino acid physicochemical properties"""
        return {
            'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'charge': 0, 'polar': False},
            'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'charge': 1, 'polar': True},
            'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'charge': 0, 'polar': True},
            'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'charge': -1, 'polar': True},
            'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'charge': 0, 'polar': False},
            'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'charge': 0, 'polar': True},
            'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'charge': -1, 'polar': True},
            'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'charge': 0, 'polar': False},
            'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'charge': 0, 'polar': True},
            'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'charge': 0, 'polar': False},
            'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'charge': 0, 'polar': False},
            'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'charge': 1, 'polar': True},
            'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'charge': 0, 'polar': False},
            'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'charge': 0, 'polar': False},
            'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'charge': 0, 'polar': False},
            'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'charge': 0, 'polar': True},
            'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'charge': 0, 'polar': True},
            'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'charge': 0, 'polar': False},
            'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'charge': 0, 'polar': True},
            'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'charge': 0, 'polar': False}
        }
    
    def _load_atomic_weights(self) -> Dict:
        """Load atomic weights for common elements"""
        return {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
            'Br': 79.904, 'I': 126.904
        }
    
    def calculate_protein_properties(self, sequence: str) -> Dict:
        """Calculate comprehensive protein properties"""
        
        # Use BioPython ProtParam
        protein_analysis = ProtParam.ProteinAnalysis(sequence)
        
        properties = {
            'length': len(sequence),
            'molecular_weight': protein_analysis.molecular_weight(),
            'isoelectric_point': protein_analysis.isoelectric_point(),
            'instability_index': protein_analysis.instability_index(),
            'gravy': protein_analysis.gravy(),  # Grand average of hydropathy
            'aromaticity': protein_analysis.aromaticity(),
            'extinction_coefficient': protein_analysis.molar_extinction_coefficient(),
            'amino_acid_composition': protein_analysis.get_amino_acids_percent()
        }
        
        # Additional calculated properties
        properties['charge_at_ph7'] = self._calculate_charge_at_ph(sequence, 7.0)
        properties['hydrophobic_moment'] = self._calculate_hydrophobic_moment(sequence)
        properties['flexibility'] = self._calculate_flexibility(sequence)
        properties['secondary_structure_propensity'] = self._predict_secondary_structure(sequence)
        
        return properties
    
    def _calculate_charge_at_ph(self, sequence: str, ph: float) -> float:
        """Calculate net charge at specific pH"""
        
        # pKa values for ionizable groups
        pka_values = {
            'N_terminus': 9.6, 'C_terminus': 2.3,
            'R': 12.5, 'K': 10.5, 'H': 6.0,
            'D': 3.9, 'E': 4.3, 'C': 8.3, 'Y': 10.9
        }
        
        charge = 0.0
        
        # N-terminus
        charge += 1 / (1 + 10**(ph - pka_values['N_terminus']))
        
        # C-terminus
        charge -= 1 / (1 + 10**(pka_values['C_terminus'] - ph))
        
        # Side chains
        for aa in sequence:
            if aa in pka_values:
                if aa in ['R', 'K', 'H']:
                    charge += 1 / (1 + 10**(ph - pka_values[aa]))
                elif aa in ['D', 'E']:
                    charge -= 1 / (1 + 10**(pka_values[aa] - ph))
                elif aa in ['C', 'Y']:
                    charge -= 1 / (1 + 10**(pka_values[aa] - ph))
        
        return charge
    
    def _calculate_hydrophobic_moment(self, sequence: str, window_size: int = 11) -> float:
        """Calculate hydrophobic moment using Eisenberg scale"""
        
        if len(sequence) < window_size:
            return 0.0
        
        max_moment = 0.0
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            
            # Calculate moment for this window
            sum_cos = 0.0
            sum_sin = 0.0
            
            for j, aa in enumerate(window):
                if aa in self.amino_acid_properties:
                    hydrophobicity = self.amino_acid_properties[aa]['hydrophobicity']
                    angle = 2 * math.pi * j / window_size
                    sum_cos += hydrophobicity * math.cos(angle)
                    sum_sin += hydrophobicity * math.sin(angle)
            
            moment = math.sqrt(sum_cos**2 + sum_sin**2) / window_size
            max_moment = max(max_moment, moment)
        
        return max_moment
    
    def _calculate_flexibility(self, sequence: str) -> float:
        """Calculate average flexibility using Vihinen scale"""
        
        flexibility_scale = {
            'A': 0.984, 'R': 1.008, 'N': 1.048, 'D': 1.068, 'C': 0.906,
            'Q': 1.037, 'E': 1.094, 'G': 1.031, 'H': 0.950, 'I': 0.927,
            'L': 0.935, 'K': 1.102, 'M': 0.952, 'F': 0.915, 'P': 1.049,
            'S': 1.046, 'T': 0.997, 'W': 0.904, 'Y': 0.929, 'V': 0.931
        }
        
        total_flexibility = 0.0
        valid_residues = 0
        
        for aa in sequence:
            if aa in flexibility_scale:
                total_flexibility += flexibility_scale[aa]
                valid_residues += 1
        
        return total_flexibility / valid_residues if valid_residues > 0 else 0.0
    
    def _predict_secondary_structure(self, sequence: str) -> Dict:
        """Predict secondary structure propensities using Chou-Fasman"""
        
        # Simplified Chou-Fasman propensities
        helix_propensity = {
            'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
            'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
            'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
            'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
        }
        
        sheet_propensity = {
            'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
            'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
            'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
            'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
        }
        
        avg_helix = np.mean([helix_propensity.get(aa, 1.0) for aa in sequence])
        avg_sheet = np.mean([sheet_propensity.get(aa, 1.0) for aa in sequence])
        avg_coil = 2.0 - avg_helix - avg_sheet  # Approximation
        
        return {
            'helix_propensity': avg_helix,
            'sheet_propensity': avg_sheet,
            'coil_propensity': max(0.0, avg_coil)
        }

class StructuralAnalysis:
    """Advanced structural analysis tools"""
    
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
    
    def analyze_structure(self, pdb_data: str) -> Dict:
        """Comprehensive structural analysis"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_data)
            f.flush()
            
            try:
                structure = self.parser.get_structure('protein', f.name)
                
                analysis = {
                    'basic_info': self._get_basic_info(structure),
                    'geometry': self._analyze_geometry(structure),
                    'secondary_structure': self._analyze_secondary_structure(structure),
                    'surface_properties': self._analyze_surface_properties(structure),
                    'cavities': self._identify_cavities(structure),
                    'quality_metrics': self._assess_quality(structure)
                }
                
                return analysis
                
            finally:
                os.unlink(f.name)
    
    def _get_basic_info(self, structure: Structure) -> Dict:
        """Extract basic structural information"""
        
        chains = list(structure.get_chains())
        residues = list(structure.get_residues())
        atoms = list(structure.get_atoms())
        
        # Get sequence
        sequences = {}
        for chain in chains:
            seq = ""
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Standard amino acid
                    try:
                        seq += three_to_one(residue.get_resname())
                    except:
                        seq += 'X'  # Unknown residue
            sequences[chain.get_id()] = seq
        
        return {
            'num_chains': len(chains),
            'num_residues': len(residues),
            'num_atoms': len(atoms),
            'sequences': sequences,
            'chain_lengths': {chain.get_id(): len(sequences[chain.get_id()]) for chain in chains}
        }
    
    def _analyze_geometry(self, structure: Structure) -> Dict:
        """Analyze structural geometry"""
        
        coords = []
        for atom in structure.get_atoms():
            coords.append(atom.get_coord())
        coords = np.array(coords)
        
        # Calculate center of mass
        center_of_mass = np.mean(coords, axis=0)
        
        # Calculate radius of gyration
        distances_from_com = np.linalg.norm(coords - center_of_mass, axis=1)
        radius_of_gyration = np.sqrt(np.mean(distances_from_com**2))
        
        # Calculate bounding box
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        dimensions = max_coords - min_coords
        
        return {
            'center_of_mass': center_of_mass.tolist(),
            'radius_of_gyration': radius_of_gyration,
            'bounding_box': {
                'min': min_coords.tolist(),
                'max': max_coords.tolist(),
                'dimensions': dimensions.tolist()
            },
            'total_volume': np.prod(dimensions)
        }
    
    def _analyze_secondary_structure(self, structure: Structure) -> Dict:
        """Analyze secondary structure using DSSP-like method"""
        
        # Simplified secondary structure assignment
        # Real implementation would use DSSP or STRIDE
        
        phi_psi_angles = []
        ss_assignments = {'H': 0, 'E': 0, 'C': 0}  # Helix, Sheet, Coil
        
        for chain in structure.get_chains():
            residues = list(chain.get_residues())
            
            for i in range(1, len(residues) - 1):
                try:
                    # Get backbone atoms
                    prev_c = residues[i-1]['C']
                    curr_n = residues[i]['N']
                    curr_ca = residues[i]['CA']
                    curr_c = residues[i]['C']
                    next_n = residues[i+1]['N']
                    
                    # Calculate phi and psi angles
                    phi = self._calculate_dihedral(prev_c, curr_n, curr_ca, curr_c)
                    psi = self._calculate_dihedral(curr_n, curr_ca, curr_c, next_n)
                    
                    phi_psi_angles.append((phi, psi))
                    
                    # Simple secondary structure assignment
                    if -180 <= phi <= -30 and -70 <= psi <= 50:
                        ss_assignments['H'] += 1  # Alpha helix
                    elif -180 <= phi <= -40 and 90 <= psi <= 180:
                        ss_assignments['E'] += 1  # Beta sheet
                    else:
                        ss_assignments['C'] += 1  # Coil
                        
                except:
                    continue
        
        total_residues = sum(ss_assignments.values())
        ss_percentages = {ss: count/total_residues*100 for ss, count in ss_assignments.items()} if total_residues > 0 else ss_assignments
        
        return {
            'phi_psi_angles': phi_psi_angles,
            'secondary_structure_counts': ss_assignments,
            'secondary_structure_percentages': ss_percentages
        }
    
    def _calculate_dihedral(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom) -> float:
        """Calculate dihedral angle between four atoms"""
        
        v1 = atom1.get_coord() - atom2.get_coord()
        v2 = atom2.get_coord() - atom3.get_coord()
        v3 = atom3.get_coord() - atom4.get_coord()
        
        # Calculate normal vectors to planes
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        
        # Normalize
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        
        # Calculate dihedral angle
        dot_product = np.dot(n1, n2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Determine sign
        cross_product = np.cross(n1, n2)
        if np.dot(cross_product, v2) < 0:
            angle = -angle
        
        return np.degrees(angle)
    
    def _analyze_surface_properties(self, structure: Structure) -> Dict:
        """Analyze surface properties"""
        
        # Simplified surface analysis
        surface_residues = []
        buried_residues = []
        
        for chain in structure.get_chains():
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Standard amino acid
                    # Simple accessibility check based on neighbors
                    ca_atom = residue.get('CA')
                    if ca_atom:
                        neighbors = 0
                        for other_chain in structure.get_chains():
                            for other_residue in other_chain:
                                if other_residue != residue and other_residue.get_id()[0] == ' ':
                                    other_ca = other_residue.get('CA')
                                    if other_ca:
                                        distance = ca_atom - other_ca
                                        if distance < 8.0:  # Within 8Å
                                            neighbors += 1
                        
                        if neighbors < 12:  # Threshold for surface exposure
                            surface_residues.append(residue.get_resname())
                        else:
                            buried_residues.append(residue.get_resname())
        
        return {
            'surface_residues': len(surface_residues),
            'buried_residues': len(buried_residues),
            'surface_composition': self._analyze_composition(surface_residues),
            'buried_composition': self._analyze_composition(buried_residues)
        }
    
    def _analyze_composition(self, residue_list: List[str]) -> Dict:
        """Analyze amino acid composition"""
        
        composition = {}
        for residue in residue_list:
            composition[residue] = composition.get(residue, 0) + 1
        
        total = len(residue_list)
        percentages = {res: count/total*100 for res, count in composition.items()} if total > 0 else {}
        
        return percentages
    
    def _identify_cavities(self, structure: Structure) -> List[Dict]:
        """Identify potential binding cavities"""
        
        # Simplified cavity detection
        # Real implementation would use CASTp, fpocket, or similar
        
        cavities = []
        
        # Grid-based approach for cavity detection
        atoms = list(structure.get_atoms())
        coords = np.array([atom.get_coord() for atom in atoms])
        
        if len(coords) == 0:
            return cavities
        
        # Define search grid
        min_coords = np.min(coords, axis=0) - 5
        max_coords = np.max(coords, axis=0) + 5
        grid_spacing = 2.0
        
        # Find potential cavity centers
        cavity_centers = []
        
        for x in np.arange(min_coords[0], max_coords[0], grid_spacing):
            for y in np.arange(min_coords[1], max_coords[1], grid_spacing):
                for z in np.arange(min_coords[2], max_coords[2], grid_spacing):
                    point = np.array([x, y, z])
                    
                    # Check if point is in a cavity (not too close to atoms)
                    distances = np.linalg.norm(coords - point, axis=1)
                    min_distance = np.min(distances)
                    
                    if 2.0 < min_distance < 8.0:  # Potential cavity
                        # Count nearby atoms
                        nearby_atoms = np.sum(distances < 10.0)
                        if nearby_atoms > 5:  # Surrounded by protein
                            cavity_centers.append({
                                'center': point.tolist(),
                                'volume': grid_spacing**3,
                                'nearby_atoms': int(nearby_atoms),
                                'accessibility': float(min_distance)
                            })
        
        # Cluster nearby cavity points
        if cavity_centers:
            # Simple clustering - merge nearby points
            clustered_cavities = []
            used = set()
            
            for i, cavity in enumerate(cavity_centers):
                if i in used:
                    continue
                
                cluster = [cavity]
                used.add(i)
                
                for j, other_cavity in enumerate(cavity_centers):
                    if j != i and j not in used:
                        distance = np.linalg.norm(np.array(cavity['center']) - np.array(other_cavity['center']))
                        if distance < 5.0:  # Merge threshold
                            cluster.append(other_cavity)
                            used.add(j)
                
                # Calculate cluster properties
                if len(cluster) >= 3:  # Minimum cavity size
                    centers = np.array([c['center'] for c in cluster])
                    clustered_cavities.append({
                        'center': np.mean(centers, axis=0).tolist(),
                        'volume': len(cluster) * grid_spacing**3,
                        'druggability_score': min(1.0, len(cluster) / 20.0)
                    })
            
            cavities = sorted(clustered_cavities, key=lambda x: x['volume'], reverse=True)[:5]
        
        return cavities
    
    def _assess_quality(self, structure: Structure) -> Dict:
        """Assess structure quality metrics"""
        
        # Simplified quality assessment
        # Real implementation would check Ramachandran plots, clashes, etc.
        
        metrics = {
            'resolution': None,  # Would be extracted from PDB header
            'r_factor': None,    # Would be extracted from PDB header
            'geometry_score': np.random.uniform(0.7, 1.0),  # Placeholder
            'clash_score': np.random.uniform(0.0, 0.3),     # Placeholder
            'completeness': np.random.uniform(0.85, 1.0)    # Placeholder
        }
        
        return metrics

class LigandAnalysis:
    """Tools for ligand analysis and property prediction"""
    
    def __init__(self):
        pass
    
    def parse_smiles(self, smiles: str) -> Dict:
        """Parse SMILES string and extract basic information"""
        
        # Simplified SMILES parsing
        # Real implementation would use RDKit
        
        # Count atoms by symbol
        atom_counts = {}
        i = 0
        while i < len(smiles):
            char = smiles[i]
            if char.isupper():
                # Start of atom symbol
                atom = char
                if i + 1 < len(smiles) and smiles[i + 1].islower():
                    atom += smiles[i + 1]
                    i += 1
                
                atom_counts[atom] = atom_counts.get(atom, 0) + 1
            i += 1
        
        # Calculate molecular weight
        atomic_weights = {
            'C': 12.011, 'H': 1.008, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
            'Br': 79.904, 'I': 126.904
        }
        
        molecular_weight = sum(atomic_weights.get(atom, 0) * count for atom, count in atom_counts.items())
        
        return {
            'smiles': smiles,
            'atom_counts': atom_counts,
            'molecular_weight': molecular_weight,
            'formula': ''.join(f"{atom}{count if count > 1 else ''}" for atom, count in sorted(atom_counts.items()))
        }
    
    def predict_lipinski_properties(self, smiles: str) -> Dict:
        """Predict Lipinski Rule of Five properties"""
        
        # Simplified property prediction
        # Real implementation would use RDKit descriptors
        
        mol_info = self.parse_smiles(smiles)
        
        # Estimate properties based on composition
        mw = mol_info['molecular_weight']
        carbon_count = mol_info['atom_counts'].get('C', 0)
        nitrogen_count = mol_info['atom_counts'].get('N', 0)
        oxygen_count = mol_info['atom_counts'].get('O', 0)
        
        # Rough estimates
        logp = (carbon_count * 0.5) - (nitrogen_count * 0.8) - (oxygen_count * 1.2)
        hbd = nitrogen_count + oxygen_count  # Simplified
        hba = nitrogen_count + oxygen_count  # Simplified
        
        properties = {
            'molecular_weight': mw,
            'logp': logp,
            'hbd': hbd,
            'hba': hba,
            'rotatable_bonds': max(0, carbon_count - 5),  # Rough estimate
            'polar_surface_area': (nitrogen_count + oxygen_count) * 20  # Rough estimate
        }
        
        # Check Lipinski violations
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1
        
        properties['lipinski_violations'] = violations
        properties['drug_like'] = violations <= 1
        
        return properties

def fetch_pdb_info(pdb_id: str) -> Dict:
    """Fetch PDB metadata from RCSB PDB API"""
    
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            info = {
                'pdb_id': pdb_id.upper(),
                'title': data.get('struct', {}).get('title', 'Unknown'),
                'resolution': data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0],
                'experimental_method': data.get('exptl', [{}])[0].get('method', 'Unknown'),
                'deposition_date': data.get('rcsb_accession_info', {}).get('initial_release_date', 'Unknown'),
                'organism': data.get('rcsb_entity_source_organism', [{}])[0].get('ncbi_scientific_name', 'Unknown')
            }
            
            return info
        else:
            return {'error': f"Could not fetch PDB info for {pdb_id}"}
            
    except Exception as e:
        return {'error': f"Error fetching PDB info: {str(e)}"}

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate RMSD between two sets of coordinates"""
    
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have the same shape")
    
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd

def superimpose_structures(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[np.ndarray, float]:
    """Superimpose two structures using Kabsch algorithm"""
    
    # Center coordinates
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    
    coords1_centered = coords1 - center1
    coords2_centered = coords2 - center2
    
    # Calculate rotation matrix using SVD
    H = coords1_centered.T @ coords2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    coords1_rotated = coords1_centered @ R.T + center2
    
    # Calculate RMSD after superposition
    rmsd = calculate_rmsd(coords1_rotated, coords2)
    
    return coords1_rotated, rmsd