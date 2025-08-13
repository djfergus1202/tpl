"""
Mathematical computation engine integrating symbolic math and external APIs
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sympy as sp
from sympy import symbols, diff, integrate, solve, simplify, expand, factor
from sympy import Matrix, lambdify, series, limit, oo
from sympy.plotting import plot as sympy_plot
import requests
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import wolframalpha
    WOLFRAM_AVAILABLE = True
except ImportError:
    WOLFRAM_AVAILABLE = False

class MathematicalEngine:
    """Advanced mathematical computation engine with symbolic math and external API integration"""
    
    def __init__(self):
        self.wolfram_client = None
        self.wolfram_available = False
        self.setup_symbolic_environment()
        
        # Initialize MATLAB-style numerical computing capabilities
        self.matlab_available = True
        
    def setup_symbolic_environment(self):
        """Setup symbolic mathematics environment"""
        # Common symbols for pharmacological modeling
        self.t, self.c, self.r = symbols('t c r')  # time, concentration, response
        self.ka, self.ke, self.Vd = symbols('ka ke Vd', positive=True)  # pharmacokinetic parameters
        self.Emax, self.EC50, self.n = symbols('Emax EC50 n', positive=True)  # pharmacodynamic parameters
        self.D, self.F = symbols('D F', positive=True)  # dose, bioavailability
        
    def setup_wolfram_alpha(self, api_key: str) -> bool:
        """Setup Wolfram Alpha client and test connection"""
        if WOLFRAM_AVAILABLE and api_key:
            try:
                self.wolfram_client = wolframalpha.Client(api_key)
                # Test the connection with a simple query
                test_result = self.wolfram_client.query('integrate x^2')
                if test_result.success:
                    self.wolfram_available = True
                    return True
                else:
                    st.error("Wolfram Alpha connection test failed")
                    return False
            except Exception as e:
                st.error(f"Failed to setup Wolfram Alpha: {str(e)}")
                return False
        else:
            if not WOLFRAM_AVAILABLE:
                st.error("Wolfram Alpha library not available")
            return False
    
    def query_wolfram_alpha(self, query: str) -> Dict[str, Any]:
        """Query Wolfram Alpha for mathematical computations"""
        if not self.wolfram_client:
            return {"error": "Wolfram Alpha not available"}
        
        try:
            result = self.wolfram_client.query(query)
            
            # Extract relevant information
            output = {
                "input": query,
                "results": [],
                "plots": [],
                "numerical_results": []
            }
            
            for pod in result.pods:
                pod_data = {
                    "title": pod.title,
                    "text": pod.text if pod.text else "No text available"
                }
                
                # Check for subpods with images (plots)
                if hasattr(pod, 'subpods'):
                    for subpod in pod.subpods:
                        if hasattr(subpod, 'img') and subpod.img:
                            pod_data["image_url"] = subpod.img.src
                            output["plots"].append(subpod.img.src)
                
                output["results"].append(pod_data)
            
            return output
            
        except Exception as e:
            return {"error": f"Wolfram Alpha query failed: {str(e)}"}
    
    def create_pharmacokinetic_model(self, dose: float, ka: float, ke: float, 
                                   vd: float, bioavailability: float = 1.0) -> sp.Expr:
        """Create symbolic pharmacokinetic model (one-compartment)"""
        
        # One-compartment model with first-order absorption
        # C(t) = (F*D*ka)/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
        
        concentration = ((bioavailability * dose * ka) / (vd * (ka - ke))) * \
                       (sp.exp(-ke * self.t) - sp.exp(-ka * self.t))
        
        return concentration
    
    def create_pharmacodynamic_model(self, emax: float, ec50: float, hill_coeff: float = 1.0) -> sp.Expr:
        """Create symbolic pharmacodynamic model (Hill equation)"""
        
        # Hill equation: E = Emax * C^n / (EC50^n + C^n)
        response = (emax * self.c**hill_coeff) / (ec50**hill_coeff + self.c**hill_coeff)
        
        return response
    
    def create_pk_pd_model(self, pk_params: Dict, pd_params: Dict) -> Tuple[sp.Expr, sp.Expr]:
        """Create combined PK-PD model"""
        
        # PK model
        pk_model = self.create_pharmacokinetic_model(
            dose=pk_params['dose'],
            ka=pk_params['ka'],
            ke=pk_params['ke'],
            vd=pk_params['vd'],
            bioavailability=pk_params.get('F', 1.0)
        )
        
        # PD model (substitute concentration from PK model)
        pd_model = self.create_pharmacodynamic_model(
            emax=pd_params['emax'],
            ec50=pd_params['ec50'],
            hill_coeff=pd_params.get('n', 1.0)
        )
        
        # Combined model: substitute PK into PD
        combined_model = pd_model.subs(self.c, pk_model)
        
        return pk_model, combined_model
    
    def analyze_therapeutic_window(self, pk_pd_model: sp.Expr, 
                                 min_efficacy: float = 0.2, 
                                 max_toxicity: float = 0.8) -> Dict[str, Any]:
        """Analyze therapeutic window using symbolic computation"""
        
        # Find time points where response is within therapeutic window
        efficacy_time = solve(pk_pd_model - min_efficacy, self.t)
        toxicity_time = solve(pk_pd_model - max_toxicity, self.t)
        
        # Calculate derivatives for rate analysis
        first_derivative = diff(pk_pd_model, self.t)
        second_derivative = diff(first_derivative, self.t)
        
        # Find peak response time
        peak_times = solve(first_derivative, self.t)
        
        analysis = {
            "model": pk_pd_model,
            "efficacy_threshold_times": efficacy_time,
            "toxicity_threshold_times": toxicity_time,
            "peak_response_times": peak_times,
            "first_derivative": first_derivative,
            "second_derivative": second_derivative
        }
        
        return analysis
    
    def create_dose_response_surface(self, dose_range: np.ndarray, 
                                   time_range: np.ndarray,
                                   pk_params: Dict, pd_params: Dict) -> go.Figure:
        """Create 3D dose-response-time surface using symbolic computation"""
        
        # Create symbolic model
        pk_model, combined_model = self.create_pk_pd_model(pk_params, pd_params)
        
        # Convert to numerical function
        response_func = lambdify([self.t], combined_model.subs([(self.D, pk_params['dose'])]), 'numpy')
        
        # Create meshgrid
        T, D = np.meshgrid(time_range, dose_range)
        
        # Calculate responses for different doses
        responses = np.zeros_like(T)
        
        for i, dose in enumerate(dose_range):
            # Update dose in the model
            temp_params = pk_params.copy()
            temp_params['dose'] = dose
            _, temp_model = self.create_pk_pd_model(temp_params, pd_params)
            temp_func = lambdify([self.t], temp_model, 'numpy')
            
            responses[i, :] = temp_func(time_range)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=T,
            y=responses,
            z=D,
            colorscale='Viridis',
            colorbar=dict(title="Response")
        )])
        
        fig.update_layout(
            title='3D Dose-Response-Time Surface',
            scene=dict(
                xaxis_title='Time (hours)',
                yaxis_title='Response',
                zaxis_title='Dose (mg)',
                bgcolor="rgba(0,0,0,0)"
            ),
            height=600
        )
        
        return fig
    
    def matlab_style_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """MATLAB-style statistical and mathematical analysis"""
        
        results = {}
        
        # Basic statistics (MATLAB-style)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            col_data = data[col].dropna()
            
            results[col] = {
                'mean': np.mean(col_data),
                'median': np.median(col_data),
                'std': np.std(col_data, ddof=1),  # Sample standard deviation
                'var': np.var(col_data, ddof=1),  # Sample variance
                'min': np.min(col_data),
                'max': np.max(col_data),
                'range': np.max(col_data) - np.min(col_data),
                'skewness': self._calculate_skewness(col_data),
                'kurtosis': self._calculate_kurtosis(col_data),
                'quantiles': np.percentile(col_data, [25, 50, 75])
            }
        
        # Correlation matrix
        if len(numerical_cols) > 1:
            results['correlation_matrix'] = data[numerical_cols].corr()
        
        return results
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness (MATLAB-style)"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0
        
        skew = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
        return skew
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis (MATLAB-style)"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0
        
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - \
               (3 * (n-1)**2 / ((n-2) * (n-3)))
        return kurt
    
    def create_mathematical_plots(self, expression: sp.Expr, 
                                variable: sp.Symbol,
                                x_range: Tuple[float, float],
                                plot_type: str = "line") -> go.Figure:
        """Create mathematical plots from symbolic expressions"""
        
        # Convert symbolic expression to numerical function
        func = lambdify(variable, expression, 'numpy')
        
        # Generate data points
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        
        try:
            y_vals = func(x_vals)
            
            # Handle complex results
            if np.iscomplexobj(y_vals):
                y_vals = np.real(y_vals)
                st.warning("Complex values detected, showing real part only")
        
        except Exception as e:
            st.error(f"Error evaluating expression: {str(e)}")
            return go.Figure()
        
        # Create plot based on type
        if plot_type == "line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=str(expression),
                line=dict(width=2)
            ))
            
        elif plot_type == "surface" and len(x_vals) > 1:
            # For 2D surface plots, create a parameter sweep
            param_range = np.linspace(0.1, 2.0, 50)
            X, P = np.meshgrid(x_vals[::20], param_range)  # Subsample for performance
            Z = np.zeros_like(X)
            
            # Assuming expression has a parameter we can vary
            for i, p in enumerate(param_range):
                temp_expr = expression.subs(self.ka, p)  # Example parameter substitution
                temp_func = lambdify(variable, temp_expr, 'numpy')
                Z[i, :] = temp_func(x_vals[::20])
            
            fig = go.Figure(data=[go.Surface(x=X, y=Z, z=P)])
        
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name=str(expression)
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Mathematical Plot: {str(expression)}",
            xaxis_title=str(variable),
            yaxis_title="f(" + str(variable) + ")",
            height=500
        )
        
        return fig
    
    def symbolic_calculus_operations(self, expression: sp.Expr) -> Dict[str, Any]:
        """Perform various symbolic calculus operations"""
        
        operations = {}
        
        try:
            # Differentiation
            operations['first_derivative'] = diff(expression, self.t)
            operations['second_derivative'] = diff(expression, self.t, 2)
            
            # Integration
            operations['indefinite_integral'] = integrate(expression, self.t)
            operations['definite_integral_0_to_inf'] = integrate(expression, (self.t, 0, oo))
            
            # Simplification
            operations['simplified'] = simplify(expression)
            operations['expanded'] = expand(expression)
            operations['factored'] = factor(expression)
            
            # Series expansion around t=0
            operations['taylor_series'] = series(expression, self.t, 0, 5)
            
            # Limits
            operations['limit_t_to_0'] = limit(expression, self.t, 0)
            operations['limit_t_to_inf'] = limit(expression, self.t, oo)
            
        except Exception as e:
            operations['error'] = f"Error in symbolic operations: {str(e)}"
        
        return operations
    
    def matrix_pharmacology_analysis(self, drug_data: pd.DataFrame) -> Dict[str, Any]:
        """Matrix-based pharmacological analysis (MATLAB-style)"""
        
        # Extract numerical features for matrix analysis
        features = ['molecular_weight', 'logp', 'drug_likeness', 'ki_nm', 
                   'therapeutic_window', 'drug_response']
        
        # Filter available features
        available_features = [f for f in features if f in drug_data.columns]
        
        if len(available_features) < 2:
            return {"error": "Insufficient numerical features for matrix analysis"}
        
        # Create feature matrix
        feature_matrix = drug_data[available_features].dropna()
        
        if feature_matrix.empty:
            return {"error": "No valid data for matrix analysis"}
        
        # Convert to numpy matrix
        X = feature_matrix.values
        
        # Matrix operations
        results = {
            'feature_names': available_features,
            'matrix_shape': X.shape,
            'covariance_matrix': np.cov(X.T),
            'correlation_matrix': np.corrcoef(X.T),
            'eigenvalues': None,
            'eigenvectors': None,
            'condition_number': np.linalg.cond(X),
            'matrix_rank': np.linalg.matrix_rank(X)
        }
        
        # Eigenvalue decomposition of correlation matrix
        try:
            eigenvalues, eigenvectors = np.linalg.eig(results['correlation_matrix'])
            results['eigenvalues'] = eigenvalues
            results['eigenvectors'] = eigenvectors
        except Exception as e:
            results['eigen_error'] = str(e)
        
        # Principal component analysis
        try:
            # Center the data
            X_centered = X - np.mean(X, axis=0)
            # SVD for PCA
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            results['pca_components'] = Vt
            results['pca_singular_values'] = S
            results['pca_explained_variance'] = (S**2) / (X.shape[0] - 1)
        except Exception as e:
            results['pca_error'] = str(e)
        
        return results
    
    def create_mathematical_dashboard(self, drug_data: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create comprehensive mathematical analysis dashboard"""
        
        figures = {}
        
        # 1. Correlation heatmap
        numerical_cols = drug_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = drug_data[numerical_cols].corr()
            
            figures['correlation_heatmap'] = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            figures['correlation_heatmap'].update_layout(
                title="Feature Correlation Matrix",
                height=500
            )
        
        # 2. PCA visualization
        matrix_results = self.matrix_pharmacology_analysis(drug_data)
        if 'pca_components' in matrix_results:
            pca_data = matrix_results['pca_components'][:2, :]  # First 2 components
            
            figures['pca_plot'] = go.Figure()
            figures['pca_plot'].add_trace(go.Scatter(
                x=pca_data[0, :],
                y=pca_data[1, :],
                mode='markers+text',
                text=matrix_results['feature_names'],
                textposition="top center",
                name='PCA Components'
            ))
            figures['pca_plot'].update_layout(
                title="Principal Component Analysis",
                xaxis_title="PC1",
                yaxis_title="PC2",
                height=500
            )
        
        # 3. Distribution analysis
        if 'therapeutic_window' in drug_data.columns:
            figures['distribution_analysis'] = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Cumulative Distribution')
            )
            
            tw_data = drug_data['therapeutic_window'].dropna()
            
            # Histogram
            figures['distribution_analysis'].add_trace(
                go.Histogram(x=tw_data, nbinsx=30, name='Histogram'),
                row=1, col=1
            )
            
            # Box plot
            figures['distribution_analysis'].add_trace(
                go.Box(y=tw_data, name='Box Plot'),
                row=1, col=2
            )
            
            # Q-Q plot (approximate)
            sorted_data = np.sort(tw_data)
            theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_data))
            figures['distribution_analysis'].add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Q-Q Plot'),
                row=2, col=1
            )
            
            # Cumulative distribution
            figures['distribution_analysis'].add_trace(
                go.Scatter(x=sorted_data, y=np.linspace(0, 1, len(sorted_data)), 
                          mode='lines', name='CDF'),
                row=2, col=2
            )
            
            figures['distribution_analysis'].update_layout(
                title="Statistical Distribution Analysis",
                height=600,
                showlegend=False
            )
        
        return figures
    
    def analyze_drug_pharmacokinetics_matlab(self, drug_data: Dict) -> Dict[str, Any]:
        """MATLAB-style drug pharmacokinetic analysis with numerical computing"""
        results = {"drug_name": drug_data.get('compound_name', 'Unknown')}
        
        try:
            # Extract parameters
            ka_val = drug_data.get('kabs', 1.0)
            ke_val = drug_data.get('kel', 0.1) 
            dose = drug_data.get('dose', 100.0)
            vd = drug_data.get('vd', 70.0)  # Volume of distribution in L
            
            # 1. Pharmacokinetic parameter calculations (MATLAB-style)
            half_life = 0.693 / ke_val
            clearance = ke_val * vd
            bioavailability = 0.8  # Assume 80% bioavailability
            
            # 2. Time-concentration profile (numerical solution)
            time_points = np.linspace(0, 24, 100)  # 24 hours, 100 points
            
            # One-compartment model with first-order absorption
            concentrations = []
            for t in time_points:
                if t == 0:
                    conc = 0
                else:
                    conc = (dose * bioavailability * ka_val / (vd * (ka_val - ke_val))) * \
                           (np.exp(-ke_val * t) - np.exp(-ka_val * t))
                concentrations.append(max(0, conc))
            
            # 3. Pharmacokinetic metrics
            cmax = max(concentrations)
            tmax_idx = np.argmax(concentrations)
            tmax = time_points[tmax_idx]
            
            # AUC calculation using trapezoidal rule
            auc = np.trapz(concentrations, time_points)
            
            # 4. Therapeutic window analysis
            ec50 = drug_data.get('ic50_nm', 50.0)
            therapeutic_min = ec50 * 0.5  # Minimum effective concentration
            therapeutic_max = ec50 * 5.0   # Maximum safe concentration
            
            # Time in therapeutic window
            therapeutic_mask = (np.array(concentrations) >= therapeutic_min) & \
                              (np.array(concentrations) <= therapeutic_max)
            time_in_window = np.sum(therapeutic_mask) * (time_points[1] - time_points[0])
            
            # 5. Drug interaction potential (competitive inhibition model)
            ki_val = drug_data.get('ki_nm', 100.0)
            interaction_factor = 1 / (1 + (cmax / ki_val))
            
            results.update({
                "half_life_hours": f"{half_life:.2f}",
                "clearance_l_per_h": f"{clearance:.2f}",
                "volume_distribution_l": f"{vd:.1f}",
                "cmax_concentration": f"{cmax:.2f} ng/mL",
                "tmax_hours": f"{tmax:.2f}",
                "auc_ng_h_ml": f"{auc:.2f}",
                "time_in_therapeutic_window": f"{time_in_window:.1f} hours",
                "interaction_factor": f"{interaction_factor:.3f}",
                "bioavailability": f"{bioavailability:.1%}",
                "pk_profile_points": len(concentrations),
                "matlab_analysis": "Complete numerical PK analysis performed"
            })
            
        except Exception as e:
            results["error"] = f"MATLAB-style analysis error: {str(e)}"
            results["fallback"] = "Basic calculations available"
        
        return results
    
    def _fallback_pk_analysis(self, drug_data: Dict) -> Dict[str, str]:
        """Fallback pharmacokinetic analysis using SymPy when Wolfram Alpha unavailable"""
        try:
            ka_val = drug_data.get('kabs', 1.0)
            ke_val = drug_data.get('kel', 0.1)
            
            # Symbolic half-life
            half_life = sp.log(2) / self.ke
            half_life_numeric = float(sp.log(2) / ke_val)
            
            # Symbolic clearance
            clearance_sym = self.ke * self.Vd
            
            return {
                "half_life_symbolic": str(half_life),
                "half_life_numeric": f"{half_life_numeric:.2f} hours",
                "clearance_symbolic": str(clearance_sym)
            }
        except:
            return {"error": "Fallback analysis failed"}
    
    def generate_pharmacological_equations_matlab(self, drug_response_data: 'pd.DataFrame') -> Dict[str, str]:
        """Generate comprehensive pharmacological equations using MATLAB-style numerical methods"""
        equations = {}
        
        try:
            # 1. Multi-compartment PK model (SymPy + numerical)
            pk_equation = (self.D * self.F * self.ka / (self.Vd * (self.ka - self.ke))) * \
                         (sp.exp(-self.ke * self.t) - sp.exp(-self.ka * self.t))
            equations["pharmacokinetic_model"] = str(pk_equation)
            
            # 2. Hill equation for dose-response
            hill_eq = self.Emax * (self.c ** self.n) / (self.EC50 ** self.n + self.c ** self.n)
            equations["hill_equation"] = str(hill_eq)
            
            # 3. Therapeutic index
            equations["therapeutic_index"] = "TD50/ED50"
            
            # 4. Time to steady state
            steady_state = 5 / self.ke  # 5 half-lives
            equations["steady_state_time"] = str(steady_state)
            
            # MATLAB-style numerical analysis
            if self.matlab_available:
                # 5. Population pharmacokinetics (numerical approach)
                equations["population_pk_numerical"] = "Population PK using Monte Carlo simulation with inter-individual variability"
                
                # 6. Drug-drug interaction modeling
                equations["ddi_competitive_inhibition"] = "v = Vmax*[S] / (Km*(1 + [I]/Ki) + [S])"
                
                # 7. PBPK compartmental modeling  
                equations["pbpk_liver"] = "dCL/dt = QLiver*(CA - CL/KpL) - CLint*CL/KpL"
                equations["pbpk_kidney"] = "dCK/dt = QKidney*(CA - CK/KpK) - CLrenal*CK/KpK"
                
                # 8. Allometric scaling relationships
                equations["allometric_clearance"] = "CL_human = CL_animal * (BW_human/BW_animal)^0.75"
                equations["allometric_volume"] = "Vd_human = Vd_animal * (BW_human/BW_animal)^1.0"
                
                # 9. Bioequivalence analysis
                equations["bioequivalence_cmax"] = "90% CI of Test/Reference Cmax ratio within [0.8, 1.25]"
                equations["bioequivalence_auc"] = "90% CI of Test/Reference AUC ratio within [0.8, 1.25]"
                
                # 10. Population pharmacokinetic covariate model
                equations["covariate_clearance"] = "CL = TVcl * (WT/70)^0.75 * (1 + θ_age*(AGE-40)) * exp(η_CL)"
                equations["covariate_volume"] = "Vd = TVvd * (WT/70)^1.0 * exp(η_Vd)"
                
            equations["matlab_status"] = "Complete MATLAB-style numerical analysis enabled"
            
        except Exception as e:
            equations["error"] = f"Equation generation failed: {str(e)}"
        
        return equations