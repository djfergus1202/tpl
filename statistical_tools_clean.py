"""
Statistical Analysis Tools for Academic Research Platform

This module provides comprehensive statistical analysis capabilities including
meta-analysis, effect size calculations, and publication bias detection.
"""

import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import warnings

class MetaAnalysisResult(NamedTuple):
    """Results from meta-analysis calculations"""
    pooled_effect: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    z_value: float
    p_value: float
    q_statistic: float
    df: int
    i_squared: float
    tau_squared: float
    method: str

class StatisticalValidator:
    """Comprehensive statistical analysis and validation toolkit"""
    
    def __init__(self):
        self.alpha = 0.05
        self.confidence_level = 0.95
    
    def calculate_effect_size_cohens_d(self, mean1: float, mean2: float, 
                                      sd1: float, sd2: float, 
                                      n1: int, n2: int) -> Dict[str, float]:
        """Calculate Cohen's d effect size"""
        
        # Pooled standard deviation
        pooled_sd = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_sd
        
        # Standard error of Cohen's d
        se_d = math.sqrt((n1 + n2) / (n1 * n2) + (cohens_d**2) / (2 * (n1 + n2)))
        
        # Confidence interval
        t_value = stats.t.ppf(1 - self.alpha/2, n1 + n2 - 2)
        ci_lower = cohens_d - t_value * se_d
        ci_upper = cohens_d + t_value * se_d
        
        return {
            'cohens_d': cohens_d,
            'standard_error': se_d,
            'confidence_interval': (ci_lower, ci_upper),
            'interpretation': self._interpret_cohens_d(abs(cohens_d))
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "Negligible"
        elif d < 0.5:
            return "Small"
        elif d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def fixed_effects_meta_analysis(self, effect_sizes: List[float], 
                                   standard_errors: List[float]) -> MetaAnalysisResult:
        """Perform fixed-effects meta-analysis"""
        
        if len(effect_sizes) != len(standard_errors):
            raise ValueError("Effect sizes and standard errors must have the same length")
        
        # Weights (inverse variance)
        weights = [1 / (se**2) for se in standard_errors]
        total_weight = sum(weights)
        
        # Weighted mean effect size
        pooled_effect = sum(es * w for es, w in zip(effect_sizes, weights)) / total_weight
        
        # Standard error
        se_pooled = math.sqrt(1 / total_weight)
        
        # Z-value and p-value
        z_value = pooled_effect / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = pooled_effect - z_critical * se_pooled
        ci_upper = pooled_effect + z_critical * se_pooled
        
        # Heterogeneity statistics
        q_statistic = sum(w * (es - pooled_effect)**2 for es, w in zip(effect_sizes, weights))
        df = len(effect_sizes) - 1
        i_squared = max(0, (q_statistic - df) / q_statistic) if q_statistic > 0 else 0
        
        return MetaAnalysisResult(
            pooled_effect=pooled_effect,
            standard_error=se_pooled,
            confidence_interval=(ci_lower, ci_upper),
            z_value=z_value,
            p_value=p_value,
            q_statistic=q_statistic,
            df=df,
            i_squared=i_squared,
            tau_squared=0.0,  # Fixed effects assumes tau² = 0
            method="Fixed Effects"
        )
    
    def random_effects_meta_analysis(self, effect_sizes: List[float], 
                                    standard_errors: List[float],
                                    method: str = "DerSimonian-Laird") -> MetaAnalysisResult:
        """Perform random-effects meta-analysis"""
        
        # First, calculate fixed effects for Q statistic
        fixed_result = self.fixed_effects_meta_analysis(effect_sizes, standard_errors)
        
        # Calculate tau-squared (between-study variance)
        if method == "DerSimonian-Laird":
            weights_fixed = [1 / (se**2) for se in standard_errors]
            sum_weights = sum(weights_fixed)
            sum_weights_squared = sum(w**2 for w in weights_fixed)
            
            c_value = sum_weights - (sum_weights_squared / sum_weights)
            tau_squared = max(0, (fixed_result.q_statistic - fixed_result.df) / c_value)
        else:
            tau_squared = 0.0
        
        # Random effects weights
        weights_random = [1 / (se**2 + tau_squared) for se in standard_errors]
        total_weight = sum(weights_random)
        
        # Weighted mean effect size
        pooled_effect = sum(es * w for es, w in zip(effect_sizes, weights_random)) / total_weight
        
        # Standard error
        se_pooled = math.sqrt(1 / total_weight)
        
        # Z-value and p-value
        z_value = pooled_effect / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = pooled_effect - z_critical * se_pooled
        ci_upper = pooled_effect + z_critical * se_pooled
        
        return MetaAnalysisResult(
            pooled_effect=pooled_effect,
            standard_error=se_pooled,
            confidence_interval=(ci_lower, ci_upper),
            z_value=z_value,
            p_value=p_value,
            q_statistic=fixed_result.q_statistic,
            df=fixed_result.df,
            i_squared=fixed_result.i_squared,
            tau_squared=tau_squared,
            method=f"Random Effects ({method})"
        )
    
    def eggers_test(self, effect_sizes: List[float], standard_errors: List[float]) -> Dict[str, float]:
        """Perform Egger's regression test for publication bias"""
        
        if len(effect_sizes) != len(standard_errors) or len(effect_sizes) < 3:
            raise ValueError("Need at least 3 studies with matching effect sizes and standard errors")
        
        # Precision (1/SE)
        precision = [1/se for se in standard_errors]
        
        # Regression: effect_size ~ precision
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(precision, effect_sizes)
        
        # Test statistic for intercept
        t_statistic = intercept / std_err if std_err > 0 else 0
        
        return {
            'intercept': intercept,
            'slope': slope,
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def beggs_test(self, effect_sizes: List[float], standard_errors: List[float]) -> Dict[str, float]:
        """Perform Begg's rank correlation test for publication bias"""
        
        if len(effect_sizes) != len(standard_errors) or len(effect_sizes) < 3:
            raise ValueError("Need at least 3 studies with matching effect sizes and standard errors")
        
        # Standardized effect sizes
        z_scores = [es/se for es, se in zip(effect_sizes, standard_errors)]
        
        # Variance of standardized effect sizes
        variances = [se**2 for se in standard_errors]
        
        # Kendall's tau correlation
        from scipy.stats import kendalltau
        tau, p_value = kendalltau(z_scores, variances)
        
        return {
            'kendall_tau': tau,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

# Alias for backward compatibility
StatisticalAnalyzer = StatisticalValidator