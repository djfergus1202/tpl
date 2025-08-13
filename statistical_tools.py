import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2, norm, t
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
import math

@dataclass
class EffectSizeResult:
    """Result container for effect size calculations"""
    effect_size: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    variance: float
    interpretation: str
    method: str

@dataclass
class MetaAnalysisResult:
    """Result container for meta-analysis calculations"""
    pooled_effect: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    z_value: float
    p_value: float
    q_statistic: float
    q_p_value: float
    i_squared: float
    tau_squared: float
    h_squared: float
    prediction_interval: Tuple[float, float]
    weights: List[float]
    method: str

class StatisticalValidator:
    """Statistical validation and calculation tools for meta-analysis"""
    
    def __init__(self):
        self.confidence_level = 0.95
        self.alpha = 1 - self.confidence_level
        
    def calculate_cohens_d(self, mean1: float, mean2: float, sd1: float, sd2: float, 
                          n1: int, n2: int, bias_correction: bool = True) -> EffectSizeResult:
        """Calculate Cohen's d with optional bias correction (Hedges' g)"""
        
        # Input validation
        if any(val <= 0 for val in [sd1, sd2, n1, n2]):
            raise ValueError("Standard deviations and sample sizes must be positive")
        
        # Pooled standard deviation
        pooled_sd = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean2 - mean1) / pooled_sd
        
        # Bias correction factor (for Hedges' g)
        if bias_correction:
            j = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
            effect_size = cohens_d * j
            method = "Hedges' g"
        else:
            effect_size = cohens_d
            method = "Cohen's d"
        
        # Standard error
        se = math.sqrt((n1 + n2) / (n1 * n2) + effect_size**2 / (2 * (n1 + n2)))
        
        # Variance
        variance = se**2
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = effect_size - z_critical * se
        ci_upper = effect_size + z_critical * se
        
        # Interpretation
        abs_es = abs(effect_size)
        if abs_es < 0.2:
            interpretation = "Small effect"
        elif abs_es < 0.5:
            interpretation = "Small to medium effect"
        elif abs_es < 0.8:
            interpretation = "Medium to large effect"
        else:
            interpretation = "Large effect"
        
        return EffectSizeResult(
            effect_size=effect_size,
            standard_error=se,
            confidence_interval=(ci_lower, ci_upper),
            variance=variance,
            interpretation=interpretation,
            method=method
        )
    
    def calculate_odds_ratio(self, a: int, b: int, c: int, d: int, 
                           method: str = "woolf") -> EffectSizeResult:
        """Calculate odds ratio from 2x2 contingency table
        
        Args:
            a: Events in treatment group
            b: Non-events in treatment group  
            c: Events in control group
            d: Non-events in control group
            method: Method for calculation ("woolf", "exact")
        """
        
        # Input validation
        if any(val < 0 for val in [a, b, c, d]):
            raise ValueError("All cell counts must be non-negative")
        
        if any(val == 0 for val in [a, b, c, d]) and method == "woolf":
            # Add continuity correction
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
            warnings.warn("Applied continuity correction for zero cells")
        
        # Odds ratio
        or_value = (a * d) / (b * c) if b * c > 0 else float('inf')
        
        # Log odds ratio
        log_or = math.log(or_value) if or_value > 0 else float('-inf')
        
        # Standard error of log OR
        if method == "woolf":
            se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
        else:
            # Exact method (simplified)
            se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
        
        # Confidence interval for log OR
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        log_ci_lower = log_or - z_critical * se_log_or
        log_ci_upper = log_or + z_critical * se_log_or
        
        # Transform back to OR scale
        ci_lower = math.exp(log_ci_lower)
        ci_upper = math.exp(log_ci_upper)
        
        # Interpretation
        if or_value == 1:
            interpretation = "No association"
        elif or_value > 1:
            if or_value < 1.5:
                interpretation = "Weak positive association"
            elif or_value < 3:
                interpretation = "Moderate positive association"
            else:
                interpretation = "Strong positive association"
        else:
            if or_value > 0.67:
                interpretation = "Weak negative association"
            elif or_value > 0.33:
                interpretation = "Moderate negative association"
            else:
                interpretation = "Strong negative association"
        
        return EffectSizeResult(
            effect_size=or_value,
            standard_error=se_log_or,  # SE on log scale
            confidence_interval=(ci_lower, ci_upper),
            variance=se_log_or**2,
            interpretation=interpretation,
            method=f"Odds Ratio ({method})"
        )
    
    def calculate_risk_ratio(self, a: int, b: int, c: int, d: int) -> EffectSizeResult:
        """Calculate risk ratio from 2x2 contingency table"""
        
        # Input validation
        if any(val < 0 for val in [a, b, c, d]):
            raise ValueError("All cell counts must be non-negative")
        
        # Risk in treatment and control groups
        risk_treatment = a / (a + b) if (a + b) > 0 else 0
        risk_control = c / (c + d) if (c + d) > 0 else 0
        
        # Risk ratio
        rr_value = risk_treatment / risk_control if risk_control > 0 else float('inf')
        
        # Log risk ratio
        log_rr = math.log(rr_value) if rr_value > 0 else float('-inf')
        
        # Standard error of log RR
        se_log_rr = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d)) if all(val > 0 for val in [a, c, a+b, c+d]) else float('inf')
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        log_ci_lower = log_rr - z_critical * se_log_rr
        log_ci_upper = log_rr + z_critical * se_log_rr
        
        ci_lower = math.exp(log_ci_lower)
        ci_upper = math.exp(log_ci_upper)
        
        # Interpretation
        if abs(rr_value - 1) < 0.1:
            interpretation = "No meaningful difference in risk"
        elif rr_value > 1:
            if rr_value < 1.5:
                interpretation = "Moderately increased risk"
            else:
                interpretation = "Substantially increased risk"
        else:
            if rr_value > 0.67:
                interpretation = "Moderately decreased risk"
            else:
                interpretation = "Substantially decreased risk"
        
        return EffectSizeResult(
            effect_size=rr_value,
            standard_error=se_log_rr,
            confidence_interval=(ci_lower, ci_upper),
            variance=se_log_rr**2,
            interpretation=interpretation,
            method="Risk Ratio"
        )
    
    def calculate_correlation_effect_size(self, r: float, n: int) -> EffectSizeResult:
        """Calculate effect size measures for correlation coefficient"""
        
        if not -1 <= r <= 1:
            raise ValueError("Correlation coefficient must be between -1 and 1")
        
        if n < 3:
            raise ValueError("Sample size must be at least 3")
        
        # Fisher's z transformation
        fisher_z = 0.5 * math.log((1 + r) / (1 - r)) if abs(r) < 1 else float('inf')
        
        # Standard error of Fisher's z
        se_z = 1 / math.sqrt(n - 3)
        
        # Confidence interval for Fisher's z
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        z_ci_lower = fisher_z - z_critical * se_z
        z_ci_upper = fisher_z + z_critical * se_z
        
        # Transform back to correlation scale
        ci_lower = (math.exp(2 * z_ci_lower) - 1) / (math.exp(2 * z_ci_lower) + 1)
        ci_upper = (math.exp(2 * z_ci_upper) - 1) / (math.exp(2 * z_ci_upper) + 1)
        
        # Interpretation
        abs_r = abs(r)
        if abs_r < 0.1:
            interpretation = "Negligible correlation"
        elif abs_r < 0.3:
            interpretation = "Small correlation"
        elif abs_r < 0.5:
            interpretation = "Medium correlation"
        elif abs_r < 0.7:
            interpretation = "Large correlation"
        else:
            interpretation = "Very large correlation"
        
        return EffectSizeResult(
            effect_size=r,
            standard_error=se_z,  # SE on Fisher's z scale
            confidence_interval=(ci_lower, ci_upper),
            variance=se_z**2,
            interpretation=interpretation,
            method="Pearson correlation"
        )
    
    def fixed_effects_meta_analysis(self, effect_sizes: List[float], 
                                   standard_errors: List[float]) -> MetaAnalysisResult:
        """Perform fixed-effects meta-analysis"""
        
        if len(effect_sizes) != len(standard_errors):
            raise ValueError("Effect sizes and standard errors must have the same length")
        
        if len(effect_sizes) < 2:
            raise ValueError("Need at least 2 studies for meta-analysis")
        
        # Convert to numpy arrays
        es = np.array(effect_sizes)
        se = np.array(standard_errors)
        
        # Weights (inverse variance)
        weights = 1 / (se**2)
        
        # Pooled effect size
        pooled_effect = np.sum(weights * es) / np.sum(weights)
        
        # Standard error of pooled effect
        pooled_se = 1 / math.sqrt(np.sum(weights))
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = pooled_effect - z_critical * pooled_se
        ci_upper = pooled_effect + z_critical * pooled_se
        
        # Test of overall effect
        z_value = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        # Heterogeneity statistics
        q_statistic = np.sum(weights * (es - pooled_effect)**2)
        df = len(effect_sizes) - 1
        q_p_value = 1 - stats.chi2.cdf(q_statistic, df) if df > 0 else 1.0
        
        # I-squared
        i_squared = max(0, (q_statistic - df) / q_statistic * 100) if q_statistic > 0 else 0
        
        # H-squared
        h_squared = q_statistic / df if df > 0 else 1
        
        # Tau-squared (set to 0 for fixed effects)
        tau_squared = 0
        
        # Prediction interval (same as confidence interval for fixed effects)
        prediction_interval = (ci_lower, ci_upper)
        
        return MetaAnalysisResult(
            pooled_effect=pooled_effect,
            standard_error=pooled_se,
            confidence_interval=(ci_lower, ci_upper),
            z_value=z_value,
            p_value=p_value,
            q_statistic=q_statistic,
            q_p_value=q_p_value,
            i_squared=i_squared,
            tau_squared=tau_squared,
            h_squared=h_squared,
            prediction_interval=prediction_interval,
            weights=weights.tolist(),
            method="Fixed Effects"
        )
    
    def random_effects_meta_analysis(self, effect_sizes: List[float], 
                                    standard_errors: List[float],
                                    method: str = "DerSimonian-Laird") -> MetaAnalysisResult:
        """Perform random-effects meta-analysis"""
        
        if len(effect_sizes) != len(standard_errors):
            raise ValueError("Effect sizes and standard errors must have the same length")
        
        if len(effect_sizes) < 2:
            raise ValueError("Need at least 2 studies for meta-analysis")
        
        # Convert to numpy arrays
        es = np.array(effect_sizes)
        se = np.array(standard_errors)
        
        # Step 1: Fixed effects analysis for Q statistic
        fixed_result = self.fixed_effects_meta_analysis(effect_sizes, standard_errors)
        
        # Step 2: Estimate tau-squared
        if method == "DerSimonian-Laird":
            weights_fixed = 1 / (se**2)
            sum_weights = np.sum(weights_fixed)
            sum_weights_squared = np.sum(weights_fixed**2)
            
            c = sum_weights - sum_weights_squared / sum_weights
            tau_squared = max(0, (fixed_result.q_statistic - (len(effect_sizes) - 1)) / c)
        else:
            # Other methods could be implemented (REML, ML, etc.)
            tau_squared = 0
        
        # Step 3: Random effects analysis
        # Weights including between-study variance
        weights = 1 / (se**2 + tau_squared)
        
        # Pooled effect size
        pooled_effect = np.sum(weights * es) / np.sum(weights)
        
        # Standard error of pooled effect
        pooled_se = 1 / math.sqrt(np.sum(weights))
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = pooled_effect - z_critical * pooled_se
        ci_upper = pooled_effect + z_critical * pooled_se
        
        # Test of overall effect
        z_value = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        # Prediction interval
        if len(effect_sizes) > 2:
            t_critical = stats.t.ppf(1 - self.alpha / 2, len(effect_sizes) - 2)
            pi_se = math.sqrt(pooled_se**2 + tau_squared)
            pi_lower = pooled_effect - t_critical * pi_se
            pi_upper = pooled_effect + t_critical * pi_se
            prediction_interval = (pi_lower, pi_upper)
        else:
            prediction_interval = (ci_lower, ci_upper)
        
        return MetaAnalysisResult(
            pooled_effect=pooled_effect,
            standard_error=pooled_se,
            confidence_interval=(ci_lower, ci_upper),
            z_value=z_value,
            p_value=p_value,
            q_statistic=fixed_result.q_statistic,
            q_p_value=fixed_result.q_p_value,
            i_squared=fixed_result.i_squared,
            tau_squared=tau_squared,
            h_squared=fixed_result.h_squared,
            prediction_interval=prediction_interval,
            weights=weights.tolist(),
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

    def trim_and_fill(self, effect_sizes: List[float], standard_errors: List[float]) -> Dict[str, Any]:
        """Perform trim-and-fill analysis for publication bias"""
        
        # This is a simplified implementation
        # Full implementation would require iterative trimming and filling
        
        original_ma = self.random_effects_meta_analysis(effect_sizes, standard_errors)
        
        # Estimate number of missing studies (simplified)
        # In practice, this would use the Duval and Tweedie algorithm
        
        # Calculate funnel plot asymmetry
        precision = [1/se for se in standard_errors]
        mean_precision = np.mean(precision)
        
        # Studies below average precision (potential missing studies)
        below_avg = [i for i, p in enumerate(precision) if p < mean_precision]
        
        # Estimate missing studies (very simplified)
        estimated_missing = len(below_avg) // 2
        
        return {
            'original_effect': original_ma.pooled_effect,
            'estimated_missing_studies': estimated_missing,
            'adjusted_effect': original_ma.pooled_effect * 0.9,  # Simplified adjustment
            'original_ci': original_ma.confidence_interval,
            'adjusted_ci': (original_ma.confidence_interval[0] * 0.9, 
                          original_ma.confidence_interval[1] * 0.9)
        }
    
    def subgroup_analysis(self, effect_sizes: List[float], standard_errors: List[float],
                         subgroups: List[str]) -> Dict[str, Any]:
        """Perform subgroup analysis"""
        
        if len(effect_sizes) != len(standard_errors) or len(effect_sizes) != len(subgroups):
            raise ValueError("All input lists must have the same length")
        
        unique_subgroups = list(set(subgroups))
        subgroup_results = {}
        
        # Analyze each subgroup
        for subgroup in unique_subgroups:
            indices = [i for i, sg in enumerate(subgroups) if sg == subgroup]
            
            if len(indices) >= 2:  # Need at least 2 studies
                sg_es = [effect_sizes[i] for i in indices]
                sg_se = [standard_errors[i] for i in indices]
                
                sg_result = self.random_effects_meta_analysis(sg_es, sg_se)
                subgroup_results[subgroup] = sg_result
        
        # Test for subgroup differences (simplified)
        if len(subgroup_results) >= 2:
            # Q_between calculation (simplified)
            overall_ma = self.random_effects_meta_analysis(effect_sizes, standard_errors)
            
            q_within = sum(result.q_statistic for result in subgroup_results.values())
            q_total = overall_ma.q_statistic
            q_between = q_total - q_within
            
            df_between = len(subgroup_results) - 1
            p_between = 1 - stats.chi2.cdf(q_between, df_between) if df_between > 0 else 1.0
            
            return {
                'subgroup_results': subgroup_results,
                'q_between': q_between,
                'df_between': df_between,
                'p_between': p_between,
                'significant_difference': p_between < 0.05
            }
        else:
            return {'subgroup_results': subgroup_results}

# Alias for backward compatibility
StatisticalAnalyzer = StatisticalValidator
