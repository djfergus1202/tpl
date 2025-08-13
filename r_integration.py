import subprocess
import tempfile
import os
import logging
from typing import Optional, Dict, Any, List
import streamlit as st

class RAnalytics:
    """R Analytics integration for statistical computing"""
    
    def __init__(self):
        self.r_available = self._check_r_installation()
        self.required_packages = [
            'meta', 'metafor', 'dmetar', 'robvis', 'forestplot',
            'dplyr', 'tidyverse', 'ggplot2', 'compute.es', 'effsize'
        ]
        
        if self.r_available:
            self._install_required_packages()
    
    def _check_r_installation(self) -> bool:
        """Check if R is installed and accessible"""
        try:
            result = subprocess.run(['R', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    def _install_required_packages(self):
        """Install required R packages"""
        try:
            install_script = f"""
            # Function to install packages if not already installed
            install_if_missing <- function(packages) {{
                for (pkg in packages) {{
                    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {{
                        install.packages(pkg, repos = "https://cran.r-project.org/", quiet = TRUE)
                        library(pkg, character.only = TRUE)
                    }}
                }}
            }}
            
            # Install required packages
            packages <- c({', '.join([f'"{pkg}"' for pkg in self.required_packages])})
            install_if_missing(packages)
            
            cat("R packages installation completed\\n")
            """
            
            self.execute_r_code(install_script, timeout=300)  # 5 minutes timeout for installation
            
        except Exception as e:
            st.warning(f"Could not install some R packages: {str(e)}")
    
    def execute_r_code(self, r_code: str, timeout: int = 60) -> str:
        """Execute R code and return output"""
        
        if not self.r_available:
            raise Exception("R is not available. Please install R to use statistical functions.")
        
        try:
            # Create temporary R script file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
                f.write(r_code)
                r_script_path = f.name
            
            # Execute R script
            result = subprocess.run(
                ['Rscript', '--vanilla', r_script_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Clean up temporary file
            os.unlink(r_script_path)
            
            if result.returncode != 0:
                error_msg = f"R execution error:\n{result.stderr}"
                raise Exception(error_msg)
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            raise Exception(f"R execution timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Error executing R code: {str(e)}")
    
    def check_r_status(self) -> bool:
        """Check R environment status"""
        return self.r_available
    
    def meta_analysis(self, effect_sizes: List[float], standard_errors: List[float], 
                     study_labels: List[str], method: str = "REML") -> Dict[str, Any]:
        """Perform meta-analysis using metafor package"""
        
        if len(effect_sizes) != len(standard_errors) or len(effect_sizes) != len(study_labels):
            raise ValueError("All input lists must have the same length")
        
        r_code = f"""
        library(metafor)
        library(meta)
        
        # Prepare data
        effect_sizes <- c({', '.join(map(str, effect_sizes))})
        standard_errors <- c({', '.join(map(str, standard_errors))})
        study_labels <- c({', '.join([f'"{label}"' for label in study_labels])})
        
        # Perform meta-analysis
        ma_result <- rma(yi = effect_sizes, sei = standard_errors, 
                        slab = study_labels, method = "{method}")
        
        # Extract results
        cat("Meta-Analysis Results\\n")
        cat("===================\\n")
        cat("Overall effect size:", round(ma_result$beta, 4), "\\n")
        cat("Standard error:", round(ma_result$se, 4), "\\n")
        cat("95% CI: [", round(ma_result$ci.lb, 4), ",", round(ma_result$ci.ub, 4), "]\\n")
        cat("Z-value:", round(ma_result$zval, 4), "\\n")
        cat("P-value:", round(ma_result$pval, 4), "\\n")
        cat("\\n")
        cat("Heterogeneity Statistics\\n")
        cat("=======================\\n")
        cat("Tau-squared:", round(ma_result$tau2, 4), "\\n")
        cat("I-squared:", round(ma_result$I2, 2), "%\\n")
        cat("H-squared:", round(ma_result$H2, 4), "\\n")
        cat("Q-statistic:", round(ma_result$QE, 4), "\\n")
        cat("Q p-value:", round(ma_result$QEp, 4), "\\n")
        """
        
        return self.execute_r_code(r_code)
    
    def publication_bias_tests(self, effect_sizes: List[float], 
                             standard_errors: List[float]) -> str:
        """Run publication bias tests"""
        
        r_code = f"""
        library(metafor)
        library(meta)
        
        # Prepare data
        effect_sizes <- c({', '.join(map(str, effect_sizes))})
        standard_errors <- c({', '.join(map(str, standard_errors))})
        
        # Meta-analysis for bias tests
        ma <- rma(yi = effect_sizes, sei = standard_errors)
        
        cat("Publication Bias Assessment\\n")
        cat("==========================\\n")
        
        # Egger's test
        egger_test <- regtest(ma, model = "lm")
        cat("Egger's Regression Test\\n")
        cat("Intercept:", round(egger_test$zval, 4), "\\n")
        cat("P-value:", round(egger_test$pval, 4), "\\n")
        
        if (egger_test$pval < 0.05) {{
            cat("Interpretation: Significant asymmetry detected (p < 0.05)\\n")
        }} else {{
            cat("Interpretation: No significant asymmetry detected (p >= 0.05)\\n")
        }}
        
        cat("\\n")
        
        # Rank correlation test (Begg's test)
        begg_test <- ranktest(ma)
        cat("Begg's Rank Correlation Test\\n")
        cat("Kendall's tau:", round(begg_test$tau, 4), "\\n")
        cat("P-value:", round(begg_test$pval, 4), "\\n")
        
        if (begg_test$pval < 0.05) {{
            cat("Interpretation: Significant publication bias detected (p < 0.05)\\n")
        }} else {{
            cat("Interpretation: No significant publication bias detected (p >= 0.05)\\n")
        }}
        
        cat("\\n")
        
        # Trim and fill
        tf_result <- trimfill(ma)
        cat("Trim and Fill Analysis\\n")
        cat("Estimated missing studies:", tf_result$k0, "\\n")
        cat("Original effect size:", round(ma$beta, 4), "\\n")
        cat("Adjusted effect size:", round(tf_result$beta, 4), "\\n")
        """
        
        return self.execute_r_code(r_code)
    
    def effect_size_calculation(self, data_type: str, **kwargs) -> str:
        """Calculate effect sizes based on data type"""
        
        if data_type == "continuous":
            # Cohen's d or Hedges' g
            r_code = f"""
            library(compute.es)
            library(effsize)
            
            # Calculate effect size from means and SDs
            n1 <- {kwargs.get('n1', 30)}
            n2 <- {kwargs.get('n2', 30)}
            mean1 <- {kwargs.get('mean1', 10)}
            mean2 <- {kwargs.get('mean2', 12)}
            sd1 <- {kwargs.get('sd1', 2)}
            sd2 <- {kwargs.get('sd2', 2)}
            
            # Cohen's d
            pooled_sd <- sqrt(((n1-1)*sd1^2 + (n2-1)*sd2^2) / (n1+n2-2))
            cohens_d <- (mean2 - mean1) / pooled_sd
            
            # Hedges' g (bias-corrected)
            hedges_g <- cohens_d * (1 - 3/(4*(n1+n2-2)))
            
            # Standard error
            se_d <- sqrt((n1+n2)/(n1*n2) + cohens_d^2/(2*(n1+n2)))
            
            # Confidence intervals
            ci_lower <- cohens_d - 1.96 * se_d
            ci_upper <- cohens_d + 1.96 * se_d
            
            cat("Effect Size Calculations (Continuous Data)\\n")
            cat("=========================================\\n")
            cat("Cohen's d:", round(cohens_d, 4), "\\n")
            cat("Hedges' g:", round(hedges_g, 4), "\\n")
            cat("Standard Error:", round(se_d, 4), "\\n")
            cat("95% CI: [", round(ci_lower, 4), ",", round(ci_upper, 4), "]\\n")
            
            # Interpretation
            if (abs(cohens_d) < 0.2) {{
                interpretation <- "Small effect"
            }} else if (abs(cohens_d) < 0.5) {{
                interpretation <- "Small to medium effect"
            }} else if (abs(cohens_d) < 0.8) {{
                interpretation <- "Medium to large effect"
            }} else {{
                interpretation <- "Large effect"
            }}
            
            cat("Interpretation:", interpretation, "\\n")
            """
            
        elif data_type == "dichotomous":
            # Odds ratio or risk ratio
            r_code = f"""
            library(compute.es)
            
            # Calculate effect size from 2x2 table
            a <- {kwargs.get('a', 20)}  # events in treatment
            b <- {kwargs.get('b', 10)}  # non-events in treatment
            c <- {kwargs.get('c', 10)}  # events in control
            d <- {kwargs.get('d', 20)}  # non-events in control
            
            # Odds ratio
            or_value <- (a * d) / (b * c)
            log_or <- log(or_value)
            se_log_or <- sqrt(1/a + 1/b + 1/c + 1/d)
            
            # Confidence intervals
            ci_lower <- exp(log_or - 1.96 * se_log_or)
            ci_upper <- exp(log_or + 1.96 * se_log_or)
            
            # Risk ratio
            risk_treat <- a / (a + b)
            risk_control <- c / (c + d)
            rr_value <- risk_treat / risk_control
            
            cat("Effect Size Calculations (Dichotomous Data)\\n")
            cat("==========================================\\n")
            cat("Odds Ratio:", round(or_value, 4), "\\n")
            cat("Log Odds Ratio:", round(log_or, 4), "\\n")
            cat("Standard Error (log OR):", round(se_log_or, 4), "\\n")
            cat("95% CI: [", round(ci_lower, 4), ",", round(ci_upper, 4), "]\\n")
            cat("Risk Ratio:", round(rr_value, 4), "\\n")
            cat("Treatment Risk:", round(risk_treat, 4), "\\n")
            cat("Control Risk:", round(risk_control, 4), "\\n")
            """
        
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        return self.execute_r_code(r_code)
    
    def heterogeneity_analysis(self, effect_sizes: List[float], 
                             standard_errors: List[float]) -> str:
        """Analyze heterogeneity in meta-analysis"""
        
        r_code = f"""
        library(metafor)
        library(meta)
        
        # Prepare data
        effect_sizes <- c({', '.join(map(str, effect_sizes))})
        standard_errors <- c({', '.join(map(str, standard_errors))})
        
        # Meta-analysis
        ma <- rma(yi = effect_sizes, sei = standard_errors)
        
        cat("Heterogeneity Analysis\\n")
        cat("=====================\\n")
        
        # Q-test
        cat("Cochran's Q-test\\n")
        cat("Q-statistic:", round(ma$QE, 4), "\\n")
        cat("Degrees of freedom:", ma$k - ma$p, "\\n")
        cat("P-value:", round(ma$QEp, 4), "\\n")
        
        if (ma$QEp < 0.05) {{
            cat("Interpretation: Significant heterogeneity (p < 0.05)\\n")
        }} else {{
            cat("Interpretation: No significant heterogeneity (p >= 0.05)\\n")
        }}
        
        cat("\\n")
        
        # I-squared
        cat("I-squared statistic\\n")
        cat("I²:", round(ma$I2, 2), "%\\n")
        
        if (ma$I2 < 25) {{
            i2_interpretation <- "Low heterogeneity"
        }} else if (ma$I2 < 50) {{
            i2_interpretation <- "Moderate heterogeneity"
        }} else if (ma$I2 < 75) {{
            i2_interpretation <- "Substantial heterogeneity"
        }} else {{
            i2_interpretation <- "Considerable heterogeneity"
        }}
        
        cat("Interpretation:", i2_interpretation, "\\n")
        
        cat("\\n")
        
        # Tau-squared
        cat("Between-study variance\\n")
        cat("Tau²:", round(ma$tau2, 4), "\\n")
        cat("Tau:", round(sqrt(ma$tau2), 4), "\\n")
        
        # H-statistic
        cat("H-statistic:", round(sqrt(ma$H2), 4), "\\n")
        """
        
        return self.execute_r_code(r_code)
    
    def sensitivity_analysis(self, effect_sizes: List[float], 
                           standard_errors: List[float],
                           study_labels: List[str]) -> str:
        """Perform leave-one-out sensitivity analysis"""
        
        r_code = f"""
        library(metafor)
        
        # Prepare data
        effect_sizes <- c({', '.join(map(str, effect_sizes))})
        standard_errors <- c({', '.join(map(str, standard_errors))})
        study_labels <- c({', '.join([f'"{label}"' for label in study_labels])})
        
        # Original meta-analysis
        ma_full <- rma(yi = effect_sizes, sei = standard_errors, slab = study_labels)
        
        cat("Sensitivity Analysis (Leave-One-Out)\\n")
        cat("===================================\\n")
        cat("Original pooled effect:", round(ma_full$beta, 4), 
            " [", round(ma_full$ci.lb, 4), ",", round(ma_full$ci.ub, 4), "]\\n")
        cat("\\n")
        
        # Leave-one-out analysis
        loo_results <- leave1out(ma_full)
        
        cat("Leave-One-Out Results\\n")
        cat("Study Excluded\\tEffect Size\\t95% CI Lower\\t95% CI Upper\\tI²\\n")
        
        for (i in 1:length(study_labels)) {{
            cat(study_labels[i], "\\t",
                round(loo_results$estimate[i], 4), "\\t",
                round(loo_results$ci.lb[i], 4), "\\t",
                round(loo_results$ci.ub[i], 4), "\\t",
                round(loo_results$I2[i], 2), "%\\n")
        }}
        
        # Identify influential studies
        cat("\\nInfluential Studies Analysis\\n")
        max_change <- max(abs(loo_results$estimate - ma_full$beta))
        influential_idx <- which.max(abs(loo_results$estimate - ma_full$beta))
        
        cat("Maximum effect size change:", round(max_change, 4), "\\n")
        cat("Most influential study:", study_labels[influential_idx], "\\n")
        cat("Effect without this study:", round(loo_results$estimate[influential_idx], 4), "\\n")
        """
        
        return self.execute_r_code(r_code)
    
    def subgroup_analysis(self, effect_sizes: List[float], 
                         standard_errors: List[float],
                         subgroup_variable: List[str]) -> str:
        """Perform subgroup analysis"""
        
        r_code = f"""
        library(metafor)
        library(meta)
        
        # Prepare data
        effect_sizes <- c({', '.join(map(str, effect_sizes))})
        standard_errors <- c({', '.join(map(str, standard_errors))})
        subgroups <- c({', '.join([f'"{group}"' for group in subgroup_variable])})
        
        cat("Subgroup Analysis\\n")
        cat("================\\n")
        
        # Overall analysis
        ma_overall <- rma(yi = effect_sizes, sei = standard_errors)
        cat("Overall effect size:", round(ma_overall$beta, 4), 
            " [", round(ma_overall$ci.lb, 4), ",", round(ma_overall$ci.ub, 4), "]\\n")
        cat("\\n")
        
        # Subgroup analysis
        unique_groups <- unique(subgroups)
        
        for (group in unique_groups) {{
            group_idx <- which(subgroups == group)
            
            if (length(group_idx) > 1) {{
                ma_subgroup <- rma(yi = effect_sizes[group_idx], 
                                 sei = standard_errors[group_idx])
                
                cat("Subgroup:", group, "\\n")
                cat("  Number of studies:", length(group_idx), "\\n")
                cat("  Effect size:", round(ma_subgroup$beta, 4), 
                    " [", round(ma_subgroup$ci.lb, 4), ",", round(ma_subgroup$ci.ub, 4), "]\\n")
                cat("  I²:", round(ma_subgroup$I2, 2), "%\\n")
                cat("  P-value:", round(ma_subgroup$pval, 4), "\\n")
                cat("\\n")
            }}
        }}
        
        # Test for subgroup differences
        ma_subgroup_test <- rma(yi = effect_sizes, sei = standard_errors, 
                              mods = ~ factor(subgroups) - 1)
        
        cat("Test for Subgroup Differences\\n")
        cat("Q_between:", round(ma_subgroup_test$QM, 4), "\\n")
        cat("df:", ma_subgroup_test$QMdf[1], "\\n")
        cat("P-value:", round(ma_subgroup_test$QMp, 4), "\\n")
        
        if (ma_subgroup_test$QMp < 0.05) {{
            cat("Interpretation: Significant differences between subgroups (p < 0.05)\\n")
        }} else {{
            cat("Interpretation: No significant differences between subgroups (p >= 0.05)\\n")
        }}
        """
        
        return self.execute_r_code(r_code)

    def generate_forest_plot_data(self, effect_sizes: List[float], 
                                 standard_errors: List[float],
                                 study_labels: List[str]) -> str:
        """Generate data for forest plot visualization"""
        
        r_code = f"""
        library(metafor)
        library(forestplot)
        
        # Prepare data
        effect_sizes <- c({', '.join(map(str, effect_sizes))})
        standard_errors <- c({', '.join(map(str, standard_errors))})
        study_labels <- c({', '.join([f'"{label}"' for label in study_labels])})
        
        # Meta-analysis
        ma <- rma(yi = effect_sizes, sei = standard_errors, slab = study_labels)
        
        # Calculate confidence intervals
        ci_lower <- effect_sizes - 1.96 * standard_errors
        ci_upper <- effect_sizes + 1.96 * standard_errors
        
        # Forest plot data output
        cat("FOREST_PLOT_DATA\\n")
        cat("study,effect_size,ci_lower,ci_upper,weight\\n")
        
        weights <- weights(ma)
        for (i in 1:length(study_labels)) {{
            cat(study_labels[i], ",", effect_sizes[i], ",", 
                ci_lower[i], ",", ci_upper[i], ",", weights[i], "\\n")
        }}
        
        # Overall effect
        cat("Overall,", ma$beta, ",", ma$ci.lb, ",", ma$ci.ub, ",100\\n")
        
        cat("\\nFOREST_PLOT_SUMMARY\\n")
        cat("Overall Effect Size:", round(ma$beta, 4), "\\n")
        cat("95% CI: [", round(ma$ci.lb, 4), ",", round(ma$ci.ub, 4), "]\\n")
        cat("I²:", round(ma$I2, 2), "%\\n")
        cat("Tau²:", round(ma$tau2, 4), "\\n")
        """
        
        return self.execute_r_code(r_code)
