"""
Post-hoc Comparisons Module
===========================

Computes estimated marginal means and pairwise contrasts from fitted LMMs.
Implements Wald-based hypothesis tests with multiplicity corrections.

Architecture Note:
    This module uses dictionaries instead of classes for data structures
    to maintain consistency with the oh_parser project style.

Note: statsmodels MixedLM doesn't provide "emmeans" natively, so we
compute them by predicting at each level of the factor while holding
other covariates at reference/mean values.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Literal, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .lmm import LMMResult
from .prepare import AnalysisDataset


# =============================================================================
# ContrastResult (dict)
# =============================================================================

ContrastResult = Dict[str, Any]


def create_contrast_result(
    outcome: str,
    factor: str,
    contrasts: Optional[pd.DataFrame] = None,
    emmeans: Optional[pd.DataFrame] = None,
    correction: str = "none",
    effect_sizes: Optional[pd.DataFrame] = None,
) -> ContrastResult:
    """
    Create a ContrastResult dictionary with all contrast analysis output.
    
    :param outcome: Name of the outcome variable
    :param factor: Factor being compared (e.g., "day_index")
    :param contrasts: DataFrame with contrast estimates, SEs, CIs, p-values
    :param emmeans: DataFrame with estimated marginal means per level
    :param correction: Multiplicity correction applied
    :param effect_sizes: DataFrame with effect size estimates (optional)
    :returns: ContrastResult dictionary
    """
    return {
        "outcome": outcome,
        "factor": factor,
        "contrasts": contrasts if contrasts is not None else pd.DataFrame(),
        "emmeans": emmeans if emmeans is not None else pd.DataFrame(),
        "correction": correction,
        "effect_sizes": effect_sizes,
    }


def summarize_contrast_result(result: ContrastResult) -> str:
    """
    Generate summary string for contrast result.
    
    :param result: ContrastResult dictionary
    :returns: Human-readable summary string
    """
    contrasts = result["contrasts"]
    n_sig = 0
    if not contrasts.empty and "p_adjusted" in contrasts.columns:
        n_sig = (contrasts["p_adjusted"] < 0.05).sum()
    n_contrasts = len(contrasts)
    
    lines = [
        f"Contrast Result: {result['outcome']}",
        f"  Factor: {result['factor']}",
        f"  Levels: {len(result['emmeans'])}",
        f"  Contrasts: {n_contrasts}",
        f"  Significant (p<0.05): {n_sig}",
        f"  Correction: {result['correction']}",
    ]
    return "\n".join(lines)


# =============================================================================
# Estimated Marginal Means
# =============================================================================

def compute_emmeans(
    result: LMMResult,
    factor: str,
    ds: AnalysisDataset,
) -> pd.DataFrame:
    """
    Compute estimated marginal means for each level of a factor.
    
    :param result: Fitted LMMResult dictionary
    :param factor: Factor variable name (e.g., "day_index", "side")
    :param ds: AnalysisDataset dictionary (needed for factor levels)
    :returns: DataFrame with level, emmean, se, ci_lower, ci_upper
    """
    if result["model"] is None:
        return pd.DataFrame()
    
    model = result["model"]
    df = ds["data"].copy()
    
    # Get factor levels
    factor_col = factor.replace("C(", "").replace(")", "")
    if factor_col not in df.columns:
        warnings.warn(f"Factor '{factor_col}' not found in data")
        return pd.DataFrame()
    
    levels = sorted(df[factor_col].dropna().unique())
    
    # Get model coefficients
    coefs = model.fe_params
    cov = model.cov_params()
    
    # Build emmeans by computing predictions at each level
    # This is a simplified approach - compute mean prediction per level
    rows = []
    
    for level in levels:
        # For categorical factors encoded as C(factor), find the corresponding coefficient
        # The reference level has no coefficient (absorbed in intercept)
        
        # Start with intercept
        if "Intercept" in coefs.index:
            emmean = coefs["Intercept"]
            var = cov.loc["Intercept", "Intercept"]
        else:
            emmean = 0
            var = 0
        
        # Add factor effect if not reference level
        # Coefficient names look like "C(day_index)[T.2]" for level 2
        coef_pattern = f"C({factor_col})[T.{level}]"
        
        if coef_pattern in coefs.index:
            emmean += coefs[coef_pattern]
            # Variance: Var(a + b) = Var(a) + Var(b) + 2*Cov(a,b)
            var += cov.loc[coef_pattern, coef_pattern]
            if "Intercept" in cov.index:
                var += 2 * cov.loc["Intercept", coef_pattern]
        
        se = np.sqrt(var) if var > 0 else np.nan
        
        rows.append({
            "level": level,
            "emmean": emmean,
            "se": se,
            "ci_lower": emmean - 1.96 * se,
            "ci_upper": emmean + 1.96 * se,
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# Pairwise Contrasts
# =============================================================================

def pairwise_contrasts(
    result: LMMResult,
    factor: str,
    ds: AnalysisDataset,
    correction: Literal["none", "holm", "bonferroni", "fdr_bh"] = "holm",
    compute_effects: bool = True,
) -> ContrastResult:
    """
    Compute pairwise contrasts between factor levels.
    
    :param result: Fitted LMMResult dictionary
    :param factor: Factor variable name (e.g., "day_index")
    :param ds: AnalysisDataset dictionary
    :param correction: Multiplicity correction method
    :param compute_effects: Whether to compute effect sizes
    :returns: ContrastResult dictionary with contrasts, emmeans, and effect sizes
    
    Example:
        >>> result = fit_lmm(ds, "EMG_intensity.mean_percent_mvc")
        >>> contrasts = pairwise_contrasts(result, "day_index", ds)
        >>> print(contrasts["contrasts"])
    """
    if result["model"] is None:
        return create_contrast_result(
            outcome=result["outcome"],
            factor=factor,
            correction=correction,
        )
    
    # Compute emmeans
    emmeans = compute_emmeans(result, factor, ds)
    
    if emmeans.empty or len(emmeans) < 2:
        return create_contrast_result(
            outcome=result["outcome"],
            factor=factor,
            emmeans=emmeans,
            correction=correction,
        )
    
    # Compute pairwise contrasts
    model = result["model"]
    coefs = model.fe_params
    cov = model.cov_params()
    
    factor_col = factor.replace("C(", "").replace(")", "")
    levels = emmeans["level"].tolist()
    
    rows = []
    se_is_approximate = False  # Track if we had to use approximation
    
    for level1, level2 in combinations(levels, 2):
        # Get emmeans for both levels
        em1 = emmeans[emmeans["level"] == level1]["emmean"].values[0]
        em2 = emmeans[emmeans["level"] == level2]["emmean"].values[0]
        
        # Difference
        diff = em1 - em2
        
        # SE of difference using delta method
        # For simple contrasts: Var(a - b) = Var(a) + Var(b) - 2*Cov(a,b)
        coef1 = f"C({factor_col})[T.{level1}]"
        coef2 = f"C({factor_col})[T.{level2}]"
        
        se1 = emmeans[emmeans["level"] == level1]["se"].values[0]
        se2 = emmeans[emmeans["level"] == level2]["se"].values[0]
        
        # Start with sum of variances (assumes independence)
        var_diff = se1**2 + se2**2
        
        # Try to incorporate covariance from model (makes SE smaller if positively correlated)
        # Note: This only works for non-reference levels; reference level contrasts use approximation
        covariance_available = coef1 in cov.index and coef2 in cov.index
        if covariance_available:
            var_diff -= 2 * cov.loc[coef1, coef2]
        else:
            # For contrasts involving reference level, we use independence assumption
            # This may be anti-conservative if levels are positively correlated
            se_is_approximate = True
        
        se_diff = np.sqrt(max(var_diff, 0))
        
        # z-test
        z_value = diff / se_diff if se_diff > 0 else np.nan
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value))) if not np.isnan(z_value) else np.nan
        
        rows.append({
            "contrast": f"{level1} - {level2}",
            "level1": level1,
            "level2": level2,
            "estimate": diff,
            "se": se_diff,
            "z_value": z_value,
            "p_value": p_value,
            "ci_lower": diff - 1.96 * se_diff,
            "ci_upper": diff + 1.96 * se_diff,
        })
    
    contrasts_df = pd.DataFrame(rows)
    
    # Warn if approximation was used
    if se_is_approximate:
        warnings.warn(
            f"SE for some contrasts involving the reference level were computed assuming "
            f"independence, which may be anti-conservative (underestimate SE) if levels "
            f"are positively correlated. Interpret p-values with caution."
        )
    
    # Apply multiplicity correction
    if not contrasts_df.empty and correction != "none":
        from .multiplicity import adjust_pvalues
        contrasts_df["p_adjusted"] = adjust_pvalues(
            contrasts_df["p_value"].values,
            method=correction,
        )
    else:
        contrasts_df["p_adjusted"] = contrasts_df["p_value"]
    
    # Compute effect sizes
    effect_sizes = None
    if compute_effects and not contrasts_df.empty:
        effect_sizes = _compute_contrast_effect_sizes(contrasts_df, result, ds)
    
    return create_contrast_result(
        outcome=result["outcome"],
        factor=factor,
        contrasts=contrasts_df,
        emmeans=emmeans,
        correction=correction,
        effect_sizes=effect_sizes,
    )


# =============================================================================
# Effect Sizes
# =============================================================================

def _compute_contrast_effect_sizes(
    contrasts_df: pd.DataFrame,
    result: LMMResult,
    ds: AnalysisDataset,
) -> pd.DataFrame:
    """
    Compute effect sizes for contrasts.
    
    Returns Cohen's d (standardized mean difference) using pooled SD.
    """
    if result["model"] is None:
        return pd.DataFrame()
    
    # Get residual SD as the standardizer
    residual_sd = np.sqrt(result["random_effects"].get("residual_var", 1))
    
    rows = []
    for _, row in contrasts_df.iterrows():
        # Cohen's d = difference / pooled_sd
        d = row["estimate"] / residual_sd if residual_sd > 0 else np.nan
        
        # SE of d (approximate)
        se_d = row["se"] / residual_sd if residual_sd > 0 else np.nan
        
        rows.append({
            "contrast": row["contrast"],
            "cohens_d": d,
            "se_d": se_d,
            "ci_lower_d": d - 1.96 * se_d if not np.isnan(se_d) else np.nan,
            "ci_upper_d": d + 1.96 * se_d if not np.isnan(se_d) else np.nan,
            "interpretation": _interpret_cohens_d(d),
        })
    
    return pd.DataFrame(rows)


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    if np.isnan(d):
        return "NA"
    
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compute_effect_size(
    result: LMMResult,
    ds: AnalysisDataset,
    method: Literal["cohens_d", "eta_squared", "r_squared"] = "cohens_d",
) -> Dict[str, float]:
    """
    Compute overall effect size for a fitted model.
    
    :param result: Fitted LMMResult dictionary
    :param ds: AnalysisDataset dictionary
    :param method: Effect size method
    :returns: Dictionary with effect size metrics
    """
    if result["model"] is None:
        return {}
    
    model = result["model"]
    
    if method == "cohens_d":
        # Not directly applicable to overall model
        # Return ICC as a measure of effect
        return {
            "icc": result["random_effects"].get("icc", np.nan),
            "note": "ICC represents proportion of variance due to between-subject differences",
        }
    
    elif method == "eta_squared":
        # Approximate eta-squared from explained variance
        # This is a rough approximation for mixed models
        if hasattr(model, "fittedvalues") and hasattr(model, "model"):
            y = model.model.endog
            y_hat = model.fittedvalues
            ss_total = np.sum((y - y.mean()) ** 2)
            ss_resid = np.sum((y - y_hat) ** 2)
            ss_model = ss_total - ss_resid
            eta_sq = ss_model / ss_total if ss_total > 0 else np.nan
            
            return {
                "eta_squared": eta_sq,
                "omega_squared": (ss_model - (len(model.fe_params) - 1) * model.scale) / (ss_total + model.scale),
            }
    
    elif method == "r_squared":
        # Pseudo R-squared for mixed models
        # Using Nakagawa's marginal and conditional R²
        if hasattr(model, "fittedvalues"):
            # Marginal R² = variance explained by fixed effects
            # Conditional R² = variance explained by fixed + random effects
            var_fixed = np.var(model.fittedvalues)
            var_random = result["random_effects"].get("group_var", 0)
            var_resid = result["random_effects"].get("residual_var", 1)
            var_total = var_fixed + var_random + var_resid
            
            r2_marginal = var_fixed / var_total if var_total > 0 else np.nan
            r2_conditional = (var_fixed + var_random) / var_total if var_total > 0 else np.nan
            
            return {
                "r2_marginal": r2_marginal,
                "r2_conditional": r2_conditional,
            }
    
    return {}


# =============================================================================
# Trend Analysis
# =============================================================================

def test_linear_trend(
    result: LMMResult,
    factor: str = "day_index",
) -> Dict[str, float]:
    """
    Test for linear trend across ordered factor levels.
    
    Uses polynomial contrast to test if there's a significant
    linear trend across days.
    
    :param result: LMMResult dictionary (fitted with day_index as numeric)
    :param factor: Factor variable name
    :returns: Dictionary with trend test results
    """
    if result["model"] is None:
        return {}
    
    coefs = result["coefficients"]
    
    # Look for numeric day_index coefficient
    factor_rows = coefs[coefs["term"].str.contains(factor, na=False)]
    
    if factor_rows.empty:
        return {"note": "Factor not found in model"}
    
    # If day_index was numeric, there's a single coefficient
    if len(factor_rows) == 1 and not factor_rows["term"].str.contains("C\\(").any():
        row = factor_rows.iloc[0]
        return {
            "linear_estimate": row["estimate"],
            "linear_se": row["std_error"],
            "linear_z": row["z_value"],
            "linear_p": row["p_value"],
            "interpretation": "positive trend" if row["estimate"] > 0 else "negative trend",
        }
    
    return {"note": "Factor is categorical; refit with numeric day_index for trend test"}
