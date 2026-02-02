"""
Model Diagnostics Module
========================

Residual analysis and assumption checking for LMM results.
Includes tests for normality, heteroscedasticity, and influential observations.

Architecture Note:
    This module uses dictionaries instead of classes for data structures
    to maintain consistency with the oh_parser project style.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .lmm import LMMResult, get_residuals, get_fitted_values


# =============================================================================
# DiagnosticsResult (dict)
# =============================================================================

DiagnosticsResult = Dict[str, Any]


def create_diagnostics_result(
    outcome: str,
    residuals_normality: Optional[Dict[str, Any]] = None,
    homoscedasticity: Optional[Dict[str, Any]] = None,
    outliers: Optional[Dict[str, Any]] = None,
    influential: Optional[Dict[str, Any]] = None,
    overall_assessment: str = "",
    diag_warnings: Optional[List[str]] = None,
) -> DiagnosticsResult:
    """
    Create a DiagnosticsResult dictionary.
    
    :param outcome: Outcome variable name
    :param residuals_normality: Shapiro-Wilk test results
    :param homoscedasticity: Levene/Breusch-Pagan test results
    :param outliers: Information about potential outliers
    :param influential: Information about influential observations
    :param overall_assessment: Summary assessment
    :param diag_warnings: List of diagnostic warnings
    :returns: DiagnosticsResult dictionary
    """
    return {
        "outcome": outcome,
        "residuals_normality": residuals_normality or {},
        "homoscedasticity": homoscedasticity or {},
        "outliers": outliers or {},
        "influential": influential or {},
        "overall_assessment": overall_assessment,
        "warnings": diag_warnings or [],
    }


def summarize_diagnostics(result: DiagnosticsResult) -> str:
    """
    Generate summary string for diagnostics result.
    
    :param result: DiagnosticsResult dictionary
    :returns: Human-readable summary string
    """
    lines = [
        f"Diagnostics: {result['outcome']}",
        f"  Residuals normality: {'OK' if result['residuals_normality'].get('is_normal', False) else 'CHECK'}",
        f"  Homoscedasticity: {'OK' if result['homoscedasticity'].get('is_ok', False) else 'CHECK'}",
        f"  Outliers: {result['outliers'].get('n_outliers', 0)} detected",
        f"  Assessment: {result['overall_assessment']}",
    ]
    if result["warnings"]:
        lines.append(f"  Warnings: {len(result['warnings'])}")
    return "\n".join(lines)


# =============================================================================
# Residual Diagnostics
# =============================================================================

def residual_diagnostics(
    result: LMMResult,
    alpha: float = 0.05,
) -> DiagnosticsResult:
    """
    Perform comprehensive residual diagnostics.
    
    :param result: Fitted LMMResult dictionary
    :param alpha: Significance level for tests
    :returns: DiagnosticsResult dictionary with all diagnostic information
    """
    warnings_list = []
    
    if result["model"] is None:
        return create_diagnostics_result(
            outcome=result["outcome"],
            residuals_normality={"note": "No model fitted"},
            homoscedasticity={"note": "No model fitted"},
            outliers={"note": "No model fitted"},
            influential={"note": "No model fitted"},
            overall_assessment="Cannot assess - model not fitted",
            diag_warnings=["Model fitting failed"],
        )
    
    residuals = get_residuals(result)
    fitted = get_fitted_values(result)
    
    if residuals is None or len(residuals) == 0:
        return create_diagnostics_result(
            outcome=result["outcome"],
            residuals_normality={"note": "No residuals available"},
            homoscedasticity={"note": "No residuals available"},
            outliers={"note": "No residuals available"},
            influential={"note": "No residuals available"},
            overall_assessment="Cannot assess - no residuals",
            diag_warnings=["No residuals available"],
        )
    
    # 1. Normality of residuals
    normality = _check_residual_normality(residuals, alpha)
    if not normality.get("is_normal", True):
        warnings_list.append("Residuals may not be normally distributed")
    
    # 2. Homoscedasticity
    homoscedasticity = _check_homoscedasticity(residuals, fitted, alpha)
    if not homoscedasticity.get("is_ok", True):
        warnings_list.append("Evidence of heteroscedasticity")
    
    # 3. Outliers
    outliers = _detect_outliers(residuals)
    if outliers.get("n_outliers", 0) > 0:
        pct = 100 * outliers["n_outliers"] / len(residuals)
        if pct > 5:
            warnings_list.append(f"High proportion of outliers: {pct:.1f}%")
    
    # 4. Influential observations (simplified)
    influential = _check_influential(residuals, fitted)
    
    # Overall assessment
    issues = []
    if not normality.get("is_normal", True):
        issues.append("non-normal residuals")
    if not homoscedasticity.get("is_ok", True):
        issues.append("heteroscedasticity")
    if outliers.get("n_outliers", 0) / len(residuals) > 0.05:
        issues.append("many outliers")
    
    if not issues:
        assessment = "OK - No major violations detected"
    elif len(issues) == 1:
        assessment = f"Minor concern: {issues[0]}"
    else:
        assessment = f"Multiple concerns: {', '.join(issues)}"
    
    return create_diagnostics_result(
        outcome=result["outcome"],
        residuals_normality=normality,
        homoscedasticity=homoscedasticity,
        outliers=outliers,
        influential=influential,
        overall_assessment=assessment,
        diag_warnings=warnings_list,
    )


def _check_residual_normality(
    residuals: pd.Series,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Check normality of residuals."""
    residuals = residuals.dropna()
    n = len(residuals)
    
    if n < 3:
        return {"note": "Too few observations", "is_normal": None}
    
    # Shapiro-Wilk (up to 5000 samples)
    test_values = residuals.values[:5000] if n > 5000 else residuals.values
    
    try:
        shapiro_stat, shapiro_p = stats.shapiro(test_values)
    except Exception:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    
    # Also do Jarque-Bera for larger samples
    try:
        jb_stat, jb_p = stats.jarque_bera(residuals)
    except Exception:
        jb_stat, jb_p = np.nan, np.nan
    
    is_normal = shapiro_p > alpha if not np.isnan(shapiro_p) else None
    
    return {
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_p": jb_p,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "is_normal": is_normal,
    }


def _check_homoscedasticity(
    residuals: pd.Series,
    fitted: pd.Series,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Check for heteroscedasticity (non-constant variance)."""
    if residuals is None or fitted is None:
        return {"note": "Missing data", "is_ok": None}
    
    # Align indices safely
    residuals = residuals.dropna()
    
    # Ensure fitted values index matches residuals
    common_idx = residuals.index.intersection(fitted.index)
    if len(common_idx) < len(residuals):
        warnings.warn(
            f"Index mismatch: {len(residuals) - len(common_idx)} residuals have no matching fitted values. "
            f"Using {len(common_idx)} aligned observations."
        )
    
    residuals = residuals.loc[common_idx]
    fitted = fitted.loc[common_idx]
    
    n = len(residuals)
    if n < 10:
        return {"note": "Too few observations", "is_ok": None}
    
    # Simple test: correlation between |residuals| and fitted values
    abs_resid = np.abs(residuals)
    corr, corr_p = stats.pearsonr(abs_resid, fitted)
    
    # Breusch-Pagan test (simplified)
    # Regress squared residuals on fitted values
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        import statsmodels.api as sm
        
        X = sm.add_constant(fitted.values)
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals.values, X)
    except Exception:
        bp_stat, bp_p = np.nan, np.nan
    
    is_ok = corr_p > alpha if not np.isnan(corr_p) else None
    
    return {
        "abs_resid_fitted_corr": corr,
        "abs_resid_fitted_p": corr_p,
        "breusch_pagan_stat": bp_stat,
        "breusch_pagan_p": bp_p,
        "is_ok": is_ok,
    }


def _detect_outliers(
    residuals: pd.Series,
    threshold: float = 3.0,
) -> Dict[str, Any]:
    """Detect outliers using standardized residuals."""
    residuals = residuals.dropna()
    
    if len(residuals) == 0:
        return {"n_outliers": 0, "outlier_indices": []}
    
    # Standardize
    mean = residuals.mean()
    std = residuals.std()
    
    if std == 0:
        return {"n_outliers": 0, "outlier_indices": [], "note": "Zero variance"}
    
    z_scores = (residuals - mean) / std
    
    # Outliers: |z| > threshold
    outlier_mask = np.abs(z_scores) > threshold
    outlier_indices = residuals[outlier_mask].index.tolist()
    
    return {
        "n_outliers": outlier_mask.sum(),
        "outlier_indices": outlier_indices,
        "threshold": threshold,
        "max_abs_z": np.abs(z_scores).max(),
    }


def _check_influential(
    residuals: pd.Series,
    fitted: pd.Series,
) -> Dict[str, Any]:
    """Check for influential observations (simplified)."""
    # Full Cook's distance requires the full model
    # Here we just flag extreme residuals as potentially influential
    
    residuals = residuals.dropna()
    n = len(residuals)
    
    if n == 0:
        return {"note": "No data"}
    
    std = residuals.std()
    if std == 0:
        return {"note": "Zero variance"}
    
    z_scores = (residuals - residuals.mean()) / std
    
    # Rough rule: observations with |z| > 2 may be influential
    potentially_influential = (np.abs(z_scores) > 2).sum()
    
    return {
        "potentially_influential": potentially_influential,
        "pct_influential": 100 * potentially_influential / n,
        "note": "Full Cook's distance requires refitting model",
    }


# =============================================================================
# Assumption Checking
# =============================================================================

def check_assumptions(
    result: LMMResult,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Check key LMM assumptions and return summary.
    
    :param result: Fitted LMMResult dictionary
    :param verbose: Print detailed output
    :returns: Dictionary with assumption checks
    """
    diag = residual_diagnostics(result)
    
    checks = {
        "normality": diag["residuals_normality"].get("is_normal", None),
        "homoscedasticity": diag["homoscedasticity"].get("is_ok", None),
        "no_outliers": diag["outliers"].get("n_outliers", 0) == 0,
        "converged": result["converged"],
    }
    
    if verbose:
        print(f"\nAssumption Check: {result['outcome']}")
        print("-" * 40)
        
        for assumption, passed in checks.items():
            status = "✓" if passed else ("✗" if passed is False else "?")
            print(f"  {assumption}: {status}")
        
        if diag["warnings"]:
            print("\nWarnings:")
            for w in diag["warnings"]:
                print(f"  - {w}")
        
        print(f"\nOverall: {diag['overall_assessment']}")
    
    return checks


# =============================================================================
# Residual Plots (Data Preparation)
# =============================================================================

def get_diagnostic_data(result: LMMResult) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Prepare data for diagnostic plots.
    
    Returns DataFrame with residuals, fitted values, standardized residuals,
    and QQ-plot coordinates.
    
    :param result: Fitted LMMResult dictionary
    :returns: Tuple of (residual_df, qq_df) or None if not available
    """
    if result["model"] is None:
        return None
    
    residuals = get_residuals(result)
    fitted = get_fitted_values(result)
    
    if residuals is None:
        return None
    
    df = pd.DataFrame({
        "residual": residuals,
        "fitted": fitted,
    })
    
    # Standardized residuals
    df["std_residual"] = (df["residual"] - df["residual"].mean()) / df["residual"].std()
    
    # QQ-plot theoretical quantiles
    n = len(df)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    sorted_resid = df["std_residual"].sort_values()
    
    qq_df = pd.DataFrame({
        "theoretical": theoretical_quantiles,
        "sample": sorted_resid.values,
    })
    
    return df, qq_df
