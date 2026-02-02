"""
Linear Mixed Models Module
==========================

Fits Linear Mixed Effects Models for continuous outcomes using statsmodels.
Designed for repeated measures (subject × day) with random intercepts.

Architecture Note:
    This module uses dictionaries instead of classes for data structures
    to maintain consistency with the oh_parser project style.

Key features:
- LMMResult dict for structured model output
- Automatic handling of categorical day_index
- Optional variance-stabilizing transforms
- Support for fixed effects formulas
- Model comparison (AIC, BIC, log-likelihood)
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMResults
from statsmodels.formula.api import mixedlm

from .prepare import AnalysisDataset
from .registry import get_outcome_info, OutcomeType, TransformType


# =============================================================================
# LMMResult (dict)
# =============================================================================

LMMResult = Dict[str, Any]


def create_lmm_result(
    outcome: str,
    model: Optional[MixedLMResults] = None,
    formula: str = "",
    coefficients: Optional[pd.DataFrame] = None,
    fit_stats: Optional[Dict[str, float]] = None,
    random_effects: Optional[Dict[str, float]] = None,
    n_obs: int = 0,
    n_groups: int = 0,
    converged: bool = False,
    transform_applied: Optional[str] = None,
    model_warnings: Optional[List[str]] = None,
) -> LMMResult:
    """
    Create an LMMResult dictionary with all model output.
    
    :param outcome: Name of the outcome variable
    :param model: Fitted statsmodels MixedLMResults object
    :param formula: Model formula used
    :param coefficients: DataFrame with estimates, SEs, CIs, p-values
    :param fit_stats: Dict with AIC, BIC, log-likelihood, etc.
    :param random_effects: Dict with random effects variance estimates
    :param n_obs: Number of observations
    :param n_groups: Number of subjects/groups
    :param converged: Whether optimization converged
    :param transform_applied: Transform applied to outcome (if any)
    :param model_warnings: List of warnings generated during fitting
    :returns: LMMResult dictionary
    """
    return {
        "outcome": outcome,
        "model": model,
        "formula": formula,
        "coefficients": coefficients if coefficients is not None else pd.DataFrame(),
        "fit_stats": fit_stats if fit_stats is not None else {},
        "random_effects": random_effects if random_effects is not None else {},
        "n_obs": n_obs,
        "n_groups": n_groups,
        "converged": converged,
        "transform_applied": transform_applied,
        "warnings": model_warnings if model_warnings is not None else [],
    }


def summarize_lmm_result(result: LMMResult) -> str:
    """
    Generate a summary string for an LMM result.
    
    :param result: LMMResult dictionary
    :returns: Human-readable summary string
    """
    lines = [
        f"LMM Result: {result['outcome']}",
        f"  Formula: {result['formula']}",
        f"  N observations: {result['n_obs']}",
        f"  N subjects: {result['n_groups']}",
        f"  Converged: {result['converged']}",
        f"  AIC: {result['fit_stats'].get('aic', np.nan):.2f}",
        f"  BIC: {result['fit_stats'].get('bic', np.nan):.2f}",
    ]
    transform = result.get("transform_applied")
    if transform and transform != TransformType.NONE:
        lines.append(f"  Transform: {transform}")
    if result["warnings"]:
        lines.append(f"  Warnings: {len(result['warnings'])}")
    return "\n".join(lines)


# =============================================================================
# Transforms
# =============================================================================

def apply_transform(
    values: pd.Series,
    transform: TransformType,
) -> pd.Series:
    """
    Apply variance-stabilizing transform to values.
    
    :param values: Series of outcome values
    :param transform: Transform type to apply
    :returns: Transformed values
    """
    if transform == TransformType.NONE:
        return values
    
    elif transform == TransformType.LOG:
        # Require positive values
        if (values <= 0).any():
            warnings.warn("LOG transform requires positive values; using LOG1P instead")
            return np.log1p(values)
        return np.log(values)
    
    elif transform == TransformType.LOG1P:
        if (values < 0).any():
            warnings.warn("LOG1P transform requires non-negative values; clipping to 0")
            values = values.clip(lower=0)
        return np.log1p(values)
    
    elif transform == TransformType.SQRT:
        if (values < 0).any():
            warnings.warn("SQRT transform requires non-negative values; clipping to 0")
            values = values.clip(lower=0)
        return np.sqrt(values)
    
    elif transform == TransformType.LOGIT:
        # Require values in (0, 1); rescale if provided as percent
        max_val = values.max(skipna=True)
        if max_val > 1 and max_val <= 100:
            warnings.warn("LOGIT transform expects proportions in [0,1]. Rescaling from percent.")
            values = values / 100.0
        eps = 1e-6
        values = values.clip(lower=eps, upper=1 - eps)
        return np.log(values / (1 - values))
    
    elif transform == TransformType.ARCSINE:
        # Require values in [0, 1]
        values = values.clip(lower=0, upper=1)
        return np.arcsin(np.sqrt(values))
    
    else:
        warnings.warn(f"Unknown transform {transform}; returning untransformed")
        return values


# =============================================================================
# Model Fitting
# =============================================================================

def fit_lmm(
    ds: AnalysisDataset,
    outcome: str,
    fixed_effects: Optional[List[str]] = None,
    random_intercept: str = "subject_id",
    transform: Optional[str] = None,
    day_as_categorical: bool = True,
    include_side: bool = True,
    formula: Optional[str] = None,
    reml: bool = False,
    **kwargs,
) -> LMMResult:
    """
    Fit a Linear Mixed Effects Model for a continuous outcome.
    
    :param ds: AnalysisDataset dictionary with data and metadata
    :param outcome: Name of the outcome variable
    :param fixed_effects: List of fixed effect variable names (default: day_index, side)
    :param random_intercept: Variable for random intercept grouping (default: subject_id)
    :param transform: Variance-stabilizing transform (None = use registry recommendation)
    :param day_as_categorical: Treat day_index as categorical (vs numeric trend)
    :param include_side: Include 'side' in model if present (default: True)
    :param formula: Override formula (statsmodels formula syntax)
    :param reml: Use REML estimation (default: False = ML for valid AIC/BIC)
    :param kwargs: Additional arguments passed to mixedlm
    :returns: LMMResult dictionary with model output
    
    Note:
        REML (Restricted Maximum Likelihood) is generally preferred for unbiased
        variance estimates, but AIC/BIC are only valid with ML estimation.
        Default is ML (reml=False) to enable model comparison via AIC/BIC.
    
    Example:
        >>> result = fit_lmm(ds, "EMG_intensity.mean_percent_mvc")
        >>> print(result["coefficients"])
    """
    model_warnings = []
    
    # Validate outcome
    if outcome not in ds["data"].columns:
        raise ValueError(f"Outcome '{outcome}' not found in dataset")
    
    # Get outcome info from registry
    info = get_outcome_info(outcome)
    
    if info["outcome_type"] not in (OutcomeType.CONTINUOUS, OutcomeType.UNKNOWN):
        model_warnings.append(
            f"Outcome type is {info['outcome_type']}; LMM assumes continuous. "
            f"Consider appropriate model family."
        )
    
    # Prepare data
    df = ds["data"].copy()
    
    # Determine transform
    if transform is None:
        transform = info["transform"]
    
    # Apply transform
    outcome_col = outcome
    if transform != TransformType.NONE:
        transformed_name = f"{outcome}_transformed"
        df[transformed_name] = apply_transform(df[outcome], transform)
        outcome_col = transformed_name
    
    # Drop missing values for this outcome
    df = df.dropna(subset=[outcome_col, random_intercept])
    
    n_obs = len(df)
    n_groups = df[random_intercept].nunique()
    
    if n_obs < 10:
        return create_lmm_result(
            outcome=outcome,
            model=None,
            formula="",
            n_obs=n_obs,
            n_groups=n_groups,
            converged=False,
            transform_applied=transform,
            model_warnings=["Insufficient observations for model fitting"],
        )
    
    # Build formula
    if formula is not None:
        model_formula = formula
    else:
        # Build fixed effects
        if fixed_effects is None:
            fixed_effects = []
            
            if "day_index" in df.columns:
                if day_as_categorical:
                    fixed_effects.append("C(day_index)")
                else:
                    fixed_effects.append("day_index")
            
            if include_side and "side" in df.columns and df["side"].nunique() > 1:
                fixed_effects.append("C(side)")
        
        # Drop rows with missing values in predictors
        # Extract raw column names (strip C() wrapper if present)
        predictor_cols = []
        for fe in fixed_effects:
            if fe.startswith("C(") and fe.endswith(")"):
                predictor_cols.append(fe[2:-1])
            else:
                predictor_cols.append(fe)
        
        # Filter to columns that exist in the dataframe
        predictor_cols = [c for c in predictor_cols if c in df.columns]
        if predictor_cols:
            df = df.dropna(subset=predictor_cols)
            n_obs = len(df)
            n_groups = df[random_intercept].nunique()
        
        # Quote outcome column name if it contains special characters
        # Characters that break statsmodels formulas: . [ ] ( ) + - * / : ^ | ~ space
        special_chars = set('.[]()+-*/:^|~ ')
        needs_quoting = any(c in outcome_col for c in special_chars)
        outcome_formula = f"Q('{outcome_col}')" if needs_quoting else outcome_col
        
        if not fixed_effects:
            # Intercept-only model
            model_formula = f"{outcome_formula} ~ 1"
        else:
            fixed_str = " + ".join(fixed_effects)
            model_formula = f"{outcome_formula} ~ {fixed_str}"
    
    # Fit model
    try:
        model = mixedlm(
            model_formula,
            data=df,
            groups=df[random_intercept],
            **kwargs,
        )
        result = model.fit(reml=reml)
        converged = result.converged
        
    except Exception as e:
        model_warnings.append(f"Model fitting failed: {str(e)}")
        return create_lmm_result(
            outcome=outcome,
            model=None,
            formula=model_formula,
            n_obs=n_obs,
            n_groups=n_groups,
            converged=False,
            transform_applied=transform,
            model_warnings=model_warnings,
        )
    
    # Extract coefficients
    coef_df = _extract_coefficients(result)
    
    # Compute LRT for day_index effect (if applicable and ML)
    lrt_stat = np.nan
    lrt_df = np.nan
    lrt_pvalue = np.nan

    if formula is None and not reml:
        if fixed_effects and any(term in fixed_effects for term in ["C(day_index)", "day_index"]):
            reduced_effects = [
                term for term in fixed_effects
                if term not in ["C(day_index)", "day_index"]
            ]
            reduced_formula = f"{outcome_formula} ~ 1" if not reduced_effects else f"{outcome_formula} ~ {' + '.join(reduced_effects)}"

            try:
                reduced_model = mixedlm(
                    reduced_formula,
                    data=df,
                    groups=df[random_intercept],
                    **kwargs,
                )
                reduced_result = reduced_model.fit(reml=False)
                lrt_stat, lrt_df, lrt_pvalue = _compute_lrt(result, reduced_result)
            except Exception as e:
                model_warnings.append(f"LRT failed: {str(e)}")
    elif reml:
        model_warnings.append("LRT skipped because REML=True (use ML for LRT)")

    # Extract fit statistics
    fit_stats = {
        "aic": result.aic,
        "bic": result.bic,
        "llf": result.llf,
        "scale": result.scale,
        "lrt_stat": lrt_stat,
        "lrt_df": lrt_df,
        "lrt_pvalue": lrt_pvalue,
    }
    
    # Extract random effects variance
    random_effects = {
        "group_var": result.cov_re.iloc[0, 0] if hasattr(result, "cov_re") else np.nan,
        "residual_var": result.scale,
    }
    
    # Calculate ICC (Intraclass Correlation Coefficient)
    total_var = random_effects["group_var"] + random_effects["residual_var"]
    if total_var > 0:
        random_effects["icc"] = random_effects["group_var"] / total_var
    else:
        random_effects["icc"] = np.nan
    
    return create_lmm_result(
        outcome=outcome,
        model=result,
        formula=model_formula,
        coefficients=coef_df,
        fit_stats=fit_stats,
        random_effects=random_effects,
        n_obs=n_obs,
        n_groups=n_groups,
        converged=converged,
        transform_applied=transform,
        model_warnings=model_warnings,
    )


def _extract_coefficients(result: MixedLMResults) -> pd.DataFrame:
    """Extract coefficient table from fitted model."""
    
    # Get summary as DataFrame
    summary_df = pd.DataFrame({
        "estimate": result.fe_params,
        "std_error": result.bse_fe,
        "z_value": result.tvalues,
        "p_value": result.pvalues,
    })
    
    # Add confidence intervals
    conf_int = result.conf_int()
    if isinstance(conf_int, pd.DataFrame):
        summary_df["ci_lower"] = conf_int.iloc[:, 0]
        summary_df["ci_upper"] = conf_int.iloc[:, 1]
    
    # Reset index to make coefficient names a column
    summary_df = summary_df.reset_index()
    summary_df = summary_df.rename(columns={"index": "term"})
    
    return summary_df


def _compute_lrt(
    full_result: MixedLMResults,
    reduced_result: MixedLMResults,
) -> Tuple[float, float, float]:
    """Compute likelihood ratio test between full and reduced models."""
    try:
        lrt_stat = 2 * (full_result.llf - reduced_result.llf)
        df_full = getattr(full_result, "df_modelwc", None)
        df_red = getattr(reduced_result, "df_modelwc", None)
        if df_full is None or df_red is None:
            df_diff = len(full_result.fe_params) - len(reduced_result.fe_params)
        else:
            df_diff = df_full - df_red
        if df_diff <= 0:
            return (np.nan, np.nan, np.nan)
        pvalue = stats.chi2.sf(lrt_stat, df_diff)
        return (float(lrt_stat), float(df_diff), float(pvalue))
    except Exception:
        return (np.nan, np.nan, np.nan)


# =============================================================================
# Batch Fitting
# =============================================================================

def fit_all_outcomes(
    ds: AnalysisDataset,
    outcomes: Optional[List[str]] = None,
    skip_degenerate: bool = True,
    **kwargs,
) -> Dict[str, LMMResult]:
    """
    Fit LMM for all (or selected) outcomes.
    
    :param ds: AnalysisDataset dictionary
    :param outcomes: Specific outcomes to fit (None = all)
    :param skip_degenerate: Skip outcomes with near-zero variance
    :param kwargs: Additional arguments passed to fit_lmm
    :returns: Dictionary mapping outcome names to LMMResult dictionaries
    """
    from .descriptive import get_non_degenerate_outcomes
    
    if outcomes is None:
        outcomes = ds["outcome_vars"]
    
    if skip_degenerate:
        outcomes = get_non_degenerate_outcomes(ds, outcomes)
    
    results = {}
    
    for outcome in outcomes:
        try:
            results[outcome] = fit_lmm(ds, outcome, **kwargs)
        except Exception as e:
            warnings.warn(f"Failed to fit model for {outcome}: {str(e)}")
            results[outcome] = create_lmm_result(
                outcome=outcome,
                model=None,
                formula="",
                n_obs=0,
                n_groups=0,
                converged=False,
                model_warnings=[f"Exception: {str(e)}"],
            )
    
    return results


# =============================================================================
# Model Diagnostics (Basic)
# =============================================================================

def get_residuals(result: LMMResult) -> Optional[pd.Series]:
    """Extract residuals from fitted model."""
    if result["model"] is None:
        return None
    return result["model"].resid


def get_fitted_values(result: LMMResult) -> Optional[pd.Series]:
    """Extract fitted values from model."""
    if result["model"] is None:
        return None
    return result["model"].fittedvalues


def get_random_effects(result: LMMResult) -> Optional[pd.DataFrame]:
    """Extract random effects (BLUPs) from model."""
    if result["model"] is None:
        return None
    
    re = result["model"].random_effects
    if re is None:
        return None
    
    # Convert to DataFrame
    rows = []
    for group, effects in re.items():
        row = {"group": group}
        if hasattr(effects, "items"):
            row.update(dict(effects))
        else:
            row["intercept"] = effects
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(results: List[LMMResult]) -> pd.DataFrame:
    """
    Compare multiple LMM results by fit statistics.
    
    :param results: List of LMMResult dictionaries (same outcome, different specs)
    :returns: DataFrame comparing AIC, BIC, log-likelihood
    """
    rows = []
    for r in results:
        rows.append({
            "outcome": r["outcome"],
            "formula": r["formula"],
            "n_obs": r["n_obs"],
            "n_groups": r["n_groups"],
            "converged": r["converged"],
            "aic": r["fit_stats"].get("aic", np.nan),
            "bic": r["fit_stats"].get("bic", np.nan),
            "llf": r["fit_stats"].get("llf", np.nan),
            "icc": r["random_effects"].get("icc", np.nan),
        })
    
    df = pd.DataFrame(rows)
    
    # Add delta AIC/BIC relative to best
    if len(df) > 1 and not df["aic"].isna().all():
        min_aic = df["aic"].min()
        min_bic = df["bic"].min()
        df["delta_aic"] = df["aic"] - min_aic
        df["delta_bic"] = df["bic"] - min_bic
    
    return df
