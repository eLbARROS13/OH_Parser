"""
Hypothesis Runner
=================

Generic execution engine for running hypothesis tests.

This module handles the common workflow:
1. Prepare data (subset, transform)
2. Fit model (LMM or OLS)
3. Extract p-value for primary predictor
4. Return structured result

The runner eliminates ~60% of repeated code across hypotheses.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import HYPOTHESES, HypothesisConfig, get_hypothesis

# Import oh_stats functions
from oh_stats import (
    prepare_daily_metrics,
    prepare_single_instance_metrics,
    add_subject_metadata,
    fit_lmm,
    summarize_lmm_result,
    prepare_from_dataframe,
    apply_holm_hypotheses,
    check_variance,
    residual_diagnostics,
)


# -----------------------------------------------------------------------------
# HypothesisResult (dictionary-based, no classes)
# -----------------------------------------------------------------------------

HypothesisResult = Dict[str, Any]


def create_hypothesis_result(
    hypothesis_id: str,
    config: HypothesisConfig,
    p_value: float,
    converged: bool = True,
    model_result: Optional[Dict] = None,
    note: str = "",
    skipped: bool = False,
    skip_reason: str = "",
) -> HypothesisResult:
    """Create a HypothesisResult dictionary."""
    return {
        "hypothesis_id": hypothesis_id,
        "config": config,
        "p_value": p_value,
        "converged": converged,
        "model_result": model_result,
        "note": note,
        "skipped": skipped,
        "skip_reason": skip_reason,
    }


def hypothesis_result_to_dict(result: HypothesisResult) -> Dict[str, Any]:
    """Convert HypothesisResult to dictionary for Holm correction."""
    return {
        "p_value": result["p_value"],
        "note": result["config"]["name"],
        "converged": result.get("converged", True),
        "skipped": result.get("skipped", False),
    }


def format_hypothesis_result(result: HypothesisResult) -> str:
    """Format HypothesisResult for display."""
    h_id = result["hypothesis_id"]
    if result.get("skipped", False):
        return f"HypothesisResult({h_id}: SKIPPED - {result.get('skip_reason', '')})"
    p_val = result["p_value"]
    if np.isnan(p_val):
        return f"HypothesisResult({h_id}: p=NaN)"
    status = "✓" if p_val < 0.05 else "✗"
    return f"HypothesisResult({h_id}: p={p_val:.4f} {status})"


# -----------------------------------------------------------------------------
# Within-Between Decomposition (for H5 exploratory analysis)
# -----------------------------------------------------------------------------

def decompose_within_between(
    df: pd.DataFrame,
    variables: List[str],
    subject_col: str = "subject_id",
    suffix_within: str = "_within",
    suffix_between: str = "_between",
) -> pd.DataFrame:
    """
    Decompose predictors into within-subject and between-subject components.
    
    This allows separating two different effects:
    - Within-subject: "On days when this person's X is higher than their own average..."
    - Between-subject: "People who generally have higher X..."
    
    These can have opposite signs! Classic example:
    - Exercise increases HR (within-person)
    - But fit people have lower resting HR (between-person)
    
    Formula:
        X_within  = X_ij - mean_i(X)           # Day deviation from person's mean
        X_between = mean_i(X) - grand_mean(X)  # Person's mean deviation from grand mean
    
    Args:
        df: DataFrame with repeated measures data
        variables: List of variable names to decompose
        subject_col: Column for subject grouping
        suffix_within: Suffix for within-subject component
        suffix_between: Suffix for between-subject component
    
    Returns:
        DataFrame with original columns plus decomposed versions
    
    Example:
        >>> df = decompose_within_between(df, ["hr_ratio_mean", "noise_mean"])
        >>> # Now has: hr_ratio_mean_within, hr_ratio_mean_between, etc.
    """
    df = df.copy()
    
    for var in variables:
        if var not in df.columns:
            continue
        
        # Grand mean (across all observations)
        grand_mean = df[var].mean()
        
        # Subject means
        subject_means = df.groupby(subject_col)[var].transform("mean")
        
        # Within-subject component: deviation from subject's own mean
        df[f"{var}{suffix_within}"] = df[var] - subject_means
        
        # Between-subject component: subject mean - grand mean
        df[f"{var}{suffix_between}"] = subject_means - grand_mean
    
    return df


def get_within_between_variables(variables: List[str]) -> dict:
    """
    Get the within and between variable names for a list of variables.
    
    Args:
        variables: Original variable names
    
    Returns:
        Dict with 'within' and 'between' lists
    """
    return {
        "within": [f"{v}_within" for v in variables],
        "between": [f"{v}_between" for v in variables],
    }


# -----------------------------------------------------------------------------
# Bootstrap (Clustered by Subject)
# -----------------------------------------------------------------------------

def _bootstrap_primary_pvalue(
    ds: Any,
    outcome: str,
    fixed_effects: List[str],
    random_intercept: str,
    primary: str,
    decompose: bool = False,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> float:
    """
    Cluster bootstrap p-value for primary predictor.
    
    Resamples subjects with replacement and refits the model.
    Returns a two-sided p-value based on bootstrap distribution.
    """
    df = ds["data"].copy()
    if random_intercept not in df.columns:
        return np.nan

    subjects = df[random_intercept].dropna().unique()
    if len(subjects) < 5:
        return np.nan

    rng = np.random.default_rng(seed)
    estimates = []

    primary_term = f"{primary}_within" if decompose else primary

    for i in range(n_bootstrap):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        boot_frames = []
        for j, subj in enumerate(sampled_subjects):
            subj_df = df[df[random_intercept] == subj].copy()
            subj_df[random_intercept] = f"boot_{i}_{j}"
            boot_frames.append(subj_df)
        boot_df = pd.concat(boot_frames, ignore_index=True)
        boot_ds = dict(ds)
        boot_ds["data"] = boot_df

        try:
            boot_result = fit_lmm(
                boot_ds,
                outcome=outcome,
                fixed_effects=fixed_effects,
                random_intercept=random_intercept,
                reml=False,
            )
            if not boot_result.get("converged", False):
                continue
            coef_df = boot_result.get("coefficients", pd.DataFrame())
            if coef_df.empty:
                continue
            mask = coef_df["term"].str.contains(primary_term, case=False, na=False)
            if not mask.any():
                continue
            estimates.append(float(coef_df.loc[mask, "estimate"].iloc[0]))
        except Exception:
            continue

    if len(estimates) < max(30, int(0.3 * n_bootstrap)):
        return np.nan

    estimates = np.array(estimates)
    # Two-sided p-value: proportion crossing 0
    p_two_sided = 2 * min(np.mean(estimates <= 0), np.mean(estimates >= 0))
    return float(min(p_two_sided, 1.0))


# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------

def _apply_logit_transform(df: pd.DataFrame, col: str, eps: float = 1e-6) -> pd.DataFrame:
    """Apply logit transformation to a proportion column."""
    df = df.copy()
    clipped = df[col].clip(eps, 1 - eps)
    df[f"{col}_logit"] = np.log(clipped / (1 - clipped))
    return df


def _apply_log_transform(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Apply log transformation to a positive-valued column.
    
    This is used to stabilize variance when heteroscedasticity is present
    (variance proportional to mean).
    """
    df = df.copy()
    # Require positive values for log transform
    if (df[col] <= 0).any():
        # Use log1p for values that might be 0
        df[f"{col}_log"] = np.log1p(df[col])
    else:
        df[f"{col}_log"] = np.log(df[col])
    return df


def _apply_sqrt_transform(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Apply square root transformation (alternative variance stabilization)."""
    df = df.copy()
    df[f"{col}_sqrt"] = np.sqrt(df[col].clip(lower=0))
    return df


def _prepare_data(
    profiles: Dict[str, dict],
    config: HypothesisConfig,
) -> tuple[Optional[Any], HypothesisConfig, Optional[str]]:
    """
    Prepare data for hypothesis testing.
    
    Returns:
        (dataset, updated_config, error_message) - dataset is None if preparation failed
        The updated_config may contain _transformed_outcome if a transform was applied.
    """
    level = config.get("level", "daily")
    
    if level == "daily":
        # Prepare daily metrics
        ds = prepare_daily_metrics(profiles)
        if ds["data"].empty:
            return None, config, "Daily metrics data is empty"
        
        # Add work_type metadata
        ds = add_subject_metadata(ds, profiles, fields=["work_type"])
        
        # Check required columns
        outcome = config["outcome"]
        if outcome not in ds["data"].columns:
            return None, config, f"Outcome '{outcome}' not found in data"
        
        for pred in config.get("predictors", []):
            if pred not in ds["data"].columns:
                return None, config, f"Predictor '{pred}' not found in data"
        
        # Apply transform if needed
        transform = config.get("transform")
        if transform == "logit":
            ds["data"] = _apply_logit_transform(ds["data"], outcome)
            # Update outcome to transformed version
            config = dict(config)  # Copy to avoid modifying original
            config["_transformed_outcome"] = f"{outcome}_logit"
        elif transform == "log":
            ds["data"] = _apply_log_transform(ds["data"], outcome)
            config = dict(config)
            config["_transformed_outcome"] = f"{outcome}_log"
        elif transform == "sqrt":
            ds["data"] = _apply_sqrt_transform(ds["data"], outcome)
            config = dict(config)
            config["_transformed_outcome"] = f"{outcome}_sqrt"
        
        # Apply within-between decomposition if requested
        if config.get("decompose_predictors", False):
            predictors = config.get("predictors", [])
            # Filter to numeric predictors that exist
            numeric_predictors = [
                p for p in predictors 
                if p in ds["data"].columns and pd.api.types.is_numeric_dtype(ds["data"][p])
            ]
            
            if numeric_predictors:
                # Apply decomposition
                ds["data"] = decompose_within_between(
                    ds["data"], 
                    numeric_predictors,
                    subject_col=config.get("random_intercept", "subject_id"),
                )
                
                # Update config with decomposed predictor names
                config = dict(config) if not isinstance(config, dict) else config
                decomposed_predictors = []
                for p in predictors:
                    if p in numeric_predictors:
                        decomposed_predictors.append(f"{p}_within")
                        decomposed_predictors.append(f"{p}_between")
                    else:
                        decomposed_predictors.append(p)
                config["predictors"] = decomposed_predictors
                config["_original_predictors"] = predictors  # Keep original for reference
        
        return ds, config, None
    
    elif level == "subject":
        # Prepare subject-level data
        single_ds = prepare_single_instance_metrics(profiles)
        daily_ds = prepare_daily_metrics(profiles)
        daily_ds = add_subject_metadata(daily_ds, profiles, fields=["work_type"])
        
        if single_ds["data"].empty:
            return None, config, "Single-instance data is empty"
        
        return {"single": single_ds, "daily": daily_ds}, config, None
    
    return None, config, f"Unknown level: {level}"


# -----------------------------------------------------------------------------
# Model Fitting
# -----------------------------------------------------------------------------

def _run_lmm(
    ds: Any,
    config: HypothesisConfig,
) -> HypothesisResult:
    """Run Linear Mixed Model hypothesis test."""
    hypothesis_id = config.get("_id", "?")
    
    # Determine outcome (may be transformed)
    outcome = config.get("_transformed_outcome", config["outcome"])
    
    # Build fixed effects list
    fixed_effects = config.get("predictors", []) + config.get("covariates", [])
    
    # Check for missing values in outcome
    if ds["data"][outcome].dropna().empty:
        return create_hypothesis_result(
            hypothesis_id=hypothesis_id,
            config=config,
            p_value=np.nan,
            skipped=True,
            skip_reason=f"Outcome '{outcome}' has no non-missing values",
        )
    
    # Avoid double-transform: if runner already transformed, disable registry transform
    transform_override = "none" if (config.get("transform") or config.get("_transformed_outcome")) else None

    # Fit model (ML by default in fit_lmm)
    result = fit_lmm(
        ds,
        outcome=outcome,
        fixed_effects=fixed_effects,
        random_intercept=config.get("random_intercept", "subject_id"),
        transform=transform_override,
    )

    # Automatic assumption corrections (log-transform) when requested
    note_parts: List[str] = []
    auto_correct = config.get("auto_correct", False)
    transform_already_applied = bool(config.get("transform") or config.get("_transformed_outcome"))

    if auto_correct and not transform_already_applied and result.get("converged", False):
        diagnostics = residual_diagnostics(result)
        needs_homoscedastic_fix = diagnostics.get("homoscedasticity", {}).get("is_ok") is False
        needs_normality_fix = diagnostics.get("residuals_normality", {}).get("is_normal") is False
        needs_fix = needs_homoscedastic_fix or needs_normality_fix

        if needs_fix:
            # Only attempt log transform for non-negative outcomes
            outcome_values = ds["data"][outcome].dropna()
            if not outcome_values.empty and outcome_values.min() >= 0:
                corrected_ds = dict(ds)
                corrected_ds["data"] = _apply_log_transform(ds["data"], outcome)
                corrected_outcome = f"{outcome}_log"

                corrected_result = fit_lmm(
                    corrected_ds,
                    outcome=corrected_outcome,
                    fixed_effects=fixed_effects,
                    random_intercept=config.get("random_intercept", "subject_id"),
                )

                if corrected_result.get("converged", False):
                    result = corrected_result
                    config = dict(config)
                    config["_transformed_outcome"] = corrected_outcome
                    note_parts.append("Auto-correction: log transform applied due to assumption violations")
                    diagnostics = residual_diagnostics(result)
                else:
                    note_parts.append("Auto-correction attempted (log transform) but refit failed; using original model")
            else:
                note_parts.append("Auto-correction skipped (outcome has negative values)")
        else:
            diagnostics = residual_diagnostics(result)
    else:
        diagnostics = residual_diagnostics(result) if result.get("converged", False) else None

    if diagnostics is not None:
        result["diagnostics"] = diagnostics
        if diagnostics.get("homoscedasticity", {}).get("is_ok") is False:
            note_parts.append("Heteroscedasticity persists (consider bootstrap/robust SE)")
        if diagnostics.get("residuals_normality", {}).get("is_normal") is False:
            note_parts.append("Residuals non-normal (consider bootstrap)")

    # Bootstrap p-value if violations persist and requested
    if config.get("bootstrap_on_violation", False) and diagnostics is not None:
        violations_persist = (
            diagnostics.get("homoscedasticity", {}).get("is_ok") is False
            or diagnostics.get("residuals_normality", {}).get("is_normal") is False
        )

        if violations_persist:
            n_boot = int(config.get("bootstrap_iterations", 200))
            boot_p = _bootstrap_primary_pvalue(
                ds,
                outcome=outcome,
                fixed_effects=fixed_effects,
                random_intercept=config.get("random_intercept", "subject_id"),
                primary=primary,
                decompose=config.get("decompose_predictors", False),
                n_bootstrap=n_boot,
            )
            if not np.isnan(boot_p):
                note_parts.append(f"Bootstrap p-value (n={n_boot}) = {boot_p:.4f}")
                result["_bootstrap_p_value"] = boot_p
    
    if not result.get("converged", False):
        return create_hypothesis_result(
            hypothesis_id=hypothesis_id,
            config=config,
            p_value=np.nan,
            converged=False,
            model_result=result,
            note="; ".join(["Model did not converge"] + note_parts) if note_parts else "Model did not converge",
        )
    
    # Extract p-value for primary predictor
    coeffs = result.get("coefficients", pd.DataFrame())
    if coeffs.empty:
        return create_hypothesis_result(
            hypothesis_id=hypothesis_id,
            config=config,
            p_value=np.nan,
            converged=False,
            model_result=result,
            note="; ".join(["No coefficients returned"] + note_parts) if note_parts else "No coefficients returned",
        )
    
    primary = config.get("primary_predictor", config["predictors"][0])

    # Compute LRT for primary predictor if requested
    use_lrt = config.get("use_lrt_primary", False)
    lrt_pvalue = np.nan
    lrt_note = ""

    if use_lrt and result.get("model") is not None:
        # Determine which term(s) to remove for reduced model
        if config.get("decompose_predictors", False):
            primary_terms = [f"{primary}_within"]
        else:
            primary_terms = [primary]

        reduced_effects = [fe for fe in fixed_effects if fe not in primary_terms]

        if len(reduced_effects) == len(fixed_effects):
            lrt_note = "LRT skipped (primary term not found in fixed effects)"
        else:
            reduced_result = fit_lmm(
                ds,
                outcome=outcome,
                fixed_effects=reduced_effects,
                random_intercept=config.get("random_intercept", "subject_id"),
                transform=transform_override,
                reml=False,
            )

            full_model = result.get("model")
            red_model = reduced_result.get("model") if reduced_result else None
            if full_model is not None and red_model is not None:
                lrt_stat = 2 * (full_model.llf - red_model.llf)
                df_diff = len(full_model.fe_params) - len(red_model.fe_params)
                if df_diff > 0:
                    lrt_pvalue = float(stats.chi2.sf(lrt_stat, df_diff))
                else:
                    lrt_note = "LRT skipped (df_diff <= 0)"
            else:
                lrt_note = "LRT skipped (model fit failed)"
    
    # For decomposed predictors, look for the _within version (day-to-day effect)
    # as the primary test, since this is typically the effect of interest
    if config.get("decompose_predictors", False):
        # Try to find the within-subject effect first
        primary_within = f"{primary}_within"
        mask_within = coeffs["term"].str.contains(primary_within, case=False, na=False)
        
        if mask_within.any():
            p_value = float(coeffs.loc[mask_within, "p_value"].iloc[0])
            
            # Also extract the between effect for reporting
            primary_between = f"{primary}_between"
            mask_between = coeffs["term"].str.contains(primary_between, case=False, na=False)
            if mask_between.any():
                p_between = float(coeffs.loc[mask_between, "p_value"].iloc[0])
                result["_p_value_between"] = p_between
            
            # Use LRT p-value if available, otherwise Wald
            primary_p = lrt_pvalue if not np.isnan(lrt_pvalue) else p_value
            if not np.isnan(lrt_pvalue):
                note_parts.append(f"Primary p-value from LRT: {lrt_pvalue:.4f}")
            if lrt_note:
                note_parts.append(lrt_note)

            return create_hypothesis_result(
                hypothesis_id=hypothesis_id,
                config=config,
                p_value=primary_p,
                converged=True,
                model_result=result,
                note="; ".join([
                    f"Within-subject p={p_value:.4f}, Between-subject p={result.get('_p_value_between', 'N/A')}",
                    *note_parts,
                ]) if note_parts else f"Within-subject p={p_value:.4f}, Between-subject p={result.get('_p_value_between', 'N/A')}",
            )
    
    # Standard case: find primary predictor
    mask = coeffs["term"].str.contains(primary, case=False, na=False)
    p_values = coeffs.loc[mask, "p_value"]
    
    if p_values.empty:
        return create_hypothesis_result(
            hypothesis_id=hypothesis_id,
            config=config,
            p_value=np.nan,
            model_result=result,
            note="; ".join([f"Primary predictor '{primary}' not found in coefficients"] + note_parts) if note_parts else f"Primary predictor '{primary}' not found in coefficients",
        )
    
    p_value = float(p_values.iloc[0])

    # Use LRT p-value if available, otherwise Wald
    primary_p = lrt_pvalue if not np.isnan(lrt_pvalue) else p_value
    if not np.isnan(lrt_pvalue):
        note_parts.append(f"Primary p-value from LRT: {lrt_pvalue:.4f}")
    if lrt_note:
        note_parts.append(lrt_note)
    
    return create_hypothesis_result(
        hypothesis_id=hypothesis_id,
        config=config,
        p_value=primary_p,
        converged=True,
        model_result=result,
        note="; ".join(note_parts) if note_parts else "",
    )


def _run_ols_h4(
    data: Dict[str, Any],
    config: HypothesisConfig,
) -> HypothesisResult:
    """
    Run OLS for H4 (OSPAQ validation).
    
    Special handling: 
    - Aggregates daily sitting to subject-level (weighted by duration if available)
    - Applies logit transform to proportion outcome
    """
    hypothesis_id = config.get("_id", "H4")
    single_ds = data["single"]
    daily_ds = data["daily"]
    
    # Check required columns
    if "ospaq_sitting_frac" not in single_ds["data"].columns:
        return create_hypothesis_result(
            hypothesis_id=hypothesis_id,
            config=config,
            p_value=np.nan,
            skipped=True,
            skip_reason="OSPAQ sitting fraction not available",
        )
    
    if "har_sentado_prop" not in daily_ds["data"].columns:
        return create_hypothesis_result(
            hypothesis_id=hypothesis_id,
            config=config,
            p_value=np.nan,
            skipped=True,
            skip_reason="HAR sitting proportion not available",
        )
    
    # Aggregate daily sitting to subject level (weekly proportion from durations if available)
    obj = daily_ds["data"].dropna(subset=["har_sentado_prop"])

    if {"har_sentado_duration_sec", "har_total_duration_sec"}.issubset(obj.columns):
        # Weekly sitting proportion = sum(sitting) / sum(total)
        obj_subject = (
            obj.groupby("subject_id")[
                ["har_sentado_duration_sec", "har_total_duration_sec"]
            ]
            .sum(min_count=1)
            .reset_index()
        )
        obj_subject["har_sentado_prop"] = (
            obj_subject["har_sentado_duration_sec"]
            / obj_subject["har_total_duration_sec"]
        )
        obj_subject = obj_subject[["subject_id", "har_sentado_prop"]]
    elif "har_total_duration_sec" in obj.columns:
        # Weighted mean by monitored duration (fallback)
        def weighted_mean(group):
            weights = group["har_total_duration_sec"]
            if weights.sum() > 0:
                return np.average(group["har_sentado_prop"], weights=weights)
            return group["har_sentado_prop"].mean()
        obj_subject = obj.groupby("subject_id").apply(weighted_mean).reset_index()
        obj_subject.columns = ["subject_id", "har_sentado_prop"]
    else:
        # Simple mean (last fallback)
        obj_subject = (
            obj.groupby("subject_id")["har_sentado_prop"]
            .mean()
            .reset_index()
        )
    
    # Merge with OSPAQ
    df = single_ds["data"][["subject_id", "ospaq_sitting_frac"]].copy()
    if "work_type" in single_ds["data"].columns:
        df["work_type"] = single_ds["data"]["work_type"]
    elif "work_type" in daily_ds["data"].columns:
        wt = daily_ds["data"].groupby("subject_id")["work_type"].first().reset_index()
        df = df.merge(wt, on="subject_id", how="left")
    
    df = df.merge(obj_subject, on="subject_id", how="inner")
    df = df.dropna(subset=["ospaq_sitting_frac", "har_sentado_prop"])
    
    if df.empty:
        return create_hypothesis_result(
            hypothesis_id=hypothesis_id,
            config=config,
            p_value=np.nan,
            skipped=True,
            skip_reason="No overlap between OSPAQ and objective sitting data",
        )
    
    # Apply logit transform to proportion outcome (if specified in config)
    transform = config.get("transform")
    outcome_col = "har_sentado_prop"
    
    if transform == "logit":
        # Logit transform: log(p / (1-p))
        # Clip to avoid log(0) or log(inf)
        eps = 1e-6
        p_clipped = df["har_sentado_prop"].clip(eps, 1 - eps)
        df["har_sentado_prop_logit"] = np.log(p_clipped / (1 - p_clipped))
        outcome_col = "har_sentado_prop_logit"
    
    # Build OLS model
    X = df[["ospaq_sitting_frac"]].copy().astype(float)
    if "work_type" in df.columns:
        work_dummies = pd.get_dummies(df["work_type"], drop_first=True, dtype=float)
        X = pd.concat([X, work_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[outcome_col].astype(float)
    
    model = sm.OLS(y, X).fit()
    p_value = model.pvalues.get("ospaq_sitting_frac", np.nan)
    
    return create_hypothesis_result(
        hypothesis_id=hypothesis_id,
        config=config,
        p_value=float(p_value),
        converged=True,
        model_result={"ols_summary": model.summary()},
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_hypothesis(
    hypothesis_id: str,
    profiles: Dict[str, dict],
    verbose: bool = True,
) -> HypothesisResult:
    """
    Run a single hypothesis test.
    
    Args:
        hypothesis_id: One of "H1", "H2", ..., "H6"
        profiles: Loaded OH profiles dictionary
        verbose: Print progress messages
    
    Returns:
        HypothesisResult dict with p-value and model details
    """
    config = dict(get_hypothesis(hypothesis_id))  # Copy to allow modification
    config["_id"] = hypothesis_id
    
    if verbose:
        print(f"\n[{hypothesis_id}] {config['name']}")
    
    # Prepare data
    data, config, error = _prepare_data(profiles, config)
    if error:
        if verbose:
            print(f"  SKIPPED: {error}")
        return create_hypothesis_result(
            hypothesis_id=hypothesis_id,
            config=config,
            p_value=np.nan,
            skipped=True,
            skip_reason=error,
        )
    
    # Run appropriate model
    model_type = config.get("model", "lmm")
    
    if model_type == "lmm":
        result = _run_lmm(data, config)
    elif model_type == "ols" and hypothesis_id == "H4":
        result = _run_ols_h4(data, config)
    else:
        result = create_hypothesis_result(
            hypothesis_id=hypothesis_id,
            config=config,
            p_value=np.nan,
            skipped=True,
            skip_reason=f"Unknown model type: {model_type}",
        )
    
    if verbose:
        if result.get("skipped", False):
            print(f"  SKIPPED: {result.get('skip_reason', '')}")
        elif not result.get("converged", True):
            print(f"  WARNING: {result.get('note', '')}")
        else:
            p_val = result["p_value"]
            if not np.isnan(p_val):
                sig = "✓ SIGNIFICANT" if p_val < 0.05 else "✗ Not significant"
                print(f"  p = {p_val:.4f} ({sig})")
            
            # Print model summary if available
            model_result = result.get("model_result")
            if model_result and "coefficients" in model_result:
                print(summarize_lmm_result(model_result))
    
    return result


def run_all(
    profiles: Dict[str, dict],
    hypotheses: Optional[List[str]] = None,
    verbose: bool = True,
    run_variance_check: bool = True,
) -> Dict[str, HypothesisResult]:
    """
    Run all (or selected) hypothesis tests.
    
    Args:
        profiles: Loaded OH profiles dictionary
        hypotheses: List of hypothesis IDs to run (default: all)
        verbose: Print progress messages
        run_variance_check: Run variance diagnostics before modeling
    
    Returns:
        Dictionary mapping hypothesis ID to HypothesisResult dict
    """
    hypotheses = hypotheses or list(HYPOTHESES.keys())
    
    if verbose:
        print("=" * 70)
        print("HYPOTHESIS TESTING (H1-H6)")
        print("=" * 70)
    
    # Run variance check on key outcomes if requested
    if run_variance_check and verbose:
        # Prepare data once for variance check
        daily_ds = prepare_daily_metrics(profiles)
        daily_ds = add_subject_metadata(daily_ds, profiles, fields=["work_type"])
        
        # Collect unique outcomes from requested hypotheses
        outcomes = set()
        for h_id in hypotheses:
            config = get_hypothesis(h_id)
            if config.get("level") == "daily":
                outcomes.add(config["outcome"])
                outcomes.update(config.get("predictors", []))
        
        # Filter to numeric columns that exist
        numeric_outcomes = [
            o for o in outcomes
            if o in daily_ds["data"].columns
            and pd.api.types.is_numeric_dtype(daily_ds["data"][o])
        ]
        
        if numeric_outcomes:
            print("\n" + "=" * 70)
            print("VARIANCE CHECK (Pre-modeling Diagnostic)")
            print("=" * 70)
            variance_df = check_variance(daily_ds, outcomes=list(numeric_outcomes))
            flagged = variance_df[variance_df["is_degenerate"]]
            print(f"\nChecked {len(variance_df)} variables:")
            print(f"  ✅ {(~variance_df['is_degenerate']).sum()} variables have sufficient variance")
            if not flagged.empty:
                print(f"  ⚠️  {len(flagged)} variables are degenerate:")
                for _, row in flagged.iterrows():
                    print(f"     - {row['outcome']}: {row['reason']}")
    
    results = {}
    for h_id in hypotheses:
        results[h_id] = run_hypothesis(h_id, profiles, verbose=verbose)
    
    return results


def apply_multiplicity_correction(
    results: Dict[str, HypothesisResult],
    method: str = "holm",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply multiplicity correction across CONFIRMATORY hypotheses only.
    
    Exploratory hypotheses (marked with exploratory=True in config) are
    excluded from the correction but reported separately.
    
    Args:
        results: Dictionary of HypothesisResult dicts
        method: Correction method ("holm" or "fdr")
        verbose: Print results
    
    Returns:
        DataFrame with corrected p-values
    """
    # Separate confirmatory from exploratory hypotheses
    confirmatory = {}
    exploratory = {}
    
    for h_id, result in results.items():
        if result.get("skipped", False):
            continue
        if result["config"].get("exploratory", False):
            exploratory[h_id] = hypothesis_result_to_dict(result)
        else:
            confirmatory[h_id] = hypothesis_result_to_dict(result)
    
    # Report exploratory hypotheses (no correction)
    if verbose and exploratory:
        print("\n" + "=" * 70)
        print("EXPLORATORY HYPOTHESES (no multiplicity correction)")
        print("=" * 70)
        for h_id, data in exploratory.items():
            p = data["p_value"]
            sig = "*" if not np.isnan(p) and p < 0.05 else ""
            print(f"  {h_id}: p = {p:.4f} {sig}")
    
    # Convert to format expected by apply_holm_hypotheses
    hyp_dict = confirmatory
    
    if not hyp_dict:
        if verbose:
            print("No valid hypotheses to correct")
        return pd.DataFrame()
    
    corrected = apply_holm_hypotheses(hyp_dict)
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"MULTIPLICITY CORRECTION ({method.upper()})")
        print("=" * 70)
        print(corrected.to_string(index=False))
    
    return corrected


def summarize_results(
    results: Dict[str, HypothesisResult],
) -> pd.DataFrame:
    """
    Create a summary table of all hypothesis results.
    
    Returns:
        DataFrame with hypothesis summaries
    """
    rows = []
    for h_id, result in results.items():
        config = result["config"]
        skipped = result.get("skipped", False)
        converged = result.get("converged", True)
        p_value = result["p_value"]
        
        rows.append({
            "Hypothesis": h_id,
            "Name": config["name"],
            "Level": config.get("level", "?"),
            "Outcome": config["outcome"],
            "Primary Predictor": config.get("primary_predictor", "?"),
            "p-value": p_value if not skipped else None,
            "Significant": p_value < 0.05 if not np.isnan(p_value) else None,
            "Status": "Skipped" if skipped else ("Converged" if converged else "Failed"),
            "Note": result.get("skip_reason", "") if skipped else result.get("note", ""),
        })
    
    return pd.DataFrame(rows)
